import time
import os

import numpy as np
import hydra
import torch

from tensordict.nn import TensorDictModule
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ValueEstimators

from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig

from utils.utils import compute_nash_conv, evaluate_policy
from matrix_games import MatrixGameFactory
from utils.losses.derid import DeepERIDLoss

def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))


@hydra.main(version_base="1.1", config_path="", config_name="deep_erid")
def train(cfg: DictConfig):
    # setting up device
    cfg.train.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    cfg.env.device = cfg.train.device

    # seed for reporoducability
    torch.manual_seed(cfg.seed)

    # sampling
    cfg.env.num_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch

    # initializing logging
    log_dir = os.path.join("tb_logs", f"DeepERID")
    writer = SummaryWriter(log_dir=log_dir)
    torchrl_logger.info(f"Tensorboard logging to: {log_dir}")

    # environments
    env = MatrixGameFactory(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.num_envs,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
    )

    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")])
    )

    env_test = MatrixGameFactory(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.num_envs,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
    )

    env_test = TransformedEnv(
        env_test,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")])
    )

    # policy

    policy_net = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=env.full_action_spec_unbatched[env.action_key].shape[-1],
            n_agents=env.n_agents,
            centralized=False,
            share_params=cfg.model.shared_params,
            device=cfg.train.device,
            depth=2,
            num_cells=64,
            activation_class=nn.Tanh,
        ),
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "logits")],
    )

    policy = ProbabilisticActor(
        policy_module,
        spec=env.full_action_spec_unbatched,
        in_keys=[("agents", "logits")],
        out_keys=[env.action_key],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )

    # critic
    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec[("agents", "observation")].shape[-1],
        n_agent_outputs=env.full_action_spec_unbatched[env.action_key].shape[-1],
        n_agents=env.n_agents,
        centralized=cfg.model.centralised_critic,
        share_params=cfg.model.shared_params,
        device=cfg.train.device,
        depth=2,
        num_cells=64,
        activation_class=nn.Tanh,
    )

    critic = ValueOperator(
        critic_net,
        in_keys=[("agents", "observation")],
        out_keys=["q_value"],
    )

    # dealing with env data

    collector = Collector(
        env,
        policy,
        device=cfg.train.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        #postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys),
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    # deep erid loss 

    loss_module = DeepERIDLoss(
        actor_network=policy,
        critic_network=critic,
        entropy_coeff=cfg.loss.entropy_eps,
        alpha=cfg.loss.alpha,
        gamma=cfg.loss.gamma,
        reward_scale=cfg.loss.reward_scale,
    )

    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
        sample_log_prob=("agents", "action_log_prob"),
    )

    if loss_module.functional:
        actor_params = list(loss_module.actor_network_params.values(True, True))
        critic_params = list(loss_module.critic_network_params.values(True, True))
    else:
        actor_params = list(loss_module.actor_network.parameters())
        critic_params = list(loss_module.critic_network.parameters())

    actor_optim = torch.optim.Adam(actor_params, lr=cfg.train.actor_lr)
    critic_optim = torch.optim.Adam(critic_params, lr=cfg.train.critic_lr)

    # training loop 

    total_time = 0
    total_frames = 0
    sampling_start = time.time()

    eval_freq = cfg.eval.frequency

    policy_history = []

    for i, tensordict_data in enumerate(collector):
        sampling_time = time.time() - sampling_start
        
        current_frames = tensordict_data.numel()
        total_frames += current_frames
        data_view = tensordict_data.reshape(-1)

        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()
        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata = replay_buffer.sample()

                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                # --- Critic Step ---
                critic_optim.zero_grad()
                loss_vals["loss_critic"].backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(critic_params, cfg.train.max_grad_norm)
                critic_optim.step()

                # --- Actor Step ---
                actor_optim.zero_grad()
                actor_loss = loss_vals["loss_objective"] + loss_vals["loss_entropy"]
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_params, cfg.train.max_grad_norm)
                actor_optim.step()

                loss_module.soft_update_target(tau=cfg.train.tau)
                # NOTE: soft_update_avg_actor is intentionally NOT called here.
                # Calling it every gradient step (×120/iter) makes the average
                # actor equivalent to the live actor with a tiny lag — defeating
                # the purpose.  It is called once per *iteration* below.

                total_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in actor_params + critic_params
                    if p.grad is not None
                ) ** 0.5

                training_tds[-1].set("grad_norm", torch.tensor(total_norm, device=cfg.train.device))

        # ── Update average actor ONCE per iteration ──────────────────────────
        # tau=0.02 per iteration means the average blends in ~2% of the current
        # policy each pass.  Over 100 iterations the effective "memory" is
        # ~50 iterations, giving a true time-average rather than a lagged copy.
        loss_module.soft_update_avg_actor(tau=0.02)

        collector.update_policy_weights_()

        training_time = time.time() - training_start

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        with torch.no_grad()        :
            sample_td = env.reset()
            q_values = loss_module._get_q_values(sample_td, use_target=False)
            q_target = loss_module._get_q_values(sample_td, use_target=True)

            print(f"Q values: {q_values[0]}")
            print(f"Q Target: {q_target[0]}")

            q_std_mean = q_values.std(dim=-1).mean().item()
            q_mean = q_values.mean().item()
            q_std_all = q_values.std().item()

            q_range = (q_values.max(dim=-1).values - q_values.min(dim=-1).values).mean().item()


        episode_r = tensordict_data.get(("next", "agents", "episode_reward"))
        episode_r = episode_r.reshape(
            cfg.env.num_envs, cfg.env.max_steps, env.n_agents, 1
        )
        final_rewards = episode_r[:, -1]
        mean_episode_reward = final_rewards.mean().item()

        avg_loss_objective = training_tds["loss_objective"].mean().item()
        avg_loss_critic = training_tds["loss_critic"].mean().item()
        avg_loss_entropy = training_tds["loss_entropy"].mean().item()
        avg_grad_norm = training_tds["grad_norm"].mean().item()

        global_step = total_frames

        nash, avg_policy = compute_nash_conv(env, policy)
        print(f" Average Policy: {avg_policy}")
        policy_history.append((global_step, avg_policy))

        for agent in range(env.n_agents):
            for action in range(env.n_actions):
                prob = avg_policy[agent, action].item()
                writer.add_scalar(f"Policy/agent{agent}_action{action}", prob, global_step)

        writer.add_scalar("Reward/mean_episode_reward", mean_episode_reward, global_step)

        writer.add_scalar("Loss/objective", avg_loss_objective, global_step)
        writer.add_scalar("Loss/critic", avg_loss_critic, global_step)
        writer.add_scalar("Loss/entropy", avg_loss_entropy, global_step)
        writer.add_scalar("Loss/total", avg_loss_objective + avg_loss_critic, global_step)

        writer.add_scalar("Grad/grad_norm", avg_grad_norm, global_step)

        writer.add_scalar("Critic/Q_std", q_std_mean, global_step)
        writer.add_scalar("Critic/q_mean", q_mean, global_step)
        writer.add_scalar("Critic/q_std_all", q_std_all, global_step)
        writer.add_scalar("Critic/q_range", q_range, global_step)
        
        writer.add_scalar("Time/sampling_time", sampling_time, global_step)
        writer.add_scalar("Time/training_time", training_time, global_step)
        writer.add_scalar("Time/iteration_time", iteration_time, global_step)

        writer.add_scalar("Frames/total_frames", total_frames, global_step)

        writer.add_scalar("Nash/Nash_Conv", nash, global_step)

        # Log the average actor policy (should converge more smoothly to NE)
        if loss_module.avg_actor_network_params is not None:
            with torch.no_grad():
                sample_td2 = env.reset()
                avg_pi = loss_module._get_action_probs(sample_td2, use_avg=True)
                avg_pi_mean = avg_pi.mean(dim=0)  # (n_agents, n_actions)
                for agent in range(env.n_agents):
                    for action in range(env.n_actions):
                        prob = avg_pi_mean[agent, action].item()
                        writer.add_scalar(f"AvgPolicy/agent{agent}_action{action}", prob, global_step)

        torchrl_logger.info(
            f"Iter {i} | "
            f"Frames: {total_frames} | "
            f"Mean Ep Reward {mean_episode_reward:.3f} | "
            f"Objective Loss {avg_loss_objective:.4f} | "
            f"Critic Loss {avg_loss_critic:.4f} | "
            f"Q Std Mean {q_std_mean:.4f} | "
            f"Q Mean {q_mean:.4f} | "
            f"Q Range {q_range:.4f} | "
            f"Nash Conv {nash:.4f}"
        )

        if i % eval_freq == 0 or i == cfg.collector.n_iters - 1:
            eval_reward = evaluate_policy(
                env_test=env_test,
                policy=policy
            )

            writer.add_scalar("Eval/mean_episode_reward", eval_reward, total_frames)

            torchrl_logger.info(
                f"Evaluation Reward: {eval_reward:.3f}"
            )

    steps = np.array([step for step, _ in policy_history])
    policies = np.stack([p.numpy() for step, p in policy_history])

    traj_path = f"policy_trajectory_{cfg.env.scenario_name}_seed{cfg.seed}.npy"
    np.save(traj_path, {
        "steps": steps,
        "policies": policies,
        "scenario_name": cfg.env.scenario_name,
        "n_agents": env.n_agents,
        "n_actions": env.n_actions,
    })

    writer.close()
    collector.shutdown()
    if not env.is_closed:
        env.close()
    if not env_test.is_closed:
        env_test.close()


if __name__ == "__main__":
    train()