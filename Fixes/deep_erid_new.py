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
    # ------------------------------------------------------------------ setup
    cfg.train.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    cfg.env.device   = cfg.train.device

    torch.manual_seed(cfg.seed)

    cfg.env.num_envs            = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames  = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size      = cfg.collector.frames_per_batch

    log_dir = os.path.join("tb_logs", f"{cfg.env.scenario_name}-seed-{cfg.seed}")
    writer  = SummaryWriter(log_dir=log_dir)
    torchrl_logger.info(f"Tensorboard logging to: {log_dir}")

    # ------------------------------------------------------------------ envs
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
        RewardSum(in_keys=[env_test.reward_key], out_keys=[("agents", "episode_reward")])
    )

    # ------------------------------------------------------------------ actor
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

    # ------------------------------------------------------------------ critic
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

    # ------------------------------------------------------------------ collector / buffer
    collector = Collector(
        env,
        policy,
        device=cfg.train.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    # ------------------------------------------------------------------ loss
    loss_module = DeepERIDLoss(
        actor_network=policy,
        critic_network=critic,
        entropy_coeff=cfg.loss.entropy_eps,
        entropy_bonus=True,
        gamma=cfg.loss.gamma,
        alpha=cfg.loss.alpha,
    )

    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
        sample_log_prob=("agents", "action_log_prob"),
    )

    # FIX: separate optimizers for actor and critic.
    #
    # Why this matters:
    #   The critic needs to converge *before* its Q values are reliable
    #   enough to guide the evolutionary dynamics.  A higher LR for the
    #   critic lets it track the moving target faster, while a lower LR
    #   for the actor prevents it from committing to a bad policy before
    #   the critic has stabilised.
    actor_params  = list(loss_module.actor_network_params.values(True, True))
    critic_params = list(loss_module.critic_network_params.values(True, True))

    actor_optim  = torch.optim.Adam(actor_params,  lr=cfg.train.actor_lr)
    critic_optim = torch.optim.Adam(critic_params, lr=cfg.train.critic_lr)

    # ------------------------------------------------------------------ training loop
    total_time   = 0
    total_frames = 0
    sampling_start = time.time()

    eval_freq      = cfg.eval.frequency
    polyak_tau     = cfg.train.polyak_tau   # e.g. 0.005
    policy_history = []

    for i, tensordict_data in enumerate(collector):
        sampling_time = time.time() - sampling_start

        total_frames += tensordict_data.numel()
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds   = []
        training_start = time.time()

        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata = replay_buffer.sample()

                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                # --- critic step ---
                critic_optim.zero_grad()
                loss_vals["loss_critic"].backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(critic_params, cfg.train.max_grad_norm)
                critic_optim.step()

                # --- actor step ---
                actor_optim.zero_grad()
                actor_loss = loss_vals["loss_objective"] + loss_vals["loss_entropy"]
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_params, cfg.train.max_grad_norm)
                actor_optim.step()

                # FIX: Polyak-update the target critic after every optimizer step.
                # Without this the target network stays frozen at random-init weights
                # and produces meaningless bootstrap targets throughout training.
                loss_module.soft_update_target(tau=polyak_tau)

                total_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in actor_params + critic_params
                    if p.grad is not None
                ) ** 0.5
                training_tds[-1].set(
                    "grad_norm", torch.tensor(total_norm, device=cfg.train.device)
                )

        collector.update_policy_weights_()

        training_time  = time.time() - training_start
        iteration_time = sampling_time + training_time
        total_time    += iteration_time
        training_tds   = torch.stack(training_tds)

        # ---------------------------------------------------------------- logging
        episode_r = tensordict_data.get(("next", "agents", "episode_reward"))
        episode_r = episode_r.reshape(cfg.env.num_envs, cfg.env.max_steps, env.n_agents, 1)
        mean_episode_reward = episode_r[:, -1].mean().item()

        avg_loss_objective = training_tds["loss_objective"].mean().item()
        avg_loss_critic    = training_tds["loss_critic"].mean().item()
        avg_loss_entropy   = training_tds["loss_entropy"].mean().item()
        avg_grad_norm      = training_tds["grad_norm"].mean().item()

        global_step = total_frames

        nash, avg_policy = compute_nash_conv(env, policy)
        print(avg_policy)
        policy_history.append((global_step, avg_policy))

        # --- Q-symmetry diagnostic: std of Q values across actions (should → 0 at NE) ---
        with torch.no_grad():
            sample_td  = env.reset()
            q_diag     = loss_module._get_q_values(sample_td, use_target=False)
            q_std_mean = q_diag.std(dim=-1).mean().item()   # lower = more symmetric

        for agent in range(env.n_agents):
            for action in range(env.n_actions):
                writer.add_scalar(
                    f"Policy/agent{agent}_action{action}",
                    avg_policy[agent, action].item(),
                    global_step,
                )

        writer.add_scalar("Reward/mean_episode_reward", mean_episode_reward, global_step)
        writer.add_scalar("Loss/objective",   avg_loss_objective,             global_step)
        writer.add_scalar("Loss/critic",      avg_loss_critic,                global_step)
        writer.add_scalar("Loss/entropy",     avg_loss_entropy,               global_step)
        writer.add_scalar("Loss/total",       avg_loss_objective + avg_loss_critic, global_step)
        writer.add_scalar("Grad/grad_norm",   avg_grad_norm,                  global_step)
        writer.add_scalar("Critic/q_std",     q_std_mean,                     global_step)
        writer.add_scalar("Time/sampling",    sampling_time,                  global_step)
        writer.add_scalar("Time/training",    training_time,                  global_step)
        writer.add_scalar("Frames/total",     total_frames,                   global_step)
        writer.add_scalar("Nash/Nash_Conv",   nash,                           global_step)

        torchrl_logger.info(
            f"Iter {i} | "
            f"Frames: {total_frames} | "
            f"Reward {mean_episode_reward:.3f} | "
            f"Obj {avg_loss_objective:.4f} | "
            f"Critic {avg_loss_critic:.4f} | "
            f"Q_std {q_std_mean:.4f} | "
            f"Nash {nash:.4f}"
        )

        if i % eval_freq == 0 or i == cfg.collector.n_iters - 1:
            eval_reward = evaluate_policy(env_test=env_test, policy=policy)
            writer.add_scalar("Eval/mean_episode_reward", eval_reward, total_frames)
            torchrl_logger.info(f"Eval reward: {eval_reward:.3f}")

    # ------------------------------------------------------------------ save
    steps    = np.array([s for s, _ in policy_history])
    policies = np.stack([p.numpy() for _, p in policy_history])

    np.save(
        f"policy_trajectory_{cfg.env.scenario_name}_seed{cfg.seed}.npy",
        {"steps": steps, "policies": policies,
         "scenario_name": cfg.env.scenario_name,
         "n_agents": env.n_agents, "n_actions": env.n_actions},
    )

    writer.close()
    collector.shutdown()
    if not env.is_closed:      env.close()
    if not env_test.is_closed: env_test.close()


if __name__ == "__main__":
    train()