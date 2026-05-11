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
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig

from utils.utils import evaluate_policy, compute_positional_entropy, compute_mean_goal_distance
from grid_worlds import GridWorldFactory


@hydra.main(version_base="1.1", config_path="", config_name="mappo")
def train(cfg: DictConfig):

    cfg.train.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    cfg.env.device   = cfg.train.device

    torch.manual_seed(cfg.seed)

    cfg.env.num_envs         = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = (
        cfg.collector.frames_per_batch * cfg.collector.n_iters
    )
    cfg.buffer.memory_size   = cfg.collector.frames_per_batch

    log_dir = os.path.join("tb_logs", cfg.sub_exp)
    writer  = SummaryWriter(log_dir=log_dir)
    torchrl_logger.info(f"Tensorboard logging to: {log_dir}")

    def make_env(seed_offset=0):
        env = GridWorldFactory(
            scenario=cfg.env.scenario_name,
            num_envs=cfg.env.num_envs,
            max_steps=cfg.env.max_steps,
            device=cfg.env.device,
            seed=cfg.seed + seed_offset,
        )
        env = TransformedEnv(
            env,
            RewardSum(
                in_keys=[env.reward_key],
                out_keys=[("agents", "episode_reward")],
            ),
        )
        return env

    env      = make_env(seed_offset=0)
    env_test = make_env(seed_offset=1)

    torchrl_logger.info(
        f"Env: {cfg.env.scenario_name} | "
        f"Agents: {env.n_agents} | "
        f"Obs dim: {env.observation_spec['agents', 'observation'].shape[-1]} | "
        f"Grid: {env.grid_size}x{env.grid_size}"
    )

    policy_net = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec[("agents", "observation")].shape[-1],
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

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec[("agents", "observation")].shape[-1],
        n_agent_outputs=1,
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
    )

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

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coeff=cfg.loss.entropy_eps,
        normalize_advantage=False,
        separate_agent_loss=True,
    )

    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
        sample_log_prob=("agents", "action_log_prob"),
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE,
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.lmbda,
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg.train.lr)

    total_time   = 0
    total_frames = 0
    sampling_start = time.time()

    eval_freq = cfg.eval.frequency

    metric_history = []

    goal = env._goals[0][0].tolist()
    
    for i, tensordict_data in enumerate(collector):
        sampling_time = time.time() - sampling_start

        with torch.no_grad():
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        current_frames = tensordict_data.numel()
        total_frames  += current_frames
        data_view      = tensordict_data.reshape(-1)

        replay_buffer.extend(data_view)

        training_tds   = []
        training_start = time.time()

        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata   = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.train.max_grad_norm
                )
                training_tds[-1].set("grad_norm", total_norm.mean())

                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()

        training_time  = time.time() - training_start
        iteration_time = sampling_time + training_time
        total_time    += iteration_time

        training_tds = torch.stack(training_tds)

        episode_r = tensordict_data.get(("next", "agents", "episode_reward"))
        episode_r = episode_r.reshape(
            cfg.env.num_envs, cfg.env.max_steps, env.n_agents, 1
        )
        mean_episode_reward = episode_r[:, -1].mean().item()

        per_agent_reward = episode_r[:, -1, :, 0].mean(0)   # [n_agents]

        avg_loss_objective = training_tds["loss_objective"].mean().item()
        avg_loss_critic    = training_tds["loss_critic"].mean().item()
        avg_loss_entropy   = training_tds["loss_entropy"].mean().item()
        avg_grad_norm      = training_tds["grad_norm"].mean().item()

        global_step = total_frames

        pos_entropy   = compute_positional_entropy(env)
        mean_goal_dist = compute_mean_goal_distance(env)
        domain_metrics = env.compute_metrics()         # scenario-specific

        if goal != env._goals[0][0].tolist():
            goal = env._goals[0][0].tolist()
            writer.add_scalar("GridWorld/Goal_Change", 1.0, global_step)
        else:
            writer.add_scalar("GridWorld/Goal_Change", 0.0, global_step)

        writer.add_scalar("Reward/mean_episode_reward", mean_episode_reward, global_step)
        for a_i in range(env.n_agents):
            writer.add_scalar(
                f"Reward/agent{a_i}_episode_reward",
                per_agent_reward[a_i].item(),
                global_step,
            )

        writer.add_scalar("Loss/objective", avg_loss_objective, global_step)
        writer.add_scalar("Loss/critic",    avg_loss_critic,    global_step)
        writer.add_scalar("Loss/entropy",   avg_loss_entropy,   global_step)
        writer.add_scalar(
            "Loss/total",
            avg_loss_objective + avg_loss_critic + avg_loss_entropy,
            global_step,
        )

        writer.add_scalar("Grad/grad_norm", avg_grad_norm, global_step)

        writer.add_scalar("Time/sampling_time",  sampling_time,  global_step)
        writer.add_scalar("Time/training_time",  training_time,  global_step)
        writer.add_scalar("Time/iteration_time", iteration_time, global_step)
        writer.add_scalar("Frames/total_frames", total_frames,   global_step)

        writer.add_scalar("GridWorld/positional_entropy", pos_entropy, global_step)

        if not np.isnan(mean_goal_dist):
            writer.add_scalar("GridWorld/mean_goal_distance", mean_goal_dist, global_step)

        for metric_name, metric_val in domain_metrics.items():
            writer.add_scalar(f"GridWorld/{metric_name}", metric_val, global_step)

        with torch.no_grad():
            obs_sample = tensordict_data.get(("agents", "observation"))[-1]  # last step slice
            # obs_sample: [num_envs, n_agents, obs_dim]
            td_sample   = tensordict_data[-1]
            td_policy   = policy(td_sample)
            logits      = td_policy.get(("agents", "logits"))        # [ne, n_agents, n_actions]
            action_probs = logits.softmax(-1).mean(0)                # [n_agents, n_actions]

        for a_i in range(env.n_agents):
            for act in range(env.n_actions):
                action_name = ["stay", "up", "down", "left", "right"][act]
                writer.add_scalar(
                    f"Policy/agent{a_i}_{action_name}",
                    action_probs[a_i, act].item(),
                    global_step,
                )

        domain_str = " | ".join(f"{k} {v:.3f}" for k, v in domain_metrics.items())
        torchrl_logger.info(
            f"Iter {i:4d} | "
            f"Frames {total_frames:8d} | "
            f"Reward {mean_episode_reward:7.3f} | "
            f"ObjLoss {avg_loss_objective:7.4f} | "
            f"CritLoss {avg_loss_critic:7.4f} | "
            f"Entropy {avg_loss_entropy:7.4f} | "
            f"PosEntropy {pos_entropy:.3f}"
            + (f" | {domain_str}" if domain_str else "")
        )

        metric_history.append({
            "step":                 global_step,
            "mean_episode_reward":  mean_episode_reward,
            "positional_entropy":   pos_entropy,
            "mean_goal_dist":       mean_goal_dist,
            **domain_metrics,
        })

        if i % eval_freq == 0 or i == cfg.collector.n_iters - 1:
            eval_reward = evaluate_policy(env_test=env_test, policy=policy)
            writer.add_scalar("Eval/mean_episode_reward", eval_reward, global_step)
            torchrl_logger.info(f"  -> Eval reward: {eval_reward:.3f}")

        sampling_start = time.time()

    traj_path = f"metric_trajectory_{cfg.env.scenario_name}_seed{cfg.seed}.npy"
    np.save(traj_path, metric_history, allow_pickle=True)
    torchrl_logger.info(f"Saved metric trajectory to {traj_path}")

    writer.close()
    collector.shutdown()
    if not env.is_closed:
        env.close()
    if not env_test.is_closed:
        env_test.close()


if __name__ == "__main__":
    train()