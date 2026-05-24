import time
import os

import hydra
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import TensorDictReplayBuffer, Composite
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv, Compose
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig

from utils.utils import DoneTransform


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(cfg, seed, groups=None):
    """Create a VMAS env wrapped with per-group RewardSum transforms.

    Returns the wrapped env and the list of group names so callers that
    create the env first can pass the discovered groups to subsequent calls.
    """
    scenario_kwargs = cfg.env.scenario if cfg.env.scenario is not None else {}
    base_env = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.num_envs,
        continuous_actions=False,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=seed,
        **scenario_kwargs,
    )

    if groups is None:
        groups = list(base_env.group_map.keys())

    reward_sums = [
        RewardSum(
            in_keys=[(group, "reward")],
            out_keys=[(group, "episode_reward")],
        )
        for group in groups
    ]

    env = TransformedEnv(base_env, Compose(*reward_sums))
    return env, groups


# ---------------------------------------------------------------------------
# Per-group model builders
# ---------------------------------------------------------------------------

def build_policy(group, env, cfg):
    """Decentralised actor for one agent group."""
    n_agents = len(env.group_map[group])
    obs_dim  = env.observation_spec[group, "observation"].shape[-1]
    act_dim  = env.full_action_spec_unbatched[group, "action"].shape[-1]

    net = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=act_dim,
            n_agents=n_agents,
            centralized=False,
            share_params=cfg.model.shared_params,
            device=cfg.train.device,
            depth=2,
            num_cells=256,
            activation_class=nn.Tanh,
        )
    )

    module = TensorDictModule(
        net,
        in_keys=[(group, "observation")],
        out_keys=[(group, "logits")],
    )

    policy = ProbabilisticActor(
        module,
        spec=Composite({(group, "action"): env.full_action_spec_unbatched[group, "action"]}),
        in_keys=[(group, "logits")],
        out_keys=[(group, "action")],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )
    return policy


def build_critic(group, env, cfg):
    """Centralised-or-decentralised critic for one agent group."""
    n_agents = len(env.group_map[group])
    obs_dim  = env.observation_spec[group, "observation"].shape[-1]

    net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=1,
        n_agents=n_agents,
        centralized=cfg.model.centralised_critic,
        share_params=cfg.model.shared_params,
        device=cfg.train.device,
        depth=2,
        num_cells=128,
        activation_class=nn.Tanh,
    )

    # Use a group-scoped out_key to prevent collisions between groups.
    return ValueOperator(net, in_keys=[(group, "observation")], out_keys=[(group, "state_value")])


def build_loss(group, policy, critic, cfg):
    """ClipPPO loss wired to the correct per-group keys."""
    loss = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coeff=cfg.loss.entropy_eps,
        normalize_advantage=False,
    )
    # All intermediate keys are group-scoped so GAE outputs for "adversary"
    # (1 agent) never collide with those for "agent" (2 agents).
    loss.set_keys(
        reward=(group, "reward"),
        action=(group, "action"),
        done=(group, "done"),
        terminated=(group, "terminated"),
        value=(group, "state_value"),
        advantage=(group, "advantage"),
        value_target=(group, "value_target"),
    )
    loss.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.loss.gamma, lmbda=cfg.loss.lmbda
    )
    return loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(env_test, policy, groups):
    policy.eval()
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        td = env_test.rollout(
            max_steps=env_test.max_steps,
            policy=policy,
            auto_reset=True,
            break_when_any_done=False,
            tensordict=env_test.reset(),
        )

    rewards = {}
    for group in groups:
        done = td.get((group, "done"))
        final_rewards = td.get((group, "episode_reward"))[done]
        rewards[group] = final_rewards.mean().item() if final_rewards.numel() > 0 else 0.0

    policy.train()
    return rewards


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.1", config_path="", config_name="mappo")
def train(cfg: DictConfig):
    # Device
    cfg.train.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    cfg.env.device   = cfg.train.device

    torch.manual_seed(cfg.seed)

    # Sampling sizes
    cfg.env.num_envs          = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size    = cfg.collector.frames_per_batch

    # Logging
    log_dir = os.path.join("tb_logs", f"{cfg.env.scenario_name}-seed-{cfg.seed}")
    writer  = SummaryWriter(log_dir=log_dir)
    torchrl_logger.info(f"Tensorboard logging to: {log_dir}")

    # -----------------------------------------------------------------------
    # Environments
    # -----------------------------------------------------------------------
    env,      groups = make_env(cfg, cfg.seed)
    env_test, _      = make_env(cfg, cfg.seed, groups=groups)

    torchrl_logger.info(f"Agent groups discovered: {groups}")

    # -----------------------------------------------------------------------
    # Per-group policies, critics and losses
    # -----------------------------------------------------------------------
    policies      = {g: build_policy(g, env, cfg) for g in groups}
    critics       = {g: build_critic(g, env, cfg) for g in groups}
    loss_modules  = {g: build_loss(g, policies[g], critics[g], cfg) for g in groups}

    # Single policy seen by the collector: runs each group's actor in sequence
    combined_policy = TensorDictSequential(*[policies[g] for g in groups])

    # -----------------------------------------------------------------------
    # Collector
    # -----------------------------------------------------------------------
    # One DoneTransform per group broadcasts the done signal to match the
    # reward shape that PPO expects.
    postproc = Compose(*[
        DoneTransform(
            reward_key=(group, "reward"),
            done_keys=env.done_keys,
        )
        for group in groups
    ])

    collector = Collector(
        env,
        combined_policy,
        device=cfg.train.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        postproc=postproc,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    # -----------------------------------------------------------------------
    # Optimiser — single Adam over all group parameters
    # -----------------------------------------------------------------------
    all_params = [p for loss in loss_modules.values() for p in loss.parameters()]
    optim      = torch.optim.Adam(params=all_params, lr=cfg.train.lr)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    total_time   = 0
    total_frames = 0
    eval_freq    = cfg.eval.frequency
    sampling_start = time.time()

    for i, tensordict_data in enumerate(collector):
        sampling_time = time.time() - sampling_start

        # GAE for every group
        with torch.no_grad():
            for loss_module in loss_modules.values():
                loss_module.value_estimator(
                    tensordict_data,
                    params=loss_module.critic_network_params,
                    target_params=loss_module.target_critic_network_params,
                )

        total_frames += tensordict_data.numel()
        replay_buffer.extend(tensordict_data.reshape(-1))

        training_tds   = {g: [] for g in groups}
        training_start = time.time()

        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata    = replay_buffer.sample()
                total_loss = torch.zeros(1, device=cfg.train.device)

                for group, loss_module in loss_modules.items():
                    loss_vals = loss_module(subdata)
                    training_tds[group].append(loss_vals.detach())
                    total_loss = total_loss + (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                total_loss.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    all_params, cfg.train.max_grad_norm
                )
                # Attach grad norm to the last recorded td of each group
                for group in groups:
                    training_tds[group][-1].set("grad_norm", total_norm.mean())

                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()

        training_time  = time.time() - training_start
        iteration_time = sampling_time + training_time
        total_time    += iteration_time
        global_step    = total_frames

        # -------------------------------------------------------------------
        # Logging — per group
        # -------------------------------------------------------------------
        for group in groups:
            tds  = torch.stack(training_tds[group])
            done = tensordict_data.get((group, "done"))
            final_rewards = tensordict_data.get((group, "episode_reward"))[done]
            mean_ep_reward = (
                final_rewards.mean().item() if final_rewards.numel() > 0 else 0.0
            )

            writer.add_scalar(f"Reward/{group}/mean_episode_reward", mean_ep_reward, global_step)
            writer.add_scalar(f"Loss/{group}/objective", tds["loss_objective"].mean().item(), global_step)
            writer.add_scalar(f"Loss/{group}/critic",    tds["loss_critic"].mean().item(),    global_step)
            writer.add_scalar(f"Loss/{group}/entropy",   tds["loss_entropy"].mean().item(),   global_step)
            writer.add_scalar(f"Grad/{group}/grad_norm", tds["grad_norm"].mean().item(),      global_step)

            torchrl_logger.info(
                f"Iter {i} | Group: {group:12s} | "
                f"Frames: {total_frames} | "
                f"Mean Ep Reward: {mean_ep_reward:.3f} | "
                f"Obj Loss: {tds['loss_objective'].mean().item():.4f} | "
                f"Critic Loss: {tds['loss_critic'].mean().item():.4f}"
            )

        writer.add_scalar("Time/sampling_time",  sampling_time,  global_step)
        writer.add_scalar("Time/training_time",  training_time,  global_step)
        writer.add_scalar("Time/iteration_time", iteration_time, global_step)
        writer.add_scalar("Frames/total_frames", total_frames,   global_step)

        # -------------------------------------------------------------------
        # Evaluation
        # -------------------------------------------------------------------
        if i % eval_freq == 0 or i == cfg.collector.n_iters - 1:
            eval_rewards = evaluate_policy(env_test, combined_policy, groups)
            for group, reward in eval_rewards.items():
                writer.add_scalar(f"Eval/{group}/mean_episode_reward", reward, total_frames)
                torchrl_logger.info(f"Eval | Group: {group:12s} | Reward: {reward:.3f}")

    writer.close()
    collector.shutdown()
    if not env.is_closed:
        env.close()
    if not env_test.is_closed:
        env_test.close()


if __name__ == "__main__":
    train()