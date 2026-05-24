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

from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig

from utils.utils import DoneTransform
from utils.losses.derid import DeepERIDLoss


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(cfg, seed, groups=None):
    """Create a VMAS env wrapped with per-group RewardSum transforms."""
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
    """Q-network (per-action values) for one agent group.

    n_agent_outputs matches the action dimension so the network outputs one
    Q-value per discrete action, which DeepERIDLoss expects.
    """
    n_agents = len(env.group_map[group])
    obs_dim  = env.observation_spec[group, "observation"].shape[-1]
    act_dim  = env.full_action_spec_unbatched[group, "action"].shape[-1]

    net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=act_dim,
        n_agents=n_agents,
        centralized=cfg.model.centralised_critic,
        share_params=cfg.model.shared_params,
        device=cfg.train.device,
        depth=2,
        num_cells=128,
        activation_class=nn.Tanh,
    )

    # Use flat "q_value" key so DeepERIDLoss._get_q_values() can find it via
    # q_out.get("q_value").  Each group's loss has its own critic instance so
    # there is no collision between groups.
    return ValueOperator(
        net,
        in_keys=[(group, "observation")],
        out_keys=["q_value"],
    )


def build_loss(group, n_agents, policy, critic, cfg):
    """DeepERID loss wired to the correct per-group keys.

    Also patches the forward pass to guarantee the action tensor always has
    shape (batch, n_agents) before derid's loss_critic runs gather() on it.
    Without the patch, a 1-agent group produces action shape (batch,) after
    derid's squeeze(-1), making unsqueeze(-1) yield a 2-D tensor that cannot
    be gathered against the 3-D q_vals tensor.
    """
    loss = DeepERIDLoss(
        actor_network=policy,
        critic_network=critic,
        entropy_coeff=cfg.loss.entropy_eps,
        alpha=cfg.loss.alpha,
        gamma=cfg.loss.gamma,
    )
    loss.set_keys(
        reward=(group, "reward"),
        action=(group, "action"),
        done=(group, "done"),
        terminated=(group, "terminated"),
        sample_log_prob=(group, "action_log_prob"),
    )

    # --- action-shape guard patched onto loss_critic directly.
    # Patching forward() doesn't work: torchrl's @dispatch decorator unwraps
    # the tensordict and calls the real forward(), bypassing any tensordict
    # we modified there.  loss_critic() is called after dispatch resolves,
    # so patching it is reliable.
    _orig_loss_critic = loss.loss_critic

    def _safe_loss_critic(tensordict):
        # derid.py line 246-248 does:
        #   if action.ndim > 1: action = action.squeeze(-1)
        #   q_selected = q_vals.gather(-1, action.unsqueeze(-1))
        # q_vals is always 3-D: (batch, n_agents, n_actions).
        # gather needs the index to also be 3-D: (batch, n_agents, 1).
        # For a 1-agent group action arrives as (batch, 1):
        #   squeeze(-1) → (batch,)  [ndim=1]
        #   unsqueeze(-1) → (batch, 1)  [ndim=2]  ← mismatch with 3-D q_vals
        # Fix: unsqueeze action to (batch, n_agents, 1) BEFORE derid's squeeze,
        # so derid's squeeze(-1) removes our trailing 1 → (batch, n_agents),
        # and then its unsqueeze(-1) restores → (batch, n_agents, 1) correctly.
        action_key = loss.tensor_keys.action
        action = tensordict.get(action_key)
        if action.ndim == 2:
            # (batch, n_agents) → (batch, n_agents, 1) so squeeze(-1) is safe
            action = action.unsqueeze(-1)
        elif action.ndim == 1:
            # (batch,) → (batch, 1, 1)
            action = action.unsqueeze(-1).unsqueeze(-1)
        # Now action is (batch, n_agents, 1): derid squeeze(-1)→(batch, n_agents)
        # then unsqueeze(-1)→(batch, n_agents, 1) matching 3-D q_vals.
        td = tensordict.copy()
        td.set(action_key, action)
        return _orig_loss_critic(td)

    loss.loss_critic = _safe_loss_critic
    # ---

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

@hydra.main(version_base="1.1", config_path="", config_name="deep_erid")
def train(cfg: DictConfig):
    # Device
    cfg.train.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    cfg.env.device   = cfg.train.device

    torch.manual_seed(cfg.seed)

    # Sampling sizes
    cfg.env.num_envs           = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size     = cfg.collector.frames_per_batch

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
    policies     = {g: build_policy(g, env, cfg) for g in groups}
    critics      = {g: build_critic(g, env, cfg) for g in groups}
    loss_modules = {
        g: build_loss(g, len(env.group_map[g]), policies[g], critics[g], cfg)
        for g in groups
    }

    # Single policy seen by the collector: runs each group's actor in sequence
    combined_policy = TensorDictSequential(*[policies[g] for g in groups])

    # -----------------------------------------------------------------------
    # Collector
    # -----------------------------------------------------------------------
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
    # Per-group optimisers (actor and critic are updated on separate schedules)
    # -----------------------------------------------------------------------
    def _params(loss_module, kind):
        if loss_module.functional:
            params = loss_module.actor_network_params if kind == "actor" else loss_module.critic_network_params
            return list(params.values(True, True))
        else:
            net = loss_module.actor_network if kind == "actor" else loss_module.critic_network
            return list(net.parameters())

    actor_optims  = {g: torch.optim.Adam(_params(loss_modules[g], "actor"),  lr=cfg.train.actor_lr)  for g in groups}
    critic_optims = {g: torch.optim.Adam(_params(loss_modules[g], "critic"), lr=cfg.train.critic_lr) for g in groups}
    actor_params  = {g: _params(loss_modules[g], "actor")  for g in groups}
    critic_params = {g: _params(loss_modules[g], "critic") for g in groups}

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    total_time   = 0
    total_frames = 0
    eval_freq    = cfg.eval.frequency
    sampling_start = time.time()

    for i, tensordict_data in enumerate(collector):
        sampling_time = time.time() - sampling_start

        total_frames += tensordict_data.numel()
        replay_buffer.extend(tensordict_data.reshape(-1))

        training_tds   = {g: [] for g in groups}
        training_start = time.time()

        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata = replay_buffer.sample()

                for group, loss_module in loss_modules.items():
                    loss_vals = loss_module(subdata)
                    training_tds[group].append(loss_vals.detach())

                    # Critic step
                    critic_optims[group].zero_grad()
                    loss_vals["loss_critic"].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(critic_params[group], cfg.train.max_grad_norm)
                    critic_optims[group].step()

                    # Actor step
                    actor_optims[group].zero_grad()
                    (loss_vals["loss_objective"] + loss_vals["loss_entropy"]).backward()
                    torch.nn.utils.clip_grad_norm_(actor_params[group], cfg.train.max_grad_norm)
                    actor_optims[group].step()

                    loss_module.soft_update_target(tau=cfg.train.tau)

                    total_norm = sum(
                        p.grad.norm().item() ** 2
                        for p in actor_params[group] + critic_params[group]
                        if p.grad is not None
                    ) ** 0.5
                    training_tds[group][-1].set(
                        "grad_norm", torch.tensor(total_norm, device=cfg.train.device)
                    )

        for loss_module in loss_modules.values():
            loss_module.soft_update_avg_actor(tau=0.02)

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

            writer.add_scalar(f"Reward/{group}/mean_episode_reward", mean_ep_reward,                    global_step)
            writer.add_scalar(f"Loss/{group}/objective",             tds["loss_objective"].mean().item(), global_step)
            writer.add_scalar(f"Loss/{group}/critic",                tds["loss_critic"].mean().item(),    global_step)
            writer.add_scalar(f"Loss/{group}/entropy",               tds["loss_entropy"].mean().item(),   global_step)
            writer.add_scalar(f"Grad/{group}/grad_norm",             tds["grad_norm"].mean().item(),      global_step)

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