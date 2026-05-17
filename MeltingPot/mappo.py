import time
import os

import numpy as np
import hydra
import torch
import torch.nn.functional as F

from tensordict.nn import TensorDictModule
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, SerialEnv, TransformedEnv
from torchrl.envs.libs.meltingpot import MeltingpotEnv
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig

from utils.utils import evaluate_policy


# ── Shared CNN backbone + MLP head for MeltingPot agents ─────────────────────
# Input: RGB obs of shape (..., n_agents, H, W, C)  (uint8, 0-255)
# Output: logits/value of shape (..., n_agents, out_dim)
class AgentCNNMLP(nn.Module):
    """
    Per-agent CNN that processes each agent's RGB frame independently.
    Expects input shape: (*, n_agents, H, W, C) as uint8 or float.
    """
    def __init__(self, obs_shape, out_dim, device):
        super().__init__()
        H, W, C = obs_shape          # e.g. 88, 88, 3
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=2, padding=1),   # -> 44x44
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> 22x22
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # -> 11x11
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out = 32 * ((H // 8) * (W // 8))   # rough; computed exactly below
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            cnn_out = self.cnn(dummy).shape[-1]

        self.mlp = nn.Sequential(
            nn.Linear(cnn_out, 128), nn.Tanh(),
            nn.Linear(128, 64),     nn.Tanh(),
            nn.Linear(64, out_dim),
        )
        self.to(device)

    def forward(self, rgb):
        # rgb: (*, n_agents, H, W, C)  uint8 -> normalise to float [0,1]
        x = rgb.float() / 255.0
        lead = x.shape[:-4]          # everything before (n_agents, H, W, C)
        n_agents, H, W, C = x.shape[-4:]
        x = x.reshape(-1, H, W, C)              # (B*n_agents, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B*n_agents, C, H, W)
        feat = self.cnn(x)                       # (B*n_agents, cnn_out)
        out  = self.mlp(feat)                    # (B*n_agents, out_dim)
        out  = out.reshape(*lead, n_agents, -1)  # (*, n_agents, out_dim)
        return out
# ─────────────────────────────────────────────────────────────────────────────


@hydra.main(version_base="1.1", config_path="", config_name="mappo")
def train(cfg: DictConfig):

    cfg.train.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    cfg.env.device   = cfg.train.device

    torch.manual_seed(cfg.seed)

    cfg.env.num_envs           = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size     = cfg.collector.frames_per_batch

    assert cfg.env.num_envs >= 1, (
        f"frames_per_batch ({cfg.collector.frames_per_batch}) must be >= max_steps ({cfg.env.max_steps})"
    )
    assert cfg.collector.frames_per_batch % cfg.train.minibatch_size == 0, (
        f"frames_per_batch ({cfg.collector.frames_per_batch}) must be divisible "
        f"by minibatch_size ({cfg.train.minibatch_size})"
    )
    # Rough memory estimate for the RGB replay buffer (uint8 stored, float32 during forward)
    _rgb_bytes = cfg.buffer.memory_size * 7 * 88 * 88 * 3  # uint8 bytes
    torchrl_logger.info(
        f"Approx RGB buffer size: {_rgb_bytes / 1e6:.1f} MB "
        f"(frames={cfg.collector.frames_per_batch}, num_envs={cfg.env.num_envs})"
    )

    log_dir = os.path.join("tb_logs", cfg.sub_exp)
    writer  = SummaryWriter(log_dir=log_dir)
    torchrl_logger.info(f"Tensorboard logging to: {log_dir}")

    # ── Probe a single raw env to read all static metadata ───────────
    # MeltingpotEnv has no n_agents attribute; read from spec shape[0].
    # Obs is a Composite under ("agents","observation") with key "RGB"
    # shape (n_agents, H, W, C).  Action is Categorical shape (n_agents,).
    _probe      = MeltingpotEnv(cfg.env.substrate_name)
    action_key  = _probe.action_key                          # ("agents","action")
    reward_key  = _probe.reward_key                          # ("agents","reward")
    action_spec = _probe.full_action_spec_unbatched
    n_agents    = action_spec[action_key].shape[0]           # 7
    n_actions   = action_spec[action_key].space.n            # 8  (Categorical)
    rgb_shape   = tuple(
        _probe.observation_spec[("agents", "observation", "RGB")].shape[1:]
    )                                                        # (H, W, C) = (88,88,3)
    _probe.close()
    # ─────────────────────────────────────────────────────────────────

    def make_env(seed_offset=0):
        def make_single():
            e = MeltingpotEnv(cfg.env.substrate_name)
            e = TransformedEnv(
                e,
                RewardSum(
                    in_keys=[e.reward_key],
                    out_keys=[("agents", "episode_reward")],
                ),
            )
            return e
        return SerialEnv(cfg.env.num_envs, make_single, device=cfg.env.device)

    env      = make_env(seed_offset=0)
    env_test = make_env(seed_offset=1)

    torchrl_logger.info(
        f"Substrate: {cfg.env.substrate_name} | "
        f"Agents: {n_agents} | "
        f"RGB shape: {rgb_shape} | "
        f"Actions: {n_actions}"
    )

    # ── Policy network ────────────────────────────────────────────────
    policy_net = AgentCNNMLP(
        obs_shape=rgb_shape,
        out_dim=n_actions,
        device=cfg.train.device,
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation", "RGB")],
        out_keys=[("agents", "logits")],
    )

    policy = ProbabilisticActor(
        policy_module,
        spec=action_spec,
        in_keys=[("agents", "logits")],
        out_keys=[action_key],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
        log_prob_key=("agents", "action_log_prob"),
    )

    # ── Critic network ────────────────────────────────────────────────
    critic_net = AgentCNNMLP(
        obs_shape=rgb_shape,
        out_dim=1,
        device=cfg.train.device,
    )

    critic = ValueOperator(
        nn.Sequential(
            critic_net,
            # squeeze the last dim so output is (*, n_agents) not (*, n_agents, 1)
            # ValueOperator expects (..., 1) -- keep it; torchrl handles both
        ),
        in_keys=[("agents", "observation", "RGB")],
        out_keys=[("agents", "state_value")],
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
        reward=reward_key,
        action=action_key,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
        sample_log_prob=("agents", "action_log_prob"),
        value=("agents", "state_value"),
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE,
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.lmbda,
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg.train.lr)

    total_time     = 0
    total_frames   = 0
    sampling_start = time.time()
    eval_freq      = cfg.eval.frequency
    metric_history = []

    for i, tensordict_data in enumerate(collector):
        sampling_time = time.time() - sampling_start

        with torch.no_grad():
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        current_frames  = tensordict_data.numel()
        total_frames   += current_frames
        data_view       = tensordict_data.reshape(-1)
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
        episode_r = episode_r.reshape(cfg.env.num_envs, cfg.env.max_steps, n_agents, 1)
        mean_episode_reward = episode_r[:, -1].mean().item()
        per_agent_reward    = episode_r[:, -1, :, 0].mean(0)   # (n_agents,)

        avg_loss_objective = training_tds["loss_objective"].mean().item()
        avg_loss_critic    = training_tds["loss_critic"].mean().item()
        avg_loss_entropy   = training_tds["loss_entropy"].mean().item()
        avg_grad_norm      = training_tds["grad_norm"].mean().item()

        global_step = total_frames

        writer.add_scalar("Reward/mean_episode_reward", mean_episode_reward, global_step)
        for a_i in range(n_agents):
            writer.add_scalar(
                f"Reward/agent{a_i}_episode_reward",
                per_agent_reward[a_i].item(), global_step,
            )

        writer.add_scalar("Loss/objective", avg_loss_objective, global_step)
        writer.add_scalar("Loss/critic",    avg_loss_critic,    global_step)
        writer.add_scalar("Loss/entropy",   avg_loss_entropy,   global_step)
        writer.add_scalar(
            "Loss/total",
            avg_loss_objective + avg_loss_critic + avg_loss_entropy, global_step,
        )
        writer.add_scalar("Grad/grad_norm",      avg_grad_norm,  global_step)
        writer.add_scalar("Time/sampling_time",  sampling_time,  global_step)
        writer.add_scalar("Time/training_time",  training_time,  global_step)
        writer.add_scalar("Time/iteration_time", iteration_time, global_step)
        writer.add_scalar("Frames/total_frames", total_frames,   global_step)

        with torch.no_grad():
            td_policy    = policy(tensordict_data[-1])
            logits       = td_policy.get(("agents", "logits"))   # (ne, na, n_actions)
            action_probs = logits.softmax(-1).mean(0)             # (na, n_actions)

        for a_i in range(n_agents):
            for act in range(n_actions):
                writer.add_scalar(
                    f"Policy/agent{a_i}_action{act}",
                    action_probs[a_i, act].item(), global_step,
                )

        torchrl_logger.info(
            f"Iter {i:4d} | "
            f"Frames {total_frames:8d} | "
            f"Reward {mean_episode_reward:7.3f} | "
            f"ObjLoss {avg_loss_objective:7.4f} | "
            f"CritLoss {avg_loss_critic:7.4f} | "
            f"Entropy {avg_loss_entropy:7.4f}"
        )

        metric_history.append({"step": global_step, "mean_episode_reward": mean_episode_reward})

        if i % eval_freq == 0 or i == cfg.collector.n_iters - 1:
            eval_reward = evaluate_policy(env_test=env_test, policy=policy)
            writer.add_scalar("Eval/mean_episode_reward", eval_reward, global_step)
            torchrl_logger.info(f"  -> Eval reward: {eval_reward:.3f}")

        sampling_start = time.time()

    traj_path = f"metric_trajectory_{cfg.env.substrate_name}_seed{cfg.seed}.npy"
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

    # ── Read static env metadata from a raw (unwrapped) env to avoid
    #    the RecursionError that occurs when __getattr__ is called on
    #    a deeply nested SerialEnv / TransformedEnv stack.
    #
    #    MeltingpotEnv does NOT expose n_agents as an attribute; it is
    #    encoded in the observation spec shape:
    #      observation_spec[("agents","observation")].shape == (n_agents, obs_dim)
    # ──────────────────────────────────────────────────────────────────
    _probe      = MeltingpotEnv(cfg.env.substrate_name)

    # ── Diagnostic: print everything we need to pick the right keys ──
    torchrl_logger.info(f"[PROBE] observation_spec:\n{_probe.observation_spec}")
    torchrl_logger.info(f"[PROBE] action_spec:\n{_probe.full_action_spec_unbatched}")
    torchrl_logger.info(f"[PROBE] reward_key : {_probe.reward_key}")
    torchrl_logger.info(f"[PROBE] action_key : {_probe.action_key}")
    torchrl_logger.info(f"[PROBE] batch_size : {_probe.batch_size}")
    # check for any 'n_agents'-like attributes
    for _attr in ("n_agents", "num_agents", "_n_agents", "n_players"):
        torchrl_logger.info(
            f"[PROBE] hasattr({_attr}): {hasattr(_probe, _attr)}"
            + (f" = {getattr(_probe, _attr)}" if hasattr(_probe, _attr) else "")
        )
    _probe.close()
    raise SystemExit("Diagnostic complete — check logs above, then remove this block.")
    # ──────────────────────────────────────────────────────────────────

    def make_env(seed_offset=0):
        def make_single():
            env = MeltingpotEnv(cfg.env.substrate_name)
            env = TransformedEnv(
                env,
                RewardSum(
                    in_keys=[env.reward_key],
                    out_keys=[("agents", "episode_reward")],
                ),
            )
            return env

        return SerialEnv(cfg.env.num_envs, make_single, device=cfg.env.device)

    env      = make_env(seed_offset=0)
    env_test = make_env(seed_offset=1)

    torchrl_logger.info(
        f"Substrate: {cfg.env.substrate_name} | "
        f"Agents: {n_agents} | "
        f"Obs dim: {obs_dim} | "
        f"Actions: {n_actions}"
    )

    policy_net = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=n_actions,
            n_agents=n_agents,
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
        spec=action_spec,
        in_keys=[("agents", "logits")],
        out_keys=[action_key],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )

    critic_net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=1,
        n_agents=n_agents,
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
        reward=reward_key,
        action=action_key,
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
            cfg.env.num_envs, cfg.env.max_steps, n_agents, 1
        )
        mean_episode_reward = episode_r[:, -1].mean().item()
        per_agent_reward    = episode_r[:, -1, :, 0].mean(0)   # (n_agents,)

        avg_loss_objective = training_tds["loss_objective"].mean().item()
        avg_loss_critic    = training_tds["loss_critic"].mean().item()
        avg_loss_entropy   = training_tds["loss_entropy"].mean().item()
        avg_grad_norm      = training_tds["grad_norm"].mean().item()

        global_step = total_frames

        writer.add_scalar("Reward/mean_episode_reward", mean_episode_reward, global_step)
        for a_i in range(n_agents):
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
        writer.add_scalar("Grad/grad_norm",      avg_grad_norm,  global_step)
        writer.add_scalar("Time/sampling_time",  sampling_time,  global_step)
        writer.add_scalar("Time/training_time",  training_time,  global_step)
        writer.add_scalar("Time/iteration_time", iteration_time, global_step)
        writer.add_scalar("Frames/total_frames", total_frames,   global_step)

        with torch.no_grad():
            td_policy    = policy(tensordict_data[-1])
            logits       = td_policy.get(("agents", "logits"))  # (ne, na, n_actions)
            action_probs = logits.softmax(-1).mean(0)            # (na, n_actions)

        for a_i in range(n_agents):
            for act in range(n_actions):
                writer.add_scalar(
                    f"Policy/agent{a_i}_action{act}",
                    action_probs[a_i, act].item(),
                    global_step,
                )

        torchrl_logger.info(
            f"Iter {i:4d} | "
            f"Frames {total_frames:8d} | "
            f"Reward {mean_episode_reward:7.3f} | "
            f"ObjLoss {avg_loss_objective:7.4f} | "
            f"CritLoss {avg_loss_critic:7.4f} | "
            f"Entropy {avg_loss_entropy:7.4f}"
        )

        metric_history.append({
            "step":                global_step,
            "mean_episode_reward": mean_episode_reward,
        })

        if i % eval_freq == 0 or i == cfg.collector.n_iters - 1:
            eval_reward = evaluate_policy(env_test=env_test, policy=policy)
            writer.add_scalar("Eval/mean_episode_reward", eval_reward, global_step)
            torchrl_logger.info(f"  -> Eval reward: {eval_reward:.3f}")

        sampling_start = time.time()

    traj_path = f"metric_trajectory_{cfg.env.substrate_name}_seed{cfg.seed}.npy"
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