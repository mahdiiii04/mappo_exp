import time
import os

import hydra
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv, SerialEnv
from torchrl.envs.libs.pettingzoo import PettingZooEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ValueEstimators

from torch.utils.tensorboard import SummaryWriter

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from utils.utils import DoneTransform
from utils.losses.derid import DeepERIDLoss

# ── MPE key constants ────────────────────────────────────────────────────────
# TorchRL's PettingZoo wrapper uses "agent" (singular), confirmed from reward spec.
REWARD_KEY = ("agent", "reward")
DONE_KEYS  = [("agent", "done"), ("agent", "terminated")]


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))


def make_env(scenario_name: str, seed: int, **scenario_kwargs) -> TransformedEnv:
    """
    Factory for a single PettingZoo MPE environment with cumulative reward tracking.
    Called once per SerialEnv worker.
    """
    base = PettingZooEnv(
        task=scenario_name,
        parallel=True,
        seed=seed,
        continuous_actions=False,
        **scenario_kwargs,
    )
    return TransformedEnv(
        base,
        RewardSum(
            in_keys=[REWARD_KEY],
            out_keys=[("agent", "episode_reward")],
        ),
    )


def evaluate_policy(env_test, policy, max_steps):
    policy.eval()

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        td = env_test.rollout(
            max_steps=max_steps,
            policy=policy,
            auto_reset=True,
            break_when_any_done=False,
            tensordict=env_test.reset(),
        )

        done = td.get(("agent", "done"))
        final_rewards = td.get(("agent", "episode_reward"))[done]
        if final_rewards.numel() > 0:
            mean_episode_reward = final_rewards.mean().item()
        else:
            mean_episode_reward = td.get(("agent", "episode_reward")).mean().item()

    policy.train()
    return mean_episode_reward


@hydra.main(version_base="1.1", config_path="", config_name="deep_erid")
def train(cfg: DictConfig):
    # ── device ────────────────────────────────────────────────────────────────
    cfg.train.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    cfg.env.device = cfg.train.device

    torch.manual_seed(cfg.seed)

    # ── frame / buffer accounting ────────────────────────────────────────────
    cfg.env.num_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch

    # ── logging ───────────────────────────────────────────────────────────────
    log_dir = os.path.join(get_original_cwd(), "tb_logs", cfg.sub_exp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    torchrl_logger.info(f"Tensorboard logging to: {log_dir}")

    # ── probe env for dims ────────────────────────────────────────────────────
    # PettingZooEnv wraps aec_to_parallel_wrapper which doesn't forward .n_agents.
    # Read everything from specs instead:
    #   obs shape is (n_agents, obs_dim)  → shape[0] / shape[-1]
    #   action dim comes from action_spec[action_key].space.n
    _probe = make_env(cfg.env.scenario_name, cfg.seed, **cfg.env.scenario)
    obs_shape  = _probe.observation_spec["agent", "observation"].shape
    n_agents   = obs_shape[0]
    obs_dim    = obs_shape[-1]
    action_key = ("agent", "action")
    action_dim = _probe.action_spec[action_key].space.n
    _probe.close()

    torchrl_logger.info(
        f"MPE env: {cfg.env.scenario_name} | "
        f"n_agents={n_agents} | obs_dim={obs_dim} | action_dim={action_dim}"
    )

    # ── environments ──────────────────────────────────────────────────────────
    def env_fn():
        return make_env(cfg.env.scenario_name, cfg.seed, **cfg.env.scenario)

    env      = SerialEnv(cfg.env.num_envs, env_fn)
    # Single env for evaluation — avoids SerialEnv lazy worker init
    # which hangs when reset() is first called mid-run.
    env_test = make_env(cfg.env.scenario_name, cfg.seed, **cfg.env.scenario)

    # ── policy ────────────────────────────────────────────────────────────────
    policy_net = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=action_dim,
            n_agents=n_agents,
            centralized=False,
            share_params=cfg.model.shared_params,
            device=cfg.train.device,
            depth=2,
            num_cells=256,
            activation_class=nn.Tanh,
        ),
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agent", "observation")],
        out_keys=[("agent", "logits")],
    )

    policy = ProbabilisticActor(
        policy_module,
        spec=env.action_spec,          # PettingZooEnv uses action_spec
        in_keys=[("agent", "logits")],
        out_keys=[action_key],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )

    # ── critic ────────────────────────────────────────────────────────────────
    critic_net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=action_dim,    # Q-value per action (DeepERID critic)
        n_agents=n_agents,
        centralized=cfg.model.centralised_critic,
        share_params=cfg.model.shared_params,
        device=cfg.train.device,
        depth=2,
        num_cells=128,
        activation_class=nn.Tanh,
    )

    critic = ValueOperator(
        critic_net,
        in_keys=[("agent", "observation")],
        out_keys=["q_value"],
    )

    # ── collector ─────────────────────────────────────────────────────────────
    collector = Collector(
        env,
        policy,
        device=cfg.train.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        postproc=DoneTransform(reward_key=REWARD_KEY, done_keys=DONE_KEYS),
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    # ── DeepERID loss ─────────────────────────────────────────────────────────
    loss_module = DeepERIDLoss(
        actor_network=policy,
        critic_network=critic,
        entropy_coeff=cfg.loss.entropy_eps,
        alpha=cfg.loss.alpha,
        gamma=cfg.loss.gamma,
    )

    loss_module.set_keys(
        reward=REWARD_KEY,
        action=action_key,
        done=("agent", "done"),
        terminated=("agent", "terminated"),
        sample_log_prob=("agent", "action_log_prob"),
    )

    if loss_module.functional:
        actor_params  = list(loss_module.actor_network_params.values(True, True))
        critic_params = list(loss_module.critic_network_params.values(True, True))
    else:
        actor_params  = list(loss_module.actor_network.parameters())
        critic_params = list(loss_module.critic_network.parameters())

    actor_optim  = torch.optim.Adam(actor_params,  lr=cfg.train.actor_lr)
    critic_optim = torch.optim.Adam(critic_params, lr=cfg.train.critic_lr)

    # ── training loop ─────────────────────────────────────────────────────────
    total_time   = 0
    total_frames = 0
    sampling_start = time.time()
    eval_freq = cfg.eval.frequency

    for i, tensordict_data in enumerate(collector):
        sampling_time = time.time() - sampling_start

        current_frames = tensordict_data.numel()
        total_frames  += current_frames
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()
        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata   = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                # --- Critic step ---
                critic_optim.zero_grad()
                loss_vals["loss_critic"].backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(critic_params, cfg.train.max_grad_norm)
                critic_optim.step()

                # --- Actor step ---
                actor_optim.zero_grad()
                actor_loss = loss_vals["loss_objective"] + loss_vals["loss_entropy"]
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_params, cfg.train.max_grad_norm)
                actor_optim.step()

                loss_module.soft_update_target(tau=cfg.train.tau)

                total_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in actor_params + critic_params
                    if p.grad is not None
                ) ** 0.5
                training_tds[-1].set(
                    "grad_norm", torch.tensor(total_norm, device=cfg.train.device)
                )

        loss_module.soft_update_avg_actor(tau=0.02)
        collector.update_policy_weights_()

        training_time  = time.time() - training_start
        iteration_time = sampling_time + training_time
        total_time    += iteration_time
        training_tds   = torch.stack(training_tds)

        # ── logging ───────────────────────────────────────────────────────────
        done = tensordict_data.get(("agent", "done"))
        final_rewards = tensordict_data.get(("agent", "episode_reward"))[done]
        if final_rewards.numel() > 0:
            mean_episode_reward = final_rewards.mean().item()
        else:
            mean_episode_reward = tensordict_data.get(("agent", "episode_reward")).mean().item()

        avg_loss_objective = training_tds["loss_objective"].mean().item()
        avg_loss_critic    = training_tds["loss_critic"].mean().item()
        avg_loss_entropy   = training_tds["loss_entropy"].mean().item()
        avg_grad_norm      = training_tds["grad_norm"].mean().item()

        global_step = total_frames

        writer.add_scalar("Reward/mean_episode_reward", mean_episode_reward, global_step)
        writer.add_scalar("Loss/objective", avg_loss_objective, global_step)
        writer.add_scalar("Loss/critic",    avg_loss_critic,    global_step)
        writer.add_scalar("Loss/entropy",   avg_loss_entropy,   global_step)
        writer.add_scalar("Loss/total",
            avg_loss_objective + avg_loss_critic + avg_loss_entropy, global_step)
        writer.add_scalar("Grad/grad_norm", avg_grad_norm, global_step)
        writer.add_scalar("Time/sampling_time",  sampling_time,  global_step)
        writer.add_scalar("Time/training_time",  training_time,  global_step)
        writer.add_scalar("Time/iteration_time", iteration_time, global_step)
        writer.add_scalar("Frames/total_frames", total_frames,   global_step)

        print(f"Reward Scale: {loss_module.reward_scale}")

        torchrl_logger.info(
            f"Iter {i} | "
            f"Frames: {total_frames} | "
            f"Mean Ep Reward {mean_episode_reward:.3f} | "
            f"Objective Loss {avg_loss_objective:.4f} | "
            f"Critic Loss {avg_loss_critic:.4f}"
        )

        if i % eval_freq == 0 or i == cfg.collector.n_iters - 1:
            eval_reward = evaluate_policy(env_test=env_test, policy=policy, max_steps=cfg.env.max_steps)
            writer.add_scalar("Eval/mean_episode_reward", eval_reward, total_frames)
            torchrl_logger.info(f"Evaluation Reward: {eval_reward:.3f}")

        sampling_start = time.time()

    writer.close()
    collector.shutdown()
    if not env.is_closed:
        env.close()
    if not env_test.is_closed:
        env_test.close()


if __name__ == "__main__":
    train()