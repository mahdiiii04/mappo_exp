import time
import os

import numpy as np
import hydra
import torch

from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule, QValueModule, SafeSequential
from torchrl.modules.models.multiagent import MultiAgentMLP, QMixer, VDNMixer
from torchrl.objectives import SoftUpdate, ValueEstimators
from torchrl.objectives.multiagent.qmixer import QMixerLoss

from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig

from utils.utils import compute_nash_conv_qnet, evaluate_policy
from matrix_games import MatrixGameFactory


@hydra.main(version_base="1.1", config_path="", config_name="qmix_vdn")
def train(cfg: DictConfig):
    # Device
    cfg.train.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    cfg.env.device = cfg.train.device

    # Seeding
    torch.manual_seed(cfg.seed)

    # Sampling
    cfg.env.num_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch

    # Logging
    log_dir = os.path.join("tb_logs", cfg.sub_exp)
    writer = SummaryWriter(log_dir=log_dir)
    torchrl_logger.info(f"Tensorboard logging to: {log_dir}")

    # Environments
    env = MatrixGameFactory(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.num_envs,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
    )
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
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
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    # Policy network
    net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=env.full_action_spec_unbatched[env.action_key].shape[-1],
        n_agents=env.n_agents,
        centralised=False,
        share_params=cfg.model.shared_parameters,
        device=cfg.train.device,
        depth=2,
        num_cells=64,
        activation_class=nn.Tanh,
    )
    module = TensorDictModule(
        net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "action_value")],
    )
    value_module = QValueModule(
        action_value_key=("agents", "action_value"),
        out_keys=[
            env.action_key,
            ("agents", "action_value"),
            ("agents", "chosen_action_value"),
        ],
        spec=env.full_action_spec_unbatched,
        action_space=None,
    )
    qnet = SafeSequential(module, value_module)

    qnet_explore = TensorDictSequential(
        qnet,
        EGreedyModule(
            eps_init=cfg.loss.eps_init,
            eps_end=cfg.loss.eps_end,
            annealing_num_steps=int(cfg.collector.total_frames * cfg.loss.eps_anneal_frac),
            action_key=env.action_key,
            spec=env.full_action_spec_unbatched,
        ),
    )

    # Mixer
    if cfg.loss.mixer_type == "qmix":
        mixer = TensorDictModule(
            module=QMixer(
                state_shape=env.observation_spec_unbatched["agents", "observation"].shape,
                mixing_embed_dim=32,
                n_agents=env.n_agents,
                device=cfg.train.device,
            ),
            in_keys=[("agents", "chosen_action_value"), ("agents", "observation")],
            out_keys=["chosen_action_value"],
        )
    elif cfg.loss.mixer_type == "vdn":
        mixer = TensorDictModule(
            module=VDNMixer(
                n_agents=env.n_agents,
                device=cfg.train.device,
            ),
            in_keys=[("agents", "chosen_action_value")],
            out_keys=["chosen_action_value"],
        )
    else:
        raise ValueError(f"Unknown mixer_type: {cfg.loss.mixer_type}. Use 'qmix' or 'vdn'.")

    # Collector
    collector = Collector(
        env,
        qnet_explore,
        device=cfg.env.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
    )

    # Replay buffer
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    # Loss — done/terminated point to top-level keys that we populate below
    loss_module = QMixerLoss(qnet, mixer, delay_value=True)
    loss_module.set_keys(
        action_value=("agents", "action_value"),
        local_value=("agents", "chosen_action_value"),
        global_value="chosen_action_value",
        action=env.action_key,
        done="done",
        terminated="terminated",
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=cfg.loss.gamma)
    target_net_updater = SoftUpdate(loss_module, eps=1 - cfg.loss.tau)

    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)

    # Training loop
    total_time = 0
    total_frames = 0
    sampling_start = time.time()

    policy_history = []

    for i, tensordict_data in enumerate(collector):
        sampling_time = time.time() - sampling_start

        # QMIX/VDN needs a single global reward/done/terminated (no agent dim).
        # mean(-2) on reward collapses [*, n_agents, 1] -> [*, 1].
        # any(-2)  on done/terminated does the same, treating the episode as
        # done globally if any agent is done.
        tensordict_data.set(
            ("next", "reward"),
            tensordict_data.get(("next", env.reward_key)).mean(-2),
        )
        del tensordict_data["next", env.reward_key]

        tensordict_data.set(
            ("next", "episode_reward"),
            tensordict_data.get(("next", "agents", "episode_reward")).mean(-2),
        )
        del tensordict_data["next", "agents", "episode_reward"]

        # Flatten done/terminated to top-level so they match reward's shape
        tensordict_data.set(
            ("next", "done"),
            tensordict_data.get(("next", "agents", "done")).any(-2),
        )
        tensordict_data.set(
            ("next", "terminated"),
            tensordict_data.get(("next", "agents", "terminated")).any(-2),
        )

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

                loss_vals["loss"].backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.train.max_grad_norm
                )
                training_tds[-1].set("grad_norm", total_norm.mean())

                optim.step()
                optim.zero_grad()
                target_net_updater.step()

        qnet_explore[1].step(frames=current_frames)  # epsilon annealing
        collector.update_policy_weights_()

        training_time = time.time() - training_start
        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # Episode reward (same approach as mappo.py)
        episode_r = tensordict_data.get(("next", "episode_reward"))
        episode_r = episode_r.reshape(cfg.env.num_envs, cfg.env.max_steps, 1)
        mean_episode_reward = episode_r[:, -1].mean().item()

        avg_loss = training_tds["loss"].mean().item()
        avg_grad_norm = training_tds["grad_norm"].mean().item()

        global_step = total_frames

        # Nash convergence
        nash, avg_policy = compute_nash_conv_qnet(env, qnet)
        print(avg_policy)
        policy_history.append((global_step, avg_policy))

        print(env._payoff)
        if cfg.env.scenario_name == "biased_rps":
            current_phase = env._current_phase[0].item()
            writer.add_scalar("Env/current_phase", current_phase, global_step)

            if not hasattr(train, "_prev_phase"):
                train._prev_phase = current_phase
            elif current_phase != train._prev_phase:
                writer.add_scalar("Nash/Nash_Conv_phase_change", nash, global_step)
                torchrl_logger.info(
                    f"PHASE CHANGE at iteration {i} | Nash Conv {nash:.4f}"
                )
                train._prev_phase = current_phase

        for agent in range(env.n_agents):
            for action in range(env.n_actions):
                prob = avg_policy[agent, action].item()
                writer.add_scalar(f"Policy/agent{agent}_action{action}", prob, global_step)

        current_eps = qnet_explore[1].eps.item()
        writer.add_scalar("Reward/mean_episode_reward", mean_episode_reward, global_step)
        writer.add_scalar("Loss/qmixer_loss", avg_loss, global_step)
        writer.add_scalar("Grad/grad_norm", avg_grad_norm, global_step)
        writer.add_scalar("Exploration/epsilon", current_eps, global_step)
        writer.add_scalar("Time/sampling_time", sampling_time, global_step)
        writer.add_scalar("Time/training_time", training_time, global_step)
        writer.add_scalar("Time/iteration_time", iteration_time, global_step)
        writer.add_scalar("Frames/total_frames", total_frames, global_step)
        writer.add_scalar("Nash/Nash_Conv", nash, global_step)

        torchrl_logger.info(
            f"Iter {i} | "
            f"Frames: {total_frames} | "
            f"Mean Ep Reward {mean_episode_reward:.3f} | "
            f"Loss {avg_loss:.4f} | "
            f"Epsilon {current_eps:.4f} | "
            f"Nash Conv {nash:.4f}"
        )

        if i % cfg.eval.frequency == 0 or i == cfg.collector.n_iters - 1:
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                eval_reward = evaluate_policy(env_test=env_test, policy=qnet)

            writer.add_scalar("Eval/mean_episode_reward", eval_reward, total_frames)
            torchrl_logger.info(f"Evaluation Reward: {eval_reward:.3f}")

        sampling_start = time.time()

    # Save policy trajectory
    steps = np.array([step for step, _ in policy_history])
    policies = np.stack([p.numpy() for _, p in policy_history])

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