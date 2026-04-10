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
from torchrl.envs import RewardSum, TransformedEnv
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

def evaluate_policy(env_test, policy):
    policy.eval()

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        td = env_test.rollout(
            max_steps=env_test.max_steps,
            policy=policy,
            auto_reset=True,
            break_when_any_done=False,
            tensordict=env_test.reset(),
        )

        #episode_rewards = td.get(("agents", "episode_reward")).sum(-1)
        #mean_episode_reward = episode_rewards.mean().item()

        done = td.get(("agents", "done"))
        final_rewards = td.get(("agents", "episode_reward"))[done]
        mean_episode_reward = final_rewards.mean().item()

    policy.train()
    return mean_episode_reward

@hydra.main(version_base="1.1", config_path="", config_name="mappo")
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
    log_dir = os.path.join("tb_logs", f"{cfg.env.scenario_name}-seed-{cfg.seed}")
    writer = SummaryWriter(log_dir=log_dir)
    torchrl_logger.info(f"Tensorboard logging to: {log_dir}")

    # environments
    env = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.num_envs,
        continuous_actions=False,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        **cfg.env.scenario,
    )

    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")])
    )

    env_test = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.num_envs,
        continuous_actions=False,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        **cfg.env.scenario,
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
            num_cells=256,
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
    critic_module = MultiAgentMLP(
        n_agent_inputs=env.observation_spec[("agents", "observation")].shape[-1],
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralized=cfg.model.centralised_critic,
        share_params=cfg.model.shared_params,
        device=cfg.train.device,
        depth=2,
        num_cells=256,
        activation_class=nn.Tanh,
    )

    critic = ValueOperator(
        critic_module,
        in_keys=[("agents", "observation")],
    )

    # dealing with env data

    collector = Collector(
        env,
        policy,
        device=cfg.train.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys),
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    # ppo loss 

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coeff=cfg.loss.entropy_eps,
        normalize_advantage=False,
    )

    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.loss.gamma, lmbda=cfg.loss.lmbda
    )

    optim = torch.optim.Adam(params=loss_module.parameters(), lr=cfg.train.lr)

    # training loop 

    total_time = 0
    total_frames = 0
    sampling_start = time.time()

    eval_freq = cfg.eval.frequency

    for i, tensordict_data in enumerate(collector):
        sampling_time = time.time() - sampling_start

        with torch.no_grad():
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,  # later scith to None 
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

                with torch.no_grad():
                    rewards = subdata.get(("next", env.reward_key))
                    values = subdata.get("state_value")
                    advantages = subdata.get("advantage")
                    logits = subdata.get(("agents", "logits"))
                    print(f"Reward: mean={rewards.mean().item():.4f}, std={rewards.std().item():.4f}")
                    print(f"Value: mean={values.mean().item():.4f}, std={values.std().item():.4f}")
                    print(f"Advantage: mean={advantages.mean().item():.5f}, std={advantages.std().item():.5f}")
                    print(f"Logit: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")

                loss_value = (
                    loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.train.max_grad_norm
                )
                training_tds[-1].set("grad_norm", total_norm.mean())

                optim.step()
                optim.zero_grad()

        
        collector.update_policy_weights_()

        training_time = time.time() - training_start

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # logging
        #mean_reward = tensordict_data.get(("agents", "episode_reward")).mean().item()
        #mean_episode_reward = tensordict_data.get(("agents", "episode_reward")).sum(-1).mean().item() # per episode

        done = tensordict_data.get(("agents", "done"))
        final_rewards = tensordict_data.get(("agents", "episode_reward"))[done]
        mean_episode_reward = final_rewards.mean().item()

        avg_loss_objective = training_tds["loss_objective"].mean().item()
        avg_loss_critic = training_tds["loss_critic"].mean().item()
        avg_loss_entropy = training_tds["loss_entropy"].mean().item()
        avg_grad_norm = training_tds["grad_norm"].mean().item()

        global_step = total_frames

        writer.add_scalar("Reward/mean_episode_reward", mean_episode_reward, global_step)
        #writer.add_scalar("Reward/mean_agent_reward", mean_reward, global_step)

        writer.add_scalar("Loss/objective", avg_loss_objective, global_step)
        writer.add_scalar("Loss/critic", avg_loss_critic, global_step)
        writer.add_scalar("Loss/entropy", avg_loss_entropy, global_step)
        writer.add_scalar("Loss/total", avg_loss_objective + avg_loss_critic + avg_loss_entropy, global_step)

        writer.add_scalar("Grad/grad_norm", avg_grad_norm, global_step)
        
        writer.add_scalar("Time/sampling_time", sampling_time, global_step)
        writer.add_scalar("Time/training_time", training_time, global_step)
        writer.add_scalar("Time/iteration_time", iteration_time, global_step)

        writer.add_scalar("Frames/total_frames", total_frames, global_step)

        torchrl_logger.info(
            f"Iter {i} | "
            f"Frames: {total_frames} | "
            f"Mean Ep Reward {mean_episode_reward:.3f} | "
            f"Objective Loss {avg_loss_objective:.4f} | "
            f"Critic Loss {avg_loss_critic:.4f}"
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

    writer.close()
    collector.shutdown()
    if not env.is_closed:
        env.close()
    if not env_test.is_closed:
        env_test.close()


if __name__ == "__main__":
    train()



