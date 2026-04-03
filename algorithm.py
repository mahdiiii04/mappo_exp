import torch
from tqdm import tqdm
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from torchrl.objectives import ClipPPOLoss, ValueEstimators

from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal


class MAPPO:

    def __init__(self, env, env_device, minibatch_size, 
                 lr=3e-4, clip_epsilon=0.2,
                 gamma=0.99, lmbda=0.9, entropy_eps=1e-4,
                 max_grad_norm=1.0, share_params_policy=True, 
                 share_params_critic=True, frames_per_batch=6_000,
                 n_iters=5, agentic_keyword="agents"):
        
        self.device = (torch.device(0) if torch.cuda.is_available() else torch.device("cpu"))
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.max_grad_norm = max_grad_norm
        self.n_iters = n_iters
        self.frames_per_batch = frames_per_batch
        self.minibatch_size = minibatch_size
        
        self.agentic_keyword = agentic_keyword

        self.share_params_policy = share_params_policy
        self.share_params_critic = share_params_critic

        self.env = env

        self.init_policy()
        self.init_critic()

        self.collector = SyncDataCollector(
            self.env,
            self.policy,
            device=env_device,
            storing_device=self.device,
            frames_per_batch=frames_per_batch,
            total_frames=self.frames_per_batch * self.n_iters,
        )

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                frames_per_batch, device=self.device
            ),
            sampler=SamplerWithoutReplacement(),
            batch_size=minibatch_size,
        )

        self.loss_module = ClipPPOLoss(
            self.policy,
            self.critic,
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=self.entropy_eps,
            normalize_advantage=False,
        )

        self.loss_module.set_keys(
            reward=self.env.reward_key,
            action=self.env.action_key,
            value=(self.agentic_keyword, "state_value"),
            done=(self.agentic_keyword, "done"),
            terminated=(self.agentic_keyword, "terminated"),
        )

        self.loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.gamma, lmbda=self.lmbda
        )

        self.GAE = self.loss_module.value_estimator

        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=lr)

    def init_policy(self):

        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=self.env.observation_spec[self.agentic_keyword, "observation"].shape[-1],
                n_agent_outputs=2 * self.env.full_action_spec[self.env.action_key].shape[-1],
                n_agents=self.env.n_agents,
                centralized=False,
                share_params=self.share_params_policy,
                device=self.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            NormalParamExtractor(),
        )

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[(self.agentic_keyword, "observation")],
            out_keys=[(self.agentic_keyword, "loc"), (self.agentic_keyword, "scale")],
        )

        self.policy = ProbabilisticActor(
            module=policy_module,
            spec=self.env.action_spec_unbatched,
            in_keys=[(self.agentic_keyword, "loc"), (self.agentic_keyword, "scale")],
            out_keys=[self.env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low" : self.env.full_action_spec_unbatched[self.env.action_key].space.low,
                "high" : self.env.full_action_spec_unbatched[self.env.action_key].space.high
            },
            return_log_prob=True,
        )

    
    def init_critic(self):

        critic_net = MultiAgentMLP(
            n_agent_inputs=self.env.observation_spec[self.agentic_keyword, "observation"].shape[-1],
            n_agent_outputs=1,
            n_agents=self.env.n_agents,
            centralized=True,
            share_params=self.share_params_critic,
            device=self.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        self.critic = TensorDictModule(
            module=critic_net,
            in_keys=[(self.agentic_keyword, "observation")],
            out_keys=[(self.agentic_keyword, "state_value")]
        )

    def train(self, num_epochs):

        pbar = tqdm(total=self.n_iters, desc="episode_reward_mean = 0")

        episode_reward_mean_list = []

        for tensordict_data in self.collector:
            tensordict_data.set(
                ("next", self.agentic_keyword, "done"),
                tensordict_data.get(("next", "done")).unsqueeze(-1).expand(tensordict_data.get_item_shape(("next", self.env.reward_key)))
            )
            tensordict_data.set(
                ("next", self.agentic_keyword, "terminated"),
                tensordict_data.get(("next", "terminated")).unsqueeze(-1).expand(tensordict_data.get_item_shape(("next", self.env.reward_key)))
            )

            with torch.no_grad():
                self.GAE(
                    tensordict_data,
                    params=self.loss_module.critic_network_params,
                    target_params=self.loss_module.target_critic_network_params,
                )

            data_view = tensordict_data.reshape(-1)
            self.replay_buffer.extend(data_view)

            for _ in range(num_epochs):
                for _ in range(self.frames_per_batch //self.minibatch_size):
                    subdata = self.replay_buffer.sample()
                    loss_vals = self.loss_module(subdata)

                    loss_value = (
                        loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                    )

                    loss_value.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), self.max_grad_norm
                    )

                    self.optim.step()
                    self.optim.zero_grad()

            self.collector.update_policy_weights_()

            done = tensordict_data.get(("next", self.agentic_keyword, "done"))
            episode_reward_mean = (
                tensordict_data.get(("next", self.agentic_keyword, "episode_reward"))[done].mean().item()
            )
            episode_reward_mean_list.append(episode_reward_mean)
            pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
            pbar.update()

        return episode_reward_mean_list            