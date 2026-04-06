import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule

from torchrl.objectives import LossModule
from torchrl.objectives.utils import (
    _clip_value_loss,
    distance_loss,
    ValueEstimators,
    default_value_kwargs
)
from torchrl.objectives.value import GAE
from torchrl.envs.utils import ExplorationType, set_exploration_type

class PPOLoss(LossModule):
    """Re-implemntation of ClipPPOLoss from torchRL"""

    default_value_estimator = ValueEstimators.GAE

    def __init__(self, actor_network: TensorDictModule, value_network: TensorDictModule,
                 gamma: float = 0.99, lmbda: float = 0.95, clip_epsilon: float = 0.2,
                 entropy_coef: float = 0, critic_coef: float = 0.5, loss_function: str = "smooth_l1",
                 normalize_advantage: bool = True
                ):
        super().__init__()

        # converting our networks from stateful to stateless
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=False,
        )

        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=True,
            compare_against=(actor_network.parameters()),
        )

        # hyperparams
        self.gamma = gamma
        self.lmbda = lmbda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coef
        self.critic_coeff = critic_coef
        self.loss_function = loss_function
        self.normalize_advantage = normalize_advantage

        # create estimator
        self.make_value_estimator(ValueEstimators.GAE)


    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        
        if value_type is None:
            value_type = self.default_value_estimator

        hp = dict(default_value_kwargs(value_type))

        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        if hasattr(self, "lmbda"):
            hp["lmbda"] = self.lmbda

        hp.update(hyperparams)

        self._value_estimator = GAE(
            value_network=self.value_network,
            **hp,
        )

        self._value_estimator.set_keys(     # which key contains the value
            value="state_value"
        )

    def loss_actor(self, tensordict: TensorDict) -> torch.Tensor:
        """
        Computes cliped PPO policy loss 
        Uses importance sampling ratio with clipping
        """

        td = tensordict.clone(recurse=False)

        log_prob_old = td.get("sample_log_prob")

        with self.actor_netwrok_params.to_module(self.actor_network):
            td = self.actor_network(td)
            log_prob_new = td.get("sample_log_prob")

        ratio = torch.exp(log_prob_new - log_prob_old)

        advantage = td.get("advantage")

        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage

        actor_loss = -torch.min(surr1, surr2).mean()

        entropy = td.get("entropy", None)
        if entropy is not None:
            actor_loss = actor_loss - self.entropy_coeff * entropy.mean()

        return actor_loss
    
    def loss_value(self, tensordict: TensorDict):
        """
        Computes value loss using GAE targets
        """
        td = tensordict.clone(recurse=False)

        with self.value_network_params.to_module(self.value_network):
            td = self.value_network(td)

        current_value = td.get("state_value").squeeze(-1)

        with self.target_value_network_params.to_module(self.value_network):
            target_td = self._value_estimator(td)

        advantage = target_td.get("advantage").squeeze(-1)
        value_target = target_td.get("value_target").squeeze(-1)

        td.set("advantage", advantage.unsqueeze(-1))
        td.set("value_target", value_target.unsqueeze(-1))

        value_loss = distance_loss(
            current_value,
            value_target,
            loss_function=self.loss_function
        )

        return value_loss, td
    
    def forward(self, tensordict: TensorDict) -> TensorDict:
        
        value_loss, tensordict = self.loss_value(tensordict)

        actor_loss = self.loss_actor(tensordict)

        total_loss = (
            actor_loss + self.critic_coeff * value_loss
        )

        return TensorDict(
            {
                "loss" : total_loss,
                "loss_actor" : actor_loss.detach(),
                "loss_value" : value_loss.detach(),
                "entropy" : tensordict.get("entropy", torch.tensor(0.0)).mean().detach(),
                "clip_fraction" : self._compute_clip_fraction(tensordict).detach(),
            },
            batch_size=[],
        )
    
    def _compute_clip_fraction(self, tensordict: TensorDict) -> torch.Tensor:

        log_prob_old = tensordict.get("sample_log_prob")

        with self.actor_network_params(self.actor_network):
            td = self.actor_network(tensordict.clone(recurse=False))
            log_prob_new = td.get("sample_log_prob")

        ratio = torch.exp(log_prob_new - log_prob_old)
        clipped = (ratio < (1.0 - self.clip_epsilon)) | (ratio > (1.0 + self.clip_epsilon))
        return clipped.float().mean()