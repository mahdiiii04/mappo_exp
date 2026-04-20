import contextlib
from copy import deepcopy
from dataclasses import dataclass

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import (
    CompositeDistribution,
    dispatch,
    ProbabilisticTensorDictSequential,
    set_composite_lp_aggregate,
    TensorDictModule,
    composite_lp_aggregate,
)
from tensordict.utils import NestedKey
from torch import distributions as d
from torchrl.modules.distributions import HAS_ENTROPY
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _cache_values,
    _get_default_device,
    _reduce,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)

class ERIDLoss(LossModule):

    @dataclass
    class _AcceptedKeys:
        action: NestedKey = "action"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        sample_log_prob: NestedKey | None = None

        def __post_init__(self):
            if self.sample_log_prob is None:
                if composite_lp_aggregate(nowarn=True):
                    self.sample_log_prob = "sample_log_prob"
                else:
                    self.sample_log_prob = "action_log_prob"

    default_keys = _AcceptedKeys
    tensor_keys: _AcceptedKeys
    _schedulable_buffers = frozenset({"entropy_coeff", "critic_coeff", "alpha"})

    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams | None
    critic_network_params: TensorDictParams | None
    target_critic_network_params: TensorDictParams | None
    target_actor_network_params: TensorDictParams | None

    def __init__(
            self,
            actor_network: ProbabilisticTensorDictSequential,
            *,
            alpha: float = 0.1,
            entropy_bonus: bool = True,
            samples_mc_entropy: int = 1,
            entropy_coeff: float = 0.01,
            loss_critic_type: str = "smooth_l1",
            gamma: float = 0.99,
            reduction: str = "mean",
            functional: bool = True,
    ):
        self._functional = functional  
        super().__init__()

        if functional:
            self.convert_to_functional(actor_network, "actor_network")
        else:
            self.actor_network = actor_network

        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_bonus = entropy_bonus
        self.reduction = reduction
        self.gamma = gamma

        device = _get_default_device(self)

        self.register_buffer(
            "alpha", torch.as_tensor(alpha, device=device)
        )
        self.register_buffer(
            "entropy_coeff", torch.as_tensor(entropy_coeff, device=device)
        )

        log_prob_keys = self.actor_network.log_prob_keys
        action_keys = self.actor_network.dist_sample_keys
        if len(log_prob_keys) > 1:
            self.set_keys(sample_log_prob=log_prob_keys, action=action_keys)
        else:
            self.set_keys(sample_log_prob=log_prob_keys[0], action=action_keys[0])

        self._value_estimator = None

    @property
    def functional(self):
        return self._functional   

    @property
    def in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
            *self.actor_network.in_keys,
            *[("next", key) for key in self.actor_network.in_keys],
        ]
        return list(set(keys))

    @property
    def out_keys(self):
        outs = ["loss_objective"]   # KL(π' || π)
        if self.entropy_bonus:
            outs.append("loss_entropy")
        return outs
    
    @set_composite_lp_aggregate(False)
    def get_entropy_bonus(self, dist: d.Distribution) -> torch.Tensor:
        if HAS_ENTROPY.get(type(dist), False):
            entropy = dist.entropy()
        else:
            x = dist.rsample((self.samples_mc_entropy,))
            log_prob = dist.log_prob(x)
            entropy = -log_prob.mean(0)
        return entropy.unsqueeze(-1)
    
    def _get_action_probs(
            self, tensordict: TensorDictBase
    ) -> torch.Tensor:
        with (
            self.actor_network_params.to_module(self.actor_network)
            if self.functional
            else contextlib.nullcontext()
        ):
            dist = self.actor_network.get_dist(tensordict)
        
        if isinstance(dist, CompositeDistribution):
            dist = dist[0]
        probs = dist.probs  # (batch, n_actions)
        return probs

    def _get_average_rewards(
            self, tensordict: TensorDictBase
    ) -> torch.Tensor:
        
        next_tensordict = tensordict.get("next")
        rewards = next_tensordict.get(self.tensor_keys.reward)
        actions = tensordict.get(self.tensor_keys.action)

        B = rewards.shape[0]
        n_agents = rewards.shape[1]
        n_actions = self._get_n_actions_from_pi(tensordict)
        
        if actions.shape[-1] == n_actions:
            actions_onehot = actions.float()
        else:
            actions_idx = actions.long()
            actions_onehot = torch.zeros(
                (B , n_agents, n_actions),
                device=actions.device,
                dtype=torch.float32,
            )
            actions_onehot.scatter_(-1, actions_idx.unsqueeze(-1), 1.0)

        weighted = rewards.expand(B, n_agents, n_actions) * actions_onehot

        sum_r = weighted.sum(dim=0)
        counts = actions_onehot.sum(dim=0)

        bar_r = sum_r / counts.clamp(min=1e-8)

        return bar_r
    
    def _get_n_actions_from_pi(
            self, tensordict: TensorDictBase
    ) -> int:
        with torch.no_grad():
            pi = self._get_action_probs(tensordict)
        return pi.shape[-1]   

    def _replicator_update(
            self, pi: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        
        pi_prime = pi * torch.exp(self.alpha * q)
        pi_prime = pi_prime / (pi_prime.sum(dim=-1, keepdim=True) + 1e-8)

        return pi_prime
    
    def _policy_update(
            self, pi: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        
        q_centered = q 
        
        q_i = q_centered.unsqueeze(-1)   # (batch, n_actions, 1)
        q_j = q_centered.unsqueeze(-2)   # (batch, 1, n_actions)

        R = torch.clamp(q_i - q_j, min=0.0)  # (batch, n_actions, n_actions)

        # Inflow to i: agents on j switch to i when i is better
        # = Σ_j π[j] * R[b, i, j]
        term1 = torch.einsum("...j,...ij->...i", pi, R)  

        # Outflow from i: agents on i switch to j when j is better
        # = π[i] * Σ_j R[b, j, i]
        term2 = pi * torch.sum(R, dim=-2)          

        pi_prime = pi + self.alpha * (term1 - term2)

        """if torch.rand(1).item() < 0.01:
            print(f"R: {R[0]}")
            print(f"Pi': {pi_prime[0]}")
        """
        # project back onto the simplex
        pi_prime = torch.clamp(pi_prime, min=0.0)
        pi_prime = pi_prime / pi_prime.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        
        return pi_prime
    
    def _kl_divergence(
            self, pi_prime: torch.Tensor, pi: torch.Tensor
    ) -> torch.Tensor:
        """
        KL(π' || π) = Σ_a π'(a) * log(π'(a) / π(a))
        """
        eps = 1e-8
        pi       = torch.clamp(pi,       min=eps)
        pi_prime = torch.clamp(pi_prime, min=eps)
        kl = (pi_prime * (pi_prime.log() - pi.log())).sum(dim=-1)  
        return kl.unsqueeze(-1)

    @dispatch
    def forward(
        self, tensordict: TensorDictBase
    ) -> TensorDictBase:
        
        tensordict = tensordict.clone(recurse=False)

        pi = self._get_action_probs(tensordict)

        bar_r = self._get_average_rewards(tensordict)

        batch_size = pi.shape[0]
        q_vals = bar_r.unsqueeze(0).expand(batch_size, -1, -1)
        
        with torch.no_grad():
            pi_prime = self._replicator_update(pi, q_vals)

        loss_kl = self._kl_divergence(pi_prime, pi)

        td_out = TensorDict({"loss_objective": loss_kl}, batch_size=[])

        if self.entropy_bonus and self.entropy_coeff > 0:
            with (
                self.actor_network_params.to_module(self.actor_network)
                if self.functional
                else contextlib.nullcontext()
            ):
                dist = self.actor_network.get_dist(tensordict)

            entropy = self.get_entropy_bonus(dist)
            td_out.set("loss_entropy", -self.entropy_coeff * entropy)

        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value
        )

        return td_out

    def make_value_estimator(self, *args, **kwargs):
        pass

    def _forward_value_estimator_keys(self, **kwargs):
        pass