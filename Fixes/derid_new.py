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


class DeepERIDLoss(LossModule):

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
            critic_network: TensorDictModule,
            *,
            alpha: float = 0.1,
            entropy_bonus: bool = True,
            samples_mc_entropy: int = 1,
            entropy_coeff: float = 0.01,
            critic_coeff: float = 1.0,
            loss_critic_type: str = "smooth_l1",
            gamma: float = 0.99,
            reduction: str = "mean",
            functional: bool = True,
    ):
        self._functional = functional
        super().__init__()

        if functional:
            self.convert_to_functional(actor_network, "actor_network")
            self.convert_to_functional(critic_network, "critic_network")
            self.target_critic_network_params = deepcopy(self.critic_network_params)
        else:
            self.actor_network = actor_network
            self.critic_network = critic_network
            self.target_critic_network_params = None

        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_bonus = entropy_bonus
        self.reduction = reduction
        self.gamma = gamma
        self.loss_critic_type = loss_critic_type

        device = _get_default_device(self)

        self.register_buffer("alpha",         torch.as_tensor(alpha,         device=device))
        self.register_buffer("entropy_coeff", torch.as_tensor(entropy_coeff, device=device))
        self.register_buffer("critic_coeff",  torch.as_tensor(critic_coeff,  device=device))

        log_prob_keys = self.actor_network.log_prob_keys
        action_keys   = self.actor_network.dist_sample_keys
        if len(log_prob_keys) > 1:
            self.set_keys(sample_log_prob=log_prob_keys, action=action_keys)
        else:
            self.set_keys(sample_log_prob=log_prob_keys[0], action=action_keys[0])

        self._value_estimator = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
            *self.critic_network.in_keys,
            *[("next", key) for key in self.critic_network.in_keys],
        ]
        return list(set(keys))

    @property
    def out_keys(self):
        outs = ["loss_objective"]
        if self.critic_coeff > 0:
            outs.append("loss_critic")
        if self.entropy_bonus:
            outs.append("loss_entropy")
        return outs

    # ------------------------------------------------------------------
    # Target network Polyak update — called by training loop each step
    # ------------------------------------------------------------------

    def soft_update_target(self, tau: float = 0.005) -> None:
        """
        Polyak average: θ_target ← τ·θ_live + (1-τ)·θ_target
        Must be called after every optimizer step in the training loop.
        """
        if not self.functional or self.target_critic_network_params is None:
            return
        with torch.no_grad():
            for p_live, p_tgt in zip(
                self.critic_network_params.values(True, True),
                self.target_critic_network_params.values(True, True),
            ):
                p_tgt.data.lerp_(p_live.data, tau)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @set_composite_lp_aggregate(False)
    def get_entropy_bonus(self, dist: d.Distribution) -> torch.Tensor:
        if HAS_ENTROPY.get(type(dist), False):
            entropy = dist.entropy()
        else:
            x = dist.rsample((self.samples_mc_entropy,))
            log_prob = dist.log_prob(x)
            entropy = -log_prob.mean(0)
        return entropy.unsqueeze(-1)

    def _get_action_probs(self, tensordict: TensorDictBase) -> torch.Tensor:
        with (
            self.actor_network_params.to_module(self.actor_network)
            if self.functional
            else contextlib.nullcontext()
        ):
            dist = self.actor_network.get_dist(tensordict)

        if isinstance(dist, CompositeDistribution):
            dist = dist[0]
        return dist.probs  # (batch, n_actions)

    def _get_q_values(
            self, tensordict: TensorDictBase, use_target: bool = False
    ) -> torch.Tensor:
        if self.functional:
            params = (
                self.target_critic_network_params
                if use_target
                else self.critic_network_params
            )
            ctx = params.to_module(self.critic_network)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            q_out = self.critic_network(tensordict)

        return q_out.get("q_value")  # (batch, n_actions)

    # ------------------------------------------------------------------
    # Smith dynamics with Q-centering
    # ------------------------------------------------------------------

    def _policy_update(self, pi: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Centered Smith dynamics.

        FIX: subtract mean(Q) before building R so the dynamics are
        translation-invariant.  Without centering, a biased critic
        (e.g. all Q values shifted positive) creates a non-zero R
        even when the policy is already at the mixed NE, causing
        perpetual cycling in games like RPS.

        R[b, i, j] = [q̃_i - q̃_j]_+   (flow rate FROM j TO i)
        Inflow  to i:   Σ_j π[j] · R[b, i, j]   → einsum "bj,bij->bi"
        Outflow from i: π[i] · Σ_j R[b, j, i]   → pi * R.sum(dim=-2)
        """
        q_c = q - q.mean(dim=-1, keepdim=True)   # centered  (batch, n_actions)

        q_i = q_c.unsqueeze(-1)   # (batch, n_actions, 1)
        q_j = q_c.unsqueeze(-2)   # (batch, 1, n_actions)

        R = torch.clamp(q_i - q_j, min=0.0)          # (batch, n_actions, n_actions)

        term1 = torch.einsum("bj,bij->bi", pi, R)     # inflow
        term2 = pi * torch.sum(R, dim=-2)             # outflow

        pi_prime = pi + self.alpha * (term1 - term2)
        pi_prime = torch.clamp(pi_prime, min=0.0)
        pi_prime = pi_prime / pi_prime.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return pi_prime

    def _kl_divergence(self, pi_prime: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """KL(π' ‖ π) – works with any leading dimensions."""
        eps = 1e-8
        pi       = torch.clamp(pi,       min=eps)
        pi_prime = torch.clamp(pi_prime, min=eps)
        kl = (pi_prime * (pi_prime.log() - pi.log())).sum(dim=-1, keepdim=True)
        return kl

    # ------------------------------------------------------------------
    # Critic loss — policy-weighted (expected-SARSA) bootstrapping
    # ------------------------------------------------------------------

    def loss_critic(self, tensordict: TensorDictBase) -> torch.Tensor:
        """
        TD loss with policy-weighted bootstrap target:

            V(s') = Σ_a π(a | s') · Q_target(s', a)

        FIX: replaces the original max-Q (DQN-style) target.

        Why this matters for mixed-NE games (RPS):
        ───────────────────────────────────────────
        At the mixed NE, Q(s, a) = 0 for all actions.
        max_a Q(s', a) is always ≥ 0 (and strictly > 0 during training
        due to asymmetric initialisation), so max-bootstrapping
        systematically inflates one action's value, breaking the
        symmetry the mixed NE requires.
        The policy-weighted value V(s') = Σ_a π(a) Q(s', a) is
        unbiased: when π is at the NE and Q values are equal,
        V(s') = Q(s', a) for any a, preserving symmetry.
        """
        q_vals = self._get_q_values(tensordict, use_target=False)  # (batch, n_actions)

        next_td = tensordict.get("next")

        with torch.no_grad():
            next_q  = self._get_q_values(next_td, use_target=True)   # (batch, n_actions)
            pi_next = self._get_action_probs(next_td)                 # (batch, n_actions)
            v_next  = (pi_next * next_q).sum(dim=-1, keepdim=True)   # (batch, 1)

            reward   = next_td.get(self.tensor_keys.reward)           # (batch, 1)
            done     = next_td.get(self.tensor_keys.done).float()     # (batch, 1)
            target_q = reward + self.gamma * (1.0 - done) * v_next   # (batch, 1)

        action = tensordict.get(self.tensor_keys.action)
        if action.ndim > 1:
            action = action.squeeze(-1)
        q_selected = q_vals.gather(-1, action.unsqueeze(-1))  # (batch, 1)

        loss = distance_loss(target_q, q_selected, loss_function=self.loss_critic_type)
        return loss.mean() if self.reduction == "mean" else loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(recurse=False)

        pi     = self._get_action_probs(tensordict)          # shape: (..., n_agents, n_actions)
        q_vals = self._get_q_values(tensordict, use_target=False)  # same shape

        # Save original shape to restore later
        orig_shape = pi.shape
        n_agents = orig_shape[-2]
        n_actions = orig_shape[-1]

        # Flatten all leading dimensions into one batch dimension
        pi_flat = pi.reshape(-1, n_actions)
        q_vals_flat = q_vals.reshape(-1, n_actions)

        with torch.no_grad():
            pi_prime_flat = self._policy_update(pi_flat.detach(), q_vals_flat.detach())

        # Restore original shape
        pi_prime = pi_prime_flat.reshape(orig_shape)

        loss_kl = self._kl_divergence(pi_prime, pi)
        td_out  = TensorDict({"loss_objective": loss_kl}, batch_size=[])

        if self.entropy_bonus and self.entropy_coeff > 0:
            with (
                self.actor_network_params.to_module(self.actor_network)
                if self.functional
                else contextlib.nullcontext()
            ):
                dist = self.actor_network.get_dist(tensordict)
            entropy = self.get_entropy_bonus(dist)
            td_out.set("loss_entropy", -self.entropy_coeff * entropy)

        if self.critic_coeff > 0:
            loss_critic = self.loss_critic(tensordict)
            td_out.set("loss_critic", self.critic_coeff * loss_critic)

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