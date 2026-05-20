# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
from copy import deepcopy
from dataclasses import dataclass

import torch
from tensordict import (
    is_tensor_collection,
    TensorDict,
    TensorDictBase,
    TensorDictParams,
)
from tensordict.nn import (
    composite_lp_aggregate,
    CompositeDistribution,
    dispatch,
    ProbabilisticTensorDictSequential,
    set_composite_lp_aggregate,
    TensorDictModule,
)
from tensordict.utils import NestedKey
from torch import distributions as d

from torchrl.modules.distributions import HAS_ENTROPY
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _cache_values,
    _clip_value_loss,
    _GAMMA_LMBDA_DEPREC_ERROR,
    _get_default_device,
    _reduce,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import (
    GAE,
    TD0Estimator,
    TD1Estimator,
    TDLambdaEstimator,
    ValueEstimatorBase,
    VTrace,
)


class A2CLoss(LossModule):
    @dataclass
    class _AcceptedKeys:
        
        advantage: NestedKey = "advantage"
        value_target: NestedKey = "value_target"
        value: NestedKey = "state_value"
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
    default_value_estimator: ValueEstimators = ValueEstimators.GAE
    _schedulable_buffers = frozenset({"entropy_coeff", "critic_coeff", "clip_value"})

    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams | None
    critic_network_params: TensorDictParams | None
    target_actor_network_params: TensorDictParams | None
    target_critic_network_params: TensorDictParams | None

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential = None,
        critic_network: TensorDictModule = None,
        *,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coeff: float | None = None,
        critic_coeff: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        gamma: float | None = None,
        separate_losses: bool = False,
        advantage_key: str | None = None,
        value_target_key: str | None = None,
        functional: bool = True,
        actor: ProbabilisticTensorDictSequential = None,
        critic: ProbabilisticTensorDictSequential = None,
        reduction: str | None = None,
        clip_value: float | None = None,
        **kwargs,
    ):
        # entropy_coef has been removed in v0.11
        if "entropy_coef" in kwargs:
            raise TypeError(
                "'entropy_coef' has been removed in torchrl v0.11. Please use 'entropy_coeff' instead."
            )

        # Set default value if None
        if entropy_coeff is None:
            entropy_coeff = 0.01

        # critic_coef has been removed in v0.11
        if "critic_coef" in kwargs:
            raise TypeError(
                "'critic_coef' has been removed in torchrl v0.11. Please use 'critic_coeff' instead."
            )

        if actor is not None:
            actor_network = actor
            del actor
        if critic is not None:
            critic_network = critic
            del critic
        if actor_network is None or critic_network is None:
            raise TypeError(
                "Missing positional arguments actor_network or critic_network."
            )
        if reduction is None:
            reduction = "mean"

        self._functional = functional
        self._out_keys = None
        super().__init__()
        self._set_deprecated_ctor_keys(
            advantage=advantage_key, value_target=value_target_key
        )

        if functional:
            self.convert_to_functional(
                actor_network,
                "actor_network",
            )
        else:
            self.actor_network = actor_network

        if separate_losses:
            # we want to make sure there are no duplicates in the params: the
            # params of critic must be refs to actor if they're shared
            policy_params = list(actor_network.parameters())
        else:
            policy_params = None
        if functional:
            self.convert_to_functional(
                critic_network, "critic_network", compare_against=policy_params
            )
        else:
            self.critic_network = critic_network
            self.target_critic_network_params = None

        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_bonus = entropy_bonus and entropy_coeff
        self.reduction = reduction

        device = _get_default_device(self)

        self.register_buffer(
            "entropy_coeff", torch.as_tensor(entropy_coeff, device=device)
        )
        if critic_coeff is not None:
            self.register_buffer(
                "critic_coeff", torch.as_tensor(critic_coeff, device=device)
            )
        else:
            self.critic_coeff = None

        if gamma is not None:
            raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)
        self.loss_critic_type = loss_critic_type

        if clip_value is not None:
            if isinstance(clip_value, float):
                clip_value = torch.tensor(clip_value)
            elif isinstance(clip_value, torch.Tensor):
                if clip_value.numel() != 1:
                    raise ValueError(
                        f"clip_value must be a float or a scalar tensor, got {clip_value}."
                    )
            else:
                raise ValueError(
                    f"clip_value must be a float or a scalar tensor, got {clip_value}."
                )
            self.register_buffer(
                "clip_value", torch.as_tensor(clip_value, device=device)
            )
        else:
            self.clip_value = None

        log_prob_keys = self.actor_network.log_prob_keys
        action_keys = self.actor_network.dist_sample_keys
        if len(log_prob_keys) > 1:
            self.set_keys(sample_log_prob=log_prob_keys, action=action_keys)
        else:
            self.set_keys(sample_log_prob=log_prob_keys[0], action=action_keys[0])

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
        if self.critic_coeff is not None:
            keys.extend(self.critic_network.in_keys)
        return list(set(keys))

    @property
    def out_keys(self):
        if self._out_keys is None:
            outs = ["loss_objective"]
            if self.critic_coeff is not None:
                outs.append("loss_critic")
            if self.entropy_bonus:
                outs.append("entropy")
                outs.append("loss_entropy")
            self._out_keys = outs
        return self._out_keys

    @out_keys.setter
    def out_keys(self, value):
        self._out_keys = value

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                advantage=self.tensor_keys.advantage,
                value_target=self.tensor_keys.value_target,
                value=self.tensor_keys.value,
                reward=self.tensor_keys.reward,
                done=self.tensor_keys.done,
                terminated=self.tensor_keys.terminated,
            )

    def reset(self) -> None:
        pass

    @set_composite_lp_aggregate(False)
    def get_entropy_bonus(self, dist: d.Distribution) -> torch.Tensor:
        if HAS_ENTROPY.get(type(dist), False):
            entropy = dist.entropy()
        else:
            x = dist.rsample((self.samples_mc_entropy,))
            log_prob = dist.log_prob(x)
            if is_tensor_collection(log_prob):
                log_prob = sum(log_prob.sum(dim="feature").values(True, True))
            entropy = -log_prob.mean(0)
        return entropy.unsqueeze(-1)

    @set_composite_lp_aggregate(False)
    def _log_probs(
        self, tensordict: TensorDictBase
    ) -> tuple[torch.Tensor, d.Distribution]:
        # current log_prob of actions
        tensordict_clone = tensordict.select(
            *self.actor_network.in_keys, strict=False
        ).copy()
        with (
            self.actor_network_params.to_module(self.actor_network)
            if self.functional
            else contextlib.nullcontext()
        ):
            dist = self.actor_network.get_dist(tensordict_clone)
        if isinstance(dist, CompositeDistribution):
            action_keys = self.tensor_keys.action
            action = tensordict.select(
                *((action_keys,) if isinstance(action_keys, NestedKey) else action_keys)
            )
        else:
            action = tensordict.get(self.tensor_keys.action)

        if action.requires_grad:
            raise RuntimeError(
                f"tensordict stored {self.tensor_keys.action} requires grad."
            )
        log_prob = dist.log_prob(action)
        if not isinstance(action, torch.Tensor):
            log_prob = sum(
                dist.log_prob(tensordict).sum(dim="feature").values(True, True)
            )
        log_prob = log_prob.unsqueeze(-1)
        return log_prob, dist

    def loss_critic(self, tensordict: TensorDictBase) -> tuple[torch.Tensor, float]:
        if self.clip_value:
            old_state_value = tensordict.get(
                self.tensor_keys.value, None
            )  # TODO: None soon to be removed
            if old_state_value is None:
                raise KeyError(
                    f"clip_value is set to {self.clip_value}, but "
                    f"the key {self.tensor_keys.value} was not found in the input tensordict. "
                    f"Make sure that the value_key passed to A2C exists in the input tensordict."
                )
            old_state_value = old_state_value.clone()

        # TODO: if the advantage is gathered by forward, this introduces an
        #  overhead that we could easily reduce.
        target_return = tensordict.get(
            self.tensor_keys.value_target, None
        )  # TODO: None soon to be removed
        if target_return is None:
            raise KeyError(
                f"the key {self.tensor_keys.value_target} was not found in the input tensordict. "
                f"Make sure you provided the right key and the value_target (i.e. the target "
                f"return) has been retrieved accordingly. Advantage classes such as GAE, "
                f"TDLambdaEstimate and TDEstimate all return a 'value_target' entry that "
                f"can be used for the value loss."
            )
        tensordict_select = tensordict.select(
            *self.critic_network.in_keys, strict=False
        )
        with (
            self.critic_network_params.to_module(self.critic_network)
            if self.functional
            else contextlib.nullcontext()
        ):
            state_value = self.critic_network(
                tensordict_select,
            ).get(self.tensor_keys.value)
        loss_value = distance_loss(
            target_return,
            state_value,
            loss_function=self.loss_critic_type,
        )
        clip_fraction = None
        if self.clip_value:
            loss_value, clip_fraction = _clip_value_loss(
                old_state_value,
                state_value,
                self.clip_value,
                target_return,
                loss_value,
                self.loss_critic_type,
            )
        self._clear_weakrefs(
            tensordict,
            "actor_network_params",
            "critic_network_params",
            "target_actor_network_params",
            "target_critic_network_params",
        )
        if self.critic_coeff is not None:
            return self.critic_coeff * loss_value, clip_fraction
        return loss_value, clip_fraction

    @property
    @_cache_values
    def _cached_detach_critic_network_params(self):
        if not self.functional:
            return None
        return self.critic_network_params.detach()
    
    def _logits(
            self, tensordict: TensorDictBase
    ) -> torch.Tensor:
        """Extracts logits for chosen action."""
        logits = tensordict.get(("agents", "logits"))
        action = tensordict.get(self.tensor_keys.action)

        if action.ndim < logits.ndim:  # if dim of action is (batch, 1) keeps as it as, if (batch) unsqueeze it for gather
            action = action.unsqueeze(-1)
        
        action_logits = logits.gather(dim=-1, index=action) # get the logits for the actual chosen actions

        return action_logits

    @dispatch()
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_detach_critic_network_params,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        log_probs, dist = self._log_probs(tensordict)

        action_logits = self._logits(tensordict)
        loss = -(log_probs * advantage)
        td_out = TensorDict({"loss_objective": loss}, batch_size=[])

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.detach().mean())  # for logging
            td_out.set("loss_entropy", -self.entropy_coeff * entropy)
        if self.critic_coeff is not None:
            loss_critic, value_clip_fraction = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
        )
        self._clear_weakrefs(
            tensordict,
            td_out,
            "actor_network_params",
            "critic_network_params",
            "target_actor_network_params",
            "target_critic_network_params",
        )
        return td_out

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator

        # Handle ValueEstimatorBase instance or class
        if isinstance(value_type, ValueEstimatorBase) or (
            isinstance(value_type, type) and issubclass(value_type, ValueEstimatorBase)
        ):
            return LossModule.make_value_estimator(self, value_type, **hyperparams)

        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        hp.update(hyperparams)

        device = _get_default_device(self)
        hp["device"] = device

        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        if value_type == ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                value_network=self.critic_network, **hp
            )
        elif value_type == ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                value_network=self.critic_network, **hp
            )
        elif value_type == ValueEstimators.GAE:
            self._value_estimator = GAE(value_network=self.critic_network, **hp)
        elif value_type == ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                value_network=self.critic_network, **hp
            )
        elif value_type == ValueEstimators.VTrace:
            # VTrace currently does not support functional call on the actor
            if self.functional:
                actor_with_params = deepcopy(self.actor_network)
                self.actor_network_params.to_module(actor_with_params)
            else:
                actor_with_params = self.actor_network
            self._value_estimator = VTrace(
                value_network=self.critic_network, actor_network=actor_with_params, **hp
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "advantage": self.tensor_keys.advantage,
            "value": self.tensor_keys.value,
            "value_target": self.tensor_keys.value_target,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
            "sample_log_prob": self.tensor_keys.sample_log_prob,
        }
        self._value_estimator.set_keys(**tensor_keys)