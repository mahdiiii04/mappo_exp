from collections.abc import Mapping
from dataclasses import dataclass
import contextlib

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
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    set_composite_lp_aggregate,
    TensorDictModule,
)

from tensordict.utils import NestedKey
from torch import distributions as d

from torchrl._utils import _standardize, logger as torchrl_logger, VERBOSE
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _cache_values,
    _clip_value_loss,
    _GAMMA_LMBDA_DEPREC_ERROR,
    _maybe_add_or_extend_key,
    _maybe_get_or_select,
    _reduce,
    _sum_td_features,
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


class PPOLoss(LossModule):

    @dataclass
    class _AcceptedKeys:

        advantage: NestedKey = "advantage"
        value_target: NestedKey = "value_target"
        value: NestedKey = "state_value"
        sample_log_prob: NestedKey | list[NestedKey] | None = None
        action: NestedKey | list[NestedKey] = "action"
        reward: NestedKey | list[NestedKey] = "reward"
        done: NestedKey | list[NestedKey] = "done"
        terminated: NestedKey | list[NestedKey] = "terminated"

        def __post_init__(self):
            if self.sample_log_prob is None:
                if composite_lp_aggregate(nowarn=True):
                    self.sample_log_prob = "sample_log_prob"
                else:
                    self.sample_log_prob = "action_log_prob"
        

    default_keys = _AcceptedKeys
    tensor_keys = _AcceptedKeys
    default_value_estimator = ValueEstimators.GAE

    actor_network: ProbabilisticTensorDictModule
    critic_network: TensorDictModule
    actor_network_parameters: TensorDictParams
    critic_network_parameters: TensorDictParams
    target_actor_network_parameters: TensorDictParams
    target_critic_network_parameters: TensorDictParams

    def __init__(
            self,
            actor_network: ProbabilisticTensorDictSequential | None = None,
            critic_network: TensorDictModule | None = None,
            *,
            entropy_bonus: bool = True,
            samples_mc_entropy: int = 1,
            entropy_coeff: float | Mapping[NestedKey, float] | None = None,
            log_explained_variance: bool = True,
            critic_coeff: float | None = None,
            loss_critic_type: str = "smooth_l1",
            normalize_advantage: bool = False,
            normalize_advantage_exclude_dims: tuple[int] = (),
            gamma: float | None = None,
            seperate_losses: bool = False,
            advantage_key: str | None = None,
            value_target_key: str | None = None,
            value_key: str | None = None,
            functional: bool = True,
            actor: ProbabilisticTensorDictSequential = None,
            critic: ProbabilisticTensorDictSequential = None,
            reduction: str | None = None,
            clip_value: float | None = None,
            device: torch.device | None = None,
            **kwargs,
    ):
        
        if actor is not None:
            actor_network = actor
            del actor
        if critic is not None:
            critic_network = critic_network
            del critic

        if "critic_coef" in kwargs:
            raise TypeError(
                "'critic coef' has been removed in torchrl v0.11. Please use 'critic_coeff' instead." 
            )
        
        if critic_coeff is None and critic_network is not None:
            critic_coeff = 1.0
        elif critic_coeff in (None, 0) and critic_network is not None:
            critic_coeff = None

        if actor_network is None or (
            critic_network is None and critic_coeff not in (None, 0.0)
        ):
            raise TypeError(
                "Missing positional arguments actor_network or critic_network."
            )
        
        if reduction is None:
            reduction = "mean"
        
        self._functional = functional
        self._in_keys = None
        self._out_keys = None
        super().__init__()

        if functional:
            self.convert_to_functional(actor_network, "actor_network")
        else:
            self.actor_network = actor_network
            self.actor_network_parameters = None
            self.target_actor_network_parameters = None

        if seperate_losses:
            policy_params = list(actor_network.parameters())
        else:
            policy_params = None
        
        if functional and critic_network is not None:
            self.convert_to_functional(critic_network, "critic_network", compare_against=policy_params)
        else:
            self.critic_network = critic_network
            self.critic_network_parameters = None
            self.target_critic_network_parameters = None
        
        self.log_explained_variance = log_explained_variance
        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_bonus = entropy_bonus
        self.seperate_losses = seperate_losses
        self.reduction = reduction

        if device is None:
            try:
                device = next(self.parameters()).device
            except(AttributeError, StopIteration):
                device = getattr(
                    torch, "get_default_device", lambda: torch.device("cpu")
                )()
        
        if "entropy_coef" in kwargs:
            raise TypeError(
                "'entropy_coef' has been removed in torchrl v0.11. Please use 'entropy_coeff' instead."
            )
        
        if entropy_coeff is None:
            entropy_coeff = 0.01

        if isinstance(entropy_coeff, Mapping):
            self._entropy_coeff_map = {k: float(v) for k,v in entropy_coeff.items()}
            self.register_buffer("entropy_coeff", torch.tensor(0.0))
        elif isinstance(entropy_coeff, (float, int, torch.Tensor)):
            coeff = (
                float(entropy_coeff)
                if not torch.is_tensor(entropy_coeff)
                else float(entropy_coeff.item())
            )
            self.register_buffer("entropy_coeff", torch.tensor(coeff))
            self._entropy_coeff_map = None
        else:
            raise TypeError("entropy_coeff must be a float or a Mapping[str, float]")
        
        if critic_coeff is not None:
            self.register_buffer("critic_coeff", torch.tensor(critic_coeff, device=device))
        else:
            self.critic_coeff = None

        self._has_critic = bool(self.critic_coeff is not None and self.critic_coeff > 0)
        self.loss_critic_type = loss_critic_type
        self.normalize_advantage = normalize_advantage
        self.normalize_advantage_exclude_dims = normalize_advantage_exclude_dims

        if gamma is not None:
            raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)
        self._set_deprecated_ctor_keys(
            advantage=advantage_key,
            value_target=value_target_key,
            value=value_key,
        )

        if clip_value is not None:
            if isinstance(clip_value, float):
                clip_value = torch.tensor(clip_value, device=device)
            elif isinstance(clip_value, torch.Tensor):
                if clip_value.numel() != 1:
                    raise TypeError(
                        f"clip_value must be a float or a scalar tensor, got {clip_value}."
                    )
            else:
                raise TypeError(
                    f"clip_value must be a float or a scalar tensor, got {clip_value}."
                )
            self.register_buffer("clip_value", clip_value.to(device))
        else:
            self.clip_value = None

        try:
            log_prob_keys = self.actor_network.log_prob_keys
            action_keys = self.actor_network.dist_sample_keys
            if len(log_prob_keys) > 1:
                self.set_keys(sample_log_prob=log_prob_keys, action=action_keys)
            else:
                self.set_keys(sample_log_prob=log_prob_keys[0], action_keys=action_keys[0])
        except AttributeError:
            pass
    
    @property
    def functional(self):
        return self._functional

    def _set_in_keys(self):
        keys = []
        _maybe_add_or_extend_key(keys, self.actor_network.in_keys)
        _maybe_add_or_extend_key(keys, self.actor_network.in_keys, "next")
        if self.critic_network is not None:
            _maybe_add_or_extend_key(keys, self.critic_network.in_keys)
        _maybe_add_or_extend_key(keys, self.tensor_keys.action)
        _maybe_add_or_extend_key(keys, self.tensor_keys.sample_log_prob)
        _maybe_add_or_extend_key(keys, self.tensor_keys.reward, "next")
        _maybe_add_or_extend_key(keys, self.tensor_keys.done, "next")
        _maybe_add_or_extend_key(keys, self.tensor_keys.terminated, "next")

        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys
    
    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @property
    def out_keys(self):
        if self.out_keys is None:
            keys = ["loss_objective"]
            if self.entropy_bonus:
                keys.extend(["entropy", "loss_entropy"])
            if self.loss_critic:
                keys.append("loss_critic")
            if self.clip_value:
                keys.append("value_clip_fraction")
            self.out_keys = keys
        return self.out_keys

    @out_keys.setter
    def out_keys(self, values):
        self.out_keys = values

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if hasattr(self, "_value_estimator") and self._value_estimator is not None:
            self._value_estimator.set_keys(
                advantage=self.tensor_keys.advantage,
                value_target=self.tensor_keys.value_target,
                value=self.tensor_keys.value,
                reward=self.tensor_keys.reward,
                done=self.tensor_keys.done,
                terminated=self.tensor_keys.terminated,
                sample_log_prob=self.tensor_keys.sample_log_prob,
            )
        self._set_in_keys()

    def reset(self) -> None:
        pass

    def _get_entropy(
        self, dist: d.Distribution, adv_shape: torch.Size      
    ) -> torch.Tensor | TensorDict:
        try:
            entropy = dist.entropy()
            if not entropy.isfinite().all():
                del entropy
                if VERBOSE:
                    torchrl_logger.info(
                        "Entropy is not finite. Using Monte Carlo Sampling"
                    )
                raise NotImplemented
        except NotImplemented:
            if VERBOSE:
                torchrl_logger.warning(
                    f"Entropy not implemented for {type(dist)} or is not finite. Using Monte Carlo Sampling."
                )
            if getattr(dist, "has_rsample", False):
                x = dist.rsample((self.samples_mc_entropy,))
            else:
                x = dist.sample((self.samples_mc_entropy,))
            with (
                set_composite_lp_aggregate(False)
                if isinstance(dist, CompositeDistribution)
                else contextlib.nullcontext()
            ):
                log_prob = dist.log_prob(x)
                if is_tensor_collection(log_prob):
                    if isinstance(self.tensor_keys.sample_log_prob, NestedKey):
                        log_prob = log_prob.get(self.tensor_keys.sample_log_prob)
                    else:
                        log_prob = log_prob.select(*self.tensor_keys.sample_log_prob)

            entropy = -log_prob.mean(0)
            if is_tensor_collection(entropy) and entropy.batch_size != adv_shape:
                entropy.batch_size = adv_shape
        return entropy.unsqueeze(-1)