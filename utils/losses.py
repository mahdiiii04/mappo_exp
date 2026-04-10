import torch
from dataclasses import dataclass

from torchrl.objectives import A2CLoss
from torchrl.objectives.utils import _reduce

from tensordict import TensorDictBase, TensorDict
from tensordict.utils import NestedKey
from tensordict.nn import (
    composite_lp_aggregate,
    dispatch
)

class NeuRDLoss(A2CLoss):

    @dataclass
    class _AcceptedKeys:
        advantage: NestedKey = "advantage"
        logits: NestedKey = "logits"
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
        

    def _logits(
            self, tensordict: TensorDictBase
    ) -> torch.Tensor:
        """Extracts logits for chosen action."""
        logits = tensordict.get(self.tensor_keys.logits)
        action = tensordict.get(self.tensor_keys.action)

        if action.ndim == 1:  # if dim of action is (batch, 1) keeps as it as, if (batch) unsqueeze it for gather
            action = action.unsqueeze(1)
        
        action_logits = logits.gather(1, action) # get the logits for the actual chosen actions

        return action_logits
    
    @dispatch
    def forward(
            self, tensordict: TensorDictBase
        ) -> TensorDictBase:
        tensordict = tensordict.clone(recurse=False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)

        if advantage is None:
            self.value_estimator(
                tensordict, 
                params=self._cached_detach_critic_network_params,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        
        action_logits = self._logits(tensordict)
        loss = -(action_logits * advantage)  # NeuRD modification
        
        td_out = TensorDict({"loss_objective": loss}, batch_size=[])

        if self.entropy_bonus:
            dist = torch.distributions.Categorical(
                logits=self.tensor_keys.get(self.tensor_keys.logits)
            )
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.detach().mean())
            td_out.set("loss_entropy", -self.entropy_coeff * entropy)

        if self.critic_coeff is not None:
            loss_critic, clip_fraction = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if clip_fraction is not None:
                td_out.set("value_clip_fraction", clip_fraction)

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

