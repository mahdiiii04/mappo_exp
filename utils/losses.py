import torch
from dataclasses import dataclass

from torchrl.objectives import A2CLoss
from torchrl.objectives.utils import _reduce

from tensordict import TensorDictBase
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
        loss = -(action_logits * advantage)
