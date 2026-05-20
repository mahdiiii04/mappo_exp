import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    Bounded,
    Unbounded,
    OneHot,
    Composite,
)

N_MOVE_ACTIONS = 5 # stay / up / down / left / right

_DR = torch.tensor([0, -1, 1, 0, 0], dtype=torch.long) # row delta
_DC = torch.tensor([0, 0, 0, -1, 1], dtype=torch.long) # column delta

###################### Base Class #########################

class GridWorldEnv(EnvBase):
    """
    Abstract class for vectorised multi-agent grid-world environments.

    MUST Implement
    --------------
    ``_compute_rewards(positions, prev_positions, td_in) -> Tensor [ne, n_agents, 1]``
        Given current and previous positions, return per-agent rewards.
    
    ``_reset_positions(reset_mask) -> Tensor [ne, n_agents, 1]``
        Return new starting positions (row, col) for each agent for every
        environment flagged in ```reset_mask``.

    ``_reset_goals(reset_mask) -> Tensor [ne, n_goals, 2]``
        Return new goal positions for each goal for every environment
        flagged in ``reset_mask``.

    Optional overrides
    ------------------
    ``_extra_obs(positions, goals) -> Tensor [ne, n_agents, extra_obs_dim]``
        Additional features in the observation. 
        Default: empty.

    ``_extra_reset_state(reset_mask) -> dict``
        Extra Tensordict entries during reset.

    ``_extra_step_state(reset_mask) -> dict``
        Extra Tensordict entries during step.
    """

    n_agents: int = 2
    n_goals: int = 2
    grid_size: int = 8
    extra_obs_dim: int = 0

    OUT_OF_BOUNDS_VIOLATION = -0.1

    def __init__(
        self,
        num_envs: int = 1,
        max_steps: int = 50,
        device: str = "cpu",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(batch_size=torch.Size([num_envs]), device=device)

        self._num_envs = num_envs
        self._max_steps = max_steps
        self.n_actions = N_MOVE_ACTIONS

        # deltas
        self.register_buffer(
            "_dr", _DR.clone()
        )
        self.register_buffer(
            "_dc", _DC.clone()
        )

        self.register_buffer(
            "_step_count", 
            torch.zeros(num_envs, dtype=torch.long, device=device),
        )

        # (row, col) for each agents
        self.register_buffer(
            "_positions",
            torch.zeros(num_envs, self.n_agents, 2, dtype=torch.long, device=device),
        )

        # (row, col) for each goal
        if self.n_goals > 0:
            self.register_buffer(
                "_goals",
                torch.zeros(num_envs, self.n_goals, 2, dtype=torch.long, device=device),
            )
        else:
            self._goals = None

        self._obs_dim = (
            2                           # own pos
            + 2 * (self.n_agents - 1)    # others's pos
            + 2 * self.n_goals          # goals's pos
            + self.extra_obs_dim
        )

        self._make_specs()
        self.set_seed(seed)

    
    def _make_specs(self) -> None:

        ne, n, a, o = self._num_envs, self.n_agents, self.n_actions, self._obs_dim
        gs = self.grid_size

        self.observation_spec = Composite(
            {
                "agents": Composite(
                    {
                        "observation": Unbounded(shape=(ne, n, o), dtype=torch.float32)
                    },
                    shape=(ne, n),
                )
            },
            shape=(ne,),
        )

        self.action_spec = Composite(
            {
                "agents": Composite(
                    {
                        "action": OneHot(n=a, shape=(ne, n, a), dtype=torch.long)
                    },
                    shape=(ne, n),
                )
            },
            shape=(ne,),
        )

        self.reward_spec = Composite(
            {
                "agents": Composite(
                    {
                       "reward": Unbounded(shape=(ne, n, 1), dtype=torch.float32)
                    },
                    shape=(ne, n),
                )
            },
            shape=(ne,),
        )

        self.done_spec = Composite(
            {
                "agents": Composite(
                    {
                        "done": Bounded(low=0, high=1, shape=(ne, n, 1), dtype=torch.bool),
                        "terminated": Bounded(low=0, high=1, shape=(ne, n, 1), dtype=torch.bool),
                    },
                    shape=(ne, n),
                )
            },
            shape=(ne,),
        )

    @property
    def reward_key(self):
        return ("agents", "reward")
    
    @property
    def action_key(self):
        return ("agents", "action")
    
    @property
    def done_keys(self):
        return [("agents", "done"), ("agents", "terminated")]
    
    @property
    def max_steps(self) -> int:
        return self._max_steps
    
    def _clamp_positions(
            self, pos: torch.Tensor
    ) -> torch.Tensor:
        """Keep positions within [0, grid_size-1]"""
        return pos.clamp(0, self.grid_size - 1)
    
    def _build_obs(
            self, positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Build observation tensor

        Params
        ------
        positions : [ne, n_agents, 2] 

        Returns
        ------
        obs : [ne, n_agents, obs_dim]
        """

        ne = self._num_envs
        gs = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
        pos_f = positions.float() / gs

        obs_parts = []
        for a_i in range(self.n_agents):
            own = pos_f[:, a_i, :]
            others = torch.cat(
                [pos_f[:, j, :] for j in range(self.n_agents) if j != a_i],
                dim=-1,
            )
            parts = [own, others]

            if self.n_goals > 0 and self._goals is not None:
                goals_f = self._goals.float() / gs
                goals_flat = goals_f.reshape(ne, -1)
                parts.append(goals_flat)

            obs_parts.append(torch.cat(parts, dim=-1))

        obs = torch.stack(obs_parts, dim=1)

        if self.extra_obs_dim > 0:
            extra = self._extra_obs(positions, self._goals)
            obs = torch.cat([obs, extra], dim=-1)

        return obs
    
    def _random_positions(
        self, ne: int, n: int, device
    ) -> torch.Tensor:
        """"Sample ``n`` random poistions across ``ne`` envs."""
        rows = torch.randint(0, self.grid_size, (ne, n), device=device)
        cols = torch.randint(0, self.grid_size, (ne, n), device=device)
        return torch.stack([rows, cols], dim=-1)
    
    def _reset(
        self, tensordict=None
    ) -> TensorDict:
        ne = self._num_envs
        dev = self.device

        if tensordict is not None and "_reset" in tensordict.keys():
            reset_mask = tensordict.get("_reset").reshape(ne)
        else:
            reset_mask = torch.ones(ne, dtype=torch.bool, device=dev)
        
        self._step_count[reset_mask] = 0

        new_pos = self._reset_positions(reset_mask)
        self._positions[reset_mask] = new_pos[reset_mask]

        if self.n_goals > 0:
            new_goals = self._reset_goals(reset_mask)
            self._goals[reset_mask] = new_goals[reset_mask]
        
        obs = self._build_obs(self._positions)
        
        td = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation" : obs
                    },
                    batch_size=[ne, self.n_agents],
                    device=dev,
                )
            },
            batch_size=[ne],
            device=dev,
        )

        extra = self._extra_reset_state(reset_mask)
        for k,v in extra.items():
            td.set(k, v)

        return td
    
    def _step(
        self, tensordict: TensorDict
    ) -> TensorDict:
        ne = self._num_envs
        dev = self.device

        action_raw = tensordict.get(("agents", "action"))
        if action_raw.dim() == 3 and action_raw.shape[-1] == self.n_actions:
            actions = action_raw.argmax(dim=-1)
        else:
            actions = action_raw.long()

        prev_positions = self._positions.clone()

        dr = self._dr[actions]
        dc = self._dc[actions]
        intended_pos = self._positions.clone()
        intended_pos[..., 0] = intended_pos[..., 0] + dr
        intended_pos[..., 1] = intended_pos[..., 1] + dc

        new_pos = intended_pos.clone()
        new_pos = self._clamp_positions(new_pos)

        violation_mask = (new_pos != intended_pos).any(dim=-1)

        self._positions = new_pos
        self._step_count.add_(1)

        rewards = self._compute_rewards(new_pos, prev_positions, tensordict)

        violation_penalty = violation_mask.unsqueeze(-1).float() * self.OUT_OF_BOUNDS_VIOLATION
        rewards = rewards + violation_penalty

        done_env = self._step_count >= self._max_steps
        done_agent = done_env[:, None, None].expand(ne, self.n_agents, 1)

        obs = self._build_obs(new_pos)

        td = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation" : obs,
                        "reward": rewards,
                        "done": done_agent.clone(),
                        "terminated": done_agent.clone(),
                    },
                    batch_size=[ne, self.n_agents],
                    device=dev,
                )
            },
            batch_size=[ne],
            device=dev,
        )

        extra = self._extra_step_state(tensordict)
        for k, v in extra.items():
            td.set(k, v)
        
        return td
    
    def _set_seed(self, seed) -> None:
        torch.manual_seed(seed)
    
    ######################### To Override ###########################

    def _reset_positions(
            self, reset_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return [ne, n_agents, 2] long tensor of starting positions."""
        ne, dev = self._num_envs, self.device
        return self._random_positions(ne, self.n_agents, dev)
    
    def _reset_goals(
            self, reset_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return [ne, n_goals, 2] long tensor of goal positions."""
        ne, dev = self._num_envs, self.device
        return self._random_positions(ne, self.n_goals, dev)
    
    def _compute_rewards(
        self, 
        positions: torch.Tensor,
        prev_positions: torch.Tensor,
        td_in: TensorDict,
    ) -> torch.Tensor:
        """Return [ne, n_agents, 1] float reward tensor."""
        raise NotImplementedError
    
    def _extra_obs(
        self, positions, goals
    ) -> torch.Tensor:
        ne, dev = self._num_envs, self.device
        return torch.zeros(ne, self.n_agents, 0, device=dev)
    
    def _extra_reset_state(
        self, reset_mask
    ) -> dict:
        return {}
    
    def _extra_step_state(
        self, td
    ) -> dict:
        return {}
    

    @torch.no_grad()
    def compute_metrics(
        self
    ) -> dict:
        """
        Return a dict of scalar metrics for the *current* positions.
        Subclasses can override to add domain-specific metrics.
        """
        return {}
    

######################### Scenarios ############################

class IndependentGridEnv(GridWorldEnv):
    """
    Each agent has its own goal.
    Reward per agent: -0.01 per step (time penalty) + 1.0 when on goal.
    Episode ends after max_steps.
    """

    n_agents = 2
    n_goals = 2
    grid_size = 8
    extra_obs_dim = 0

    STEP_PENALTY = -0.01
    GOAL_REWARD = 1.0

    def _compute_rewards(
        self, positions, prev_positions, td_in
    ):
        ne = self._num_envs
        dev = self.device

        rewards = torch.full(
            (ne, self.n_agents, 1), 
            self.STEP_PENALTY, 
            dtype=torch.float32,
            device=dev,
        )

        for i in range(self.n_agents):
            on_goal = (
                (positions[:, i, 0] == self._goals[:, i, 0]) &
                (positions[:, i, 1] == self._goals[:, i, 1])
            )
            rewards[:, i, 0] += on_goal.float() * self.GOAL_REWARD

        return rewards
    
    @torch.no_grad()
    def compute_metrics(self) -> dict:
        ne = self._num_envs
        on = sum(
            ((self._positions[:, i, 0] == self._goals[:, i, 0])
            & (self._positions[:, i, 1] == self._goals[:, i, 1])).float().mean()
            for i in range(self.n_agents)
        ) / self.n_agents
        return {"success_rate": on.item()}
    

class CooperativeNavEnv(GridWorldEnv):
    """
    All agents must reach the same goal.

    Agents are rewarded when they are all simultaneously on the goal.
    They also get a small reward for getting close to it (shaping).
    """

    n_agents = 2
    n_goals = 1
    grid_size = 8
    extra_obs_dim = 0

    ALL_ON_GOAL_REWARD = 2.0
    STEP_PENALTY = -0.01
    SHAPING_COEF = 0.1

    def _dist_to_goal(
        self, positions
    ):
        """L1 distance from each agent to the goal"""
        goal = self._goals[:, 0, :] # [ne, 2]
        diff = (positions - goal[:, None, :]).abs()
        return diff.sum(-1).float()
    
    def _compute_rewards(
        self, positions, prev_positions, td_in
    ):
        ne = self._num_envs
        dev = self.device

        rewards = torch.full(
            (ne, self.n_agents, 1),
            self.STEP_PENALTY,
            dtype=torch.float32,
            device=dev,
        )

        #shaping
        prev_dist = self._dist_to_goal(prev_positions)
        curr_dist = self._dist_to_goal(positions)
        shaping = (prev_dist - curr_dist) * self.SHAPING_COEF
        rewards[:, :, 0] += shaping

        goal = self._goals[:, 0, :]
        on_goal = (
            (positions[:, :, 0] == goal[:, None, 0]) &
            (positions[:, :, 1] == goal[:, None, 1])
        )
        all_on = on_goal.all(dim=1, keepdim=True).expand_as(on_goal)
        rewards[:, :, 0] += all_on.float() * self.ALL_ON_GOAL_REWARD

        return rewards

    @torch.no_grad()
    def compute_metrics(self) -> dict:
        goal = self._goals[:, 0, :]
        on = (
            (self._positions[:, :, 0] == goal[:, None, 0]) &
            (self._positions[:, :, 1] == goal[:, None, 1])
        )
        all_on = on.all(dim=1).float().mean()
        any_on = on.any(dim=1).float().mean()

        return {
            "all_on_goal_rate": all_on.item(),
            "any_on_goal_rate": any_on.item(),
        }

class AsymmetricNavEnv(GridWorldEnv):
    """
    N agents, N goals.

    Agents must coordinate to ALL occupy the SAME goal simultaneously.
    If all agents are on goal i:
        - agent i gets H_REWARD
        - all other agents get L_REWARD
    """

    n_agents = 2
    n_goals = 2
    grid_size = 8
    extra_obs_dim = 0

    H_REWARD = 2.0
    L_REWARD = 1.0
    STEP_PENALTY = -0.01
    SHAPING_COEF = 0.1


    def _dist_to_goal_idx(
        self, positions: torch.Tensor, goal_idx: int
    ) -> torch.Tensor:
        """
        L1 distance from every agent to a specific goal.
        """
        goal = self._goals[:, goal_idx, :]         
        diff = (positions - goal[:, None, :]).abs()  
        return diff.sum(-1).float()                  

    def _min_dist_to_any_goal(
        self, positions: torch.Tensor
    ) -> torch.Tensor:
        """
        For each agent, L1 distance to the nearest goal.
        """
        all_dists = torch.stack(
            [self._dist_to_goal_idx(positions, g) for g in range(self.n_goals)],
            dim=-1,
        )
        return all_dists.min(dim=-1).values          

    def _compute_rewards(
        self, positions, prev_positions, td_in
    ) -> torch.Tensor:

        ne  = self._num_envs
        dev = self.device

        rewards = torch.full(
            (ne, self.n_agents, 1),
            self.STEP_PENALTY,
            dtype=torch.float32,
            device=dev,
        )

        # Role-aware shaping: agent i is shaped toward goal i (its H_REWARD goal),
        # not the nearest goal. Ensures the shaping gradient always points agents
        # toward the goal that actually pays them H_REWARD.
        for i in range(self.n_agents):
            goal_i   = self._goals[:, i, :]                              # [ne, 2]
            prev_d   = (prev_positions[:, i] - goal_i).abs().sum(-1).float()
            curr_d   = (positions[:, i]      - goal_i).abs().sum(-1).float()
            rewards[:, i, 0] += (prev_d - curr_d) * self.SHAPING_COEF

        for i in range(self.n_goals):
            goal = self._goals[:, i, :]    

            on_goal = (
                (positions[:, :, 0] == goal[:, None, 0]) &
                (positions[:, :, 1] == goal[:, None, 1])
            )                              

            all_on = on_goal.all(dim=1)    

            rewards[:, :, 0] += all_on[:, None].float() * self.L_REWARD

            rewards[:, i, 0] += all_on.float() * (self.H_REWARD - self.L_REWARD)

        return rewards

    @torch.no_grad()
    def compute_metrics(self) -> dict:
        metrics = {}

        any_all_on = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)

        for i in range(self.n_goals):
            goal   = self._goals[:, i, :]           
            on     = (
                (self._positions[:, :, 0] == goal[:, None, 0]) &
                (self._positions[:, :, 1] == goal[:, None, 1])
            )                                      
            all_on = on.all(dim=1)                   
            any_all_on |= all_on

            metrics[f"all_on_goal_{i}_rate"] = all_on.float().mean().item()

        metrics["success_rate"] = any_all_on.float().mean().item()
        return metrics

class ShiftingAsymmetricNavEnv(GridWorldEnv):
    """
    N agents, N goals.

    The H_REWARD assignment rotates every `reward_phase_length` total steps:
      Phase 0: agent i gets H_REWARD at goal i
      Phase 1: agent i gets H_REWARD at goal (i+1) % n_goals
      ...
    This forces agents to renegotiate roles when the phase flips.
    """

    n_agents = 2
    n_goals = 2
    grid_size = 8
    extra_obs_dim = 0 

    H_REWARD = 2.0
    L_REWARD = 1.0
    STEP_PENALTY = -0.01
    SHAPING_COEF = 0.1

    def __init__(
        self, 
        reward_phase_length: int = 1000, 
        **kwargs
    ):
        self._reward_phase_length = reward_phase_length
        super().__init__(**kwargs)
        self.register_buffer(
            "_phase",
            torch.zeros(self._num_envs, dtype=torch.long, device=self.device),
        )
        self.register_buffer(
            "_global_steps",
            torch.zeros(self._num_envs, dtype=torch.long, device=self.device),
        )

    def _h_reward_goal_for_agent(
        self, agent_idx: int
    ) -> torch.Tensor:
        """
        Returns [ne] long tensor: which goal gives H_REWARD to agent_idx
        given the current phase per env.
        """
        return (agent_idx + self._phase) % self.n_goals

    def _min_dist_to_any_goal(
        self, positions
    ):
        all_dists = torch.stack(
            [
                (positions - self._goals[:, g, :][:, None, :])
                .abs().sum(-1).float()
                for g in range(self.n_goals)
            ],
            dim=-1,
        )
        return all_dists.min(dim=-1).values

    def _compute_rewards(
        self, positions, prev_positions, td_in
    ):
        ne  = self._num_envs
        dev = self.device

        # update phase
        self._global_steps.add_(1)
        self._phase = (
            self._global_steps // self._reward_phase_length
        ) % self.n_goals

        rewards = torch.full(
            (ne, self.n_agents, 1),
            self.STEP_PENALTY,
            dtype=torch.float32,
            device=dev,
        )

        # shaping toward nearest goal
        prev_min = self._min_dist_to_any_goal(prev_positions)
        curr_min = self._min_dist_to_any_goal(positions)

        shaping = (prev_min - curr_min) * self.SHAPING_COEF
        rewards[:, :, 0] += shaping

        # coordination rewards
        for g in range(self.n_goals):

            goal = self._goals[:, g, :]

            # [ne, n_agents]
            on_goal = (
                (positions[:, :, 0] == goal[:, None, 0]) &
                (positions[:, :, 1] == goal[:, None, 1])
            )

            # ONLY reward if ALL agents coordinated
            # on the SAME goal simultaneously
            all_on = on_goal.all(dim=1)  # [ne]

            if not all_on.any():
                continue

            for a in range(self.n_agents):

                # which goal currently gives high reward
                # to this agent
                h_goal = self._h_reward_goal_for_agent(a)

                # envs where THIS goal is the preferred one
                is_h_goal = (h_goal == g)

                # coordinated on preferred goal
                h_reward_mask = all_on & is_h_goal

                # coordinated on non-preferred goal
                l_reward_mask = all_on & (~is_h_goal)

                rewards[:, a, 0] += (
                    h_reward_mask.float() * self.H_REWARD
                )

                rewards[:, a, 0] += (
                    l_reward_mask.float() * self.L_REWARD
                )

        return rewards

    @torch.no_grad()
    def compute_metrics(self):
        metrics = {"current_phase": self._phase.float().mean().item()}
        for g in range(self.n_goals):
            goal = self._goals[:, g, :]
            on = (
                (self._positions[:, :, 0] == goal[:, None, 0]) &
                (self._positions[:, :, 1] == goal[:, None, 1])
            )
            metrics[f"any_on_goal_{g}"] = on.any(dim=1).float().mean().item()
            metrics[f"all_on_goal_{g}"] = on.all(dim=1).float().mean().item()
        return metrics

class ExclusiveNavEnv(GridWorldEnv):
    """
    N agents, N goals. Agents must split across goals (exclusive reward).
    
    The H_REWARD assignment rotates every `reward_phase_length` total steps:
      Phase 0: agent i gets H_REWARD at goal i
      Phase 1: agent i gets H_REWARD at goal (i+1) % n_goals
      ...
    This forces agents to renegotiate roles when the phase flips.
    
    Exclusivity: if multiple agents occupy the same goal, nobody gets the bonus.
    """

    n_agents = 2
    n_goals = 2
    grid_size = 8
    extra_obs_dim = 2  # each agent's H_REWARD goal (row, col) — phase-aware & agent-specific

    H_REWARD = 10.0
    L_REWARD = 1.0     # reduced so splitting is clearly better than piling
    STEP_PENALTY = -0.01
    SHAPING_COEF = 0.1
    COLLISION_PENALTY = -5.0  # penalty if 2+ agents on same goal

    def __init__(self, reward_phase_length: int = 1000, **kwargs):
        self._reward_phase_length = reward_phase_length
        super().__init__(**kwargs)
        # Stagger starting steps so parallel envs span all phases from the
        # very first collection batch — no more all-envs-same-phase batches.
        stagger = torch.randint(
            0,
            reward_phase_length * self.n_goals,
            (self._num_envs,),
            device=self.device,
        )
        self.register_buffer(
            "_global_steps",
            stagger,
        )
        # _phase[ne] tracks which rotation offset is active per env
        self.register_buffer(
            "_phase",
            (stagger // reward_phase_length) % self.n_goals,
        )

    def _extra_obs(self, positions, goals):
        # Return each agent's currently-assigned H_REWARD goal position,
        # normalised to [0, 1].  Shape: [ne, n_agents, 2].
        # This tells every agent *exactly* where it should go right now,
        # making the phase signal unambiguous and agent-specific.
        ne  = self._num_envs
        dev = self.device
        gs  = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
        idx = torch.arange(ne, device=dev)
        assigned = torch.stack(
            [
                self._goals[idx, self._h_reward_goal_for_agent(a)]  # [ne, 2]
                for a in range(self.n_agents)
            ],
            dim=1,  # [ne, n_agents, 2]
        )
        return assigned.float() / gs

    def _h_reward_goal_for_agent(self, agent_idx: int) -> torch.Tensor:
        """
        Returns [ne] long tensor: which goal gives H_REWARD to agent_idx
        given the current phase per env.
        """
        return (agent_idx + self._phase) % self.n_goals

    def _min_dist_to_any_goal(self, positions):
        all_dists = torch.stack(
            [
                (positions - self._goals[:, g, :][:, None, :])
                .abs().sum(-1).float()
                for g in range(self.n_goals)
            ],
            dim=-1,
        )
        return all_dists.min(dim=-1).values

    def _compute_rewards(self, positions, prev_positions, td_in):
        ne  = self._num_envs
        dev = self.device

        # Increment global steps and check for phase flip
        self._global_steps.add_(1)
        new_phase = (self._global_steps // self._reward_phase_length) % self.n_goals
        phase_changed = new_phase != self._phase
        self._phase = new_phase

        rewards = torch.full(
            (ne, self.n_agents, 1), self.STEP_PENALTY,
            dtype=torch.float32, device=dev,
        )

        # Phase-aware potential shaping: each agent shaped toward its
        # currently-assigned H_REWARD goal, not just the nearest goal.
        # This prevents shaping from reinforcing the wrong assignment
        # after a phase flip.
        idx = torch.arange(ne, device=dev)
        for a in range(self.n_agents):
            assigned_goal = self._goals[idx, self._h_reward_goal_for_agent(a)]  # [ne, 2]
            prev_dist = (prev_positions[:, a] - assigned_goal).abs().sum(-1).float()
            curr_dist = (positions[:, a]      - assigned_goal).abs().sum(-1).float()
            rewards[:, a, 0] += (prev_dist - curr_dist) * self.SHAPING_COEF

        # For each goal: count occupancy, assign rewards
        for g in range(self.n_goals):
            goal = self._goals[:, g, :]  # [ne, 2]
            on_goal = (
                (positions[:, :, 0] == goal[:, None, 0]) &
                (positions[:, :, 1] == goal[:, None, 1])
            )  # [ne, n_agents]

            n_on_goal = on_goal.sum(dim=1)  # [ne]
            exclusive = (n_on_goal == 1)     # only one agent here
            contested = (n_on_goal > 1)      # multiple agents collided

            for a in range(self.n_agents):
                h_goal = self._h_reward_goal_for_agent(a)  # [ne]
                is_h_goal = (h_goal == g)                   # [ne] bool

                # Only reward if this agent is alone on the goal
                agent_exclusive = on_goal[:, a] & exclusive  # [ne]

                h_bonus = agent_exclusive & is_h_goal
                l_bonus = agent_exclusive & ~is_h_goal

                rewards[:, a, 0] += h_bonus.float() * self.H_REWARD
                rewards[:, a, 0] += l_bonus.float() * self.L_REWARD

            # Collision penalty: all agents on a contested goal are penalised
            collision_agents = on_goal & contested[:, None]  # [ne, n_agents]
            rewards[:, :, 0] += collision_agents.float() * self.COLLISION_PENALTY

        return rewards

    @torch.no_grad()
    def compute_metrics(self):
        metrics = {"current_phase": self._phase.float().mean().item()}
        for g in range(self.n_goals):
            goal = self._goals[:, g, :]
            on = (
                (self._positions[:, :, 0] == goal[:, None, 0]) &
                (self._positions[:, :, 1] == goal[:, None, 1])
            )
            metrics[f"any_on_goal_{g}"] = on.any(dim=1).float().mean().item()
            metrics[f"exclusive_on_goal_{g}"] = (
                on.sum(dim=1) == 1
            ).float().mean().item()
        return metrics

class NSGridWorldMixin:
    """
    Adds non-stationarity (goal swap every phase_length steps)
    to any GridWorldEnv subclass.

    Usage:  class NSFoo(NSGridWorldMixin, FooEnv): ...
    """
    def __init__(self, phase_length: int = 550, **kwargs):
        self._phase_length = phase_length
        self._goals_initialized = False
        super().__init__(**kwargs)
        self.register_buffer("_total_steps",
            torch.zeros(self._num_envs, dtype=torch.long, device=self.device))
        self.register_buffer("_episode_count",
            torch.zeros(self._num_envs, dtype=torch.long, device=self.device))

    def _reset(self, tensordict=None):
        ne, dev = self._num_envs, self.device
        reset_mask = (
            tensordict.get("_reset").reshape(ne)
            if tensordict is not None and "_reset" in tensordict.keys()
            else torch.ones(ne, dtype=torch.bool, device=dev)
        )
        if self._goals_initialized:
            self._episode_count[reset_mask] += 1
        self._step_count[reset_mask] = 0
        new_pos = self._reset_positions(reset_mask)
        self._positions[reset_mask] = new_pos[reset_mask]
        if not self._goals_initialized:
            init_goals = self._reset_goals(reset_mask)
            self._goals[reset_mask] = init_goals[reset_mask]
            self._goals_initialized = True
        obs = self._build_obs(self._positions)
        return TensorDict(
            {
                "agents": TensorDict({"observation": obs},
                                     batch_size=[ne, self.n_agents], device=dev),
                "goals_changed":  torch.zeros(ne, dtype=torch.bool, device=dev),
                "episode_changed": torch.zeros(ne, dtype=torch.bool, device=dev),
                "total_steps":    self._total_steps.clone(),
                "episode_count":  self._episode_count.clone(),
            },
            batch_size=[ne], device=dev,
        )

    def _step(self, tensordict: TensorDict):
        self._total_steps.add_(1)
        goal_changed = (self._total_steps % self._phase_length == 0) & (self._total_steps > 0)
        if goal_changed.any():
            fresh = self._reset_goals(goal_changed)
            self._goals[goal_changed] = fresh[goal_changed]
        td = super()._step(tensordict)
        episode_changed = self._step_count >= self._max_steps
        td.set("goal_changed",    goal_changed)
        td.set("episode_changed", episode_changed)
        td.set("total_steps",     self._total_steps.clone())
        td.set("episode_count",   self._episode_count.clone())
        return td


class NSCooperativeNavEnv(NSGridWorldMixin, CooperativeNavEnv):
    """Non-stationary cooperative navigation."""

class NSAsymmetricNavEnv(NSGridWorldMixin, AsymmetricNavEnv):
    """
    Non-stationary asymmetric navigation.

    Fixes over the plain mixin wrapper:

    1. **Phase-aware shaping**: each agent is shaped toward the goal that
       currently gives *it* H_REWARD (agent i → goal i in phase 0,
       goal (i+1)%n_goals in phase 1, …) rather than toward the nearest
       goal.  After a goal swap the shaping immediately pulls agents to
       the right target instead of reinforcing the stale assignment.

    2. **Phase signal in observation**: ``extra_obs_dim = 2 * n_agents``
       — each agent receives the (row, col) of its currently-preferred
       goal, normalised to [0, 1].  This makes the non-stationarity
       directly observable so algorithms can condition on it.

    The H_REWARD assignment follows the same rotation as
    ShiftingAsymmetricNavEnv:
        phase p  →  agent i prefers goal (i + p) % n_goals
    and the phase advances every ``phase_length`` *total* env steps
    (tracked by NSGridWorldMixin._total_steps).
    """

    # Each agent gets 2 extra floats: (row, col) of its preferred goal.
    # extra_obs_dim is *per agent* (base class appends [ne, n_agents, extra_obs_dim]).
    extra_obs_dim = 2

    def __init__(self, phase_length: int = 550, **kwargs):
        # Store before super().__init__ so it is available during spec
        # building (which reads extra_obs_dim).
        self._asym_phase_length = phase_length
        super().__init__(phase_length=phase_length, **kwargs)
        # _phase[ne]: which rotation offset is active, 0 … n_goals-1
        self.register_buffer(
            "_asym_phase",
            torch.zeros(self._num_envs, dtype=torch.long, device=self.device),
        )

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _preferred_goal_for_agent(self, agent_idx: int) -> torch.Tensor:
        """[ne] long tensor: index of the goal that gives H_REWARD to agent_idx."""
        return (agent_idx + self._asym_phase) % self.n_goals

    def _update_phase(self) -> None:
        """Advance phase counter based on total env steps (from mixin)."""
        self._asym_phase = (
            self._total_steps // self._asym_phase_length
        ) % self.n_goals

    # ------------------------------------------------------------------
    # Extra observation: preferred-goal position per agent
    # ------------------------------------------------------------------

    def _extra_obs(self, positions, goals) -> torch.Tensor:
        """
        Returns [ne, n_agents, 2] — each agent's currently-preferred
        goal position, normalised to [0, 1].
        """
        ne  = self._num_envs
        dev = self.device
        gs  = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
        idx = torch.arange(ne, device=dev)
        preferred = torch.stack(
            [
                self._goals[idx, self._preferred_goal_for_agent(a)]  # [ne, 2]
                for a in range(self.n_agents)
            ],
            dim=1,  # [ne, n_agents, 2]
        )
        return preferred.float() / gs   # [ne, n_agents, 2]

    # ------------------------------------------------------------------
    # Reward: phase-aware shaping + existing H/L structure
    # ------------------------------------------------------------------

    def _compute_rewards(self, positions, prev_positions, td_in) -> torch.Tensor:
        ne  = self._num_envs
        dev = self.device

        # Advance the phase counter (uses _total_steps kept by mixin).
        self._update_phase()

        rewards = torch.full(
            (ne, self.n_agents, 1),
            self.STEP_PENALTY,
            dtype=torch.float32,
            device=dev,
        )

        # --- Phase-aware potential shaping ----------------------------
        # Shape each agent toward *its* preferred goal, not the nearest
        # goal.  This ensures the shaping gradient flips correctly when
        # the phase changes and never points agents at the wrong target.
        idx = torch.arange(ne, device=dev)
        for a in range(self.n_agents):
            g_idx   = self._preferred_goal_for_agent(a)          # [ne]
            pg      = self._goals[idx, g_idx]                    # [ne, 2]
            prev_d  = (prev_positions[:, a] - pg).abs().sum(-1).float()
            curr_d  = (positions[:, a]      - pg).abs().sum(-1).float()
            rewards[:, a, 0] += (prev_d - curr_d) * self.SHAPING_COEF

        # --- Coordination reward (inherited structure) ----------------
        # All agents must be on the *same* goal simultaneously.
        # The agent whose index matches the goal gets H_REWARD;
        # all others get L_REWARD.
        for g in range(self.n_goals):
            goal   = self._goals[:, g, :]                        # [ne, 2]
            on_goal = (
                (positions[:, :, 0] == goal[:, None, 0]) &
                (positions[:, :, 1] == goal[:, None, 1])
            )                                                     # [ne, n_agents]
            all_on = on_goal.all(dim=1)                          # [ne]

            if not all_on.any():
                continue

            for a in range(self.n_agents):
                pref_g      = self._preferred_goal_for_agent(a)  # [ne]
                is_h        = (pref_g == g)                      # [ne] bool
                h_mask      = all_on & is_h
                l_mask      = all_on & ~is_h
                rewards[:, a, 0] += h_mask.float() * self.H_REWARD
                rewards[:, a, 0] += l_mask.float() * self.L_REWARD

        return rewards

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_metrics(self) -> dict:
        metrics = {"current_phase": self._asym_phase.float().mean().item()}
        for g in range(self.n_goals):
            goal   = self._goals[:, g, :]
            on     = (
                (self._positions[:, :, 0] == goal[:, None, 0]) &
                (self._positions[:, :, 1] == goal[:, None, 1])
            )
            all_on = on.all(dim=1).float()
            any_on = on.any(dim=1).float()
            metrics[f"all_on_goal_{g}_rate"] = all_on.mean().item()
            metrics[f"any_on_goal_{g}_rate"] = any_on.mean().item()
        return metrics

class NSShiftingNAvEnv(NSGridWorldMixin, ShiftingAsymmetricNavEnv):
    """Non-stationary shifting navigation."""

class NSExclusiveNavEnv(NSGridWorldMixin, ExclusiveNavEnv):
    """Non-stationary exclusive navigation."""
    
class NSRoleShiftNavEnv(AsymmetricNavEnv):
    """
    Non-stationary asymmetric navigation — role shift only.

    Goal *positions* are fixed for the entire training run (set once at
    the first reset, never randomised again).  The only non-stationarity
    is that the H_REWARD assignment rotates every ``phase_length`` total
    environment steps:

        phase p  →  agent i gets H_REWARD at goal (i + p) % n_goals

    This isolates the role-renegotiation challenge from the goal-relocation
    challenge, making it a clean test of whether an algorithm can adapt to
    a shifting incentive structure with no perceptual change in the env.

    Observation augmentation (``extra_obs_dim = 2``): each agent receives
    the (row, col) of its currently-preferred goal, normalised to [0, 1],
    so the phase flip is directly observable.
    """

    extra_obs_dim = 2

    def __init__(self, phase_length: int = 550, **kwargs):
        self._rs_phase_length = phase_length
        super().__init__(**kwargs)
        self.register_buffer(
            "_total_steps",
            torch.zeros(self._num_envs, dtype=torch.long, device=self.device),
        )
        self.register_buffer(
            "_rs_phase",
            torch.zeros(self._num_envs, dtype=torch.long, device=self.device),
        )
        self._goals_initialized = False

    # ------------------------------------------------------------------
    # Goals: fixed after first reset, never re-randomised
    # ------------------------------------------------------------------

    def _reset(self, tensordict=None):
        ne, dev = self._num_envs, self.device
        reset_mask = (
            tensordict.get("_reset").reshape(ne)
            if tensordict is not None and "_reset" in tensordict.keys()
            else torch.ones(ne, dtype=torch.bool, device=dev)
        )
        self._step_count[reset_mask] = 0
        new_pos = self._reset_positions(reset_mask)
        self._positions[reset_mask] = new_pos[reset_mask]

        # Goals set once on first reset, then frozen forever.
        if not self._goals_initialized:
            init_goals = self._reset_goals(reset_mask)
            self._goals[reset_mask] = init_goals[reset_mask]
            self._goals_initialized = True
        # Note: subsequent resets do NOT touch self._goals.

        obs = self._build_obs(self._positions)
        return TensorDict(
            {
                "agents": TensorDict(
                    {"observation": obs},
                    batch_size=[ne, self.n_agents],
                    device=dev,
                )
            },
            batch_size=[ne],
            device=dev,
        )

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _preferred_goal_for_agent(self, agent_idx: int) -> torch.Tensor:
        """[ne] index of the goal that currently pays agent_idx H_REWARD."""
        return (agent_idx + self._rs_phase) % self.n_goals

    def _update_phase(self) -> None:
        self._rs_phase = (
            self._total_steps // self._rs_phase_length
        ) % self.n_goals

    # ------------------------------------------------------------------
    # Extra observation: preferred-goal position per agent
    # ------------------------------------------------------------------

    def _extra_obs(self, positions, goals) -> torch.Tensor:
        """[ne, n_agents, 2] — each agent's preferred goal pos, normalised."""
        ne  = self._num_envs
        dev = self.device
        gs  = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
        idx = torch.arange(ne, device=dev)
        preferred = torch.stack(
            [
                self._goals[idx, self._preferred_goal_for_agent(a)]
                for a in range(self.n_agents)
            ],
            dim=1,  # [ne, n_agents, 2]
        )
        return preferred.float() / gs

    # ------------------------------------------------------------------
    # Step: tick total_steps, update phase, then run base step
    # ------------------------------------------------------------------

    def _step(self, tensordict: TensorDict) -> TensorDict:
        self._total_steps.add_(1)
        self._update_phase()
        return super()._step(tensordict)

    # ------------------------------------------------------------------
    # Reward: phase-aware shaping + phase-aware H/L assignment
    # ------------------------------------------------------------------

    def _compute_rewards(self, positions, prev_positions, td_in) -> torch.Tensor:
        ne  = self._num_envs
        dev = self.device

        rewards = torch.full(
            (ne, self.n_agents, 1),
            self.STEP_PENALTY,
            dtype=torch.float32,
            device=dev,
        )

        # Phase-aware shaping: agent a → its currently-preferred goal.
        idx = torch.arange(ne, device=dev)
        for a in range(self.n_agents):
            g_idx  = self._preferred_goal_for_agent(a)           # [ne]
            pg     = self._goals[idx, g_idx]                     # [ne, 2]
            prev_d = (prev_positions[:, a] - pg).abs().sum(-1).float()
            curr_d = (positions[:, a]      - pg).abs().sum(-1).float()
            rewards[:, a, 0] += (prev_d - curr_d) * self.SHAPING_COEF

        # Phase-aware H/L: whoever meets at a goal, check current preferred.
        for g in range(self.n_goals):
            goal    = self._goals[:, g, :]
            on_goal = (
                (positions[:, :, 0] == goal[:, None, 0]) &
                (positions[:, :, 1] == goal[:, None, 1])
            )
            all_on = on_goal.all(dim=1)
            if not all_on.any():
                continue
            for a in range(self.n_agents):
                pref_g = self._preferred_goal_for_agent(a)       # [ne]
                is_h   = (pref_g == g)
                rewards[:, a, 0] += (all_on & is_h).float()  * self.H_REWARD
                rewards[:, a, 0] += (all_on & ~is_h).float() * self.L_REWARD

        return rewards

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_metrics(self) -> dict:
        metrics = {"current_phase": self._rs_phase.float().mean().item()}
        for g in range(self.n_goals):
            goal  = self._goals[:, g, :]
            on    = (
                (self._positions[:, :, 0] == goal[:, None, 0]) &
                (self._positions[:, :, 1] == goal[:, None, 1])
            )
            all_on = on.all(dim=1).float()
            metrics[f"all_on_goal_{g}_rate"] = all_on.mean().item()

        # Phase-correct coordination: are they meeting at the goal that
        # currently maximises total reward?
        idx         = torch.arange(self._num_envs, device=self.device)
        hot_g       = self._rs_phase                             # [ne]
        hot_goal    = self._goals[idx, hot_g]
        on_hot      = (
            (self._positions[:, :, 0] == hot_goal[:, None, 0]) &
            (self._positions[:, :, 1] == hot_goal[:, None, 1])
        ).all(dim=1).float()
        cold_g      = 1 - hot_g
        cold_goal   = self._goals[idx, cold_g]
        on_cold     = (
            (self._positions[:, :, 0] == cold_goal[:, None, 0]) &
            (self._positions[:, :, 1] == cold_goal[:, None, 1])
        ).all(dim=1).float()
        metrics["phase_correct_rate"] = on_hot.mean().item()
        metrics["phase_stale_rate"]   = on_cold.mean().item()
        return metrics


#################### 3-Agent Variants ######################

class AsymmetricNavEnv3(AsymmetricNavEnv):
    """
    3-agent, 3-goal asymmetric navigation (stationary).

    Identical to AsymmetricNavEnv except:
      - n_agents = n_goals = 3
      - grid_size = 10  (gives more room; 8x8 gets crowded with 3 agents + 3 goals)

    Reward structure:
      All 3 agents must occupy the same goal simultaneously.
      Agent i gets H_REWARD when all meet at goal i, L_REWARD otherwise.
      Role-aware shaping: agent i shaped toward goal i.
    """
    n_agents  = 3
    n_goals   = 3
    grid_size = 10


class NSAsymmetricNavEnv3(NSAsymmetricNavEnv):
    """
    3-agent, 3-goal non-stationary asymmetric navigation.

    Identical to NSAsymmetricNavEnv except n_agents = n_goals = 3 and
    grid_size = 10.

    Phase rotation cycles all three role assignments simultaneously:
        phase 0: agent 0 → goal 0,  agent 1 → goal 1,  agent 2 → goal 2
        phase 1: agent 0 → goal 1,  agent 1 → goal 2,  agent 2 → goal 0
        phase 2: agent 0 → goal 2,  agent 1 → goal 0,  agent 2 → goal 1

    This is strictly harder than the 2-agent case: there is no simple
    "mirror" policy MAPPO can fall back on — every agent must adopt a
    completely new role on each flip.
    """
    n_agents  = 3
    n_goals   = 3
    grid_size = 8


class NSRoleShiftNavEnv3(NSRoleShiftNavEnv):
    """
    3-agent, 3-goal role-shift-only non-stationary navigation.

    Goals are fixed for the entire run.  Only the H_REWARD assignment
    rotates every phase_length steps (same 3-way cycle as
    NSAsymmetricNavEnv3).  Hardest possible test: nothing in the
    observations changes except the preferred-goal hint, yet the
    entire coordination equilibrium must be renegotiated.
    """
    n_agents  = 3
    n_goals   = 3
    grid_size = 10


#################### Registery + Factory ######################

_REGISTRY: dict[str, type[GridWorldEnv]] = {
    "independent_gird": IndependentGridEnv,
    "cooperative_nav": CooperativeNavEnv,
    "ns_cooperative_nav": NSCooperativeNavEnv,
    "asymmetric_nav": AsymmetricNavEnv,
    "ns_asymmetric_nav": NSAsymmetricNavEnv,
    "asymmetric_nav_3": AsymmetricNavEnv3,
    "ns_asymmetric_nav_3": NSAsymmetricNavEnv3,
    "ns_role_shift_nav": NSRoleShiftNavEnv,
    "ns_role_shift_nav_3": NSRoleShiftNavEnv3,
    "shifting_nav": ShiftingAsymmetricNavEnv,
    "ns_shifting_nav": NSShiftingNAvEnv,
    "exclusive_nav": ExclusiveNavEnv,
    "ns_exclusive_nav": NSExclusiveNavEnv,
}

def GridWorldFactory(
    scenario: str,
    num_envs: int = 1,
    max_steps: int = 50,
    device: str = "cpu",
    seed: int = 0,
    **kwargs,
):
    if scenario not in _REGISTRY:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Available: {sorted(_REGISTRY.keys())}."
        )
    cls = _REGISTRY[scenario]
    return cls(
        num_envs=num_envs,
        max_steps=max_steps,
        device=device,
        seed=seed,
        **kwargs,
    )