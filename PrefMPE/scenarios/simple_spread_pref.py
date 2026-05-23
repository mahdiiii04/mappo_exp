"""
simple_spread_pref.py
---------------------
Modified VMAS simple_spread where every agent has a *preferred* landmark that
yields a bonus reward, and preferences are **reshuffled every N env steps**
independently per environment.

Per-environment step tracking
------------------------------
`_env_steps[b]` counts how many steps environment b has taken since its last
preference reshuffle.  When `_env_steps[b] >= pref_switch_interval` for any b,
only *those* environments get new preferences and their counter resets.  This
is correct even when different envs reset (and therefore start counting from 0)
at different times.

Observation per agent  (dim = 4 + n_agents*2 + (n_agents-1)*2 + n_agents + 1)
  - own velocity                       (2)
  - own position                       (2)
  - relative pos to each landmark      (n_agents × 2)
  - relative pos to each other agent   ((n_agents-1) × 2)
  - one-hot preferred landmark index   (n_agents)
  - steps-until-next-switch (normalised to [0,1])  (1)   ← new

Reward per agent:
  - shared spread penalty   -sum_over_landmarks( min_agent_dist_to_landmark )
  - preference bonus        +pref_bonus  if dist(agent, pref_landmark) < coverage_radius
  - collision penalty       -collision_penalty  for each colliding agent pair
"""

import torch
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


# Fixed per-agent colors so agents are visually distinguishable
_AGENT_COLORS = [Color.BLUE, Color.GREEN, Color.RED, Color.GRAY, Color.LIGHT_GREEN]

# Landmark base colors — overwritten at runtime to match the owning agent's color
_LANDMARK_COLORS = [Color.BLUE, Color.GREEN, Color.RED, Color.GRAY, Color.LIGHT_GREEN]


class Scenario(BaseScenario):
    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.n_agents: int = kwargs.get("n_agents", 3)
        self.pref_bonus: float = kwargs.get("pref_bonus", 1.5)
        self.coverage_radius: float = kwargs.get("coverage_radius", 0.15)
        self.collision_penalty: float = kwargs.get("collision_penalty", 1.0)

        # Number of env steps between preference reshuffles (required, no None).
        self.pref_switch_interval: int = int(kwargs.get("pref_switch_interval", 200))

        world = World(batch_dim, device, dt=0.1, drag=0.25)

        # ---- agents ----
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                mass=1.0,
                shape=Sphere(radius=0.05),
                max_speed=None,
                color=_AGENT_COLORS[i % len(_AGENT_COLORS)],
            )
            world.add_agent(agent)

        # ---- landmarks ----
        for i in range(self.n_agents):
            lm = Landmark(
                name=f"landmark_{i}",
                collide=False,
                shape=Sphere(radius=0.05),
                color=_LANDMARK_COLORS[i % len(_LANDMARK_COLORS)],
            )
            world.add_landmark(lm)

        # preferences[b, a] = landmark index preferred by agent a in env b
        self.preferences: torch.Tensor = self._random_preferences(batch_dim, device)

        # Per-env step counters; reset to 0 when preferences are reshuffled
        self._env_steps: torch.Tensor = torch.zeros(
            batch_dim, dtype=torch.long, device=device
        )

        return world

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_world_at(self, env_index=None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + self.world.landmarks,
            self.world,
            env_index,
            min_dist_between_entities=0.15,
            x_bounds=(-1.0, 1.0),
            y_bounds=(-1.0, 1.0),
        )
        if env_index is None:
            self.preferences = self._random_preferences(
                self.world.batch_dim, self.world.device
            )
            self._env_steps.zero_()
        else:
            self.preferences[env_index] = torch.randperm(
                self.n_agents, device=self.world.device
            )
            self._env_steps[env_index] = 0

        self._apply_preference_colors(env_index)

    # ------------------------------------------------------------------
    # Step hook – per-environment preference switching
    # ------------------------------------------------------------------
    def post_step(self):
        self._env_steps += 1

        # Find which environments have reached the switch interval
        switch_mask = self._env_steps >= self.pref_switch_interval  # [B]
        if not switch_mask.any():
            return

        n_switch = switch_mask.sum().item()
        new_prefs = self._random_preferences(int(n_switch), self.world.device)
        self.preferences[switch_mask] = new_prefs
        self._env_steps[switch_mask] = 0

        self._apply_preference_colors(switch_mask=switch_mask)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def reward(self, agent: Agent) -> torch.Tensor:
        agent_idx = self.world.agents.index(agent)
        batch = self.world.batch_dim
        device = self.world.device

        rew = torch.zeros(batch, device=device)

        # ---- shared spread penalty ----
        all_agent_pos = torch.stack(
            [a.state.pos for a in self.world.agents], dim=1
        )  # [B, n_agents, 2]

        for lm in self.world.landmarks:
            lm_pos = lm.state.pos.unsqueeze(1)               # [B, 1, 2]
            dists = torch.linalg.norm(all_agent_pos - lm_pos, dim=-1)  # [B, n_agents]
            rew -= dists.min(dim=-1).values

        # ---- preference bonus ----
        pref_lm_idx = self.preferences[:, agent_idx]         # [B]
        all_lm_pos = torch.stack(
            [lm.state.pos for lm in self.world.landmarks], dim=1
        )  # [B, n_lm, 2]
        batch_idx = torch.arange(batch, device=device)
        pref_lm_pos = all_lm_pos[batch_idx, pref_lm_idx]    # [B, 2]

        dist_to_pref = torch.linalg.norm(agent.state.pos - pref_lm_pos, dim=-1)
        rew += self.pref_bonus * (dist_to_pref < self.coverage_radius).float()

        # ---- collision penalty ----
        for other in self.world.agents:
            if other is agent:
                continue
            dist = torch.linalg.norm(agent.state.pos - other.state.pos, dim=-1)
            rew -= self.collision_penalty * (
                dist < agent.shape.radius + other.shape.radius  # type: ignore[union-attr]
            ).float()

        return rew

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def observation(self, agent: Agent) -> torch.Tensor:
        agent_idx = self.world.agents.index(agent)
        batch = self.world.batch_dim
        device = self.world.device

        obs_parts = [
            agent.state.vel,   # [B, 2]
            agent.state.pos,   # [B, 2]
        ]

        for lm in self.world.landmarks:
            obs_parts.append(lm.state.pos - agent.state.pos)

        for other in self.world.agents:
            if other is not agent:
                obs_parts.append(other.state.pos - agent.state.pos)

        # One-hot preferred landmark
        pref_lm_idx = self.preferences[:, agent_idx]
        pref_onehot = torch.zeros(batch, self.n_agents, device=device)
        pref_onehot.scatter_(1, pref_lm_idx.unsqueeze(1), 1.0)
        obs_parts.append(pref_onehot)

        # Steps until next switch, normalised to [0, 1]
        # 1.0 = just switched (full interval remaining), 0.0 = switch imminent
        steps_remaining = (
            (self.pref_switch_interval - self._env_steps).float()
            / self.pref_switch_interval
        ).unsqueeze(1)  # [B, 1]
        obs_parts.append(steps_remaining)

        return torch.cat(obs_parts, dim=-1)

    # ------------------------------------------------------------------
    # Done / Info
    # ------------------------------------------------------------------
    def done(self) -> torch.Tensor:
        return torch.zeros(
            self.world.batch_dim, dtype=torch.bool, device=self.world.device
        )

    def info(self, agent: Agent) -> dict:
        agent_idx = self.world.agents.index(agent)
        pref_lm_idx = self.preferences[:, agent_idx]
        all_lm_pos = torch.stack(
            [lm.state.pos for lm in self.world.landmarks], dim=1
        )
        batch_idx = torch.arange(self.world.batch_dim, device=self.world.device)
        pref_lm_pos = all_lm_pos[batch_idx, pref_lm_idx]
        dist_to_pref = torch.linalg.norm(agent.state.pos - pref_lm_pos, dim=-1)
        return {
            "pref_landmark_idx": pref_lm_idx,
            "dist_to_pref_landmark": dist_to_pref,
            "steps_until_switch": self.pref_switch_interval - self._env_steps,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _random_preferences(
        self, batch_dim: int, device: torch.device
    ) -> torch.Tensor:
        """[batch_dim, n_agents] where each row is a permutation of landmark indices."""
        return torch.stack(
            [torch.randperm(self.n_agents, device=device) for _ in range(batch_dim)]
        )

    def _apply_preference_colors(self, env_index=None, switch_mask=None):
        """
        Color each landmark to match its currently assigned agent, making
        preference changes immediately visible during rendering.

        env_index : int or None  — update a single env (used on reset)
        switch_mask : BoolTensor — update only envs where mask is True (used in post_step)
        If both are None, update every env.
        """
        # Determine which batch indices to update
        if switch_mask is not None:
            indices = switch_mask.nonzero(as_tuple=False).squeeze(1).tolist()
        elif env_index is not None:
            indices = [env_index]
        else:
            indices = list(range(self.world.batch_dim))

        # VMAS landmark colors are per-instance (not batched), so we colour by
        # the *first* env that changed.  For vectorised rendering each env is
        # its own process anyway, so env 0 governs the shared color attribute.
        if not indices:
            return
        ref = indices[0]
        for agent_idx, lm in enumerate(self.world.landmarks):
            assigned_agent_idx = self.preferences[ref, agent_idx].item()
            lm.color = _AGENT_COLORS[int(assigned_agent_idx) % len(_AGENT_COLORS)]