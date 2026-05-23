"""
simple_spread_pref.py
---------------------
A modified VMAS simple_spread scenario where every agent has a *preferred*
landmark.  Covering any landmark still contributes to the shared spread reward,
but an agent earns an additional bonus when it covers its own preferred landmark.

Preference identity (which landmark belongs to which agent) can be rotated
periodically so agents must re-learn a dynamic assignment.

Observation per agent (concatenation, dim = 4 + n_agents*2 + (n_agents-1)*2 + n_agents)
  - own velocity          (2)
  - own position          (2)
  - relative pos to each landmark  (n_agents × 2)
  - relative pos to each other agent  ((n_agents-1) × 2)
  - one-hot preferred landmark index  (n_agents)

Reward per agent:
  - shared spread penalty  -sum_over_landmarks( min_agent_dist_to_landmark )
  - preference bonus       +pref_bonus  if dist(agent, pref_landmark) < coverage_radius
  - collision penalty      -collision_penalty  for each agent pair in contact
"""

import torch
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.n_agents: int = kwargs.get("n_agents", 3)
        self.pref_bonus: float = kwargs.get("pref_bonus", 1.5)
        self.coverage_radius: float = kwargs.get("coverage_radius", 0.15)
        self.collision_penalty: float = kwargs.get("collision_penalty", 1.0)

        # Optional periodic preference switching (steps between shuffles, None = fixed)
        self.pref_switch_interval = kwargs.get("pref_switch_interval", None)
        self._step_count: int = 0

        world = World(batch_dim, device, dt=0.1, drag=0.25)

        # ---- agents ----
        agent_colors = [Color.BLUE, Color.GREEN, Color.RED,
                        Color.GRAY, Color.LIGHT_GREEN]
        for i in range(self.n_agents):
            color = agent_colors[i % len(agent_colors)]
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                mass=1.0,
                shape=Sphere(radius=0.05),
                max_speed=None,
                color=color,
            )
            world.add_agent(agent)

        # ---- landmarks ----
        landmark_colors = [Color.BLACK, Color.YELLOW, Color.GRAY,
                           Color.LIGHT_GREEN, Color.BLUE]
        for i in range(self.n_agents):
            color = landmark_colors[i % len(landmark_colors)]
            lm = Landmark(
                name=f"landmark_{i}",
                collide=False,
                shape=Sphere(radius=0.05),
                color=color,
            )
            world.add_landmark(lm)

        # preferences[b, a] = index of the preferred landmark for agent a in env b
        self.preferences = self._random_preferences(batch_dim, device)

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
        # Assign new random preferences on reset
        if env_index is None:
            self.preferences = self._random_preferences(
                self.world.batch_dim, self.world.device
            )
        else:
            self.preferences[env_index] = torch.randperm(
                self.n_agents, device=self.world.device
            )

    # ------------------------------------------------------------------
    # Step hook – preference switching
    # ------------------------------------------------------------------
    def post_step(self):
        if self.pref_switch_interval is None:
            return
        self._step_count += 1
        if self._step_count % self.pref_switch_interval == 0:
            self.preferences = self._random_preferences(
                self.world.batch_dim, self.world.device
            )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def reward(self, agent: Agent) -> torch.Tensor:
        agent_idx = self.world.agents.index(agent)
        batch = self.world.batch_dim
        device = self.world.device

        rew = torch.zeros(batch, device=device)

        # ---- shared spread penalty ----
        # For every landmark, penalise by the distance to the nearest agent.
        all_agent_pos = torch.stack(
            [a.state.pos for a in self.world.agents], dim=1
        )  # [B, n_agents, 2]

        for lm in self.world.landmarks:
            lm_pos = lm.state.pos.unsqueeze(1)  # [B, 1, 2]
            dists = torch.linalg.norm(all_agent_pos - lm_pos, dim=-1)  # [B, n_agents]
            rew -= dists.min(dim=-1).values

        # ---- preference bonus ----
        # Bonus for this agent being within coverage_radius of its preferred landmark.
        pref_lm_idx = self.preferences[:, agent_idx]  # [B]
        all_lm_pos = torch.stack(
            [lm.state.pos for lm in self.world.landmarks], dim=1
        )  # [B, n_lm, 2]
        batch_idx = torch.arange(batch, device=device)
        pref_lm_pos = all_lm_pos[batch_idx, pref_lm_idx]  # [B, 2]

        dist_to_pref = torch.linalg.norm(agent.state.pos - pref_lm_pos, dim=-1)  # [B]
        covered_pref = (dist_to_pref < self.coverage_radius).float()
        rew += self.pref_bonus * covered_pref

        # ---- collision penalty ----
        for other in self.world.agents:
            if other is agent:
                continue
            dist = torch.linalg.norm(
                agent.state.pos - other.state.pos, dim=-1
            )
            collision_radius = (
                agent.shape.radius + other.shape.radius  # type: ignore[union-attr]
            )
            rew -= self.collision_penalty * (dist < collision_radius).float()

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

        # Relative positions to all landmarks
        for lm in self.world.landmarks:
            obs_parts.append(lm.state.pos - agent.state.pos)  # [B, 2]

        # Relative positions to other agents
        for other in self.world.agents:
            if other is not agent:
                obs_parts.append(other.state.pos - agent.state.pos)  # [B, 2]

        # One-hot encoding of preferred landmark
        pref_lm_idx = self.preferences[:, agent_idx]  # [B]
        pref_onehot = torch.zeros(batch, self.n_agents, device=device)
        pref_onehot.scatter_(1, pref_lm_idx.unsqueeze(1), 1.0)
        obs_parts.append(pref_onehot)  # [B, n_agents]

        return torch.cat(obs_parts, dim=-1)

    # ------------------------------------------------------------------
    # Done / Info
    # ------------------------------------------------------------------
    def done(self) -> torch.Tensor:
        # Episode length is controlled externally via max_steps
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
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _random_preferences(
        self, batch_dim: int, device: torch.device
    ) -> torch.Tensor:
        """Return [batch_dim, n_agents] tensor where row b is a permutation of
        0..n_agents-1, i.e. each agent gets a unique preferred landmark."""
        return torch.stack(
            [
                torch.randperm(self.n_agents, device=device)
                for _ in range(batch_dim)
            ]
        )