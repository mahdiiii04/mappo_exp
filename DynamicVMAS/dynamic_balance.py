#  Dynamic Balance scenario: extends the built-in VMAS balance scenario so that
#  the goal landmark teleports to a new random position every `goal_move_every`
#  steps.  All other mechanics (physics, rewards, observations, done) are
#  inherited unchanged from the original scenario.

import torch

from vmas.scenarios.balance import Scenario as BalanceScenario
from vmas.simulator.utils import ScenarioUtils


class Scenario(BalanceScenario):
    """Balance scenario with a periodically-moving goal.

    Extra kwargs (passed via env.scenario in mappo.yaml / VmasEnv):
        goal_move_every (int): how many steps between goal relocations.
                               Default: 25.
        goal_move_x_range (tuple): (low, high) for the new goal x position.
                                   Default: (-1.0, 1.0).
        goal_move_y_range (tuple): (low, high) for the new goal y position.
                                   Default: (0.0, world.y_semidim).
    """

    # ------------------------------------------------------------------
    # World creation  (extend parent to grab extra kwargs & add counter)
    # ------------------------------------------------------------------

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Pull out our custom kwargs before the parent sees them (it calls
        # ScenarioUtils.check_kwargs_consumed which would raise on unknowns).
        self.goal_move_every = kwargs.pop("goal_move_every", 25)
        self.goal_x_low = kwargs.pop("goal_move_x_low", -1.0)
        self.goal_x_high = kwargs.pop("goal_move_x_high", 1.0)
        self.goal_y_low = kwargs.pop("goal_move_y_low", 0.0)
        # y_high is resolved after the world is built (we need y_semidim)
        self._goal_y_high_override = kwargs.pop("goal_move_y_high", None)

        world = super().make_world(batch_dim, device, **kwargs)

        # Per-environment step counter (shape: [batch_dim])
        self._step_count = torch.zeros(
            batch_dim, device=device, dtype=torch.long
        )

        return world

    # ------------------------------------------------------------------
    # Reset  (also reset per-env counters)
    # ------------------------------------------------------------------

    def reset_world_at(self, env_index: int = None):
        super().reset_world_at(env_index)
        if env_index is None:
            self._step_count[:] = 0
        else:
            self._step_count[env_index] = 0

    # ------------------------------------------------------------------
    # Reward hook  – advance counter and teleport goal when due
    # ------------------------------------------------------------------

    def reward(self, agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            # Increment step counter for every env
            self._step_count += 1

            # Mask: which envs are due for a goal move this step?
            move_mask = (self._step_count % self.goal_move_every) == 0

            if move_mask.any():
                n_move = int(move_mask.sum().item())
                device = self.world.device
                goal_y_high = (
                    self._goal_y_high_override
                    if self._goal_y_high_override is not None
                    else self.world.y_semidim
                )

                new_x = torch.empty(n_move, device=device).uniform_(
                    self.goal_x_low, self.goal_x_high
                )
                new_y = torch.empty(n_move, device=device).uniform_(
                    self.goal_y_low, goal_y_high
                )
                new_pos = torch.stack([new_x, new_y], dim=-1)  # (n_move, 2)

                # Update only the envs in the mask (vectorised write)
                self.package.goal.state.pos[move_mask] = new_pos

                # Re-initialise the potential-based shaping for moved envs so
                # the agents are not penalised/rewarded for the goal teleport.
                self.global_shaping[move_mask] = (
                    torch.linalg.vector_norm(
                        self.package.state.pos[move_mask]
                        - self.package.goal.state.pos[move_mask],
                        dim=1,
                    )
                    * self.shaping_factor
                )

        return super().reward(agent)