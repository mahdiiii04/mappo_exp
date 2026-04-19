import torch

from tensordict import TensorDict

from torchrl.envs import EnvBase
from torchrl.data import (
    Bounded,
    Composite,
    OneHot,
    Unbounded,
)

############## Base Class ###############
class MatrixGameEnv(EnvBase):
    """
    Abstract class for 2-player, N-action matrix games.

    Subclass must set class attributes ``n_agents`` and ``n_actions``
    and implement ``_build_payoff(**kwargs) -> torch.Tensor`` which 
    returns a tensor of shape ``[n_agents, n_action, n_actions]`` 
    where ``payoff[i, a0, a1]`` is the reward for agent ``i`` when
    the joint action is ``(a0, a1)``.
    """

    n_agents: int = 2
    n_actions: int = 2

    def __init__(
            self,
            num_envs: int = 1,
            max_steps: int = 1,
            device: str = "cpu",
            seed: int = 0,
            **kwargs,
    ):
        super().__init__(batch_size=torch.Size([num_envs]), device=device)

        self._num_envs = num_envs
        self._max_steps = max_steps

        self._obs_dim = self.n_agents * self.n_actions # joint one-hot encoded obs for all agents

        self.register_buffer(
            "_step_count",
            torch.zeros(num_envs, dtype=torch.long, device=device),
        )

        self._make_specs()

        payoff = self._build_payoff(**kwargs)
        if not isinstance(payoff, torch.Tensor):
            raise TypeError("_build_payoff must return a torch.Tensor.")
        self.register_buffer(
            "_payoff",
            payoff.to(device)
        )

        self.set_seed(seed)

    ################# Specs ########################
    
    def _make_specs(self) -> None:
        ne = self._num_envs
        n = self.n_agents
        a = self.n_actions
        o = self._obs_dim

        self.observation_spec = Composite(
            {
                "agents": Composite(
                    {"observation" : Unbounded(shape=(ne, n, o), dtype=torch.float32)},
                    shape=(ne, n),
                )
            },
            shape=(ne,),
        )

        self.action_spec = Composite(
            {
                "agents": Composite(
                    {"action": OneHot(n=a, shape=(ne, n, a), dtype=torch.long)},
                    shape=(ne, n),
                )
            },
            shape=(ne,)
        )

        self.reward_spec = Composite(
            {
                "agents": Composite(
                    {"reward": Unbounded(shape=(ne, n, 1), dtype=torch.float32)},
                    shape=(ne, n),
                )
            },
            shape=(ne,),
        )

        # Per-agent done/terminated keys - like Vmas
        self.done_spec = Composite(
            {
                "agents": Composite(
                    {
                        "done": Bounded(low=0, high=1, shape=(ne, n, 1), dtype=torch.bool),
                        "terminated": Bounded(low=0, high=1, shape=(ne, n, 1), dtype=torch.bool),
                    },
                    shape=(ne, n),
                ),
            },
            shape=(ne,),
        )

    @property
    def reward_key(self):
        return ("agents", "reward")

    @property
    def action_key(self):
        return ("agents","action")
    
    @property
    def done_keys(self):
        return [("agents", "done"), ("agents", "terminated")]
    
    @property
    def max_steps(self) -> int:
        return self._max_steps
    
    def _reset(self, tensordict=None):
        ne = self._num_envs
        dev = self.device

        if tensordict is not None and "_reset" in tensordict.keys():
            reset_mask = tensordict.get("_reset").reshape(ne)
        else:
            reset_mask = torch.ones(ne, dtype=torch.bool, device=dev)

        self._step_count[reset_mask] = 0
        obs = torch.zeros(ne, self.n_agents, self._obs_dim, device=dev)

        return TensorDict(
            {
                "agents": TensorDict(
                    {"observation" : obs},
                    batch_size=[ne, self.n_agents],
                    device=dev,
                )
            },
            batch_size=[ne],
            device=dev,
        )
    
    def _step(self, tensordict: TensorDict):
        ne = self._num_envs
        dev = self.device

        action_raw = tensordict.get(("agents", "action"))

        if action_raw.dim() == 3:
            actions = action_raw.argmax(dim=-1)
            actions_oh = action_raw.float()
        else:
            actions = action_raw.long()
            actions_oh = torch.zeros(
                ne, self.n_agents, self.n_actions,
                device=dev, dtype=torch.float32,
            ).scatter_(-1, actions.unsqueeze(-1), 1.0)

        rewards = self._compute_rewards(actions)

        self._step_count.add_(1)
        done_env = self._step_count >= self._max_steps

        done_agent = done_env[:, None, None].expand(ne, self.n_agents, 1)

        joint_oh = actions_oh.reshape(ne, -1)
        obs_next = joint_oh[:, None, :].expand(ne, self.n_agents, -1).clone()

        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": obs_next,
                        "reward": rewards,
                        "done": done_agent.clone(),
                        "terminated": done_agent.clone(),
                    },
                    batch_size=[ne, self.n_agents],
                    device=dev,
                ),
            },
            batch_size=[ne],
            device=dev,
        )
    
    def _set_seed(self, seed) -> None:
        torch.manual_seed(seed)

    def _build_payoff(self, **kwargs) -> torch.Tensor: ##### to implement
        """
        Returns a float tensor of shape ``[num_envs, n_agents, n_actions]``
        where ``payoff[i, a0, a1]`` is the payoff of agent ``i`` when the
        joint action is ``(a0, a1)``. called once during ``__init__``.
        """
        raise NotImplemented
    
    def _compute_rewards(
            self, actions: torch.Tensor
    ) -> torch.Tensor:
        
        a0 = actions[:, 0]
        a1 = actions[:, 1]
        rewards = torch.stack(
            [self._payoff[i, a0, a1] for i in range(self.n_agents)],
            dim=1
        ).unsqueeze(-1)
        return rewards


#################### Games ######################

class PrisonersDilemmaEnv(MatrixGameEnv):
    """
    Classic 2-player Prisonner's Dilemma 

    Action
    ------
        0 Cooperate (C)
        1 Defect (D)
    
    payoffs (T > R > P > S, 2R > T+S ensures mutual cooperation is socially optimal)
    ------
        (C, C) -> (R, R) mutual reward
        (C, D) -> (S, T) sucker / temptation
        (D, C) -> (T, S) temptation / sucker
        (D, D) -> (P, P) mutual punishement

    default: T=3, R=2, P=1, S=0
    """
    n_agents = 2
    n_actions = 2

    def __init__(
            self,
            T: float = 3.,
            R: float = 2.,
            P: float = 1.,
            S: float = 0.,
            **kwargs
    ):
        self.T, self.R, self.P, self.S = T, R, P, S
        super().__init__(**kwargs)

    def _build_payoff(self, **_) -> torch.Tensor:
        T, R, P, S = self.T, self.R, self.P, self.S

        return torch.tensor(
            [
                [[R, S], [T, P]], # agent 0
                [[R, T], [S, P]], # agent 1
            ],
            dtype=torch.float32,
        )

class RockPaperScissorsEnv(MatrixGameEnv):
    """
    2-player Rock Paper Scissors.

    Actions
    ------
        0 Rock
        1 Paper
        2 Scissors

    Payoffs
    ------
    Win -> +1, Loss -> -1, Draw -> 0.
    """
    n_agents = 2
    n_actions = 3

    def _build_payoff(self, **_) -> torch.Tensor:
        
        outcome = torch.tensor(
            [
                [0., -1., 1.],
                [1., 0., -1.],
                [-1., 1., 0.],
            ], dtype=torch.float32,
        )

        return torch.stack([outcome, -outcome], dim=0)
    
class StagHuntEnv(MatrixGameEnv):
    """
    2-player Stag Hunt game.

    Actions
    ------
        0 Hunt Stag
        1 Hunt Hare
    
    Payoffs
    ------
        (Stag, Stag) -> (stag_reward, stag_reward)
        (Stag, Hare) -> (0, hare_reward)
        (Hare, Stag) -> (hare_reward, 0)
        (Hare, Hare) -> (hare_reward, hare_reward)

    default: stag_reward = 4, hare_reward = 1.
    """

    n_agents = 2
    n_actions = 2

    def __init__(
            self,
            stag_reward: float = 4.,
            hare_reward: float = 1.,
            **kwargs
    ):
        self._stag = stag_reward
        self._hare = hare_reward
        super().__init__(**kwargs)

    def _build_payoff(self, **_):
        S, H = self._stag, self._hare
        return torch.tensor(
            [
                [[S, 0], [H, H]],
                [[S, H], [0, H]],
            ],
            dtype=torch.float32,
        )
    
class BattleOfSexesEnv(MatrixGameEnv):
    """
    Battle of the Sexes

    Actions
    ------
        0 Football (agent's 0 preference)
        1 Opera    (agent's 1 preference)

    Payoffs
    ------
        (Football, Football) -> (football_reward, coordination_reward)
        (Opera, Opera)       -> (coordination_reward, opera_reward)
        (Football, Opera)    -> (0,0)
        (Opera, Football)    -> (0,0)

    default: football_reward = opera_reward = 2, coordination_reward = 1.
    """

    n_agents = 2
    n_actions = 2

    def __init__(
            self,
            football_reward: float = 2.,
            opera_reward: float = 2.,
            coordination_reward: float = 1.,
            **kwargs):
        
        self._football = football_reward
        self._opera = opera_reward
        self._coord = coordination_reward
        super().__init__(**kwargs)

    def _build_payoff(self, **_):
        f, o, c = self._football, self._opera, self._coord
        return torch.tensor(
            [
                [[f, 0], [0, c]],
                [[c, 0], [0, o]],
            ],
            dtype=torch.float32,
        )
    

class BiasedRPSEnv(MatrixGameEnv):
    """"
    A modified version of rock paper scissors with a value ``v``
    periodically assigned to a different matchup.

    Actions
    ------
        0 Rock
        1 Paper
        2 Scissors
    
    Payoffs
    -----
        At the start, the matchup ``(Rock, Paper)`` is augmented to 
        ``(-v, v)``, then after ``phase_length``, it returns to normal 
        and the matchup ``(Paper, Scissors)`` is augmented to 
        ``(-v, v)``, then after ``phase_length``, it returns to normal
        and the matchup ``(Scissors, Rock)`` is augmented to 
        ``(-v, v)``, and then after ```phase_length``, we start over.

    default: v = 6, phase_length = 3000.
    """

    n_agents = 2
    n_actions = 3

    def __init__(
            self,
            v: float = 6.,
            phase_length : int = 3000,
            **kwargs,
    ):
        self._v = v
        self._phase_length = phase_length
        self._current_phase = 0
        super().__init__(**kwargs)

        self.register_buffer(
            "_total_steps",
            torch.zeros(self._num_envs, dtype=torch.long, device=self.device),
        )

    def _build_payoff(self, **_) -> torch.Tensor:

        v = self._v
        phase = self._current_phase

        outcome = torch.tensor([
            [ 0., -1.,  1.],   # Rock
            [ 1.,  0., -1.],   # Paper
            [-1.,  1.,  0.],   # Scissors
        ], dtype=torch.float32)

        if phase == 0:
            outcome[0, 1] = -v
            outcome[1, 0] = v
        elif phase == 1:
            outcome[1, 2] = -v
            outcome[2, 1] = v
        else:
            outcome[2, 0] = -v
            outcome[0, 2] = v

        return torch.stack([outcome, -outcome], dim=0)

    def _step(self, tensordict: TensorDict):
        self._total_steps.add_(1)
        
        td = super()._step(tensordict)

        if (self._total_steps[0] % self._phase_length == 0) and (self._total_steps[0] > 0):
            self._current_phase = (self._current_phase + 1) % 3
            self._payoff = self._build_payoff().to(self.device)

        return td
    
    def _reset(self, tensordict=None):
        td = super()._reset(tensordict)

        return td
    
class StaticBiasedRPSEnv(MatrixGameEnv):
    """
    A static version of baised rps.

    only rock paper interaction is amplified
    """
    n_agents = 2
    n_actions = 3

    def _build_payoff(self, **_) -> torch.Tensor:
        
        outcome = torch.tensor(
            [
                [0., -2., 1.],
                [2., 0., -1.],
                [-1., 1., 0.],
            ], dtype=torch.float32,
        )

        return torch.stack([outcome, -outcome], dim=0)

#################### Registry ###################

_REGISTRY: dict[str, type[MatrixGameEnv]] = {
    "prisoners_dilemma": PrisonersDilemmaEnv,
    "rock_paper_scissors": RockPaperScissorsEnv,
    "stag_hunt": StagHuntEnv,
    "battle_of_sexes": BattleOfSexesEnv,
    "biased_rps": BiasedRPSEnv,
    "static_biased_rps": StaticBiasedRPSEnv,
}


##################### Factory ####################

def MatrixGameFactory(
        scenario: str,
        num_envs: int = 1,
        max_steps: int = 1,
        device: str = "cpu",
        seed: int = 0,
        **kwargs,
) -> MatrixGameEnv:
    """
    Matrix Games API Interface.

    Parameters
    ---------
    scenario : str
        One if implemented games (e.g., ``prisonners_dilemma``).
    num_envs : int
        Number of parallel environments (= batch size).
    max_steps : int
        Episode length (number of repeated interactions).
    device : str
        Torch device.
    seed : int
        RNG seed.
    **kwargs :
        Game-specific keyword arguments forwarded to constructors.
    """

    if scenario not in _REGISTRY:
        raise ValueError(
            f"Unknown scenario '{scenario}'."
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

