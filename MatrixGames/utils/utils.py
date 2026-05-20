import torch

import numpy as np

from torchrl.envs.utils import ExplorationType, set_exploration_type
from tensordict import TensorDict

from matrix_games import MatrixGameEnv

@torch.no_grad()
def compute_nash_conv(
    env: MatrixGameEnv,
    policy,
    device: str | torch.device = None,
) -> tuple[float, torch.Tensor]:   # return type corrected
    if device is None:
        device = env.device

    td = env.rollout(
        max_steps=env.max_steps,
        policy=policy,
        auto_reset=True,
        tensordict=None,
    )

    num_episodes = td.batch_size[0]
    max_steps = env.max_steps
    n_agents = env.n_agents
    n_actions = env.n_actions

    obs = td.get(("agents", "observation"))
    flat_obs = obs.reshape(-1, n_agents, obs.shape[-1])

    input_td = TensorDict(
        {"agents": {"observation": flat_obs}},
        batch_size=[flat_obs.shape[0]],
        device=device,
    )
    dist = policy.get_dist(input_td)

    probs = dist.probs.reshape(num_episodes, max_steps, n_agents, n_actions)
    avg_pi = probs.mean(dim=(0, 1))          # shape (n_agents, n_actions)

    payoff = env._payoff.to(device)

    u = torch.zeros(n_agents, device=device)
    u[0] = torch.einsum("a,ab,b->", avg_pi[0], payoff[0], avg_pi[1])
    u[1] = torch.einsum("a,ab,b->", avg_pi[0], payoff[1], avg_pi[1])

    br_0 = (payoff[0] @ avg_pi[1]).max()
    br_1 = (avg_pi[0] @ payoff[1]).max()

    nash_conv = (br_0 - u[0] + br_1 - u[1]).item()

    return nash_conv, avg_pi.clone().cpu()

def evaluate_policy(env_test, policy):
    policy.eval()

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        td = env_test.rollout(
            max_steps=env_test.max_steps,
            policy=policy,
            auto_reset=True,
            break_when_any_done=False,
            tensordict=env_test.reset(),
        )

        episode_r = td.get(("next", "agents", "episode_reward"))
        # rollout on batched env usually gives [max_steps, num_envs, n_agents, 1] or [num_envs, max_steps, ...]
        if episode_r.ndim == 4 and episode_r.shape[1] == env_test.max_steps:
            final_rewards = episode_r[:, -1]      # time-major
        else:
            final_rewards = episode_r[-1]         # fallback (works in both common cases)

        mean_episode_reward = final_rewards.mean().item()

    policy.train()
    return mean_episode_reward

@torch.no_grad()
def compute_relative_nash_conv(
    env: MatrixGameEnv,
    policy,                    
    device: str | torch.device = None,
    relative: bool = True,
) -> tuple[float, torch.Tensor]:
    if device is None:
        device = env.device

    td = env.rollout(
        max_steps=env.max_steps,
        policy=policy,
        auto_reset=True,
        tensordict=None,
    )

    num_episodes = td.batch_size[0]
    max_steps = env.max_steps
    n_agents = env.n_agents
    n_actions = env.n_actions

    obs = td.get(("agents", "observation"))
    flat_obs = obs.reshape(-1, n_agents, obs.shape[-1])

    input_td = TensorDict({"agents": {"observation": flat_obs}}, 
                          batch_size=[flat_obs.shape[0]], device=device)
    
    dist = policy.get_dist(input_td)
    probs = dist.probs.reshape(num_episodes, max_steps, n_agents, n_actions)

    avg_pi = probs.mean(dim=(0, 1))

    payoff = env._payoff.to(device)

    u = torch.zeros(n_agents, device=device)
    for i in range(n_agents):
        u[i] = torch.einsum("a,ab,b->", avg_pi[i], payoff[i], avg_pi[1 - i])

    br_0 = (payoff[0] @ avg_pi[1]).max()
    br_1 = (payoff[1].T @ avg_pi[0]).max()

    raw_nash_conv = (br_0 - u[0] + br_1 - u[1]).item()

    if relative:
        # Better normalization for biased RPS with factor v
        v = getattr(env, '_v', 6.0)
        # Theoretical max exploitability per agent is (v-1), total for both agents is 2*(v-1)
        max_exploit = 2 * (v - 1)          # <--- This is the key change
        nash_conv = raw_nash_conv / max_exploit
    else:
        nash_conv = raw_nash_conv

    return nash_conv, avg_pi.clone().cpu()

# ── Add this function to utils/utils.py ──────────────────────────────────────

@torch.no_grad()
def compute_nash_conv_qnet(
    env,
    qnet,
    device=None,
    temperature: float = 1.0,
) -> tuple[float, torch.Tensor]:
    """Nash convergence for Q-value policies (QMIX / VDN / IQL).

    Instead of sampling a distribution, we run the Q-network over a rollout,
    then convert Q-values to a soft policy via softmax(Q / temperature).
    temperature=1.0 gives a reasonable soft policy; lower values approach
    greedy (one-hot), higher values approach uniform.
    """
    if device is None:
        device = env.device

    n_agents  = env.n_agents
    n_actions = env.n_actions

    # ── collect one rollout (greedy, no exploration) ─────────────────────────
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.rollout(
            max_steps=env.max_steps,
            policy=qnet,
            auto_reset=True,
            tensordict=None,
        )

    num_episodes = td.batch_size[0]
    max_steps    = env.max_steps

    obs      = td.get(("agents", "observation"))           # [E, T, n_agents, obs_dim]
    flat_obs = obs.reshape(-1, n_agents, obs.shape[-1])    # [E*T, n_agents, obs_dim]

    input_td = TensorDict(
        {"agents": {"observation": flat_obs}},
        batch_size=[flat_obs.shape[0]],
        device=device,
    )

    # Forward pass → action_value shape: [E*T, n_agents, n_actions]
    out_td      = qnet(input_td)
    action_vals = out_td.get(("agents", "action_value"))   # [E*T, n_agents, n_actions]

    # Softmax over actions to get a differentiable policy proxy
    probs = torch.softmax(action_vals / temperature, dim=-1)
    probs = probs.reshape(num_episodes, max_steps, n_agents, n_actions)

    avg_pi = probs.mean(dim=(0, 1))   # [n_agents, n_actions]

    payoff = env._payoff.to(device)

    u    = torch.zeros(n_agents, device=device)
    u[0] = torch.einsum("a,ab,b->", avg_pi[0], payoff[0], avg_pi[1])
    u[1] = torch.einsum("a,ab,b->", avg_pi[0], payoff[1], avg_pi[1])

    br_0 = (payoff[0] @ avg_pi[1]).max()
    br_1 = (avg_pi[0] @ payoff[1]).max()

    nash_conv = (br_0 - u[0] + br_1 - u[1]).item()

    return nash_conv, avg_pi.clone().cpu()