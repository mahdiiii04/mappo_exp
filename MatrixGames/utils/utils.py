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
) -> float:
    """Compute NashConv using the empirical time-averaged policy from full rollouts.
    No hard-coded num_episodes anymore — it uses whatever batch_size your env has.
    """
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

    avg_pi = probs.mean(dim=(0, 1))          

    payoff = env._payoff.to(device)          
    u = torch.zeros(n_agents, device=device)
    for i in range(n_agents):
        # u_i = π_i @ payoff_i @ π_{-i}^T
        u[i] = torch.einsum("a,ab,b->", avg_pi[i], payoff[i], avg_pi[1 - i])

    br_0 = (payoff[0] @ avg_pi[1]).max()          # agent 0 vs π1
    br_1 = (payoff[1].T @ avg_pi[0]).max()        # agent 1 vs π0

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

def evaluate_policy_tabular(env_test, policy, n_episodes=10):
    """Evaluate tabular policy over several episodes."""
    total_rewards = []
    for _ in range(n_episodes):
        td = env_test.reset()
        done = False
        ep_reward = 0
        while not done:
            with torch.no_grad():
                actions = policy(td)  # one‑hot
                td = env_test.step(td.set(("agents", "action"), actions))
            reward = td.get(("next", "agents", "reward")).sum().item()  # sum over agents
            ep_reward += reward
            done = td.get(("next", "agents", "done")).all().item()
        total_rewards.append(ep_reward)
    return np.mean(total_rewards)

def compute_nash_conv_tabular(env, policy_probs):
    """
    Compute NashConv for a tabular policy in a 2‑player matrix game.
    
    Args:
        env: MatrixGameEnv instance (must have n_agents==2)
        policy_probs: torch.Tensor of shape (n_agents, n_actions)
    
    Returns:
        nash_conv (float), policy_probs (numpy array)
    """
    n_agents = env.n_agents
    n_actions = env.n_actions
    payoff = env._payoff  # shape (n_agents, n_actions, n_actions)

    # Current expected payoff for each agent
    current_values = torch.zeros(n_agents, device=policy_probs.device)
    for agent in range(n_agents):
        other = 1 - agent
        for a in range(n_actions):
            p_a = policy_probs[agent, a]
            for o in range(n_actions):
                p_o = policy_probs[other, o]
                if agent == 0:
                    r = payoff[agent, a, o]
                else:
                    r = payoff[agent, o, a]
                current_values[agent] += p_a * p_o * r

    # Best response payoff for each agent
    best_response_values = torch.zeros(n_agents, device=policy_probs.device)
    for agent in range(n_agents):
        other = 1 - agent
        best = -float('inf')
        for a in range(n_actions):
            expected = 0.0
            for o in range(n_actions):
                p_o = policy_probs[other, o]
                if agent == 0:
                    r = payoff[agent, a, o]
                else:
                    r = payoff[agent, o, a]
                expected += p_o * r
            best = max(best, expected)
        best_response_values[agent] = best

    nash_conv = (best_response_values - current_values).sum().item()
    return nash_conv, policy_probs.cpu().numpy()