import torch
import json
import os

import numpy as np

from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

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
def compute_positional_entropy(env) -> float:
    """
    Proxy for agent exploration: average Shannon entropy of each agent's
    empirical row and column distributions across parallel envs.

    Uses current _positions buffer. Returns a scalar in [0, log(grid_size)].
    """
    positions = env._positions          # [ne, n_agents, 2]
    gs        = env.grid_size
    ne        = env._num_envs
    entropies = []

    for a_i in range(env.n_agents):
        for dim in range(2):            # row, col
            coords = positions[:, a_i, dim]   # [ne]
            counts = torch.bincount(coords, minlength=gs).float()
            probs  = counts / counts.sum().clamp(min=1e-8)
            probs  = probs.clamp(min=1e-10)
            h      = -(probs * probs.log()).sum().item()
            entropies.append(h)

    return float(np.mean(entropies))


@torch.no_grad()
def compute_mean_goal_distance(env) -> float:
    """
    Average L1 distance between each agent and its assigned goal(s).
    Returns NaN for scenarios without goals (e.g. PredatorPrey).
    """
    if env.n_goals == 0 or env._goals is None:
        return float("nan")

    positions = env._positions.float()    # [ne, n_agents, 2]
    goals     = env._goals.float()        # [ne, n_goals, 2]

    # pair agent i with goal min(i, n_goals-1)
    dists = []
    for i in range(env.n_agents):
        g_i   = min(i, env.n_goals - 1)
        goal  = goals[:, g_i, :]                           # [ne, 2]
        d     = (positions[:, i, :] - goal).abs().sum(-1)  # [ne]
        dists.append(d.mean().item())

    return float(np.mean(dists))

@torch.no_grad()
def save_episode_trace(
    env_factory,
    policy,
    critic,
    loss_module,
    file_path: str,
):
    env = env_factory()
    td = env.reset()

    if loss_module.functional:
        actor_ctx = loss_module.actor_network_params.to_module(policy)
        critic_ctx = loss_module.critic_network_params.to_module(critic)
    else:
        from contextlib import nullcontext
        actor_ctx = nullcontext()
        critic_ctx = nullcontext()

    episode_trace = []

    while True:
        # --- grid representation ---
        gs = env.grid_size
        positions = env._positions[0].cpu()
        goals = env._goals[0].cpu() if env.n_goals > 0 else None

        grid = [['.' for _ in range(gs)] for _ in range(gs)]
        for a_i, (r, c) in enumerate(positions):
            grid[r.item()][c.item()] = str(a_i)
        if goals is not None:
            for g_i, (r, c) in enumerate(goals):
                grid[r.item()][c.item()] = f'G{g_i}'
        grid_rows = [' '.join(row) for row in grid]

        # --- policy ---
        with actor_ctx, set_exploration_type(ExplorationType.DETERMINISTIC):
            policy_out = policy(td.clone())

        action_td = policy_out.get(("agents", "action"))
        logits = policy_out.get(("agents", "logits"))
        probs = logits.softmax(dim=-1)

        # --- critic (proper input) ---
        obs_tensor = env._build_obs(env._positions)   # (1, n_agents, obs_dim)
        # Build TensorDict as expected by the ValueOperator
        td_critic = TensorDict(
            {
                "agents": TensorDict(
                    {"observation": obs_tensor},
                    batch_size=[1, env.n_agents],
                    device=env.device,
                )
            },
            batch_size=[1],
            device=env.device,
        )
        with critic_ctx:
            q_out = critic(td_critic)
        q_vals = q_out.get("q_value")   # (1, n_agents, n_actions)

        # --- per-agent info ---
        step_info = {"step": len(episode_trace), "grid": grid_rows, "agents": []}
        for a in range(env.n_agents):
            if action_td.ndim == 3 and action_td.shape[-1] == env.n_actions:
                act_idx = action_td[0, a].argmax().item()
            else:
                act_idx = action_td[0, a].item()

            step_info["agents"].append({
                "agent_id": a,
                "action": act_idx,
                "q_values": [round(v, 4) for v in q_vals[0, a].tolist()],
                "action_probs": [round(p, 4) for p in probs[0, a].tolist()],
            })

        episode_trace.append(step_info)

        td = env.step(policy_out)
        done = td.get(("agents", "done")).any() or td.get(("agents", "terminated")).any()
        if done or len(episode_trace) >= env.max_steps:
            break

    env.close()

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(episode_trace, f, indent=2)

    return episode_trace