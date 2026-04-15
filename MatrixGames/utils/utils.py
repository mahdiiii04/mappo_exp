import torch

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