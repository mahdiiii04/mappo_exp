from tensordict import unravel_key
from torchrl.envs import Transform


def swap_last(source, dest):
    source = unravel_key(source)
    dest = unravel_key(dest)
    if isinstance(source, str):
        if isinstance(dest, str):
            return dest
        return dest[-1]
    if isinstance(dest, str):
        return source[:-1] + (dest,)
    return source[:-1] + (dest[-1],)

class DoneTransform(Transform):
    def __init__(self, reward_key, done_keys):
        super().__init__()
        self.reward_key = reward_key
        self.done_keys = done_keys

    def forward(self, tensordict):
        for done_key in self.done_keys:
            new_name = swap_last(self.reward_key, done_key)
            tensordict.set(
                ("next", new_name),
                tensordict.get(("next", done_key)).reshape(tensordict.get(("next", self.reward_key)).shape),
            )

        return tensordict

def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))

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

        #episode_rewards = td.get(("agents", "episode_reward")).sum(-1)
        #mean_episode_reward = episode_rewards.mean().item()

        done = td.get(("agents", "done"))
        final_rewards = td.get(("agents", "episode_reward"))[done]
        mean_episode_reward = final_rewards.mean().item()

    policy.train()
    return mean_episode_reward