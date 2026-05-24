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
            val = tensordict.get(("next", done_key), default=None)
            if val is None:
                # Key doesn't exist for this group/environment — skip.
                continue
            new_name = swap_last(self.reward_key, done_key)
            tensordict.set(
                ("next", new_name),
                val.unsqueeze(-1).expand(
                    tensordict.get(("next", self.reward_key)).shape
                ),
            )
        return tensordict