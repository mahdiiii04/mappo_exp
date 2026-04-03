import torch
import argparse
import json

from torchrl.envs import PettingZooEnv, RewardSum, TransformedEnv
from torchrl.envs.utils import check_env_specs

from algorithm import MAPPO


def main():
    parser = argparse.ArgumentParser(description="Training script")

    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--name', type=str, default='mpe/simple_spread_v3')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--agentic_keyword', type=str, default="agents")
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lmbda', type=float, default=0.9)
    parser.add_argument('--max_grad_norm', type=float, default=1.00)
    parser.add_argument('--share_params_policy', type=bool, default=True)
    parser.add_argument('--share_params_critic', type=bool, default=True)
    parser.add_argument('--frames_per_batch', type=int, default=6_000)
    parser.add_argument('--n_iters', type=int, default=5)
    parser.add_argument('--minibatch_size', type=int, default=128)
    parser.add_argument('--entropy_eps', type=float, default=1e-4)


    args = parser.parse_args()

    env_device = (
        torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    )   

    env = PettingZooEnv(
        task=args.name,
        parallel=True,
        device=env_device,          
        N=args.n_agents,                  
        local_ratio=0.5,
        max_cycles=args.max_steps,        
        continuous_actions=True
    )

    env.n_agents = args.n_agents

    env = TransformedEnv(
        env,
        RewardSum(in_keys=[(args.agentic_keyword, "reward")], out_keys=[(args.agentic_keyword, "episode_reward")])
    )

    check_env_specs(env)

    mappo = MAPPO(env, env_device, minibatch_size=args.minibatch_size,
                  lr=args.learning_rate, clip_epsilon=args.clip_epsilon,
                  gamma=args.gamma, lmbda=args.lmbda, entropy_eps=args.entropy_eps,
                  max_grad_norm=args.max_grad_norm, share_params_policy=args.share_params_policy,
                  share_params_critic=args.share_params_critic, frames_per_batch=args.frames_per_batch,
                  n_iters=args.n_iters, agentic_keyword=args.agentic_keyword
            )      

    result = mappo.train(num_epochs=args.n_epochs)

    with open('run_0000001', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)



if __name__ == "__main__":
    main()
