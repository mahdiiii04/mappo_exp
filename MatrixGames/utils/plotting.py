import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_policy_trajectory(file_path: str):
    data = np.load(file_path, allow_pickle=True).item()
    steps = data["steps"]
    policies = data["policies"]
    scenario = data["scenario_name"]
    n_actions = data["n_actions"]

    plt.figure(figsize=(8, 6))

    if n_actions == 2:
        p0 = policies[:, 0, 0]
        p1 = policies[:, 1, 0]
        plt.plot(p0, p1, 'b-', alpha=0.7, label="Policy trajectory")
        plt.scatter(p0[0], p1[0], c='green', s=80, label="Start")
        plt.scatter(p0[-1], p1[-1], c='red', s=80, label="End (NE)")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Agent 0 prob(action 0)")
        plt.ylabel("Agent 1 prob(action 0)")
        plt.title(f"Policy trajectory - {scenario} (2 actions)")
        plt.grid(True)
        plt.legend()
    elif n_actions == 3:
        p = policies[:, 0]   # use agent 0 (symmetric)
        x = 0.5 * (p[:, 1] + 2 * p[:, 2])          # Paper + 2*Scissors
        y = np.sqrt(3) / 2 * p[:, 1]                # height for Paper
        plt.plot(x, y, 'b-', alpha=0.7)
        plt.scatter(x[0], y[0], c='green', s=80)
        plt.scatter(x[-1], y[-1], c='red', s=80)
        # Draw simplex triangle
        plt.plot([0, 0.5, 1, 0], [0, np.sqrt(3)/2, 0, 0], 'k-', lw=1)
        plt.text(0, 0, 'Rock', ha='center')
        plt.text(1, 0, 'Scissors', ha='center')
        plt.text(0.5, np.sqrt(3)/2, 'Paper', ha='center')
        plt.axis('equal')
        plt.title(f"Policy trajectory — {scenario} (simplex)")
    else:
        plt.plot(steps, policies[:, 0, :], label=[f"Agent 0 action {a}" for a in range(n_actions)])
        plt.title(f"Raw probabilities — {scenario}")

    plt.tight_layout()
    plt.savefig(f"simplex_{scenario}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_policy_trajectory.py <filename.npy>")
        print("Example: python plot_policy_trajectory.py policy_trajectory_stag_hunt_seed42.npy")
        sys.exit(1)

    file_name = sys.argv[1]
    plot_policy_trajectory(file_name)