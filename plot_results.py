import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="Environment name")
    args = parser.parse_args()

    env_dir = f"./results/{args.env}"
    if not os.path.exists(env_dir):
        print(f"No results found for environment {args.env}")
        return

    policies = [d for d in os.listdir(env_dir) if os.path.isdir(os.path.join(env_dir, d))]
    if not policies:
        print(f"No policies found in {env_dir}")
        return

    plt.figure(figsize=(12, 8))

    for policy in sorted(policies):
        policy_dir = os.path.join(env_dir, policy)
        seed_files = glob(os.path.join(policy_dir, "*.npy"))
        if not seed_files:
            continue

        evaluations = []
        for seed_file in seed_files:
            eval_data = np.load(seed_file)
            evaluations.append(eval_data)

        if not evaluations:
            continue

        # Stack into (num_seeds, num_timesteps)
        eval_array = np.array(evaluations)
        mean_rewards = np.mean(eval_array, axis=0)
        std_rewards = np.std(eval_array, axis=0)

        timesteps = np.array([0 if i == 0 else i * 5000 for i in range(len(mean_rewards))])  # Assuming eval_freq=5000, adjust if needed

        plt.plot(timesteps, mean_rewards, label=policy, marker='o')
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)

    plt.xlabel('Timestep')
    plt.ylabel('Average Reward')
    plt.title(f'Policy Comparison on {args.env} (Averaged over Seeds)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./results/{args.env}_comparison.png")
    print(f"Plot saved to ./results/{args.env}_comparison.png")

    # Print final means
    print("\nFinal Average Rewards:")
    for policy in sorted(policies):
        policy_dir = os.path.join(env_dir, policy)
        seed_files = glob(os.path.join(policy_dir, "*.npy"))
        if not seed_files:
            continue
        final_rewards = []
        for seed_file in seed_files:
            eval_data = np.load(seed_file)
            final_rewards.append(eval_data[-1])
        mean_final = np.mean(final_rewards)
        std_final = np.std(final_rewards)
        print(f"{policy}: {mean_final:.3f} ± {std_final:.3f}")

if __name__ == "__main__":
    main()