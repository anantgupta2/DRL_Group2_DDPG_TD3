import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob

def smooth(data, window):
    if window <= 1:
        return data
    return np.convolve(data, np.ones(window) / window, mode='same')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs='+', default=['Ant-v5', 'InvertedPendulum-v5', 'Walker2d-v5'], help="List of environment names")
    parser.add_argument("--smooth_window", type=int, default=5, help="Window size for smoothing (1 = no smoothing)")
    args = parser.parse_args()

    for env in args.envs:
        env_dir = f"./results/{env}"
        if not os.path.exists(env_dir):
            print(f"No results found for environment {env}")
            continue

        policies = [d for d in os.listdir(env_dir) if os.path.isdir(os.path.join(env_dir, d))]
        if not policies:
            print(f"No policies found in {env_dir}")
            continue

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
            num_seeds = len(evaluations)
            mean_rewards = np.mean(eval_array, axis=0)
            std_errors = np.std(eval_array, axis=0) / np.sqrt(num_seeds)

            if args.smooth_window > 1:
                mean_to_plot = smooth(mean_rewards, args.smooth_window)
                std_to_plot = smooth(std_errors, args.smooth_window)
            else:
                mean_to_plot = mean_rewards
                std_to_plot = std_errors

            timesteps = np.array([0 if i == 0 else i * 5000 for i in range(len(mean_rewards))])  # Assuming eval_freq=5000, adjust if needed

            plt.plot(timesteps, mean_to_plot, label=policy, marker='o')
            plt.fill_between(timesteps, mean_to_plot - std_to_plot, mean_to_plot + std_to_plot, alpha=0.3)

        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        plt.title(f'Policy Comparison on {env} (Averaged over Seeds)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./results/{env}_comparison.png")
        print(f"Plot saved to ./results/{env}_comparison.png")

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
            std_error_final = np.std(final_rewards) / np.sqrt(len(final_rewards))
            print(f"{policy}: {mean_final:.3f} ± {std_error_final:.3f}")

if __name__ == "__main__":
    main()