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
    parser.add_argument(
        "--envs",
        nargs='+',
        default=['Walker2d-v5', 'Hopper-v5', 'InvertedDoublePendulum-v5'],
        help="List of environment names"
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=5,
        help="Window size for smoothing (1 = no smoothing)"
    )
    args = parser.parse_args()

    # List of alphas we are comparing
    alphas = ['0.1', '0.25', '0.5', '0.75', '0.9']
    # Corresponding directory names under each env
    policies = [f'GaussianTD3/{a}' for a in alphas]

    # For collecting final results across all envs for LaTeX table
    # table_data[env][alpha] = (mean_final, std_error_final)
    table_data = {}

    for env in args.envs:
        env_dir = f"./results/{env}"
        if not os.path.exists(env_dir):
            print(f"No results found for environment {env}")
            continue

        plt.figure(figsize=(12, 8))

        # Per-env dict: alpha -> (mean_final, stderr_final)
        env_results = {}

        for alpha, policy in sorted(zip(alphas, policies), key=lambda x: float(x[0])):
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

            timesteps = np.array(
                [0 if i == 0 else i * 5000 for i in range(len(mean_rewards))]
            )  # Assuming eval_freq=5000

            # Nice legend label with alpha
            label = f"GaussianTD3, $\\alpha={alpha}$"
            plt.plot(timesteps, mean_to_plot, label=label, marker='o')
            plt.fill_between(
                timesteps,
                mean_to_plot - std_to_plot,
                mean_to_plot + std_to_plot,
                alpha=0.3
            )

            # Final stats for this alpha/env
            final_rewards = [eval_data[-1] for eval_data in evaluations]
            mean_final = np.mean(final_rewards)
            std_error_final = np.std(final_rewards) / np.sqrt(len(final_rewards))
            env_results[alpha] = (mean_final, std_error_final)

        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        plt.title(f'GaussianTD3 $\\alpha$ Comparison on {env} (Averaged over Seeds)')
        plt.legend()
        plt.grid(True)

        os.makedirs("./results", exist_ok=True)
        plot_path = f"./results/{env}_comparison_of_alpha.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")

        # Print final means for this env
        print(f"\nFinal Average Rewards for {env}:")
        if not env_results:
            print("  No alpha configs with data.")
        else:
            for alpha in sorted(env_results.keys(), key=float):
                mean_final, std_error_final = env_results[alpha]
                print(f"alpha={alpha}: {mean_final:.3f} ± {std_error_final:.3f}")

            # Store for table (only if we had something)
            table_data[env] = env_results

    # After looping over all envs, create a LaTeX table
    if table_data:
        latex_path = "./results/gaussian_td3_alpha_table.tex"
        with open(latex_path, "w") as f:
            # Column spec: 1 for env + one per alpha
            col_spec = "l" + "c" * len(alphas)

            f.write("\\begin{table}[ht]\n")
            f.write("    \\centering\n")
            f.write(f"    \\begin{{tabular}}{{{col_spec}}}\n")
            f.write("        \\hline\n")

            # Header row
            header_cells = ["Environment"] + [f"$\\alpha={a}$" for a in alphas]
            header_line = " & ".join(header_cells) + " \\\\\n"
            f.write("        " + header_line)
            f.write("        \\hline\n")

            # Body rows: one per env
            for env in sorted(table_data.keys()):
                env_results = table_data[env]
                env_name = env.replace("_", "\\_")  # just in case

                row_cells = [env_name]
                for a in alphas:
                    if a in env_results:
                        mean_final, std_error_final = env_results[a]
                        cell = f"{mean_final:.1f} $\\pm$ {std_error_final:.1f}"
                    else:
                        cell = "--"
                    row_cells.append(cell)

                row_line = " & ".join(row_cells) + " \\\\\n"
                f.write("        " + row_line)

            f.write("        \\hline\n")
            f.write("    \\end{tabular}\n")
            f.write("    \\caption{Final average return (mean $\\pm$ standard error over seeds) for different $\\alpha$ values in GaussianTD3.}\n")
            f.write("    \\label{tab:gaussian_td3_alpha}\n")
            f.write("\\end{table}\n")

        print(f"\nLaTeX table written to {latex_path}")
    else:
        print("No data collected for LaTeX table (no environments had results).")

if __name__ == "__main__":
    main()
