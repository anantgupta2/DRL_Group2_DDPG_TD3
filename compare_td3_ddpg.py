import csv
import math

# Read times
times_data = {}
with open('results/training_times.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        env = row['Env']
        policy = row['Policy']
        seed = int(row['Seed'])
        time_str = row['Time (HH:MM)']
        h, m = map(int, time_str.split(':'))
        time_min = h * 60 + m
        if env not in times_data:
            times_data[env] = {'TD3': [], 'DDPG': []}
        times_data[env][policy].append(time_min)

# Read rewards
rewards_data = {}
with open('results/final_rewards.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        env = row['Env']
        policy = row['Policy']
        seed = int(row['Seed'])
        reward = float(row['Final Reward'])
        if env not in rewards_data:
            rewards_data[env] = {'TD3': [], 'DDPG': []}
        rewards_data[env][policy].append(reward)

# Get envs
envs = sorted(set(times_data.keys()) | set(rewards_data.keys()))

# Function to calc mean ± se
def calc_stats(values):
    if not values:
        return 'N/A'
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return f"{mean:.1f} ± 0.0"
    variance = sum((x - mean)**2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    se = std / math.sqrt(n)
    return f"{mean:.1f} ± {se:.1f}"

# Results
results = []
for env in envs:
    td3_times = times_data.get(env, {}).get('TD3', [])
    ddpg_times = times_data.get(env, {}).get('DDPG', [])
    td3_rewards = rewards_data.get(env, {}).get('TD3', [])
    ddpg_rewards = rewards_data.get(env, {}).get('DDPG', [])
    
    td3_time_str = calc_stats(td3_times)
    ddpg_time_str = calc_stats(ddpg_times)
    td3_reward_str = calc_stats(td3_rewards)
    ddpg_reward_str = calc_stats(ddpg_rewards)
    
    results.append((env, td3_reward_str, td3_time_str, ddpg_reward_str, ddpg_time_str))

# LaTeX
latex = """\\begin{table}[h]
\\centering
\\begin{tabular}{l c c c c}
\\hline
\\multirow{2}{*}{Environment} & \\multicolumn{2}{c}{TD3} & \\multicolumn{2}{c}{DDPG} \\\\
\\cline{2-5}
& Reward & Time (min) & Reward & Time (min) \\\\
\\hline
"""

for env, tr, tt, dr, dt in results:
    latex += f"{env} & {tr} & {tt} & {dr} & {dt} \\\\\n"

latex += """\\hline
\\end{tabular}
\\caption{Comparison of TD3 and DDPG: Final Rewards and Training Time}
\\end{table}
"""

print(latex)