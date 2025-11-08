import numpy as np
import torch
import gymnasium as gym
import argparse
import os
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import time

import utils
import TD3
import OurDDPG
import DDPG
import QR_TD3


def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	avg_reward = 0.
	for _ in tqdm(range(eval_episodes), desc="Evaluating"):
		state = eval_env.reset(seed = seed + 100)[0]
		done = False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, terminated, truncated, _ = eval_env.step(action)
			done = terminated or truncated
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# parser.add_argument("--alpha", default =0.5)
	# parser.add_argument("--K", default=4)
	args = parser.parse_args()

	start_time = time.time()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	# env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
		print(f"Policy TD3 initialized with state_dim={state_dim}, action_dim={action_dim}, max_action={max_action}")
	elif args.policy == "QRTD3":
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = QR_TD3.QRTD3(**kwargs)
		print(f"Policy QRTD3 initialized with state_dim={state_dim}, action_dim={action_dim}, max_action={max_action}")
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
		print(f"Policy OurDDPG initialized with state_dim={state_dim}, action_dim={action_dim}, max_action={max_action}")
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)
		print(f"Policy DDPG initialized with state_dim={state_dim}, action_dim={action_dim}, max_action={max_action}")

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")
		print(f"Model loaded from {policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	print(f"Replay buffer initialized with capacity for {state_dim} state and {action_dim} action dimensions")
	
	# Evaluate untrained policy
	print("Evaluating untrained policy...")
	evaluations = [eval_policy(policy, args.env, args.seed)]

	state = env.reset(seed = args.seed + 100)[0]
	done = False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	print(f"Starting training for {args.max_timesteps} timesteps with start_timesteps={args.start_timesteps}, eval_freq={args.eval_freq}")
	for t in tqdm(range(int(args.max_timesteps)), desc="Training"):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, terminated, truncated, _ = env.step(action) 
		done = terminated or truncated
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True=
			# Reset environment
			state = env.reset(seed = args.seed + 100)[0]
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			os.makedirs(f"./results/{args.env}/{args.policy}", exist_ok=True)
			if args.save_model: policy.save(f"./models/{file_name}")

	print("Training completed.")
	end_time = time.time()
	duration = end_time - start_time
	hours = int(duration // 3600)
	minutes = int((duration % 3600) // 60)
	time_str = f"{hours:02d}:{minutes:02d}"
	np.save(f"./results/{args.env}/{args.policy}/{args.seed}.npy", evaluations)
	print(f"Final evaluation: {evaluations[-1]:.3f}")
	if args.save_model:
		print(f"Model saved to ./models/{file_name}")
	print(f"Results saved to ./results/{args.env}/{args.policy}/{args.seed}.npy")

	# Append to final rewards table
	final_rewards_file = "./results/final_rewards.csv"
	file_exists = os.path.isfile(final_rewards_file)
	with open(final_rewards_file, 'a', newline='') as csvfile:
		writer = csv.writer(csvfile)
		if not file_exists:
			writer.writerow(["Policy", "Env", "Seed", "Final Reward"])
		writer.writerow([args.policy, args.env, args.seed, f"{evaluations[-1]:.3f}"])

	print(f"Final reward appended to ./results/final_rewards.csv")

	# Append to training times table
	training_times_file = "./results/training_times.csv"
	file_exists = os.path.isfile(training_times_file)
	with open(training_times_file, 'a', newline='') as csvfile:
		writer = csv.writer(csvfile)
		if not file_exists:
			writer.writerow(["Policy", "Env", "Seed", "Time (HH:MM)"])
		writer.writerow([args.policy, args.env, args.seed, time_str])

	print(f"Training time appended to ./results/training_times.csv")
