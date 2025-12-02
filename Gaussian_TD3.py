import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class GaussianCritic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(GaussianCritic, self).__init__()
		
		# Q1 architecture - outputs mean and log_std
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.mean_head1 = nn.Linear(256, 1)
		self.log_std_head1 = nn.Linear(256, 1)

		# Q2 architecture - outputs mean and log_std
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.mean_head2 = nn.Linear(256, 1)
		self.log_std_head2 = nn.Linear(256, 1)
		
		# Constrain std to reasonable range
		self.min_log_std = -5
		self.max_log_std = 1

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		# Q1 distribution
		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		mean1 = self.mean_head1(q1)
		log_std1 = self.log_std_head1(q1)
		log_std1 = torch.clamp(log_std1, self.min_log_std, self.max_log_std)

		# Q2 distribution
		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		mean2 = self.mean_head2(q2)
		log_std2 = self.log_std_head2(q2)
		log_std2 = torch.clamp(log_std2, self.min_log_std, self.max_log_std)

		return (mean1, log_std1), (mean2, log_std2)

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		mean1 = self.mean_head1(q1)
		log_std1 = self.log_std_head1(q1)
		log_std1 = torch.clamp(log_std1, self.min_log_std, self.max_log_std)
		return mean1, log_std1
	
	def gaussian_nll_loss(self, mean_pred, log_std_pred, target):
		"""
		Negative log-likelihood loss for Gaussian distribution.
		"""
		std_pred = torch.exp(log_std_pred)
		
		# NLL = 0.5 * log(2π) + log(σ) + (target - μ)² / (2σ²)
		nll = 0.5 * np.log(2 * np.pi) + log_std_pred + \
		      0.5 * ((target - mean_pred) / std_pred) ** 2
		
		return nll.mean()

	def loss(self, dist1, dist2, target):
		"""
		Total critic loss for both Gaussian critics.
		"""
		mean1, log_std1 = dist1
		mean2, log_std2 = dist2
		
		loss1 = self.gaussian_nll_loss(mean1, log_std1, target)
		loss2 = self.gaussian_nll_loss(mean2, log_std2, target)
		
		return loss1 + loss2


class GaussianTD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=0.1,  # CVaR level (worst 25%)
		critic_lr=3e-4,
		actor_lr=3e-4
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

		self.critic = GaussianCritic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha
		self.total_it = 0
		
		# Precompute CVaR quantile for standard normal
		self.z_alpha = norm.ppf(alpha)  # e.g., -0.674 for alpha=0.25

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()
	
	def compute_cvar(self, mean, std, n_samples=50):
		"""
		Compute CVaR_alpha for Gaussian distribution using Monte Carlo sampling.
		
		Args:
			mean: [B, 1] predicted mean Q-values
			std: [B, 1] predicted standard deviations
			n_samples: number of samples to draw from each distribution
		
		Returns:
			cvar: [B, 1] CVaR values (mean of worst alpha% samples)
		"""
		batch_size = mean.shape[0]
		
		# Sample n_samples from each Gaussian distribution
		# mean: [B, 1], std: [B, 1]
		# We want samples: [B, n_samples]
		
		# Create normal samples: [B, n_samples]
		epsilon = torch.randn(batch_size, n_samples, device=mean.device)
		samples = mean + std * epsilon  # Broadcasting: [B,1] + [B,1] * [B,n_samples]
		
		# Sort samples along the sample dimension
		sorted_samples, _ = torch.sort(samples, dim=1)  # [B, n_samples], ascending
		
		# Take the worst alpha% of samples
		n_worst = max(1, int(np.ceil(self.alpha * n_samples)))
		worst_samples = sorted_samples[:, :n_worst]  # [B, n_worst]
		
		# CVaR = mean of worst samples
		cvar = worst_samples.mean(dim=1, keepdim=True)  # [B, 1]
		
		return cvar

	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		
		# Scale rewards for stability
		reward_scale = 0.1
		reward = reward * reward_scale

		with torch.no_grad():
			# Target policy smoothing
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Get target distributions
			(mean1_tgt, log_std1_tgt), (mean2_tgt, log_std2_tgt) = \
				self.critic_target(next_state, next_action)
			
			# Take minimum mean (conservative, like TD3)
			target_mean = torch.min(mean1_tgt, mean2_tgt)
			
			# Reshape for broadcasting
			reward = reward.view(-1, 1)
			not_done = not_done.view(-1, 1)
			
			# Bellman backup (deterministic target for stability)
			target = reward + not_done * self.discount * target_mean

		# Get current distributions
		(mean1, log_std1), (mean2, log_std2) = self.critic(state, action)

		# Critic loss (negative log-likelihood)
		critic_loss = self.critic.loss(
			(mean1, log_std1), (mean2, log_std2), target
		)

		# Optimize critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		critic_grad_norm = torch.nn.utils.clip_grad_norm_(
			self.critic.parameters(), max_norm=10.0
		)
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Get Q distribution for current policy
			mean1, log_std1 = self.critic.Q1(state, self.actor(state))
			std1 = torch.exp(log_std1)
			
			# Compute CVaR (risk-aware objective)
			cvar = self.compute_cvar(mean1, std1)
			actor_loss = -cvar.mean()
			#actor_loss = -mean1.mean()
			# Diagnostic logging every 500 steps
			if self.total_it % 500 == 0:
				with torch.no_grad():
					mean_val = mean1.mean().item()
					std_val = std1.mean().item()
					cvar_val = cvar.mean().item()
					
					# 5th and 95th percentiles for visualization
					q05 = (mean1 - 1.645 * std1).mean().item()
					q95 = (mean1 + 1.645 * std1).mean().item()
					
					target_mean_val = target.mean().item()
					rew_mean = (reward * (1/reward_scale)).mean().item()
					
					print(f"\n{'='*70}")
					print(f"Step {self.total_it:7d} | Gaussian Distributional TD3")
					print(f"{'='*70}")
					print(f"Reward:       mean={rew_mean:8.2f}")
					print(f"Target:       mean={target_mean_val:8.2f}")
					print(f"Q Dist:       μ={mean_val:7.2f}  σ={std_val:6.2f}")
					print(f"Q Range:      [5%: {q05:7.2f}, 95%: {q95:7.2f}]")
					print(f"CVaR_{int(self.alpha*100)}%:      {cvar_val:8.2f}")
					print(f"Critic loss:  {critic_loss.item():.4f}")
					print(f"Critic grad:  {critic_grad_norm:.4f}")
					print(f"Actor loss:   {actor_loss.item():.4f}")
					print(f"{'='*70}\n")
			
			# Optimize actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
			self.actor_optimizer.step()

			# Update target networks
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)