import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, K=5, kappa=1.0):
		super(Critic, self).__init__()
		self.kappa = kappa
		self.K = K
		
		# Quantile fractions: [0.1, 0.3, 0.5, 0.7, 0.9] for K=5
		self.register_buffer(
			"taus",
			((torch.arange(K, device=device, dtype=torch.float32) + 0.5) / K).view(1, K)
		)
		
		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, K)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, K)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1
	
	def quantile_huber_loss(self, pred, target):
		"""
		Quantile Regression Loss with Huber smoothing.
		
		pred:   [B, K] - predicted quantiles
		target: [B, K] - target quantiles from Bellman backup
		"""
		# TD errors
		td_error = target - pred  # [B, K]
		
		# Huber loss: smooth L1
		abs_td = td_error.abs()
		huber = torch.where(
			abs_td <= self.kappa,
			0.5 * td_error.pow(2),
			self.kappa * (abs_td - 0.5 * self.kappa)
		)
		
		# Quantile weighting: asymmetric penalty
		quantile_weight = torch.abs(
			self.taus - (td_error.detach() < 0).float()
		)
		
		return (quantile_weight * huber).mean()

	def quantile_huber_loss_pairwise(self, pred, target):
		diff = target.unsqueeze(1) - pred.unsqueeze(2)  
		tau = self.taus.unsqueeze(-1)  
		quantile_weight = torch.abs(tau - (diff.detach() < 0).float())
		abs_diff = diff.abs()
		huber = torch.where(
			abs_diff <= self.kappa,
			0.5 * (diff ** 2) / self.kappa,
			abs_diff - 0.5 * self.kappa
		)
		return (quantile_weight * huber).mean()
	
	def loss(self, q1, q2, target_q):
		#return self.quantile_huber_loss(q1, target_q) + self.quantile_huber_loss(q2, target_q)
		return self.quantile_huber_loss_pairwise(q1, target_q) + self.quantile_huber_loss_pairwise(q2, target_q)
	

class QRTD3(object):
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
		K=5,  
		alpha=0.5,
		critic_lr=1e-4,  # REDUCED learning rate
		actor_lr=1e-4
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

		self.critic = Critic(state_dim, action_dim, K=K).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.K = K
		self.alpha = alpha
		self.total_it = 0
		
		# For debugging
		self.q_history = []

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Target policy smoothing
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Twin Q targets
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)  # [B, K]
			
			# Reshape for broadcasting
			reward = reward.view(-1, 1)  # [B, 1]
			not_done = not_done.view(-1, 1)  # [B, 1]
			
			# Bellman backup
			target_Q = reward + not_done * self.discount * target_Q  # [B, K]
			
			# CLIP targets to prevent explosion
			target_Q = torch.clamp(target_Q, -1000, 1000)

		# Current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Critic loss
		critic_loss = self.critic.loss(current_Q1, current_Q2, target_Q)

		# Optimize critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		
		# Gradient clipping
		critic_grad_norm = torch.nn.utils.clip_grad_norm_(
			self.critic.parameters(), max_norm=10.0
		)
		
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Actor objective: maximize CVaR (risk-aware, focusing on worst-case scenarios)
			q1 = self.critic.Q1(state, self.actor(state))  # [B, K]
			q1_sorted, _ = torch.sort(q1, dim=-1)  # Sort to get ordered quantiles
			m = max(1, int(np.ceil(self.alpha * self.K)))
			cvar = q1_sorted[:, :m].mean(dim=-1)  # [B]
			actor_loss = -cvar.mean()
			
			# Logging every 500 steps
			if self.total_it % 5000 == 0:
				with torch.no_grad():
					# Current Q statistics
					q_mean = q1.mean().item()
					q_std = q1.std().item()
					q_min = q1.min().item()
					q_max = q1.max().item()
					cvar_val = cvar.mean().item()
					median_idx = self.K // 2
					median_val = q1_sorted[:, median_idx].mean().item()
					
					# Check if quantiles are ordered
					q1_sorted_check, _ = torch.sort(q1, dim=-1)
					disorder = (q1 - q1_sorted_check).abs().mean().item()
					
					# Target Q statistics
					tgt_mean = target_Q.mean().item()
					tgt_std = target_Q.std().item()
					
					# Reward statistics
					rew_mean = reward.mean().item()
					rew_std = reward.std().item()
					
					print(f"\n{'='*70}")
					print(f"Step {self.total_it:7d} | Diagnostics (CVaR_{int(self.alpha*100)}%)")
					print(f"{'='*70}")
					print(f"Reward:       mean={rew_mean:8.2f}  std={rew_std:7.2f}")
					print(f"Target Q:     mean={tgt_mean:8.2f}  std={tgt_std:7.2f}")
					print(f"Current Q:    mean={q_mean:8.2f}  std={q_std:7.2f}")
					print(f"Q range:      [{q_min:8.2f}, {q_max:8.2f}]")
					print(f"CVaR_{int(self.alpha*100)}%:      {cvar_val:8.2f}")
					print(f"Median (50%): {median_val:8.2f}")
					print(f"Disorder:     {disorder:.6f}")
					print(f"Critic loss:  {critic_loss.item():.4f}")
					print(f"Critic grad:  {critic_grad_norm:.4f}")
					print(f"Actor loss:   {actor_loss.item():.4f}")
					print(f"{'='*70}\n")
			
			# Logging every 500 steps
			if self.total_it % 5000 == 0:
				with torch.no_grad():
					# Statistics
					q_mean = q1.mean().item()
					q_std = q1.std().item()
					q_min = q1.min().item()
					q_max = q1.max().item()
					
					target_mean = target_Q.mean().item()
					rew_mean = (reward).mean().item()  # Unscale for display
					
					# Check if quantiles are ordered
					q1_sorted, _ = torch.sort(q1, dim=-1)
					disorder = (q1 - q1_sorted).abs().mean().item()
					
					# Print diagnostics
					print(f"\n{'='*70}")
					print(f"Step {self.total_it:7d} | Diagnostics")
					print(f"{'='*70}")
					print(f"  Reward (unscaled):  mean={rew_mean:8.2f}")
					print(f"  Target Q:           mean={target_mean:8.2f}")
					print(f"  Current Q:          mean={q_mean:8.2f}  std={q_std:7.2f}")
					print(f"  Q range:            [{q_min:8.2f}, {q_max:8.2f}]")
					print(f"  Quantile disorder:  {disorder:.6f}")
					print(f"  Critic loss:        {critic_loss.item():.4f}")
					print(f"  Critic grad norm:   {critic_grad_norm:.4f}")
					print(f"  Actor loss:         {actor_loss.item():.4f}")
					print(f"{'='*70}\n")
					
					# Track Q evolution
					self.q_history.append({
						'step': self.total_it,
						'q_mean': q_mean,
						'q_min': q_min,
						'q_max': q_max
					})
			
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