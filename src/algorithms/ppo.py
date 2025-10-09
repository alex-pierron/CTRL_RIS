import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.amp as amp
from torch.distributions import Normal


class RolloutBuffer:
    """
    On-policy rollout storage for PPO with GAE-Lambda for parallel environments.
    Stores: states, actions, rewards, dones, values, logprobs, env_ids.
    Unlike off-policy buffers (ReplayBuffer), this stores trajectories sequentially
    and computes advantages using Generalized Advantage Estimation (GAE).
    Handles multiple parallel environments by tracking which environment each transition belongs to.
    """
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, device: torch.device, n_rollout_envs: int = 1):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.n_rollout_envs = n_rollout_envs
        self.reset()

    def reset(self):
        """Initialize/reset all buffers to zeros and reset pointer."""
        self.states = torch.zeros((self.buffer_size, self.state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.buffer_size, self.action_dim), dtype=torch.float32, device=self.device)
        # ADDED: Buffer to store raw (pre-tanh) actions for correct log_prob evaluation
        self.raw_actions = torch.zeros((self.buffer_size, self.action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.buffer_size, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.buffer_size, 1), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((self.buffer_size, 1), dtype=torch.float32, device=self.device)
        self.logprobs = torch.zeros((self.buffer_size, 1), dtype=torch.float32, device=self.device)
        # ADDED: Track which environment each transition belongs to
        self.env_ids = torch.zeros((self.buffer_size, 1), dtype=torch.long, device=self.device)
        self.ptr = 0

    # CHANGED: Added raw_action and env_id to the signature
    def add(self, state, action, raw_action, reward, done, value, logprob, env_id=0):
        """
        Add a single transition to the buffer.
        """
        # Check if buffer is full
        if self.ptr >= self.buffer_size:
            return False
            
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        # ADDED: Store the raw action
        self.raw_actions[idx] = raw_action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value
        self.logprobs[idx] = logprob
        # ADDED: Store environment ID
        self.env_ids[idx] = env_id
        self.ptr += 1
        return True

    def compute_advantages(self, last_values: torch.Tensor, gamma: float, gae_lambda: float):
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE) for parallel environments.
        
        Args:
            last_values: Tensor of shape (n_rollout_envs, 1) containing the last value estimates for each environment
        """
        advantages = torch.zeros_like(self.rewards, device=self.device)
        
        # Process each environment separately to handle different episode lengths
        for env_id in range(self.n_rollout_envs):
            # Get indices for this environment
            env_mask = (self.env_ids[:self.ptr].squeeze() == env_id)
            env_indices = torch.where(env_mask)[0]
            
            if len(env_indices) == 0:
                continue
                
            # Get the last value for this environment
            last_value = last_values[env_id] if last_values.dim() > 0 else last_values
            
            # Compute advantages for this environment's trajectory
            last_gae = 0.0
            for i in reversed(range(len(env_indices))):
                step_idx = env_indices[i]
                
                if i == len(env_indices) - 1:
                    # Last step of this environment's trajectory
                    next_non_terminal = 1.0 - self.dones[step_idx]
                    next_value = last_value
                else:
                    # Not the last step
                    next_step_idx = env_indices[i + 1]
                    next_non_terminal = 1.0 - self.dones[step_idx]
                    next_value = self.values[next_step_idx]
                
                delta = self.rewards[step_idx] + gamma * next_value * next_non_terminal - self.values[step_idx]
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                advantages[step_idx] = last_gae
        
        # CHANGED: Returns are now computed correctly from advantages and values
        self.returns = advantages + self.values
        self.advantages = advantages
        
        # Trim buffers to actual size
        self.states = self.states[:self.ptr]
        self.actions = self.actions[:self.ptr]
        self.raw_actions = self.raw_actions[:self.ptr] # ADDED
        self.logprobs = self.logprobs[:self.ptr]
        self.values = self.values[:self.ptr]
        self.returns = self.returns[:self.ptr]
        self.advantages = self.advantages[:self.ptr]
        self.env_ids = self.env_ids[:self.ptr] # ADDED

    def get_minibatches(self, minibatch_size: int):
        """
        Generate randomized minibatches for SGD updates.
        """
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)
        
        for start in range(0, self.ptr, minibatch_size):
            end = start + minibatch_size
            mb_idx = indices[start:end]
            yield (
                self.states[mb_idx],
                self.actions[mb_idx],
                self.raw_actions[mb_idx], # ADDED raw_actions to the minibatch
                self.logprobs[mb_idx],
                self.advantages[mb_idx],
                self.returns[mb_idx],
                self.values[mb_idx],
                self.env_ids[mb_idx], # ADDED env_ids to the minibatch
            )


class ActorNetwork(nn.Module):
    """
    PPO Actor network.
    """
    def __init__(self, state_dim, action_dim, N_t, K, P_max, actor_linear_layers=[128, 128, 128]):
        super().__init__()
        self.N_t = N_t
        self.K = K
        self.P_max = P_max
        self.tensored_P_max = torch.tensor(P_max, dtype=torch.float32)

        self.action_dim = action_dim
        input_dim = state_dim
        self.linear_layers = nn.ModuleList()
        for layer_dim in actor_linear_layers:
            self.linear_layers.append(nn.Linear(input_dim, layer_dim))
            input_dim = layer_dim

        self.mean_output = nn.Linear(actor_linear_layers[-1], action_dim)
        self.log_std_param = nn.Parameter(torch.zeros(action_dim))

    # CHANGED: Simplified this projection operator, as the original was a bit complex
    # and might not be backprop-friendly if grads were ever needed (though here they aren't).
    def actor_W_projection_operator(self, raw_W):
        """
        Projects a batch of beamforming matrices W onto the set defined by the power constraint.

        Args:
            raw_W (ndarray): Batch of raw beamforming matrices of shape (batch_size, rows, columns).
        Returns:
            ndarray: Batch of projected beamforming matrices of the same shape as raw_W.
        """

        # Ensure float type
        raw_W = raw_W.detach()

        # Frobenius norms for each matrix in the batch
        frobenius_norms = torch.linalg.norm(raw_W, dim=(1, 2), ord='fro')

        # Traces of (W * W^H)
        traces = torch.einsum('bij,bji->b', raw_W, raw_W.conj().transpose(1, 2)).real

        # Mask of matrices exceeding power constraint
        exceed_mask = traces > self.tensored_P_max

        # Scaling factors
        scaling_factors = torch.where(
            exceed_mask,
            (self.tensored_P_max.sqrt() / frobenius_norms),
            torch.ones_like(frobenius_norms)
        )

        # Reshape scaling factors for broadcasting
        scaling_factors = scaling_factors.view(-1, 1, 1)

        # Apply scaling
        projected_W = raw_W * scaling_factors

        return projected_W
    

    def actor_process_raw_actions(self, raw_actions):
        """
        Transforms the raw model output into the phase noise matrix Theta and
        beamforming matrix W while ensuring unit modulus and power constraints.
        Works for batch processing.

        Args:
            raw_actions (torch.Tensor): Raw output of the model for actions to take.
                                        Shape: (batch_size, action_dim).
        Returns:
            torch.Tensor: Processed actions of the same shape as raw_actions.
        """
        batch_size = raw_actions.shape[0]
        actions = torch.zeros_like(raw_actions)  # Shape: (batch_size, action_dim)

        # Splitting raw actions into W-related and Theta-related components
        # Splitting raw actions into W-related and Theta-related components
        W_raw_actions = raw_actions[:, :2 * self.N_t * self.K]  # Shape: (batch_size, 2 * N_t * K)
        theta_actions = raw_actions[:, 2 * self.N_t * self.K:]  # Shape: (batch_size, 2 * N_r)

        # Process Theta actions
        theta_real = theta_actions[:, 0::2]  # Real parts of Theta diagonal (Shape: (batch_size, M))
        theta_imag = theta_actions[:, 1::2]  # Imaginary parts of Theta diagonal (Shape: (batch_size, M))
        magnitudes = torch.sqrt(theta_real**2 + theta_imag**2)  # Shape: (batch_size, M)

        # Avoid division by zero
        magnitudes = torch.where(magnitudes == 0, 1, magnitudes)
        normalized_theta_real = theta_real / magnitudes  # Shape: (batch_size, M)
        normalized_theta_imag = theta_imag / magnitudes  # Shape: (batch_size, M)

        # Store normalized Theta components in actions
        actions[:, 2 * self.N_t * self.K::2] = normalized_theta_real
        actions[:, 2 * self.N_t * self.K + 1::2] = normalized_theta_imag

        # Process W actions
        W_raw_actions = W_raw_actions.reshape(batch_size, self.K, 2 * self.N_t)  # Shape: (batch_size, K, 2 * N_t)
        W_real = W_raw_actions[:, :, 0::2]  # Real parts of W (Shape: (batch_size, K, N_t))
        W_imag = W_raw_actions[:, :, 1::2]  # Imaginary parts of W (Shape: (batch_size, K, N_t))
        raw_W = W_real + 1j * W_imag  # Convert to complex matrix (Shape: (batch_size, K, N_t))

        # Project W onto the power constraint set
        W = self.actor_W_projection_operator(raw_W)  # Shape: (batch_size, K, N_t)

        # Store processed W components in actions
        flattened_real = W.real.flatten(start_dim=1)  # Shape: (batch_size, K * N_t)
        flattened_imag = W.imag.flatten(start_dim=1)  # Shape: (batch_size, K * N_t)
        actions[:, :2 * self.N_t * self.K:2] = flattened_real
        actions[:, 1:2 * self.N_t * self.K:2] = flattened_imag

        return actions

    def forward(self, x):
        """
        Forward pass to compute policy parameters (mean and log_std).
        """
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        mean = self.mean_output(x) # CHANGED: Removed tanh and batch norm from here. Apply tanh after sampling.
        log_std = self.log_std_param.clamp(-10, 2).expand_as(mean)
        return mean, log_std

    # CHANGED: Renamed to get_distribution
    def get_distribution(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        return Normal(mean, std)
    
    # CHANGED: Renamed to get_action_and_log_prob
    def get_action_and_log_prob(self, state, deterministic=False):
        dist = self.get_distribution(state)
        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample() # Use reparameterization trick

        # Compute log prob, accounting for the tanh squashing
        log_prob = dist.log_prob(raw_action)
        squashed_action = torch.tanh(raw_action)
        # This correction is crucial for stability and correctness of squashed action spaces
        log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Apply domain-specific processing AFTER squashing
        processed_action = self.actor_process_raw_actions(squashed_action)

        return processed_action, log_prob, raw_action

    # CHANGED: Now takes raw_actions as input to correctly evaluate them
    def evaluate_actions(self, states, raw_actions):
        """
        Evaluate log probabilities and entropy for given state-action pairs.
        """
        dist = self.get_distribution(states)
        
        # Compute log_prob of the pre-tanh action
        log_probs = dist.log_prob(raw_actions)
        
        # Correct for the tanh squashing transformation
        squashed_actions = torch.tanh(raw_actions)
        log_probs -= torch.log(1 - squashed_actions.pow(2) + 1e-6)
        
        log_probs = log_probs.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_probs, entropy


class CriticNetwork(nn.Module):
    """
    PPO Critic network.
    """
    def __init__(self, state_dim, critic_linear_layers=[128, 128]):
        super().__init__()
        input_dim = state_dim
        self.linear_layers = nn.ModuleList()
        for layer_dim in critic_linear_layers:
            self.linear_layers.append(nn.Linear(input_dim, layer_dim))
            input_dim = layer_dim
        self.output = nn.Linear(critic_linear_layers[-1], 1)

    def forward(self, state):
        x = state
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        return self.output(x)


class PPO:
    """
    Proximal Policy Optimization (PPO-Clip) implementation.
    """
    def __init__(self, state_dim, action_dim, N_t, K, P_max,
                 n_rollout_envs:int, 
                 actor_model=ActorNetwork, critic_model=CriticNetwork,
                 actor_linear_layers=[128, 128, 128], critic_linear_layers=[128, 128],
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 optimizer="adam",
                 actor_lr=3e-4, critic_lr=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
                 rollout_size=2048, ppo_epochs=10, minibatch_size=64,
                 using_loss_scaling: bool = False,
                 seed=42):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.N_t = N_t
        self.K = K
        self.P_max = P_max
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_size = rollout_size
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.device = device
        self.n_rollout_envs = n_rollout_envs
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actor = actor_model(state_dim, action_dim, N_t, K, P_max, actor_linear_layers).to(self.device)
        self.critic = critic_model(state_dim, critic_linear_layers).to(self.device)

        opt_cls = {"adam": optim.Adam, "adamw": optim.AdamW}[optimizer.lower()]
        self.actor_optimizer = opt_cls(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = opt_cls(self.critic.parameters(), lr=critic_lr)

        self.using_loss_scaling = using_loss_scaling and (self.device.type == 'cuda')
        self.scaler = torch.GradScaler() if self.using_loss_scaling else None

        self.rollout = RolloutBuffer(rollout_size, state_dim, action_dim, self.device, n_rollout_envs)
        self.total_it = 0

    # CHANGED: Now returns raw_action as well
    def select_action(self, state, eval_mode=False):
        """
        Select action using current policy.
        """
        self.actor.eval()
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            action, logprob, raw_action = self.actor.get_action_and_log_prob(state_t, deterministic=eval_mode)
                
        return action.squeeze(0).cpu(), logprob.squeeze(0), raw_action.squeeze(0)

    # CHANGED: Modified to store raw_action and env_id
    def store_transition(self, state, action, raw_action, reward, next_state, done=False, logprob=None, value=None, env_id=0):
        """
        Store transition in rollout buffer.
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        action_t = action if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.float32, device=self.device)
        raw_action_t = raw_action if isinstance(raw_action, torch.Tensor) else torch.tensor(raw_action, dtype=torch.float32, device=self.device)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_t = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        
        if value is None:
            with torch.no_grad():
                value = self.critic(state_t.unsqueeze(0)).squeeze(0)
                
        if logprob is None:
            logprob_t = torch.zeros(1, device=self.device)
        else:
            logprob_t = logprob.to(device=self.device)
        
        if logprob_t.dim() == 0:
            logprob_t = logprob_t.unsqueeze(0)
        if logprob_t.dim() == 1:
            logprob_t = logprob_t.unsqueeze(-1)
            
        self.rollout.add(state_t, action_t, raw_action_t, reward_t, done_t, 
                        value.detach().unsqueeze(-1) if value.dim() == 0 else value.detach(), 
                        logprob_t, env_id)

    def training(self, batch_size=None):
        """
        Perform PPO training update.
        """
        if self.rollout.ptr < self.minibatch_size:
            return 0.0, 0.0, 0.0
            
        self.actor.train()
        self.critic.train()
        self.total_it += 1

        with torch.no_grad():
            # Get last states for each environment to compute last values
            last_states = []
            for env_id in range(self.n_rollout_envs):
                # Find the last state for this environment
                env_mask = (self.rollout.env_ids[:self.rollout.ptr].squeeze() == env_id)
                env_indices = torch.where(env_mask)[0]
                if len(env_indices) > 0:
                    last_state_idx = env_indices[-1]
                    last_states.append(self.rollout.states[last_state_idx])
                else:
                    # If no transitions for this environment, use zeros
                    last_states.append(torch.zeros(self.state_dim, device=self.device))
            
            last_states = torch.stack(last_states)  # Shape: (n_rollout_envs, state_dim)
            last_values = self.critic(last_states)  # Shape: (n_rollout_envs, 1)

        self.rollout.compute_advantages(last_values=last_values, gamma=self.gamma, gae_lambda=self.gae_lambda)
        
        actor_losses, critic_losses = [], []
        mean_reward = self.rollout.rewards.mean().item()

        for _ in range(self.ppo_epochs):
            for (states, actions, raw_actions, old_logprobs, advantages, returns, old_values, env_ids) in self.rollout.get_minibatches(self.minibatch_size):
                
                # CHANGED: Advantage normalization per minibatch for stability
                advs = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # --- Policy (Actor) Loss ---
                # CHANGED: Pass raw_actions to evaluate_actions
                new_logprobs, entropy = self.actor.evaluate_actions(states, raw_actions)
                ratio = torch.exp(new_logprobs - old_logprobs)
                
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advs
                
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                # --- Value (Critic) Loss ---
                values = self.critic(states)
                critic_loss = F.mse_loss(values, returns)
                
                # --- Separate Updates ---
                # Actor update
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic update
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        self.rollout.reset()
        
        return np.mean(actor_losses), np.mean(critic_losses), mean_reward

    def save_models(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), os.path.join(directory, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, "critic.pth"))

    def load_models(self, directory):
        self.actor.load_state_dict(torch.load(os.path.join(directory, "actor.pth"), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(directory, "critic.pth"), map_location=self.device))