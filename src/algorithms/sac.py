import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.amp as amp
from torch.distributions import Normal
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from src.environment.ris_modules import process_raw_actions_torch
import numpy as np
import os 


class RunningMeanStd:
    """Online normalization statistics for vector observations."""

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-8)
        self.count = total_count


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, N_t, K, P_max, actor_linear_layers = [128,128,128],
                 w_action_mapping: str = "projection"):
        self.N_t = N_t
        self.K = K
        self.P_max = P_max
        self.w_action_mapping = w_action_mapping
        super(ActorNetwork, self).__init__()
        input_dim = state_dim
        self.linear_layers = nn.ModuleList()

        for layer_dim in actor_linear_layers:
            self.linear_layers.append(nn.Linear(input_dim, layer_dim))
            input_dim = layer_dim
        
        # For SAC, we need separate outputs for mean and log_std
        self.mean_output = nn.Linear(actor_linear_layers[-1], action_dim)
        self.log_std_output = nn.Linear(actor_linear_layers[-1], action_dim)
        
        # Clamp bounds for log_std
        self.log_std_min = -5
        self.log_std_max = 2

    def forward(self, x):
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        
        # Get mean and log_std (no tanh on mean; tanh is applied to samples)
        mean = self.mean_output(x)
        log_std = self.log_std_output(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state, device=None):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample()
        raw_action = torch.tanh(x_t)
        action = process_raw_actions_torch(
            raw_action,
            self.N_t,
            self.K,
            self.P_max,
            device or raw_action.device,
            w_action_mapping=self.w_action_mapping,
        )

        # Tanh squashing correction for valid SAC entropy term.
        log_prob = normal.log_prob(x_t).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - raw_action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob, raw_action
    

    def get_action(self, state, device=None):
        """Get deterministic action for evaluation"""
        mean, _ = self.forward(state)
        raw_action = torch.tanh(mean)
        action = process_raw_actions_torch(
            raw_action,
            self.N_t,
            self.K,
            self.P_max,
            device or raw_action.device,
            w_action_mapping=self.w_action_mapping,
        )
        return action

    def forward_raw(self, x):
        """Returns raw actions (before constraint processing), aligned with TD3/DDPG."""
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        x = torch.tanh(self.mean_output(x))
        return x



class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, critic_linear_layers = [128,128]):
        super(CriticNetwork, self).__init__()
        self.output = nn.Linear(critic_linear_layers[-1], 1)                   # Output
        input_dim = state_dim + action_dim
        self.linear_layers = nn.ModuleList() # Input and intermediate layers 
        for layer_dim in critic_linear_layers:
            self.linear_layers.append(nn.Linear(input_dim, layer_dim)) # Input and intermediate layers
            input_dim = layer_dim

    def forward(self, state, action):
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        x = self.output(x)  # No activation for output
        return x
    


class SAC:
    """Soft Actor-Critic implementation with optional Prioritized Experience Replay.

    Parameters:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        N_t (int): Number of transmit antennas.
        K (int): Number of users.
        P_max (float): Maximum power constraint.
        actor_model (class, optional): Actor neural network model class. Default is ActorNetwork.
        critic_model (class, optional): Critic neural network model class. Default is CriticNetwork.
        device (torch.device, optional): Device to run the computations on (CPU or GPU).
        actor_lr (float, optional): Learning rate for the actor optimizer. Default is 0.0003.
        critic_lr (float, optional): Learning rate for the critic optimizer. Default is 0.0003.
        alpha_lr (float, optional): Learning rate for the temperature parameter. Default is 0.0003.
        gamma (float, optional): Discount factor for future rewards. Default is 0.99.
        tau (float, optional): Soft update parameter for target networks. Default is 0.005.
        alpha (float, optional): Initial temperature parameter. Default is 0.2.
        automatic_entropy_tuning (bool, optional): Whether to automatically tune entropy. Default is True.
        target_entropy (float, optional): Target entropy for automatic tuning. Default is -action_dim.
        buffer_size (int, optional): Maximum size of the replay buffer. Default is 1000000.
        seed (int, optional): Seed for random number generators. Default is 42.
        actor_frequency_update (int, optional): Frequency of actor updates. Default is 1.
        critic_frequency_update (int, optional): Frequency of critic updates. Default is 1.
        use_per (bool, optional): Whether to use Prioritized Experience Replay. Default is False.
        per_alpha (float, optional): PER prioritization exponent. Default is 0.6.
        per_beta_start (float, optional): PER initial importance sampling weight. Default is 0.4.
        per_beta_frames (int, optional): Frames to anneal PER beta to 1.0. Default is 100000.
        per_epsilon (float, optional): Small constant for PER priorities. Default is 1e-6.
    """
    def __init__(self, state_dim, action_dim, N_t, K, P_max,
                 action_space_type: str = "continuous",
                 actor_model=ActorNetwork, critic_model=CriticNetwork,  # Changed to None for example
                 actor_linear_layers=[128,128,128],
                 critic_linear_layers=[128,128],
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 optimizer = "adam",
                 actor_lr=0.0003, critic_lr=0.0003, alpha_lr=0.0003, gamma=0.99, tau=0.0005,
                 critic_tau=0.0005,
                 alpha=0.2, automatic_entropy_tuning=True, target_entropy=None,
                 buffer_size=1000000, seed=42,
                 actor_frequency_update: int = 1,
                 critic_frequency_update: int = 1,
                 action_noise_scale:float  = 0,
                 using_loss_scaling: bool = False,
                 gradient_clip_norm: float = 5.0,
                 target_q_clip: float = 100.0,
                 reward_clip: float = 10.0,
                 # PER parameters
                 use_per: bool = False,
                 per_alpha: float = 0.6,
                 per_beta_start: float = 0.4,
                 per_beta_frames: int = 100000,
                 per_epsilon: float = 1e-6,
                 obs_norm_enabled: bool = False,
                 obs_norm_clip: float = 5.0,
                 w_action_mapping: str = "projection"):
        
        # Validate action space type
        # NOTE: SAC is designed for continuous action spaces only
        if action_space_type != "continuous":
            raise ValueError(
                f"SAC requires continuous action space, got '{action_space_type}'. "
                f"For discrete action spaces, use DQN instead."
            )
        
        self.action_space_type = action_space_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.N_t = N_t
        self.K = K
        self.P_max = P_max
        self.total_it = 0
        self.device = device
        self.device_string = str(self.device)
        self.actor_frequency_update = actor_frequency_update
        self.critic_frequency_update = critic_frequency_update
        self.using_loss_scaling = using_loss_scaling
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.use_per = use_per
        self.per_epsilon = per_epsilon
        self.gradient_clip_norm = gradient_clip_norm
        self.target_q_clip = target_q_clip
        self.reward_clip = reward_clip
        self.last_training_stats = {}
        self.obs_norm_enabled = obs_norm_enabled
        self.obs_norm_clip = obs_norm_clip
        self.w_action_mapping = w_action_mapping
        self.obs_rms = RunningMeanStd(state_dim) if obs_norm_enabled else None

        if self.device.type == 'cuda':
            self.gpu_used = True
        else:
            self.gpu_used = False

        if self.gpu_used:
            self.tau = torch.tensor(tau, device=self.device)
            self.critic_tau = torch.tensor(critic_tau, device=self.device)
        else:
            self.tau = tau
            self.critic_tau = critic_tau

        self.action_noise_scale = action_noise_scale
        self.scaler = torch.GradScaler() if self.gpu_used else None

        # Independent RNGs
        self.network_numpy_rng = np.random.default_rng(seed)    # for exploration noise
        torch.manual_seed(seed * 2)                             # for torch ops
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed * 2)
            torch.cuda.manual_seed_all(seed * 2)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize networks (assuming ActorNetwork and CriticNetwork exist)
        if actor_model is None:
            raise ValueError("actor_model must be provided")
        if critic_model is None:
            raise ValueError("critic_model must be provided")
            
        self.actor = actor_model(state_dim=state_dim, action_dim=action_dim,
                                 actor_linear_layers=actor_linear_layers,
                                N_t=self.N_t, K=self.K, P_max=self.P_max,
                                w_action_mapping=self.w_action_mapping).to(self.device)

        # Two Q-networks for SAC
        self.critic1 = critic_model(state_dim, action_dim, critic_linear_layers=critic_linear_layers).to(self.device)
        self.critic2 = critic_model(state_dim, action_dim, critic_linear_layers=critic_linear_layers).to(self.device)

        # Target Q-networks
        self.target_critic1 = critic_model(state_dim, action_dim, critic_linear_layers=critic_linear_layers).to(self.device)
        self.target_critic2 = critic_model(state_dim, action_dim, critic_linear_layers=critic_linear_layers).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        OPTIMIZERS = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "rmsprop": optim.RMSprop,
            "sgd": optim.SGD,
            "adagrad": optim.Adagrad,
            "adamax": optim.Adamax,
            "nadam": optim.NAdam,
            # extend if needed
        }
        # ensure case-insensitivity
        optimizer_name = optimizer.lower()
        self.actor_optimizer =  OPTIMIZERS[optimizer_name](self.actor.parameters(), lr=actor_lr )
        self.critic1_optimizer = OPTIMIZERS[optimizer_name](self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = OPTIMIZERS[optimizer_name](self.critic2.parameters(), lr=critic_lr)


        # Temperature parameter
        if automatic_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = OPTIMIZERS[optimizer_name]([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha, device=device)

        # Initialize replay buffer (either classic or prioritized)
        if self.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size=buffer_size,
                state_dim=state_dim,
                action_dim=action_dim,
                numpy_rng=self.network_numpy_rng,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_frames=per_beta_frames,
                epsilon=per_epsilon
            )
            print(f"\n SAC using Prioritized Experience Replay (alpha={per_alpha}, beta_start={per_beta_start}) ")
        else:
            # Assuming ReplayBuffer is your original implementation
            self.replay_buffer = ReplayBuffer(
                buffer_size=buffer_size,
                state_dim=state_dim,
                action_dim=action_dim,
                numpy_rng=self.network_numpy_rng
            )
            print(f"\n SAC using Standard Experience Replay")

    def select_action(self, state, eval_mode=False):
        """
        Selects an action based on the current state using the actor network.
        """
        self.actor.eval()
        
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        state = self._normalize_state_tensor(state, update_stats=False)
        
        if eval_mode:
            # Deterministic action for evaluation
            with torch.no_grad():
                action = self.actor.get_action(state)
        else:
            # Stochastic action for exploration
            with torch.no_grad():
                action, _, _ = self.actor.sample(state)
        
        return action.detach().squeeze(0).to('cpu')

    def select_noised_action(self, state, noise_scale=None):
        """
        Dumb function to avoid compatibility problems with other similar functions. For SAC, we don't need additional noise as entropy provides exploration
        """
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        state = self._normalize_state_tensor(state, update_stats=False)
        with torch.no_grad():
            action, _, _ = self.actor.sample(state)
        action = action.detach().squeeze(0)
        
        noised_action = action
            
        return action.to('cpu'), noised_action.to('cpu')

    def _sample_from_buffer(self, batch_size):
        """
        Samples from the replay buffer, handling both standard and prioritized buffers.
        
        Returns:
            If using standard buffer: (states, actions, rewards, next_states)
            If using PER: (states, actions, rewards, next_states, weights, indices)
        """
        if self.use_per:
            return self.replay_buffer.sample(batch_size)
        else:
            batch = self.replay_buffer.sample(batch_size)
            # Add dummy weights and indices for compatibility
            dummy_weights = torch.ones(batch_size, device=self.device)
            dummy_indices = np.arange(batch_size)
            return (*batch, dummy_weights, dummy_indices)

    def _normalize_state_tensor(self, state_tensor, update_stats=False):
        """Normalize states with running statistics when enabled."""
        if not self.obs_norm_enabled:
            return state_tensor
        if update_stats:
            self.obs_rms.update(state_tensor.detach().cpu().numpy())
        mean = torch.as_tensor(self.obs_rms.mean, device=state_tensor.device, dtype=state_tensor.dtype)
        var = torch.as_tensor(self.obs_rms.var, device=state_tensor.device, dtype=state_tensor.dtype)
        normalized = (state_tensor - mean) / torch.sqrt(var + 1e-8)
        return torch.clamp(normalized, -self.obs_norm_clip, self.obs_norm_clip)

    def _compute_state_diagnostics(self, states, actions):
        """Compute diagnostics to quantify effective state usage."""
        diagnostics = {}
        with torch.no_grad():
            diagnostics["state_std_mean"] = float(states.std(dim=0).mean().detach().cpu())
            diagnostics["state_std_max"] = float(states.std(dim=0).max().detach().cpu())
            centered_state = states - states.mean(dim=0, keepdim=True)
            centered_action = actions - actions.mean(dim=0, keepdim=True)
            covariance = centered_state.T @ centered_action / max(1, states.shape[0] - 1)
            diagnostics["state_action_cov_mean_abs"] = float(covariance.abs().mean().detach().cpu())
            block_slices = self._estimate_state_blocks(states.shape[1])
            for block_name, block_slice in block_slices.items():
                if block_slice.stop > block_slice.start:
                    block_std = states[:, block_slice].std(dim=0).mean()
                    diagnostics[f"state_block_std_{block_name}"] = float(block_std.detach().cpu())

        state_for_grad = states.detach().clone().requires_grad_(True)
        diag_actions, _, _ = self.actor.sample(state_for_grad)
        q_values = self.critic1(state_for_grad, diag_actions)
        grad = torch.autograd.grad(
            outputs=q_values.mean(),
            inputs=state_for_grad,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        diagnostics["dq_ds_norm_mean"] = float(grad.norm(dim=-1).mean().detach().cpu())
        diagnostics["dq_ds_norm_max"] = float(grad.norm(dim=-1).max().detach().cpu())
        return diagnostics

    def _estimate_state_blocks(self, state_dim):
        """Infer standard block boundaries from known dimensional constraints."""
        m_dim = max(0, (self.action_dim - 2 * self.N_t * self.K) // 2)
        fixed_tail = m_dim + self.action_dim + 4 * self.K
        remaining = max(0, state_dim - fixed_tail)
        prev_rate_candidates = [4, 4 * self.K + 3]
        prev_rate_dim = 0
        for candidate in prev_rate_candidates:
            if remaining - candidate >= 0:
                prev_rate_dim = candidate
                break
        channel_dim = max(0, remaining - prev_rate_dim)

        idx = 0
        blocks = {
            "rates": slice(idx, idx + prev_rate_dim),
        }
        idx += prev_rate_dim
        blocks["channels"] = slice(idx, idx + channel_dim)
        idx += channel_dim
        blocks["phases"] = slice(idx, idx + m_dim)
        idx += m_dim
        blocks["prev_action"] = slice(idx, idx + self.action_dim)
        idx += self.action_dim
        blocks["powers"] = slice(idx, min(state_dim, idx + 4 * self.K))
        return blocks

    def _calculate_td_errors(self, states, actions, rewards, next_states, q1_values, q2_values):
        """
        Calculates TD errors for priority updates in PER.
        """
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            if self.target_q_clip is not None:
                target_q = torch.clamp(target_q, -self.target_q_clip, self.target_q_clip)
            clipped_rewards = rewards
            if self.reward_clip is not None:
                clipped_rewards = torch.clamp(clipped_rewards, -self.reward_clip, self.reward_clip)
            target_values = clipped_rewards + self.gamma * target_q
            
            # Calculate TD errors for both critics (use minimum for priority)
            td_error1 = torch.abs(q1_values - target_values)
            td_error2 = torch.abs(q2_values - target_values)
            td_errors = torch.min(td_error1, td_error2)
            
        return td_errors.detach().cpu().numpy().flatten()

    def training(self, batch_size):
        """
        Performs a training step on a batch of experiences sampled from the replay buffer.
        """
        self.actor.train()
        self.total_it += 1
        target_q_mean = float("nan")
        
        if self.gpu_used:
            self.actor.to(self.device)
        
        # Sample from buffer (handles both standard and PER)
        if self.use_per:
            state, actions, rewards, next_state, weights, indices = self._sample_from_buffer(batch_size)
        else:
            state, actions, rewards, next_state, weights, indices = self._sample_from_buffer(batch_size)
        
        if self.gpu_used:
            state, actions, rewards, next_state = (t.to(self.device, non_blocking=True) 
                                                  for t in (state, actions, rewards, next_state))
            if self.use_per:
                weights = weights.to(self.device, non_blocking=True)

        # Observation normalization shared by actor/critics.
        state = self._normalize_state_tensor(state, update_stats=True)
        next_state = self._normalize_state_tensor(next_state, update_stats=True)

        # Update Critics
        if self.total_it % self.critic_frequency_update == 0:
            updated_critic = True
            with torch.no_grad():
                next_actions, next_log_probs, _ = self.actor.sample(next_state)
                target_q1 = self.target_critic1(next_state, next_actions)
                target_q2 = self.target_critic2(next_state, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
                if self.target_q_clip is not None:
                    target_q = torch.clamp(target_q, -self.target_q_clip, self.target_q_clip)
                target_q_mean = float(target_q.mean().detach().cpu())
                # Optional done masking; if not available, assume zeros
                dones = torch.zeros_like(rewards)
                clipped_rewards = rewards
                if self.reward_clip is not None:
                    clipped_rewards = torch.clamp(clipped_rewards, -self.reward_clip, self.reward_clip)
                y = clipped_rewards + self.gamma * (1 - dones) * target_q

            if self.using_loss_scaling and self.scaler:
                with amp.autocast():
                    q1_values = self.critic1(state, actions)
                    q2_values = self.critic2(state, actions)
                    
                    # Apply importance sampling weights if using PER
                    if self.use_per:
                        critic1_loss = (weights * F.mse_loss(q1_values, y, reduction='none')).mean()
                        critic2_loss = (weights * F.mse_loss(q2_values, y, reduction='none')).mean()
                    else:
                        critic1_loss = F.mse_loss(q1_values, y)
                        critic2_loss = F.mse_loss(q2_values, y)

                self.critic1_optimizer.zero_grad()
                self.scaler.scale(critic1_loss).backward()
                self.scaler.unscale_(self.critic1_optimizer)
                torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.critic1_optimizer)

                self.critic2_optimizer.zero_grad()
                self.scaler.scale(critic2_loss).backward()
                self.scaler.unscale_(self.critic2_optimizer)
                torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.critic2_optimizer)
                self.scaler.update()
            else:

                q1_values = self.critic1(state, actions)
                q2_values = self.critic2(state, actions)
                
                # Apply importance sampling weights if using PER
                if self.use_per:
                    critic1_loss = (weights * F.mse_loss(q1_values, y, reduction='none')).mean()
                    critic2_loss = (weights * F.mse_loss(q2_values, y, reduction='none')).mean()
                else:
                    critic1_loss = F.mse_loss(q1_values, y)
                    critic2_loss = F.mse_loss(q2_values, y)

                self.critic1_optimizer.zero_grad()
                critic1_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.gradient_clip_norm)
                self.critic1_optimizer.step()

                self.critic2_optimizer.zero_grad()
                critic2_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.gradient_clip_norm)
                self.critic2_optimizer.step()

            # Update priorities if using PER
            if self.use_per:
                td_errors = self._calculate_td_errors(state, actions, rewards, next_state, q1_values, q2_values)
                new_priorities = td_errors + self.per_epsilon
                self.replay_buffer.update_priorities(indices, new_priorities)

            update_target_critics = True
        else:
            updated_critic = False
            with torch.no_grad():
                q1_values = self.critic1(state, actions)
                q2_values = self.critic2(state, actions)
                # NOTE: When critic is not being updated, we compute a diagnostic loss
                # against rewards only (no Bellman target) for logging purposes.
                critic1_loss = F.mse_loss(q1_values, rewards)
                critic2_loss = F.mse_loss(q2_values, rewards)
            update_target_critics = False

        # Update Actor
        if self.total_it % self.actor_frequency_update == 0:
            updated_actor = True
            new_actions, log_probs, _ = self.actor.sample(state)
            q1_new = self.critic1(state, new_actions)
            q2_new = self.critic2(state, new_actions)
            q_new = torch.min(q1_new, q2_new)
            
            # Apply importance sampling weights if using PER
            if self.use_per:
                actor_loss = (weights * (self.alpha * log_probs - q_new)).mean()
            else:
                actor_loss = (self.alpha * log_probs - q_new).mean()

            if self.using_loss_scaling and self.scaler:
                self.actor_optimizer.zero_grad()
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(self.actor_optimizer)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.actor_optimizer)
            else:
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_norm)
                self.actor_optimizer.step()

            # Update temperature parameter
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()

            self.update_target_networks(update_target_critics)
        else:
            updated_actor = False
            with torch.no_grad():
                new_actions, log_probs, _ = self.actor.sample(state)
                q1_new = self.critic1(state, new_actions)
                q2_new = self.critic2(state, new_actions)
                q_new = torch.min(q1_new, q2_new)
                
                if self.use_per:
                    actor_loss = (weights * (self.alpha * log_probs - q_new)).mean()
                else:
                    actor_loss = (self.alpha * log_probs - q_new).mean()
                
                if self.automatic_entropy_tuning:
                    alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                else:
                    alpha_loss = torch.tensor(0.0)

            self.update_target_networks(update_target_critics=update_target_critics)

        if self.gpu_used:
            actor_loss = actor_loss.to('cpu')
            critic1_loss = critic1_loss.to('cpu')

        alpha_value = float(self.alpha.detach().cpu()) if isinstance(self.alpha, torch.Tensor) else float(self.alpha)
        self.last_training_stats = {
            "alpha": alpha_value,
            "log_prob_mean": float(log_probs.mean().detach().cpu()),
            "q1_mean": float(q1_values.mean().detach().cpu()),
            "q2_mean": float(q2_values.mean().detach().cpu()),
            "q_new_mean": float(q_new.mean().detach().cpu()),
            "target_q_mean": target_q_mean,
            "action_norm_mean": float(new_actions.norm(dim=-1).mean().detach().cpu()),
        }
        if self.total_it % 50 == 0:
            self.last_training_stats.update(self._compute_state_diagnostics(state, actions))

        return float(actor_loss.detach().cpu()), float(critic1_loss.detach().cpu()), rewards, updated_actor, updated_critic

    def get_last_training_stats(self):
        """Returns lightweight diagnostics from the last training step."""
        return dict(self.last_training_stats)

    def update_target_networks(self, update_target_critics=True):
        """
        Updates the target networks using soft updates.
        """
        with torch.no_grad():
            if update_target_critics:
                for target_param, param in zip(self.target_critic1.parameters(), 
                                             self.critic1.parameters()):
                    target_param.data.copy_(self.tau * param.data + 
                                          (1 - self.tau) * target_param.data)
                
                for target_param, param in zip(self.target_critic2.parameters(), 
                                             self.critic2.parameters()):
                    target_param.data.copy_(self.tau * param.data + 
                                          (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, batch_size=None):
        """
        Stores a transition in the replay buffer.
        """
        if self.obs_norm_enabled:
            self.obs_rms.update(np.asarray(state, dtype=np.float32))
            self.obs_rms.update(np.asarray(next_state, dtype=np.float32))
        self.replay_buffer.add(state, action, reward, next_state, batch_size=batch_size)

    def get_buffer_info(self):
        """
        Returns information about the current buffer state.
        """
        info = {
            'buffer_type': 'PER' if self.use_per else 'Standard',
            'buffer_size': self.replay_buffer.size,
            'buffer_capacity': self.replay_buffer.buffer_size,
            'buffer_filled': self.replay_buffer.buffer_filled
        }
        
        if self.use_per:
            info['beta'] = self.replay_buffer._get_beta()
            info['frame'] = self.replay_buffer.frame
            info['max_priority'] = self.replay_buffer.max_priority
            
        return info

    def save_models(self, directory):
        """
        Saves all networks to the specified directory.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Define file paths
        actor_path = os.path.join(directory, "actor.pth")
        critic1_path = os.path.join(directory, "critic1.pth")
        critic2_path = os.path.join(directory, "critic2.pth")
        target_critic1_path = os.path.join(directory, "target_critic1.pth")
        target_critic2_path = os.path.join(directory, "target_critic2.pth")

        # Save models
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)
        torch.save(self.target_critic1.state_dict(), target_critic1_path)
        torch.save(self.target_critic2.state_dict(), target_critic2_path)

        # Save temperature parameter if using automatic tuning
        if self.automatic_entropy_tuning:
            alpha_path = os.path.join(directory, "log_alpha.pth")
            torch.save(self.log_alpha, alpha_path)

        # Save additional info about buffer type
        config_path = os.path.join(directory, "config.pth")
        config = {
            'use_per': self.use_per,
            'buffer_info': self.get_buffer_info()
        }
        torch.save(config, config_path)

    def load_models(self, directory):
        """
        Loads all networks from the specified directory.
        """
        actor_path = os.path.join(directory, "actor.pth")
        critic1_path = os.path.join(directory, "critic1.pth")
        critic2_path = os.path.join(directory, "critic2.pth")
        target_critic1_path = os.path.join(directory, "target_critic1.pth")
        target_critic2_path = os.path.join(directory, "target_critic2.pth")

        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic1.load_state_dict(torch.load(critic1_path, map_location=self.device))
        self.critic2.load_state_dict(torch.load(critic2_path, map_location=self.device))
        self.target_critic1.load_state_dict(torch.load(target_critic1_path, map_location=self.device))
        self.target_critic2.load_state_dict(torch.load(target_critic2_path, map_location=self.device))

        if self.automatic_entropy_tuning:
            alpha_path = os.path.join(directory, "log_alpha.pth")
            if os.path.exists(alpha_path):
                self.log_alpha = torch.load(alpha_path, map_location=self.device)
                self.alpha = self.log_alpha.exp()

        # Load config if available
        config_path = os.path.join(directory, "config.pth")
        if os.path.exists(config_path):
            config = torch.load(config_path, map_location=self.device)
            print(f"Loaded model with buffer type: {config.get('buffer_info', {}).get('buffer_type', 'Unknown')}")