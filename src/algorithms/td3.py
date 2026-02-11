import torch
import torch.nn as nn
import torch.amp as amp
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from src.environment.ris_modules import process_raw_actions_torch
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
        # Input & intermediate layers
        input_dim = state_dim
        self.linear_layers = nn.ModuleList()
        for layer_dim in actor_linear_layers:
            self.linear_layers.append(nn.Linear(input_dim, layer_dim))
            input_dim = layer_dim

        # Output layer
        self.output = nn.Linear(actor_linear_layers[-1], action_dim)


    def forward(self, x):
        """Forward pass: linear layers + tanh. Returns raw actions (constraints applied externally)."""
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        x = F.tanh(self.output(x))
        return x

    def forward_raw(self, x):
        """Alias for forward: returns raw actions (before constraint processing)."""
        return self.forward(x)



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
    

class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) implementation with optional Prioritized Experience Replay.
    
    NOTE: TD3 is designed for continuous action spaces. For discrete action spaces, use DQN instead.
    This implementation includes a check to ensure it's only used with continuous action spaces.

    Parameters:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        N_t (int): Number of transmit antennas.
        K (int): Number of users.
        P_max (float): Maximum power constraint at the Base Station.
        action_space_type (str): Type of action space - must be "continuous" for TD3.
        actor_model (class, optional): Actor neural network model class. Default is ActorNetwork.
        critic_model (class, optional): Critic neural network model class. Default is CriticNetwork.
        device (torch.device, optional): Device to run the computations on (CPU or GPU).
        actor_lr (float, optional): Learning rate for the actor optimizer. Default is 0.0001.
        critic_lr (float, optional): Learning rate for the critic optimizer. Default is 0.0005.
        gamma (float, optional): Discount factor for future rewards. Default is 0.99.
        tau (float, optional): Soft update parameter for target networks. Default is 0.01.
        critic_tau (float, optional): Soft update parameter for the critic target network. Default is 0.01.
        buffer_size (int, optional): Maximum size of the replay buffer. Default is 10000.
        seed (int, optional): Seed for random number generators. Default is 42.
        actor_frequency_update (int, optional): Frequency of actor updates. Default is 1.
        critic_frequency_update (int, optional): Frequency of critic updates. Default is 1.
        using_loss_scaling (bool, optional): Whether to use mixed precision training. Default is False.
        use_per (bool, optional): Whether to use Prioritized Experience Replay. Default is False.
        per_alpha (float, optional): PER prioritization exponent. Default is 0.6.
        per_beta_start (float, optional): PER initial importance sampling weight. Default is 0.4.
        per_beta_frames (int, optional): Frames to anneal PER beta to 1.0. Default is 100000.
        per_epsilon (float, optional): Small constant for PER priorities. Default is 1e-6.
    """
    def __init__(self, state_dim, action_dim, N_t, K, P_max,
                 action_space_type: str = "continuous",
                 actor_model=ActorNetwork, critic_model=CriticNetwork,
                 actor_linear_layers=[128,128,128],
                 critic_linear_layers=[128,128],
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 optimizer = "adam",
                 actor_lr=0.0001, critic_lr=0.0005, gamma=0.99, tau=0.01, critic_tau=0.01,
                 buffer_size=100000, seed=42,
                 actor_frequency_update: int = 1,
                 critic_frequency_update: int = 1,
                 action_noise_scale:float  = 0,
                 using_loss_scaling: bool = False,
                 # PER parameters
                 use_per: bool = False,
                 per_alpha: float = 0.6,
                 per_beta_start: float = 0.4,
                 per_beta_frames: int = 100000,
                 per_epsilon: float = 1e-6,
                 obs_norm_enabled: bool = False,
                 obs_norm_clip: float = 5.0,
                 target_policy_noise: float = 0.2,
                 target_noise_clip: float = 0.5,
                 w_action_mapping: str = "projection"):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.N_t = N_t
        self.K = K
        self.P_max = P_max
        self.total_it = 0
        self.device = device
        self.device_string = str(self.device)
        self.actor_frequency_update = actor_frequency_update
        self.critic_frequency_update = critic_frequency_update
        self.using_loss_scaling = using_loss_scaling
        self.use_per = use_per
        self.per_epsilon = per_epsilon
        self.obs_norm_enabled = obs_norm_enabled
        self.obs_norm_clip = obs_norm_clip
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.w_action_mapping = w_action_mapping
        self.obs_rms = RunningMeanStd(state_dim) if obs_norm_enabled else None
        self.last_training_stats = {}

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

        self.scaler = torch.GradScaler(self.device)

        self.network_numpy_rng = np.random.default_rng(seed)
        torch.manual_seed(seed * 2)

        # Initialize networks (assuming ActorNetwork and CriticNetwork exist)
        if actor_model is None:
            raise ValueError("actor_model must be provided")
        if critic_model is None:
            raise ValueError("critic_model must be provided")
        
        # Initialize actor networks
        self.actor = actor_model(state_dim=state_dim, action_dim=action_dim,
                                actor_linear_layers=actor_linear_layers,
                                N_t=self.N_t, K=self.K, P_max=self.P_max,
                                w_action_mapping=self.w_action_mapping).to(self.device)
        self.target_actor = actor_model(state_dim=state_dim, action_dim=action_dim,
                                       actor_linear_layers=actor_linear_layers,
                                       N_t=self.N_t, K=self.K, P_max=self.P_max,
                                       w_action_mapping=self.w_action_mapping).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Initialize critic networks
        self.critic_loss_max_clamp = P_max * 3
        self.critic_1 = critic_model(state_dim, action_dim, critic_linear_layers=critic_linear_layers).to(self.device)
        self.critic_2 = critic_model(state_dim, action_dim, critic_linear_layers=critic_linear_layers).to(self.device)
        
        # Target critics
        self.target_critic_1 = critic_model(state_dim, action_dim, critic_linear_layers=critic_linear_layers).to(self.device)
        self.target_critic_2 = critic_model(state_dim, action_dim, critic_linear_layers=critic_linear_layers).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # mapping from argument string → optimizer class
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
        self.actor_optimizer = OPTIMIZERS[optimizer_name](self.actor.parameters(), lr=actor_lr, maximize=True) # we want to maximize the state-value function (the actor)
        # Common optimizer for the critics
        self.q_optimizer = OPTIMIZERS[optimizer_name](list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=critic_lr)

        # Initialize replay buffer (either classic or prioritized)
        if self.use_per:
            # Assuming PrioritizedReplayBuffer is available
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
            print(f" TD3 using Prioritized Experience Replay (alpha={per_alpha}, beta_start={per_beta_start})")
        else:
            self.replay_buffer = ReplayBuffer(
                buffer_size=buffer_size,
                state_dim=state_dim,
                action_dim=action_dim,
                numpy_rng=self.network_numpy_rng
            )
            print(" TD3 using Standard Experience Replay")

    def select_action(self, state):
        """
        Selects an action based on the current state using the actor network.
        """
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        state = self._normalize_state_tensor(state, update_stats=False)

        with torch.no_grad():
            raw_action = self.actor(state)
            action = process_raw_actions_torch(
                raw_action, self.N_t, self.K, self.P_max, self.device,
                w_action_mapping=self.w_action_mapping
            ).squeeze(0).cpu()

        return action


    def select_noised_action(self, state, noise_scale=0.01):
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        state = self._normalize_state_tensor(state, update_stats=False)

        with torch.no_grad():
            raw_action = self.actor.forward_raw(state).squeeze(0)

        # Generate noise using your seeded numpy RNG
        noise_np = self.network_numpy_rng.normal(loc=0.0, scale=self.action_noise_scale, size=raw_action.shape)
        noise = torch.tensor(noise_np, dtype=torch.float32, device=self.device)

        raw_action_noised = raw_action + noise

        # Process actions via environment module
        with torch.no_grad():
            noised_action = process_raw_actions_torch(
                raw_action_noised.unsqueeze(0), self.N_t, self.K, self.P_max, self.device,
                w_action_mapping=self.w_action_mapping
            ).squeeze(0)
            clean_action = process_raw_actions_torch(
                raw_action.unsqueeze(0), self.N_t, self.K, self.P_max, self.device,
                w_action_mapping=self.w_action_mapping
            ).squeeze(0)

        # Return to CPU for rollout
        return clean_action.cpu(), noised_action.cpu()

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
        if not self.obs_norm_enabled:
            return state_tensor
        if update_stats:
            self.obs_rms.update(state_tensor.detach().cpu().numpy())
        mean = torch.as_tensor(self.obs_rms.mean, device=state_tensor.device, dtype=state_tensor.dtype)
        var = torch.as_tensor(self.obs_rms.var, device=state_tensor.device, dtype=state_tensor.dtype)
        normalized = (state_tensor - mean) / torch.sqrt(var + 1e-8)
        return torch.clamp(normalized, -self.obs_norm_clip, self.obs_norm_clip)

    def _compute_state_diagnostics(self, states, actions):
        diagnostics = {}
        with torch.no_grad():
            diagnostics["state_std_mean"] = float(states.std(dim=0).mean().detach().cpu())
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
        raw_actions = self.actor(state_for_grad)
        actor_actions = process_raw_actions_torch(
            raw_actions, self.N_t, self.K, self.P_max, self.device,
            w_action_mapping=self.w_action_mapping
        )
        q_values = self.critic_1(state_for_grad, actor_actions)
        grad = torch.autograd.grad(
            outputs=q_values.mean(),
            inputs=state_for_grad,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        diagnostics["dq_ds_norm_mean"] = float(grad.norm(dim=-1).mean().detach().cpu())
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
            target_actions = process_raw_actions_torch(
                self.target_actor(next_states), self.N_t, self.K, self.P_max, self.device,
                w_action_mapping=self.w_action_mapping
            )
            target_q1 = self.target_critic_1(next_states, target_actions)
            target_q2 = self.target_critic_2(next_states, target_actions)
            target_q_values = torch.min(target_q1, target_q2)
            target_values = rewards + self.gamma * target_q_values
            
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

        state = self._normalize_state_tensor(state, update_stats=True)
        next_state = self._normalize_state_tensor(next_state, update_stats=True)

        # Compute target values
        with torch.no_grad():
            target_raw_actions = self.target_actor(next_state)
            policy_noise = torch.randn_like(target_raw_actions) * self.target_policy_noise
            policy_noise = torch.clamp(policy_noise, -self.target_noise_clip, self.target_noise_clip)
            target_raw_actions = torch.clamp(target_raw_actions + policy_noise, -1.0, 1.0)
            target_actions = process_raw_actions_torch(
                target_raw_actions, self.N_t, self.K, self.P_max, self.device,
                w_action_mapping=self.w_action_mapping
            )
            target_q1 = self.target_critic_1(next_state, target_actions)
            target_q2 = self.target_critic_2(next_state, target_actions)
            target_q_values = torch.min(target_q1, target_q2)
            y = rewards + self.gamma * target_q_values

        # Update Critics
        if self.using_loss_scaling:
            if self.total_it % self.critic_frequency_update == 0:
                updated_critic = True
                with amp.autocast(self.device_string):
                    q1_values = self.critic_1(state, actions)
                    q2_values = self.critic_2(state, actions)
                    
                    # Apply importance sampling weights if using PER
                    if self.use_per:
                        critic_loss = (weights * F.mse_loss(q1_values, y, reduction='none')).mean()
                        critic_loss_2 = (weights * F.mse_loss(q2_values, y, reduction='none')).mean()
                    else:
                        critic_loss = F.mse_loss(q1_values, y)
                        critic_loss_2 = F.mse_loss(q2_values, y)

                    common_critic_loss = critic_loss + critic_loss_2

                # Optimize Critics
                self.q_optimizer.zero_grad()
                self.scaler.scale(common_critic_loss).backward()
                self.scaler.step(self.q_optimizer)
                self.scaler.update()

                # Update priorities if using PER
                if self.use_per:
                    td_errors = self._calculate_td_errors(state, actions, rewards, next_state, q1_values, q2_values)
                    new_priorities = td_errors + self.per_epsilon
                    self.replay_buffer.update_priorities(indices, new_priorities)

                update_target_critic = True
            else:
                updated_critic = False
                with torch.no_grad():
                    with amp.autocast(self.device_string):
                        q1_values = self.critic_1(state, actions)
                        q2_values = self.critic_2(state, actions)
                        
                        if self.use_per:
                            critic_loss = (weights * F.mse_loss(q1_values, y, reduction='none')).mean()
                            critic_loss_2 = (weights * F.mse_loss(q2_values, y, reduction='none')).mean()
                        else:
                            critic_loss = F.mse_loss(q1_values, y)
                            critic_loss_2 = F.mse_loss(q2_values, y)
                update_target_critic = False

            # Update Actor
            if self.total_it % self.actor_frequency_update == 0:
                updated_actor = True
                with amp.autocast(self.device_string):
                    actor_actions = process_raw_actions_torch(
                        self.actor(state), self.N_t, self.K, self.P_max, self.device,
                        w_action_mapping=self.w_action_mapping
                    )
                    actor_q_values = self.critic_1(state, actor_actions)
                    
                    # Apply importance sampling weights if using PER
                    if self.use_per:
                        actor_loss = (weights * actor_q_values).mean()
                    else:
                        actor_loss = actor_q_values.mean()

                self.actor_optimizer.zero_grad()
                self.scaler.scale(actor_loss).backward()
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
                self.update_target_networks(update_target_critic=update_target_critic)
            else:
                updated_actor = False
                with torch.no_grad():
                    with amp.autocast(self.device_string):
                        actor_actions = process_raw_actions_torch(
                            self.actor(state), self.N_t, self.K, self.P_max, self.device,
                            w_action_mapping=self.w_action_mapping
                        )
                        actor_q_values = self.critic_1(state, actor_actions)
                        
                        if self.use_per:
                            actor_loss = (weights * actor_q_values).mean()
                        else:
                            actor_loss = actor_q_values.mean()
                self.update_target_networks(update_target_actor=False, update_target_critic=update_target_critic)

        else:
            # Without loss scaling
            if self.total_it % self.critic_frequency_update == 0:
                updated_critic = True
                q1_values = self.critic_1(state, actions)
                q2_values = self.critic_2(state, actions)

                # Apply importance sampling weights if using PER
                if self.use_per:
                    critic_loss = (weights * F.mse_loss(q1_values, y, reduction='none')).mean()
                    critic_loss_2 = (weights * F.mse_loss(q2_values, y, reduction='none')).mean()
                else:
                    critic_loss = F.mse_loss(q1_values, y)
                    critic_loss_2 = F.mse_loss(q2_values, y)

                common_critic_loss = critic_loss + critic_loss_2

                # Optimize Critics
                self.q_optimizer.zero_grad()
                common_critic_loss.backward()
                self.q_optimizer.step()

                # Update priorities if using PER
                if self.use_per:
                    td_errors = self._calculate_td_errors(state, actions, rewards, next_state, q1_values, q2_values)
                    new_priorities = td_errors + self.per_epsilon
                    self.replay_buffer.update_priorities(indices, new_priorities)

                update_target_critic = True
            else:
                updated_critic = False
                with torch.no_grad():
                    q1_values = self.critic_1(state, actions)
                    q2_values = self.critic_2(state, actions)
                    
                    if self.use_per:
                        critic_loss = (weights * F.mse_loss(q1_values, y, reduction='none')).mean()
                        critic_loss_2 = (weights * F.mse_loss(q2_values, y, reduction='none')).mean()
                    else:
                        critic_loss = F.mse_loss(q1_values, y)
                        critic_loss_2 = F.mse_loss(q2_values, y)
                        
                update_target_critic = False

            # Update Actor
            if self.total_it % self.actor_frequency_update == 0:
                updated_actor = True
                actor_actions = process_raw_actions_torch(
                    self.actor(state), self.N_t, self.K, self.P_max, self.device,
                    w_action_mapping=self.w_action_mapping
                )
                actor_q_values = self.critic_1(state, actor_actions)
                
                # Apply importance sampling weights if using PER
                if self.use_per:
                    actor_loss = (weights * actor_q_values).mean()
                else:
                    actor_loss = actor_q_values.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.update_target_networks(update_target_critic=update_target_critic)
            else:
                updated_actor = False
                with torch.no_grad():
                    actor_actions = process_raw_actions_torch(
                        self.actor(state), self.N_t, self.K, self.P_max, self.device,
                        w_action_mapping=self.w_action_mapping
                    )
                    actor_q_values = self.critic_1(state, actor_actions)
                    
                    if self.use_per:
                        actor_loss = (weights * actor_q_values).mean()
                    else:
                        actor_loss = actor_q_values.mean()
                self.update_target_networks(update_target_actor=False, update_target_critic=update_target_critic)

        if self.gpu_used:
            actor_loss = actor_loss.to('cpu')
            critic_loss = critic_loss.to('cpu')

        self.last_training_stats = {
            "q1_mean": float(q1_values.mean().detach().cpu()),
            "q2_mean": float(q2_values.mean().detach().cpu()),
            "target_q_mean": float(target_q_values.mean().detach().cpu()),
            "actor_q_mean": float(actor_q_values.mean().detach().cpu()),
            "policy_noise_abs_mean": float(policy_noise.abs().mean().detach().cpu()),
        }
        if self.total_it % 50 == 0:
            self.last_training_stats.update(self._compute_state_diagnostics(state, actions))

        return actor_loss, critic_loss, rewards, updated_actor, updated_critic

    def get_last_training_stats(self):
        """Returns lightweight diagnostics from the last training step."""
        return dict(self.last_training_stats)

    def update_target_networks(self, update_target_actor=True, update_target_critic=True):
        """
        Updates the target networks using soft updates.
        """
        with torch.no_grad():
            if update_target_actor:
                for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            if update_target_critic:
                for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                    target_param.data.copy_(self.critic_tau * param.data + (1 - self.critic_tau) * target_param.data)
                for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                    target_param.data.copy_(self.critic_tau * param.data + (1 - self.critic_tau) * target_param.data)

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
        Saves the actor, critic, and target networks to the specified directory.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Define the file paths
        actor_path = os.path.join(directory, "actor.pth")
        critic_1_path = os.path.join(directory, "critic_1.pth")
        critic_2_path = os.path.join(directory, "critic_2.pth")
        target_actor_path = os.path.join(directory, "target_actor.pth")
        target_critic_1_path = os.path.join(directory, "target_critic_1.pth")
        target_critic_2_path = os.path.join(directory, "target_critic_2.pth")
        
        # Save the models
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.target_actor.state_dict(), target_actor_path)
        torch.save(self.critic_1.state_dict(), critic_1_path)
        torch.save(self.target_critic_1.state_dict(), target_critic_1_path)
        torch.save(self.critic_2.state_dict(), critic_2_path)
        torch.save(self.target_critic_2.state_dict(), target_critic_2_path)

        # Save additional info about buffer type
        config_path = os.path.join(directory, "config.pth")
        config = {
            'use_per': self.use_per,
            'buffer_info': self.get_buffer_info()
        }
        torch.save(config, config_path)

    def load_models(self, directory):
        """
        Loads the actor, critic, and target networks from the specified directory.
        """
        actor_path = os.path.join(directory, "actor.pth")
        critic_1_path = os.path.join(directory, "critic_1.pth")
        critic_2_path = os.path.join(directory, "critic_2.pth")
        target_actor_path = os.path.join(directory, "target_actor.pth")
        target_critic_1_path = os.path.join(directory, "target_critic_1.pth")
        target_critic_2_path = os.path.join(directory, "target_critic_2.pth")

        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.target_actor.load_state_dict(torch.load(target_actor_path, map_location=self.device))
        self.critic_1.load_state_dict(torch.load(critic_1_path, map_location=self.device))
        self.target_critic_1.load_state_dict(torch.load(target_critic_1_path, map_location=self.device))
        self.critic_2.load_state_dict(torch.load(critic_2_path, map_location=self.device))
        self.target_critic_2.load_state_dict(torch.load(target_critic_2_path, map_location=self.device))

        # Load config if available
        config_path = os.path.join(directory, "config.pth")
        if os.path.exists(config_path):
            config = torch.load(config_path, map_location=self.device)
            print(f"Loaded TD3 model with buffer type: {config.get('buffer_info', {}).get('buffer_type', 'Unknown')}")