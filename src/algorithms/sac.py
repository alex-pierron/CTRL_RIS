import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.amp as amp
from torch.distributions import Normal
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import numpy as np
import os 

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, N_t, K, P_max, actor_linear_layers = [128,128,128]):
        self.N_t = N_t
        self.K = K
        self.P_max = P_max
        self.tensored_P_max = torch.tensor(P_max)
        self.numpied_P_max = self.tensored_P_max.detach().numpy()
        super(ActorNetwork, self).__init__()
        input_dim = state_dim
        self.linear_layers = nn.ModuleList()

        for layer_dim in actor_linear_layers:
            self.linear_layers.append(nn.Linear(input_dim, layer_dim))
            input_dim = layer_dim
        self.output = nn.Linear(actor_linear_layers[-1], action_dim)
        
        # For SAC, we need separate outputs for mean and log_std
        self.mean_output = nn.Linear(actor_linear_layers[-1], action_dim)
        self.log_std_output = nn.Linear(actor_linear_layers[-1], action_dim)
        
        # Clamp bounds for log_std
        self.log_std_min = -10
        self.log_std_max = 2
        
        self.batch_norm = nn.BatchNorm1d(action_dim)

    def actor_W_projection_operator(self, raw_W):
        """
        Projects a batch of beamforming matrices W onto the set defined by the power constraint.
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
        """
        batch_size = raw_actions.shape[0]
        actions = torch.zeros_like(raw_actions)

        # Splitting raw actions into W-related and Theta-related components
        W_raw_actions = raw_actions[:, :2 * self.N_t * self.K]
        theta_actions = raw_actions[:, 2 * self.N_t * self.K:]

        # Process Theta actions
        theta_real = theta_actions[:, 0::2]
        theta_imag = theta_actions[:, 1::2]
        magnitudes = torch.sqrt(theta_real**2 + theta_imag**2)

        # Avoid division by zero
        magnitudes = torch.where(magnitudes == 0, 1, magnitudes)
        normalized_theta_real = theta_real / magnitudes
        normalized_theta_imag = theta_imag / magnitudes

        # Store normalized Theta components in actions
        actions[:, 2 * self.N_t * self.K::2] = normalized_theta_real
        actions[:, 2 * self.N_t * self.K + 1::2] = normalized_theta_imag

        # Process W actions
        W_raw_actions = W_raw_actions.reshape(batch_size, self.K, 2 * self.N_t)
        W_real = W_raw_actions[:, :, 0::2]
        W_imag = W_raw_actions[:, :, 1::2]
        raw_W = W_real + 1j * W_imag

        # Project W onto the power constraint set
        W = self.actor_W_projection_operator(raw_W)

        # Store processed W components in actions
        flattened_real = W.real.flatten(start_dim=1)
        flattened_imag = W.imag.flatten(start_dim=1)
        actions[:, :2 * self.N_t * self.K:2] = flattened_real
        actions[:, 1:2 * self.N_t * self.K:2] = flattened_imag

        return actions

    def forward(self, x):
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        
        # Get mean and log_std
        mean = torch.tanh(self.mean_output(x))
        log_std = self.log_std_output(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def sample(self, state):
        """Sample action with reparameterization trick"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Create normal distribution and sample
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Apply tanh and process actions
        raw_action = torch.tanh(x_t)
        raw_action = self.batch_norm(raw_action)
        action = self.actor_process_raw_actions(raw_action)
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t).sum(dim=-1, keepdim=True)
        # Adjust for tanh transformation
        log_prob -= torch.log(1 - torch.tanh(x_t).pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        return action, log_prob, raw_action

    def get_action(self, state):
        """Get deterministic action for evaluation"""
        mean, _ = self.forward(state)
        mean = self.batch_norm(mean)
        action = self.actor_process_raw_actions(mean)
        return action

    def forward_raw(self, x):
        """
        Forward pass without applying actor_process_raw_actions.
        Returns raw actions (before constraints), aligned with TD3/DDPG.
        """
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        x = torch.tanh(self.mean_output(x))
        x = self.batch_norm(x)
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
                 # PER parameters
                 use_per: bool = False,
                 per_alpha: float = 0.6,
                 per_beta_start: float = 0.4,
                 per_beta_frames: int = 100000,
                 per_epsilon: float = 1e-6):
        
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
                                N_t=self.N_t, K=self.K, P_max=self.P_max)

        # Two Q-networks for SAC
        self.critic1 = critic_model(state_dim, action_dim, critic_linear_layers=critic_linear_layers).to(self.device)
        self.critic2 = critic_model(state_dim, action_dim, critic_linear_layers=critic_linear_layers).to(self.device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

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
        self.actor_optimizer =  OPTIMIZERS[optimizer_name](self.actor.parameters(), lr=actor_lr )#, maximize = True)
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
            print(f"Using Prioritized Experience Replay (alpha={per_alpha}, beta_start={per_beta_start}) \n")
        else:
            # Assuming ReplayBuffer is your original implementation
            self.replay_buffer = ReplayBuffer(
                buffer_size=buffer_size,
                state_dim=state_dim,
                action_dim=action_dim,
                numpy_rng=self.network_numpy_rng
            )
            print("Using standard Experience Replay")

    def select_action(self, state, eval_mode=False):
        """
        Selects an action based on the current state using the actor network.
        """
        if self.gpu_used:
            self.actor.to("cpu")
        self.actor.eval()
        
        state = torch.tensor(state, dtype=torch.float32, device='cpu').unsqueeze(0)
        
        if eval_mode:
            # Deterministic action for evaluation
            action = self.actor.get_action(state)
        else:
            # Stochastic action for exploration
            action, _, _ = self.actor.sample(state)
        
        return action.detach().squeeze(0)

    def select_noised_action(self, state, noise_scale=None):
        """
        Dumb function to avoid compatibility problems with other similar functions. For SAC, we don't need additional noise as entropy provides exploration
        """
        if self.gpu_used:
            self.actor.to("cpu")
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float32, device='cpu').unsqueeze(0)
        action, _, _ = self.actor.sample(state)
        action = action.detach().squeeze(0)
        
        noised_action = action
            
        return action, noised_action

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

    def _calculate_td_errors(self, states, actions, rewards, next_states, q1_values, q2_values):
        """
        Calculates TD errors for priority updates in PER.
        """
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_values = rewards + self.gamma * target_q
            
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

        # Update Critics
        if self.total_it % self.critic_frequency_update == 0:
            updated_critic = True
            with torch.no_grad():
                next_actions, next_log_probs, _ = self.actor.sample(next_state)
                target_q1 = self.target_critic1(next_state, next_actions)
                target_q2 = self.target_critic2(next_state, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
                y = rewards + self.gamma * target_q

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
                self.scaler.step(self.critic1_optimizer)

                self.critic2_optimizer.zero_grad()
                self.scaler.scale(critic2_loss).backward()
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
                self.critic1_optimizer.step()

                self.critic2_optimizer.zero_grad()
                critic2_loss.backward()
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
                critic1_loss = F.mse_loss(q1_values, y if 'y' in locals() else rewards)
                critic2_loss = F.mse_loss(q2_values, y if 'y' in locals() else rewards)
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
                self.scaler.step(self.actor_optimizer)
            else:
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
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

        return (actor_loss, critic1_loss, rewards, updated_actor, updated_critic)

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