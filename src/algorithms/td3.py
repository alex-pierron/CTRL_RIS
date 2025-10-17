import torch
import torch.nn as nn
import torch.amp as amp
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import os 


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, N_t, K, P_max, actor_linear_layers = [128,128,128]):
        self.N_t = N_t
        self.K = K
        self.P_max = P_max
        self.tensored_P_max = torch.tensor(P_max)
        self.numpied_P_max = self.tensored_P_max.detach().numpy()
        # self.executing_on_gpu = False
        super(ActorNetwork, self).__init__()
        # Input & intermediate layers
        input_dim = state_dim
        self.linear_layers = nn.ModuleList()
        for layer_dim in actor_linear_layers:
            self.linear_layers.append(nn.Linear(input_dim, layer_dim))
            input_dim = layer_dim

        # Output layer
        self.output = nn.Linear(actor_linear_layers[-1], action_dim)
        self.batch_norm = nn.BatchNorm1d(action_dim)


    def actor_W_projection_operator(self, raw_W):
        """
        Projects a batch of beamforming matrices W onto the set defined by the power constraint.

        Args:
            raw_W (ndarray): Batch of raw beamforming matrices of shape (batch_size, rows, columns).
        Returns:
            ndarray: Batch of projected beamforming matrices of the same shape as raw_W.
        """

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
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        x = F.tanh(self.output(x))  # tanh activation for output
        x = self.batch_norm(x)
        x = self.actor_process_raw_actions(x)
        return x

    def forward_raw(self, x):
        """
        Forward pass without applying actor_process_raw_actions.
        Returns raw actions (before constraints).
        """
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        x = torch.tanh(self.output(x))  # raw tanh output
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
    

class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) implementation with optional Prioritized Experience Replay.

    Parameters:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        N_t (int): Number of transmit antennas.
        K (int): Number of users.
        P_max (float): Maximum power constraint at the Base Station.
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
                 per_epsilon: float = 1e-6):
        
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
                                N_t=self.N_t, K=self.K, P_max=self.P_max).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, maximize=True)
        self.target_actor = actor_model(state_dim=state_dim, action_dim=action_dim,
                                       actor_linear_layers=actor_linear_layers,
                                       N_t=self.N_t, K=self.K, P_max=self.P_max).to(self.device)
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

        with torch.no_grad():
            action = self.actor(state).squeeze(0).cpu()  # always return CPU action for rollout

        return action


    def select_noised_action(self, state, noise_scale=0.01):
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            raw_action = self.actor.forward_raw(state).squeeze(0)

        # Generate noise using your seeded numpy RNG
        noise_np = self.network_numpy_rng.normal(loc=0.0, scale=self.action_noise_scale, size=raw_action.shape)
        noise = torch.tensor(noise_np, dtype=torch.float32, device=self.device)

        raw_action_noised = raw_action + noise

        # Process actions on GPU
        with torch.no_grad():
            noised_action = self.actor.actor_process_raw_actions(raw_action_noised.unsqueeze(0)).squeeze(0)
            clean_action = self.actor.actor_process_raw_actions(raw_action.unsqueeze(0)).squeeze(0)

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

    def _calculate_td_errors(self, states, actions, rewards, next_states, q1_values, q2_values):
        """
        Calculates TD errors for priority updates in PER.
        """
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
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

        # Compute target values
        with torch.no_grad():
            target_actions = self.target_actor(next_state)
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
                    actor_actions = self.actor(state)
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
                        actor_actions = self.actor(state)
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
                actor_actions = self.actor(state)
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
                    actor_actions = self.actor(state)
                    actor_q_values = self.critic_1(state, actor_actions)
                    
                    if self.use_per:
                        actor_loss = (weights * actor_q_values).mean()
                    else:
                        actor_loss = actor_q_values.mean()
                self.update_target_networks(update_target_actor=False, update_target_critic=update_target_critic)

        if self.gpu_used:
            actor_loss = actor_loss.to('cpu')
            critic_loss = critic_loss.to('cpu')

        return actor_loss, critic_loss, rewards, updated_actor, updated_critic

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
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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