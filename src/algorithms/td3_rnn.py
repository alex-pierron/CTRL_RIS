"""
TD3 algorithm with RNN support (LSTM/GRU).

This module provides TD3 implementation with RNN-based actor and critic networks
for sequential processing using LSTM or GRU architectures.
"""

import torch
import torch.nn as nn
import torch.amp as amp
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, SequenceReplayBuffer, SequencePrioritizedReplayBuffer
from .rnn_networks import RNNActorNetwork, RNNCriticNetwork
import os


class TD3_RNN:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) implementation with RNN support.
    
    This class uses RNN-based actor and critic networks for sequential processing.
    Supports both LSTM and GRU architectures.
    """
    
    def __init__(self, state_dim, action_dim, N_t, K, P_max,
                 actor_linear_layers=[128, 128, 128],
                 critic_linear_layers=[128, 128],
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 optimizer="adam",
                 actor_lr=0.0001, critic_lr=0.0005, gamma=0.99, tau=0.01, critic_tau=0.01,
                 buffer_size=100000, seed=42,
                 actor_frequency_update: int = 1,
                 critic_frequency_update: int = 1,
                 action_noise_scale: float = 0,
                 using_loss_scaling: bool = False,
                 # PER parameters
                 use_per: bool = True,
                 per_alpha: float = 0.6,
                 per_beta_start: float = 0.4,
                 per_beta_frames: int = 100000,
                 per_epsilon: float = 1e-6,
                 # RNN parameters
                 rnn_type: str = 'lstm',
                 rnn_hidden_size: int = 128,
                 rnn_num_layers: int = 1,
                 sequence_length: int = 1):
        
        # Store basic parameters
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
        
        # RNN parameters
        self.rnn_type = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.sequence_length = sequence_length

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

        # Initialize RNN-based networks
        self.actor = RNNActorNetwork(
            state_dim=state_dim, action_dim=action_dim, N_t=N_t, K=K, P_max=P_max,
            rnn_type=rnn_type, rnn_hidden_size=rnn_hidden_size, rnn_num_layers=rnn_num_layers,
            actor_linear_layers=actor_linear_layers, sequence_length=sequence_length
        ).to(self.device)
        
        self.critic_1 = RNNCriticNetwork(
            state_dim=state_dim, action_dim=action_dim,
            rnn_type=rnn_type, rnn_hidden_size=rnn_hidden_size, rnn_num_layers=rnn_num_layers,
            critic_linear_layers=critic_linear_layers, sequence_length=sequence_length
        ).to(self.device)
        
        self.critic_2 = RNNCriticNetwork(
            state_dim=state_dim, action_dim=action_dim,
            rnn_type=rnn_type, rnn_hidden_size=rnn_hidden_size, rnn_num_layers=rnn_num_layers,
            critic_linear_layers=critic_linear_layers, sequence_length=sequence_length
        ).to(self.device)
        
        self.target_actor = RNNActorNetwork(
            state_dim=state_dim, action_dim=action_dim, N_t=N_t, K=K, P_max=P_max,
            rnn_type=rnn_type, rnn_hidden_size=rnn_hidden_size, rnn_num_layers=rnn_num_layers,
            actor_linear_layers=actor_linear_layers, sequence_length=sequence_length
        ).to(self.device)
        
        self.target_critic_1 = RNNCriticNetwork(
            state_dim=state_dim, action_dim=action_dim,
            rnn_type=rnn_type, rnn_hidden_size=rnn_hidden_size, rnn_num_layers=rnn_num_layers,
            critic_linear_layers=critic_linear_layers, sequence_length=sequence_length
        ).to(self.device)
        
        self.target_critic_2 = RNNCriticNetwork(
            state_dim=state_dim, action_dim=action_dim,
            rnn_type=rnn_type, rnn_hidden_size=rnn_hidden_size, rnn_num_layers=rnn_num_layers,
            critic_linear_layers=critic_linear_layers, sequence_length=sequence_length
        ).to(self.device)
        
        print(f"TD3 using RNN architecture: {rnn_type.upper()} (hidden_size={rnn_hidden_size}, num_layers={rnn_num_layers})")

        # Initialize target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        OPTIMIZERS = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "rmsprop": optim.RMSprop,
            "sgd": optim.SGD,
            "adagrad": optim.Adagrad,
            "adamax": optim.Adamax,
            "nadam": optim.NAdam,
        }
        
        optimizer_name = optimizer.lower()
        self.actor_optimizer = OPTIMIZERS[optimizer_name](self.actor.parameters(), lr=actor_lr, maximize=True)
        self.q_optimizer = OPTIMIZERS[optimizer_name](list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=critic_lr)

        # Initialize replay buffer - use sequence-aware buffers for RNN training
        if self.sequence_length > 1:
            if self.use_per:
                self.replay_buffer = SequencePrioritizedReplayBuffer(
                    buffer_size=buffer_size,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    numpy_rng=self.network_numpy_rng,
                    sequence_length=sequence_length,
                    episode_boundaries=True,
                    alpha=per_alpha,
                    beta_start=per_beta_start,
                    beta_frames=per_beta_frames,
                    epsilon=per_epsilon
                )
                print(f"TD3 RNN using Sequence Prioritized Experience Replay (seq_len={sequence_length}, alpha={per_alpha})")
            else:
                self.replay_buffer = SequenceReplayBuffer(
                    buffer_size=buffer_size,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    numpy_rng=self.network_numpy_rng,
                    sequence_length=sequence_length,
                    episode_boundaries=True
                )
                print(f"TD3 RNN using Sequence Experience Replay (seq_len={sequence_length})")
        else:
            # Use standard buffers for single-step training (backward compatibility)
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
                print(f"TD3 using Prioritized Experience Replay (alpha={per_alpha}, beta_start={per_beta_start})")
            else:
                self.replay_buffer = ReplayBuffer(
                    buffer_size=buffer_size,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    numpy_rng=self.network_numpy_rng
                )
                print("TD3 using standard Experience Replay")

    def select_action(self, state, hidden_states=None):
        """Selects an action based on the current state."""
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, new_hidden_states = self.actor(state, hidden_states)
            # Store hidden states internally for next call
            self.actor.set_hidden_states(new_hidden_states)
            return action.squeeze(0).cpu()

    def select_noised_action(self, state, noise_scale=0.01, hidden_states=None):
        """Selects a noised action for exploration."""
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            raw_action, new_hidden_states = self.actor.forward_raw(state, hidden_states)
            # Store hidden states internally for next call
            self.actor.set_hidden_states(new_hidden_states)

        # Generate noise
        noise_np = self.network_numpy_rng.normal(loc=0.0, scale=self.action_noise_scale, size=raw_action.shape)
        noise = torch.tensor(noise_np, dtype=torch.float32, device=self.device)
        raw_action_noised = raw_action + noise

        # Process actions
        with torch.no_grad():
            noised_action = self.actor.actor_process_raw_actions(raw_action_noised.unsqueeze(0)).squeeze(0)
            clean_action = self.actor.actor_process_raw_actions(raw_action.unsqueeze(0)).squeeze(0)

        return clean_action.cpu(), noised_action.cpu()

    def _sample_from_buffer(self, batch_size):
        """Samples from the replay buffer, handling both standard and prioritized buffers."""
        if self.use_per:
            return self.replay_buffer.sample(batch_size)
        else:
            batch = self.replay_buffer.sample(batch_size)
            dummy_weights = torch.ones(batch_size, device=self.device)
            dummy_indices = np.arange(batch_size)
            return (*batch, dummy_weights, dummy_indices)

    def _calculate_td_errors(self, states, actions, rewards, next_states, q1_values, q2_values):
        """Calculates TD errors for priority updates in PER."""
        with torch.no_grad():
            target_actions, _ = self.target_actor(next_states)
            target_q1, _ = self.target_critic_1(next_states, target_actions)
            target_q2, _ = self.target_critic_2(next_states, target_actions)
            target_q_values = torch.min(target_q1, target_q2)
            target_values = rewards + self.gamma * target_q_values
            
            # Calculate TD errors for both critics (use minimum for priority)
            td_error1 = torch.abs(q1_values - target_values)
            td_error2 = torch.abs(q2_values - target_values)
            td_errors = torch.min(td_error1, td_error2)
            
        return td_errors.detach().cpu().numpy().flatten()

    def training(self, batch_size):
        """Performs a training step on a batch of experiences sampled from the replay buffer."""
        self.actor.train()
        self.total_it += 1
        
        # Sample from buffer
        state, actions, rewards, next_state, weights, indices = self._sample_from_buffer(batch_size)
        
        if self.gpu_used:
            state, actions, rewards, next_state = (t.to(self.device, non_blocking=True) 
                                                  for t in (state, actions, rewards, next_state))
            if self.use_per:
                weights = weights.to(self.device, non_blocking=True)

        # Compute target values
        with torch.no_grad():
            target_actions, _ = self.target_actor(next_state)
            target_q1, _ = self.target_critic_1(next_state, target_actions)
            target_q2, _ = self.target_critic_2(next_state, target_actions)
            target_q_values = torch.min(target_q1, target_q2)
            y = rewards + self.gamma * target_q_values

        # Update Critics
        if self.using_loss_scaling:
            if self.total_it % self.critic_frequency_update == 0:
                updated_critic = True
                with amp.autocast(self.device_string):
                    q1_values, _ = self.critic_1(state, actions)
                    q2_values, _ = self.critic_2(state, actions)
                    
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
                        q1_values, _ = self.critic_1(state, actions)
                        q2_values, _ = self.critic_2(state, actions)
                        
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
                    actor_actions, _ = self.actor(state)
                    actor_q_values, _ = self.critic_1(state, actor_actions)
                    
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
                        actor_actions, _ = self.actor(state)
                        actor_q_values, _ = self.critic_1(state, actor_actions)
                        
                        if self.use_per:
                            actor_loss = (weights * actor_q_values).mean()
                        else:
                            actor_loss = actor_q_values.mean()
                self.update_target_networks(update_target_actor=False, update_target_critic=update_target_critic)

        else:
            # Without loss scaling
            if self.total_it % self.critic_frequency_update == 0:
                updated_critic = True
                q1_values, _ = self.critic_1(state, actions)
                q2_values, _ = self.critic_2(state, actions)

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
                    q1_values, _ = self.critic_1(state, actions)
                    q2_values, _ = self.critic_2(state, actions)
                    
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
                actor_actions, _ = self.actor(state)
                actor_q_values, _ = self.critic_1(state, actor_actions)
                
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
                    actor_actions, _ = self.actor(state)
                    actor_q_values, _ = self.critic_1(state, actor_actions)
                    
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
        """Updates the target networks using soft updates."""
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
        """Stores a transition in the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, batch_size=batch_size)

    def get_buffer_info(self):
        """Returns information about the current buffer state."""
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
        """Saves the actor, critic, and target networks to the specified directory."""
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
            'use_rnn': True,
            'rnn_type': self.rnn_type,
            'use_per': self.use_per,
            'buffer_info': self.get_buffer_info()
        }
        torch.save(config, config_path)

    def load_models(self, directory):
        """Loads the actor, critic, and target networks from the specified directory."""
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
            print(f"Loaded TD3 RNN model with buffer type: {config.get('buffer_info', {}).get('buffer_type', 'Unknown')}")
