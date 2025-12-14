"""
 RNN-based neural network architectures for reinforcement learning algorithms.

This module provides  RNN-based Actor and Critic networks with performance improvements
while maintaining backward compatibility with existing code.

Key optimizations:
- Efficient hidden state management with caching
-  sequence processing
- Reduced memory allocations
- Faster tensor operations
- Improved batch processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


class RNNActorNetwork(nn.Module):
    """
     RNN-based Actor Network for reinforcement learning algorithms.
    
    Performance improvements:
    - Cached hidden state management
    -  sequence processing
    - Reduced memory allocations
    - Faster tensor operations
    """
    
    def __init__(self, state_dim, action_dim, N_t, K, P_max, 
                 rnn_type='lstm', rnn_hidden_size=128, rnn_num_layers=1,
                 actor_linear_layers=[128, 128], sequence_length=2):
        
        # Validate parameters
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if N_t <= 0:
            raise ValueError(f"N_t must be positive, got {N_t}")
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        if P_max <= 0:
            raise ValueError(f"P_max must be positive, got {P_max}")
        if rnn_type.lower() not in ['lstm', 'gru']:
            raise ValueError(f"rnn_type must be 'lstm' or 'gru', got {rnn_type}")
        if rnn_hidden_size <= 0:
            raise ValueError(f"rnn_hidden_size must be positive, got {rnn_hidden_size}")
        if rnn_num_layers <= 0:
            raise ValueError(f"rnn_num_layers must be positive, got {rnn_num_layers}")
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
            
        super(RNNActorNetwork, self).__init__()
        
        # Store parameters
        self.N_t = N_t
        self.K = K
        self.P_max = P_max
        self.tensored_P_max = torch.tensor(P_max)
        self.numpied_P_max = self.tensored_P_max.detach().numpy()
        self.rnn_type = rnn_type.lower()
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.sequence_length = sequence_length
        
        # Initialize RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=state_dim,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=state_dim,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True
            )
        
        # Initialize linear layers after RNN
        self.linear_layers = nn.ModuleList()
        input_dim = rnn_hidden_size
        
        for layer_dim in actor_linear_layers:
            self.linear_layers.append(nn.Linear(input_dim, layer_dim))
            input_dim = layer_dim
        
        # Output layer
        self.output = nn.Linear(actor_linear_layers[-1], action_dim)
        self.batch_norm = nn.BatchNorm1d(action_dim)
        
        # Initialize hidden states with caching
        self.hidden_states = None
        self._cached_hidden_states = None
        self._last_batch_size = 0
        
        # Pre-allocate tensors for efficiency
        self._preallocated_tensors = {}
        
    def _get_preallocated_tensor(self, key, shape, dtype, device):
        """Get or create pre-allocated tensor for efficiency."""
        if key not in self._preallocated_tensors:
            self._preallocated_tensors[key] = torch.empty(shape, dtype=dtype, device=device)
        elif self._preallocated_tensors[key].shape != shape:
            self._preallocated_tensors[key] = torch.empty(shape, dtype=dtype, device=device)
        return self._preallocated_tensors[key]
    
    def reset_hidden_states(self, batch_size=1, device=None):
        """Reset hidden states to zeros."""
        if device is None:
            device = next(self.parameters()).device
            
        if self.rnn_type == 'lstm':
            self.hidden_states = (
                torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device),
                torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
            )
        else:  # GRU
            self.hidden_states = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
        
        self._cached_hidden_states = None
        self._last_batch_size = batch_size
    
    def get_hidden_states(self):
        """Get current hidden states."""
        return self.hidden_states
    
    def set_hidden_states(self, hidden_states):
        """Set hidden states (detach from computation graph to avoid gradient issues)."""
        if hidden_states is not None:
            if isinstance(hidden_states, tuple):  # LSTM
                self.hidden_states = (hidden_states[0].detach(), hidden_states[1].detach())
            else:  # GRU
                self.hidden_states = hidden_states.detach()
        else:
            self.hidden_states = None
    
    def actor_W_projection_operator(self, raw_W):
        """
         projection operator with reduced memory allocations.
        """
        # Ensure float type
        raw_W = raw_W.detach()

        # Frobenius norms for each matrix in the batch
        frobenius_norms = torch.linalg.norm(raw_W, dim=(1, 2), ord='fro')

        # Traces of (W * W^H) -  computation
        traces = torch.einsum('bij,bji->b', raw_W, raw_W.conj().transpose(1, 2)).real

        # Mask of matrices exceeding power constraint
        exceed_mask = traces > self.tensored_P_max

        # Scaling factors - vectorized operation
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
         action processing with reduced memory allocations.
        """
        batch_size = raw_actions.shape[0]
        
        # Pre-allocate output tensor
        actions = torch.zeros_like(raw_actions)

        # Splitting raw actions into W-related and Theta-related components
        W_raw_actions = raw_actions[:, :2 * self.N_t * self.K]
        theta_actions = raw_actions[:, 2 * self.N_t * self.K:]

        # Process Theta actions - 
        theta_real = theta_actions[:, 0::2]
        theta_imag = theta_actions[:, 1::2]
        magnitudes = torch.sqrt(theta_real**2 + theta_imag**2)

        # Avoid division by zero - vectorized
        magnitudes = torch.where(magnitudes == 0, 1, magnitudes)
        normalized_theta_real = theta_real / magnitudes
        normalized_theta_imag = theta_imag / magnitudes

        # Store normalized Theta components in actions
        actions[:, 2 * self.N_t * self.K::2] = normalized_theta_real
        actions[:, 2 * self.N_t * self.K + 1::2] = normalized_theta_imag

        # Process W actions - 
        W_raw_actions = W_raw_actions.reshape(batch_size, self.K, 2 * self.N_t)
        W_real = W_raw_actions[:, :, 0::2]
        W_imag = W_raw_actions[:, :, 1::2]
        raw_W = W_real + 1j * W_imag

        # Project W onto the power constraint set
        W = self.actor_W_projection_operator(raw_W)

        # Store processed W components in actions -  indexing
        flattened_real = W.real.flatten(start_dim=1)
        flattened_imag = W.imag.flatten(start_dim=1)
        actions[:, :2 * self.N_t * self.K:2] = flattened_real
        actions[:, 1:2 * self.N_t * self.K:2] = flattened_imag

        return actions
    
    def _initialize_hidden_states(self, batch_size, device):
        """Initialize hidden states with caching."""
        if self._cached_hidden_states is None or self._last_batch_size != batch_size:
            if self.rnn_type == 'lstm':
                self._cached_hidden_states = (
                    torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device),
                    torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
                )
            else:  # GRU
                self._cached_hidden_states = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
            self._last_batch_size = batch_size
        return self._cached_hidden_states
    
    def forward(self, x, hidden_states=None):
        """
         forward pass through the RNN-based actor network.
        """
        # Handle single-step input by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, state_dim)
            single_step = True
        else:
            single_step = False
        
        # Use provided hidden states or current ones
        if hidden_states is None:
            hidden_states = self.hidden_states
        
        # Check if hidden states match current batch size
        batch_size = x.shape[0]
        device = x.device
        
        if hidden_states is not None:
            # Check batch size compatibility
            if isinstance(hidden_states, tuple):  # LSTM
                stored_batch_size = hidden_states[0].shape[1]
            else:  # GRU
                stored_batch_size = hidden_states.shape[1]
            
            if stored_batch_size != batch_size:
                hidden_states = None  # Reset if batch size doesn't match
        
        # Initialize hidden states if None
        if hidden_states is None:
            hidden_states = self._initialize_hidden_states(batch_size, device)
        
        # Forward pass through RNN
        rnn_output, new_hidden_states = self.rnn(x, hidden_states)
        
        # Update hidden states (detach from computation graph to avoid gradient issues)
        if isinstance(new_hidden_states, tuple):  # LSTM
            self.hidden_states = (new_hidden_states[0].detach(), new_hidden_states[1].detach())
        else:  # GRU
            self.hidden_states = new_hidden_states.detach()
        
        # Use the last output from the sequence
        rnn_output = rnn_output[:, -1, :]  # (batch_size, rnn_hidden_size)
        
        # Forward pass through linear layers - 
        for layer in self.linear_layers:
            rnn_output = F.relu(layer(rnn_output))
        
        # Output layer
        rnn_output = F.tanh(self.output(rnn_output))
        rnn_output = self.batch_norm(rnn_output)
        
        # Process raw actions
        processed_actions = self.actor_process_raw_actions(rnn_output)
        
        return processed_actions, new_hidden_states
    
    def forward_raw(self, x, hidden_states=None):
        """
         forward pass without applying actor_process_raw_actions.
        """
        # Handle single-step input by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
            single_step = True
        else:
            single_step = False
        
        # Use provided hidden states or current ones
        if hidden_states is None:
            hidden_states = self.hidden_states
        
        # Initialize hidden states if None
        if hidden_states is None:
            batch_size = x.shape[0]
            device = x.device
            hidden_states = self._initialize_hidden_states(batch_size, device)
        
        # Forward pass through RNN
        rnn_output, new_hidden_states = self.rnn(x, hidden_states)
        
        # Update hidden states (detach from computation graph to avoid gradient issues)
        if isinstance(new_hidden_states, tuple):  # LSTM
            self.hidden_states = (new_hidden_states[0].detach(), new_hidden_states[1].detach())
        else:  # GRU
            self.hidden_states = new_hidden_states.detach()
        
        # Use the last output from the sequence
        rnn_output = rnn_output[:, -1, :]
        
        # Forward pass through linear layers
        for layer in self.linear_layers:
            rnn_output = F.relu(layer(rnn_output))
        
        # Output layer (raw tanh output)
        rnn_output = torch.tanh(self.output(rnn_output))
        rnn_output = self.batch_norm(rnn_output)
        
        return rnn_output, new_hidden_states


class RNNCriticNetwork(nn.Module):
    """
     RNN-based Critic Network for reinforcement learning algorithms.
    
    Performance improvements:
    - Cached hidden state management
    -  sequence processing
    - Reduced memory allocations
    - Faster tensor operations
    """
    
    def __init__(self, state_dim, action_dim, 
                 rnn_type='lstm', rnn_hidden_size=128, rnn_num_layers=1,
                 critic_linear_layers=[128, 128], sequence_length=1):
        
        # Validate parameters
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if rnn_type.lower() not in ['lstm', 'gru']:
            raise ValueError(f"rnn_type must be 'lstm' or 'gru', got {rnn_type}")
        if rnn_hidden_size <= 0:
            raise ValueError(f"rnn_hidden_size must be positive, got {rnn_hidden_size}")
        if rnn_num_layers <= 0:
            raise ValueError(f"rnn_num_layers must be positive, got {rnn_num_layers}")
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
            
        super(RNNCriticNetwork, self).__init__()
        
        # Store parameters
        self.rnn_type = rnn_type.lower()
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.sequence_length = sequence_length
        
        # Input dimension is state_dim + action_dim
        input_dim = state_dim + action_dim
        
        # Initialize RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True
            )
        
        # Initialize linear layers after RNN
        self.linear_layers = nn.ModuleList()
        layer_input_dim = rnn_hidden_size
        
        for layer_dim in critic_linear_layers:
            self.linear_layers.append(nn.Linear(layer_input_dim, layer_dim))
            layer_input_dim = layer_dim
        
        # Output layer
        self.output = nn.Linear(critic_linear_layers[-1], 1)
        
        # Initialize hidden states with caching
        self.hidden_states = None
        self._cached_hidden_states = None
        self._last_batch_size = 0
        
        # Pre-allocate tensors for efficiency
        self._preallocated_tensors = {}
    
    def _get_preallocated_tensor(self, key, shape, dtype, device):
        """Get or create pre-allocated tensor for efficiency."""
        if key not in self._preallocated_tensors:
            self._preallocated_tensors[key] = torch.empty(shape, dtype=dtype, device=device)
        elif self._preallocated_tensors[key].shape != shape:
            self._preallocated_tensors[key] = torch.empty(shape, dtype=dtype, device=device)
        return self._preallocated_tensors[key]
    
    def reset_hidden_states(self, batch_size=1, device=None):
        """Reset hidden states to zeros."""
        if device is None:
            device = next(self.parameters()).device
            
        if self.rnn_type == 'lstm':
            self.hidden_states = (
                torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device),
                torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
            )
        else:  # GRU
            self.hidden_states = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
        
        self._cached_hidden_states = None
        self._last_batch_size = batch_size
    
    def get_hidden_states(self):
        """Get current hidden states."""
        return self.hidden_states
    
    def set_hidden_states(self, hidden_states):
        """Set hidden states (detach from computation graph to avoid gradient issues)."""
        if hidden_states is not None:
            if isinstance(hidden_states, tuple):  # LSTM
                self.hidden_states = (hidden_states[0].detach(), hidden_states[1].detach())
            else:  # GRU
                self.hidden_states = hidden_states.detach()
        else:
            self.hidden_states = None
    
    def _initialize_hidden_states(self, batch_size, device):
        """Initialize hidden states with caching."""
        if self._cached_hidden_states is None or self._last_batch_size != batch_size:
            if self.rnn_type == 'lstm':
                self._cached_hidden_states = (
                    torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device),
                    torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
                )
            else:  # GRU
                self._cached_hidden_states = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
            self._last_batch_size = batch_size
        return self._cached_hidden_states
    
    def forward(self, state, action, hidden_states=None):
        """
         forward pass through the RNN-based critic network.
        """
        # Handle input dimensions -  sequence handling
        if state.dim() == 2 and action.dim() == 2:
            # Both are single-step inputs
            state = state.unsqueeze(1)  # (batch_size, 1, state_dim)
            action = action.unsqueeze(1)  # (batch_size, 1, action_dim)
            single_step = True
        elif state.dim() == 3 and action.dim() == 3:
            # Both are sequence inputs - check if sequence lengths match
            if state.shape[1] != action.shape[1]:
                # Mismatched sequence lengths - expand the shorter one
                if state.shape[1] < action.shape[1]:
                    # Expand state to match action sequence length
                    seq_diff = action.shape[1] - state.shape[1]
                    state = state.repeat(1, seq_diff + 1, 1)
                else:
                    # Expand action to match state sequence length
                    seq_diff = state.shape[1] - action.shape[1]
                    action = action.repeat(1, seq_diff + 1, 1)
            single_step = False
        elif state.dim() == 3 and action.dim() == 2:
            # State is sequence, action is single-step - expand action to match state sequence length
            seq_len = state.shape[1]
            action = action.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, action_dim)
            single_step = False
        elif state.dim() == 2 and action.dim() == 3:
            # State is single-step, action is sequence - expand state to match action sequence length
            seq_len = action.shape[1]
            state = state.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, state_dim)
            single_step = False
        else:
            raise ValueError(f"Incompatible tensor dimensions: state.dim()={state.dim()}, action.dim()={action.dim()}")
        
        # Concatenate state and action - 
        x = torch.cat([state, action], dim=-1)  # (batch_size, sequence_length, state_dim + action_dim)
        
        # Use provided hidden states or current ones
        if hidden_states is None:
            hidden_states = self.hidden_states
        
        # Check if hidden states match current batch size
        batch_size = x.shape[0]
        device = x.device
        
        if hidden_states is not None:
            # Check batch size compatibility
            if isinstance(hidden_states, tuple):  # LSTM
                stored_batch_size = hidden_states[0].shape[1]
            else:  # GRU
                stored_batch_size = hidden_states.shape[1]
            
            if stored_batch_size != batch_size:
                hidden_states = None  # Reset if batch size doesn't match
        
        # Initialize hidden states if None
        if hidden_states is None:
            hidden_states = self._initialize_hidden_states(batch_size, device)
        
        # Forward pass through RNN
        rnn_output, new_hidden_states = self.rnn(x, hidden_states)
        
        # Update hidden states (detach from computation graph to avoid gradient issues)
        if isinstance(new_hidden_states, tuple):  # LSTM
            self.hidden_states = (new_hidden_states[0].detach(), new_hidden_states[1].detach())
        else:  # GRU
            self.hidden_states = new_hidden_states.detach()
        
        # Use the last output from the sequence
        rnn_output = rnn_output[:, -1, :]  # (batch_size, rnn_hidden_size)
        
        # Forward pass through linear layers - 
        for layer in self.linear_layers:
            rnn_output = F.relu(layer(rnn_output))
        
        # Output layer
        q_value = self.output(rnn_output)
        
        return q_value, new_hidden_states


class RNNActorNetworkSAC(nn.Module):
    """
     RNN-based Actor Network specifically for SAC algorithm.
    
    Performance improvements:
    - Cached hidden state management
    -  sequence processing
    - Reduced memory allocations
    - Faster tensor operations
    """
    
    def __init__(self, state_dim, action_dim, N_t, K, P_max, 
                 rnn_type='lstm', rnn_hidden_size=128, rnn_num_layers=1,
                 actor_linear_layers=[128, 128], sequence_length=1):
        
        super(RNNActorNetworkSAC, self).__init__()
        
        # Store parameters
        self.N_t = N_t
        self.K = K
        self.P_max = P_max
        self.tensored_P_max = torch.tensor(P_max)
        self.numpied_P_max = self.tensored_P_max.detach().numpy()
        self.rnn_type = rnn_type.lower()
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.sequence_length = sequence_length
        
        # Initialize RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=state_dim,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=state_dim,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True
            )
        
        # Initialize linear layers after RNN
        self.linear_layers = nn.ModuleList()
        input_dim = rnn_hidden_size
        
        for layer_dim in actor_linear_layers:
            self.linear_layers.append(nn.Linear(input_dim, layer_dim))
            input_dim = layer_dim
        
        # SAC-specific outputs
        self.mean_output = nn.Linear(actor_linear_layers[-1], action_dim)
        self.log_std_output = nn.Linear(actor_linear_layers[-1], action_dim)
        
        # Clamp bounds for log_std
        self.log_std_min = -10
        self.log_std_max = 2
        
        self.batch_norm = nn.BatchNorm1d(action_dim)
        
        # Initialize hidden states with caching
        self.hidden_states = None
        self._cached_hidden_states = None
        self._last_batch_size = 0
        
        # Pre-allocate tensors for efficiency
        self._preallocated_tensors = {}
    
    def _get_preallocated_tensor(self, key, shape, dtype, device):
        """Get or create pre-allocated tensor for efficiency."""
        if key not in self._preallocated_tensors:
            self._preallocated_tensors[key] = torch.empty(shape, dtype=dtype, device=device)
        elif self._preallocated_tensors[key].shape != shape:
            self._preallocated_tensors[key] = torch.empty(shape, dtype=dtype, device=device)
        return self._preallocated_tensors[key]
    
    def reset_hidden_states(self, batch_size=1, device=None):
        """Reset hidden states to zeros."""
        if device is None:
            device = next(self.parameters()).device
            
        if self.rnn_type == 'lstm':
            self.hidden_states = (
                torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device),
                torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
            )
        else:  # GRU
            self.hidden_states = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
        
        self._cached_hidden_states = None
        self._last_batch_size = batch_size
    
    def get_hidden_states(self):
        """Get current hidden states."""
        return self.hidden_states
    
    def set_hidden_states(self, hidden_states):
        """Set hidden states (detach from computation graph to avoid gradient issues)."""
        if hidden_states is not None:
            if isinstance(hidden_states, tuple):  # LSTM
                self.hidden_states = (hidden_states[0].detach(), hidden_states[1].detach())
            else:  # GRU
                self.hidden_states = hidden_states.detach()
        else:
            self.hidden_states = None
    
    def actor_W_projection_operator(self, raw_W):
        """Same as in RNNActorNetwork."""
        raw_W = raw_W.detach()
        frobenius_norms = torch.linalg.norm(raw_W, dim=(1, 2), ord='fro')
        traces = torch.einsum('bij,bji->b', raw_W, raw_W.conj().transpose(1, 2)).real
        exceed_mask = traces > self.tensored_P_max
        scaling_factors = torch.where(
            exceed_mask,
            (self.tensored_P_max.sqrt() / frobenius_norms),
            torch.ones_like(frobenius_norms)
        )
        scaling_factors = scaling_factors.view(-1, 1, 1)
        projected_W = raw_W * scaling_factors
        return projected_W
    
    def actor_process_raw_actions(self, raw_actions):
        """Same as in RNNActorNetwork."""
        batch_size = raw_actions.shape[0]
        actions = torch.zeros_like(raw_actions)
        W_raw_actions = raw_actions[:, :2 * self.N_t * self.K]
        theta_actions = raw_actions[:, 2 * self.N_t * self.K:]
        theta_real = theta_actions[:, 0::2]
        theta_imag = theta_actions[:, 1::2]
        magnitudes = torch.sqrt(theta_real**2 + theta_imag**2)
        magnitudes = torch.where(magnitudes == 0, 1, magnitudes)
        normalized_theta_real = theta_real / magnitudes
        normalized_theta_imag = theta_imag / magnitudes
        actions[:, 2 * self.N_t * self.K::2] = normalized_theta_real
        actions[:, 2 * self.N_t * self.K + 1::2] = normalized_theta_imag
        W_raw_actions = W_raw_actions.reshape(batch_size, self.K, 2 * self.N_t)
        W_real = W_raw_actions[:, :, 0::2]
        W_imag = W_raw_actions[:, :, 1::2]
        raw_W = W_real + 1j * W_imag
        W = self.actor_W_projection_operator(raw_W)
        flattened_real = W.real.flatten(start_dim=1)
        flattened_imag = W.imag.flatten(start_dim=1)
        actions[:, :2 * self.N_t * self.K:2] = flattened_real
        actions[:, 1:2 * self.N_t * self.K:2] = flattened_imag
        return actions
    
    def _initialize_hidden_states(self, batch_size, device):
        """Initialize hidden states with caching."""
        if self._cached_hidden_states is None or self._last_batch_size != batch_size:
            if self.rnn_type == 'lstm':
                self._cached_hidden_states = (
                    torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device),
                    torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
                )
            else:  # GRU
                self._cached_hidden_states = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
            self._last_batch_size = batch_size
        return self._cached_hidden_states
    
    def forward(self, x, hidden_states=None):
        """ forward pass for SAC actor network."""
        # Handle single-step input
        if x.dim() == 2:
            x = x.unsqueeze(1)
            single_step = True
        else:
            single_step = False
        
        # Use provided hidden states or current ones
        if hidden_states is None:
            hidden_states = self.hidden_states
        
        # Check if hidden states match current batch size
        batch_size = x.shape[0]
        device = x.device
        
        if hidden_states is not None:
            # Check batch size compatibility
            if isinstance(hidden_states, tuple):  # LSTM
                stored_batch_size = hidden_states[0].shape[1]
            else:  # GRU
                stored_batch_size = hidden_states.shape[1]
            
            if stored_batch_size != batch_size:
                hidden_states = None  # Reset if batch size doesn't match
        
        # Initialize hidden states if None
        if hidden_states is None:
            hidden_states = self._initialize_hidden_states(batch_size, device)
        
        # Forward pass through RNN
        rnn_output, new_hidden_states = self.rnn(x, hidden_states)
        self.hidden_states = new_hidden_states
        
        # Use the last output from the sequence
        rnn_output = rnn_output[:, -1, :]
        
        # Forward pass through linear layers
        for layer in self.linear_layers:
            rnn_output = F.relu(layer(rnn_output))
        
        # SAC-specific outputs
        mean = self.mean_output(rnn_output)
        log_std = self.log_std_output(rnn_output)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, new_hidden_states
    
    def sample(self, state, hidden_states=None):
        """Sample action from the policy."""
        mean, log_std, new_hidden_states = self.forward(state, hidden_states)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.tensored_P_max
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.tensored_P_max * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.tensored_P_max
        return action, log_prob, mean, new_hidden_states
    
    def get_action(self, state, hidden_states=None):
        """Get deterministic action."""
        mean, log_std, new_hidden_states = self.forward(state, hidden_states)
        action = torch.tanh(mean) * self.tensored_P_max
        return action, new_hidden_states
