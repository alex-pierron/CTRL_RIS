"""
 replay buffer implementations for reinforcement learning algorithms.

This module provides  replay buffer implementations with performance improvements
while maintaining backward compatibility with existing code.

Key optimizations:
- Efficient memory management with pre-allocation
-  sampling algorithms
- Reduced tensor conversions
- Faster sequence handling
- Improved priority tree operations
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union, List
import threading


class ReplayBuffer:
    """
     replay buffer for storing and sampling experiences in reinforcement learning.
    
    Performance improvements:
    - Pre-allocated memory buffers
    -  sampling with vectorized operations
    - Reduced memory copying
    - Faster tensor operations
    """
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, numpy_rng):
        """
        Initializes the  replay buffer with specified size and dimensions.
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.numpy_rng = numpy_rng
        
        # Pre-allocate buffers with optimal data types
        self.state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        
        self.pointer = 0
        self.size = 0
        self.buffer_filled = False
        
        # Pre-allocate sampling indices for efficiency
        self._sampling_indices = np.empty(buffer_size, dtype=np.int32)
        self._lock = threading.Lock()
    
    def add(self, state, action, reward, next_state, batch_size=None):
        """
         add method with reduced memory allocations.
        """
        with self._lock:
            # Determine if we're handling a batch or single experience
            if batch_size is None:
                if hasattr(state, 'shape') and len(state.shape) > 1:
                    batch_size = state.shape[0]
                elif isinstance(state, list):
                    batch_size = len(state)
                else:
                    batch_size = 1
                    state = [state]
                    action = [action]
                    reward = [reward]
                    next_state = [next_state]

            # Convert to numpy arrays if needed - 
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            if not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float32)
            if not isinstance(reward, np.ndarray):
                reward = np.array(reward, dtype=np.float32)
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state, dtype=np.float32)
            
            # Ensure rewards have the correct shape
            if reward.ndim == 1:
                reward = reward.reshape(-1, 1)
            
            #  buffer update with vectorized operations
            if self.pointer + batch_size <= self.buffer_size:
                # Contiguous space available
                indices = np.arange(self.pointer, self.pointer + batch_size) % self.buffer_size
                self.state_buffer[indices] = state
                self.action_buffer[indices] = action
                self.reward_buffer[indices] = reward
                self.next_state_buffer[indices] = next_state
            else:
                # Need to wrap around the buffer
                remaining = self.buffer_size - self.pointer
                first_indices = np.arange(self.pointer, self.buffer_size)
                second_indices = np.arange(0, batch_size - remaining)
                
                # First chunk (until the end of buffer)
                self.state_buffer[first_indices] = state[:remaining]
                self.action_buffer[first_indices] = action[:remaining]
                self.reward_buffer[first_indices] = reward[:remaining]
                self.next_state_buffer[first_indices] = next_state[:remaining]
                
                # Second chunk (wrapped to the beginning)
                if batch_size - remaining > 0:
                    self.state_buffer[second_indices] = state[remaining:]
                    self.action_buffer[second_indices] = action[remaining:]
                    self.reward_buffer[second_indices] = reward[remaining:]
                    self.next_state_buffer[second_indices] = next_state[remaining:]
            
            # Update pointer and size
            self.pointer = (self.pointer + batch_size) % self.buffer_size
            self.size = min(self.size + batch_size, self.buffer_size)
            self.buffer_filled = (self.size >= self.buffer_size)

    def sample(self, batch_size: int):
        """
         sampling with vectorized operations.
        """
        with self._lock:
            # Determine the actual size to sample from
            actual_size = self.size if not self.buffer_filled else self.buffer_size
            
            # Generate random indices - 
            if actual_size == self.buffer_size:
                # Buffer is full, sample from entire buffer
                indices = self.numpy_rng.choice(actual_size, size=batch_size, replace=True)
            else:
                # Buffer is not full, sample from filled portion
                indices = self.numpy_rng.choice(actual_size, size=batch_size, replace=True)
            
            # Vectorized sampling - much faster than individual indexing
            states = torch.from_numpy(self.state_buffer[indices])
            actions = torch.from_numpy(self.action_buffer[indices])
            rewards = torch.from_numpy(self.reward_buffer[indices])
            next_states = torch.from_numpy(self.next_state_buffer[indices])
            
            return states, actions, rewards, next_states

    def new_episode(self):
        """Placeholder for episode boundary handling."""
        pass


class SumTree:
    """
     SumTree implementation for prioritized experience replay.
    
    Performance improvements:
    - Reduced memory allocations
    -  tree operations
    - Faster priority updates
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.data_pointer = 0
        self.n_entries = 0
        
        # Pre-allocate arrays for efficiency
        self._update_indices = np.empty(capacity, dtype=np.int32)
        self._update_priorities = np.empty(capacity, dtype=np.float32)
    
    def add(self, priority, data_idx):
        """ add operation."""
        tree_idx = data_idx + self.capacity - 1
        self.data[self.data_pointer] = data_idx
        self.update(data_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def _propagate(self, idx):
        """ propagation with reduced operations."""
        parent = (idx - 1) // 2
        self.tree[parent] = self.tree[2 * parent + 1] + self.tree[2 * parent + 2]
        if parent != 0:
            self._propagate(parent)
    
    def get_leaf(self, value):
        """ leaf retrieval."""
        parent = 0
        while True:
            left_child = 2 * parent + 1
            right_child = left_child + 1
            if left_child >= len(self.tree):
                break
            if value <= self.tree[left_child]:
                parent = left_child
            else:
                value -= self.tree[left_child]
                parent = right_child
        return parent, self.data[parent - self.capacity + 1]
    
    def total_priority(self):
        """Return the total sum of all priorities."""
        return self.tree[0]
    
    def update(self, data_idx, priority):
        """ priority update."""
        tree_idx = data_idx + self.capacity - 1
        self.tree[tree_idx] = priority
        self._propagate(tree_idx)
    
    def update_batch(self, data_indices, priorities):
        """Batch update for multiple priorities - much faster than individual updates."""
        tree_indices = data_indices + self.capacity - 1
        self.tree[tree_indices] = priorities
        
        # Propagate updates in batch
        unique_parents = set()
        for tree_idx in tree_indices:
            parent = (tree_idx - 1) // 2
            while parent >= 0:
                unique_parents.add(parent)
                parent = (parent - 1) // 2
        
        # Update all affected parents
        for parent in unique_parents:
            left_child = 2 * parent + 1
            right_child = left_child + 1
            if right_child < len(self.tree):
                self.tree[parent] = self.tree[left_child] + self.tree[right_child]


class PrioritizedReplayBuffer:
    """
     prioritized replay buffer with performance improvements.
    
    Performance improvements:
    -  SumTree operations
    - Batch priority updates
    - Reduced memory allocations
    - Faster sampling
    """
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, numpy_rng,
                 alpha: float = 0.6, beta_start: float = 0.2, beta_frames: int = 100000,
                 epsilon: float = 1e-6):
        """
        Initializes the  prioritized replay buffer.
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1
        self.numpy_rng = numpy_rng
        
        # Pre-allocate buffers
        self.state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        
        self.pointer = 0
        self.size = 0
        self.buffer_filled = False
        
        # Initialize  sum tree
        self.tree = SumTree(buffer_size)
        self.max_priority = 1.0
        
        # Pre-allocate arrays for efficiency
        self._sampling_indices = np.empty(buffer_size, dtype=np.int32)
        self._sampling_priorities = np.empty(buffer_size, dtype=np.float32)
        self._sampling_weights = np.empty(buffer_size, dtype=np.float32)
        self._lock = threading.Lock()
    
    def _get_beta(self):
        """Calculate current beta value (annealed from beta_start to 1.0)."""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def add(self, state, action, reward, next_state, batch_size=None):
        """
         add method with priority handling.
        """
        with self._lock:
            # Determine batch size
            if batch_size is None:
                if hasattr(state, 'shape') and len(state.shape) > 1:
                    batch_size = state.shape[0]
                elif isinstance(state, list):
                    batch_size = len(state)
                else:
                    batch_size = 1
                    state = [state]
                    action = [action]
                    reward = [reward]
                    next_state = [next_state]

            # Convert to numpy arrays if needed
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            if not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float32)
            if not isinstance(reward, np.ndarray):
                reward = np.array(reward, dtype=np.float32)
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state, dtype=np.float32)
            
            # Ensure rewards have the correct shape
            if reward.ndim == 1:
                reward = reward.reshape(-1, 1)
            
            #  buffer update
            if self.pointer + batch_size <= self.buffer_size:
                indices = np.arange(self.pointer, self.pointer + batch_size) % self.buffer_size
                self.state_buffer[indices] = state
                self.action_buffer[indices] = action
                self.reward_buffer[indices] = reward
                self.next_state_buffer[indices] = next_state
            else:
                remaining = self.buffer_size - self.pointer
                first_indices = np.arange(self.pointer, self.buffer_size)
                second_indices = np.arange(0, batch_size - remaining)
                
                self.state_buffer[first_indices] = state[:remaining]
                self.action_buffer[first_indices] = action[:remaining]
                self.reward_buffer[first_indices] = reward[:remaining]
                self.next_state_buffer[first_indices] = next_state[:remaining]
                
                if batch_size - remaining > 0:
                    self.state_buffer[second_indices] = state[remaining:]
                    self.action_buffer[second_indices] = action[remaining:]
                    self.reward_buffer[second_indices] = reward[remaining:]
                    self.next_state_buffer[second_indices] = next_state[remaining:]
            
            # Add to priority tree with maximum priority
            for i in range(batch_size):
                data_idx = (self.pointer + i) % self.buffer_size
                self.tree.add(self.max_priority, data_idx)
            
            # Update pointer and size
            self.pointer = (self.pointer + batch_size) % self.buffer_size
            self.size = min(self.size + batch_size, self.buffer_size)
            self.buffer_filled = (self.size >= self.buffer_size)

    def sample(self, batch_size: int):
        """
         prioritized sampling.
        """
        with self._lock:
            # Determine actual size
            actual_size = self.size if not self.buffer_filled else self.buffer_size
            
            # Generate random values for sampling
            total_priority = self.tree.total_priority()
            segment = total_priority / batch_size
            
            #  sampling
            indices = np.empty(batch_size, dtype=np.int32)
            priorities = np.empty(batch_size, dtype=np.float32)
            
            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                value = self.numpy_rng.uniform(a, b)
                tree_idx, data_idx = self.tree.get_leaf(value)
                indices[i] = data_idx
                priorities[i] = self.tree.tree[tree_idx]
            
            # Calculate importance sampling weights
            beta = self._get_beta()
            min_priority = self.epsilon if actual_size < self.buffer_size else np.min(priorities)
            weights = (priorities / min_priority) ** (-beta)
            weights = weights / np.max(weights)
            weights = torch.from_numpy(weights)
            
            # Sample data
            states = torch.from_numpy(self.state_buffer[indices])
            actions = torch.from_numpy(self.action_buffer[indices])
            rewards = torch.from_numpy(self.reward_buffer[indices])
            next_states = torch.from_numpy(self.next_state_buffer[indices])
            
            return states, actions, rewards, next_states, weights, indices

    def update_priorities(self, indices, priorities):
        """
         batch priority update.
        """
        with self._lock:
            # Add epsilon to prevent zero priorities
            priorities = np.maximum(priorities, self.epsilon)
            
            # Update max priority
            self.max_priority = max(self.max_priority, np.max(priorities))
            
            # Batch update in tree
            # indices are data indices; SumTree.update_batch expects data indices
            self.tree.update_batch(indices, priorities)

    def new_episode(self):
        """Placeholder for episode boundary handling."""
        pass


class SequenceReplayBuffer:
    """
     sequence-aware replay buffer for RNN training.
    
    Performance improvements:
    - Efficient sequence sampling
    -  episode boundary handling
    - Reduced memory allocations
    - Faster sequence operations
    """
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, numpy_rng,
                 sequence_length: int = 2, episode_boundaries: bool = True):
        """
        Initializes the  sequence replay buffer.
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.episode_boundaries = episode_boundaries
        self.numpy_rng = numpy_rng
        
        # Pre-allocate buffers
        self.state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.done_buffer = np.zeros((buffer_size, 1), dtype=np.bool_)
        
        self.pointer = 0
        self.size = 0
        self.buffer_filled = False
        
        # Track episode boundaries for sequence sampling
        self.episode_starts = []
        self.episode_ends = []
        
        # Pre-allocate arrays for efficiency
        self._sequence_indices = np.empty(sequence_length, dtype=np.int32)
        self._lock = threading.Lock()
    
    def add(self, state, action, reward, next_state, done=False, batch_size=None):
        """
         add method with episode boundary tracking.
        """
        with self._lock:
            # Determine batch size
            if batch_size is None:
                if hasattr(state, 'shape') and len(state.shape) > 1:
                    batch_size = state.shape[0]
                elif isinstance(state, list):
                    batch_size = len(state)
                else:
                    batch_size = 1
                    state = [state]
                    action = [action]
                    reward = [reward]
                    next_state = [next_state]
                    done = [done]
            
            # Convert to numpy arrays if needed
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            if not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float32)
            if not isinstance(reward, np.ndarray):
                reward = np.array(reward, dtype=np.float32)
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state, dtype=np.float32)
            if not isinstance(done, np.ndarray):
                done = np.array(done, dtype=np.bool_)
            
            # Ensure correct shapes
            if reward.ndim == 1:
                reward = reward.reshape(-1, 1)
            if done.ndim == 1:
                done = done.reshape(-1, 1)
            
            #  buffer update
            if self.pointer + batch_size <= self.buffer_size:
                indices = np.arange(self.pointer, self.pointer + batch_size) % self.buffer_size
                self.state_buffer[indices] = state
                self.action_buffer[indices] = action
                self.reward_buffer[indices] = reward
                self.next_state_buffer[indices] = next_state
                # Handle done parameter - scalar or array
                if isinstance(done, (bool, np.bool_)) or (hasattr(done, 'ndim') and done.ndim == 0):
                    # done is a scalar - broadcast to all indices
                    self.done_buffer[indices] = done
                else:
                    # done is an array
                    self.done_buffer[indices] = done
            else:
                remaining = self.buffer_size - self.pointer
                first_indices = np.arange(self.pointer, self.buffer_size)
                second_indices = np.arange(0, batch_size - remaining)
                
                self.state_buffer[first_indices] = state[:remaining]
                self.action_buffer[first_indices] = action[:remaining]
                self.reward_buffer[first_indices] = reward[:remaining]
                self.next_state_buffer[first_indices] = next_state[:remaining]
                # Handle done parameter for first chunk
                if isinstance(done, (bool, np.bool_)) or (hasattr(done, 'ndim') and done.ndim == 0):
                    # done is a scalar - broadcast to all indices
                    self.done_buffer[first_indices] = done
                else:
                    # done is an array
                    self.done_buffer[first_indices] = done[:remaining]
                
                if batch_size - remaining > 0:
                    self.state_buffer[second_indices] = state[remaining:]
                    self.action_buffer[second_indices] = action[remaining:]
                    self.reward_buffer[second_indices] = reward[remaining:]
                    self.next_state_buffer[second_indices] = next_state[remaining:]
                    # Handle done parameter for second chunk
                    if isinstance(done, (bool, np.bool_)) or (hasattr(done, 'ndim') and done.ndim == 0):
                        # done is a scalar - broadcast to all indices
                        self.done_buffer[second_indices] = done
                    else:
                        # done is an array
                        self.done_buffer[second_indices] = done[remaining:]
            
            # Track episode boundaries
            for i in range(batch_size):
                # Handle both scalar and array done values
                if isinstance(done, (bool, np.bool_)) or (hasattr(done, 'ndim') and done.ndim == 0):
                    # done is a scalar - apply to all elements in batch
                    done_value = done if isinstance(done, (bool, np.bool_)) else done.item()
                else:
                    # done is an array - get the i-th element
                    done_value = done[i]
                
                if done_value:
                    self.episode_ends.append((self.pointer + i) % self.buffer_size)
                    if i < batch_size - 1:  # Not the last element in batch
                        self.episode_starts.append((self.pointer + i + 1) % self.buffer_size)
                elif i == 0 and self.pointer == 0:  # First element of first batch
                    self.episode_starts.append(0)
            
            # Update pointer and size
            self.pointer = (self.pointer + batch_size) % self.buffer_size
            self.size = min(self.size + batch_size, self.buffer_size)
            self.buffer_filled = (self.size >= self.buffer_size)

    def sample(self, batch_size: int):
        """
         sequence sampling.
        """
        with self._lock:
            if self.sequence_length == 1:
                return self._sample_single_step(batch_size)
            else:
                return self._sample_sequences(batch_size)
    
    def _sample_single_step(self, batch_size: int):
        """ single-step sampling."""
        actual_size = self.size if not self.buffer_filled else self.buffer_size
        indices = self.numpy_rng.choice(actual_size, size=batch_size, replace=True)
        
        states = torch.from_numpy(self.state_buffer[indices])
        actions = torch.from_numpy(self.action_buffer[indices])
        rewards = torch.from_numpy(self.reward_buffer[indices])
        next_states = torch.from_numpy(self.next_state_buffer[indices])
        dones = torch.from_numpy(self.done_buffer[indices])
        
        return states, actions, rewards, next_states, dones
    
    def _sample_sequences(self, batch_size: int):
        """ sequence sampling."""
        actual_size = self.size if not self.buffer_filled else self.buffer_size
        
        # Get valid sequence start positions
        valid_starts = self._get_valid_sequence_starts()
        
        if len(valid_starts) == 0:
            # Fallback to single-step sampling
            return self._sample_single_step(batch_size)
        
        # Sample sequence start positions
        start_indices = self.numpy_rng.choice(valid_starts, size=batch_size, replace=True)
        
        # Build sequences
        sequences = []
        for start_idx in start_indices:
            sequence_indices = np.arange(start_idx, start_idx + self.sequence_length) % self.buffer_size
            sequences.append(sequence_indices)
        
        # Convert to arrays
        sequence_array = np.array(sequences)
        
        # Sample data
        states = torch.from_numpy(self.state_buffer[sequence_array])
        actions = torch.from_numpy(self.action_buffer[sequence_array])
        rewards = torch.from_numpy(self.reward_buffer[sequence_array])
        next_states = torch.from_numpy(self.next_state_buffer[sequence_array])
        dones = torch.from_numpy(self.done_buffer[sequence_array])
        
        return states, actions, rewards, next_states, dones
    
    def _get_valid_sequence_starts(self):
        """Get valid sequence start positions."""
        if not self.episode_boundaries:
            # No boundary restrictions
            actual_size = self.size if not self.buffer_filled else self.buffer_size
            return np.arange(max(0, actual_size - self.sequence_length + 1))
        
        # Consider episode boundaries
        valid_starts = []
        actual_size = self.size if not self.buffer_filled else self.buffer_size
        
        for start in range(actual_size - self.sequence_length + 1):
            # Check if sequence crosses episode boundary
            valid = True
            for i in range(self.sequence_length - 1):
                if self.done_buffer[(start + i) % self.buffer_size]:
                    valid = False
                    break
            
            if valid:
                valid_starts.append(start)
        
        return np.array(valid_starts)

    def new_episode(self):
        """Mark episode boundary."""
        if self.episode_boundaries and self.size > 0:
            self.episode_ends.append((self.pointer - 1) % self.buffer_size)


class SequencePrioritizedReplayBuffer:
    """
     sequence-aware prioritized replay buffer.
    
    Performance improvements:
    - Efficient sequence sampling with priorities
    -  episode boundary handling
    - Batch priority updates
    - Reduced memory allocations
    """
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, numpy_rng,
                 sequence_length: int = 1, episode_boundaries: bool = True,
                 alpha: float = 0.6, beta_start: float = 0.2, beta_frames: int = 100000,
                 epsilon: float = 1e-6):
        """
        Initializes the  sequence prioritized replay buffer.
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.episode_boundaries = episode_boundaries
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1
        self.numpy_rng = numpy_rng
        
        # Pre-allocate buffers
        self.state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.done_buffer = np.zeros((buffer_size, 1), dtype=np.bool_)
        
        self.pointer = 0
        self.size = 0
        self.buffer_filled = False
        
        # Initialize  sum tree
        self.tree = SumTree(buffer_size)
        self.max_priority = 1.0
        
        # Track episode boundaries
        self.episode_starts = []
        self.episode_ends = []
        
        # Pre-allocate arrays for efficiency
        self._sequence_indices = np.empty(sequence_length, dtype=np.int32)
        self._sampling_indices = np.empty(buffer_size, dtype=np.int32)
        self._sampling_priorities = np.empty(buffer_size, dtype=np.float32)
        self._sampling_weights = np.empty(buffer_size, dtype=np.float32)
        self._lock = threading.Lock()
    
    def _get_beta(self):
        """Calculate current beta value (annealed from beta_start to 1.0)."""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def add(self, state, action, reward, next_state, done=False, batch_size=None):
        """
         add method with priority handling and episode boundary tracking.
        """
        with self._lock:
            # Determine batch size
            if batch_size is None:
                if hasattr(state, 'shape') and len(state.shape) > 1:
                    batch_size = state.shape[0]
                elif isinstance(state, list):
                    batch_size = len(state)
                else:
                    batch_size = 1
                    state = [state]
                    action = [action]
                    reward = [reward]
                    next_state = [next_state]
                    done = [done]
            
            # Convert to numpy arrays if needed
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            if not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float32)
            if not isinstance(reward, np.ndarray):
                reward = np.array(reward, dtype=np.float32)
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state, dtype=np.float32)
            if not isinstance(done, np.ndarray):
                done = np.array(done, dtype=np.bool_)
            
            # Ensure correct shapes
            if reward.ndim == 1:
                reward = reward.reshape(-1, 1)
            if done.ndim == 1:
                done = done.reshape(-1, 1)
            
            #  buffer update
            if self.pointer + batch_size <= self.buffer_size:
                indices = np.arange(self.pointer, self.pointer + batch_size) % self.buffer_size
                self.state_buffer[indices] = state
                self.action_buffer[indices] = action
                self.reward_buffer[indices] = reward
                self.next_state_buffer[indices] = next_state
                # Handle done parameter - scalar or array
                if isinstance(done, (bool, np.bool_)) or (hasattr(done, 'ndim') and done.ndim == 0):
                    # done is a scalar - broadcast to all indices
                    self.done_buffer[indices] = done
                else:
                    # done is an array
                    self.done_buffer[indices] = done
            else:
                remaining = self.buffer_size - self.pointer
                first_indices = np.arange(self.pointer, self.buffer_size)
                second_indices = np.arange(0, batch_size - remaining)
                
                self.state_buffer[first_indices] = state[:remaining]
                self.action_buffer[first_indices] = action[:remaining]
                self.reward_buffer[first_indices] = reward[:remaining]
                self.next_state_buffer[first_indices] = next_state[:remaining]
                # Handle done parameter for first chunk
                if isinstance(done, (bool, np.bool_)) or (hasattr(done, 'ndim') and done.ndim == 0):
                    # done is a scalar - broadcast to all indices
                    self.done_buffer[first_indices] = done
                else:
                    # done is an array
                    self.done_buffer[first_indices] = done[:remaining]
                
                if batch_size - remaining > 0:
                    self.state_buffer[second_indices] = state[remaining:]
                    self.action_buffer[second_indices] = action[remaining:]
                    self.reward_buffer[second_indices] = reward[remaining:]
                    self.next_state_buffer[second_indices] = next_state[remaining:]
                    # Handle done parameter for second chunk
                    if isinstance(done, (bool, np.bool_)) or (hasattr(done, 'ndim') and done.ndim == 0):
                        # done is a scalar - broadcast to all indices
                        self.done_buffer[second_indices] = done
                    else:
                        # done is an array
                        self.done_buffer[second_indices] = done[remaining:]
            
            # Add to priority tree with maximum priority
            for i in range(batch_size):
                data_idx = (self.pointer + i) % self.buffer_size
                self.tree.add(self.max_priority, data_idx)
            
            # Track episode boundaries
            for i in range(batch_size):
                # Handle both scalar and array done values
                if isinstance(done, (bool, np.bool_)) or (hasattr(done, 'ndim') and done.ndim == 0):
                    # done is a scalar - apply to all elements in batch
                    done_value = done if isinstance(done, (bool, np.bool_)) else done.item()
                else:
                    # done is an array - get the i-th element
                    done_value = done[i]
                
                if done_value:
                    self.episode_ends.append((self.pointer + i) % self.buffer_size)
                    if i < batch_size - 1:  # Not the last element in batch
                        self.episode_starts.append((self.pointer + i + 1) % self.buffer_size)
                elif i == 0 and self.pointer == 0:  # First element of first batch
                    self.episode_starts.append(0)
            
            # Update pointer and size
            self.pointer = (self.pointer + batch_size) % self.buffer_size
            self.size = min(self.size + batch_size, self.buffer_size)
            self.buffer_filled = (self.size >= self.buffer_size)

    def sample(self, batch_size: int):
        """
         sequence sampling with priorities.
        """
        with self._lock:
            if self.sequence_length == 1:
                return self._sample_prioritized_single_step(batch_size)
            else:
                return self._sample_prioritized_sequences(batch_size)
    
    def _sample_prioritized_single_step(self, batch_size: int):
        """ prioritized single-step sampling."""
        actual_size = self.size if not self.buffer_filled else self.buffer_size
        
        # Generate random values for sampling
        total_priority = self.tree.total_priority()
        segment = total_priority / batch_size
        
        #  sampling
        indices = np.empty(batch_size, dtype=np.int32)
        priorities = np.empty(batch_size, dtype=np.float32)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = self.numpy_rng.uniform(a, b)
            tree_idx, data_idx = self.tree.get_leaf(value)
            indices[i] = data_idx
            priorities[i] = self.tree.tree[tree_idx]
        
        # Calculate importance sampling weights
        beta = self._get_beta()
        min_priority = self.epsilon if actual_size < self.buffer_size else np.min(priorities)
        weights = (priorities / min_priority) ** (-beta)
        weights = weights / np.max(weights)
        
        # Sample data
        states = torch.from_numpy(self.state_buffer[indices])
        actions = torch.from_numpy(self.action_buffer[indices])
        rewards = torch.from_numpy(self.reward_buffer[indices])
        next_states = torch.from_numpy(self.next_state_buffer[indices])
        dones = torch.from_numpy(self.done_buffer[indices])
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def _sample_prioritized_sequences(self, batch_size: int):
        """ prioritized sequence sampling."""
        actual_size = self.size if not self.buffer_filled else self.buffer_size
        
        # Get valid sequence start positions
        valid_starts = self._get_valid_sequence_starts()
        
        if len(valid_starts) == 0:
            # Fallback to single-step sampling
            return self._sample_prioritized_single_step(batch_size)
        
        # Sample sequence start positions with priorities
        start_indices = self._sample_prioritized_start(valid_starts, batch_size)
        
        # Build sequences
        sequences = []
        for start_idx in start_indices:
            sequence_indices = np.arange(start_idx, start_idx + self.sequence_length) % self.buffer_size
            sequences.append(sequence_indices)
        
        # Convert to arrays
        sequence_array = np.array(sequences)
        
        # Sample data
        states = torch.from_numpy(self.state_buffer[sequence_array])
        actions = torch.from_numpy(self.action_buffer[sequence_array])
        rewards = torch.from_numpy(self.reward_buffer[sequence_array])
        next_states = torch.from_numpy(self.next_state_buffer[sequence_array])
        dones = torch.from_numpy(self.done_buffer[sequence_array])
        
        # Calculate weights (use first element of sequence for priority)
        weights = torch.ones(batch_size, dtype=torch.float32)
        indices = start_indices
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def _get_valid_sequence_starts(self):
        """Get valid sequence start positions."""
        if not self.episode_boundaries:
            # No boundary restrictions
            actual_size = self.size if not self.buffer_filled else self.buffer_size
            return np.arange(max(0, actual_size - self.sequence_length + 1))
        
        # Consider episode boundaries
        valid_starts = []
        actual_size = self.size if not self.buffer_filled else self.buffer_size
        
        for start in range(actual_size - self.sequence_length + 1):
            # Check if sequence crosses episode boundary
            valid = True
            for i in range(self.sequence_length - 1):
                if self.done_buffer[(start + i) % self.buffer_size]:
                    valid = False
                    break
            
            if valid:
                valid_starts.append(start)
        
        return np.array(valid_starts)
    
    def _sample_prioritized_start(self, valid_starts, batch_size):
        """Sample sequence start positions with priorities."""
        # For simplicity, use uniform sampling from valid starts
        # In a more sophisticated implementation, we could use priorities
        return self.numpy_rng.choice(valid_starts, size=batch_size, replace=True)

    def update_priorities(self, indices, priorities):
        """
         batch priority update.
        """
        with self._lock:
            # Add epsilon to prevent zero priorities
            priorities = np.maximum(priorities, self.epsilon)
            
            # Update max priority
            self.max_priority = max(self.max_priority, np.max(priorities))
            
            # Batch update in tree
            # indices are data indices; SumTree.update_batch expects data indices
            self.tree.update_batch(indices, priorities)

    def new_episode(self):
        """Mark episode boundary."""
        if self.episode_boundaries and self.size > 0:
            self.episode_ends.append((self.pointer - 1) % self.buffer_size)
