import numpy as np
import torch

class ReplayBuffer:
    """
    A replay buffer for storing and sampling experiences in reinforcement learning.

    Parameters:
        buffer_size (int): Maximum size of the replay buffer.
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        numpy_rng (np.random.Generator): Random number generator for sampling.
    """
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, numpy_rng):
        """
        Initializes the replay buffer with specified size and dimensions.

        Parameters:
            buffer_size (int): Maximum size of the replay buffer.
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            numpy_rng (np.random.Generator): Random number generator for sampling.
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_buffer = np.zeros((buffer_size, state_dim))
        self.action_buffer = np.zeros((buffer_size, action_dim))
        self.reward_buffer = np.zeros((buffer_size, 1))
        self.next_state_buffer = np.zeros((buffer_size, state_dim))
        # self.done_buffer = np.zeros((buffer_size, 1))
        self.pointer = 0
        self.size = 0
        self.numpy_rng = numpy_rng
        self.buffer_filled = False

    def add(self, state, action, reward, next_state):
        """
        Adds a new experience to the replay buffer.

        Parameters:
            state (array-like): Current state.
            action (array-like): Action taken.
            reward (float): Reward received.
            next_state (array-like): Next state.
        """
        idx = self.pointer % self.buffer_size
        self.state_buffer[idx] = state
        self.action_buffer[idx] = action
        self.reward_buffer[idx] = reward
        self.next_state_buffer[idx] = next_state
        # self.done_buffer[idx] = done
        self.pointer += 1
        self.size = min(self.size + 1, self.buffer_size)
        self.buffer_filled = (self.pointer >= self.buffer_size)

    def add(self, state, action, reward, next_state, batch_size=None):
        """
        Adds experiences to the replay buffer. Supports both single experiences and batches.

        Parameters:
            state (array-like): Current state or batch of states.
            action (array-like): Action taken or batch of actions.
            reward (float or array-like): Reward received or batch of rewards.
            next_state (array-like): Next state or batch of states.
            batch_size (int, optional): Size of the batch. If None, inferred from inputs.
        """
        # Determine if we're handling a batch or single experience
        if batch_size is None:
            # Try to infer batch size from the shape of the inputs
            if hasattr(state, 'shape') and len(state.shape) > 1:
                batch_size = state.shape[0]
            elif isinstance(state, list):
                batch_size = len(state)
            else:
                # Single experience case
                batch_size = 1
                # If single items, convert to batches of size 1 for consistent handling
                if not hasattr(state, 'shape') or len(state.shape) == 1:
                    state = [state]
                    action = [action]
                    reward = [reward]
                    next_state = [next_state]

        # Convert to numpy arrays if needed and ensure correct shapes
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        if not isinstance(reward, np.ndarray):
            reward = np.array(reward)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)
        
        # Ensure rewards have the correct shape (batch_size, 1)
        if reward.ndim == 1:
            reward = reward.reshape(-1, 1)
        
        # Check if we have enough space in the buffer
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
        Samples a batch of experiences from the replay buffer.

        Parameters:
            batch_size (int): Size of the batch to sample.

        Returns:
            tuple: A tuple containing the sampled states, actions, rewards, and next states.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} experiences from buffer with only {self.size} experiences")
        
        indices = self.numpy_rng.choice(self.size, size = batch_size)
        return (
            torch.FloatTensor(self.state_buffer[indices]),
            torch.FloatTensor(self.action_buffer[indices]),
            torch.FloatTensor(self.reward_buffer[indices]),
            torch.FloatTensor(self.next_state_buffer[indices]),
            #torch.FloatTensor(self.done_buffer[indices]),
        )
    

    def new_episode(self):
        """
        Resets the replay buffer with specified size and dimensions.

        Parameters:
            buffer_size (int): New maximum size of the replay buffer.
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
        """
        self.state_buffer = np.zeros((self.buffer_size, self.state_dim))
        self.action_buffer = np.zeros((self.buffer_size, self.action_dim))
        self.reward_buffer = np.zeros((self.buffer_size, 1))
        self.next_state_buffer = np.zeros((self.buffer_size, self.state_dim))
        # self.done_buffer = np.zeros((buffer_size, 1))
        self.pointer = 0
        self.size = 0
        self.buffer_filled = False


class SumTree:
    """
    A sum tree data structure for efficient priority-based sampling.
    Used internally by the PrioritizedReplayBuffer.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0
        
    def add(self, priority, data_idx):
        """Add priority value to the tree."""
        tree_idx = data_idx + self.capacity - 1
        self.tree[tree_idx] = priority
        self._propagate(tree_idx)
        
    def _propagate(self, idx):
        """Propagate priority changes up the tree."""
        parent = (idx - 1) // 2
        if parent >= 0:
            self.tree[parent] = self.tree[2 * parent + 1] + self.tree[2 * parent + 2]
            self._propagate(parent)
    
    def get_leaf(self, value):
        """Retrieve a leaf node index based on priority value."""
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
                    
        data_idx = leaf_idx - self.capacity + 1
        return data_idx, self.tree[leaf_idx]
    
    def total_priority(self):
        """Return the total sum of all priorities."""
        return self.tree[0]
    
    def update(self, data_idx, priority):
        """Update priority for a specific data index."""
        tree_idx = data_idx + self.capacity - 1
        self.tree[tree_idx] = priority
        self._propagate(tree_idx)


class PrioritizedReplayBuffer:
    """
    A prioritized replay buffer for storing and sampling experiences in reinforcement learning.
    Implements Prioritized Experience Replay (PER) as described in Schaul et al. (2015).

    Parameters:
        buffer_size (int): Maximum size of the replay buffer.
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        numpy_rng (np.random.Generator): Random number generator for sampling.
        alpha (float): Prioritization exponent (0=uniform, 1=full prioritization).
        beta_start (float): Initial importance sampling weight exponent.
        beta_frames (int): Number of frames over which to anneal beta to 1.0.
        epsilon (float): Small constant to prevent zero priorities.
    """
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, numpy_rng,
                 alpha: float = 0.6, beta_start: float = 0.2, beta_frames: int = 100000,
                 epsilon: float = 1e-6):
        """
        Initializes the prioritized replay buffer with specified size and dimensions.
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1
        
        # Initialize buffers
        self.state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        
        self.pointer = 0
        self.size = 0
        self.numpy_rng = numpy_rng
        self.buffer_filled = False
        
        # Initialize sum tree for priority sampling
        self.tree = SumTree(buffer_size)
        self.max_priority = 1.0

    def _get_beta(self):
        """Calculate current beta value (annealed from beta_start to 1.0)."""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def add(self, state, action, reward, next_state, batch_size=None):
        """
        Adds experiences to the replay buffer with maximum priority.

        Parameters:
            state (array-like): Current state or batch of states.
            action (array-like): Action taken or batch of actions.
            reward (float or array-like): Reward received or batch of rewards.
            next_state (array-like): Next state or batch of states.
            batch_size (int, optional): Size of the batch. If None, inferred from inputs.
        """
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

        # Convert to numpy arrays if needed
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        if not isinstance(reward, np.ndarray):
            reward = np.array(reward).reshape(-1, 1)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)

        for i in range(batch_size):
            idx = self.pointer % self.buffer_size
            
            # Store experience
            self.state_buffer[idx] = state[i] if batch_size > 1 else state[0]
            self.action_buffer[idx] = action[i] if batch_size > 1 else action[0]
            self.reward_buffer[idx] = reward[i] if batch_size > 1 else reward[0]
            self.next_state_buffer[idx] = next_state[i] if batch_size > 1 else next_state[0]
            
            # Add to tree with maximum priority (for new experiences)
            priority = self.max_priority ** self.alpha
            self.tree.add(priority, idx)
            
            self.pointer += 1
            self.size = min(self.size + 1, self.buffer_size)
            
        self.buffer_filled = (self.size >= self.buffer_size)

    def sample(self, batch_size: int):
        """
        Samples a batch of experiences from the replay buffer using prioritized sampling.

        Parameters:
            batch_size (int): Size of the batch to sample.

        Returns:
            tuple: A tuple containing:
                - sampled states (torch.FloatTensor)
                - sampled actions (torch.FloatTensor) 
                - sampled rewards (torch.FloatTensor)
                - sampled next states (torch.FloatTensor)
                - importance sampling weights (torch.FloatTensor)
                - sample indices (np.ndarray) - needed for priority updates
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
            
        batch_indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        # Calculate priority segment size
        priority_segment = self.tree.total_priority() / batch_size
        
        beta = self._get_beta()
        
        for i in range(batch_size):
            # Sample from each priority segment
            low = priority_segment * i
            high = priority_segment * (i + 1)
            value = self.numpy_rng.uniform(low, high)
            
            # Get leaf node from sum tree
            idx, priority = self.tree.get_leaf(value)
            batch_indices[i] = idx
            priorities[i] = priority
        
        # Calculate importance sampling weights
        sampling_probabilities = priorities / self.tree.total_priority()
        # Avoid division by zero
        sampling_probabilities = np.maximum(sampling_probabilities, self.epsilon)
        
        weights = (self.size * sampling_probabilities) ** (-beta)
        weights = weights / weights.max()  # Normalize weights
        
        # Get experiences
        states = self.state_buffer[batch_indices]
        actions = self.action_buffer[batch_indices]
        rewards = self.reward_buffer[batch_indices]
        next_states = self.next_state_buffer[batch_indices]
        
        self.frame += 1
        
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(weights),
            batch_indices
        )

    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled experiences based on TD errors.

        Parameters:
            indices (array-like): Indices of experiences to update.
            priorities (array-like): New priority values (typically |TD error| + epsilon).
        """
        for idx, priority in zip(indices, priorities):
            # Clip priority and apply alpha exponent
            priority = max(priority, self.epsilon)
            priority = priority ** self.alpha
            
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def new_episode(self):
        """
        Resets the replay buffer.
        """
        self.state_buffer = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        
        self.pointer = 0
        self.size = 0
        self.buffer_filled = False
        self.tree = SumTree(self.buffer_size)
        self.max_priority = 1.0
        self.frame = 1


class SequenceReplayBuffer:
    """
    A sequence-aware replay buffer for RNN-based reinforcement learning.
    
    This buffer stores sequences of experiences and can sample consecutive sequences
    for RNN training while maintaining backward compatibility with single-step training.
    
    Parameters:
        buffer_size (int): Maximum size of the replay buffer.
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        numpy_rng (np.random.Generator): Random number generator for sampling.
        sequence_length (int): Length of sequences to sample for RNN training.
        episode_boundaries (bool): Whether to respect episode boundaries when sampling sequences.
    """
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, numpy_rng,
                 sequence_length: int = 2, episode_boundaries: bool = True):
        """
        Initializes the sequence-aware replay buffer.
        
        Parameters:
            buffer_size (int): Maximum size of the replay buffer.
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            numpy_rng (np.random.Generator): Random number generator for sampling.
            sequence_length (int): Length of sequences to sample for RNN training.
            episode_boundaries (bool): Whether to respect episode boundaries when sampling sequences.
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.episode_boundaries = episode_boundaries
        self.numpy_rng = numpy_rng
        
        # Initialize buffers
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
    
    def add(self, state, action, reward, next_state, done=False, batch_size=None):
        """
        Adds experiences to the replay buffer.
        
        Parameters:
            state (array-like): Current state or batch of states.
            action (array-like): Action taken or batch of actions.
            reward (float or array-like): Reward received or batch of rewards.
            next_state (array-like): Next state or batch of states.
            done (bool or array-like): Whether the episode ended.
            batch_size (int, optional): Size of the batch. If None, inferred from inputs.
        """
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
                done = [done]
        
        # Convert to numpy arrays if needed
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        if not isinstance(reward, np.ndarray):
            reward = np.array(reward).reshape(-1, 1)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)
        if not isinstance(done, np.ndarray):
            # Handle both single boolean and array cases
            if isinstance(done, (bool, np.bool_)):
                done = np.array([done] * batch_size).reshape(-1, 1)
            else:
                done = np.array(done).reshape(-1, 1)
        
        # Store experiences
        for i in range(batch_size):
            idx = self.pointer % self.buffer_size
            
            # Store experience
            self.state_buffer[idx] = state[i] if batch_size > 1 else state[0]
            self.action_buffer[idx] = action[i] if batch_size > 1 else action[0]
            self.reward_buffer[idx] = reward[i] if batch_size > 1 else reward[0]
            self.next_state_buffer[idx] = next_state[i] if batch_size > 1 else next_state[0]
            self.done_buffer[idx] = done[i] if batch_size > 1 else done[0]
            
            # Track episode boundaries
            if self.episode_boundaries and (done[i] if batch_size > 1 else done[0]):
                self.episode_ends.append(idx)
                if self.pointer + 1 < self.buffer_size:
                    self.episode_starts.append((self.pointer + 1) % self.buffer_size)
                else:
                    self.episode_starts.append(0)
            
            self.pointer += 1
            self.size = min(self.size + 1, self.buffer_size)
        
        self.buffer_filled = (self.size >= self.buffer_size)
    
    def sample(self, batch_size: int):
        """
        Samples a batch of experiences from the replay buffer.
        
        For sequence_length > 1, samples consecutive sequences.
        For sequence_length = 1, behaves like a standard replay buffer.
        
        Parameters:
            batch_size (int): Size of the batch to sample.
            
        Returns:
            tuple: A tuple containing the sampled states, actions, rewards, next states, and dones.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        if self.sequence_length == 1:
            # Standard single-step sampling
            return self._sample_single_step(batch_size)
        else:
            # Sequence sampling for RNN training
            return self._sample_sequences(batch_size)
    
    def _sample_single_step(self, batch_size: int):
        """Sample individual transitions (backward compatibility)."""
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} experiences from buffer with only {self.size} experiences")
        
        indices = self.numpy_rng.choice(self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state_buffer[indices]),
            torch.FloatTensor(self.action_buffer[indices]),
            torch.FloatTensor(self.reward_buffer[indices]),
            torch.FloatTensor(self.next_state_buffer[indices]),
            torch.BoolTensor(self.done_buffer[indices])
        )
    
    def _sample_sequences(self, batch_size: int):
        """Sample sequences of consecutive transitions for RNN training."""
        if self.sequence_length > self.size:
            raise ValueError(f"Cannot sample sequences of length {self.sequence_length} from buffer with only {self.size} experiences")
        
        sequences = []
        sequence_indices = []
        
        for _ in range(batch_size):
            # Find valid sequence start positions
            valid_starts = self._get_valid_sequence_starts()
            
            if len(valid_starts) == 0:
                # Fallback to single-step sampling if no valid sequences
                return self._sample_single_step(batch_size)
            
            # Sample a random start position
            start_idx = self.numpy_rng.choice(valid_starts)
            
            # Get consecutive sequence
            sequence_idx = np.arange(start_idx, start_idx + self.sequence_length) % self.buffer_size
            sequence_indices.append(sequence_idx)
            
            # Extract sequence data
            states = self.state_buffer[sequence_idx]
            actions = self.action_buffer[sequence_idx]
            rewards = self.reward_buffer[sequence_idx]
            next_states = self.next_state_buffer[sequence_idx]
            dones = self.done_buffer[sequence_idx]
            
            sequences.append({
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_states,
                'dones': dones
            })
        
        # Convert to tensors with proper shape for RNN input
        # Shape: (batch_size, sequence_length, feature_dim)
        states = torch.FloatTensor(np.array([seq['states'] for seq in sequences]))
        actions = torch.FloatTensor(np.array([seq['actions'] for seq in sequences]))
        rewards = torch.FloatTensor(np.array([seq['rewards'] for seq in sequences]))
        next_states = torch.FloatTensor(np.array([seq['next_states'] for seq in sequences]))
        dones = torch.BoolTensor(np.array([seq['dones'] for seq in sequences]))
        
        return states, actions, rewards, next_states, dones
    
    def _get_valid_sequence_starts(self):
        """Get valid starting positions for sequences that don't cross episode boundaries."""
        if not self.episode_boundaries or len(self.episode_ends) == 0:
            # If no episode boundaries or no episodes ended, all positions are valid
            return list(range(self.size))
        
        valid_starts = []
        
        # Check each possible start position
        for start in range(self.size):
            end = (start + self.sequence_length - 1) % self.buffer_size
            
            # Check if sequence crosses episode boundary
            crosses_boundary = False
            for episode_end in self.episode_ends:
                if start <= episode_end < end or (end < start and (start <= episode_end or episode_end < end)):
                    crosses_boundary = True
                    break
            
            if not crosses_boundary:
                valid_starts.append(start)
        
        return valid_starts
    
    def new_episode(self):
        """Reset the replay buffer."""
        self.state_buffer = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.done_buffer = np.zeros((self.buffer_size, 1), dtype=np.bool_)
        
        self.pointer = 0
        self.size = 0
        self.buffer_filled = False
        self.episode_starts = []
        self.episode_ends = []


class SequencePrioritizedReplayBuffer:
    """
    A sequence-aware prioritized replay buffer for RNN-based reinforcement learning.
    
    This buffer combines sequence sampling with prioritized experience replay,
    maintaining backward compatibility with single-step training.
    
    Parameters:
        buffer_size (int): Maximum size of the replay buffer.
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        numpy_rng (np.random.Generator): Random number generator for sampling.
        sequence_length (int): Length of sequences to sample for RNN training.
        episode_boundaries (bool): Whether to respect episode boundaries when sampling sequences.
        alpha (float): Prioritization exponent (0=uniform, 1=full prioritization).
        beta_start (float): Initial importance sampling weight exponent.
        beta_frames (int): Number of frames over which to anneal beta to 1.0.
        epsilon (float): Small constant to prevent zero priorities.
    """
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, numpy_rng,
                 sequence_length: int = 1, episode_boundaries: bool = True,
                 alpha: float = 0.6, beta_start: float = 0.2, beta_frames: int = 100000,
                 epsilon: float = 1e-6):
        """
        Initializes the sequence-aware prioritized replay buffer.
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
        
        # Initialize buffers
        self.state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.done_buffer = np.zeros((buffer_size, 1), dtype=np.bool_)
        
        self.pointer = 0
        self.size = 0
        self.buffer_filled = False
        
        # Initialize sum tree for priority sampling
        self.tree = SumTree(buffer_size)
        self.max_priority = 1.0
        
        # Track episode boundaries
        self.episode_starts = []
        self.episode_ends = []
    
    def _get_beta(self):
        """Calculate current beta value (annealed from beta_start to 1.0)."""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def add(self, state, action, reward, next_state, done=False, batch_size=None):
        """
        Adds experiences to the replay buffer with maximum priority.
        """
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
                done = [done]
        
        # Convert to numpy arrays if needed
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        if not isinstance(reward, np.ndarray):
            reward = np.array(reward).reshape(-1, 1)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)
        if not isinstance(done, np.ndarray):
            # Handle both single boolean and array cases
            if isinstance(done, (bool, np.bool_)):
                done = np.array([done] * batch_size).reshape(-1, 1)
            else:
                done = np.array(done).reshape(-1, 1)

        for i in range(batch_size):
            idx = self.pointer % self.buffer_size
            
            # Store experience
            self.state_buffer[idx] = state[i] if batch_size > 1 else state[0]
            self.action_buffer[idx] = action[i] if batch_size > 1 else action[0]
            self.reward_buffer[idx] = reward[i] if batch_size > 1 else reward[0]
            self.next_state_buffer[idx] = next_state[i] if batch_size > 1 else next_state[0]
            self.done_buffer[idx] = done[i] if batch_size > 1 else done[0]
            
            # Add to tree with maximum priority (for new experiences)
            priority = self.max_priority ** self.alpha
            self.tree.add(priority, idx)
            
            # Track episode boundaries
            if self.episode_boundaries and (done[i] if batch_size > 1 else done[0]):
                self.episode_ends.append(idx)
                if self.pointer + 1 < self.buffer_size:
                    self.episode_starts.append((self.pointer + 1) % self.buffer_size)
                else:
                    self.episode_starts.append(0)
            
            self.pointer += 1
            self.size = min(self.size + 1, self.buffer_size)
        
        self.buffer_filled = (self.size >= self.buffer_size)
    
    def sample(self, batch_size: int):
        """
        Samples a batch of experiences from the replay buffer using prioritized sampling.
        
        For sequence_length > 1, samples consecutive sequences.
        For sequence_length = 1, behaves like a standard prioritized replay buffer.
        
        Returns:
            tuple: A tuple containing:
                - sampled states (torch.FloatTensor)
                - sampled actions (torch.FloatTensor) 
                - sampled rewards (torch.FloatTensor)
                - sampled next states (torch.FloatTensor)
                - sampled dones (torch.BoolTensor)
                - importance sampling weights (torch.FloatTensor)
                - sample indices (np.ndarray) - needed for priority updates
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        if self.sequence_length == 1:
            # Standard prioritized single-step sampling
            return self._sample_prioritized_single_step(batch_size)
        else:
            # Sequence sampling for RNN training
            return self._sample_prioritized_sequences(batch_size)
    
    def _sample_prioritized_single_step(self, batch_size: int):
        """Sample individual transitions with prioritization (backward compatibility)."""
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} experiences from buffer with only {self.size} experiences")
        
        batch_indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        # Calculate priority segment size
        priority_segment = self.tree.total_priority() / batch_size
        
        beta = self._get_beta()
        
        for i in range(batch_size):
            # Sample from each priority segment
            low = priority_segment * i
            high = priority_segment * (i + 1)
            value = self.numpy_rng.uniform(low, high)
            
            # Get leaf node from sum tree
            idx, priority = self.tree.get_leaf(value)
            batch_indices[i] = idx
            priorities[i] = priority
        
        # Calculate importance sampling weights
        sampling_probabilities = priorities / self.tree.total_priority()
        sampling_probabilities = np.maximum(sampling_probabilities, self.epsilon)
        
        weights = (self.size * sampling_probabilities) ** (-beta)
        weights = weights / weights.max()  # Normalize weights
        
        # Get experiences
        states = self.state_buffer[batch_indices]
        actions = self.action_buffer[batch_indices]
        rewards = self.reward_buffer[batch_indices]
        next_states = self.next_state_buffer[batch_indices]
        dones = self.done_buffer[batch_indices]
        
        self.frame += 1
        
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.BoolTensor(dones),
            torch.FloatTensor(weights),
            batch_indices
        )
    
    def _sample_prioritized_sequences(self, batch_size: int):
        """Sample sequences with prioritization for RNN training."""
        if self.sequence_length > self.size:
            raise ValueError(f"Cannot sample sequences of length {self.sequence_length} from buffer with only {self.size} experiences")
        
        sequences = []
        sequence_indices = []
        all_weights = []
        all_priorities = []
        
        for _ in range(batch_size):
            # Find valid sequence start positions
            valid_starts = self._get_valid_sequence_starts()
            
            if len(valid_starts) == 0:
                # Fallback to single-step sampling if no valid sequences
                return self._sample_prioritized_single_step(batch_size)
            
            # Sample a start position based on priorities
            start_idx = self._sample_prioritized_start(valid_starts)
            
            # Get consecutive sequence
            sequence_idx = np.arange(start_idx, start_idx + self.sequence_length) % self.buffer_size
            sequence_indices.append(sequence_idx)
            
            # Calculate priority for the sequence (average of individual priorities)
            sequence_priorities = []
            for idx in sequence_idx:
                tree_idx = idx + self.buffer_size - 1
                if tree_idx < len(self.tree.tree):
                    sequence_priorities.append(self.tree.tree[tree_idx])
                else:
                    sequence_priorities.append(self.max_priority)
            
            avg_priority = np.mean(sequence_priorities)
            all_priorities.append(avg_priority)
            
            # Extract sequence data
            states = self.state_buffer[sequence_idx]
            actions = self.action_buffer[sequence_idx]
            rewards = self.reward_buffer[sequence_idx]
            next_states = self.next_state_buffer[sequence_idx]
            dones = self.done_buffer[sequence_idx]
            
            sequences.append({
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_states,
                'dones': dones
            })
        
        # Calculate importance sampling weights
        beta = self._get_beta()
        total_priority = self.tree.total_priority()
        sampling_probabilities = np.array(all_priorities) / total_priority
        sampling_probabilities = np.maximum(sampling_probabilities, self.epsilon)
        
        weights = (self.size * sampling_probabilities) ** (-beta)
        weights = weights / weights.max()  # Normalize weights
        
        # Convert to tensors with proper shape for RNN input
        states = torch.FloatTensor(np.array([seq['states'] for seq in sequences]))
        actions = torch.FloatTensor(np.array([seq['actions'] for seq in sequences]))
        rewards = torch.FloatTensor(np.array([seq['rewards'] for seq in sequences]))
        next_states = torch.FloatTensor(np.array([seq['next_states'] for seq in sequences]))
        dones = torch.BoolTensor(np.array([seq['dones'] for seq in sequences]))
        
        self.frame += 1
        
        return states, actions, rewards, next_states, dones, torch.FloatTensor(weights), sequence_indices
    
    def _get_valid_sequence_starts(self):
        """Get valid starting positions for sequences that don't cross episode boundaries."""
        if not self.episode_boundaries or len(self.episode_ends) == 0:
            return list(range(self.size))
        
        valid_starts = []
        for start in range(self.size):
            end = (start + self.sequence_length - 1) % self.buffer_size
            
            crosses_boundary = False
            for episode_end in self.episode_ends:
                if start <= episode_end < end or (end < start and (start <= episode_end or episode_end < end)):
                    crosses_boundary = True
                    break
            
            if not crosses_boundary:
                valid_starts.append(start)
        
        return valid_starts
    
    def _sample_prioritized_start(self, valid_starts):
        """Sample a start position from valid starts based on priorities."""
        if len(valid_starts) == 1:
            return valid_starts[0]
        
        # Calculate priorities for valid starts
        start_priorities = []
        for start in valid_starts:
            tree_idx = start + self.buffer_size - 1
            if tree_idx < len(self.tree.tree):
                start_priorities.append(self.tree.tree[tree_idx])
            else:
                start_priorities.append(self.max_priority)
        
        # Convert to probabilities
        start_priorities = np.array(start_priorities)
        probabilities = start_priorities / start_priorities.sum()
        
        # Sample based on probabilities
        return self.numpy_rng.choice(valid_starts, p=probabilities)
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled experiences based on TD errors.
        
        Parameters:
            indices (array-like): Indices of experiences to update.
            priorities (array-like): New priority values (typically |TD error| + epsilon).
        """
        if isinstance(indices, list) and len(indices) > 0 and isinstance(indices[0], np.ndarray):
            # Handle sequence indices (list of arrays)
            for sequence_idx in indices:
                for idx in sequence_idx:
                    priority = max(priorities.mean(), self.epsilon)  # Use average priority for sequence
                    priority = priority ** self.alpha
                    self.tree.update(idx, priority)
                    self.max_priority = max(self.max_priority, priority)
        else:
            # Handle single indices
            for idx, priority in zip(indices, priorities):
                priority = max(priority, self.epsilon)
                priority = priority ** self.alpha
                self.tree.update(idx, priority)
                self.max_priority = max(self.max_priority, priority)
    
    def new_episode(self):
        """Reset the replay buffer."""
        self.state_buffer = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.done_buffer = np.zeros((self.buffer_size, 1), dtype=np.bool_)
        
        self.pointer = 0
        self.size = 0
        self.buffer_filled = False
        self.tree = SumTree(self.buffer_size)
        self.max_priority = 1.0
        self.frame = 1
        self.episode_starts = []
        self.episode_ends = []