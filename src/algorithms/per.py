import numpy as np
import torch

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