import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum
from collections import deque

class DifficultyLevel(IntEnum):
    """Enumeration for difficulty levels."""
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4
    LEVEL_5 = 5

class Outcome(IntEnum):
    """Enumeration for episode outcomes."""
    SUCCESS = 1
    FAILURE = 2
    SEVERE_FAILURE = 3

@dataclass
class DifficultyConfig:
    """Configuration for a specific difficulty level."""
    level: DifficultyLevel
    grid_limits: np.ndarray  # Shape: (2, 2) for [[x_min, x_max], [y_min, y_max]]
    angle_is_max: bool  # True if angle is maximum, False if minimum
    angle_value: float  # The angle constraint value
    fully_random: bool  # True for fully random generation within limits
    new_min_distance_between_eavesdropper_and_users: float # Minimal distance to ensure between an eavesdropper and the users
    new_max_distance_between_eavesdropper_and_users: float # Maximum distance to ensure between an eavesdropper and the closest user
    
@dataclass
class EpisodeRecord:
    """Record of a single episode."""
    difficulty_level: DifficultyLevel
    outcome: Outcome
    episode_id: int

class TaskManager:
    """
    Manages difficulty progression for episode generation.
    Includes buffer-based statistics and adaptive difficulty management.
    """
    
    def __init__(
        self,
        num_users: int,
        user_limits, 
        RIS_position,
        num_steps_per_episode: int,
        downlink_uplink_eavesdropper_bools: list,
        thresholds: np.ndarray, # ? to rename 
        difficulty_configs: Optional[Dict[DifficultyLevel, DifficultyConfig]] = None,
        Buffer_Size: int = 10,  # Buffer size
        H: float = 4.0,  # Multiplier for episode threshold
        num_environments: int = 1,
        random_seed: Optional[int] = None
    ):
        """
        Initialize TaskManager.
        
        Args:
            num_steps_per_episode: Number of steps for a single episode
            downlink_uplink_eavesdropper_bools: List of booleans to know if downlink, uplink and eavesdroppers are respectivly considered
            thresholds: Thresholds that will be used for computing success for an episode based on downlink, uplink and total eavesdropped signal per user.
            difficulty_configs: Custom difficulty configurations
            Buffer_Size: Buffer size for episode statistics
            H: Multiplier for episode threshold before difficulty reduction
            num_environments: Number of parallel environments
            random_seed: Random seed for reproducibility
        """
        self.num_users = num_users
        self.num_steps_per_episode = num_steps_per_episode
        self.user_limits = user_limits 
        self.RIS_position = RIS_position
        self._is_downlink_used = downlink_uplink_eavesdropper_bools[0]
        self._is_uplink_used = downlink_uplink_eavesdropper_bools[1]
        self._are_eavesdroppers_used = downlink_uplink_eavesdropper_bools[2]
        self.thresholds = thresholds
        self.Buffer_Size = Buffer_Size
        self.H = H
        self.num_environments = num_environments
        self.rng = np.random.default_rng(random_seed)
        
        self.min_angle = np.arctan((self.RIS_position[1]-self.user_limits[1][1])/(self.user_limits[0][1]-self.RIS_position[0]))

        self.max_angle = np.arctan((self.RIS_position[1]-self.user_limits[1][0])/(self.user_limits[0][0]-self.RIS_position[0]))

        
        self.difference_angle = self.max_angle - self.min_angle
        
        # Initialize difficulty configurations
        self.difficulty_configs = difficulty_configs or self._create_default_configs()
        
        # Current state
        self.current_max_level = DifficultyLevel.LEVEL_1
        self.current_episode = 0
        self.episodes_used_current_level = 0
        
        # Buffer for episode statistics (FIFO)
        self.episode_buffer = deque(maxlen=Buffer_Size)
        
    def _create_default_configs(self) -> Dict[DifficultyLevel, DifficultyConfig]:
        """Create default difficulty configurations."""
        return {
            DifficultyLevel.LEVEL_1: DifficultyConfig(
                level=DifficultyLevel.LEVEL_1,
                grid_limits=np.array([[120, 200], [40,100]]),
                angle_is_max=False,
                angle_value=self.difference_angle/(2*self.num_users),  #
                fully_random=False,
                new_min_distance_between_eavesdropper_and_users = 25,
                new_max_distance_between_eavesdropper_and_users = np.inf
            ),
            DifficultyLevel.LEVEL_2: DifficultyConfig(
                level=DifficultyLevel.LEVEL_2,
                grid_limits=np.array([[100, 200], [0, 100]]),
                angle_is_max=False,
                angle_value=self.difference_angle/(4*self.num_users),  # 
                fully_random=False,
                new_min_distance_between_eavesdropper_and_users = 10,
                new_max_distance_between_eavesdropper_and_users = 50
            ),
            DifficultyLevel.LEVEL_3: DifficultyConfig(
                level=DifficultyLevel.LEVEL_3,
                grid_limits=np.array([[100, 200], [0, 100]]),
                angle_is_max=False,
                angle_value=self.difference_angle/(6 * self.num_users),  # 
                fully_random=False,
                new_min_distance_between_eavesdropper_and_users = 5,
                new_max_distance_between_eavesdropper_and_users = 30
            ),
            DifficultyLevel.LEVEL_4: DifficultyConfig(
                level=DifficultyLevel.LEVEL_4,
                grid_limits=np.array([[100, 200], [0, 100]]),
                angle_is_max=True,
                angle_value=self.difference_angle/(6 * self.num_users),  #
                fully_random=False,
                new_min_distance_between_eavesdropper_and_users = 2,
                new_max_distance_between_eavesdropper_and_users = 15
            ),
            DifficultyLevel.LEVEL_5: DifficultyConfig(
                level=DifficultyLevel.LEVEL_5,
                grid_limits=np.array([[100, 200], [0, 100]]),
                angle_is_max=True,
                angle_value=np.pi/12,  # 15 degrees max
                fully_random=True, # * max grid generation
                new_min_distance_between_eavesdropper_and_users = 0.5,
                new_max_distance_between_eavesdropper_and_users = np.inf
            )
        }
    
    def _get_level_probabilities(self, n: int) -> Dict[DifficultyLevel, float]:
        """Calculate probability distribution for difficulty level n."""
        probabilities = {}
        
        if n == 1:
            # Only level 1 available
            probabilities[DifficultyLevel.LEVEL_1] = 1.0
        elif n == 2:
            # Special case for n=2
            probabilities[DifficultyLevel.LEVEL_2] = 0.8
            probabilities[DifficultyLevel.LEVEL_1] = 0.2
        else:
            # General case for n >= 3
            probabilities[DifficultyLevel(n)] = 0.65  # Current level
            probabilities[DifficultyLevel(n-1)] = 0.20  # Previous level
            
            # Distribute remaining 0.15 among levels 1 to n-2 using dict comprehension
            remaining_prob = 0.15 / (n - 2)
            probabilities.update({
                DifficultyLevel(level): remaining_prob 
                for level in range(1, n - 1)
            })
        return probabilities
    
    def _sample_difficulty_levels(self, num_samples: int) -> np.ndarray:
        """Sample multiple difficulty levels at once based on current probabilities."""
        probabilities = self._get_level_probabilities(self.current_max_level)
        levels = np.array(list(probabilities.keys()), dtype=int)
        probs = np.array(list(probabilities.values()))
        
        selected_levels = self.rng.choice(levels, size=num_samples, p=probs)
        return selected_levels
    
    def generate_episode_configs(self) -> List[Tuple[np.ndarray, bool, float, bool]]:
        """
        Generate configurations for all environments.
        
        Returns:
            List of tuples, one per environment, each containing:
            (grid_limits, angle_is_max, angle_value, fully_random)
        """
        # Sample all difficulty levels at once
        self.selected_levels = self._sample_difficulty_levels(self.num_environments)
        
        # Vectorized config generation
        configs = [
            (
                self.difficulty_configs[DifficultyLevel(level)].grid_limits.copy(),
                self.difficulty_configs[DifficultyLevel(level)].angle_is_max,
                self.difficulty_configs[DifficultyLevel(level)].angle_value,
                self.difficulty_configs[DifficultyLevel(level)].fully_random,
                self.difficulty_configs[DifficultyLevel(level)].new_min_distance_between_eavesdropper_and_users,
                self.difficulty_configs[DifficultyLevel(level)].new_max_distance_between_eavesdropper_and_users,
            )
            for level in self.selected_levels
        ]
        
        self.current_episode += 1
        return configs
    
    def update_episode_outcomes(self, outcomes: List[Outcome]):
        """Update buffer with multiple episode outcomes from parallel environments."""

        final_outcomes = list(zip(self.selected_levels, outcomes))
        records = [
            EpisodeRecord(
                difficulty_level=difficulty_level,
                outcome=outcome,
                episode_id=self.current_episode
            )
            for difficulty_level, outcome in final_outcomes
        ]
        
        # Extend buffer with all records at once
        self.episode_buffer.extend(records)

        self.episodes_used_current_level += self.num_environments
        
        # Check if we need to advance difficulty or reduce it
        self._check_difficulty_progression()
    
    def _check_difficulty_progression(self):
        """Check if difficulty should be advanced or reduced."""
        # Check for difficulty advancement
        if self._is_difficulty_solved():
            if self.current_max_level < DifficultyLevel.LEVEL_5:
                self.current_max_level = DifficultyLevel(self.current_max_level + 1)
                self._reset_buffer_for_new_level()
        
        # Check for difficulty reduction
        elif self._should_reduce_difficulty():
            if self.current_max_level > DifficultyLevel.LEVEL_1:
                self.current_max_level = DifficultyLevel(self.current_max_level - 1)
                self._reset_buffer_for_new_level()
    


    def _is_difficulty_solved(self) -> bool:
        """Check if current difficulty level has been solved."""
        if len(self.episode_buffer) < self.Buffer_Size:
            return False
        
        # Calculate success and severe failure rates

        # Calculate success rate in buffer
            
        success_count = sum(1 for record in self.episode_buffer if record.outcome == Outcome.SUCCESS)
        severe_failure_count = sum(1 for record in self.episode_buffer if record.outcome == Outcome.SEVERE_FAILURE)
        
        success_rate = success_count / len(self.episode_buffer)
        severe_failure_rate = severe_failure_count / len(self.episode_buffer)
        
        return success_rate > 0.9 and severe_failure_rate < 0.005
    
    def _should_reduce_difficulty(self) -> bool:
        """Check if difficulty should be reduced."""
        if self.episodes_used_current_level < self.H * self.Buffer_Size:
            return False
        
        if len(self.episode_buffer) == 0:
            return False
        
        # Calculate success rate in buffer
        success_count = sum(1 for record in self.episode_buffer if record.outcome == Outcome.SUCCESS)
        success_rate = success_count / len(self.episode_buffer)
        
        return success_rate < 0.4
    


    def compute_episodes_outcome(self, downlink_sum = None, uplink_sum = None , best_eavesdropper_sum = None) -> Outcome:
        """
        Backward-compat wrapper kept for signature compatibility. Use the robust
        implementation below. This alias forwards to the main implementation.
        """
        return self._compute_episodes_outcome_impl(
            downlink_sum=downlink_sum,
            uplink_sum=uplink_sum,
            best_eavesdropper_sum=best_eavesdropper_sum
        )



    def _compute_episodes_outcome_impl(self,
                                 downlink_sum=None, uplink_sum=None, best_eavesdropper_sum=None) -> Outcome:
        """
        Compute the outcome of an episode based on several arrays and specific conditions.

        Args:
            downlink_sum: Numpy array or list of downlink sums.
            uplink_sum: Numpy array or list of uplink sums.
            best_eavesdropper_sum: (Optional) Numpy array or list for eavesdropping sums.

        Returns:
            List of outcomes per episode.
        """
        def compute_condition_result(values, threshold, num_users, for_eavesdropper=False):
            """Helper to compute condition results."""
            values = np.asarray(values)
            # Normalize to (batch, users) by flattening all non-batch dims
            if values.ndim == 0:
                # Single scalar → treat as one episode, one user
                values2d = values.reshape(1, 1)
            elif values.ndim == 1:
                # (users,) → (1, users)
                values2d = values.reshape(1, -1)
            else:
                # (batch, ...users_dims...) → (batch, users)
                batch = values.shape[0]
                users = int(np.prod(values.shape[1:]))
                values2d = values.reshape(batch, users)

            # Sum condition over users axis
            if for_eavesdropper:
                condition = np.sum(values2d < threshold, axis=1) / num_users
            else:
                condition = np.sum(values2d > threshold, axis=1) / num_users
            return np.where(
                condition == 1, Outcome.SUCCESS.value,
                np.where(condition >= 0.51, Outcome.FAILURE.value, Outcome.SEVERE_FAILURE.value)
            )

        # Precompute meaned arrays if relevant
        downlink_meaned = (
            np.array(downlink_sum) / self.num_steps_per_episode 
            if self._is_downlink_used else None
        )
        uplink_meaned = (
            np.array(uplink_sum) / self.num_steps_per_episode 
            if self._is_uplink_used else None
        )

        results = []

        if self._is_downlink_used:
            downlink_result = compute_condition_result(downlink_meaned, self.thresholds[0], self.num_users)
            results.append(downlink_result)

        if self._is_uplink_used:
            uplink_result = compute_condition_result(uplink_meaned, self.thresholds[1], self.num_users)
            results.append(uplink_result)

        if self._are_eavesdroppers_used:
            if self._is_downlink_used:
                downlink_eavesdropper_result = compute_condition_result(
                    downlink_meaned, self.thresholds[0], self.num_users, for_eavesdropper=True
                )
                results.append(downlink_eavesdropper_result)
            
            if self._is_uplink_used:
                uplink_eavesdropper_result = compute_condition_result(
                    uplink_meaned, self.thresholds[1], self.num_users, for_eavesdropper=True
                )
                results.append(uplink_eavesdropper_result)

        if not results:
            return []
        # Combine all results by taking the maximum severity per episode
        # Normalize each to 1D then stack
        flat_results = [np.asarray(r).reshape(-1) for r in results]
        outcomes = np.max(np.vstack(flat_results), axis=0)
        return [Outcome(int(o)) for o in outcomes.tolist()]


    def _reset_buffer_for_new_level(self):
        """Reset buffer and counters for new difficulty level."""
        self.episode_buffer.clear()
        self.episodes_used_current_level = 0
    
    def get_buffer_statistics(self) -> Dict:
        """Get statistics from the current buffer."""
        if len(self.episode_buffer) == 0:
            return {
                'buffer_size': 0,
                'success_rate': 0.0,
                'failure_rate': 0.0,
                'severe_failure_rate': 0.0,
                'level_distribution': {}
            }
        
        # Count outcomes
        success_count = sum(1 for record in self.episode_buffer if record.outcome == Outcome.SUCCESS)
        failure_count = sum(1 for record in self.episode_buffer if record.outcome == Outcome.FAILURE)
        severe_failure_count = sum(1 for record in self.episode_buffer if record.outcome == Outcome.SEVERE_FAILURE)
        
        buffer_size = len(self.episode_buffer)
        
        # Count level distribution
        level_counts = {}
        for record in self.episode_buffer:
            level = record.difficulty_level
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            'buffer_size': buffer_size,
            'success_rate': success_count / buffer_size,
            'failure_rate': failure_count / buffer_size,
            'severe_failure_rate': severe_failure_count / buffer_size,
            'level_distribution': level_counts,
            'is_buffer_full': buffer_size == self.Buffer_Size
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the task manager."""
        buffer_stats = self.get_buffer_statistics()
        
        stats = {
            'current_episode': self.current_episode,
            'current_max_level': self.current_max_level,
            'episodes_used_current_level': self.episodes_used_current_level,
            'level_probabilities': self._get_level_probabilities(self.current_max_level),
            'buffer_stats': buffer_stats,
            'difficulty_solved': self._is_difficulty_solved(),
            'should_reduce_difficulty': self._should_reduce_difficulty(),
            'episodes_threshold_for_reduction': self.H * self.Buffer_Size
        }
        return stats
    
    def reset(self):
        """Reset the task manager to initial state."""
        self.current_max_level = DifficultyLevel.LEVEL_1
        self.current_episode = 0
        self.episodes_used_current_level = 0
        self.episode_buffer.clear()
    
    def save_state(self) -> Dict:
        """Save current state for persistence."""
        return {
            'current_max_level': int(self.current_max_level),
            'current_episode': self.current_episode,
            'episodes_used_current_level': self.episodes_used_current_level,
            'episode_buffer': [(int(r.difficulty_level), int(r.outcome), r.episode_id) for r in self.episode_buffer],
            'Buffer_Size': self.Buffer_Size,
            'H': self.H,
            'num_environments': self.num_environments
        }
    
    def load_state(self, state: Dict):
        """Load previously saved state."""
        self.current_max_level = DifficultyLevel(state['current_max_level'])
        self.current_episode = state['current_episode']
        self.episodes_used_current_level = state['episodes_used_current_level']
        self.Buffer_Size = state['Buffer_Size']
        self.H = state['H']
        self.num_environments = state['num_environments']
        
        # Reconstruct episode buffer
        self.episode_buffer = deque(maxlen=self.Buffer_Size)
        for level, outcome, episode_id in state['episode_buffer']:
            record = EpisodeRecord(
                difficulty_level=DifficultyLevel(level),
                outcome=Outcome(outcome),
                episode_id=episode_id
            )
            self.episode_buffer.append(record)