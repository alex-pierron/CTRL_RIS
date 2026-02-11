"""Curriculum scheduler used by RL trainers.

`TaskManager` responsibilities:
- sample one difficulty level per environment rollout,
- build per-episode difficulty payloads for environment reset,
- aggregate episode outcomes,
- adapt current maximum reachable level using buffer statistics.

Difficulty progression policy:
- Advance level when buffer is full and success is high with almost no severe
  failures.
- Reduce level after enough attempts if success is persistently low.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .default_difficulty_modes import create_default_difficulty_configs
from .types import DifficultyConfig, DifficultyLevel, EpisodeDifficultyConfig, Outcome


@dataclass
class EpisodeRecord:
    """Record of one episode used in moving-window progression statistics."""

    difficulty_level: DifficultyLevel
    outcome: Outcome
    episode_id: int


class TaskManager:
    """Manage curriculum progression and produce per-episode difficulty configs.

    Interface contract:
    - `generate_episode_configs()` returns one `EpisodeDifficultyConfig` per env.
    - The returned list must be passed unchanged to `VecEnv.reset(...)`.
    - `update_episode_outcomes(...)` consumes one `Outcome` per sampled env and
      updates curriculum progression state.
    """

    def __init__(
        self,
        num_users: int,
        user_limits,
        RIS_position,
        num_steps_per_episode: int,
        downlink_uplink_eavesdropper_bools: list,
        thresholds: np.ndarray,
        eavesdropper_thresholds: Optional[Sequence[float]] = None,
        difficulty_configs: Optional[Dict[DifficultyLevel, DifficultyConfig]] = None,
        Buffer_Size: int = 10,
        H: float = 4.0,
        num_environments: int = 1,
        random_seed: Optional[int] = None,
    ):
        """Initialize curriculum manager and progression state.

        Args:
            num_users: Number of legitimate users.
            user_limits: Spawn rectangle used to derive default angular limits.
            RIS_position: RIS 2D coordinates.
            num_steps_per_episode: Episode horizon used to normalize sums.
            downlink_uplink_eavesdropper_bools: Flags
                `[downlink_enabled, uplink_enabled, eavesdropper_enabled]`.
            thresholds: Success thresholds for downlink/uplink means.
            eavesdropper_thresholds: Optional dedicated thresholds used when
                evaluating eavesdropper outcomes as
                `[downlink_reference, uplink_reference]`. If omitted, fallback
                behavior is applied for backward compatibility.
            difficulty_configs: Optional custom per-level templates.
            Buffer_Size: FIFO size used to estimate solved/unsolved level state.
            H: Multiplier controlling when level reduction is allowed.
            num_environments: Number of parallel environments.
            random_seed: Optional deterministic seed.

        Invariants:
            - `current_max_level` bounds sampling support.
            - The buffer stores recent outcomes across all environments.
            - One call to `generate_episode_configs` increments `current_episode`.
        """
        self.num_users = num_users
        self.num_steps_per_episode = num_steps_per_episode
        self.user_limits = user_limits
        self.RIS_position = RIS_position
        self._is_downlink_used = downlink_uplink_eavesdropper_bools[0]
        self._is_uplink_used = downlink_uplink_eavesdropper_bools[1]
        self._are_eavesdroppers_used = downlink_uplink_eavesdropper_bools[2]
        thresholds_array = np.asarray(thresholds, dtype=float).reshape(-1)
        if thresholds_array.size < 2:
            raise ValueError(
                "thresholds must provide at least two values: "
                "[downlink_threshold, uplink_threshold]."
            )
        self.thresholds = thresholds_array
        self.eavesdropper_thresholds = self._resolve_eavesdropper_thresholds(
            thresholds_array=thresholds_array,
            eavesdropper_thresholds=eavesdropper_thresholds,
        )
        self.Buffer_Size = Buffer_Size
        self.H = H
        self.num_environments = num_environments
        self.rng = np.random.default_rng(random_seed)

        self.min_angle = np.arctan(
            (self.RIS_position[1] - self.user_limits[1][1])
            / (self.user_limits[0][1] - self.RIS_position[0])
        )
        self.max_angle = np.arctan(
            (self.RIS_position[1] - self.user_limits[1][0])
            / (self.user_limits[0][0] - self.RIS_position[0])
        )
        self.difference_angle = self.max_angle - self.min_angle

        self.difficulty_configs = difficulty_configs or create_default_difficulty_configs(
            num_users=self.num_users,
            difference_angle=self.difference_angle,
        )

        self.current_max_level = DifficultyLevel.LEVEL_1
        self.current_episode = 0
        self.episodes_used_current_level = 0

        self.episode_buffer = deque(maxlen=Buffer_Size)
        self.selected_levels = np.array([], dtype=int)

    def _get_level_probabilities(self, n: int) -> Dict[DifficultyLevel, float]:
        """Return the sampling distribution restricted to levels <= `n`."""
        probabilities: Dict[DifficultyLevel, float] = {}

        if n == 1:
            probabilities[DifficultyLevel.LEVEL_1] = 1.0
        elif n == 2:
            probabilities[DifficultyLevel.LEVEL_2] = 0.8
            probabilities[DifficultyLevel.LEVEL_1] = 0.2
        else:
            probabilities[DifficultyLevel(n)] = 0.65
            probabilities[DifficultyLevel(n - 1)] = 0.20
            remaining_prob = 0.15 / (n - 2)
            probabilities.update(
                {
                    DifficultyLevel(level): remaining_prob
                    for level in range(1, n - 1)
                }
            )
        return probabilities

    def _sample_difficulty_levels(self, num_samples: int) -> np.ndarray:
        """Sample one difficulty level per environment rollout."""
        probabilities = self._get_level_probabilities(self.current_max_level)
        levels = np.array(list(probabilities.keys()), dtype=int)
        probs = np.array(list(probabilities.values()))
        return self.rng.choice(levels, size=num_samples, p=probs)

    def generate_episode_configs(self) -> List[EpisodeDifficultyConfig]:
        """Generate one explicit difficulty payload per environment."""
        self.selected_levels = self._sample_difficulty_levels(self.num_environments)
        configs = [
            EpisodeDifficultyConfig.from_level_config(
                self.difficulty_configs[DifficultyLevel(level)]
            )
            for level in self.selected_levels
        ]
        self.current_episode += 1
        return configs

    def update_episode_outcomes(self, outcomes: List[Outcome]):
        """Update progression state from outcomes produced by current episode."""
        if len(outcomes) != len(self.selected_levels):
            raise ValueError(
                "Outcomes length must match previously sampled levels: "
                f"{len(outcomes)} != {len(self.selected_levels)}"
            )

        final_outcomes = list(zip(self.selected_levels, outcomes))
        records = [
            EpisodeRecord(
                difficulty_level=difficulty_level,
                outcome=outcome,
                episode_id=self.current_episode,
            )
            for difficulty_level, outcome in final_outcomes
        ]

        self.episode_buffer.extend(records)
        self.episodes_used_current_level += self.num_environments
        self._check_difficulty_progression()

    def _check_difficulty_progression(self):
        """Advance or reduce `current_max_level` based on windowed outcomes."""
        if self._is_difficulty_solved():
            if self.current_max_level < DifficultyLevel.LEVEL_5:
                self.current_max_level = DifficultyLevel(self.current_max_level + 1)
                self._reset_buffer_for_new_level()
        elif self._should_reduce_difficulty():
            if self.current_max_level > DifficultyLevel.LEVEL_1:
                self.current_max_level = DifficultyLevel(self.current_max_level - 1)
                self._reset_buffer_for_new_level()

    def _is_difficulty_solved(self) -> bool:
        """Return True when performance is stably strong on the current buffer."""
        if len(self.episode_buffer) < self.Buffer_Size:
            return False

        success_count = sum(
            1 for record in self.episode_buffer if record.outcome == Outcome.SUCCESS
        )
        severe_failure_count = sum(
            1
            for record in self.episode_buffer
            if record.outcome == Outcome.SEVERE_FAILURE
        )
        success_rate = success_count / len(self.episode_buffer)
        severe_failure_rate = severe_failure_count / len(self.episode_buffer)
        return success_rate > 0.9 and severe_failure_rate < 0.05

    def _should_reduce_difficulty(self) -> bool:
        """Return True when level appears too hard despite enough attempts."""
        if self.episodes_used_current_level < self.H * self.Buffer_Size:
            return False
        if len(self.episode_buffer) == 0:
            return False

        success_count = sum(
            1 for record in self.episode_buffer if record.outcome == Outcome.SUCCESS
        )
        success_rate = success_count / len(self.episode_buffer)
        return success_rate < 0.4

    def compute_episodes_outcome(
        self,
        downlink_sum=None,
        uplink_sum=None,
        best_eavesdropper_sum=None,
    ) -> Outcome:
        """Compute one outcome per environment episode."""
        return self._compute_episodes_outcome_impl(
            downlink_sum=downlink_sum,
            uplink_sum=uplink_sum,
            best_eavesdropper_sum=best_eavesdropper_sum,
        )

    def _normalize_to_batch_users(self, values: Any) -> np.ndarray:
        """Normalize arrays to shape `(batch, users_flat)`."""
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr.reshape(arr.shape[0], -1)

    def _compute_condition_result(
        self,
        values: Any,
        threshold: float,
        success_if_greater: bool,
    ) -> np.ndarray:
        """Convert per-user means into outcome severity values."""
        values2d = self._normalize_to_batch_users(values)
        num_users = values2d.shape[1]
        if success_if_greater:
            ratio = np.sum(values2d > threshold, axis=1) / num_users
        else:
            ratio = np.sum(values2d < threshold, axis=1) / num_users
        return np.where(
            ratio == 1.0,
            Outcome.SUCCESS.value,
            np.where(ratio >= 0.51, Outcome.FAILURE.value, Outcome.SEVERE_FAILURE.value),
        )

    def _ensure_same_batch_size(self, values: Sequence[np.ndarray]) -> None:
        """Raise on inconsistent batch dimensions across condition arrays."""
        if not values:
            return
        batch_sizes = {v.shape[0] for v in values}
        if len(batch_sizes) > 1:
            raise ValueError(f"Inconsistent batch sizes for outcomes: {sorted(batch_sizes)}")

    def _get_eavesdropper_threshold(self) -> float:
        """Pick threshold used for best-eavesdropper success condition."""
        if self._is_downlink_used and self._is_uplink_used:
            return min(
                float(self.eavesdropper_thresholds[0]),
                float(self.eavesdropper_thresholds[1]),
            )
        if self._is_downlink_used:
            return float(self.eavesdropper_thresholds[0])
        if self._is_uplink_used:
            return float(self.eavesdropper_thresholds[1])
        return float(self.eavesdropper_thresholds[0])

    def _resolve_eavesdropper_thresholds(
        self,
        thresholds_array: np.ndarray,
        eavesdropper_thresholds: Optional[Sequence[float]],
    ) -> np.ndarray:
        """Resolve dedicated eavesdropper thresholds with backward compatibility."""
        if eavesdropper_thresholds is not None:
            eav_array = np.asarray(eavesdropper_thresholds, dtype=float).reshape(-1)
            if eav_array.size < 2:
                raise ValueError(
                    "eavesdropper_thresholds must provide two values: "
                    "[downlink_reference, uplink_reference]."
                )
            return eav_array[:2]

        # Backward compatibility path:
        # - 4+ values in thresholds: [dl, ul, eav_dl, eav_ul]
        # - 3 values in thresholds: [dl, ul, eav_shared]
        # - 2 values in thresholds: [dl, ul] reused for eavesdropper condition
        if thresholds_array.size >= 4:
            return np.asarray([thresholds_array[2], thresholds_array[3]], dtype=float)
        if thresholds_array.size == 3:
            return np.asarray([thresholds_array[2], thresholds_array[2]], dtype=float)
        return np.asarray([thresholds_array[0], thresholds_array[1]], dtype=float)

    def _compute_episodes_outcome_impl(
        self,
        downlink_sum=None,
        uplink_sum=None,
        best_eavesdropper_sum=None,
    ) -> Outcome:
        """Implementation behind `compute_episodes_outcome`."""
        downlink_meaned = None
        uplink_meaned = None
        eavesdropper_meaned = None

        if self._is_downlink_used:
            if downlink_sum is None:
                raise ValueError("downlink_sum is required when downlink is enabled.")
            downlink_meaned = (
                np.asarray(downlink_sum, dtype=float) / self.num_steps_per_episode
            )

        if self._is_uplink_used:
            if uplink_sum is None:
                raise ValueError("uplink_sum is required when uplink is enabled.")
            uplink_meaned = (
                np.asarray(uplink_sum, dtype=float) / self.num_steps_per_episode
            )

        if self._are_eavesdroppers_used:
            if best_eavesdropper_sum is None:
                raise ValueError(
                    "best_eavesdropper_sum is required when eavesdroppers are enabled."
                )
            eavesdropper_meaned = (
                np.asarray(best_eavesdropper_sum, dtype=float)
                / self.num_steps_per_episode
            )

        normalized_values = [
            self._normalize_to_batch_users(v)
            for v in (downlink_meaned, uplink_meaned, eavesdropper_meaned)
            if v is not None
        ]
        self._ensure_same_batch_size(normalized_values)

        results = []
        if downlink_meaned is not None:
            results.append(
                self._compute_condition_result(
                    downlink_meaned,
                    float(self.thresholds[0]),
                    success_if_greater=True,
                )
            )
        if uplink_meaned is not None:
            results.append(
                self._compute_condition_result(
                    uplink_meaned,
                    float(self.thresholds[1]),
                    success_if_greater=True,
                )
            )
        if eavesdropper_meaned is not None:
            results.append(
                self._compute_condition_result(
                    eavesdropper_meaned,
                    self._get_eavesdropper_threshold(),
                    success_if_greater=False,
                )
            )

        if not results:
            return []

        outcomes = np.max(
            np.vstack([np.asarray(r).reshape(-1) for r in results]),
            axis=0,
        )
        return [Outcome(int(outcome)) for outcome in outcomes.tolist()]

    def _reset_buffer_for_new_level(self):
        """Reset statistics when changing max reachable level."""
        self.episode_buffer.clear()
        self.episodes_used_current_level = 0

    def get_buffer_statistics(self) -> Dict:
        """Return current moving-window statistics."""
        if len(self.episode_buffer) == 0:
            return {
                "buffer_size": 0,
                "success_rate": 0.0,
                "failure_rate": 0.0,
                "severe_failure_rate": 0.0,
                "level_distribution": {},
            }

        success_count = sum(
            1 for record in self.episode_buffer if record.outcome == Outcome.SUCCESS
        )
        failure_count = sum(
            1 for record in self.episode_buffer if record.outcome == Outcome.FAILURE
        )
        severe_failure_count = sum(
            1
            for record in self.episode_buffer
            if record.outcome == Outcome.SEVERE_FAILURE
        )
        buffer_size = len(self.episode_buffer)

        level_counts = {}
        for record in self.episode_buffer:
            level = record.difficulty_level
            level_counts[level] = level_counts.get(level, 0) + 1

        return {
            "buffer_size": buffer_size,
            "success_rate": success_count / buffer_size,
            "failure_rate": failure_count / buffer_size,
            "severe_failure_rate": severe_failure_count / buffer_size,
            "level_distribution": level_counts,
            "is_buffer_full": buffer_size == self.Buffer_Size,
        }

    def get_statistics(self) -> Dict:
        """Return full scheduler state for debug/logging purposes."""
        buffer_stats = self.get_buffer_statistics()
        return {
            "current_episode": self.current_episode,
            "current_max_level": self.current_max_level,
            "episodes_used_current_level": self.episodes_used_current_level,
            "level_probabilities": self._get_level_probabilities(self.current_max_level),
            "buffer_stats": buffer_stats,
            "difficulty_solved": self._is_difficulty_solved(),
            "should_reduce_difficulty": self._should_reduce_difficulty(),
            "episodes_threshold_for_reduction": self.H * self.Buffer_Size,
        }

    def reset(self):
        """Reset scheduler state to initial level and empty statistics."""
        self.current_max_level = DifficultyLevel.LEVEL_1
        self.current_episode = 0
        self.episodes_used_current_level = 0
        self.episode_buffer.clear()

    def save_state(self) -> Dict:
        """Serialize scheduler state."""
        return {
            "current_max_level": int(self.current_max_level),
            "current_episode": self.current_episode,
            "episodes_used_current_level": self.episodes_used_current_level,
            "episode_buffer": [
                (int(r.difficulty_level), int(r.outcome), r.episode_id)
                for r in self.episode_buffer
            ],
            "Buffer_Size": self.Buffer_Size,
            "H": self.H,
            "num_environments": self.num_environments,
        }

    def load_state(self, state: Dict):
        """Load previously serialized scheduler state."""
        self.current_max_level = DifficultyLevel(state["current_max_level"])
        self.current_episode = state["current_episode"]
        self.episodes_used_current_level = state["episodes_used_current_level"]
        self.Buffer_Size = state["Buffer_Size"]
        self.H = state["H"]
        self.num_environments = state["num_environments"]

        self.episode_buffer = deque(maxlen=self.Buffer_Size)
        for level, outcome, episode_id in state["episode_buffer"]:
            record = EpisodeRecord(
                difficulty_level=DifficultyLevel(level),
                outcome=Outcome(outcome),
                episode_id=episode_id,
            )
            self.episode_buffer.append(record)

