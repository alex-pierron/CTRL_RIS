"""Typed contracts used by curriculum difficulty management.

This module contains the stable schemas shared across:
- the curriculum scheduler (`TaskManager`),
- vectorized env wrappers (`DummyVecEnv` / `SubprocVecEnv`),
- environment reset flow (`RIS_Duplex.reset`),
- position generation constraints (`PositionGenerator`).
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class DifficultyLevel(IntEnum):
    """Enumeration for curriculum levels ordered from easiest to hardest."""

    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4
    LEVEL_5 = 5


class Outcome(IntEnum):
    """Episode outcome severity used by curriculum progression logic."""

    SUCCESS = 1
    FAILURE = 2
    SEVERE_FAILURE = 3


@dataclass
class DifficultyConfig:
    """Template configuration attached to a single difficulty level.

    This is a static definition of a level. At episode generation time, each
    selected level is converted to `EpisodeDifficultyConfig` which is the
    transport payload consumed by environments.
    """

    level: DifficultyLevel
    grid_limits: np.ndarray
    angle_is_max: bool
    angle_value: float
    fully_random: bool
    new_min_distance_between_eavesdropper_and_users: float
    new_max_distance_between_eavesdropper_and_users: float


@dataclass
class EpisodeDifficultyConfig:
    """Concrete, per-episode config passed to env reset.

    Field semantics:
    - `grid_limits`: spawn area used for users/eavesdroppers generation.
    - `angle_is_max`: semantic flag preserved for compatibility with current
      positioning API.
    - `angle_value`: angular constraint value in radians.
    - `min_distance_eavesdropper_users`: minimum distance from each eavesdropper
      to every user.
    - `max_distance_eavesdropper_users`: maximum distance to closest user.
    - `fully_random`: if true, angle constraints are bypassed for users.
    """

    grid_limits: np.ndarray
    angle_is_max: bool
    angle_value: float
    min_distance_eavesdropper_users: float
    max_distance_eavesdropper_users: float
    fully_random: bool

    def as_position_generator_args(self) -> tuple:
        """Return ordered args expected by `PositionGenerator` update method."""
        return (
            self.grid_limits,
            self.angle_is_max,
            self.angle_value,
            self.min_distance_eavesdropper_users,
            self.max_distance_eavesdropper_users,
            self.fully_random,
        )

    @classmethod
    def from_level_config(cls, config: DifficultyConfig) -> "EpisodeDifficultyConfig":
        """Build an episode payload from one level template."""
        return cls(
            grid_limits=config.grid_limits.copy(),
            angle_is_max=config.angle_is_max,
            angle_value=config.angle_value,
            min_distance_eavesdropper_users=config.new_min_distance_between_eavesdropper_and_users,
            max_distance_eavesdropper_users=config.new_max_distance_between_eavesdropper_and_users,
            fully_random=config.fully_random,
        )

    @classmethod
    def from_any(cls, raw_config: Any) -> "EpisodeDifficultyConfig":
        """Coerce legacy tuple/dict/object payloads to explicit episode config.

        Supported inputs:
        - `EpisodeDifficultyConfig`
        - dict with required keys
        - tuple/list length 6 in legacy positional order
        - duck-typed object exposing required attributes

        Raises:
            ValueError: if `raw_config` cannot be interpreted.
        """
        required_keys = (
            "grid_limits",
            "angle_is_max",
            "angle_value",
            "min_distance_eavesdropper_users",
            "max_distance_eavesdropper_users",
            "fully_random",
        )

        if isinstance(raw_config, EpisodeDifficultyConfig):
            return cls(
                grid_limits=np.asarray(raw_config.grid_limits).copy(),
                angle_is_max=bool(raw_config.angle_is_max),
                angle_value=float(raw_config.angle_value),
                min_distance_eavesdropper_users=float(raw_config.min_distance_eavesdropper_users),
                max_distance_eavesdropper_users=float(raw_config.max_distance_eavesdropper_users),
                fully_random=bool(raw_config.fully_random),
            )

        if isinstance(raw_config, dict):
            if not all(key in raw_config for key in required_keys):
                raise ValueError(f"Invalid difficulty config dict keys: {list(raw_config.keys())}")
            return cls(
                grid_limits=np.asarray(raw_config["grid_limits"]).copy(),
                angle_is_max=bool(raw_config["angle_is_max"]),
                angle_value=float(raw_config["angle_value"]),
                min_distance_eavesdropper_users=float(raw_config["min_distance_eavesdropper_users"]),
                max_distance_eavesdropper_users=float(raw_config["max_distance_eavesdropper_users"]),
                fully_random=bool(raw_config["fully_random"]),
            )

        if isinstance(raw_config, (list, tuple)) and len(raw_config) == 6:
            return cls(
                grid_limits=np.asarray(raw_config[0]).copy(),
                angle_is_max=bool(raw_config[1]),
                angle_value=float(raw_config[2]),
                min_distance_eavesdropper_users=float(raw_config[3]),
                max_distance_eavesdropper_users=float(raw_config[4]),
                fully_random=bool(raw_config[5]),
            )

        if all(hasattr(raw_config, key) for key in required_keys):
            return cls(
                grid_limits=np.asarray(getattr(raw_config, "grid_limits")).copy(),
                angle_is_max=bool(getattr(raw_config, "angle_is_max")),
                angle_value=float(getattr(raw_config, "angle_value")),
                min_distance_eavesdropper_users=float(getattr(raw_config, "min_distance_eavesdropper_users")),
                max_distance_eavesdropper_users=float(getattr(raw_config, "max_distance_eavesdropper_users")),
                fully_random=bool(getattr(raw_config, "fully_random")),
            )

        raise ValueError(f"Unsupported difficulty config format: {type(raw_config)!r}")

