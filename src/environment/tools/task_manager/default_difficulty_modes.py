"""Default curriculum difficulty modes for `TaskManager`.

The defaults are intentionally opinionated and progressive:

- Level 1: narrow user area, large angular separation, strict eavesdropper
  minimum distance. Easiest setup.
- Level 2: wider area, reduced angle separation, eavesdroppers allowed closer.
- Level 3: same area, tighter angular separation, closer eavesdropper range.
- Level 4: level 3 geometry + tighter eavesdropper proximity constraints.
- Level 5: full random positioning in broad area, smallest minimum distance.

These modes can be replaced by user-provided configurations, but this module
defines the production-safe baseline used when no custom config is provided.
"""

from typing import Dict

import numpy as np

from .types import DifficultyConfig, DifficultyLevel


def create_default_difficulty_configs(
    num_users: int,
    difference_angle: float,
) -> Dict[DifficultyLevel, DifficultyConfig]:
    """Build the canonical 5-level difficulty dictionary.

    Args:
        num_users: Number of legitimate users. Used to scale angular spacing.
        difference_angle: Total reachable angular range around RIS in radians.

    Returns:
        Mapping `DifficultyLevel -> DifficultyConfig` with all levels populated.

    Notes:
        - Angular constraints become tighter as level increases.
        - Spatial area and eavesdropper constraints become progressively harder.
        - Level 5 intentionally enables `fully_random` to remove structured
          angular spacing while keeping broad area coverage.
    """
    return {
        DifficultyLevel.LEVEL_1: DifficultyConfig(
            level=DifficultyLevel.LEVEL_1,
            grid_limits=np.array([[120, 200], [40, 100]], dtype=float),
            angle_is_max=False,
            angle_value=difference_angle / (2 * num_users),
            fully_random=False,
            new_min_distance_between_eavesdropper_and_users=25.0,
            new_max_distance_between_eavesdropper_and_users=np.inf,
        ),
        DifficultyLevel.LEVEL_2: DifficultyConfig(
            level=DifficultyLevel.LEVEL_2,
            grid_limits=np.array([[100, 200], [0, 100]], dtype=float),
            angle_is_max=False,
            angle_value=difference_angle / (4 * num_users),
            fully_random=False,
            new_min_distance_between_eavesdropper_and_users=10.0,
            new_max_distance_between_eavesdropper_and_users=50.0,
        ),
        DifficultyLevel.LEVEL_3: DifficultyConfig(
            level=DifficultyLevel.LEVEL_3,
            grid_limits=np.array([[100, 200], [0, 100]], dtype=float),
            angle_is_max=False,
            angle_value=difference_angle / (6 * num_users),
            fully_random=False,
            new_min_distance_between_eavesdropper_and_users=5.0,
            new_max_distance_between_eavesdropper_and_users=30.0,
        ),
        DifficultyLevel.LEVEL_4: DifficultyConfig(
            level=DifficultyLevel.LEVEL_4,
            grid_limits=np.array([[100, 200], [0, 100]], dtype=float),
            angle_is_max=True,
            angle_value=difference_angle / (6 * num_users),
            fully_random=False,
            new_min_distance_between_eavesdropper_and_users=2.0,
            new_max_distance_between_eavesdropper_and_users=15.0,
        ),
        DifficultyLevel.LEVEL_5: DifficultyConfig(
            level=DifficultyLevel.LEVEL_5,
            grid_limits=np.array([[100, 200], [0, 100]], dtype=float),
            angle_is_max=True,
            angle_value=np.pi / 12,  # 15 degrees
            fully_random=True,
            new_min_distance_between_eavesdropper_and_users=0.5,
            new_max_distance_between_eavesdropper_and_users=np.inf,
        ),
    }


def validate_difficulty_configs(configs: Dict[DifficultyLevel, DifficultyConfig]) -> None:
    """Validate that a custom/default config dict satisfies expected invariants.

    Args:
        configs: Mapping from difficulty level to config object.

    Raises:
        ValueError: If required levels are missing or data is malformed.
    """
    expected_levels = {
        DifficultyLevel.LEVEL_1,
        DifficultyLevel.LEVEL_2,
        DifficultyLevel.LEVEL_3,
        DifficultyLevel.LEVEL_4,
        DifficultyLevel.LEVEL_5,
    }
    provided_levels = set(configs.keys())
    if provided_levels != expected_levels:
        missing = sorted(expected_levels - provided_levels)
        extra = sorted(provided_levels - expected_levels)
        raise ValueError(
            f"Invalid difficulty level keys. missing={missing}, extra={extra}"
        )

    for level, cfg in configs.items():
        grid = np.asarray(cfg.grid_limits, dtype=float)
        if grid.shape != (2, 2):
            raise ValueError(f"{level}: grid_limits must be shape (2, 2), got {grid.shape}")
        if not (grid[0, 0] < grid[0, 1] and grid[1, 0] < grid[1, 1]):
            raise ValueError(f"{level}: grid_limits bounds are invalid: {grid}")
        if cfg.new_min_distance_between_eavesdropper_and_users < 0:
            raise ValueError(f"{level}: min eavesdropper distance must be >= 0")
        if (
            cfg.new_max_distance_between_eavesdropper_and_users
            < cfg.new_min_distance_between_eavesdropper_and_users
        ):
            raise ValueError(f"{level}: max eavesdropper distance must be >= min distance")
        if cfg.angle_value < 0:
            raise ValueError(f"{level}: angle_value must be >= 0")

