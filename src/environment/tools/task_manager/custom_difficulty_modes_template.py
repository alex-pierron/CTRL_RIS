"""Didactic template for user-defined curriculum difficulty modes.

How to use:
1) Copy this file or import `build_custom_difficulty_configs`.
2) Edit each level according to your task geometry and desired progression.
3) Pass the resulting dictionary to:
   `TaskManager(..., difficulty_configs=my_configs)`.

The structure must remain: `Dict[DifficultyLevel, DifficultyConfig]`.
"""

from typing import Dict

import numpy as np

from .default_difficulty_modes import validate_difficulty_configs
from .types import DifficultyConfig, DifficultyLevel


def build_custom_difficulty_configs(
    num_users: int,
    difference_angle: float,
) -> Dict[DifficultyLevel, DifficultyConfig]:
    """Create an example custom curriculum.

    This implementation is intentionally simple and documented, so external
    users can modify values safely.
    """
    # Example strategy:
    # - Start with a restricted area and strong angular spacing.
    # - Gradually widen area and tighten angular spacing.
    # - Keep eavesdroppers farther at low levels, allow proximity later.
    custom_configs: Dict[DifficultyLevel, DifficultyConfig] = {
        DifficultyLevel.LEVEL_1: DifficultyConfig(
            level=DifficultyLevel.LEVEL_1,
            grid_limits=np.array([[130, 190], [30, 90]], dtype=float),
            angle_is_max=False,
            angle_value=difference_angle / (2 * num_users),
            fully_random=False,
            new_min_distance_between_eavesdropper_and_users=20.0,
            new_max_distance_between_eavesdropper_and_users=np.inf,
        ),
        DifficultyLevel.LEVEL_2: DifficultyConfig(
            level=DifficultyLevel.LEVEL_2,
            grid_limits=np.array([[110, 200], [10, 100]], dtype=float),
            angle_is_max=False,
            angle_value=difference_angle / (3 * num_users),
            fully_random=False,
            new_min_distance_between_eavesdropper_and_users=12.0,
            new_max_distance_between_eavesdropper_and_users=60.0,
        ),
        DifficultyLevel.LEVEL_3: DifficultyConfig(
            level=DifficultyLevel.LEVEL_3,
            grid_limits=np.array([[100, 200], [0, 100]], dtype=float),
            angle_is_max=False,
            angle_value=difference_angle / (5 * num_users),
            fully_random=False,
            new_min_distance_between_eavesdropper_and_users=6.0,
            new_max_distance_between_eavesdropper_and_users=35.0,
        ),
        DifficultyLevel.LEVEL_4: DifficultyConfig(
            level=DifficultyLevel.LEVEL_4,
            grid_limits=np.array([[100, 200], [0, 100]], dtype=float),
            angle_is_max=True,
            angle_value=difference_angle / (6 * num_users),
            fully_random=False,
            new_min_distance_between_eavesdropper_and_users=3.0,
            new_max_distance_between_eavesdropper_and_users=20.0,
        ),
        DifficultyLevel.LEVEL_5: DifficultyConfig(
            level=DifficultyLevel.LEVEL_5,
            grid_limits=np.array([[100, 200], [0, 100]], dtype=float),
            angle_is_max=True,
            angle_value=np.pi / 10,
            fully_random=True,
            new_min_distance_between_eavesdropper_and_users=1.0,
            new_max_distance_between_eavesdropper_and_users=np.inf,
        ),
    }

    # Always validate before returning so users fail fast on bad edits.
    validate_difficulty_configs(custom_configs)
    return custom_configs

