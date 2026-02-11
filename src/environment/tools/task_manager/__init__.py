"""Task manager package.

Public API is intentionally re-exported here to keep imports stable:
- `from src.environment.tools.task_manager import TaskManager`
- `from src.environment.tools.task_manager import EpisodeDifficultyConfig`
"""

from .core import TaskManager
from .default_difficulty_modes import create_default_difficulty_configs, validate_difficulty_configs
from .types import DifficultyConfig, DifficultyLevel, EpisodeDifficultyConfig, Outcome

__all__ = [
    "TaskManager",
    "DifficultyLevel",
    "Outcome",
    "DifficultyConfig",
    "EpisodeDifficultyConfig",
    "create_default_difficulty_configs",
    "validate_difficulty_configs",
]

