"""
Rotation puzzle task implementation.

This module contains the task-specific implementation:
    - config.py   : Task-specific configuration (TaskConfig)
    - generator.py: Task generation logic (TaskGenerator)
    - prompts.py  : Task prompts/instructions (get_prompt)
"""

from .config import TaskConfig
from .generator import TaskGenerator
from .prompts import get_prompt

__all__ = ["TaskConfig", "TaskGenerator", "get_prompt"]
