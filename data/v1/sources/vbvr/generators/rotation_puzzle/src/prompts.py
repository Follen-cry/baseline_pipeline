"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ROTATION PUZZLE TASK PROMPTS                               ║
║                                                                               ║
║  Prompts for pipe puzzle rotation tasks.                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ══════════════════════════════════════════════════════════════════════════════
#  ROTATION PUZZLE PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = """Solve this rotation puzzle by rotating the four squares to connect the pipe paths. Each square can be rotated 90 degrees clockwise or counterclockwise. Rotate the squares so that all pipe paths connect to form a continuous path. Keep the camera view fixed in the top-down perspective and maintain all square positions unchanged. Stop the video when all pipes are connected and the puzzle is solved."""


def get_prompt(task_type: str = "default") -> str:
    """
    Get prompt for rotation puzzle task.
    
    Args:
        task_type: Type of task (currently only "default" is supported)
        
    Returns:
        Prompt string
    """
    return PROMPT_TEMPLATE


def get_all_prompts(task_type: str = "default") -> list[str]:
    """Get all prompts for a given task type."""
    return [PROMPT_TEMPLATE]
