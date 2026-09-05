"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           YOUR TASK PROMPTS                                   ║
║                                                                               ║
║  CUSTOMIZE THIS FILE to define prompts/instructions for your task.            ║
║  Prompts are selected based on task type and returned to the model.           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import List


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED, PARAMETERIZED PROMPT (参考 O-7 风格，但针对两步变换)
# ══════════════════════════════════════════════════════════════════════════════

def get_prompt(
    task_type: str = "default",
    source_color: str = "color A",
    target_color: str = "color B",
    source_shape: str = "shape A",
    target_shape: str = "shape D",
    movement_direction: str = "up or down",
) -> str:
    """
    生成统一的分步指令，明确两步变换：先颜色变化，后位置移动。整体长度控制在约 100 词以内。
    
    Args:
        task_type: Type of task (for compatibility)
        source_color: Initial color name
        target_color: Target color name
        source_shape: Shape in the example row
        target_shape: Shape in the question row
        movement_direction: Direction of movement (e.g., "up", "down", "up or down")
    
    Returns:
        Unified prompt string with clear step-by-step instructions
    """
    prompt_template = (
        "The scene shows a sequential analogy A→B→C :: D→?→? with three shapes per row. "
        "On the top row, a {source_color} {source_shape} becomes {target_color}, then moves {movement_direction}. "
        "On the bottom row, apply the same two-step transformation to the {target_shape}: "
        "first change color from {source_color} to {target_color}, then move {movement_direction}. "
        "Show both transformations sequentially in the video."
    )
    return prompt_template.format(
        source_color=source_color,
        target_color=target_color,
        source_shape=source_shape,
        target_shape=target_shape,
        movement_direction=movement_direction,
    )


def get_all_prompts(task_type: str = "default") -> List[str]:
    """返回当前任务类型的所有可用提示（单一模板）。"""
    return [get_prompt(task_type=task_type)]
