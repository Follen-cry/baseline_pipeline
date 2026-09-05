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
#  UNIFIED, PARAMETERIZED PROMPT (参考 O-7 / O-1 风格)
# ══════════════════════════════════════════════════════════════════════════════

def get_prompt(
    task_type: str = "default",
    source_shape: str = "shape A",
    target_shape: str = "shape D",
    movement_direction: str = "up",
) -> str:
    """
    生成统一的分步指令，明确两步转换过程并应用到 D → ? → ?。整体长度控制在约 100 词以内。
    
    Args:
        task_type: Type of task (for future extension)
        source_shape: Shape name in the example sequence (A)
        target_shape: Shape name in the question sequence (D)
        movement_direction: Movement direction (up/down/up_small/down_small/etc.)
    
    Returns:
        Unified prompt string with clear step-by-step instructions
    """
    # Map movement names to readable descriptions
    movement_desc = {
        "up": "up",
        "down": "down",
        "up_small": "up by a small amount",
        "down_small": "down by a small amount",
        "up_large": "up by a large amount",
        "down_large": "down by a large amount",
        "up_tiny": "up by a tiny amount",
        "down_tiny": "down by a tiny amount",
        "up_huge": "up by a huge amount",
        "down_huge": "down by a huge amount",
    }.get(movement_direction, movement_direction)
    
    prompt_template = (
        "The scene shows an analogy A→B→C :: D→?→? with two rows of shapes and arrows. "
        "On the top row, a filled {source_shape} first becomes an outline-only {source_shape} (step 1), "
        "then moves {movement_desc} (step 2). "
        "On the bottom row, the {target_shape} starts filled. "
        "Apply the same two-step transformation: first convert it to outline-only style, "
        "then move it {movement_desc}, keeping its shape and size the same while only the style and position change."
    )
    return prompt_template.format(
        source_shape=source_shape,
        target_shape=target_shape,
        movement_desc=movement_desc,
    )


def get_all_prompts(task_type: str = "default") -> List[str]:
    """返回当前任务类型的所有可用提示（单一模板）。"""
    return [get_prompt(task_type=task_type)]
