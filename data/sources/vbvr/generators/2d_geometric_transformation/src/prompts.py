"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           YOUR TASK PROMPTS                                   ║
║                                                                               ║
║  CUSTOMIZE THIS FILE to define prompts/instructions for your task.            ║
║  Prompts are selected based on task type and returned to the model.           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ══════════════════════════════════════════════════════════════════════════════
#  DEFINE YOUR PROMPTS (Unified, parameterized template)
# ══════════════════════════════════════════════════════════════════════════════

def get_prompt(
    shape_type: str = "2D polygon",
    rotation_direction: str = "clockwise",
    task_type: str = "default"
) -> str:
    """
    Generate a unified, step-by-step prompt for 2D geometric transformation task.
    
    This prompt clearly explains the process:
    1. Observe the initial state (shape, rotation center, target outline)
    2. Understand the rotation transformation rule
    3. Rotate the shape around the center point in specified direction
    4. Align with the target outline
    
    Args:
        shape_type: Type of 2D shape (for future extension, currently unused)
        rotation_direction: Rotation direction - "clockwise" or "counterclockwise"
        task_type: Type of task (for future extension, currently unused)
    
    Returns:
        Unified prompt string with clear step-by-step instructions including rotation direction
    
    Word count: under 100 words
    """
    # Determine rotation direction description
    if rotation_direction == "clockwise":
        direction_desc = "clockwise"
    else:
        direction_desc = "counterclockwise"
    
    prompt_template = (
        "The scene shows a colored 2D polygon, a rotation center marked by a small circular marker, "
        "and a dashed target outline indicating the final orientation. Only orientation changes. "
        "Rotate the polygon in the {direction} direction. First note the polygon’s initial orientation and the marked center, "
        "then read the dashed outline to know the target orientation. Rotate the polygon {direction} around the center "
        "until it completely overlaps the dashed outline with no offset. Keep size unchanged, keep a line to the center, "
        "and end with the polygon exactly matching the dashed outline."
    )
    
    return prompt_template.format(direction=direction_desc)
