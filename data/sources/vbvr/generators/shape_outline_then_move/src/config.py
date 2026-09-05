"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           YOUR TASK CONFIGURATION                             ║
║                                                                               ║
║  CUSTOMIZE THIS FILE to define your task-specific settings.                   ║
║  Inherits common settings from core.GenerationConfig                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import colorsys
from pydantic import Field
from core import GenerationConfig


def _generate_shape_colors():
    """Generate 400 distinct colors: 40 hand-picked + 360 HSV-generated."""
    # Original 40 carefully selected colors
    base_colors = [
        # Primary colors
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        # Secondary colors
        (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 0, 128), (128, 255, 0), (0, 255, 128),
        # Tertiary colors
        (192, 64, 0), (64, 192, 0), (0, 192, 64), (0, 64, 192), (192, 0, 64), (64, 0, 192),
        # Pastel colors
        (255, 182, 193), (173, 216, 230), (144, 238, 144), (255, 218, 185), (221, 160, 221), (176, 224, 230),
        # Deep colors
        (139, 0, 0), (0, 100, 0), (0, 0, 139), (184, 134, 11), (128, 0, 128), (0, 128, 128),
        # Bright colors
        (255, 69, 0), (50, 205, 50), (30, 144, 255), (255, 20, 147), (255, 215, 0), (138, 43, 226),
        # Additional distinct colors
        (165, 42, 42), (70, 130, 180), (220, 20, 60), (255, 140, 0),
    ]
    
    # Generate 360 additional colors using HSV space for better distribution
    generated_colors = []
    for h in range(0, 360, 10):
        for s in [0.5, 0.65, 0.8, 0.95]:
            for v in [0.6, 0.75, 0.9]:
                if len(generated_colors) >= 360:
                    break
                rgb = colorsys.hsv_to_rgb(h / 360, s, v)
                generated_colors.append(tuple(int(c * 255) for c in rgb))
            if len(generated_colors) >= 360:
                break
        if len(generated_colors) >= 360:
            break
    
    all_colors = base_colors + generated_colors[:360]
    return all_colors


class TaskConfig(GenerationConfig):
    """
    Your task-specific configuration.
    
    CUSTOMIZE THIS CLASS to add your task's hyperparameters.
    
    Inherited from GenerationConfig:
        - num_samples: int          # Number of samples to generate
        - domain: str               # Task domain name
        - difficulty: Optional[str] # Difficulty level
        - random_seed: Optional[int] # For reproducibility
        - output_dir: Path          # Where to save outputs
        - image_size: tuple[int, int] # Image dimensions
    """
    
    # ══════════════════════════════════════════════════════════════════════════
    #  OVERRIDE DEFAULTS
    # ══════════════════════════════════════════════════════════════════════════
    
    domain: str = Field(default="shape_outline_then_move")
    # 1:1 视频和图片分辨率，统一为 1024 x 1024
    image_size: tuple[int, int] = Field(default=(1024, 1024))
    
    # ══════════════════════════════════════════════════════════════════════════
    #  VIDEO SETTINGS (Optional but recommended)
    # ══════════════════════════════════════════════════════════════════════════
    
    generate_videos: bool = Field(
        default=True,
        description="Whether to generate ground truth videos"
    )
    
    # 固定为 16 fps，配合约 60 帧的视频长度，时长约 3.75 秒 < 5 秒
    video_fps: int = Field(
        default=16,
        description="Video frame rate (fixed 16 fps)"
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    #  TASK-SPECIFIC SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    
    # Shape matching specific settings
    shape_size: int = Field(
        default=160,
        description="Size of individual shapes in pixels"
    )
    
    margin: int = Field(
        default=80,
        description="Margin around shapes"
    )
    
    arrow_length: int = Field(
        default=60,
        description="Length of transformation arrows"
    )
    
    question_mark_size: int = Field(
        default=60,
        description="Size of question mark"
    )
    
    # Shape colors - expanded to 400 distinct colors
    shape_colors: list[tuple[int, int, int]] = Field(
        default_factory=_generate_shape_colors,
        description="List of 400 distinct RGB colors for shapes"
    )