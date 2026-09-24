"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           YOUR TASK CONFIGURATION                             ║
║                                                                               ║
║  CUSTOMIZE THIS FILE to define your task-specific settings.                   ║
║  Inherits common settings from core.GenerationConfig                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from pydantic import Field
from core import GenerationConfig


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
    
    domain: str = Field(default="rotation_puzzle")
    image_size: tuple[int, int] = Field(default=(1024, 1024))  # Canvas size per spec
    
    # ══════════════════════════════════════════════════════════════════════════
    #  VIDEO SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    
    generate_videos: bool = Field(
        default=True,
        description="Whether to generate ground truth videos"
    )
    
    video_fps: int = Field(
        default=16,
        description="Video frame rate"
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    #  ROTATION PUZZLE TASK SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    
    canvas_size: tuple[int, int] = Field(
        default=(1024, 1024),
        description="Canvas size (width, height) in pixels"
    )
    
    background_color: tuple[int, int, int] = Field(
        default=(248, 250, 252),  # #f8fafc
        description="Background color (RGB)"
    )
    
    num_squares: int = Field(
        default=4,
        description="Number of squares (always 4 for 2×2 grid)"
    )
    
    square_size: int = Field(
        default=220,
        description="Size of each square in pixels"
    )
    
    square_spacing: int = Field(
        default=30,
        description="Spacing between squares in pixels"
    )
    
    pipe_color: tuple[int, int, int] = Field(
        default=(59, 130, 246),  # #3b82f6 (blue)
        description="Pipe path color (RGB) - will be randomly selected from palette"
    )
    
    pipe_color_palette: list[tuple[int, int, int]] = Field(
        default=[
            (59, 130, 246),   # Blue
            (239, 68, 68),    # Red
            (34, 197, 94),    # Green
            (168, 85, 247),   # Purple
            (249, 115, 22),   # Orange
            (236, 72, 153),   # Pink
            (14, 165, 233),   # Sky Blue
            (234, 179, 8),    # Yellow
            (20, 184, 166),   # Teal
            (245, 158, 11),   # Amber
            (99, 102, 241),   # Indigo
            (139, 92, 246),   # Violet
            (244, 63, 94),    # Rose
            (16, 185, 129),   # Emerald
            (251, 146, 60),   # Orange (lighter)
            (217, 70, 239),   # Fuchsia
            (6, 182, 212),    # Cyan
            (132, 204, 22),   # Lime
            (251, 191, 36),   # Yellow (amber)
            (225, 29, 72),    # Red (deep)
        ],
        description="20 distinct pipe colors for visual variety"
    )
    
    pipe_width: int = Field(
        default=14,
        description="Width of pipe paths in pixels"
    )
    
    square_border_width: int = Field(
        default=3,
        description="Width of square border in pixels"
    )
    
    square_border_color: tuple[int, int, int] = Field(
        default=(100, 116, 139),  # #64748b (slate-500)
        description="Square border color (RGB)"
    )
    
    difficulty: str = Field(
        default="medium",
        description="Difficulty level: 'easy', 'medium', 'hard'"
    )
