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
    2D Geometric Transformation - Planar Rotation task configuration.

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

    domain: str = Field(default="transformation_worlds_2d_geometric_transformation")
    image_size: tuple[int, int] = Field(default=(1024, 1024))

    # ══════════════════════════════════════════════════════════════════════════
    #  VIDEO SETTINGS
    # ══════════════════════════════════════════════════════════════════════════

    generate_videos: bool = Field(
        default=True,
        description="Whether to generate ground truth videos"
    )

    video_fps: int = Field(
        default=15,
        description="Video frame rate"
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  TASK-SPECIFIC SETTINGS
    # ══════════════════════════════════════════════════════════════════════════

    object_size: int = Field(
        default=180,
        description="Size of the 2D object in pixels (scaled for 1024x1024)"
    )

    rotation_center_radius: int = Field(
        default=12,
        description="Radius of the rotation center point marker in pixels"
    )

    target_outline_width: int = Field(
        default=4,
        description="Width of the target outline in pixels"
    )

    min_rotation_angle: int = Field(
        default=30,
        description="Minimum rotation angle in degrees"
    )

    max_rotation_angle: int = Field(
        default=330,
        description="Maximum rotation angle in degrees"
    )
