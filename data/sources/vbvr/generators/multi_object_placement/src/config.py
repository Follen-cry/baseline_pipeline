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
    
    domain: str = Field(default="multi_object_placement")
    image_size: tuple[int, int] = Field(default=(1024, 1024))
    
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
    #  TASK-SPECIFIC SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    
    # ══════════════════════════════════════════════════════════════════════════
    #  TASK-SPECIFIC SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    
    # Multi-object placement specific parameters
    num_objects_range: tuple[int, int] = Field(
        default=(2, 5), 
        description="Range of number of objects to place"
    )
    
    shapes: list[str] = Field(
        default=["circle", "square", "rectangle", "triangle"],
        description="Available 2D object shapes"
    )
    
    colors: list[str] = Field(
        default=["red", "green", "blue", "yellow", "purple", "orange"],
        description="Available object colors"
    )
    
    object_size_range: tuple[int, int] = Field(
        # Scaled up ~1.5× for 1024×1024 canvas (size is treated as "radius"/half-size in rendering)
        default=(38, 75),
        description="Range of object sizes in pixels"
    )
    
    marker_size_ratio: float = Field(
        default=0.5,
        description="Ratio of marker size to object size"
    )
    
    # Animation settings
    animation_duration: float = Field(
        default=3.0,
        description="Animation duration in seconds"
    )
    
    border_gradient: bool = Field(
        default=True,
        description="Whether to use gradient borders for objects"
    )
    
    # Layout settings
    min_distance: int = Field(
        # Extra gap (in pixels) between object and marker in the initial frame
        default=120,
        description="Minimum distance between objects and markers in initial frame"
    )
    
    min_object_spacing: int = Field(
        # Extra gap (in pixels) between objects to avoid any occlusion
        default=120,
        description="Minimum spacing between objects in initial frame"
    )
    
    min_marker_spacing: int = Field(
        # Extra gap (in pixels) between star markers so that final objects (placed on markers) never overlap
        default=220,
        description="Minimum spacing between markers (stars) to ensure final objects are well separated"
    )
    
    background_color: str = Field(
        default="white",
        description="Background color of the scene"
    )
