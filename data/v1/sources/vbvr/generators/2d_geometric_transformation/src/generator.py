"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           YOUR TASK GENERATOR                                 ║
║                                                                               ║
║  CUSTOMIZE THIS FILE to implement your data generation logic.                 ║
║  Replace the example implementation with your own task.                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import random
import math
import tempfile
from pathlib import Path
from typing import Set, Tuple, List
from PIL import Image, ImageDraw

from core import BaseGenerator, TaskPair, ImageRenderer
from core.video_utils import VideoGenerator
from .config import TaskConfig
from .prompts import get_prompt

# Scaling checker hint: this task has ample randomization dimensions.
ESTIMATED_COMBINATIONS_LOWER_BOUND = 10000


class TaskGenerator(BaseGenerator):
    """
    2D Geometric Transformation - Planar Rotation task generator.

    Generates tasks showing 2D shape rotation in the plane around a specified center point.

    Required:
        - generate_task_pair(task_id) -> TaskPair

    The base class provides:
        - self.config: Your TaskConfig instance
        - generate_dataset(): Loops and calls generate_task_pair() for each sample
    """

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.renderer = ImageRenderer(image_size=config.image_size)

        # Initialize video generator if enabled
        self.video_generator = None
        if config.generate_videos and VideoGenerator.is_available():
            self.video_generator = VideoGenerator(fps=config.video_fps, output_format="mp4")

        # Deduplication to avoid repeated samples within one dataset run
        self._seen_signatures: Set[tuple] = set()

    def generate_task_pair(self, task_id: str) -> TaskPair:
        """Generate one 2D planar rotation task pair."""

        # Generate task data (object shape, rotation angle, center point)
        task_data = self._generate_task_data()

        # Render images
        first_image = self._render_initial_state(task_data)
        final_image = self._render_final_state(task_data)

        # Generate video (optional)
        video_path = None
        if self.config.generate_videos and self.video_generator:
            video_path = self._generate_video(first_image, final_image, task_id, task_data)

        # Select prompt (using unified parameterized template)
        prompt = get_prompt(
            shape_type=task_data.get("shape_type", "2D polygon"),
            rotation_direction=task_data.get("rotation_direction", "clockwise"),
            task_type=task_data.get("type", "default")
        )

        
        # Build metadata with objects (symbol, center, color)
        objects = self._build_objects_metadata(task_data)
        
        optimized_task_data = {
            "shape_type": task_data["shape_type"],
            "rotation_angle": task_data["rotation_angle"],
            "rotation_direction": task_data["rotation_direction"],
            "initial_angle": task_data["initial_angle"],
            "objects": objects
        }
        
        metadata = self._build_metadata(task_id, optimized_task_data)
        
        
        
        return TaskPair(
            task_id=task_id,
            domain=self.config.domain,
            prompt=prompt,
            first_image=first_image,
            final_image=final_image,
            ground_truth_video=video_path,
            metadata=metadata
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  TASK-SPECIFIC METHODS
    # ══════════════════════════════════════════════════════════════════════════

    def _generate_task_data(self) -> dict:
        """
        Generate random 2D shape planar rotation configuration.

        Creates:
        - A 2D polygon shape (represented as vertices)
        - A rotation center point
        - A rotation angle
        - Initial and target orientations
        """
        width, height = self.config.image_size
        object_size = self.config.object_size

        max_attempts = 200
        last_payload = None

        for _ in range(max_attempts):
            # Generate a random 2D polygon shape
            shape_type = random.choice(["L_shape", "T_shape", "trapezoid", "arrow"])
            base_shape = self._create_base_shape(shape_type, object_size)

            # Random rotation center within the central area
            center_margin = object_size
            rotation_center = (
                random.randint(center_margin, width - center_margin),
                random.randint(center_margin, height - center_margin)
            )

            # Random rotation angle (positive = counterclockwise, negative = clockwise)
            base_angle = random.randint(
                self.config.min_rotation_angle,
                self.config.max_rotation_angle
            )

            # Random rotation direction (clockwise or counterclockwise)
            rotation_direction = random.choice(["clockwise", "counterclockwise"])
            
            # In screen coordinates (y-axis down):
            # - Standard rotation matrix: positive angle = clockwise (visually)
            # - Standard rotation matrix: negative angle = counterclockwise (visually)
            # So for clockwise, we use positive angle; for counterclockwise, we use negative angle
            if rotation_direction == "clockwise":
                rotation_angle = base_angle  # Positive = clockwise in screen coords
            else:  # counterclockwise
                rotation_angle = -base_angle  # Negative = counterclockwise in screen coords

            # Random initial angle (to position the object initially)
            initial_angle = random.randint(0, 360)

            # Random color for the object
            object_color = (
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200)
            )

            # Calculate initial and final shapes
            initial_shape = self._rotate_and_translate_shape(
                base_shape,
                initial_angle,
                rotation_center
            )

            final_shape = self._rotate_and_translate_shape(
                base_shape,
                initial_angle + rotation_angle,
                rotation_center
            )

            payload = {
                "base_shape": base_shape,
                "initial_shape": initial_shape,
                "final_shape": final_shape,
                "rotation_center": rotation_center,
                "rotation_angle": rotation_angle,
                "rotation_direction": rotation_direction,
                "initial_angle": initial_angle,
                "object_color": object_color,
                "shape_type": shape_type,
                "type": "default"
            }
            last_payload = payload

            signature = self._build_signature(
                base_shape=base_shape,
                rotation_center=rotation_center,
                rotation_angle=rotation_angle,
                initial_angle=initial_angle,
                shape_type=shape_type,
                object_color=object_color
            )
            if signature not in self._seen_signatures:
                self._seen_signatures.add(signature)
                return payload

        # Fallback if duplicates persisted
        return last_payload

    def _build_objects_metadata(self, task_data: dict) -> list:
        """
        Build objects metadata with symbol, center position, and color.
        
        Args:
            task_data: Task data dictionary
        
        Returns:
            List of objects, each containing symbol, center, color, initial_angle, and rotation_angle
        """
        objects = [{
            "symbol": task_data["shape_type"],
            "center": list(task_data["rotation_center"]),
            "color": list(task_data["object_color"]),
            "initial_angle": task_data["initial_angle"],
            "rotation_angle": task_data["rotation_angle"]
        }]
        
        return objects

    def _build_signature(
        self,
        base_shape: List[Tuple[int, int]],
        rotation_center: Tuple[int, int],
        rotation_angle: float,
        initial_angle: float,
        shape_type: str,
        object_color: Tuple[int, int, int]
    ) -> tuple:
        """Hashable signature used for deduplication within a dataset run."""
        base_sig = tuple((int(x), int(y)) for x, y in base_shape)
        return (
            base_sig,
            tuple(int(c) for c in rotation_center),
            round(rotation_angle, 2),
            round(initial_angle, 2),
            shape_type,
            tuple(int(c) for c in object_color)
        )

    def _create_base_shape(self, shape_type: str, size: int) -> list:
        """
        Create a base 2D polygon shape with one vertex at origin (0, 0).
        This ensures the shape connects to the rotation center point after transformation.

        Returns a list of (x, y) vertices defining the polygon.
        """
        half_size = size // 2

        if shape_type == "L_shape":
            # L-shaped polygon - starting from origin
            thickness = size // 3
            return [
                (0, 0),  # Connect to center point
                (thickness, 0),
                (thickness, half_size),
                (half_size, half_size),
                (half_size, half_size + thickness),
                (0, half_size + thickness)
            ]

        elif shape_type == "T_shape":
            # T-shaped polygon - starting from origin
            thickness = size // 3
            return [
                (0, 0),  # Connect to center point
                (half_size, 0),
                (half_size, thickness),
                (thickness // 2 + size // 6, thickness),
                (thickness // 2 + size // 6, half_size + thickness),
                (-thickness // 2 + size // 6, half_size + thickness),
                (-thickness // 2 + size // 6, thickness),
                (0, thickness)
            ]

        elif shape_type == "trapezoid":
            # Trapezoid shape - starting from origin
            offset = size // 4
            return [
                (0, 0),  # Connect to center point
                (half_size - offset, 0),
                (half_size, half_size),
                (0, half_size)
            ]

        elif shape_type == "arrow":
            # Arrow shape - starting from origin
            thickness = size // 4
            arrow_head = size // 3
            return [
                (0, 0),  # Connect to center point (arrow tip at origin)
                (half_size - arrow_head, 0),
                (half_size - arrow_head, -thickness),
                (half_size, thickness),
                (half_size - arrow_head, thickness * 3),
                (half_size - arrow_head, thickness * 2),
                (0, thickness * 2)
            ]

        else:
            # Default: simple rectangle - starting from origin
            return [
                (0, 0),  # Connect to center point
                (half_size, 0),
                (half_size, size),
                (0, size)
            ]

    def _rotate_and_translate_shape(
        self,
        shape: list,
        angle_degrees: float,
        center: tuple
    ) -> list:
        """
        Rotate shape around origin and translate to center position.

        Args:
            shape: List of (x, y) vertices
            angle_degrees: Rotation angle in degrees
            center: (cx, cy) translation target

        Returns:
            Transformed list of (x, y) vertices
        """
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        rotated_shape = []
        for x, y in shape:
            # Rotate around origin
            # Note: In screen coordinates (y-axis down), we need to negate sin
            # to match visual clockwise/counterclockwise expectations
            # Standard math rotation: positive angle = counterclockwise (y-axis up)
            # Screen rotation: positive angle = clockwise (y-axis down)
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a

            # Translate to center
            tx = rx + center[0]
            ty = ry + center[1]

            rotated_shape.append((tx, ty))

        return rotated_shape

    def _render_initial_state(self, task_data: dict) -> Image.Image:
        """
        Render initial state: shape at initial orientation + rotation center + target outline.

        Layout:
        - Light gray background
        - Solid colored 2D shape at initial position
        - Rotation center marked with a small circle
        - Target outline showing where the shape should end up
        """
        width, height = self.config.image_size
        img = Image.new('RGB', (width, height), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        rotation_center = task_data["rotation_center"]
        initial_shape = task_data["initial_shape"]
        final_shape = task_data["final_shape"]

        # Draw target outline first (unfilled, dashed effect, connected to center)
        self._draw_dashed_polygon(
            draw,
            final_shape,
            outline_color=(100, 100, 100),
            width=self.config.target_outline_width
        )
        
        # Draw line from center to first vertex of target outline to show connection
        if final_shape:
            draw.line(
                [rotation_center, final_shape[0]],
                fill=(100, 100, 100),
                width=1
            )

        # Draw initial object (filled, connected to center)
        draw.polygon(
            initial_shape,
            fill=task_data["object_color"],
            outline=(50, 50, 50),
            width=2
        )
        
        # Draw line from center to first vertex of initial shape to show connection
        if initial_shape:
            draw.line(
                [rotation_center, initial_shape[0]],
                fill=(50, 50, 50),
                width=2
            )

        # Draw rotation center marker (on top)
        self._draw_rotation_center(draw, rotation_center)

        return img

    def _render_final_state(self, task_data: dict) -> Image.Image:
        """
        Render final state: shape rotated to target position + rotation center.

        Shows the result of planar rotation transformation.
        """
        width, height = self.config.image_size
        img = Image.new('RGB', (width, height), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        rotation_center = task_data["rotation_center"]
        final_shape = task_data["final_shape"]

        # Draw final object (filled, at target position, connected to center)
        draw.polygon(
            final_shape,
            fill=task_data["object_color"],
            outline=(50, 50, 50),
            width=2
        )
        
        # Draw line from center to first vertex of final shape to show connection
        if final_shape:
            draw.line(
                [rotation_center, final_shape[0]],
                fill=(50, 50, 50),
                width=2
            )

        # Draw rotation center marker (on top)
        self._draw_rotation_center(draw, rotation_center)

        return img

    def _draw_rotation_center(self, draw: ImageDraw.Draw, center: tuple):
        """Draw rotation center point marker."""
        cx, cy = center
        radius = self.config.rotation_center_radius

        # Outer circle (white)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=2
        )

        # Inner dot (black)
        inner_radius = radius // 3
        draw.ellipse(
            [cx - inner_radius, cy - inner_radius, cx + inner_radius, cy + inner_radius],
            fill=(0, 0, 0),
            outline=None
        )

    def _draw_dashed_polygon(
        self,
        draw: ImageDraw.Draw,
        vertices: list,
        outline_color: tuple,
        width: int,
        dash_length: int = 10,
        gap_length: int = 5
    ):
        """
        Draw a polygon outline with dashed lines.

        Args:
            draw: ImageDraw object
            vertices: List of (x, y) points
            outline_color: RGB color tuple
            width: Line width
            dash_length: Length of each dash
            gap_length: Length of gaps between dashes
        """
        n = len(vertices)
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]

            self._draw_dashed_line(
                draw, x1, y1, x2, y2,
                outline_color, width, dash_length, gap_length
            )

    def _draw_dashed_line(
        self,
        draw: ImageDraw.Draw,
        x1: float, y1: float, x2: float, y2: float,
        color: tuple, width: int,
        dash_length: int = 10, gap_length: int = 5
    ):
        """Draw a dashed line between two points."""
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx * dx + dy * dy)

        if distance == 0:
            return

        # Normalize direction
        dx /= distance
        dy /= distance

        # Draw dashes
        current_distance = 0
        dash_pattern = dash_length + gap_length

        while current_distance < distance:
            # Start of dash
            start_x = x1 + dx * current_distance
            start_y = y1 + dy * current_distance

            # End of dash
            end_distance = min(current_distance + dash_length, distance)
            end_x = x1 + dx * end_distance
            end_y = y1 + dy * end_distance

            # Draw dash segment
            draw.line(
                [(start_x, start_y), (end_x, end_y)],
                fill=color,
                width=width
            )

            current_distance += dash_pattern

    def _generate_video(
        self,
        first_image: Image.Image,
        final_image: Image.Image,
        task_id: str,
        task_data: dict
    ) -> str:
        """Generate animation showing 2D shape planar rotation process."""
        temp_dir = Path(tempfile.gettempdir()) / f"{self.config.domain}_videos"
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_path = temp_dir / f"{task_id}_ground_truth.mp4"

        # Create animation frames
        frames = self._create_rotation_animation(task_data)

        result = self.video_generator.create_video_from_frames(
            frames,
            video_path
        )

        return str(result) if result else None

    def _create_rotation_animation(
        self,
        task_data: dict,
        hold_frames: int = 15,
        transition_frames: int = 40
    ) -> list:
        """
        Create animation frames showing 2D shape planar rotation process.

        Animation:
        1. Hold initial state (shape at start, target outline visible, rotation center marked)
        2. Transition: Shape gradually rotates in the plane around center point
        3. Hold final state (shape at target position, rotation center marked)
        """
        frames = []
        width, height = self.config.image_size

        # Initial frames - hold longer to show target outline clearly
        initial_frame = self._render_initial_state(task_data)
        for _ in range(hold_frames):
            frames.append(initial_frame.copy())

        # Transition frames: rotate object step by step
        rotation_angle = task_data["rotation_angle"]
        initial_angle = task_data["initial_angle"]
        base_shape = task_data["base_shape"]
        rotation_center = task_data["rotation_center"]
        object_color = task_data["object_color"]

        for i in range(transition_frames):
            progress = (i + 1) / transition_frames
            current_angle = initial_angle + rotation_angle * progress

            # Calculate current shape position
            current_shape = self._rotate_and_translate_shape(
                base_shape,
                current_angle,
                rotation_center
            )

            # Create frame
            img = Image.new('RGB', (width, height), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)

            # Draw target outline - keep it visible longer, only fade out near the end
            # Keep outline visible until progress > 0.8, then fade out gradually
            if progress < 0.8:
                # Keep outline fully visible for most of the rotation
                outline_alpha = 1.0
            else:
                # Fade out in the last 20% of the animation
                fade_progress = (progress - 0.8) / 0.2
                outline_alpha = max(0, 1 - fade_progress)
            
            if outline_alpha > 0.1:
                self._draw_dashed_polygon(
                    draw,
                    task_data["final_shape"],
                    outline_color=(100, 100, 100),
                    width=self.config.target_outline_width
                )
                # Draw line from center to first vertex of target outline to show connection
                if task_data["final_shape"]:
                    draw.line(
                        [rotation_center, task_data["final_shape"][0]],
                        fill=(100, 100, 100),
                        width=1
                    )

            # Draw rotating object (on top of outline, connected to center)
            draw.polygon(
                current_shape,
                fill=object_color,
                outline=(50, 50, 50),
                width=2
            )
            
            # Draw line from center to first vertex of current shape to show connection
            if current_shape:
                draw.line(
                    [rotation_center, current_shape[0]],
                    fill=(50, 50, 50),
                    width=2
                )

            # Draw rotation center marker (on top)
            self._draw_rotation_center(draw, rotation_center)

            frames.append(img)

        # Final frames
        final_frame = self._render_final_state(task_data)
        for _ in range(hold_frames):
            frames.append(final_frame.copy())

        return frames
