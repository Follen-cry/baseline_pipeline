"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ROTATION PUZZLE TASK GENERATOR                            ║
║                                                                               ║
║  Generates pipe puzzle tasks where squares with L-shaped pipe patterns      ║
║  need to be rotated to connect all pipe paths.                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import random
import math
import tempfile
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from PIL import Image, ImageDraw

from core import BaseGenerator, TaskPair, ImageRenderer
from core.video_utils import VideoGenerator
from .config import TaskConfig
from .prompts import get_prompt


class TaskGenerator(BaseGenerator):
    """
    Rotation puzzle task generator.
    
    Generates pipe puzzle tasks where 4 squares with L-shaped pipe patterns
    are arranged in a 2×2 grid. Squares need to be rotated to connect all pipes.
    """
    
    # L-shaped pipe patterns: (top, right, bottom, left) connections
    # 1 = connection, 0 = no connection
    PIPE_PATTERNS = [
        (1, 1, 0, 0),  # Top-Right L
        (0, 1, 1, 0),  # Right-Bottom L
        (0, 0, 1, 1),  # Bottom-Left L
        (1, 0, 0, 1),  # Left-Top L
    ]
    
    # Difficulty: number of squares that need rotation
    DIFFICULTY_ROTATIONS = {
        "easy": (1, 2),      # 1-2 squares need rotation
        "medium": (2, 3),    # 2-3 squares need rotation
        "hard": (3, 4),      # 3-4 squares need rotation
    }
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.renderer = ImageRenderer(image_size=config.canvas_size)
    
    def generate_task_pair(self, task_id: str) -> TaskPair:
        """Generate one rotation puzzle task pair."""
        
        # Generate puzzle data
        puzzle_data = self._generate_puzzle_data()
        
        # Render images
        first_image = self._render_first_frame(puzzle_data)
        final_image = self._render_final_frame(puzzle_data)
        
        # Generate video (optional)
        video_path = None
        if self.config.generate_videos and VideoGenerator.is_available():
            video_path = self._generate_video(first_image, final_image, task_id, puzzle_data)
        
        # Get prompt
        prompt = get_prompt()
        
        # Build object-centric metadata
        task_data = self._build_objects_metadata(puzzle_data)
        metadata = self._build_metadata(task_id, task_data)
        
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
    #  PUZZLE DATA GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def _generate_puzzle_data(self) -> dict:
        """Generate puzzle data with squares, positions, and rotations.
        
        Ensures that in the final state (all squares at 0°), pipes form a continuous square loop.
        For 2×2 grid to form a square loop:
        - Top-left (0): connects right and bottom
        - Top-right (1): connects left and bottom  
        - Bottom-left (2): connects right and top
        - Bottom-right (3): connects left and top
        """
        canvas_w, canvas_h = self.config.canvas_size
        square_size = self.config.square_size
        spacing = self.config.square_spacing
        
        # Randomly select a pipe color from the palette
        pipe_color = random.choice(self.config.pipe_color_palette)
        
        # Calculate grid layout (2×2)
        total_width = 2 * square_size + spacing
        total_height = 2 * square_size + spacing
        start_x = (canvas_w - total_width) // 2
        start_y = (canvas_h - total_height) // 2
        
        # Define patterns for each position that form a square loop at 0°
        # Pattern format: (top, right, bottom, left)
        solved_patterns = [
            (0, 1, 1, 0),  # Top-left (0): Right-Bottom L - connects right and bottom
            (0, 0, 1, 1),  # Top-right (1): Bottom-Left L - connects left and bottom
            (1, 1, 0, 0),  # Bottom-left (2): Top-Right L - connects right and top
            (1, 0, 0, 1),  # Bottom-right (3): Left-Top L - connects left and top
        ]
        
        # Determine how many squares need rotation based on difficulty
        min_rot, max_rot = self.DIFFICULTY_ROTATIONS.get(
            self.config.difficulty, (2, 3)
        )
        num_rotations = random.randint(min_rot, max_rot)
        
        # Generate squares with positions
        squares = []
        rotation_indices = random.sample(range(4), num_rotations)
        
        for i in range(4):
            row = i // 2
            col = i % 2
            x = start_x + col * (square_size + spacing)
            y = start_y + row * (square_size + spacing)
            
            # Get the pattern that forms the square loop at 0°
            pattern = solved_patterns[i]
            
            # Target angle is always 0° (solved state)
            target_angle = 0
            
            # Assign initial angle based on difficulty
            if i in rotation_indices:
                # This square needs rotation - assign a random angle from 1-359
                # Exclude 0° (which is the target) and multiples of 360
                initial_angle = random.randint(1, 359)
            else:
                # This square is already correct (at 0°)
                initial_angle = 0
            
            squares.append({
                "index": i,
                "position": (x, y),
                "pipe_pattern": pattern,
                "initial_angle": initial_angle,
                "target_angle": target_angle,
            })
        
        return {
            "squares": squares,
            "canvas_size": self.config.canvas_size,
            "square_size": square_size,
            "difficulty": self.config.difficulty,
            "pipe_color": pipe_color,
        }
    
    # ══════════════════════════════════════════════════════════════════════════
    #  IMAGE RENDERING
    # ══════════════════════════════════════════════════════════════════════════
    
    def _render_first_frame(self, puzzle_data: dict) -> Image.Image:
        """Render first frame with squares at initial rotations."""
        return self._render_frame(puzzle_data, use_initial_angles=True)
    
    def _render_final_frame(self, puzzle_data: dict) -> Image.Image:
        """Render final frame with squares at target rotations (all 0°)."""
        return self._render_frame(puzzle_data, use_initial_angles=False)
    
    def _render_frame(self, puzzle_data: dict, use_initial_angles: bool) -> Image.Image:
        """Render a frame with squares at specified angles."""
        canvas_w, canvas_h = puzzle_data["canvas_size"]
        img = Image.new('RGB', (canvas_w, canvas_h), self.config.background_color)
        draw = ImageDraw.Draw(img)
        
        square_size = puzzle_data["square_size"]
        pipe_color = puzzle_data["pipe_color"]
        
        for square in puzzle_data["squares"]:
            x, y = square["position"]
            pattern = square["pipe_pattern"]
            angle = square["initial_angle"] if use_initial_angles else square["target_angle"]
            
            # Draw square with pipe pattern at specified rotation
            self._draw_square_with_pipes(
                draw, x, y, square_size, pattern, angle, pipe_color
            )
        
        return img
    
    def _draw_square_with_pipes(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        size: int,
        pattern: Tuple[int, int, int, int],
        rotation: int,
        pipe_color: Tuple[int, int, int]
    ):
        """
        Draw a square with L-shaped pipe pattern at specified rotation.
        
        Args:
            draw: ImageDraw object
            x, y: Top-left corner of square
            size: Size of square
            pattern: (top, right, bottom, left) connections (1 or 0)
            rotation: Rotation angle in degrees (0, 15, 30, ..., 345)
            pipe_color: RGB color for the pipe
        """
        # Draw square background (white) with border
        draw.rectangle(
            [x, y, x + size, y + size],
            fill=(255, 255, 255),
            outline=self.config.square_border_color,
            width=self.config.square_border_width
        )
        
        # Create a temporary image for the pipe pattern at 0°
        # Reduced from 2x to 1.5x for better performance while maintaining quality
        temp_size = int(size * 1.5)  # 220 * 1.5 = 330px
        temp_img = Image.new('RGBA', (temp_size, temp_size), (255, 255, 255, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        
        # Draw L-shaped pipe at 0° rotation on temp image
        pipe_width = int(self.config.pipe_width * 1.5)  # Scale up proportionally
        half_width = pipe_width // 2
        center_x = temp_size // 2
        center_y = temp_size // 2
        edge_offset = pipe_width // 2
        
        # Find which two sides are connected based on pattern
        top, right, bottom, left = pattern
        connections = []
        if top:
            connections.append("top")
        if right:
            connections.append("right")
        if bottom:
            connections.append("bottom")
        if left:
            connections.append("left")
        
        # Draw L-shaped pipe on temp image
        if len(connections) == 2:
            self._draw_l_pipe_on_temp(
                temp_draw, 0, 0, temp_size, connections, pipe_color, pipe_width
            )
        
        # Rotate the temp image - use BILINEAR for faster performance
        if rotation != 0:
            temp_img = temp_img.rotate(-rotation, resample=Image.BILINEAR, expand=False)
        
        # Paste the rotated pipe onto the main image
        # First, create a white background at the target position
        bg = Image.new('RGB', (size, size), (255, 255, 255))
        
        # Resize temp_img to final size - use BILINEAR for faster performance
        temp_img_resized = temp_img.resize((size, size), Image.BILINEAR)
        
        # Extract just the pipe (non-white pixels) from temp_img_resized
        # Create a mask from the alpha channel
        if temp_img_resized.mode == 'RGBA':
            bg.paste(temp_img_resized, (0, 0), temp_img_resized)
        
        # Now paste bg onto the main image at position (x, y)
        from PIL import ImageDraw as ID
        main_img = draw._image
        main_img.paste(bg, (x, y))
    
    def _draw_l_pipe_on_temp(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        size: int,
        connections: List[str],
        color: Tuple[int, int, int],
        width: int
    ):
        """Draw an L-shaped pipe on a temporary image (for rotation)."""
        center_x = x + size // 2
        center_y = y + size // 2
        half_width = width // 2
        edge_offset = width // 2
        
        # Draw L-shaped path based on which two sides are connected
        if "top" in connections and "right" in connections:
            # Top-Right L
            draw.rectangle(
                [center_x - half_width, y + edge_offset, center_x + half_width, center_y + half_width],
                fill=color
            )
            draw.rectangle(
                [center_x - half_width, center_y - half_width, x + size - edge_offset, center_y + half_width],
                fill=color
            )
            draw.ellipse(
                [center_x - half_width, center_y - half_width, center_x + half_width, center_y + half_width],
                fill=color
            )
        
        elif "right" in connections and "bottom" in connections:
            # Right-Bottom L
            draw.rectangle(
                [center_x - half_width, center_y - half_width, x + size - edge_offset, center_y + half_width],
                fill=color
            )
            draw.rectangle(
                [center_x - half_width, center_y - half_width, center_x + half_width, y + size - edge_offset],
                fill=color
            )
            draw.ellipse(
                [center_x - half_width, center_y - half_width, center_x + half_width, center_y + half_width],
                fill=color
            )
        
        elif "bottom" in connections and "left" in connections:
            # Bottom-Left L
            draw.rectangle(
                [center_x - half_width, center_y - half_width, center_x + half_width, y + size - edge_offset],
                fill=color
            )
            draw.rectangle(
                [x + edge_offset, center_y - half_width, center_x + half_width, center_y + half_width],
                fill=color
            )
            draw.ellipse(
                [center_x - half_width, center_y - half_width, center_x + half_width, center_y + half_width],
                fill=color
            )
        
        elif "left" in connections and "top" in connections:
            # Left-Top L
            draw.rectangle(
                [x + edge_offset, center_y - half_width, center_x + half_width, center_y + half_width],
                fill=color
            )
            draw.rectangle(
                [center_x - half_width, y + edge_offset, center_x + half_width, center_y + half_width],
                fill=color
            )
            draw.ellipse(
                [center_x - half_width, center_y - half_width, center_x + half_width, center_y + half_width],
                fill=color
            )
    
    def _draw_l_pipe(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        size: int,
        connections: List[str],
        color: Tuple[int, int, int],
        width: int
    ):
        """Draw an L-shaped pipe connecting two adjacent sides."""
        center_x = x + size // 2
        center_y = y + size // 2
        half_width = width // 2
        
        # Define connection points on each side (at edge of square)
        edge_offset = width // 2  # Distance from edge
        
        # Draw L-shaped path based on which two sides are connected
        if "top" in connections and "right" in connections:
            # Top-Right L: vertical from top to center, horizontal from center to right
            # Vertical segment: top edge to center
            draw.rectangle(
                [center_x - half_width, y + edge_offset, center_x + half_width, center_y + half_width],
                fill=color
            )
            # Horizontal segment: center to right edge
            draw.rectangle(
                [center_x - half_width, center_y - half_width, x + size - edge_offset, center_y + half_width],
                fill=color
            )
            # Corner connection
            draw.ellipse(
                [center_x - half_width, center_y - half_width, center_x + half_width, center_y + half_width],
                fill=color
            )
        
        elif "right" in connections and "bottom" in connections:
            # Right-Bottom L: horizontal from right to center, vertical from center to bottom
            # Horizontal segment: right edge to center
            draw.rectangle(
                [center_x - half_width, center_y - half_width, x + size - edge_offset, center_y + half_width],
                fill=color
            )
            # Vertical segment: center to bottom edge
            draw.rectangle(
                [center_x - half_width, center_y - half_width, center_x + half_width, y + size - edge_offset],
                fill=color
            )
            # Corner connection
            draw.ellipse(
                [center_x - half_width, center_y - half_width, center_x + half_width, center_y + half_width],
                fill=color
            )
        
        elif "bottom" in connections and "left" in connections:
            # Bottom-Left L: vertical from bottom to center, horizontal from center to left
            # Vertical segment: bottom edge to center
            draw.rectangle(
                [center_x - half_width, center_y - half_width, center_x + half_width, y + size - edge_offset],
                fill=color
            )
            # Horizontal segment: center to left edge
            draw.rectangle(
                [x + edge_offset, center_y - half_width, center_x + half_width, center_y + half_width],
                fill=color
            )
            # Corner connection
            draw.ellipse(
                [center_x - half_width, center_y - half_width, center_x + half_width, center_y + half_width],
                fill=color
            )
        
        elif "left" in connections and "top" in connections:
            # Left-Top L: horizontal from left to center, vertical from center to top
            # Horizontal segment: left edge to center
            draw.rectangle(
                [x + edge_offset, center_y - half_width, center_x + half_width, center_y + half_width],
                fill=color
            )
            # Vertical segment: center to top edge
            draw.rectangle(
                [center_x - half_width, y + edge_offset, center_x + half_width, center_y + half_width],
                fill=color
            )
            # Corner connection
            draw.ellipse(
                [center_x - half_width, center_y - half_width, center_x + half_width, center_y + half_width],
                fill=color
            )
    
    # ══════════════════════════════════════════════════════════════════════════
    #  VIDEO GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def _generate_video(
        self,
        first_image: Image.Image,
        final_image: Image.Image,
        task_id: str,
        puzzle_data: dict
    ) -> Optional[str]:
        """Generate ground truth video showing smooth rotation of squares."""
        if not VideoGenerator.is_available():
            return None
        
        temp_dir = Path(tempfile.gettempdir()) / f"{self.config.domain}_videos"
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_path = temp_dir / f"{task_id}_ground_truth.mp4"
        
        # Generate rotation animation frames
        frames = self._create_rotation_animation_frames(puzzle_data)
        
        video_generator = VideoGenerator(fps=self.config.video_fps, output_format="mp4")
        result = video_generator.create_video_from_frames(frames, video_path)
        
        return str(result) if result else None
    
    def _create_rotation_animation_frames(
        self,
        puzzle_data: dict,
        hold_frames: int = 16,
        rotation_frames: int = 64
    ) -> List[Image.Image]:
        """Create animation frames showing smooth rotation from initial to final state.
        
        This method generates frames with smooth, continuous rotation animation.
        Each square rotates smoothly from its initial angle to target angle (0°).
        """
        frames = []
        
        # Hold initial position
        first_frame = self._render_first_frame(puzzle_data)
        for _ in range(hold_frames):
            frames.append(first_frame)
        
        # Calculate rotation data for each square
        squares = puzzle_data["squares"]
        rotation_data = []
        
        for square in squares:
            initial_angle = square["initial_angle"]
            target_angle = square["target_angle"]
            
            # Calculate shortest rotation path
            diff = target_angle - initial_angle
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            
            rotation_data.append({
                "initial": initial_angle,
                "target": target_angle,
                "diff": diff
            })
        
        # Generate smooth rotation frames
        for frame_idx in range(rotation_frames + 1):
            # Calculate progress (0.0 to 1.0)
            progress = frame_idx / rotation_frames if rotation_frames > 0 else 1.0
            
            # Apply easing function for smoother animation (ease-in-out)
            # This makes the rotation start slow, speed up in middle, and slow down at end
            if progress < 0.5:
                eased_progress = 2 * progress * progress
            else:
                eased_progress = 1 - 2 * (1 - progress) * (1 - progress)
            
            # Create frame data for this progress point
            step_data = puzzle_data.copy()
            step_data["squares"] = []
            
            for i, square in enumerate(squares):
                rot_data = rotation_data[i]
                
                # Calculate current angle based on progress
                current_angle = rot_data["initial"] + rot_data["diff"] * eased_progress
                
                # Normalize angle to 0-360 range
                current_angle = current_angle % 360
                
                new_square = square.copy()
                new_square["initial_angle"] = current_angle
                step_data["squares"].append(new_square)
            
            # Render frame for this progress point
            frame = self._render_frame(step_data, use_initial_angles=True)
            frames.append(frame)
        
        # Hold final position
        final_frame = self._render_final_frame(puzzle_data)
        for _ in range(hold_frames):
            frames.append(final_frame)
        
        return frames
    
    # ══════════════════════════════════════════════════════════════════════════
    #  METADATA BUILDING
    # ══════════════════════════════════════════════════════════════════════════
    
    def _build_objects_metadata(self, puzzle_data: dict) -> Dict[str, Any]:
        """
        Build object-centric metadata for rotation puzzle task.
        
        Args:
            puzzle_data: Puzzle data dictionary containing squares and puzzle info
            
        Returns:
            Dictionary with object-centric metadata
        """
        squares = puzzle_data["squares"]
        square_size = puzzle_data["square_size"]
        
        objects = []
        for square in squares:
            x, y = square["position"]
            # Calculate center coordinates
            center_x = round(x + square_size / 2, 2)
            center_y = round(y + square_size / 2, 2)
            
            # Calculate rotation angle (difference between initial and target)
            initial_angle = square["initial_angle"]
            target_angle = square["target_angle"]
            rotation_angle = (target_angle - initial_angle) % 360
            if rotation_angle > 180:
                rotation_angle -= 360
            
            # Convert pipe_pattern tuple to readable format
            top, right, bottom, left = square["pipe_pattern"]
            pipe_pattern = {
                "top": bool(top),
                "right": bool(right),
                "bottom": bool(bottom),
                "left": bool(left)
            }
            
            objects.append({
                "symbol": f"square_{square['index']}",
                "index": square["index"],
                "position": [x, y],
                "center": [center_x, center_y],
                "pipe_pattern": pipe_pattern,
                "initial_angle": initial_angle,
                "target_angle": target_angle,
                "rotation_angle": round(rotation_angle, 2)
            })
        
        # Build task-specific metadata
        task_data = {
            "difficulty": puzzle_data["difficulty"],
            "square_size": square_size,
            "pipe_color": puzzle_data["pipe_color"],
            "num_squares": len(squares),
            "objects": objects
        }
        
        return task_data
    