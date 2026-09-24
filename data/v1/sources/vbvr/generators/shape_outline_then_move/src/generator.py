"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SHAPE OUTLINE-THEN-MOVE TASK GENERATOR                    ║
║                                                                               ║
║  Generates two-step sequential tasks (A→B→C :: D→?→?)                        ║
║  Example: filled_circle → outline_circle → outline_circle_moved_up           ║
║           filled_square → outline_square → outline_square_moved_up            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import random
import tempfile
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Any

from core import BaseGenerator, TaskPair, ImageRenderer
from core.video_utils import VideoGenerator
from .config import TaskConfig
from .prompts import get_prompt


class TaskGenerator(BaseGenerator):
    """
    Shape outline-then-move task generator.
    
    Creates visual analogies in the format A→B→C :: D→?→?
    where shapes undergo two sequential transformations: fill-to-outline then vertical movement.
    """
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.renderer = ImageRenderer(image_size=config.image_size)
        
        # Initialize video generator if enabled
        self.video_generator = None
        if config.generate_videos and VideoGenerator.is_available():
            self.video_generator = VideoGenerator(fps=config.video_fps, output_format="mp4")
        
        # Shape definitions - expanded set for more variety
        self.base_shapes = [
            "square", "triangle", "circle", "diamond", "pentagon", "hexagon",
            "rectangle", "oval", "star", "heart", "cross", "arrow", "trapezoid",
            "rhombus", "octagon", "crescent", "plus", "minus", "L_shape", "T_shape"
        ]
        
        # Multiple colors for variety - from config (expanded to 400 colors)
        self.shape_colors = config.shape_colors
        
        # Fill styles - expanded with more variations
        self.fill_styles = {
            "filled": {"fill": True, "outline_width": 2},
            "outline": {"fill": False, "outline_width": 3},
            "thick_outline": {"fill": False, "outline_width": 5},
            "thin_outline": {"fill": False, "outline_width": 2}
        }
        
        # Movement directions and distances - expanded with more variations
        self.movements = {
            "center": 0,        # No movement (center position)
            "up": -60,          # Move up by 60 pixels
            "down": 60,         # Move down by 60 pixels
            "up_small": -40,    # Move up by 40 pixels
            "down_small": 40,   # Move down by 40 pixels
            "up_large": -80,    # Move up by 80 pixels
            "down_large": 80,   # Move down by 80 pixels
            "up_tiny": -25,     # Move up by 25 pixels
            "down_tiny": 25,    # Move down by 25 pixels
            "up_huge": -100,    # Move up by 100 pixels
            "down_huge": 100,   # Move down by 100 pixels
        }
        
        # Generate all valid transformation combinations dynamically
        self.valid_transformations = self._generate_all_valid_transformations()
        
        # Track generated combinations to prevent duplicates
        self.generated_combinations = set()
    
    def _generate_all_valid_transformations(self) -> List[Tuple[str, str, str, str]]:
        """Generate all valid transformation combinations dynamically."""
        transformations = []
        style_names = list(self.fill_styles.keys())
        move_names = list(self.movements.keys())
        
        # Generate all combinations where styles and movements are different
        for style_from in style_names:
            for style_to in style_names:
                if style_from != style_to:  # Styles must be different
                    for move_from in move_names:
                        for move_to in move_names:
                            if move_from != move_to:  # Movements must be different
                                transformations.append((style_from, style_to, move_from, move_to))
        
        return transformations
    
    def generate_task_pair(self, task_id: str) -> TaskPair:
        """Generate one shape outline-then-move task pair."""
        
        # Generate task data
        task_data = self._generate_task_data()
        
        # Render images
        first_image = self._render_initial_state(task_data)
        final_image = self._render_final_state(task_data)
        
        # Generate video (optional)
        video_path = None
        if self.config.generate_videos and self.video_generator:
            video_path = self._generate_video(first_image, final_image, task_id, task_data)
        
        # Select prompt with task-specific parameters
        prompt = get_prompt(
            task_type=task_data.get("transformation_type", "default"),
            source_shape=task_data.get("shape_a", "shape"),
            target_shape=task_data.get("shape_d", "shape"),
            movement_direction=task_data.get("move_to", "up"),
        )
        
        
        # Build metadata with objects (symbol, center, color, style, movement)
        objects = self._build_objects_metadata(task_data)
        
        optimized_task_data = {
            "transformation_type": task_data.get("transformation_type", "outline_then_move"),
            "style_from": task_data["style_from"],
            "style_to": task_data["style_to"],
            "move_from": task_data["move_from"],
            "move_to": task_data["move_to"],
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
    #  TASK DATA GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def _generate_task_data(self) -> Dict[str, Any]:
        """Generate outline-then-move transformation task data with duplicate prevention."""
        
        # Calculate total possible unique combinations
        num_shapes = len(self.base_shapes)
        num_transformations = len(self.valid_transformations)
        num_colors = len(self.shape_colors)
        max_unique_combinations = num_shapes * (num_shapes - 1) * num_transformations * num_colors
        
        # If we haven't exhausted all combinations, ensure uniqueness
        if len(self.generated_combinations) < max_unique_combinations:
            max_attempts = 1000  # Increase attempts for better coverage
            for attempt in range(max_attempts):
                # Select two different shapes for the analogy
                shape_example, shape_question = random.sample(self.base_shapes, 2)
                # Select a valid transformation sequence (style and movement changes)
                style_from, style_to, move_from, move_to = random.choice(self.valid_transformations)
                # Select color
                color = random.choice(self.shape_colors)
                
                # Create a unique identifier for this combination
                combination_key = (shape_example, shape_question, style_from, style_to, move_from, move_to, color)
                
                # Check if this combination has been used before
                if combination_key not in self.generated_combinations:
                    self.generated_combinations.add(combination_key)
                    return self._generate_outline_then_move_task(shape_example, shape_question, style_from, style_to, move_from, move_to, color)
            
            # If we still can't find a unique combination after many attempts,
            # generate all remaining combinations systematically
            return self._generate_systematic_unique_combination()
        
        # If we've exhausted unique combinations, allow duplicates but warn
        if len(self.generated_combinations) == max_unique_combinations:
            print(f"⚠️  Warning: Generated all {max_unique_combinations} unique combinations. Allowing duplicates for remaining tasks.")
        
        shape_example, shape_question = random.sample(self.base_shapes, 2)
        style_from, style_to, move_from, move_to = random.choice(self.valid_transformations)
        color = random.choice(self.shape_colors)
        return self._generate_outline_then_move_task(shape_example, shape_question, style_from, style_to, move_from, move_to, color)
    
    def _generate_systematic_unique_combination(self) -> Dict[str, Any]:
        """Generate a unique combination systematically when random selection fails."""
        # Generate all possible combinations and find one not yet used
        for shape_example in self.base_shapes:
            for shape_question in self.base_shapes:
                if shape_example == shape_question:
                    continue
                for style_from, style_to, move_from, move_to in self.valid_transformations:
                    for color in self.shape_colors:
                        combination_key = (shape_example, shape_question, style_from, style_to, move_from, move_to, color)
                        if combination_key not in self.generated_combinations:
                            self.generated_combinations.add(combination_key)
                            return self._generate_outline_then_move_task(shape_example, shape_question, style_from, style_to, move_from, move_to, color)
        
        # This should never happen if our math is correct
        raise RuntimeError("Failed to generate unique combination - this should not happen!")
    
    def _generate_outline_then_move_task(self, shape_example: str, shape_question: str, style_from: str, style_to: str, move_from: str, move_to: str, color: tuple) -> Dict[str, Any]:
        """Generate an outline-then-move transformation task."""
        
        return {
            "transformation_type": "outline_then_move",
            # Example sequence: A → B → C
            "shape_a": shape_example,  # Original
            "shape_b": shape_example,  # After step 1 (outline change)
            "shape_c": shape_example,  # After step 2 (movement)
            # Question sequence: D → ? → ?
            "shape_d": shape_question,  # Original
            "shape_e": shape_question,  # After step 1 (outline change) - first ?
            "shape_f": shape_question,  # After step 2 (movement) - second ?
            "style_from": style_from,
            "style_to": style_to,
            "move_from": move_from,
            "move_to": move_to,
            "color": color,
            "description": f"Step 1: {style_from} → {style_to}, Step 2: {move_from} → {move_to}"
        }
    
    def _build_objects_metadata(self, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build objects metadata with symbol, center position, color, style, and movement.
        
        Args:
            task_data: Task data dictionary
        
        Returns:
            List of objects (A, B, C, D, E, F), each containing symbol, center, color, style, and movement
        """
        width, height = self.config.image_size
        margin = self.config.margin
        base_shape_size = self.config.shape_size
        
        # Calculate positions (same as in _render_initial_state and _render_final_state)
        total_content_width = width - 2 * margin
        step_width = total_content_width // 3
        shape_spacing = step_width * 0.75
        arrow_width = step_width * 0.25
        
        # Vertical positioning: same as render functions
        safe_top = margin + base_shape_size // 2 + 120
        safe_bottom = height - margin - base_shape_size // 2 - 120
        top_row_y = max(safe_top, min(height // 3, safe_bottom - 200))
        bottom_row_y = max(safe_top + 200, min(2 * height // 3, safe_bottom))
        
        # Get movement offset
        move_offset = self.movements.get(task_data.get("move_to", "center"), 0)
        
        # Get color
        color_rgb = list(task_data.get("color", self.shape_colors[0]))
        
        objects = [
            # Example row: A → B → C
            {
                "symbol": task_data["shape_a"],
                "center": [margin + shape_spacing//2, top_row_y],
                "color": color_rgb,
                "style": task_data["style_from"],
                "movement": "center"
            },
            {
                "symbol": task_data["shape_b"],
                "center": [margin + shape_spacing + arrow_width + shape_spacing//2, top_row_y],
                "color": color_rgb,
                "style": task_data["style_to"],
                "movement": "center"
            },
            {
                "symbol": task_data["shape_c"],
                "center": [margin + 2*shape_spacing + 2*arrow_width + shape_spacing//2, top_row_y + move_offset],
                "color": color_rgb,
                "style": task_data["style_to"],
                "movement": task_data["move_to"]
            },
            # Question row: D → E → F
            {
                "symbol": task_data["shape_d"],
                "center": [margin + shape_spacing//2, bottom_row_y],
                "color": color_rgb,
                "style": task_data["style_from"],
                "movement": "center"
            },
            {
                "symbol": task_data["shape_e"],
                "center": [margin + shape_spacing + arrow_width + shape_spacing//2, bottom_row_y],
                "color": color_rgb,
                "style": task_data["style_to"],
                "movement": "center"
            },
            {
                "symbol": task_data["shape_f"],
                "center": [margin + 2*shape_spacing + 2*arrow_width + shape_spacing//2, bottom_row_y + move_offset],
                "color": color_rgb,
                "style": task_data["style_to"],
                "movement": task_data["move_to"]
            }
        ]
        
        return objects
    
    # ══════════════════════════════════════════════════════════════════════════
    #  IMAGE RENDERING
    # ══════════════════════════════════════════════════════════════════════════
    
    def _render_initial_state(self, task_data: Dict[str, Any]) -> Image.Image:
        """Render the initial state with A→B→C :: D→?→? layout."""
        img = self.renderer.create_blank_image()
        draw = ImageDraw.Draw(img)
        
        width, height = self.config.image_size
        margin = self.config.margin
        base_shape_size = self.config.shape_size
        
        # Layout positions for sequential format optimized for 1:1 (1024x1024) canvas
        # A  →  B  →  C
        # D  →  ?  →  ?
        
        # Calculate safe positions to avoid overlap and boundary issues
        # For 1024x1024: use center area with safe margins
        total_content_width = width - 2 * margin
        step_width = total_content_width // 3
        shape_spacing = step_width * 0.75  # Leave space between shapes
        arrow_width = step_width * 0.25
        
        # Vertical positioning: center the two rows with safe spacing
        # Top row: upper third, bottom row: lower third (with center buffer)
        top_row_y = height // 3
        bottom_row_y = 2 * height // 3
        
        # Ensure shapes don't overlap and stay within bounds
        # Check max movement: largest is 100px, shape radius is base_shape_size//2 = 40
        # So max extent from center: 40 + 100 = 140px
        # Safe zone: margin + 140 < position < height - margin - 140
        safe_top = margin + base_shape_size // 2 + 120  # 40 + 120 = 160
        safe_bottom = height - margin - base_shape_size // 2 - 120  # 1024 - 40 - 120 = 864
        
        # Adjust row positions to be within safe zone
        top_row_y = max(safe_top, min(top_row_y, safe_bottom - 200))
        bottom_row_y = max(safe_top + 200, min(bottom_row_y, safe_bottom))
        
        positions = {
            # Example row (top)
            "A": (margin + shape_spacing//2, top_row_y),
            "arrow1": (margin + shape_spacing + arrow_width//2, top_row_y),
            "B": (margin + shape_spacing + arrow_width + shape_spacing//2, top_row_y),
            "arrow2": (margin + 2*shape_spacing + arrow_width + arrow_width//2, top_row_y),
            "C": (margin + 2*shape_spacing + 2*arrow_width + shape_spacing//2, top_row_y),
            
            # Question row (bottom)
            "D": (margin + shape_spacing//2, bottom_row_y),
            "arrow3": (margin + shape_spacing + arrow_width//2, bottom_row_y),
            "question1": (margin + shape_spacing + arrow_width + shape_spacing//2, bottom_row_y),
            "arrow4": (margin + 2*shape_spacing + arrow_width + arrow_width//2, bottom_row_y),
            "question2": (margin + 2*shape_spacing + 2*arrow_width + shape_spacing//2, bottom_row_y)
        }
        
        # Draw example sequence: A → B → C
        color = task_data["color"]
        # A and B should always be at center (baseline) - only C gets the movement
        self._draw_shape_at_position(draw, task_data["shape_a"], positions["A"], base_shape_size,
                                   task_data["style_from"], "center", color)  # A: Original (center)
        self._draw_arrow(draw, positions["arrow1"])
        self._draw_shape_at_position(draw, task_data["shape_b"], positions["B"], base_shape_size,
                                   task_data["style_to"], "center", color)  # B: Style changed (center)
        self._draw_arrow(draw, positions["arrow2"])
        self._draw_shape_at_position(draw, task_data["shape_c"], positions["C"], base_shape_size,
                                   task_data["style_to"], task_data["move_to"], color)  # C: Style + Position changed
        
        # Draw question sequence: D → ? → ?
        # D should always be at baseline (center) position, not affected by move_from
        self._draw_shape_at_position(draw, task_data["shape_d"], positions["D"], base_shape_size,
                                   task_data["style_from"], "center", color)  # D: Original (always center)
        self._draw_arrow(draw, positions["arrow3"])
        self._draw_question_mark(draw, positions["question1"])  # First ?
        self._draw_arrow(draw, positions["arrow4"])
        self._draw_question_mark(draw, positions["question2"])  # Second ?
        
        return img
    
    def _render_final_state(self, task_data: Dict[str, Any]) -> Image.Image:
        """Render the final state with both answers revealed."""
        img = self.renderer.create_blank_image()
        draw = ImageDraw.Draw(img)
        
        width, height = self.config.image_size
        margin = self.config.margin
        base_shape_size = self.config.shape_size
        
        # Same layout as initial state (optimized for 1:1 canvas)
        total_content_width = width - 2 * margin
        step_width = total_content_width // 3
        shape_spacing = step_width * 0.75
        arrow_width = step_width * 0.25
        
        # Vertical positioning: same as initial state
        safe_top = margin + base_shape_size // 2 + 120
        safe_bottom = height - margin - base_shape_size // 2 - 120
        top_row_y = max(safe_top, min(height // 3, safe_bottom - 200))
        bottom_row_y = max(safe_top + 200, min(2 * height // 3, safe_bottom))
        
        positions = {
            # Example row (top)
            "A": (margin + shape_spacing//2, top_row_y),
            "arrow1": (margin + shape_spacing + arrow_width//2, top_row_y),
            "B": (margin + shape_spacing + arrow_width + shape_spacing//2, top_row_y),
            "arrow2": (margin + 2*shape_spacing + arrow_width + arrow_width//2, top_row_y),
            "C": (margin + 2*shape_spacing + 2*arrow_width + shape_spacing//2, top_row_y),
            
            # Question row (bottom)
            "D": (margin + shape_spacing//2, bottom_row_y),
            "arrow3": (margin + shape_spacing + arrow_width//2, bottom_row_y),
            "E": (margin + shape_spacing + arrow_width + shape_spacing//2, bottom_row_y),
            "arrow4": (margin + 2*shape_spacing + arrow_width + arrow_width//2, bottom_row_y),
            "F": (margin + 2*shape_spacing + 2*arrow_width + shape_spacing//2, bottom_row_y)
        }
        
        # Draw example sequence: A → B → C (same as initial)
        color = task_data["color"]
        # A and B should always be at center (baseline) - only C gets the movement
        self._draw_shape_at_position(draw, task_data["shape_a"], positions["A"], base_shape_size,
                                   task_data["style_from"], "center", color)  # A: Original (center)
        self._draw_arrow(draw, positions["arrow1"])
        self._draw_shape_at_position(draw, task_data["shape_b"], positions["B"], base_shape_size,
                                   task_data["style_to"], "center", color)  # B: Style changed (center)
        self._draw_arrow(draw, positions["arrow2"])
        self._draw_shape_at_position(draw, task_data["shape_c"], positions["C"], base_shape_size,
                                   task_data["style_to"], task_data["move_to"], color)  # C: Style + Position changed
        
        # Draw answer sequence: D → E → F (answers revealed)
        # D and E should always be at center (baseline) - only F gets the movement
        self._draw_shape_at_position(draw, task_data["shape_d"], positions["D"], base_shape_size,
                                   task_data["style_from"], "center", color)  # D: Original (center)
        self._draw_arrow(draw, positions["arrow3"])
        self._draw_shape_at_position(draw, task_data["shape_e"], positions["E"], base_shape_size,
                                   task_data["style_to"], "center", color)  # E: Style changed (center)
        self._draw_arrow(draw, positions["arrow4"])
        self._draw_shape_at_position(draw, task_data["shape_f"], positions["F"], base_shape_size,
                                   task_data["style_to"], task_data["move_to"], color)  # F: Style + Position changed
        
        return img
    
    def _draw_shape_at_position(self, draw: ImageDraw.Draw, shape: str, position: Tuple[int, int], base_size: int, style_name: str, move_name: str, color: tuple = None):
        """Draw a shape at the specified position with the given style and movement offset."""
        x, y = position
        
        # Use provided color or default
        if color is None:
            color = self.shape_colors[0]  # Default to first color
        
        # Apply movement offset
        if move_name in self.movements:
            y += self.movements[move_name]
        
        # Boundary check: ensure shape stays within image bounds
        width, height = self.config.image_size
        margin = self.config.margin
        half_size = base_size // 2
        
        # Clamp position to safe bounds
        x = max(margin + half_size, min(x, width - margin - half_size))
        y = max(margin + half_size, min(y, height - margin - half_size))
        
        style_config = self.fill_styles[style_name]
        fill_color = color if style_config["fill"] else None
        outline_color = color
        outline_width = style_config["outline_width"]
        
        self._draw_base_shape(draw, shape, x, y, base_size, fill_color, outline_color, outline_width)
    
    def _draw_base_shape(self, draw: ImageDraw.Draw, shape: str, x: int, y: int, size: int, fill_color, outline_color, outline_width: int):
        """Draw a basic geometric shape with specified fill and outline."""
        half_size = size // 2
        
        if shape == "square":
            draw.rectangle([x-half_size, y-half_size, x+half_size, y+half_size], 
                         fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "circle":
            draw.ellipse([x-half_size, y-half_size, x+half_size, y+half_size], 
                        fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "triangle":
            points = [
                (x, y-half_size),  # top
                (x-half_size, y+half_size),  # bottom left
                (x+half_size, y+half_size)   # bottom right
            ]
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "diamond":
            points = [
                (x, y-half_size),  # top
                (x+half_size, y),  # right
                (x, y+half_size),  # bottom
                (x-half_size, y)   # left
            ]
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "pentagon":
            points = []
            for i in range(5):
                angle = i * 2 * math.pi / 5 - math.pi/2  # Start from top
                px = x + half_size * math.cos(angle)
                py = y + half_size * math.sin(angle)
                points.append((px, py))
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "hexagon":
            points = []
            for i in range(6):
                angle = i * 2 * math.pi / 6
                px = x + half_size * math.cos(angle)
                py = y + half_size * math.sin(angle)
                points.append((px, py))
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "rectangle":
            # Rectangle (wider than tall)
            width_factor = 1.4
            rect_width = int(half_size * width_factor)
            rect_height = int(half_size * 0.7)
            draw.rectangle([x-rect_width, y-rect_height, x+rect_width, y+rect_height], 
                         fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "oval":
            # Oval (wider than tall)
            width_factor = 1.4
            oval_width = int(half_size * width_factor)
            oval_height = int(half_size * 0.7)
            draw.ellipse([x-oval_width, y-oval_height, x+oval_width, y+oval_height], 
                        fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "star":
            # 5-pointed star
            points = []
            outer_radius = half_size
            inner_radius = half_size * 0.4
            
            for i in range(10):  # 5 outer + 5 inner points
                if i % 2 == 0:  # Outer points
                    angle = i * math.pi / 5 - math.pi/2
                    px = x + outer_radius * math.cos(angle)
                    py = y + outer_radius * math.sin(angle)
                else:  # Inner points
                    angle = i * math.pi / 5 - math.pi/2
                    px = x + inner_radius * math.cos(angle)
                    py = y + inner_radius * math.sin(angle)
                points.append((px, py))
            
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "heart":
            # Heart shape using curves (approximate with polygon)
            points = [
                (x, y + half_size),                    # bottom point
                (x - half_size*0.7, y),              # left curve
                (x - half_size*0.3, y - half_size*0.5), # left top
                (x, y - half_size*0.2),              # center top
                (x + half_size*0.3, y - half_size*0.5),  # right top
                (x + half_size*0.7, y),               # right curve
            ]
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "cross":
            # Cross shape
            thickness = half_size // 4
            # Vertical bar
            draw.rectangle([x-thickness, y-half_size, x+thickness, y+half_size],
                         fill=fill_color, outline=outline_color, width=outline_width)
            # Horizontal bar
            draw.rectangle([x-half_size, y-thickness, x+half_size, y+thickness],
                         fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "arrow":
            # Arrow pointing right
            points = [
                (x-half_size, y-half_size//2),  # left top
                (x, y-half_size//2),            # middle top
                (x, y-half_size),               # tip top
                (x+half_size, y),               # tip point
                (x, y+half_size),               # tip bottom
                (x, y+half_size//2),            # middle bottom
                (x-half_size, y+half_size//2)   # left bottom
            ]
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "trapezoid":
            # Trapezoid (wider at bottom)
            top_width = half_size // 2
            points = [
                (x-top_width, y-half_size),     # top left
                (x+top_width, y-half_size),     # top right
                (x+half_size, y+half_size),     # bottom right
                (x-half_size, y+half_size)      # bottom left
            ]
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "rhombus":
            # Rhombus (diamond with different proportions)
            points = [
                (x, y-half_size),               # top
                (x+half_size*0.7, y),           # right
                (x, y+half_size),               # bottom
                (x-half_size*0.7, y)            # left
            ]
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "octagon":
            # Regular octagon
            points = []
            for i in range(8):
                angle = i * 2 * math.pi / 8
                px = x + half_size * math.cos(angle)
                py = y + half_size * math.sin(angle)
                points.append((px, py))
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "crescent":
            # Crescent moon shape (two overlapping circles)
            # Draw larger circle
            draw.ellipse([x-half_size, y-half_size, x+half_size, y+half_size],
                        fill=fill_color, outline=outline_color, width=outline_width)
            # Draw smaller circle to create crescent (using background color)
            offset = half_size // 3
            smaller_radius = int(half_size * 0.7)
            draw.ellipse([x-smaller_radius+offset, y-smaller_radius, x+smaller_radius+offset, y+smaller_radius],
                        fill=(255,255,255), outline=outline_color, width=outline_width)
        
        elif shape == "plus":
            # Plus sign (thicker cross)
            thickness = half_size // 3
            # Vertical bar
            draw.rectangle([x-thickness, y-half_size, x+thickness, y+half_size],
                         fill=fill_color, outline=outline_color, width=outline_width)
            # Horizontal bar
            draw.rectangle([x-half_size, y-thickness, x+half_size, y+thickness],
                         fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "minus":
            # Minus sign (horizontal bar)
            thickness = half_size // 4
            draw.rectangle([x-half_size, y-thickness, x+half_size, y+thickness],
                         fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "L_shape":
            # L shape
            thickness = half_size // 3
            # Vertical part
            draw.rectangle([x-half_size, y-half_size, x-half_size+thickness, y+half_size],
                         fill=fill_color, outline=outline_color, width=outline_width)
            # Horizontal part
            draw.rectangle([x-half_size, y+half_size-thickness, x+half_size, y+half_size],
                         fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "T_shape":
            # T shape
            thickness = half_size // 3
            # Horizontal top part
            draw.rectangle([x-half_size, y-half_size, x+half_size, y-half_size+thickness],
                         fill=fill_color, outline=outline_color, width=outline_width)
            # Vertical part
            draw.rectangle([x-thickness//2, y-half_size, x+thickness//2, y+half_size],
                         fill=fill_color, outline=outline_color, width=outline_width)
    
    def _draw_arrow(self, draw: ImageDraw.Draw, position: Tuple[int, int]):
        """Draw a right-pointing arrow."""
        x, y = position
        length = 40  # Shorter arrows for sequential layout
        
        # Arrow shaft
        draw.line([x-length//2, y, x+length//2-8, y], fill=(0,0,0), width=2)
        
        # Arrow head
        points = [
            (x+length//2, y),
            (x+length//2-10, y-6),
            (x+length//2-10, y+6)
        ]
        draw.polygon(points, fill=(0,0,0))
    
    def _draw_question_mark(self, draw: ImageDraw.Draw, position: Tuple[int, int]):
        """Draw a question mark."""
        x, y = position
        size = self.config.question_mark_size
        
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except:
            font = ImageFont.load_default()
        
        # Get text bounds for centering
        bbox = draw.textbbox((0, 0), "?", font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        text_x = x - w // 2
        text_y = y - h // 2
        
        draw.text((text_x, text_y), "?", font=font, fill=(100, 100, 100))
    
    # ══════════════════════════════════════════════════════════════════════════
    #  VIDEO GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def _generate_video(self, first_image: Image.Image, final_image: Image.Image, task_id: str, task_data: Dict[str, Any]) -> str:
        """Generate ground truth video showing the transformation."""
        temp_dir = Path(tempfile.gettempdir()) / f"{self.config.domain}_videos"
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_path = temp_dir / f"{task_id}_ground_truth.mp4"
        
        # Create animation frames
        frames = self._create_transformation_frames(first_image, final_image, task_data)
        
        result = self.video_generator.create_video_from_frames(frames, video_path)
        return str(result) if result else None
    
    def _create_transformation_frames(self, first_image: Image.Image, final_image: Image.Image, task_data: Dict[str, Any], hold_frames: int = 12, step_frames: int = 20) -> List[Image.Image]:
        """Create animation frames showing the two-step sequential transformation."""
        frames = []
        
        # Hold initial state
        for _ in range(hold_frames):
            frames.append(first_image.copy())
        
        # Create two-step animation: first ? then second ?
        frames.extend(self._create_sequential_morph_frames(task_data, step_frames))
        
        # Hold final state
        for _ in range(hold_frames):
            frames.append(final_image.copy())
        
        return frames
    
    def _create_sequential_morph_frames(self, task_data: Dict[str, Any], step_frames: int) -> List[Image.Image]:
        """Create frames showing the sequential two-step transformation."""
        frames = []
        
        width, height = self.config.image_size
        margin = self.config.margin
        base_shape_size = self.config.shape_size
        
        # Calculate positions matching the layout (optimized for 1:1 canvas)
        total_content_width = width - 2 * margin
        step_width = total_content_width // 3
        shape_spacing = step_width * 0.75
        arrow_width = step_width * 0.25
        
        # Use same safe positioning as render functions
        safe_top = margin + base_shape_size // 2 + 120
        safe_bottom = height - margin - base_shape_size // 2 - 120
        bottom_row_y = max(safe_top + 200, min(2 * height // 3, safe_bottom))
        
        question1_pos = (margin + shape_spacing + arrow_width + shape_spacing//2, bottom_row_y)
        question2_pos = (margin + 2*shape_spacing + 2*arrow_width + shape_spacing//2, bottom_row_y)
        
        shape_d = task_data["shape_d"]
        style_from_config = self.fill_styles[task_data["style_from"]]
        style_to_config = self.fill_styles[task_data["style_to"]]
        move_offset = self.movements.get(task_data["move_to"], 0)
        
        # Step 1: Reveal first ? (fill-to-outline change)
        for i in range(step_frames):
            img = self._render_static_elements(task_data, base_shape_size, shape_spacing, arrow_width, margin, width, height)
            draw = ImageDraw.Draw(img)
            
            # Interpolate fill-to-outline for first question mark
            progress = i / (step_frames - 1) if step_frames > 1 else 1.0
            
            # Create intermediate fill style
            if progress < 0.5:
                # First half: gradually reduce fill opacity
                color = task_data["color"]
                fill_alpha = int(255 * (1 - progress * 2))
                fill_color = (*color, fill_alpha)
                outline_width = style_from_config["outline_width"]
            else:
                # Second half: no fill, just outline
                fill_color = None
                outline_width = style_to_config["outline_width"]
            
            # Draw first answer (style changed, no movement yet)
            self._draw_base_shape(draw, shape_d, question1_pos[0], question1_pos[1], base_shape_size,
                                fill_color, color, outline_width)
            
            # Keep second question mark
            self._draw_question_mark(draw, question2_pos)
            
            frames.append(img)
        
        # Step 2: Reveal second ? (movement)
        for i in range(step_frames):
            img = self._render_static_elements(task_data, base_shape_size, shape_spacing, arrow_width, margin, width, height)
            draw = ImageDraw.Draw(img)
            
            # First answer is now complete (style changed, no movement)
            # Apply the "move_from" position (should be center/no movement for first answer)
            color = task_data["color"]
            fill_color = None
            outline_width = style_to_config["outline_width"]
            
            # First answer should always stay at center position (no movement offset)
            first_answer_y = question1_pos[1]
            # Don't apply move_from offset - first answer stays at baseline
            
            self._draw_base_shape(draw, shape_d, question1_pos[0], first_answer_y, base_shape_size,
                                fill_color, color, outline_width)
            
            # Interpolate movement for second question mark
            progress = i / (step_frames - 1) if step_frames > 1 else 1.0
            current_y_offset = move_offset * progress
            current_y = question2_pos[1] + current_y_offset
            
            # Draw second answer (style changed + movement)
            self._draw_base_shape(draw, shape_d, question2_pos[0], current_y, base_shape_size,
                                fill_color, color, outline_width)
            
            frames.append(img)
        
        return frames
    
    def _render_static_elements(self, task_data: Dict[str, Any], base_shape_size: int, shape_spacing: int, arrow_width: int, margin: int, width: int, height: int) -> Image.Image:
        """Render the static elements that don't change during animation."""
        img = self.renderer.create_blank_image()
        draw = ImageDraw.Draw(img)
        
        # Use same layout as other rendering functions (optimized for 1:1 canvas)
        total_content_width = width - 2 * margin
        step_width = total_content_width // 3
        shape_spacing = step_width * 0.75
        arrow_width = step_width * 0.25
        
        # Use same safe positioning
        safe_top = margin + base_shape_size // 2 + 120
        safe_bottom = height - margin - base_shape_size // 2 - 120
        top_row_y = max(safe_top, min(height // 3, safe_bottom - 200))
        bottom_row_y = max(safe_top + 200, min(2 * height // 3, safe_bottom))
        
        positions = {
            # Example row (top) - these never change
            "A": (margin + shape_spacing//2, top_row_y),
            "arrow1": (margin + shape_spacing + arrow_width//2, top_row_y),
            "B": (margin + shape_spacing + arrow_width + shape_spacing//2, top_row_y),
            "arrow2": (margin + 2*shape_spacing + arrow_width + arrow_width//2, top_row_y),
            "C": (margin + 2*shape_spacing + 2*arrow_width + shape_spacing//2, top_row_y),
            
            # Question row (bottom) - D and arrows are static
            "D": (margin + shape_spacing//2, bottom_row_y),
            "arrow3": (margin + shape_spacing + arrow_width//2, bottom_row_y),
            "arrow4": (margin + 2*shape_spacing + arrow_width + arrow_width//2, bottom_row_y),
        }
        
        # Draw static example sequence: A → B → C
        color = task_data["color"]
        # A and B should always be at center (baseline) - only C gets the movement
        self._draw_shape_at_position(draw, task_data["shape_a"], positions["A"], base_shape_size,
                                   task_data["style_from"], "center", color)
        self._draw_arrow(draw, positions["arrow1"])
        self._draw_shape_at_position(draw, task_data["shape_b"], positions["B"], base_shape_size,
                                   task_data["style_to"], "center", color)
        self._draw_arrow(draw, positions["arrow2"])
        self._draw_shape_at_position(draw, task_data["shape_c"], positions["C"], base_shape_size,
                                   task_data["style_to"], task_data["move_to"], color)
        
        # Draw static question elements: D and arrows
        # D should always be at baseline (center) position
        self._draw_shape_at_position(draw, task_data["shape_d"], positions["D"], base_shape_size,
                                   task_data["style_from"], "center", color)
        self._draw_arrow(draw, positions["arrow3"])
        self._draw_arrow(draw, positions["arrow4"])
        
        return img
