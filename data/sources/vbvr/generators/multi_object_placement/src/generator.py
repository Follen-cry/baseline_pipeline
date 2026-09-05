"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           TASK-SPECIFIC GENERATOR                             ║
║                                                                               ║
║  Implement your task-specific generation logic here.                          ║
║  This generator will be called by the core framework to create samples.       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image, ImageDraw

from src.config import TaskConfig
from core.base_generator import BaseGenerator
from core.schemas import TaskPair
from core.video_utils import VideoGenerator
from src.prompts import get_prompt


class TaskGenerator(BaseGenerator):
    """
    Generator for multi-object placement tasks.
    
    Generates tasks where objects need to be placed on corresponding colored markers.
    """
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.color_map = {
            "red": (255, 100, 100),
            "green": (100, 255, 100),
            "blue": (100, 100, 255),
            "yellow": (255, 255, 100),
            "purple": (255, 100, 255),
            "orange": (255, 165, 100)
        }
    
    def generate_task_pair(self, task_id: str) -> TaskPair:
        """Generate a single multi-object placement task."""
        
        # Generate objects and markers (with strict non-occlusion constraints)
        objects, markers = self._generate_objects_and_markers()
        
        # Create initial and final frames
        initial_frame = self._create_frame(objects, markers, show_objects=True, show_markers=True)
        final_frame = self._create_frame(objects, markers, show_objects=True, show_markers=True, final_positions=True)
        
        # Generate animation if enabled
        video_path = None
        if self.config.generate_videos and VideoGenerator.is_available():
            # Generate animation frames
            animation_frames = self._generate_animation_frames(objects, markers)
            
            # Create video directory
            video_dir = self.config.output_dir / f"{self.config.domain}_task" / task_id
            video_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate video
            video_generator = VideoGenerator(fps=self.config.video_fps, output_format="mp4")
            video_path = video_generator.create_video_from_frames(
                animation_frames, 
                video_dir / "ground_truth.mp4",
                size=self.config.image_size
            )
        
        # Get prompt
        prompt = get_prompt()
        
        # Build task_data dict from task parameters
        task_data = {
            "objects": [
                {
                    "id": obj["id"],
                    "shape": obj["shape"],
                    "color": obj["color"],
                    "size": obj["size"],
                    "position": obj["position"],
                }
                for obj in objects
            ],
            "markers": [
                {
                    "id": marker["id"],
                    "color": marker["color"],
                    "size": marker["size"],
                    "position": marker["position"],
                }
                for marker in markers
            ],
        }
        
        # Build metadata
        metadata = self._build_metadata(task_id, task_data)
        
        
        
        return TaskPair(
            task_id=task_id,
            domain=self.config.domain,
            prompt=prompt,
            first_image=initial_frame,
            final_image=final_frame,
            ground_truth_video=str(video_path) if video_path else None,
            metadata=metadata
        )
    
    def _generate_objects_and_markers(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate objects and their corresponding markers."""

        last_error: Optional[Exception] = None

        # Rectangle appears visually smaller than square under the current drawing rule:
        # - square area ~ (2s)^2 = 4s^2
        # - rectangle area ~ (1.5s) * (s) = 1.5s^2
        # To match square's visual footprint while keeping the rectangle aspect ratio,
        # scale rectangle size by sqrt(4/1.5) ≈ 1.633.
        RECT_SIZE_SCALE = 1.633

        # We may need to retry sampling if strict spacing cannot be satisfied.
        for _attempt in range(60):
            try:
                num_objects = random.randint(*self.config.num_objects_range)

                # Randomly select shapes and colors
                shapes = random.sample(self.config.shapes, min(num_objects, len(self.config.shapes)))
                colors = random.sample(self.config.colors, num_objects)

                objects: List[Dict[str, Any]] = []
                markers: List[Dict[str, Any]] = []

                # Track placed items for collision checking (include shape for better radius estimation)
                placed_items: List[Dict[str, Any]] = []

                # Use a uniform size for all objects (use middle of the range)
                min_size, max_size = self.config.object_size_range
                uniform_size = (min_size + max_size) // 2

                # 1) Generate all object initial positions
                for i in range(num_objects):
                    shape = shapes[i % len(shapes)]
                    color = colors[i]

                    size = uniform_size
                    if shape == "rectangle":
                        size = max(1, int(round(uniform_size * RECT_SIZE_SCALE)))
                    obj_x, obj_y = self._find_non_colliding_position(size, placed_items, is_object=True)

                    obj = {
                        "id": i,
                        "shape": shape,
                        "color": color,
                        "size": size,
                        "position": (obj_x, obj_y),
                    }
                    objects.append(obj)
                    placed_items.append(
                        {
                            "position": (obj_x, obj_y),
                            "size": size,
                            "is_marker": False,
                            "shape": shape,
                        }
                    )

                # 2) Determine uniform marker size (star arm length)
                min_object_size = min(obj["size"] for obj in objects)
                marker_size = int(min_object_size * self.config.marker_size_ratio)

                # 3) Generate marker positions using smart layout to prevent final frame overlap
                marker_positions = self._generate_smart_marker_layout(
                    len(objects), marker_size, placed_items, objects
                )

                # 4) Create marker objects with generated positions
                for i, obj in enumerate(objects):
                    marker_x, marker_y = marker_positions[i]
                    marker = {
                        "id": i,
                        "color": obj["color"],
                        "size": marker_size,
                        "position": (marker_x, marker_y),
                    }
                    markers.append(marker)

                # 5) Final strict validation: no occlusion in initial or final (except object over its own marker).
                if not self._validate_no_occlusion(objects, markers):
                    raise ValueError("Layout validation failed (occlusion detected). Retrying...")

                return objects, markers
            except Exception as e:
                last_error = e
                continue

        raise ValueError(f"Could not generate a valid non-occluding layout after many attempts: {last_error}")
    

    
    def _find_non_colliding_position(self, size: int, placed_items: List[Dict], is_object: bool = False) -> Tuple[int, int]:
        """Find a non-colliding position for a new item."""
        
        image_width, image_height = self.config.image_size
        # Use a conservative margin based on effective radius to keep full shape in-frame.
        margin = int(self._effective_radius(size, shape="square")) + 10
        
        for attempt in range(200):  # Try up to 200 times
            if is_object:
                # Objects need more space from edges
                x = random.randint(margin, image_width - margin)
                y = random.randint(margin, image_height - margin)
            else:
                # Markers can be closer to edges
                x = random.randint(margin, image_width - margin)
                y = random.randint(margin, image_height - margin)
            
            if not self._has_collision(x, y, size, placed_items, is_marker=not is_object):
                return x, y
        
        # If we can't find a non-colliding position, raise an error
        raise ValueError("Could not find non-colliding position after 200 attempts")
    
    def _generate_smart_marker_layout(self, num_markers: int, marker_size: int, 
                                     placed_items: List[Dict], objects: List[Dict]) -> List[Tuple[int, int]]:
        """Generate marker positions using smart layout to prevent final frame overlap."""
        
        image_width, image_height = self.config.image_size
        
        # Find maximum object effective radius to ensure space for final placement
        max_obj_r = max(self._effective_radius(obj["size"], obj.get("shape")) for obj in objects) if objects else 25
        
        # Calculate safe margins for markers - use min_distance to ensure no overlap with objects
        safety_buffer = max(20, self.config.min_distance)
        marker_margin = int(self._effective_radius(marker_size, shape="star") + max_obj_r + safety_buffer)
        
        # Ensure margin doesn't exceed image boundaries
        max_margin_x = image_width // 2
        max_margin_y = image_height // 2
        marker_margin = min(marker_margin, max_margin_x, max_margin_y)
        
        marker_positions = []
        
        # Try different layout strategies
        strategies = [
            self._generate_grid_layout,
            self._generate_circular_layout,
            self._generate_scatter_layout
        ]
        
        for strategy in strategies:
            try:
                positions = strategy(num_markers, marker_size, marker_margin, image_width, image_height, placed_items, objects)
                if positions:
                    return positions
            except ValueError:
                continue
        
        # Fallback: simple random placement with constraints
        return self._generate_fallback_layout(num_markers, marker_size, marker_margin, image_width, image_height, placed_items, objects)
    
    def _generate_grid_layout(self, num_markers: int, marker_size: int, margin: int, 
                             width: int, height: int, placed_items: List[Dict], objects: List[Dict]) -> List[Tuple[int, int]]:
        """Generate markers in a grid pattern."""
        
        positions = []
        
        # Calculate minimum spacing needed
        # Required center distance between markers should guarantee final objects never overlap.
        min_marker_gap = getattr(self.config, 'min_marker_spacing', 80)
        max_obj_r = max(self._effective_radius(obj["size"], obj.get("shape")) for obj in objects) if objects else 50
        required_spacing = 2 * max_obj_r + min_marker_gap
        
        # Calculate grid dimensions
        cols = int(math.ceil(math.sqrt(num_markers)))
        rows = int(math.ceil(num_markers / cols))
        
        # Calculate cell size with spacing - ensure minimum spacing between markers
        cell_width = (width - 2 * margin) / cols
        cell_height = (height - 2 * margin) / rows
        
        # Ensure cells are large enough for required spacing
        min_cell_size = min(cell_width, cell_height)
        if min_cell_size < required_spacing:
            raise ValueError("Grid layout not possible - image too small for required spacing")
        
        for i in range(num_markers):
            row = i // cols
            col = i % cols
            
            # Calculate position with some randomness within the cell
            x = margin + col * cell_width + cell_width / 2
            y = margin + row * cell_height + cell_height / 2
            
            # Add small random offset within cell bounds (reduced to maintain spacing)
            max_offset_x = min((cell_width - required_spacing) / 2, cell_width / 6)
            max_offset_y = min((cell_height - required_spacing) / 2, cell_height / 6)
            x += random.uniform(-max_offset_x, max_offset_x)
            y += random.uniform(-max_offset_y, max_offset_y)
            
            # Check for collisions with existing items
            if not self._has_collision(int(x), int(y), marker_size, placed_items, is_marker=True):
                positions.append((int(x), int(y)))
            else:
                raise ValueError("Collision detected in grid layout")
        
        return positions
    
    def _generate_circular_layout(self, num_markers: int, marker_size: int, margin: int,
                                 width: int, height: int, placed_items: List[Dict], objects: List[Dict]) -> List[Tuple[int, int]]:
        """Generate markers in a circular pattern."""
        
        positions = []
        
        # Calculate minimum spacing needed
        min_marker_gap = getattr(self.config, 'min_marker_spacing', 80)
        max_obj_r = max(self._effective_radius(obj["size"], obj.get("shape")) for obj in objects) if objects else 50
        required_spacing = 2 * max_obj_r + min_marker_gap
        
        # Calculate circle parameters
        center_x = width // 2
        center_y = height // 2
        available_width = width - 2 * margin
        available_height = height - 2 * margin
        
        # Calculate minimum radius needed based on spacing and number of markers
        # Circumference = 2 * pi * radius, need at least num_markers * required_spacing
        min_radius = (num_markers * required_spacing) / (2 * math.pi)
        
        # Use the smaller dimension for radius, but ensure it's at least min_radius
        max_radius = min(available_width, available_height) // 2
        radius = max(min_radius, max_radius * 0.7)  # Use 70% of available to ensure spacing
        
        if radius < marker_size * 2 or radius < min_radius:
            raise ValueError("Circular layout not possible - radius too small for required spacing")
        
        # Place markers around the circle
        for i in range(num_markers):
            angle = 2 * math.pi * i / num_markers
            
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Check for collisions with existing items
            if not self._has_collision(int(x), int(y), marker_size, placed_items, is_marker=True):
                positions.append((int(x), int(y)))
            else:
                raise ValueError("Collision detected in circular layout")
        
        return positions
    
    def _generate_scatter_layout(self, num_markers: int, marker_size: int, margin: int,
                                width: int, height: int, placed_items: List[Dict], objects: List[Dict]) -> List[Tuple[int, int]]:
        """Generate markers with smart scattering to avoid clustering."""
        
        positions = []
        
        # Calculate minimum distance between markers - use config value or fallback to larger spacing
        min_marker_gap = getattr(self.config, 'min_marker_spacing', 80)
        max_obj_r = max(self._effective_radius(obj["size"], obj.get("shape")) for obj in objects) if objects else 50
        min_distance = 2 * max_obj_r + min_marker_gap
        
        for i in range(num_markers):
            placed = False
            
            for attempt in range(200):  # Try more times for each marker
                x = random.randint(margin, width - margin)
                y = random.randint(margin, height - margin)
                
                # Check collision with existing placed items
                if not self._has_collision(x, y, marker_size, placed_items, is_marker=True):
                    
                    # Check minimum distance from other proposed markers
                    collision = False
                    for pos_x, pos_y in positions:
                        distance = math.sqrt((x - pos_x) ** 2 + (y - pos_y) ** 2)
                        if distance < min_distance:
                            collision = True
                            break
                    
                    if not collision:
                        positions.append((x, y))
                        placed = True
                        break
            
            if not placed:
                raise ValueError(f"Could not place marker {i} in scatter layout")
        
        return positions
    
    def _generate_fallback_layout(self, num_markers: int, marker_size: int, margin: int,
                                 width: int, height: int, placed_items: List[Dict], objects: List[Dict]) -> List[Tuple[int, int]]:
        """Fallback layout using simple constraints."""
        
        positions: List[Tuple[int, int]] = []
        
        for i in range(num_markers):
            for attempt in range(200):
                x = random.randint(margin, width - margin)
                y = random.randint(margin, height - margin)
                
                # Collision check against initial objects AND already-placed markers
                tmp_items = list(placed_items)
                for px, py in positions:
                    tmp_items.append(
                        {
                            "position": (px, py),
                            "size": marker_size,
                            "is_marker": True,
                            "shape": "star",
                        }
                    )

                if not self._has_collision(x, y, marker_size, tmp_items, is_marker=True):
                    positions.append((x, y))
                    break
            else:
                raise ValueError(f"Could not place marker {i} in fallback layout")
        
        return positions
    
    def _check_collision_between_items(self, x1: int, y1: int, size1: int, x2: int, y2: int, size2: int) -> bool:
        """Check collision between two items."""
        # Simple distance-based collision check for two items
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        collision_distance = (size1 + size2) / 2 + 3  # Small buffer
        
        if distance < collision_distance:
            return True
            
        # Additional AABB check if items are very close
        if distance < (size1 + size2):
            return self._check_aabb_collision(x1, y1, size1, x2, y2, size2)
        
        return False
    
    def _has_collision(self, x: int, y: int, size: int, placed_items: List[Dict], is_marker: bool = False) -> bool:
        """Check if a new item would collide with existing items."""
        
        new_shape = "star" if is_marker else "square"
        new_r = self._effective_radius(size, new_shape)

        for item in placed_items:
            item_x, item_y = item["position"]
            item_size = item["size"]
            item_is_marker = item["is_marker"]
            item_shape = item.get("shape", "star" if item_is_marker else None)
            item_r = self._effective_radius(item_size, item_shape)
            
            # Calculate distance between centers
            distance = math.sqrt((x - item_x) ** 2 + (y - item_y) ** 2)
            
            # Treat config values as EXTRA GAP (in pixels) beyond the sum of radii.
            if is_marker and item_is_marker:
                extra_gap = getattr(self.config, "min_marker_spacing", 80)
            elif (not is_marker) and (not item_is_marker):
                extra_gap = getattr(self.config, "min_object_spacing", 30)
            else:
                extra_gap = getattr(self.config, "min_distance", 40)

            collision_distance = new_r + item_r + extra_gap
            
            if distance < collision_distance:
                return True
            
            # Only do AABB check if items are very close
            if distance < (new_r + item_r):
                if self._check_aabb_collision(x, y, size, item_x, item_y, item_size):
                    return True
        
        return False

    def _effective_radius(self, size: int, shape: Optional[str]) -> float:
        """
        Conservative effective radius for collision checks.
        In rendering, `size` is a half-size/radius for most shapes.
        """
        if shape is None:
            return float(size)
        s = shape.lower()
        if s == "circle":
            return float(size)
        if s == "square":
            return float(size) * math.sqrt(2)
        if s == "triangle":
            return float(size) * math.sqrt(2)
        if s == "rectangle":
            # width = 1.5*size, height = size -> half-width=0.75*size, half-height=0.5*size
            return math.sqrt((0.75 * size) ** 2 + (0.5 * size) ** 2) * 1.1
        if s == "star":
            # Star fits within a square of half-size `size`
            return float(size) * math.sqrt(2)
        return float(size) * math.sqrt(2)

    def _validate_no_occlusion(self, objects: List[Dict[str, Any]], markers: List[Dict[str, Any]]) -> bool:
        """
        Strictly validate:
        - Initial frame: object-object, marker-marker, and object-marker (any pair) do not overlap.
        - Final frame: objects placed at their matching marker positions do not overlap each other.
          (Object covering its own marker is expected/allowed.)
        """
        # Build helper lists with shapes for radius estimation
        obj_items = [
            {
                "id": o["id"],
                "shape": o["shape"],
                "size": o["size"],
                "pos": o["position"],
            }
            for o in objects
        ]
        mk_items = [
            {
                "id": m["id"],
                "shape": "star",
                "size": m["size"],
                "pos": m["position"],
                "color": m["color"],
            }
            for m in markers
        ]

        # Initial: object-object
        for i in range(len(obj_items)):
            for j in range(i + 1, len(obj_items)):
                if self._pairs_too_close(obj_items[i], obj_items[j], extra_gap=self.config.min_object_spacing):
                    return False

        # Initial: marker-marker
        for i in range(len(mk_items)):
            for j in range(i + 1, len(mk_items)):
                if self._pairs_too_close(mk_items[i], mk_items[j], extra_gap=self.config.min_marker_spacing):
                    return False

        # Initial: object-marker (all pairs)
        for o in obj_items:
            for m in mk_items:
                if self._pairs_too_close(o, m, extra_gap=self.config.min_distance):
                    return False

        # Final: object-object at marker positions
        final_objs = []
        for o in obj_items:
            # Find its matching marker by color
            obj_color = next(x["color"] for x in objects if x["id"] == o["id"])
            target_marker = next(m for m in mk_items if m["color"] == obj_color)
            final_objs.append({**o, "pos": target_marker["pos"]})

        for i in range(len(final_objs)):
            for j in range(i + 1, len(final_objs)):
                if self._pairs_too_close(final_objs[i], final_objs[j], extra_gap=self.config.min_object_spacing):
                    return False

        return True

    def _pairs_too_close(self, a: Dict[str, Any], b: Dict[str, Any], extra_gap: int) -> bool:
        ax, ay = a["pos"]
        bx, by = b["pos"]
        dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        ar = self._effective_radius(a["size"], a.get("shape"))
        br = self._effective_radius(b["size"], b.get("shape"))
        return dist < (ar + br + extra_gap)
    
    def _check_aabb_collision(self, x1: int, y1: int, size1: int, x2: int, y2: int, size2: int) -> bool:
        """Check AABB (Axis-Aligned Bounding Box) collision for rectangular items."""
        
        # Assuming square items with half-size
        half_size1 = size1
        half_size2 = size2
        
        # Check if rectangles overlap
        return not (x1 + half_size1 < x2 - half_size2 or
                   x1 - half_size1 > x2 + half_size2 or
                   y1 + half_size1 < y2 - half_size2 or
                   y1 - half_size1 > y2 + half_size2)
    
    def _create_frame(self, objects: List[Dict], markers: List[Dict], 
                     show_objects: bool = True, show_markers: bool = True, 
                     final_positions: bool = False) -> Image.Image:
        """Create a single frame with objects and markers."""
        
        width, height = self.config.image_size
        img = Image.new('RGB', (width, height), self.config.background_color)
        draw = ImageDraw.Draw(img)
        
        # Draw markers first (behind objects)
        if show_markers:
            for marker in markers:
                self._draw_marker(draw, marker)
        
        # Draw objects
        if show_objects:
            for obj in objects:
                if final_positions:
                    # Move object to its target position (marker position)
                    marker = next(m for m in markers if m["color"] == obj["color"])
                    obj_copy = obj.copy()
                    obj_copy["position"] = marker["position"]
                    self._draw_object_with_gradient_border(draw, obj_copy)
                else:
                    self._draw_object_with_gradient_border(draw, obj)
        
        return img
    
    def _draw_marker(self, draw: ImageDraw.Draw, marker: Dict[str, Any]) -> None:
        """Draw a colored marker (four-pointed star)."""
        
        x, y = marker["position"]
        size = marker["size"]
        color = self.color_map.get(marker["color"], (128, 128, 128))
        
        # Draw four-pointed star
        self._draw_four_pointed_star(draw, x, y, size, color)
    
    def _draw_four_pointed_star(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        """Draw a four-pointed star marker."""
        
        # Calculate points for a four-pointed star (like a plus sign with sharp ends)
        # Star points: top, right, bottom, left
        arm_length = size
        arm_width = size // 3
        
        # Top point
        top_point = (x, y - arm_length)
        # Right point  
        right_point = (x + arm_length, y)
        # Bottom point
        bottom_point = (x, y + arm_length)
        # Left point
        left_point = (x - arm_length, y)
        
        # Inner points (to create the star shape)
        inner_top = (x, y - arm_width)
        inner_right = (x + arm_width, y)
        inner_bottom = (x, y + arm_width)
        inner_left = (x - arm_width, y)
        
        # Draw the four-pointed star as a polygon
        star_points = [
            top_point,          # Top
            inner_right,        # Inner right
            right_point,        # Right
            inner_bottom,       # Inner bottom
            bottom_point,       # Bottom
            inner_left,         # Inner left
            left_point,         # Left
            inner_top           # Inner top
        ]
        
        # Draw filled star
        draw.polygon(star_points, fill=color, outline=color)
    
    def _draw_object_with_gradient_border(self, draw: ImageDraw.Draw, obj: Dict[str, Any]) -> None:
        """Draw a 2D object with solid fill only (no border)."""
        
        x, y = obj["position"]
        size = obj["size"]
        color = self.color_map.get(obj["color"], (128, 128, 128))
        shape = obj["shape"]
        
        # Always draw solid objects without any borders
        self._draw_simple_object(draw, x, y, size, color, shape)
    
    def _draw_object_with_gradient(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int], shape: str) -> None:
        """Draw object with gradient border effect."""
        
        # First draw the main solid object
        self._draw_simple_object(draw, x, y, size, color, shape)
        
        # Then add gradient border on top
        border_width = max(2, size // 10)
        
        # Draw gradient border (lighter outer ring)
        for i in range(border_width):
            alpha = 1.0 - (i / border_width)
            # Lighten the color for gradient effect
            lighter_color = tuple(min(255, int(c * (1 + alpha * 0.3))) for c in color)
            
            if shape == "circle":
                draw.ellipse(
                    [x - size - i, y - size - i, x + size + i, y + size + i],
                    outline=lighter_color,
                    width=1
                )
            elif shape == "square":
                draw.rectangle(
                    [x - size - i, y - size - i, x + size + i, y + size + i],
                    outline=lighter_color,
                    width=1
                )
            elif shape == "rectangle":
                # Draw as wider rectangle
                rect_size = size * 1.5
                draw.rectangle(
                    [x - rect_size - i, y - size - i, x + rect_size + i, y + size + i],
                    outline=lighter_color,
                    width=1
                )
            elif shape == "triangle":
                # Draw triangular border
                triangle_points = [
                    (x, y - size - i),                    # Top
                    (x - size - i, y + size + i),         # Bottom-left
                    (x + size + i, y + size + i)          # Bottom-right
                ]
                draw.polygon(triangle_points, outline=lighter_color, width=1)
    
    def _draw_simple_object(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int], shape: str) -> None:
        """Draw simple 2D object."""
        
        if shape == "circle":
            self._draw_circle(draw, x, y, size, color)
        elif shape == "square":
            self._draw_square(draw, x, y, size, color)
        elif shape == "rectangle":
            self._draw_rectangle(draw, x, y, size, color)
        elif shape == "triangle":
            self._draw_triangle(draw, x, y, size, color)
    
    def _draw_circle(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        """Draw a simple circle with solid fill only."""
        
        draw.ellipse(
            [x - size, y - size, x + size, y + size],
            fill=color
        )
    
    def _draw_square(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        """Draw a simple square with solid fill only."""
        
        draw.rectangle(
            [x - size, y - size, x + size, y + size],
            fill=color
        )
    
    def _draw_rectangle(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        """Draw a simple rectangle with solid fill only."""
        
        width = size * 1.5
        height = size
        
        draw.rectangle(
            [x - width//2, y - height//2, x + width//2, y + height//2],
            fill=color
        )
    
    def _draw_triangle(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        """Draw a simple triangle with solid fill only."""
        
        triangle_points = [
            (x, y - size),                    # Top
            (x - size, y + size),             # Bottom-left
            (x + size, y + size)              # Bottom-right
        ]
        
        draw.polygon(triangle_points, fill=color)
    
    def _generate_animation_frames(self, objects: List[Dict], markers: List[Dict]) -> List[Image.Image]:
        """Generate animation frames showing objects moving to their target positions."""
        
        frames = []
        fps = self.config.video_fps
        duration = self.config.animation_duration
        num_frames = int(fps * duration)
        
        for frame_idx in range(num_frames):
            # Create frame
            width, height = self.config.image_size
            img = Image.new('RGB', (width, height), self.config.background_color)
            draw = ImageDraw.Draw(img)
            
            # Draw markers
            for marker in markers:
                self._draw_marker(draw, marker)
            
            # Draw objects at interpolated positions
            progress = frame_idx / (num_frames - 1)  # 0 to 1
            
            for obj in objects:
                start_x, start_y = obj["position"]
                
                # Find corresponding marker
                marker = next(m for m in markers if m["color"] == obj["color"])
                target_x, target_y = marker["position"]
                
                # Interpolate position with boundary constraints
                current_x = int(start_x + (target_x - start_x) * progress)
                current_y = int(start_y + (target_y - start_y) * progress)
                
                # Constrain position within image boundaries
                width, height = self.config.image_size
                margin = obj["size"] + 10  # Increased margin for safety
                
                # More aggressive boundary constraints to ensure objects stay within bounds
                current_x = max(margin, min(width - margin, current_x))
                current_y = max(margin, min(height - margin, current_y))
                
                # Create temporary object with current position
                temp_obj = obj.copy()
                temp_obj["position"] = (current_x, current_y)
                
                self._draw_object_with_gradient_border(draw, temp_obj)
            
            frames.append(img)
        
        return frames