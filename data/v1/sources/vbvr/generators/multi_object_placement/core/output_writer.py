"""Output writer for standard format."""

import shutil
from pathlib import Path
from typing import List
from .schemas import TaskPair
from .image_utils import ImageRenderer


class OutputWriter:
    """Writes tasks to standard folder structure."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_task_pair(self, task_pair: TaskPair) -> Path:
        """Write single task to disk."""
        task_dir = self.output_dir / f"{task_pair.domain}_task" / task_pair.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Write images
        ImageRenderer.ensure_rgb(task_pair.first_image).save(task_dir / "first_frame.png")
        
        if task_pair.final_image:
            ImageRenderer.ensure_rgb(task_pair.final_image).save(task_dir / "final_frame.png")
        
        # Write prompt
        (task_dir / "prompt.txt").write_text(task_pair.prompt)
        
        # Write video if provided (preserve original extension)
        if task_pair.ground_truth_video and Path(task_pair.ground_truth_video).exists():
            video_src = Path(task_pair.ground_truth_video)
            video_ext = video_src.suffix  # .mp4 or .avi
            video_dst = task_dir / f"ground_truth{video_ext}"
            
            # Only copy if source and destination are different
            if video_src.resolve() != video_dst.resolve():
                shutil.copy(video_src, video_dst)
            elif not video_dst.exists():
                # If already in correct location, just ensure it exists with correct name
                video_dst.touch()
        
        
        # Write metadata if provided
        if task_pair.metadata is not None:
            import json
            (task_dir / "metadata.json").write_text(
                json.dumps(task_pair.metadata, ensure_ascii=False, indent=2)
            )
        
        return task_dir
    
    def write_dataset(self, task_pairs: List[TaskPair]) -> Path:
        """Write all tasks to disk."""
        for pair in task_pairs:
            self.write_task_pair(pair)
        return self.output_dir
