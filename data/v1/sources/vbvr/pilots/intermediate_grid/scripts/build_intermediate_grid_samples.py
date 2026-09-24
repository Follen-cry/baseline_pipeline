#!/usr/bin/env python3
"""
Build the VBVR 2d_geometric_transformation "intermediate grid" eval task.

Source datapoints: the 10 already-generated raw samples at
Evaluation/VBVR-CustomEval/eval_samples/raw/2d_geometric_transformation/
(first_frame.png, final_frame.png, prompt.txt, metadata.json each). Those
samples only ship the first/final frame of a planar rotation; the two
intermediate frames were never rendered or saved anywhere.

Since the rotation is fully deterministic (metadata.json records shape_type,
rotation_center, initial_angle, rotation_angle, object_color), this script
re-renders the two missing intermediate frames pixel-faithfully by reusing
the ORIGINAL generator's private rendering methods (same base-shape
construction, same rotate-and-translate math, same dashed-outline/rotation-
center drawing code) from:
    VBVR-DataFactory/O-6_2d_geometric_transformation_data-generator

For each sample this writes: first_frame.png, intermediate_1.png,
intermediate_2.png, final_frame.png, gt_grid.png (2x2 composite),
prompt_original.txt, prompt_grid.txt, metadata.json -- self-contained under
data/v1/sources/vbvr/pilots/intermediate_grid/samples/<NN>/, plus a top-level manifest.json.

Before trusting the re-render, this script asserts that re-deriving the
FIRST frame from metadata.json byte-matches the original first_frame.png.
If that assertion ever fails, the generator's rendering code has drifted
from what produced these samples and the intermediate frames should not be
trusted without re-checking.
"""
import json
import os
import shutil
import sys

from PIL import Image, ImageChops

GENERATOR_ROOT = "/scratch/network/ssd2/junlin/ssl_mllm/VBVR-DataFactory/O-6_2d_geometric_transformation_data-generator"
RAW_TASK_DIR = (
    "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/eval_samples/raw/"
    "2d_geometric_transformation/transformation_worlds_2d_geometric_transformation_task"
)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.dirname(HERE)  # data/v1/sources/vbvr/pilots/intermediate_grid
SAMPLES_OUT = os.path.join(OUT_ROOT, "samples")
NUM_SAMPLES = 10
GRID_CELL = 512  # each frame resized to this before composing the 2x2 grid

sys.path.insert(0, GENERATOR_ROOT)
from src import TaskGenerator, TaskConfig  # noqa: E402

# Progress fractions (of the full initial_angle -> initial_angle+rotation_angle
# sweep) at which the two intermediate frames are sampled. Evenly spaced
# thirds give a natural 4-point temporal sequence: 0, 1/3, 2/3, 1.
INTERMEDIATE_PROGRESS = [1 / 3, 2 / 3]

GRID_PROMPT_WRAPPER = (
    "You are given two frames of a 2D shape transformation sequence, arranged "
    "as the top-left and bottom-right cells of an otherwise-empty 2x2 grid: "
    "the top-left cell is the FIRST frame (starting state) and the "
    "bottom-right cell is the FINAL frame (ending state). Infer the "
    "intermediate transformation steps that connect them, then generate the "
    "two missing intermediate frames and output a single complete 2x2 grid "
    "image with this exact layout:\n"
    "  - top-left: the given first frame (reproduce it unchanged)\n"
    "  - top-right: the first inferred intermediate frame\n"
    "  - bottom-left: the second inferred intermediate frame\n"
    "  - bottom-right: the given final frame (reproduce it unchanged)\n"
    "The four cells must read in temporal order top-left -> top-right -> "
    "bottom-left -> bottom-right, showing the transformation progressing "
    "smoothly and consistently with the rule described below. Do not alter "
    "the given first and final frames, do not add gutters/labels beyond the "
    "four frames, and output only the single 2x2 grid image.\n\n"
    "Task: {prompt}"
)


def render_progress(gen, config, base_shape, rotation_center, final_shape,
                     object_color, initial_angle, rotation_angle, progress):
    """Re-implements TaskGenerator._render_initial_state's drawing order for
    an arbitrary point along the rotation sweep (progress=0 reproduces the
    initial-state render exactly; this is used to validate fidelity)."""
    from PIL import ImageDraw

    width, height = config.image_size
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    current_angle = initial_angle + rotation_angle * progress
    current_shape = gen._rotate_and_translate_shape(base_shape, current_angle, rotation_center)

    gen._draw_dashed_polygon(
        draw, final_shape, outline_color=(100, 100, 100), width=config.target_outline_width
    )
    if final_shape:
        draw.line([rotation_center, final_shape[0]], fill=(100, 100, 100), width=1)

    draw.polygon(current_shape, fill=object_color, outline=(50, 50, 50), width=2)
    if current_shape:
        draw.line([rotation_center, current_shape[0]], fill=(50, 50, 50), width=2)

    gen._draw_rotation_center(draw, rotation_center)
    return img


def images_equal(a: Image.Image, b: Image.Image) -> bool:
    if a.size != b.size:
        return False
    diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
    return diff.getbbox() is None


def compose_grid(first_img, inter1_img, inter2_img, final_img, cell=GRID_CELL):
    grid = Image.new("RGB", (cell * 2, cell * 2), color=(240, 240, 240))
    grid.paste(first_img.resize((cell, cell), Image.LANCZOS), (0, 0))
    grid.paste(inter1_img.resize((cell, cell), Image.LANCZOS), (cell, 0))
    grid.paste(inter2_img.resize((cell, cell), Image.LANCZOS), (0, cell))
    grid.paste(final_img.resize((cell, cell), Image.LANCZOS), (cell, cell))
    return grid


def main():
    config = TaskConfig(num_samples=1, generate_videos=False)
    gen = TaskGenerator(config)

    os.makedirs(SAMPLES_OUT, exist_ok=True)
    manifest = []

    for i in range(NUM_SAMPLES):
        sample_id = f"{i:08d}"
        src_dir = os.path.join(
            RAW_TASK_DIR, f"transformation_worlds_2d_geometric_transformation_{sample_id}"
        )
        meta = json.load(open(os.path.join(src_dir, "metadata.json")))
        params = meta["parameters"]
        obj = params["objects"][0]

        shape_type = params["shape_type"]
        rotation_angle = params["rotation_angle"]
        initial_angle = params["initial_angle"]
        rotation_center = tuple(obj["center"])
        object_color = tuple(obj["color"])

        base_shape = gen._create_base_shape(shape_type, config.object_size)
        final_shape = gen._rotate_and_translate_shape(
            base_shape, initial_angle + rotation_angle, rotation_center
        )

        # Fidelity check: re-deriving the FIRST frame from metadata.json must
        # byte-match the original first_frame.png before we trust any
        # re-rendered intermediate frame.
        first_frame_orig = Image.open(os.path.join(src_dir, "first_frame.png"))
        first_frame_rerendered = render_progress(
            gen, config, base_shape, rotation_center, final_shape,
            object_color, initial_angle, rotation_angle, progress=0.0
        )
        if not images_equal(first_frame_orig, first_frame_rerendered):
            raise RuntimeError(
                f"sample {sample_id}: re-rendered first frame does not match "
                f"original -- generator rendering code has drifted, do not "
                f"trust re-rendered intermediate frames"
            )

        inter1_img = render_progress(
            gen, config, base_shape, rotation_center, final_shape,
            object_color, initial_angle, rotation_angle, progress=INTERMEDIATE_PROGRESS[0]
        )
        inter2_img = render_progress(
            gen, config, base_shape, rotation_center, final_shape,
            object_color, initial_angle, rotation_angle, progress=INTERMEDIATE_PROGRESS[1]
        )

        out_dir = os.path.join(SAMPLES_OUT, f"{i:02d}")
        os.makedirs(out_dir, exist_ok=True)

        shutil.copy(os.path.join(src_dir, "first_frame.png"), os.path.join(out_dir, "first_frame.png"))
        shutil.copy(os.path.join(src_dir, "final_frame.png"), os.path.join(out_dir, "final_frame.png"))
        inter1_img.save(os.path.join(out_dir, "intermediate_1.png"))
        inter2_img.save(os.path.join(out_dir, "intermediate_2.png"))

        final_frame_img = Image.open(os.path.join(src_dir, "final_frame.png"))
        grid_img = compose_grid(first_frame_orig, inter1_img, inter2_img, final_frame_img)
        grid_img.save(os.path.join(out_dir, "gt_grid.png"))

        raw_prompt = open(os.path.join(src_dir, "prompt.txt")).read().strip()
        grid_prompt = GRID_PROMPT_WRAPPER.format(prompt=raw_prompt)
        with open(os.path.join(out_dir, "prompt_original.txt"), "w") as f:
            f.write(raw_prompt)
        with open(os.path.join(out_dir, "prompt_grid.txt"), "w") as f:
            f.write(grid_prompt)

        sample_meta = {
            "sample_id": f"{i:02d}",
            "task": "2d_geometric_transformation",
            "format": "intermediate_grid",
            "source_raw_dir": src_dir,
            "generator_repo": meta["generation"]["git"]["repo"],
            "generator_commit": meta["generation"]["git"]["commit"],
            "parameters": params,
            "intermediate_progress": INTERMEDIATE_PROGRESS,
            "grid_layout": {
                "top_left": "first_frame",
                "top_right": "intermediate_1",
                "bottom_left": "intermediate_2",
                "bottom_right": "final_frame",
                "cell_size": GRID_CELL,
                "grid_size": GRID_CELL * 2,
            },
            "fidelity_check": "re-rendered first frame byte-matches original first_frame.png",
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(sample_meta, f, indent=2)

        manifest.append({
            "sample_id": f"{i:02d}",
            "sample_dir": out_dir,
            "source": src_dir,
        })
        print(f"OK sample {i:02d}  (fidelity check passed)")

    with open(os.path.join(OUT_ROOT, "manifest.json"), "w") as f:
        json.dump({
            "task": "2d_geometric_transformation",
            "format": "intermediate_grid",
            "input": "first_frame + final_frame",
            "target": "2x2 grid: [first, intermediate_1 / intermediate_2, final]",
            "num_samples": NUM_SAMPLES,
            "samples": manifest,
        }, f, indent=2)

    print(f"\nDone: {NUM_SAMPLES} samples written to {SAMPLES_OUT}")


if __name__ == "__main__":
    main()
