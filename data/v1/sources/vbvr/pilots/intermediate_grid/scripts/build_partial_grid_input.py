#!/usr/bin/env python3
"""
Build the "composed single-image input" variant of the intermediate-grid task.

Instead of feeding first_frame and final_frame as two separate conditioning
images (2 images in, 1 grid image out -- the format used by
build_intermediate_grid_samples.py and run_intermediate_grid_infer*.py), this
composes them into ONE partial 2x2 grid image up front:
  - top-left: first_frame
  - bottom-right: final_frame
  - top-right, bottom-left: blank (background color, nothing drawn)

That single partial-grid image becomes the ONE input image, and the target
is the same-canvas, same-size complete grid (gt_grid.png) with the two blank
cells filled in. This reframes the task as a same-canvas image edit (like
target_pred's first_frame -> final_frame contract) rather than a multi-image
composition -- so VAE pixel-level conditioning is meaningful again (input and
output are the same 2x2 canvas), unlike the 2-separate-images format.

Only builds samples 00 and 01 (the 2 examples requested for this zero-shot
check), writing into their existing per-sample dirs:
  data/v1/sources/vbvr/pilots/intermediate_grid/samples/<NN>/partial_grid_input.png
  data/v1/sources/vbvr/pilots/intermediate_grid/samples/<NN>/prompt_partial_grid.txt
"""
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(os.path.dirname(HERE), "samples")
CELL = 512
SAMPLE_IDS = ["00", "01"]

BG_COLOR = (240, 240, 240)  # matches the O-6 generator's own frame background

PARTIAL_GRID_PROMPT_WRAPPER = (
    "You are given a single 2x2 grid image. The top-left cell shows the "
    "FIRST frame (starting state) of a 2D shape transformation, and the "
    "bottom-right cell shows the FINAL frame (ending state). The top-right "
    "and bottom-left cells are blank. Fill in these two blank cells with the "
    "two intermediate frames of the transformation, in temporal order "
    "(top-right = first intermediate step, bottom-left = second "
    "intermediate step), so that reading top-left -> top-right -> "
    "bottom-left -> bottom-right shows the transformation progressing "
    "smoothly and consistently with the rule described below. Keep the "
    "top-left and bottom-right cells exactly as given, do not add "
    "gutters/labels beyond the four cells, and output only the single "
    "complete 2x2 grid image.\n\n"
    "Task: {prompt}"
)


def compose_partial_grid(first_img, final_img, cell=CELL):
    grid = Image.new("RGB", (cell * 2, cell * 2), color=BG_COLOR)
    grid.paste(first_img.resize((cell, cell), Image.LANCZOS), (0, 0))
    grid.paste(final_img.resize((cell, cell), Image.LANCZOS), (cell, cell))
    # top-right (cell, 0) and bottom-left (0, cell) left as BG_COLOR: blank
    return grid


def main():
    for sid in SAMPLE_IDS:
        sdir = os.path.join(SAMPLES_DIR, sid)
        first_img = Image.open(os.path.join(sdir, "first_frame.png"))
        final_img = Image.open(os.path.join(sdir, "final_frame.png"))

        partial_grid = compose_partial_grid(first_img, final_img)
        partial_grid.save(os.path.join(sdir, "partial_grid_input.png"))

        raw_prompt = open(os.path.join(sdir, "prompt_original.txt")).read().strip()
        wrapped = PARTIAL_GRID_PROMPT_WRAPPER.format(prompt=raw_prompt)
        with open(os.path.join(sdir, "prompt_partial_grid.txt"), "w") as f:
            f.write(wrapped)

        print(f"OK sample {sid}: partial_grid_input.png + prompt_partial_grid.txt")


if __name__ == "__main__":
    main()
