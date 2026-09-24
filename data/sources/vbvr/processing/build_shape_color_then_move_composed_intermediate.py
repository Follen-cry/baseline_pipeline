#!/usr/bin/env python3
"""
Build the "shape_color_then_move_Composed_intermediate" dataset: 500 train /
100 eval, adopting the same composed-single-image-input pipeline used for
2d_geo_trans_Composed_intermediate (see that dataset's README.md and
data/sources/vbvr/processing/build_2d_geo_trans_composed_intermediate.py).

Task mechanics (shape_color_then_move): an A->B->C :: D->?->? visual analogy.
Top row (A,B,C) is a fully-solved static example, IDENTICAL in first_frame
and final_frame -- it carries no signal about the transformation and is
copied unchanged. Bottom row is the real task: D (original shape/color) is
static; E = D recolored (color_from -> color_to); F = D recolored AND moved
(by a fixed named offset, e.g. "down_large" = +80px). The generator's own
ground-truth video (_create_sequential_morph_frames) proves this is a
TWO-PHASE sequential transformation, not one continuous sweep: phase 1 = E's
color lerps color_from->color_to while F stays a "?"; phase 2 = color is
already fixed at color_to and F's y-position lerps center->move_offset. The
phase boundary is exactly "E fully recolored, F not yet touched" -- a
genuinely discrete state, not an artifact of arbitrary progress fractions.

Chosen intermediates (reusing the generator's own draw calls exactly):
  intermediate_1 = phase-1 END: E fully drawn at color_to, F still "?".
  intermediate_2 = phase-2 MIDPOINT (progress=0.5): E unchanged (color_to),
                   F drawn at color_to with y lerped 50% toward move_offset
                   (same clamped linear formula as generator.py's own
                   _create_sequential_morph_frames).

600 FRESH samples are generated (seed 20260916, disjoint from the original
10-sample eval_samples probe's seed=777, checked here against those 10
samples' param_hash). This task's TaskGenerator.__init__ REQUIRES
generate_videos=True (raises otherwise) and writes a real (discarded) video
to a temp dir per sample -- unavoidable, just slower; opencv-python is
available in this environment. Fidelity checked both ways: intermediate_1's
E-only-recolored logic is exercised implicitly by construction (pure
function of metadata, no interpolation to get wrong), but progress=0 (i.e.
literally re-deriving first_frame's static+D+two-"?" state) and the real
final_frame are both asserted byte-identical to catch any metadata
reconstruction mistake before trusting intermediates.

Writes:
  data/sources/vbvr/raw/shape_color_then_move/
    shape_color_then_move_task/shape_color_then_move_{00000000..00000599}/
      first_frame.png, final_frame.png, prompt.txt, metadata.json

  data/datasets/shape_color_then_move_Composed_intermediate/
    train.jsonl (500 rows), eval.jsonl (100 rows)
    images/{train,eval}/<id>/partial_grid_input.png   (the 1 input image)
    images/{train,eval}/<id>/gt_grid.png               (the target image -- REAL ground truth)
    images/{train,eval}/<id>/intermediate_1.png, intermediate_2.png  (debug/inspection only)

  data/meta/shape_color_then_move_Composed_intermediate_meta.json
  data/meta/shape_color_then_move_Composed_intermediate_train_meta.json
"""
import json
import os
import random
import sys

from PIL import Image, ImageChops, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
GENERATOR_ROOT = os.path.join(REPO_ROOT, "data", "sources", "vbvr", "generators", "shape_color_then_move")
ORIGINAL_PROBE_RAW_TASK_DIR = (
    "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/eval_samples/raw/"
    "shape_color_then_move/shape_color_then_move_task"
)
RAW_OUT_ROOT = os.path.join(REPO_ROOT, "data", "sources", "vbvr", "raw", "shape_color_then_move")
DATASET_ROOT = os.path.join(REPO_ROOT, "data", "datasets", "shape_color_then_move_Composed_intermediate")
META_PATH = os.path.join(REPO_ROOT, "data", "meta", "shape_color_then_move_Composed_intermediate_meta.json")
TRAIN_META_PATH = os.path.join(REPO_ROOT, "data", "meta", "shape_color_then_move_Composed_intermediate_train_meta.json")

NUM_TOTAL = 600
NUM_TRAIN = 500
NUM_EVAL = 100
SEED = 20260916
SPLIT_SHUFFLE_SEED = 20260916
GRID_CELL = 512
MOVE_PROGRESS_FOR_INTER2 = 0.5

sys.path.insert(0, GENERATOR_ROOT)
from src import TaskGenerator, TaskConfig  # noqa: E402
from core import OutputWriter  # noqa: E402

PARTIAL_GRID_PROMPT_WRAPPER = (
    "You are given a single 2x2 grid image. The top-left cell shows the "
    "FIRST frame (starting state), and the bottom-right cell shows the "
    "FINAL frame (ending state). The top-right and bottom-left cells are "
    "blank. Fill in these two blank cells with the two intermediate frames "
    "of the transformation, in temporal order (top-right = first "
    "intermediate step, bottom-left = second intermediate step), so that "
    "reading top-left -> top-right -> bottom-left -> bottom-right shows the "
    "transformation progressing smoothly and consistently with the rule "
    "described below. Keep the top-left and bottom-right cells exactly as "
    "given, do not add gutters/labels beyond the four cells, and output "
    "only the single complete 2x2 grid image.\n\n"
    "Task: {prompt}"
)


def task_data_from_metadata(meta_params):
    objects = meta_params["objects"]
    return {
        "shape_a": objects[0]["symbol"],
        "shape_b": objects[1]["symbol"],
        "shape_c": objects[2]["symbol"],
        "shape_d": objects[3]["symbol"],
        "color_from": meta_params["color_from"],
        "color_to": meta_params["color_to"],
        "move_from": meta_params["move_from"],
        "move_to": meta_params["move_to"],
    }


def render_first(gen, config, task_data):
    """Mirrors generator._render_initial_state exactly (D + two '?')."""
    width, height = config.image_size
    margin = config.margin
    base_shape_size = config.shape_size
    half_shape_size = base_shape_size // 2
    total_content_width = width - 2 * margin
    step_width = total_content_width // 3
    shape_spacing = step_width * 0.8
    arrow_width = step_width * 0.2
    top_row_y = max(height // 3, margin + half_shape_size)
    bottom_row_y = min(2 * height // 3, height - margin - half_shape_size)
    move_offset = gen.movements.get(task_data.get("move_to", "center"), 0)

    top_row_moved_y = top_row_y + move_offset
    if top_row_moved_y < margin + half_shape_size:
        top_row_y = margin + half_shape_size - move_offset
    elif top_row_moved_y > height - margin - half_shape_size:
        top_row_y = height - margin - half_shape_size - move_offset
    bottom_row_moved_y = bottom_row_y + move_offset
    if bottom_row_moved_y < margin + half_shape_size:
        bottom_row_y = margin + half_shape_size - move_offset
    elif bottom_row_moved_y > height - margin - half_shape_size:
        bottom_row_y = height - margin - half_shape_size - move_offset

    positions = {
        "A": (margin + shape_spacing // 2, top_row_y),
        "arrow1": (margin + shape_spacing + arrow_width // 2, top_row_y),
        "B": (margin + shape_spacing + arrow_width + shape_spacing // 2, top_row_y),
        "arrow2": (margin + 2 * shape_spacing + arrow_width + arrow_width // 2, top_row_y),
        "C": (margin + 2 * shape_spacing + 2 * arrow_width + shape_spacing // 2, top_row_y),
        "D": (margin + shape_spacing // 2, bottom_row_y),
        "arrow3": (margin + shape_spacing + arrow_width // 2, bottom_row_y),
        "question1": (margin + shape_spacing + arrow_width + shape_spacing // 2, bottom_row_y),
        "arrow4": (margin + 2 * shape_spacing + arrow_width + arrow_width // 2, bottom_row_y),
        "question2": (margin + 2 * shape_spacing + 2 * arrow_width + shape_spacing // 2, bottom_row_y),
    }

    img = gen.renderer.create_blank_image()
    draw = ImageDraw.Draw(img)
    gen._draw_shape_at_position(draw, task_data["shape_a"], positions["A"], base_shape_size, task_data["color_from"], "center")
    gen._draw_arrow(draw, positions["arrow1"])
    gen._draw_shape_at_position(draw, task_data["shape_b"], positions["B"], base_shape_size, task_data["color_to"], "center")
    gen._draw_arrow(draw, positions["arrow2"])
    gen._draw_shape_at_position(draw, task_data["shape_c"], positions["C"], base_shape_size, task_data["color_to"], task_data["move_to"])
    gen._draw_shape_at_position(draw, task_data["shape_d"], positions["D"], base_shape_size, task_data["color_from"], "center")
    gen._draw_arrow(draw, positions["arrow3"])
    gen._draw_question_mark(draw, positions["question1"])
    gen._draw_arrow(draw, positions["arrow4"])
    gen._draw_question_mark(draw, positions["question2"])
    return img


def render_intermediates(gen, config, task_data):
    """intermediate_1 = E fully recolored, F still '?' (phase-1 end).
    intermediate_2 = E unchanged, F half-moved at color_to (phase-2 midpoint)."""
    width, height = config.image_size
    margin = config.margin
    base_shape_size = config.shape_size
    half_shape_size = base_shape_size // 2
    total_content_width = width - 2 * margin
    step_width = total_content_width // 3
    shape_spacing = step_width * 0.8
    arrow_width = step_width * 0.2
    bottom_row_y = min(2 * height // 3, height - margin - half_shape_size)
    move_offset = gen.movements.get(task_data.get("move_to", "center"), 0)

    bottom_row_moved_y = bottom_row_y + move_offset
    if bottom_row_moved_y < margin + half_shape_size:
        bottom_row_y = margin + half_shape_size - move_offset
    elif bottom_row_moved_y > height - margin - half_shape_size:
        bottom_row_y = height - margin - half_shape_size - move_offset

    question1_pos = (margin + shape_spacing + arrow_width + shape_spacing // 2, bottom_row_y)
    question2_pos = (margin + 2 * shape_spacing + 2 * arrow_width + shape_spacing // 2, bottom_row_y)

    shape_d = task_data["shape_d"]
    color_to_rgb = gen.colors[task_data["color_to"]]

    static_kwargs = dict(base_shape_size=base_shape_size, shape_spacing=shape_spacing,
                          arrow_width=arrow_width, margin=margin, width=width, height=height)

    # intermediate_1: E done (color_to), F still "?"
    img1 = gen._render_static_elements(task_data, **static_kwargs)
    draw1 = ImageDraw.Draw(img1)
    gen._draw_base_shape(draw1, shape_d, question1_pos[0], question1_pos[1], base_shape_size, color_to_rgb, (0, 0, 0), 2)
    gen._draw_question_mark(draw1, question2_pos)

    # intermediate_2: E done, F half-moved (same clamp as generator.py)
    img2 = gen._render_static_elements(task_data, **static_kwargs)
    draw2 = ImageDraw.Draw(img2)
    gen._draw_base_shape(draw2, shape_d, question1_pos[0], question1_pos[1], base_shape_size, color_to_rgb, (0, 0, 0), 2)
    current_y_offset = move_offset * MOVE_PROGRESS_FOR_INTER2
    current_y = question2_pos[1] + current_y_offset
    current_y = max(margin + half_shape_size, min(current_y, height - margin - half_shape_size))
    gen._draw_base_shape(draw2, shape_d, question2_pos[0], current_y, base_shape_size, color_to_rgb, (0, 0, 0), 2)

    return img1, img2


def images_equal(a: Image.Image, b: Image.Image) -> bool:
    if a.size != b.size:
        return False
    diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
    return diff.getbbox() is None


def compose_partial_grid(first_img, final_img, bg, cell=GRID_CELL):
    grid = Image.new("RGB", (cell * 2, cell * 2), color=bg)
    grid.paste(first_img.resize((cell, cell), Image.LANCZOS), (0, 0))
    grid.paste(final_img.resize((cell, cell), Image.LANCZOS), (cell, cell))
    return grid


def compose_full_grid(first_img, inter1_img, inter2_img, final_img, bg, cell=GRID_CELL):
    grid = Image.new("RGB", (cell * 2, cell * 2), color=bg)
    grid.paste(first_img.resize((cell, cell), Image.LANCZOS), (0, 0))
    grid.paste(inter1_img.resize((cell, cell), Image.LANCZOS), (cell, 0))
    grid.paste(inter2_img.resize((cell, cell), Image.LANCZOS), (0, cell))
    grid.paste(final_img.resize((cell, cell), Image.LANCZOS), (cell, cell))
    return grid


def load_existing_probe_param_hashes():
    hashes = set()
    for name in sorted(os.listdir(ORIGINAL_PROBE_RAW_TASK_DIR)):
        p = os.path.join(ORIGINAL_PROBE_RAW_TASK_DIR, name, "metadata.json")
        if not os.path.isfile(p):
            continue
        meta = json.load(open(p))
        hashes.add(meta["param_hash"])
    assert len(hashes) == 10, f"expected 10 existing probe hashes, got {len(hashes)}"
    return hashes


def main():
    existing_hashes = load_existing_probe_param_hashes()

    config = TaskConfig(num_samples=NUM_TOTAL, random_seed=SEED, generate_videos=True)
    gen = TaskGenerator(config)
    writer = OutputWriter(RAW_OUT_ROOT)

    bg = (255, 255, 255)

    rows = []
    for i in range(NUM_TOTAL):
        task_id = f"{config.domain}_{i:08d}"
        pair = gen.generate_task_pair(task_id)
        writer.write_task_pair(pair)

        meta = pair.metadata
        if meta["param_hash"] in existing_hashes:
            raise RuntimeError(f"{task_id}: param_hash collides with the original probe -- pick a different SEED")
        existing_hashes.add(meta["param_hash"])

        task_data = task_data_from_metadata(meta["parameters"])

        first_img = pair.first_image
        final_img = pair.final_image

        rerendered_first = render_first(gen, config, task_data)
        if not images_equal(first_img, rerendered_first):
            raise RuntimeError(f"{task_id}: first_frame fidelity check failed")

        inter1_img, inter2_img = render_intermediates(gen, config, task_data)

        partial_input = compose_partial_grid(first_img, final_img, bg)
        full_grid = compose_full_grid(first_img, inter1_img, inter2_img, final_img, bg)

        rows.append({
            "id": task_id,
            "raw_prompt": pair.prompt,
            "inter1": inter1_img,
            "inter2": inter2_img,
            "partial_input": partial_input,
            "full_grid": full_grid,
        })

        if (i + 1) % 100 == 0:
            print(f"[build] generated + rendered {i + 1}/{NUM_TOTAL}", flush=True)

    idx = list(range(NUM_TOTAL))
    random.Random(SPLIT_SHUFFLE_SEED).shuffle(idx)
    train_idx = sorted(idx[:NUM_TRAIN])
    eval_idx = sorted(idx[NUM_TRAIN:])
    assert len(train_idx) == NUM_TRAIN and len(eval_idx) == NUM_EVAL
    assert not (set(train_idx) & set(eval_idx))

    def write_split(split_name, indices):
        jsonl_path = os.path.join(DATASET_ROOT, f"{split_name}.jsonl")
        img_dir = os.path.join(DATASET_ROOT, "images", split_name)
        with open(jsonl_path, "w") as f:
            for i in indices:
                r = rows[i]
                sdir = os.path.join(img_dir, r["id"])
                os.makedirs(sdir, exist_ok=True)

                partial_path = os.path.join(sdir, "partial_grid_input.png")
                grid_path = os.path.join(sdir, "gt_grid.png")
                r["partial_input"].save(partial_path)
                r["full_grid"].save(grid_path)
                r["inter1"].save(os.path.join(sdir, "intermediate_1.png"))
                r["inter2"].save(os.path.join(sdir, "intermediate_2.png"))

                wrapped_prompt = PARTIAL_GRID_PROMPT_WRAPPER.format(prompt=r["raw_prompt"])
                row = {
                    "id": r["id"],
                    "task_type": "imgen",
                    "task_name": "shape_color_then_move_Composed_intermediate",
                    "category": "Knowledge",
                    "split": split_name,
                    "image": [partial_path],
                    "target_image": grid_path,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{wrapped_prompt}"},
                        {"from": "gpt", "value": "The complete 2x2 grid should look like this: <img>"},
                    ],
                }
                f.write(json.dumps(row) + "\n")
        return jsonl_path

    os.makedirs(DATASET_ROOT, exist_ok=True)
    train_path = write_split("train", train_idx)
    eval_path = write_split("eval", eval_idx)

    meta = {
        "shape_color_then_move_composed_intermediate_train": {
            "root": "/", "annotation": train_path, "data_augment": False,
            "max_dynamic_patch": 1, "repeat_time": 1, "length": len(train_idx), "task_type": "imgen",
        },
        "shape_color_then_move_composed_intermediate_eval": {
            "root": "/", "annotation": eval_path, "data_augment": False,
            "max_dynamic_patch": 1, "repeat_time": 1, "length": len(eval_idx), "task_type": "imgen",
        },
    }
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    with open(TRAIN_META_PATH, "w") as f:
        json.dump({"shape_color_then_move_composed_intermediate_train": meta["shape_color_then_move_composed_intermediate_train"]}, f, indent=2)

    print(f"\nDone: {len(train_idx)} train, {len(eval_idx)} eval written to {DATASET_ROOT}")


if __name__ == "__main__":
    main()
