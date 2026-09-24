#!/usr/bin/env python3
"""
Build the "rotation_puzzle_Composed_intermediate" dataset: 500 train / 100
eval, adopting the same composed-single-image-input pipeline used for
2d_geo_trans_Composed_intermediate (see that dataset's README.md and
data/v1/sources/vbvr/processing/build_2d_geo_trans_composed_intermediate.py).

Task mechanics (rotation_puzzle): a fixed 2x2 grid of 4 squares, each with a
fixed L-shaped pipe pattern per grid position. first_frame = squares at
randomized initial_angle (some already solved at 0 depending on difficulty);
final_frame = all 4 squares at target_angle=0. The generator's own ground
truth video (_create_rotation_animation_frames) rotates all 4 squares
SIMULTANEOUSLY under one shared progress fraction -- structurally identical
to the plain rotation task's single-progress-parameter sweep. metadata.json
already stores each square's initial_angle, target_angle(0), and
rotation_angle (the exact shortest-path signed diff), so no wrap-around
recomputation is needed.

600 FRESH samples are generated (seed 20260916, disjoint from the original
10-sample eval_samples probe's seed=777, checked here against those 10
samples' param_hash). The 2 intermediate frames per sample are rendered by
reconstructing a puzzle_data dict from metadata and calling the generator's
own `_render_frame(step_data, use_initial_angles=True)` at progress 1/3 and
2/3 -- LINEAR interpolation (not the ground-truth video's ease-in-out), for
consistency with how every other task in this pipeline builds its
intermediates. Fidelity checked both ways: progress=0 against first_frame.png,
progress=1 against final_frame.png.

Writes:
  data/v1/sources/vbvr/raw/rotation_puzzle/
    rotation_puzzle_task/rotation_puzzle_{00000000..00000599}/
      first_frame.png, final_frame.png, prompt.txt, metadata.json

  data/v1/datasets/rotation_puzzle_Composed_intermediate/
    train.jsonl (500 rows), eval.jsonl (100 rows)
    images/{train,eval}/<id>/partial_grid_input.png   (the 1 input image)
    images/{train,eval}/<id>/gt_grid.png               (the target image -- REAL ground truth)
    images/{train,eval}/<id>/intermediate_1.png, intermediate_2.png  (debug/inspection only)

  data/v1/meta/rotation_puzzle_Composed_intermediate_meta.json
  data/v1/meta/rotation_puzzle_Composed_intermediate_train_meta.json
"""
import json
import os
import random
import sys

from PIL import Image, ImageChops

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
GENERATOR_ROOT = os.path.join(REPO_ROOT, "data", "v1", "sources", "vbvr", "generators", "rotation_puzzle")
ORIGINAL_PROBE_RAW_TASK_DIR = (
    "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/eval_samples/raw/"
    "rotation_puzzle/rotation_puzzle_task"
)
RAW_OUT_ROOT = os.path.join(REPO_ROOT, "data", "v1", "sources", "vbvr", "raw", "rotation_puzzle")
DATASET_ROOT = os.path.join(REPO_ROOT, "data", "v1", "datasets", "rotation_puzzle_Composed_intermediate")
META_PATH = os.path.join(REPO_ROOT, "data", "v1", "meta", "rotation_puzzle_Composed_intermediate_meta.json")
TRAIN_META_PATH = os.path.join(REPO_ROOT, "data", "v1", "meta", "rotation_puzzle_Composed_intermediate_train_meta.json")

NUM_TOTAL = 600
NUM_TRAIN = 500
NUM_EVAL = 100
SEED = 20260916
SPLIT_SHUFFLE_SEED = 20260916
GRID_CELL = 512
INTERMEDIATE_PROGRESS = [1 / 3, 2 / 3]

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


def build_puzzle_data(meta_task_data, canvas_size):
    squares = []
    for obj in meta_task_data["objects"]:
        pp = obj["pipe_pattern"]
        squares.append({
            "index": obj["index"],
            "position": tuple(obj["position"]),
            "pipe_pattern": (int(pp["top"]), int(pp["right"]), int(pp["bottom"]), int(pp["left"])),
            "initial_angle": obj["initial_angle"],
            "target_angle": obj["target_angle"],
            "rotation_angle": obj["rotation_angle"],
        })
    return {
        "squares": squares,
        "canvas_size": canvas_size,
        "square_size": meta_task_data["square_size"],
        "difficulty": meta_task_data["difficulty"],
        "pipe_color": tuple(meta_task_data["pipe_color"]),
    }


def render_progress(gen, puzzle_data, progress):
    step_data = dict(puzzle_data)
    step_data["squares"] = []
    for sq in puzzle_data["squares"]:
        new_sq = dict(sq)
        new_sq["initial_angle"] = sq["initial_angle"] + sq["rotation_angle"] * progress
        step_data["squares"].append(new_sq)
    return gen._render_frame(step_data, use_initial_angles=True)


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

    config = TaskConfig(num_samples=NUM_TOTAL, random_seed=SEED, generate_videos=False)
    gen = TaskGenerator(config)
    writer = OutputWriter(RAW_OUT_ROOT)

    bg = tuple(config.background_color)

    rows = []
    for i in range(NUM_TOTAL):
        task_id = f"{config.domain}_{i:08d}"
        pair = gen.generate_task_pair(task_id)
        writer.write_task_pair(pair)

        meta = pair.metadata
        if meta["param_hash"] in existing_hashes:
            raise RuntimeError(f"{task_id}: param_hash collides with the original probe -- pick a different SEED")
        existing_hashes.add(meta["param_hash"])

        puzzle_data = build_puzzle_data(meta["parameters"], config.canvas_size)

        first_img = pair.first_image
        final_img = pair.final_image

        rerendered_first = render_progress(gen, puzzle_data, progress=0.0)
        if not images_equal(first_img, rerendered_first):
            raise RuntimeError(f"{task_id}: progress=0 fidelity check failed")
        rerendered_final = render_progress(gen, puzzle_data, progress=1.0)
        if not images_equal(final_img, rerendered_final):
            raise RuntimeError(f"{task_id}: progress=1 fidelity check failed")

        inter1_img = render_progress(gen, puzzle_data, progress=INTERMEDIATE_PROGRESS[0])
        inter2_img = render_progress(gen, puzzle_data, progress=INTERMEDIATE_PROGRESS[1])

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
                    "task_name": "rotation_puzzle_Composed_intermediate",
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
        "rotation_puzzle_composed_intermediate_train": {
            "root": "/", "annotation": train_path, "data_augment": False,
            "max_dynamic_patch": 1, "repeat_time": 1, "length": len(train_idx), "task_type": "imgen",
        },
        "rotation_puzzle_composed_intermediate_eval": {
            "root": "/", "annotation": eval_path, "data_augment": False,
            "max_dynamic_patch": 1, "repeat_time": 1, "length": len(eval_idx), "task_type": "imgen",
        },
    }
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    with open(TRAIN_META_PATH, "w") as f:
        json.dump({"rotation_puzzle_composed_intermediate_train": meta["rotation_puzzle_composed_intermediate_train"]}, f, indent=2)

    print(f"\nDone: {len(train_idx)} train, {len(eval_idx)} eval written to {DATASET_ROOT}")


if __name__ == "__main__":
    main()
