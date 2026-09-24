#!/usr/bin/env python3
"""
Build the "multi_object_placement_Composed_intermediate" dataset: 500 train /
100 eval, adopting the same composed-single-image-input pipeline used for
2d_geo_trans_Composed_intermediate (see that dataset's README.md and
data/v1/sources/vbvr/processing/build_2d_geo_trans_composed_intermediate.py).

Task mechanics (multi_object_placement): N colored shapes (2-5, random) each
move toward a same-colored four-pointed-star marker. All N objects move
SIMULTANEOUSLY under one shared progress fraction -- confirmed by reading the
generator's own `_generate_animation_frames`: each object's position is
linearly interpolated `start + (target - start) * progress`, then clamped to
stay `size+10` px from the canvas edge. This is structurally identical to the
rotation task's single-progress-parameter sweep, just N independent position
interpolants instead of one angle.

600 FRESH samples are generated (seed 20260916, disjoint from the original
10-sample eval_samples probe's seed=777, checked here against those 10
samples' param_hash) using the vendored generator at
data/v1/sources/vbvr/generators/multi_object_placement/. The 2 intermediate
frames per sample are rendered by re-running the exact interpolation+clamp
formula from `_generate_animation_frames` at progress 1/3 and 2/3, using the
generator's own `_draw_marker`/`_draw_object_with_gradient_border` draw
calls. Fidelity is checked TWO ways (unlike the rotation task's one-sided
check): progress=0 must equal the real first_frame.png AND progress=1 must
equal the real final_frame.png, since final_frame is rendered by a different
generator method (_create_frame with final_positions=True, no clamp) whose
agreement with the clamped interpolation formula at progress=1 is worth
verifying rather than assuming.

Writes:
  data/v1/sources/vbvr/raw/multi_object_placement/
    multi_object_placement_task/multi_object_placement_{00000000..00000599}/
      first_frame.png, final_frame.png, prompt.txt, metadata.json

  data/v1/datasets/multi_object_placement_Composed_intermediate/
    train.jsonl (500 rows), eval.jsonl (100 rows)
    images/{train,eval}/<id>/partial_grid_input.png   (the 1 input image)
    images/{train,eval}/<id>/gt_grid.png               (the target image -- REAL ground truth)
    images/{train,eval}/<id>/intermediate_1.png, intermediate_2.png  (debug/inspection only)

  data/v1/meta/multi_object_placement_Composed_intermediate_meta.json
  data/v1/meta/multi_object_placement_Composed_intermediate_train_meta.json
"""
import json
import os
import random
import sys

from PIL import Image, ImageChops, ImageColor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
GENERATOR_ROOT = os.path.join(REPO_ROOT, "data", "v1", "sources", "vbvr", "generators", "multi_object_placement")
ORIGINAL_PROBE_RAW_TASK_DIR = (
    "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/eval_samples/raw/"
    "multi_object_placement/multi_object_placement_task"
)
RAW_OUT_ROOT = os.path.join(REPO_ROOT, "data", "v1", "sources", "vbvr", "raw", "multi_object_placement")
DATASET_ROOT = os.path.join(REPO_ROOT, "data", "v1", "datasets", "multi_object_placement_Composed_intermediate")
META_PATH = os.path.join(REPO_ROOT, "data", "v1", "meta", "multi_object_placement_Composed_intermediate_meta.json")
TRAIN_META_PATH = os.path.join(REPO_ROOT, "data", "v1", "meta", "multi_object_placement_Composed_intermediate_train_meta.json")

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


def render_progress(gen, config, objects, markers, progress):
    width, height = config.image_size
    img = Image.new("RGB", (width, height), config.background_color)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    for marker in markers:
        gen._draw_marker(draw, marker)

    for obj in objects:
        start_x, start_y = obj["position"]
        marker = next(m for m in markers if m["color"] == obj["color"])
        target_x, target_y = marker["position"]

        current_x = int(start_x + (target_x - start_x) * progress)
        current_y = int(start_y + (target_y - start_y) * progress)

        margin = obj["size"] + 10
        current_x = max(margin, min(width - margin, current_x))
        current_y = max(margin, min(height - margin, current_y))

        temp_obj = dict(obj)
        temp_obj["position"] = (current_x, current_y)
        gen._draw_object_with_gradient_border(draw, temp_obj)

    return img


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

    bg = ImageColor.getrgb(config.background_color) if isinstance(config.background_color, str) else tuple(config.background_color)

    rows = []
    for i in range(NUM_TOTAL):
        task_id = f"{config.domain}_{i:08d}"
        pair = gen.generate_task_pair(task_id)
        writer.write_task_pair(pair)

        meta = pair.metadata
        if meta["param_hash"] in existing_hashes:
            raise RuntimeError(f"{task_id}: param_hash collides with the original probe -- pick a different SEED")
        existing_hashes.add(meta["param_hash"])

        params = meta["parameters"]
        objects = [
            {"id": o["id"], "shape": o["shape"], "color": o["color"], "size": o["size"], "position": tuple(o["position"])}
            for o in params["objects"]
        ]
        markers = [
            {"id": m["id"], "color": m["color"], "size": m["size"], "position": tuple(m["position"])}
            for m in params["markers"]
        ]

        first_img = pair.first_image
        final_img = pair.final_image

        rerendered_first = render_progress(gen, config, objects, markers, progress=0.0)
        if not images_equal(first_img, rerendered_first):
            raise RuntimeError(f"{task_id}: progress=0 fidelity check failed")
        rerendered_final = render_progress(gen, config, objects, markers, progress=1.0)
        if not images_equal(final_img, rerendered_final):
            raise RuntimeError(f"{task_id}: progress=1 fidelity check failed")

        inter1_img = render_progress(gen, config, objects, markers, progress=INTERMEDIATE_PROGRESS[0])
        inter2_img = render_progress(gen, config, objects, markers, progress=INTERMEDIATE_PROGRESS[1])

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
                    "task_name": "multi_object_placement_Composed_intermediate",
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

    train_path = write_split("train", train_idx)
    eval_path = write_split("eval", eval_idx)

    meta = {
        "multi_object_placement_composed_intermediate_train": {
            "root": "/", "annotation": train_path, "data_augment": False,
            "max_dynamic_patch": 1, "repeat_time": 1, "length": len(train_idx), "task_type": "imgen",
        },
        "multi_object_placement_composed_intermediate_eval": {
            "root": "/", "annotation": eval_path, "data_augment": False,
            "max_dynamic_patch": 1, "repeat_time": 1, "length": len(eval_idx), "task_type": "imgen",
        },
    }
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    with open(TRAIN_META_PATH, "w") as f:
        json.dump({"multi_object_placement_composed_intermediate_train": meta["multi_object_placement_composed_intermediate_train"]}, f, indent=2)

    print(f"\nDone: {len(train_idx)} train, {len(eval_idx)} eval written to {DATASET_ROOT}")


if __name__ == "__main__":
    main()
