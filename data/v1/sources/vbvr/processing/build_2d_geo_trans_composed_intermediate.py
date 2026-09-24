#!/usr/bin/env python3
"""
Build the "2d_geo_trans_Composed_intermediate" dataset: 500 train / 100 eval,
scaled up from the 10-sample composed-input probe at
data/v1/sources/vbvr/pilots/intermediate_grid/scripts/build_partial_grid_input.py +
Evaluation/vbvr_baseline_runners/run_intermediate_grid_infer_composed.py.

Format (unchanged from the probe): a single partial 2x2 grid image is the
ONE input (top-left=first_frame, bottom-right=final_frame, top-right/
bottom-left=blank), and the target is the SAME canvas with the two blank
cells filled in with the REAL rendered intermediate frames (not a
placeholder) -- this is a genuine same-canvas image edit, so a real trained
recipe on this dataset should VAE-condition on the single input image (see
run_intermediate_grid_infer_composed.py's vae_cond_last_frame=True default).

600 FRESH 2d_geometric_transformation samples are generated (seed 20260916,
disjoint by construction from the original 10-sample probe's seed=777 -- and
defensively checked here against those 10 samples' param_hash to guarantee
zero overlap) using the vendored generator at
data/v1/sources/vbvr/generators/2d_geometric_transformation/. The 2 intermediate
frames per sample are rendered deterministically from each sample's own
metadata.json the same fidelity-checked way as
data/v1/sources/vbvr/pilots/intermediate_grid/scripts/build_intermediate_grid_samples.py (re-derived
first frame asserted byte-identical to the generator's own first_frame.png
before any intermediate frame is trusted).

Writes:
  data/v1/sources/vbvr/raw/2d_geometric_transformation/
    transformation_worlds_2d_geometric_transformation_task/
      transformation_worlds_2d_geometric_transformation_{00000000..00000599}/
        first_frame.png, final_frame.png, prompt.txt, metadata.json   (raw generator output, untouched)

  data/v1/datasets/2d_geo_trans_Composed_intermediate/
    train.jsonl (500 rows), eval.jsonl (100 rows)
    images/{train,eval}/<id>/partial_grid_input.png   (the 1 input image)
    images/{train,eval}/<id>/gt_grid.png               (the target image -- REAL ground truth)
    images/{train,eval}/<id>/intermediate_1.png, intermediate_2.png  (debug/inspection only, not referenced by the jsonl)

  data/v1/meta/2d_geo_trans_Composed_intermediate_meta.json
"""
import json
import os
import random
import sys

from PIL import Image, ImageDraw, ImageChops

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))  # baseline_pipeline/
GENERATOR_ROOT = os.path.join(REPO_ROOT, "data", "v1", "sources", "vbvr", "generators", "2d_geometric_transformation")
ORIGINAL_PROBE_RAW_TASK_DIR = (
    "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/VBVR-CustomEval/eval_samples/raw/"
    "2d_geometric_transformation/transformation_worlds_2d_geometric_transformation_task"
)
RAW_OUT_ROOT = os.path.join(REPO_ROOT, "data", "v1", "sources", "vbvr", "raw", "2d_geometric_transformation")
DATASET_ROOT = os.path.join(REPO_ROOT, "data", "v1", "datasets", "2d_geo_trans_Composed_intermediate")
META_PATH = os.path.join(REPO_ROOT, "data", "v1", "meta", "2d_geo_trans_Composed_intermediate_meta.json")

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

# Identical to build_partial_grid_input.py's wrapper -- kept in sync by hand,
# this is the exact prompt shape scaled up here.
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


def render_progress(gen, config, base_shape, rotation_center, final_shape,
                     object_color, initial_angle, rotation_angle, progress):
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


def compose_partial_grid(first_img, final_img, cell=GRID_CELL, bg=(240, 240, 240)):
    grid = Image.new("RGB", (cell * 2, cell * 2), color=bg)
    grid.paste(first_img.resize((cell, cell), Image.LANCZOS), (0, 0))
    grid.paste(final_img.resize((cell, cell), Image.LANCZOS), (cell, cell))
    return grid


def compose_full_grid(first_img, inter1_img, inter2_img, final_img, cell=GRID_CELL, bg=(240, 240, 240)):
    grid = Image.new("RGB", (cell * 2, cell * 2), color=bg)
    grid.paste(first_img.resize((cell, cell), Image.LANCZOS), (0, 0))
    grid.paste(inter1_img.resize((cell, cell), Image.LANCZOS), (cell, 0))
    grid.paste(inter2_img.resize((cell, cell), Image.LANCZOS), (0, cell))
    grid.paste(final_img.resize((cell, cell), Image.LANCZOS), (cell, cell))
    return grid


def load_existing_probe_param_hashes():
    hashes = set()
    for i in range(10):
        p = os.path.join(
            ORIGINAL_PROBE_RAW_TASK_DIR,
            f"transformation_worlds_2d_geometric_transformation_{i:08d}",
            "metadata.json",
        )
        meta = json.load(open(p))
        hashes.add(meta["param_hash"])
    assert len(hashes) == 10
    return hashes


def main():
    existing_hashes = load_existing_probe_param_hashes()

    config = TaskConfig(num_samples=NUM_TOTAL, random_seed=SEED, generate_videos=False)
    gen = TaskGenerator(config)
    writer = OutputWriter(RAW_OUT_ROOT)

    rows = []
    for i in range(NUM_TOTAL):
        task_id = f"{config.domain}_{i:08d}"
        pair = gen.generate_task_pair(task_id)
        writer.write_task_pair(pair)

        meta = pair.metadata
        if meta["param_hash"] in existing_hashes:
            raise RuntimeError(
                f"{task_id}: param_hash collides with the original 10-sample probe "
                f"-- pick a different SEED"
            )
        existing_hashes.add(meta["param_hash"])

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

        first_img = pair.first_image
        final_img = pair.final_image

        rerendered_first = render_progress(
            gen, config, base_shape, rotation_center, final_shape,
            object_color, initial_angle, rotation_angle, progress=0.0
        )
        if not images_equal(first_img, rerendered_first):
            raise RuntimeError(f"{task_id}: fidelity check failed, do not trust intermediate frames")

        inter1_img = render_progress(
            gen, config, base_shape, rotation_center, final_shape,
            object_color, initial_angle, rotation_angle, progress=INTERMEDIATE_PROGRESS[0]
        )
        inter2_img = render_progress(
            gen, config, base_shape, rotation_center, final_shape,
            object_color, initial_angle, rotation_angle, progress=INTERMEDIATE_PROGRESS[1]
        )

        partial_input = compose_partial_grid(first_img, final_img)
        full_grid = compose_full_grid(first_img, inter1_img, inter2_img, final_img)

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
                    "task_name": "2d_geo_trans_Composed_intermediate",
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
        "2d_geo_trans_composed_intermediate_train": {
            "root": "/", "annotation": train_path, "data_augment": False,
            "max_dynamic_patch": 1, "repeat_time": 1, "length": len(train_idx), "task_type": "imgen",
        },
        "2d_geo_trans_composed_intermediate_eval": {
            "root": "/", "annotation": eval_path, "data_augment": False,
            "max_dynamic_patch": 1, "repeat_time": 1, "length": len(eval_idx), "task_type": "imgen",
        },
    }
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {len(train_idx)} train, {len(eval_idx)} eval written to {DATASET_ROOT}")
    print(f"meta: {META_PATH}")


if __name__ == "__main__":
    main()
