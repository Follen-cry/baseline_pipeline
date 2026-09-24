#!/usr/bin/env python3
"""Build multi_object_placement_Composed_intermediate_cross: same composed
2x2-grid task as multi_object_placement_Composed_intermediate, but
  * a dashed "+" divider is drawn across the grid (input AND target), and the
    prompt tells the model about it;
  * 3000 train / 20 eval, freshly generated with a new seed (disjoint param_hash
    from the original 600-sample set).

Reuses the fidelity-checked rendering helpers from the original build script.
"""
import json
import os
import random
import sys

from PIL import Image, ImageColor, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_multi_object_placement_composed_intermediate as base  # noqa: E402

REPO_ROOT = base.REPO_ROOT
GENERATOR_ROOT = base.GENERATOR_ROOT
from src import TaskGenerator, TaskConfig  # noqa: E402
from core import OutputWriter  # noqa: E402

NAME = "multi_object_placement_Composed_intermediate_cross"
RAW_OUT_ROOT = os.path.join(REPO_ROOT, "data", "sources", "vbvr", "raw", "multi_object_placement_cross")
DATASET_ROOT = os.path.join(REPO_ROOT, "data", "datasets", NAME)
META_PATH = os.path.join(REPO_ROOT, "data", "meta", f"{NAME}_meta.json")
TRAIN_META_PATH = os.path.join(REPO_ROOT, "data", "meta", f"{NAME}_train_meta.json")
OLD_RAW_TASK_DIR = os.path.join(REPO_ROOT, "data", "sources", "vbvr", "raw", "multi_object_placement", "multi_object_placement_task")

NUM_TRAIN = 3000
NUM_EVAL = 20
NUM_TOTAL = NUM_TRAIN + NUM_EVAL
SEED = 20260918
SPLIT_SEED = 20260918

DIV_COLOR = (90, 90, 90)
DIV_WIDTH = 6
DASH_ON, DASH_OFF = 24, 16

PROMPT_WRAPPER = (
    "You are given a single 2x2 grid image. A dashed cross (a vertical and a "
    "horizontal dashed line meeting at the center) divides the image into four "
    "equal cells. The top-left cell shows the FIRST frame (starting state), and "
    "the bottom-right cell shows the FINAL frame (ending state). The top-right "
    "and bottom-left cells are blank. Fill in these two blank cells with the two "
    "intermediate frames of the transformation, in temporal order (top-right = "
    "first intermediate step, bottom-left = second intermediate step), so that "
    "reading top-left -> top-right -> bottom-left -> bottom-right shows the "
    "transformation progressing smoothly and consistently with the rule "
    "described below. Keep the top-left and bottom-right cells exactly as given, "
    "keep the dashed cross divider in the same place in the output, do not add "
    "any other gutters or labels, and output only the single complete 2x2 grid "
    "image.\n\nTask: {prompt}"
)


def draw_cross(grid):
    w, h = grid.size
    d = ImageDraw.Draw(grid)
    cx, cy = w // 2, h // 2
    half = DIV_WIDTH // 2
    for y in range(0, h, DASH_ON + DASH_OFF):
        d.rectangle([cx - half, y, cx + half - 1, min(y + DASH_ON, h) - 1], fill=DIV_COLOR)
    for x in range(0, w, DASH_ON + DASH_OFF):
        d.rectangle([x, cy - half, min(x + DASH_ON, w) - 1, cy + half - 1], fill=DIV_COLOR)
    return grid


def old_hashes():
    hs = set()
    for n in os.listdir(OLD_RAW_TASK_DIR):
        p = os.path.join(OLD_RAW_TASK_DIR, n, "metadata.json")
        if os.path.isfile(p):
            hs.add(json.load(open(p))["param_hash"])
    return hs


def main():
    seen = old_hashes()
    config = TaskConfig(num_samples=NUM_TOTAL, random_seed=SEED, generate_videos=False)
    gen = TaskGenerator(config)
    writer = OutputWriter(RAW_OUT_ROOT)
    bg = ImageColor.getrgb(config.background_color) if isinstance(config.background_color, str) else tuple(config.background_color)

    idx = list(range(NUM_TOTAL))
    random.Random(SPLIT_SEED).shuffle(idx)
    eval_set = set(idx[NUM_TRAIN:])

    files = {s: open(os.path.join(DATASET_ROOT, f"{s}.jsonl"), "w") for s in ()}
    os.makedirs(DATASET_ROOT, exist_ok=True)
    files = {s: open(os.path.join(DATASET_ROOT, f"{s}.jsonl"), "w") for s in ("train", "eval")}
    counts = {"train": 0, "eval": 0}
    skipped = 0

    for i in range(NUM_TOTAL):
        task_id = f"{config.domain}_cross_{i:08d}"
        pair = gen.generate_task_pair(task_id)
        meta = pair.metadata
        if meta["param_hash"] in seen:
            skipped += 1
            continue
        seen.add(meta["param_hash"])
        writer.write_task_pair(pair)

        params = meta["parameters"]
        objects = [{"id": o["id"], "shape": o["shape"], "color": o["color"], "size": o["size"], "position": tuple(o["position"])} for o in params["objects"]]
        markers = [{"id": m["id"], "color": m["color"], "size": m["size"], "position": tuple(m["position"])} for m in params["markers"]]

        first_img, final_img = pair.first_image, pair.final_image
        if not base.images_equal(first_img, base.render_progress(gen, config, objects, markers, 0.0)):
            raise RuntimeError(f"{task_id}: progress=0 fidelity check failed")
        if not base.images_equal(final_img, base.render_progress(gen, config, objects, markers, 1.0)):
            raise RuntimeError(f"{task_id}: progress=1 fidelity check failed")
        inter1 = base.render_progress(gen, config, objects, markers, base.INTERMEDIATE_PROGRESS[0])
        inter2 = base.render_progress(gen, config, objects, markers, base.INTERMEDIATE_PROGRESS[1])

        partial = draw_cross(base.compose_partial_grid(first_img, final_img, bg))
        full = draw_cross(base.compose_full_grid(first_img, inter1, inter2, final_img, bg))

        split = "eval" if i in eval_set else "train"
        sdir = os.path.join(DATASET_ROOT, "images", split, task_id)
        os.makedirs(sdir, exist_ok=True)
        partial_path = os.path.join(sdir, "partial_grid_input.png")
        grid_path = os.path.join(sdir, "gt_grid.png")
        partial.save(partial_path)
        full.save(grid_path)

        row = {
            "id": task_id, "task_type": "imgen", "task_name": NAME, "category": "Knowledge", "split": split,
            "image": [partial_path], "target_image": grid_path,
            "conversations": [
                {"from": "human", "value": "<image>\n" + PROMPT_WRAPPER.format(prompt=pair.prompt)},
                {"from": "gpt", "value": "The complete 2x2 grid should look like this: <img>"},
            ],
        }
        files[split].write(json.dumps(row) + "\n")
        counts[split] += 1
        if (i + 1) % 250 == 0:
            print(f"[build] {i + 1}/{NUM_TOTAL}", flush=True)

    for f in files.values():
        f.close()

    def entry(split):
        return {"root": "/", "annotation": os.path.join(DATASET_ROOT, f"{split}.jsonl"), "data_augment": False,
                "max_dynamic_patch": 1, "repeat_time": 1, "length": counts[split], "task_type": "imgen"}

    meta = {f"{NAME}_train": entry("train"), f"{NAME}_eval": entry("eval")}
    json.dump(meta, open(META_PATH, "w"), indent=2)
    json.dump({f"{NAME}_train": meta[f"{NAME}_train"]}, open(TRAIN_META_PATH, "w"), indent=2)
    print(f"Done: train={counts['train']} eval={counts['eval']} skipped_dupes={skipped}")


if __name__ == "__main__":
    main()
