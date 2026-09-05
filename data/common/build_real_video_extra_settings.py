"""Generic S1 (variable-direction, no caption) derivation -- and, where a
source's own build_gen() never got caption-injection wired in, a captioned
S2 -- from any SelectionExample-shaped anchors_{split}.jsonl (EPIC/NWM/
panda70m_epic_ssl/panda70m_ssl all write this same shape: example_id,
context_paths, candidate_paths, candidate_types, answer_index, metadata).

Why a separate generic script instead of extending each source's own
build_gen(): EPIC and NWM already have action/noaction variants that ARE
S2/S0 respectively (their own real narration/pose-instruction text serves as
"C"), so they only need S1 added. panda70m_epic_ssl never got caption
injection into its single noaction variant, so it needs both S1 and a real
S2. Rather than duplicate the direction-reassignment logic four times with
source-specific glue, this reads the shared anchor shape directly as JSON.

Usage:
    python data/scripts/build_real_video_extra_settings.py \
        --anchors-jsonl data/datasets/epic_ssl/anchors/anchors_train.jsonl \
        --out-jsonl data/datasets/epic_ssl/epic_gen_S1/epic_gen_S1_train.jsonl \
        --out-meta data/meta/epic_gen_S1_meta.json --meta-key epic_gen_S1 \
        --direction mixed --scene-phrase "an egocentric kitchen video" \
        --setting-label S1 --max-dynamic-patch 3
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DIRECTIONS = ("forward", "backward")


def pick_direction(direction_arg: str, rng: random.Random) -> str:
    if direction_arg == "mixed":
        return "backward" if rng.random() < 0.5 else "forward"
    if direction_arg not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction_arg!r}")
    return direction_arg


def direction_ctx_target_cond(ctx_paths: list[str], target_path: str, direction: str):
    """Same free reinterpretation as VBVR/panda70m: anchors are CONTEXT_SIZE
    context frames + 1 target = CONTEXT_SIZE+1 total raw points, so backward
    just re-points which points are context vs. target -- zero new frames."""
    if direction == "backward":
        ctx = list(ctx_paths[1:]) + [target_path]
        return ctx, ctx_paths[0], ctx_paths[1]
    return list(ctx_paths), target_path, ctx_paths[-1]


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--meta-key", required=True)
    ap.add_argument("--direction", choices=["forward", "backward", "mixed"], default="mixed")
    ap.add_argument("--scene-phrase", default="a real-world video",
                     help='e.g. "an egocentric kitchen video", "a first-person camera view"')
    ap.add_argument("--caption-metadata-key", default=None,
                     help="if set, look up example['metadata'][this_key] and inject it as "
                          "'Video description: ...' (makes this a captioned S2 instead of S1)")
    ap.add_argument("--setting-label", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default="/")
    ap.add_argument("--max-dynamic-patch", type=int, default=3)
    args = ap.parse_args()

    anchors = load_jsonl(args.anchors_jsonl)
    rows = []
    for idx, ex in enumerate(anchors):
        ctx = ex["context_paths"]
        target = ex["candidate_paths"][ex["answer_index"]]
        rng = random.Random(f"{args.seed}:{ex['example_id']}:direction")
        direction = pick_direction(args.direction, rng)
        new_ctx, new_target, cond = direction_ctx_target_cond(ctx, target, direction)
        n = len(new_ctx)

        caption = None
        if args.caption_metadata_key:
            caption = (ex.get("metadata") or {}).get(args.caption_metadata_key)

        head = "".join("<image>\n" for _ in range(n))
        if direction == "backward":
            closing = (
                f"These are {n} consecutive frames from {args.scene_phrase}, occurring "
                f"immediately AFTER the frame you must predict. Based on the direction and "
                f"rate of motion established across them, predict the frame that occurred "
                f"immediately BEFORE the first frame shown."
            )
            gpt_value = "The preceding frame should look like this: <img>"
        else:
            closing = (
                f"These are {n} consecutive frames from {args.scene_phrase}. Observe these "
                f"frames and produce the next frame of the video."
            )
            gpt_value = "The next frame should look like this: <img>"
        prompt = head + (f"Video description: {caption.strip()}\n\n" if caption else "") + closing

        row = {
            "id": idx, "task_type": "imgen",
            "direction": direction, "has_caption": bool(caption),
            "image": new_ctx, "target_image": new_target, "cond_image": cond,
            "conversations": [{"from": "human", "value": prompt},
                               {"from": "gpt", "value": gpt_value}],
        }
        if args.setting_label:
            row["setting"] = args.setting_label
        rows.append(row)

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    json.dump({args.meta_key: {
        "root": args.root, "annotation": str(out_jsonl), "data_augment": False,
        "max_dynamic_patch": args.max_dynamic_patch, "repeat_time": 1,
        "length": len(rows), "task_type": "imgen"}}, open(out_meta, "w"), indent=2)

    print(f"{len(rows)} rows -> {out_jsonl}  (meta -> {out_meta})")


if __name__ == "__main__":
    main()
