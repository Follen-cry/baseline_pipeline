"""Recover, for each VBVR row in S4's base set (S2_train.jsonl), the
task-specific caption prefix that S4's human turn should keep (S2's MCQ-era
closing gets replaced -- see ../../../recipes/s4_progression/README.md).

VBVR's S2 training rows don't store the caption text as its own field (it's
only embedded in the rendered `conversations[0]` string), so this re-derives
it the same way vbvr_next_frame_make_dual.py does for S3's MCQ prompt: read
the window's `window_meta.json` (sibling of frame_0/1/2.png + target.png,
still living at the not-yet-migrated SAMPLES_ROOT) for `raw_parameters`, then
call the same `scene_description()` used everywhere else in this pipeline --
never re-implement or string-parse the rendered prompt.

Usage:
    python build_s4_caption_prefix.py \
        --s2-jsonl ../../../data/datasets/final_s0s3/S2_train.jsonl \
        --out-jsonl /tmp/vbvr_s4_caption_prefix.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vbvr_next_frame_prompt_adapt import scene_description

SAMPLES_ROOT = Path("/scratch/network/ssd/junlin/vbvr_next_frame/samples")


def caption_prefix_for_row(row: dict) -> str:
    window_dir = Path(row["image"][0]).parent
    meta_path = window_dir / "window_meta.json"
    meta = json.loads(meta_path.read_text())
    return scene_description(row["task_name"], meta.get("raw_parameters") or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    n, misses = 0, 0
    with open(args.s2_jsonl) as f, open(args.out_jsonl, "w") as out:
        for line in f:
            row = json.loads(line)
            if row.get("source") != "vbvr":
                continue
            try:
                prefix = caption_prefix_for_row(row)
            except (FileNotFoundError, KeyError) as e:
                misses += 1
                print(f"[WARN] {row.get('id')}: {e}")
                continue
            out.write(json.dumps({
                "context_paths": row["image"],
                "caption_prefix": prefix,
            }) + "\n")
            n += 1
    print(f"vbvr: wrote {n} caption prefixes ({misses} misses) -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
