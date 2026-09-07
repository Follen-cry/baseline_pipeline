"""Derive S4 (progression-description + next-frame text-and-image
prediction) from S2's exact (image, cond_image, target_image) window set --
see README.md in this dir for the full design/decision log.

This script does NOT produce a training-ready dataset by itself: the GPT
turn's ground-truth text (progression description + next-frame text
prediction) has to come from an external VLM labeling pass that hasn't
been run yet (model/final prompt still open -- see README.md). What this
script produces:

  S4_skeleton.jsonl            one row per S2_train.jsonl row, with the new
                                human turn already built and everything else
                                (image/cond_image/target_image/source/id/...)
                                carried over from S2 unchanged. `conversations`
                                has only the human turn; the gpt turn is
                                added by fill_s4_captions.py once labeling
                                is done.
  S4_captioning_manifest.jsonl one row per S2 row: the exact prompt + image
                                paths (context frames + target, in order) to
                                hand to whichever VLM gets chosen -- ready to
                                feed a labeling script/notebook directly.

Usage:
    python derive_s4_rows.py \
        --s2-jsonl ../../data/datasets/final_s0s3/S2_train.jsonl \
        --caption-prefix-dir /path/to/dir/with/5/*_s4_caption_prefix.jsonl \
        --out-dir ../../data/datasets/s4_progression
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from s4_progression_prompt import build_s4_human_prompt, build_vlm_label_prompt

N_CONTEXT = 3  # confirmed uniform across all 5 sources for S0-S2 (2026-09-06)

SOURCE_CAPTION_FILES = {
    "vbvr": "vbvr_s4_caption_prefix.jsonl",
    "epic": "epic_s4_caption_prefix.jsonl",
    "nwm": "nwm_s4_caption_prefix.jsonl",
    "panda70m_epic": "panda70m_epic_s4_caption_prefix.jsonl",
    "panda70m_v1": "panda70m_v1_s4_caption_prefix.jsonl",
}


def load_caption_prefixes(caption_prefix_dir: Path) -> dict[tuple, str]:
    lookup = {}
    for source, fname in SOURCE_CAPTION_FILES.items():
        path = caption_prefix_dir / fname
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                lookup[tuple(d["context_paths"])] = d["caption_prefix"]
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-jsonl", required=True)
    ap.add_argument("--caption-prefix-dir", required=True,
                     help="dir containing the 5 sources' *_s4_caption_prefix.jsonl "
                          "(from each source's build_s4_caption_prefix.py)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lookup = load_caption_prefixes(Path(args.caption_prefix_dir))

    n_rows, by_source = 0, {}
    with open(args.s2_jsonl) as f, \
         open(out_dir / "S4_skeleton.jsonl", "w") as skel_out, \
         open(out_dir / "S4_captioning_manifest.jsonl", "w") as manifest_out:
        for line in f:
            row = json.loads(line)
            key = tuple(row["image"])
            if key not in lookup:
                raise RuntimeError(
                    f"row {row.get('id')} (source={row.get('source')}): no caption "
                    "prefix found -- run each source's build_s4_caption_prefix.py "
                    "first and point --caption-prefix-dir at their combined output."
                )
            caption_prefix = lookup[key]
            human_prompt = build_s4_human_prompt(N_CONTEXT, caption_prefix)

            skeleton_row = dict(row)
            skeleton_row["setting"] = "S4"
            skeleton_row["conversations"] = [{"from": "human", "value": human_prompt}]
            skel_out.write(json.dumps(skeleton_row, ensure_ascii=False) + "\n")

            manifest_out.write(json.dumps({
                "id": row["id"],
                "source": row["source"],
                "task_name": row.get("task_name"),
                "context_paths": row["image"],
                "target_image": row["target_image"],
                "caption_prefix": caption_prefix,
                "vlm_prompt": build_vlm_label_prompt(N_CONTEXT, caption_prefix),
                "vlm_images": list(row["image"]) + [row["target_image"]],
            }, ensure_ascii=False) + "\n")

            n_rows += 1
            by_source[row["source"]] = by_source.get(row["source"], 0) + 1

    print(f"S4: wrote {n_rows} skeleton rows + manifest entries {by_source} -> {out_dir}")


if __name__ == "__main__":
    main()
