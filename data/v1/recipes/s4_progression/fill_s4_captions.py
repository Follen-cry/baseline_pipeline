"""Merges VLM-labeled GPT turns (from label_with_vlm.py's --out-jsonl output)
into S4_skeleton.jsonl's human-only rows, producing the final, training-ready
S4_train.jsonl + final_S4_meta.json (matching the final_s0s3/*_meta.json
convention).

Rows whose labeling failed to parse (parsed_ok=false, e.g. a malformed VLM
response or a request error) are EXCLUDED from S4_train.jsonl rather than
included with broken/missing text -- their ids are written to
S4_dropped_ids.txt for visibility/retry, since silently shipping a row with
no real GPT turn would be worse than a slightly smaller dataset.

Usage:
    python fill_s4_captions.py \
        --skeleton ../../data/v1/datasets/s4_progression/S4_skeleton.jsonl \
        --labels ../../data/v1/datasets/s4_progression/labels/labels_shard0.jsonl \
                 ../../data/v1/datasets/s4_progression/labels/labels_shard1.jsonl \
                 ../../data/v1/datasets/s4_progression/labels/labels_shard2.jsonl \
        --out-jsonl ../../data/v1/datasets/s4_progression/S4_train.jsonl \
        --out-meta ../../data/v1/meta/final_S4_meta.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_labels(paths: list[Path]) -> dict[str, dict]:
    labels = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                labels[d["id"]] = d
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--labels", required=True, nargs="+")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--meta-key", default="final_S4")
    args = ap.parse_args()

    labels = load_labels([Path(p) for p in args.labels])

    kept, dropped, by_source = [], [], {}
    with open(args.skeleton) as f:
        for line in f:
            row = json.loads(line)
            rid = row["id"]
            label = labels.get(rid)
            if label is None or not label.get("parsed_ok"):
                dropped.append(rid)
                continue
            row["conversations"].append({"from": "gpt", "value": label["gpt_turn"]})
            kept.append(row)
            by_source[row["source"]] = by_source.get(row["source"], 0) + 1

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, row in enumerate(kept):
            row["orig_id"] = row["id"]
            row["id"] = f"{args.meta_key}_{i:06d}"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if dropped:
        dropped_path = out_path.parent / "S4_dropped_ids.txt"
        dropped_path.write_text("\n".join(dropped) + "\n")
        print(f"WARNING: {len(dropped)} rows dropped (missing/failed label) -> {dropped_path}")

    meta_path = Path(args.out_meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {args.meta_key: {
        "root": "/", "annotation": str(out_path.resolve()), "data_augment": False,
        "max_dynamic_patch": 3, "repeat_time": 1,
        "length": len(kept), "task_type": "imgen",
    }}
    json.dump(meta, open(meta_path, "w"), indent=2)

    print(f"S4: wrote {len(kept)} rows ({by_source}) -> {out_path}")
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
