"""Recover the EPIC-Kitchens caption prefix (real EPIC-100 narration action
text) for each S4 base-set (S2_train.jsonl) row, by joining against
epic_gen_action's own manifest_train.jsonl on context_paths. See
../../../common/s4_caption_lookup.py for why the join key is context_paths
(not row id) and ../../../recipes/s4_progression/README.md for the S2->S4
prompt-adaptation decision log.

Usage:
    python build_s4_caption_prefix.py \
        --s2-jsonl ../../../data/datasets/final_s0s3/S2_train.jsonl \
        --manifest ../../../data/datasets/epic_ssl/epic_gen_action/manifest_train.jsonl \
        --out-jsonl /tmp/epic_s4_caption_prefix.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from s4_caption_lookup import derive_source_caption_prefixes

SOURCE_NAME = "epic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-jsonl", required=True)
    ap.add_argument("--manifest", required=True, help="epic_gen_action/manifest_train.jsonl")
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    def extract(d: dict) -> str:
        n = len(d["context_paths"])
        return (f"These are {n} consecutive first-person camera frames from an "
                f"egocentric kitchen video. Action being performed after the last "
                f"frame: {d['action_text']}")

    rows = derive_source_caption_prefixes(
        Path(args.s2_jsonl), SOURCE_NAME, [Path(args.manifest)], extract=extract,
    )
    with open(args.out_jsonl, "w") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
    print(f"epic: wrote {len(rows)} caption prefixes -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
