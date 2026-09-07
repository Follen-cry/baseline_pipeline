"""Recover the panda70m_epic caption prefix (real Panda-70M video caption)
for each S4 base-set (S2_train.jsonl) row, by joining against this source's
own anchors_train.jsonl on context_paths (see
../../../common/s4_caption_lookup.py). panda70m_epic's S2 is built by the
shared data/common/build_real_video_extra_settings.py, which reads
`metadata.caption` straight off the same anchors file -- no separate
manifest was written for the gen_S2 variant, so we read the anchors file
directly instead.

Usage:
    python build_s4_caption_prefix.py \
        --s2-jsonl ../../../data/datasets/final_s0s3/S2_train.jsonl \
        --anchors ../../../data/datasets/panda70m_epic_ssl/anchors/anchors_train.jsonl \
        --out-jsonl /tmp/panda70m_epic_s4_caption_prefix.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from s4_caption_lookup import derive_source_caption_prefixes

SOURCE_NAME = "panda70m_epic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-jsonl", required=True)
    ap.add_argument("--anchors", required=True, help="panda70m_epic_ssl/anchors/anchors_train.jsonl")
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    rows = derive_source_caption_prefixes(
        Path(args.s2_jsonl), SOURCE_NAME, [Path(args.anchors)],
        extract=lambda d: f"Video description: {d['metadata']['caption'].strip()}",
    )
    with open(args.out_jsonl, "w") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
    print(f"panda70m_epic: wrote {len(rows)} caption prefixes -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
