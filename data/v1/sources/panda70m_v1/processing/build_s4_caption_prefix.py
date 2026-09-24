"""Recover the panda70m_v1 caption prefix (real Panda-70M video caption) for
each S4 base-set (S2_train.jsonl) row, by joining against
panda70m_v1_gen_S2's own manifest_train.jsonl on context_paths (see
../../../common/s4_caption_lookup.py). Matches build_panda70m_ssl.py's own
build_gen_prompt()'s "Video description: {caption}" formatting verbatim.

Usage:
    python build_s4_caption_prefix.py \
        --s2-jsonl ../../../data/v1/datasets/final_s0s3/S2_train.jsonl \
        --manifest ../../../data/datasets/panda70m_ssl_v1/panda70m_v1_gen_S2/manifest_train.jsonl \
        --out-jsonl /tmp/panda70m_v1_s4_caption_prefix.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from s4_caption_lookup import derive_source_caption_prefixes

SOURCE_NAME = "panda70m_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-jsonl", required=True)
    ap.add_argument("--manifest", required=True, help="panda70m_v1_gen_S2/manifest_train.jsonl")
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    rows = derive_source_caption_prefixes(
        Path(args.s2_jsonl), SOURCE_NAME, [Path(args.manifest)],
        extract=lambda d: f"Video description: {d['caption'].strip()}",
    )
    with open(args.out_jsonl, "w") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
    print(f"panda70m_v1: wrote {len(rows)} caption prefixes -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
