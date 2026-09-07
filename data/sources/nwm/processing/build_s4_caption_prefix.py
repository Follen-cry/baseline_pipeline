"""Recover the NWM caption prefix for each S4 base-set (S2_train.jsonl) row,
by joining against nwm_gen_action's own manifest_train.jsonl on
context_paths (see ../../../common/s4_caption_lookup.py).

Unlike the other 4 sources, NWM's S2 prompt jumps straight into "Action to
execute from the last frame: ..." with no scene-setting sentence (EPIC says
"egocentric kitchen video", panda70m says "Video description: ..."). Per the
2026-09-06 decision (../../../recipes/s4_progression/README.md), S4 prepends
a fixed domain-framing sentence here -- this is the one caption_prefix in
the whole pipeline that isn't just "reuse S2's text verbatim".

Also fixes, for S4 only, the stale "Observe these four frames" hardcoding in
build_nwm_ssl.py's build_gen_prompt_action (CONTEXT_SIZE was shrunk 4->3 on
2026-08-31 but that string literal wasn't updated) -- S4's new prefix uses
the actual context length via s4_progression_prompt.nwm_domain_prefix(n),
not a hardcoded word. S0-S3's existing data/code is untouched.

Usage:
    python build_s4_caption_prefix.py \
        --s2-jsonl ../../../data/datasets/final_s0s3/S2_train.jsonl \
        --manifest ../../../data/datasets/nwm_ssl/nwm_gen_action/manifest_train.jsonl \
        --out-jsonl /tmp/nwm_s4_caption_prefix.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from s4_caption_lookup import derive_source_caption_prefixes
from s4_progression_prompt import nwm_domain_prefix

SOURCE_NAME = "nwm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-jsonl", required=True)
    ap.add_argument("--manifest", required=True, help="nwm_gen_action/manifest_train.jsonl")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--n-context", type=int, default=3)
    args = ap.parse_args()

    domain_prefix = nwm_domain_prefix(args.n_context)

    rows = derive_source_caption_prefixes(
        Path(args.s2_jsonl), SOURCE_NAME, [Path(args.manifest)],
        extract=lambda d: f"{domain_prefix} Action to execute from the last frame: {d['action_text']}",
    )
    with open(args.out_jsonl, "w") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
    print(f"nwm: wrote {len(rows)} caption prefixes -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
