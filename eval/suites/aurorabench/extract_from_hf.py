#!/usr/bin/env python
"""One-off: materialize McGill-NLP/aurora-bench's parquet-embedded images to
a json + images/ layout on disk that dataset.py can read (mirrors what the
official GitHub repo ships, per its README: "if you simply want a json and
image folders (I personally find parquet confusing)").

The HF `input.path` field is not a stable per-row id (AURORA chains edits,
so a filename like `368667-output1.png` is reused as another row's
*input*), so this assigns a fresh `<source>_<running-index>` key per row
instead.

Usage:
  python extract_from_hf.py --out /scratch/local/ssd/junlin/data/AuroraBench
"""
import argparse
import json
import os
from collections import Counter

import pandas as pd
from huggingface_hub import hf_hub_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo_id", default="McGill-NLP/aurora-bench")
    args = ap.parse_args()

    parquet_path = hf_hub_download(
        repo_id=args.repo_id, repo_type="dataset", filename="data/test-00000-of-00001.parquet",
    )
    df = pd.read_parquet(parquet_path)

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    items = []
    per_source_idx = {}
    for _, row in df.iterrows():
        source = row["source"]
        i = per_source_idx.get(source, 0)
        per_source_idx[source] = i + 1
        key = f"{source}_{i:03d}"
        img_dir = os.path.join(args.out, "images", source)
        os.makedirs(img_dir, exist_ok=True)
        rel_path = f"images/{source}/{key}.png"
        with open(os.path.join(args.out, rel_path), "wb") as f:
            f.write(row["input"]["bytes"])
        items.append({
            "key": key,
            "source": source,
            "instruction": row["instruction"],
            "input": rel_path,
            "orig_filename": row["input"]["path"],
        })

    with open(os.path.join(args.out, "test.json"), "w") as f:
        json.dump(items, f, indent=2)

    print(f"wrote {len(items)} items -> {args.out}/test.json")
    print(Counter(it["source"] for it in items))


if __name__ == "__main__":
    main()
