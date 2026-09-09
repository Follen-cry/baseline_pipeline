#!/usr/bin/env python
"""Generate AURORA-Bench edits with InternVL-U.

Writes <out>/images/<source>/<key>.png plus a resumable
<out>/gen_manifest.json ({item_key: image_path}).

Usage (env: internvlu):
  CUDA_VISIBLE_DEVICES=<g> python gen_aurorabench_internvlu.py \
      --data_root /scratch/local/ssd/junlin/data/AuroraBench \
      --out /scratch/local/ssd/junlin/results/AuroraBench/InternVL-U --limit 0

For parallel sharding across GPUs/nodes, pass --shard_idx/--num_shards (each
shard gets every Nth item and writes its own gen_manifest_shard{i}.json to
the same --out dir; skip-on-resume checks the PNG on disk directly, so shards
and re-runs never race on a shared manifest file).

Model-package dependency (same known gap as
eval/suites/magicbrush/inference/gen_magicbrush_internvlu.py): the
`internvlu` package (InternVLUPipeline) lives at
Model_Related/InternVLU/InternVL-U in the old ssl_mllm tree, which is NOT
part of this repo's `training/models/internvl-u` submodule. PKG below points
there directly. See baseline_pipeline_repo_plan memory / PROVENANCE.md if
this gets addressed.
"""
import argparse
import json
import os
import sys
import traceback

import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SUITE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _SUITE_ROOT)
from dataset import load_items, filter_missing, item_key  # noqa: E402

PKG = "/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL-U"

DEFAULT_CKPT = (
    "/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/"
    "snapshots/f012d760e69712bb47f7d3d09a24280f346cee01"
)

# Defensive cap matching suites/risebench/inference/gen_risebench_internvlu.py --
# AURORA-Bench's images top out at 1.23MP (well under this), but very large
# inputs crash the pipeline's VAE decode (see that script's comment for the
# measured safe/unsafe boundary), so guard here too in case the data changes.
MAX_PIXELS = 1_800_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--model_path", default=DEFAULT_CKPT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all items")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    sys.path.insert(0, PKG)
    from internvlu import InternVLUPipeline

    items = filter_missing(load_items(args.data_root))
    if args.limit:
        items = items[: args.limit]
    items = items[args.shard_idx :: args.num_shards]

    img_root = os.path.join(args.out, "images")
    os.makedirs(img_root, exist_ok=True)
    suffix = f"_shard{args.shard_idx}" if args.num_shards > 1 else ""
    manifest_path = os.path.join(args.out, f"gen_manifest{suffix}.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    print(f"[gen][internvlu] shard {args.shard_idx}/{args.num_shards}: {len(items)} items", flush=True)
    print("[gen][internvlu] loading pipeline...", flush=True)
    pipe = InternVLUPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    done, fail = 0, 0
    for idx, it in enumerate(items):
        key = item_key(it)
        out_png = os.path.join(img_root, it.gen_filename)
        if os.path.exists(out_png):
            manifest[key] = os.path.abspath(out_png)
            done += 1
            continue
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        try:
            src = Image.open(it.input_path).convert("RGB")
            w, h = src.size
            if w * h > MAX_PIXELS:
                scale = (MAX_PIXELS / (w * h)) ** 0.5
                src = src.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            with torch.no_grad():
                result = pipe(
                    prompt=it.instruction,
                    image=src,
                    generation_mode="image",
                    height=src.size[1],
                    width=src.size[0],
                    generator=torch.Generator(device=pipe.device).manual_seed(args.seed),
                )
            out_img = result.images[0]
            out_img.save(out_png)
            manifest[key] = os.path.abspath(out_png)
            done += 1
        except Exception as e:
            fail += 1
            print(f"[gen][internvlu][FAIL] {key}: {e}", flush=True)
            traceback.print_exc()
        if (idx + 1) % 10 == 0:
            json.dump(manifest, open(manifest_path, "w"), indent=2)
            print(f"[gen][internvlu] {idx + 1}/{len(items)} done={done} fail={fail}", flush=True)

    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"[gen][internvlu] FINISHED done={done} fail={fail} -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
