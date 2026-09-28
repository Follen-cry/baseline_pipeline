#!/usr/bin/env python
"""Run one checkpoint over the suite manifest with the settings in config.json.

Writes outputs/<model>/<out_rel> (skips existing -> resumable) and appends
failures to outputs/<model>/_logs/failures_shard<i>.jsonl.

  CUDA_VISIBLE_DEVICES=0 python gen.py --model T0 --shard_idx 0 --num_shards 4 [--limit N]
(env: internvlu)
"""
import argparse
import json
import os
import sys
import time
import traceback

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "..", "outputs")
PKG = "/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL-U"


def area_size(w, h, side, mult):
    """v2 training rule: keep aspect ratio, area ~ side^2, sides rounded to `mult`."""
    s = (side * side / (w * h)) ** 0.5
    return max(mult, round(w * s / mult) * mult), max(mult, round(h * s / mult) * mult)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.jsonl"))
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_root", default=OUT_ROOT)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    inf = cfg["inference"]

    rows = [json.loads(l) for l in open(a.manifest)]
    if a.limit:
        rows = rows[: a.limit]
    rows = rows[a.shard_idx :: a.num_shards]
    out_dir = os.path.join(a.out_root, a.model)
    log_dir = os.path.join(out_dir, "_logs")
    os.makedirs(log_dir, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(out_dir, r["out_rel"]))]
    print(f"[gen] {a.model} shard {a.shard_idx}/{a.num_shards}: {len(rows)} rows, {len(todo)} to do", flush=True)
    if not todo:
        return

    sys.path.insert(0, PKG)
    from internvlu import InternVLUPipeline
    pipe = InternVLUPipeline.from_pretrained(cfg["models"][a.model], torch_dtype=torch.bfloat16).to("cuda")

    fail_log = open(os.path.join(log_dir, f"failures_shard{a.shard_idx}.jsonl"), "a")
    t0, n = time.time(), 0
    for r in todo:
        out = os.path.join(out_dir, r["out_rel"])
        try:
            src = Image.open(r["inputs"][0]).convert("RGB")
            w, h = area_size(*src.size, inf["gen_area_side"], inf["round_to"])
            src = src.resize((w, h), Image.LANCZOS)
            with torch.no_grad():
                img = pipe(prompt=r["instruction"], image=src, generation_mode="image", height=h, width=w,
                           num_inference_steps=inf["num_inference_steps"], all_cfg_scale=inf["all_cfg_scale"],
                           part_cfg_scale=inf["part_cfg_scale"],
                           generator=torch.Generator(device="cuda").manual_seed(inf["seed"])).images[0]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            img.save(out + ".tmp.png")
            os.replace(out + ".tmp.png", out)  # atomic: a killed job never leaves a half-written PNG
            n += 1
        except Exception as e:
            traceback.print_exc()
            fail_log.write(json.dumps(dict(uid=r["uid"], error=repr(e), time=time.time())) + "\n")
            fail_log.flush()
        if n and n % 10 == 0:
            print(f"[gen] {a.model} s{a.shard_idx} {n}/{len(todo)} {(time.time() - t0) / n:.1f}s/it", flush=True)
    print(f"[gen] {a.model} s{a.shard_idx} DONE {n}/{len(todo)} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
