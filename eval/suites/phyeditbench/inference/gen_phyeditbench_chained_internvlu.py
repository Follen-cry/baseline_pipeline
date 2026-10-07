#!/usr/bin/env python
"""Chained (genuine multi-turn) InternVL-U generation over PhyEditBench trajectories.

**Not upstream's official protocol.** PhyEditBench's own TypeA/B/C step-wise
tasks (see dataset.py) each feed the *ground-truth* intermediate frame as
input, so they're independent single-turn edits even though they trace a
4-state trajectory (see suites/phyeditbench/README.md's Protocol section).
This script instead chains the model's own output: step 1 runs on the real
input image, step 2 runs on the model's own step-1 output (not the GT
intermediate_1), step 3 runs on the model's own step-2 output — a real
multi-turn demonstration of compounding drift, built for
`eval/suites/phyeditbench`'s artifact rather than for benchmark scoring.

Takes an explicit --trajectories json (list of
{primary, sub, id, input_path, steps: [s1, s2, s3]}) rather than dataset.py's
full 1225-item loader, since this is meant for a small hand-picked subset,
not a full-benchmark run.

Writes <out>/chained/<primary>/<sub>/<id>/turn{1,2,3}.png plus a resumable
<out>/chain_manifest.json ({"<primary>/<sub>/<id>": {"turn1": path, "turn2":
path, "turn3": path}}).

Usage (env: internvlu):
  CUDA_VISIBLE_DEVICES=<g> python gen_phyeditbench_chained_internvlu.py \
      --trajectories chain_trajectories.json \
      --out /scratch/local/ssd/junlin/results/PhyEditBench/InternVL-U

Same PKG/MAX_PIXELS/DEFAULT_CKPT as gen_phyeditbench_internvlu.py — see that
script's docstring for the known dependency-gap note.
"""
import argparse
import json
import os
import sys
import traceback

import torch
from PIL import Image

PKG = "/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/InternVL-U"

DEFAULT_CKPT = (
    "/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/"
    "snapshots/f012d760e69712bb47f7d3d09a24280f346cee01"
)

MAX_PIXELS = 1_800_000


def _downscale(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w * h > MAX_PIXELS:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True, help="json list of {primary, sub, id, input_path, steps}")
    ap.add_argument("--model_path", default=DEFAULT_CKPT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, PKG)
    from internvlu import InternVLUPipeline

    trajectories = json.load(open(args.trajectories))

    chain_root = os.path.join(args.out, "chained")
    os.makedirs(chain_root, exist_ok=True)
    manifest_path = os.path.join(args.out, "chain_manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    print(f"[chain][internvlu] {len(trajectories)} trajectories x 3 turns", flush=True)
    print("[chain][internvlu] loading pipeline...", flush=True)
    pipe = InternVLUPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    done, fail = 0, 0
    for idx, traj in enumerate(trajectories):
        key = f"{traj['primary']}/{traj['sub']}/{traj['id']}"
        traj_dir = os.path.join(chain_root, traj["primary"], traj["sub"], traj["id"])
        os.makedirs(traj_dir, exist_ok=True)
        entry = manifest.get(key, {})

        try:
            current = Image.open(traj["input_path"]).convert("RGB")
            for turn_idx, instruction in enumerate(traj["steps"], start=1):
                turn_key = f"turn{turn_idx}"
                out_png = os.path.join(traj_dir, f"{turn_key}.png")
                if turn_key in entry and os.path.exists(out_png):
                    current = Image.open(out_png).convert("RGB")
                    continue
                src = _downscale(current)
                with torch.no_grad():
                    result = pipe(
                        prompt=instruction,
                        image=src,
                        generation_mode="image",
                        height=src.size[1],
                        width=src.size[0],
                        generator=torch.Generator(device=pipe.device).manual_seed(args.seed),
                    )
                current = result.images[0]
                current.save(out_png)
                entry[turn_key] = os.path.abspath(out_png)
            manifest[key] = entry
            done += 1
        except Exception as e:
            fail += 1
            print(f"[chain][internvlu][FAIL] {key}: {e}", flush=True)
            traceback.print_exc()

        json.dump(manifest, open(manifest_path, "w"), indent=2)
        print(f"[chain][internvlu] {idx + 1}/{len(trajectories)} traj done={done} fail={fail}", flush=True)

    print(f"[chain][internvlu] FINISHED done={done} fail={fail} -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
