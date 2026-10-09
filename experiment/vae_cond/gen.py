#!/usr/bin/env python
"""Generate F3 from F0-F2 with the BASE InternVL-U under three VAE-condition settings (resumable).

  CUDA_VISIBLE_DEVICES=0 python gen.py [--conditions none first all] [--limit N]      (env: internvlu)

All 3 frames are always passed to the pipeline as `image=[F0, F1, F2]`, so the ViT always sees all three
(and the prompt keeps its three `<image>` slots). Only the VAE side is changed. In the stock pipeline every
image before the target is VAE-encoded (`_prepare_diffusion_inputs` -> `conditional_image`); we wrap that
method and drop entries from the *VAE-only* tensors `conditional_image` and `image_grid_thw_gen_cond`
(kept index-aligned). The ViT-side tensors (`pixel_values`, `encoder_image_token_mask`, `image_fhw_cond`) are
untouched.

  none  -> keep 0 VAE cond images  (decoder gets conditional_input=None)
  first -> keep F0 only
  all   -> keep F0, F1, F2         (stock pipeline behaviour)

Same noise seed for every condition, so differences come from the conditioning alone.
Writes OUTPUTS/<cond>/<dp_id>.png and OUTPUTS/_meta/<cond>__<dp_id>.json (what was actually fed to ViT / VAE).
"""
import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

from common import CFG, datapoints, path

KEEP = {"none": 0, "first": 1, "all": 3}  # number of leading frames kept on the VAE side


def area_size(w, h, side, mult):
    """v2 training/eval rule (same as v2_suite_1_sub/scripts/gen.py): keep aspect, area ~ side^2, sides rounded to `mult`."""
    s = (side * side / (w * h)) ** 0.5
    return max(mult, round(w * s / mult) * mult), max(mult, round(h * s / mult) * mult)


def restrict_vae_cond(inputs, keep, record):
    """Subset the VAE-side conditioning of `_prepare_diffusion_inputs`' output to the first `keep` frames per row."""
    cond, grid = inputs["conditional_image"], inputs["image_grid_thw_gen_cond"]
    if cond is None:  # no images at all (never the case here)
        return inputs
    assert len(cond) == len(grid)
    record["vae_cond_before"] = [int(c.shape[0]) for c in cond]
    inputs["conditional_image"] = [c[:keep] for c in cond]
    inputs["image_grid_thw_gen_cond"] = [g[:keep] for g in grid]
    record["vae_cond_after"] = [int(c.shape[0]) for c in inputs["conditional_image"]]
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(KEEP), choices=list(KEEP))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model_path", default=CFG["base_model"])
    ap.add_argument("--size", default="", help="WxH override for frames AND output (default: area-512 rule), e.g. 688x400")
    ap.add_argument("--tag", default="", help="output subfolder suffix, e.g. _688x400 -> outputs/<cond>_688x400/")
    ap.add_argument("--categories", nargs="*", default=[], help="only these datapoint categories")
    a = ap.parse_args()
    inf = CFG["inference"]
    out_root = path("outputs_root")
    dps = [d for d in datapoints() if not a.categories or d["category"] in a.categories][: a.limit or None]
    todo = [(c, d) for d in dps for c in a.conditions if not os.path.exists(os.path.join(out_root, c + a.tag, d["id"] + ".png"))]
    print(f"[gen] {len(dps)} datapoints x {a.conditions}: {len(todo)} to do", flush=True)
    if not todo:
        return

    sys.path.insert(0, CFG["internvlu_pkg"])
    from internvlu import InternVLUPipeline
    from internvlu.processing_internvlu import InternVLUProcessor
    from internvlu.diffusion.internvlu_transformer import InternVLUTransformer2DModel
    from patches import allow_multi_image_uncond, fix_multi_cond_token_slice
    allow_multi_image_uncond(InternVLUProcessor)  # stock processor asserts on >1 image in the uncond CFG row
    fix_multi_cond_token_slice(InternVLUTransformer2DModel)  # stock decoder crashes on >1 VAE cond image (`all`)
    pipe = InternVLUPipeline.from_pretrained(a.model_path, torch_dtype=torch.bfloat16).to("cuda")

    state, cur = {}, {}  # state: filled by the wrapper per call; cur["keep"]: set per generation
    orig = pipe._prepare_diffusion_inputs

    def wrapped(**kw):
        state.clear()
        state["vit_pixel_values"] = list(kw["pixel_values"].shape)  # (tiles, 3, 448, 448): ViT input, never modified
        return restrict_vae_cond(orig(**kw), cur["keep"], state)

    pipe._prepare_diffusion_inputs = wrapped

    os.makedirs(os.path.join(out_root, "_meta"), exist_ok=True)
    t0 = time.time()
    for n, (cond, d) in enumerate(todo, 1):
        cur["keep"] = KEEP[cond]
        src = [Image.open(p).convert("RGB") for p in d["frames"]]
        w, h = tuple(map(int, a.size.split("x"))) if a.size else area_size(*src[0].size, inf["gen_area_side"], inf["round_to"])
        src = [im.resize((w, h), Image.LANCZOS) for im in src]  # F0, F1, F2 chronological; all go to the ViT
        with torch.no_grad():
            # image=[src]: one entry per prompt, each entry = this prompt's 3 frames (a bare list of 3 would be read as 3 prompts)
            img = pipe(prompt=d["prompt"], image=[src], generation_mode=inf["generation_mode"], height=h, width=w,
                       num_inference_steps=inf["num_inference_steps"], all_cfg_scale=inf["all_cfg_scale"],
                       part_cfg_scale=inf["part_cfg_scale"],
                       generator=torch.Generator(device="cuda").manual_seed(inf["seed"])).images[0]
        out = os.path.join(out_root, cond + a.tag, d["id"] + ".png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        img.save(out + ".tmp.png")
        os.replace(out + ".tmp.png", out)  # atomic: a killed job never leaves a half-written PNG
        meta = {"id": d["id"], "condition": cond, "n_frames_to_vit": len(src), "gen_size_wh": [w, h],
                **state}
        json.dump(meta, open(os.path.join(out_root, "_meta", f"{cond}{a.tag}__{d['id']}.json"), "w"))
        print(f"[gen] {n}/{len(todo)} {cond:5s} {d['id']}  vae_cond(before->after)={meta.get('vae_cond_before')}->"
              f"{meta.get('vae_cond_after')}  {(time.time() - t0) / n:.1f}s/it", flush=True)
    print(f"[gen] DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
