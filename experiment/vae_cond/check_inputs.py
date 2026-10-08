#!/usr/bin/env python
"""CPU-only wiring check (no model weights, no GPU): run the real processor on datapoint 0 and the real pipeline mask
logic, then confirm for each condition that the ViT still gets all 3 frames and the VAE side keeps 0 / 1 / 3.

  python check_inputs.py        (env: internvlu)
"""
import json
import sys
import types

import torch
from PIL import Image

from common import CFG, datapoints
from gen import KEEP, area_size, restrict_vae_cond
from patches import allow_multi_image_uncond

sys.path.insert(0, CFG["internvlu_pkg"])
from internvlu import InternVLUPipeline
from internvlu.processing_internvlu import InternVLUProcessor

allow_multi_image_uncond(InternVLUProcessor)
inf = CFG["inference"]
d = datapoints()[0]
proc = InternVLUProcessor.from_pretrained(CFG["base_model"] + "/processor")
proc.template_name = json.load(open(CFG["base_model"] + "/vlm/config.json"))["template"]  # the pipeline does this in __init__
src = [Image.open(p).convert("RGB") for p in d["frames"]]
w, h = area_size(*src[0].size, inf["gen_area_side"], inf["round_to"])
src = [im.resize((w, h), Image.LANCZOS) for im in src]
inp = proc(prompt=d["prompt"], image=[src], generation_mode="image", padding=True, return_tensors="pt", height=h, width=w)
ids = inp["input_ids"]
img_ctx, img_start = proc.tokenizer.convert_tokens_to_ids(["<IMG_CONTEXT>", "<img>"])
print("rows (none/text/all drop):", ids.shape[0], " generation_flags:", inp["generation_flags"].tolist())
print("ViT pixel_values:", tuple(inp["pixel_values"].shape), " VAE pixel_values_gen:", tuple(inp["pixel_values_gen"].shape))
print("per-row <IMG_CONTEXT> tokens:", (ids == img_ctx).sum(1).tolist(), " per-row <img> starts:", (ids == img_start).sum(1).tolist())

# real mask logic from the pipeline, with a stand-in `self` that only supplies the <img> token id
fake = types.SimpleNamespace(vlm=types.SimpleNamespace(img_start_token_id=img_start))
gen_m, in_m, _ = InternVLUPipeline._prepare_image_hidden_state_mask(
    fake, input_ids=ids, attention_mask=inp["attention_mask"], generation_flags=inp["generation_flags"])
cond = [inp["pixel_values_gen"][m] for m in in_m]
grid = [inp["image_grid_thw_gen"][m] for m in gen_m]
print("stock VAE cond images per row:", [int(c.shape[0]) for c in cond], " grids per row:", [tuple(g.shape) for g in grid])
for name, keep in KEEP.items():
    rec = {}
    out = restrict_vae_cond({"conditional_image": list(cond), "image_grid_thw_gen_cond": list(grid)}, keep, rec)
    assert all(c.shape[0] == g.shape[0] for c, g in zip(out["conditional_image"], out["image_grid_thw_gen_cond"]))
    n = sum(int(c.shape[0]) for c in out["conditional_image"])
    print(f"  {name:5s}: VAE cond per row {rec['vae_cond_after']}  (total {n}; 0 => decoder conditional_input=None)")
print("ViT side identical across conditions: pixel_values is never touched.")
