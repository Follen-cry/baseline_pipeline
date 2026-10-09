#!/usr/bin/env python
"""CPU-only: dump what the ViT and the VAE branch actually receive for one datapoint's F2, to check for cropping/resizing.

  python inspect_preproc.py [category]  ->  outputs/preproc_<category>.png   (env: internvlu)
Panels: original F2 | after our resize | ViT input (de-normalised pixel_values[tile]) | VAE-branch input (pixel_values_gen)
"""
import json
import sys

import numpy as np
import torch
from PIL import Image

from common import CFG, datapoints, path
from gen import area_size

sys.path.insert(0, CFG["internvlu_pkg"])
from internvlu.processing_internvlu import InternVLUProcessor
from patches import allow_multi_image_uncond

allow_multi_image_uncond(InternVLUProcessor)
cat = sys.argv[1] if len(sys.argv) > 1 else "tension"
d = [x for x in datapoints() if x["category"] == cat][0]
proc = InternVLUProcessor.from_pretrained(CFG["base_model"] + "/processor")
proc.template_name = json.load(open(CFG["base_model"] + "/vlm/config.json"))["template"]
inf = CFG["inference"]
orig = [Image.open(p).convert("RGB") for p in d["frames"]]
w, h = area_size(*orig[0].size, inf["gen_area_side"], inf["round_to"])
src = [im.resize((w, h), Image.LANCZOS) for im in orig]
inp = proc(prompt=d["prompt"], image=[src], generation_mode="image", padding=True, return_tensors="pt", height=h, width=w)
ipc, gpc = proc.image_processor, proc.image_gen_processor
print("ViT image_processor:", type(ipc).__name__, {k: getattr(ipc, k, None) for k in ("image_size", "min_dynamic_patch", "max_dynamic_patch", "use_thumbnail", "do_center_crop", "crop_size", "do_resize")})
print("VAE gen processor:", type(gpc).__name__, {k: getattr(gpc, k, None) for k in ("min_pixels", "max_pixels", "patch_size", "merge_size", "do_resize")})
print("our resized frame (w,h):", (w, h), " requested gen (h,w):", (h, w))
print("ViT pixel_values:", tuple(inp["pixel_values"].shape), " VAE pixel_values_gen:", tuple(inp["pixel_values_gen"].shape),
      " image_grid_thw (frames 0-2):", inp["image_grid_thw_gen"][:3].tolist())

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
vit = (inp["pixel_values"][2].float() * std + mean).clamp(0, 1)  # tile 2 = F2 of the first row (1 tile per frame)
vit = Image.fromarray((vit.permute(1, 2, 0).numpy() * 255).astype("uint8"))
g = inp["pixel_values_gen"][2].float()
t = inp["image_grid_thw_gen"][2].tolist()
ps = gpc.patch_size; g = g[:, : t[1] * ps, : t[2] * ps]
print("VAE-branch value range:", float(g.min()), float(g.max()), " valid region (h,w) px:", g.shape[1:])
gen = Image.fromarray(((g.clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).numpy() * 255).astype("uint8"))
panels = [("original F2 %dx%d" % orig[2].size, orig[2]), ("our resize %dx%d" % (w, h), src[2]),
          ("ViT input %dx%d" % vit.size, vit), ("VAE-branch input %dx%d" % gen.size, gen)]
H = 300
tiles = [im.resize((round(im.width * H / im.height), H)) for _, im in panels]
sheet = Image.new("RGB", (sum(t.width for t in tiles) + 10 * len(tiles), H + 20), "white")
x = 5
from PIL import ImageDraw
dr = ImageDraw.Draw(sheet)
for (lab, _), t_ in zip(panels, tiles):
    sheet.paste(t_, (x, 20)); dr.text((x, 4), lab, fill="black"); x += t_.width + 10
out = path("outputs_root") + f"/preproc_{cat}.png"
sheet.save(out)
print("wrote", out)
