#!/usr/bin/env python
"""Per (datapoint, size, condition) similarity of the generated frame to GT F3 and to the last input F2 -> outputs/metrics.csv.

  python metrics.py          (env: internvlu; CPU is fine, ~360 images)
Columns: mad_gt / mad_f2   mean absolute pixel difference (0-255), output resized to the reference (lower = closer)
         clip_gt / clip_f2 cosine similarity of CLIP ViT-L/14-336 image embeddings (higher = same scene/content)
CLIP is a content/scene signal, MAD a layout/color signal. Neither measures whether the *physics* changed correctly.
"""
import csv
import os

import numpy as np
import torch
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from common import CONDS, SIZES, datapoints, path, size_tag

OUT = path("outputs_root")
CLIP = "/scratch/network/ssd2/junlin/huggingface/hub/models--openai--clip-vit-large-patch14-336"
snap = os.path.join(CLIP, "snapshots", sorted(os.listdir(os.path.join(CLIP, "snapshots")))[0])
model = CLIPVisionModelWithProjection.from_pretrained(snap).eval()
MEAN, STD = torch.tensor([0.4815, 0.4578, 0.4082]).view(1, 3, 1, 1), torch.tensor([0.2686, 0.2613, 0.2758]).view(1, 3, 1, 1)


@torch.no_grad()
def emb(im):
    # squash-resize to 336x336 (no center crop, which the stock CLIP processor would apply) + CLIP normalisation
    a = torch.from_numpy(np.asarray(im.convert("RGB").resize((336, 336), Image.BICUBIC), dtype="f4") / 255.0).permute(2, 0, 1)[None]
    e = model(pixel_values=(a - MEAN) / STD).image_embeds
    return torch.nn.functional.normalize(e, dim=-1)[0]


def mad(a, b):
    b = b.resize(a.size, Image.LANCZOS)
    return float(np.abs(np.asarray(a.convert("RGB"), dtype="f4") - np.asarray(b.convert("RGB"), dtype="f4")).mean())


rows = []
for d in datapoints():
    gt, f2 = Image.open(d["target_image"]), Image.open(d["frames"][2])
    e_gt, e_f2 = emb(gt), emb(f2)
    for s in SIZES:
        for c in CONDS:
            im = Image.open(os.path.join(OUT, c + size_tag(s), d["id"] + ".png"))
            e = emb(im)
            rows.append({"id": d["id"], "category": d["category"], "size": s, "condition": c,
                         "mad_gt": round(mad(im, gt), 2), "mad_f2": round(mad(im, f2), 2),
                         "clip_gt": round(float(e @ e_gt), 4), "clip_f2": round(float(e @ e_f2), 4)})
    # reference: how similar are F2 and GT themselves (the "just copy the last frame" ceiling)
    rows.append({"id": d["id"], "category": d["category"], "size": "-", "condition": "copy_F2",
                 "mad_gt": round(mad(f2, gt), 2), "mad_f2": 0.0, "clip_gt": round(float(e_f2 @ e_gt), 4), "clip_f2": 1.0})
with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote", os.path.join(OUT, "metrics.csv"), len(rows), "rows")
for s in SIZES + ["-"]:
    for c in CONDS + ["copy_F2"]:
        r = [x for x in rows if x["size"] == s and x["condition"] == c]
        if r:
            print(f"{s:8s} {c:8s} n={len(r)}  MAD gt {np.mean([x['mad_gt'] for x in r]):5.1f} f2 {np.mean([x['mad_f2'] for x in r]):5.1f}"
                  f" | CLIP gt {np.mean([x['clip_gt'] for x in r]):.3f} f2 {np.mean([x['clip_f2'] for x in r]):.3f}")
