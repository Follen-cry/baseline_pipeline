#!/usr/bin/env python
"""Edit magnitude of the Cause-2 probe data itself: mean |target - source| (0-1 RGB, both at 512x512)
on a fixed random 400-row sample of edit_sft/train.jsonl, per source.
Writes results/diagnosis/cause2_ftdata_editmag.csv   (env: internvlu)"""
import csv, json, os, random
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open("/scratch/network/ssd/junlin/ssl_eval/diag_sft/data/train.jsonl")]
random.Random(0).shuffle(R)
vals = {"magicbrush_train": [], "pica100k": []}
for r in R[:400]:
    s = Image.open(r["image"][0]).convert("RGB")
    t = Image.open(r["target_image"]).convert("RGB").resize(s.size)
    s, t = [np.asarray(x.resize((512, 512)), np.float32) / 255 for x in (s, t)]
    vals["magicbrush_train" if r["id"].startswith("mb_") else "pica100k"].append(float(np.abs(t - s).mean()))
with open(os.path.join(HERE, "..", "results", "diagnosis", "cause2_ftdata_editmag.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "n", "mean", "median", "frac_below_0.03"])
    for k, v in vals.items():
        v = np.array(v)
        w.writerow([k, len(v), round(float(v.mean()), 4), round(float(np.median(v)), 4), round(float((v < 0.03).mean()), 3)])
