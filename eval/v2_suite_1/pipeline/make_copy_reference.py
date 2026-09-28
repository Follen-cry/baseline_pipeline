#!/usr/bin/env python
"""Copy-input reference 'model': outputs/copy/<out_rel> = the input image resized exactly as gen.py
resizes the conditioning image (area ~512^2, sides x16). Judged like a model, never counted as one:
it calibrates how much each judge rewards doing nothing. Also writes results/rule/editmag.csv:
per item and model, mean absolute pixel change output vs (resized) input, in [0,1]  (env: internvlu)."""
import csv, json, os
import numpy as np
from PIL import Image
from gen import area_size
HERE = os.path.dirname(os.path.abspath(__file__)); EVAL = os.path.dirname(HERE)
cfg = json.load(open(os.path.join(HERE, "config.json")))["inference"]
rows = [json.loads(l) for l in open(os.path.join(HERE, "manifest.jsonl"))]
out = []
for r in rows:
    src = Image.open(r["inputs"][0]).convert("RGB")
    w, h = area_size(*src.size, cfg["gen_area_side"], cfg["round_to"])
    src = src.resize((w, h), Image.LANCZOS)
    p = os.path.join(EVAL, "outputs", "copy", r["out_rel"])
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True); src.save(p)
    a = np.asarray(src, np.float32) / 255
    for m in ["base", "T0", "T2", "T3"]:
        o = Image.open(os.path.join(EVAL, "outputs", m, r["out_rel"])).convert("RGB").resize((w, h), Image.LANCZOS)
        out.append((r["bench"], r["uid"], m, "edit_mad", float(np.abs(np.asarray(o, np.float32) / 255 - a).mean())))
with open(os.path.join(EVAL, "results", "rule", "editmag.csv"), "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["bench", "uid", "model", "metric", "value"]); wr.writerows(out)
print(len(out), "editmag rows")
