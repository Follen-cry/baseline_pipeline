#!/usr/bin/env python
"""Cause-1 probe: degenerate 'models' written into outputs/ so the unchanged judge harness scores them.

  random  an unrelated real image: the input of a task from a *different* benchmark (fixed seed),
          resized to the generation size -> should score near the floor on any valid judge
  gt      the ground-truth target where the benchmark has one (PhyEditBench next/final state,
          MagicBrush GT, RISEBench reference_img) -> should score near the ceiling
(copy = input unchanged already exists: pipeline/make_copy_reference.py)

Also writes results/rule/editmag_ref.csv with the same edit-magnitude metric for random/gt.
(env: internvlu)
"""
import csv
import json
import os
import random
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EVAL, "pipeline"))
from gen import area_size  # noqa: E402

cfg = json.load(open(os.path.join(EVAL, "pipeline", "config.json")))["inference"]
rows = [json.loads(l) for l in open(os.path.join(EVAL, "pipeline", "manifest.jsonl"))]
rng = random.Random("20260928:random-baseline")


def gt_path(r):
    j = r["judge"]
    if r["bench"] == "phyeditbench":
        return j["ref"]
    if r["bench"] == "magicbrush":
        return j["gt"]
    if r["bench"] == "risebench" and j.get("reference_img"):
        return j["reference_img"]
    return None


def put(model, r, img, size):
    p = os.path.join(EVAL, "outputs", model, r["out_rel"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    img.convert("RGB").resize(size, Image.LANCZOS).save(p)
    return p


out = []
for r in rows:
    src = Image.open(r["inputs"][0]).convert("RGB")
    size = area_size(*src.size, cfg["gen_area_side"], cfg["round_to"])
    a = np.asarray(src.resize(size, Image.LANCZOS), np.float32) / 255
    other = rng.choice([x for x in rows if x["bench"] != r["bench"]])
    for model, img in (("random", Image.open(other["inputs"][0])),
                       ("gt", Image.open(gt_path(r)) if gt_path(r) else None)):
        if img is None:
            continue
        p = put(model, r, img, size)
        o = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
        out.append((r["bench"], r["uid"], model, "edit_mad", float(np.abs(o - a).mean())))
with open(os.path.join(EVAL, "results", "rule", "editmag_ref.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["bench", "uid", "model", "metric", "value"])
    w.writerows(out)
print({m: sum(1 for x in out if x[2] == m) for m in ("random", "gt")})
