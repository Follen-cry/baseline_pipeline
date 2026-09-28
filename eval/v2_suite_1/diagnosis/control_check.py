#!/usr/bin/env python
"""Control check for the Cause-2 probe (asked after the first write-up):
1. null fine-tune (base through the identical train/save/merge path with lr 0, 2 steps;
   run_edit_sft_probe.sh with LR=0 GEN_DECODER_LR=0 EXTRA="--max_steps 2") vs original base:
   pixel comparison of the 150 subset outputs (weight comparison: weight_diff.py -> weight_diff.csv).
2. what the real 160-step fine-tune did to base, per benchmark and in edit size.
Writes results/diagnosis/control_null_pixels.csv, control_base_ft.csv   (env: internvlu)"""
import csv, json, os, sys
import numpy as np
import pandas as pd
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); EVAL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EVAL, "pipeline"))
import analyze as A  # noqa: E402
from gen import area_size  # noqa: E402
OUT = os.path.join(EVAL, "results", "diagnosis")
U = open(os.path.join(HERE, "cause2_subset_uids.txt")).read().split()
m = A.manifest()
diff, mads = [], {"base": [], "base_editsft": []}
for u in U:
    r = m.loc[u]
    a = np.asarray(Image.open(os.path.join(EVAL, "outputs", "base", r["out_rel"])).convert("RGB"), np.int16)
    b = np.asarray(Image.open(os.path.join(EVAL, "outputs", "base_nullft", r["out_rel"])).convert("RGB"), np.int16)
    diff.append(float(np.abs(a - b).mean()) / 255)
    src = Image.open(r["inputs"][0]).convert("RGB")
    sz = area_size(*src.size, 512, 16)
    x = np.asarray(src.resize(sz, Image.LANCZOS), np.float32) / 255
    for k in mads:
        o = np.asarray(Image.open(os.path.join(EVAL, "outputs", k, r["out_rel"])).convert("RGB").resize(sz), np.float32) / 255
        mads[k].append(float(np.abs(o - x).mean()))
d = np.array(diff)
pd.DataFrame([dict(n=len(d), identical=int((d == 0).sum()), mean_pixel_diff=float(d.mean()), max_pixel_diff=float(d.max()))]) \
    .to_csv(os.path.join(OUT, "control_null_pixels.csv"), index=False)
A.MODELS, A.REF = ["base", "base_editsft"], "__none__"
df, _ = A.item_scores()
df = df[df.uid.isin(U)]
rows = []
for b, (met, lo, hi) in A.PRIMARY.items():
    s = df[(df.metric == met) & df.uid.isin(m.index[m.bench == b])]
    if len(s):
        w = ((s.pivot_table(index="uid", columns="model", values="value") - lo) / (hi - lo)).dropna()
        rows.append(dict(item=f"{b} (normalised)", base=w["base"].mean(), base_ft=w["base_editsft"].mean(), n=len(w)))
for met, lab in (("consistency", "PhyEditBench consistency (1-10)"), ("instruction_following", "PhyEditBench instruction following (1-10)"),
                 ("physical_plausibility", "PhyEditBench physical plausibility (1-10)"), ("Reasoning", "RISE reasoning (1-5)"),
                 ("ApprConsistency", "RISE consistency (1-5)")):
    w = df[df.metric == met].pivot_table(index="uid", columns="model", values="value").dropna()
    rows.append(dict(item=lab, base=w["base"].mean(), base_ft=w["base_editsft"].mean(), n=len(w)))
for k, lab in (("no_edit", "no-edit outputs (edit size < 0.03)"), ("median", "median edit size (mean |out - in|)")):
    f = (lambda v: float((np.array(v) < 0.03).mean())) if k == "no_edit" else (lambda v: float(np.median(v)))
    rows.append(dict(item=lab, base=f(mads["base"]), base_ft=f(mads["base_editsft"]), n=len(U)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "control_base_ft.csv"), index=False)
print(pd.read_csv(os.path.join(OUT, "control_null_pixels.csv")).to_string()); print(pd.DataFrame(rows).round(3).to_string())
