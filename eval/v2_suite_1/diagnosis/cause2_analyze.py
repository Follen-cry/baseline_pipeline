#!/usr/bin/env python
"""Cause 2 (format mismatch): does a small, identical instruction-editing fine-tune close the gap?

Same fixed 150-item subset (cause2_subset_uids.txt) for 8 systems: base/T0/T2/T3 (original outputs)
and base/T0/T2/T3 _editsft (after run_edit_sft_probe.sh). Primary per-item metric normalised to 0-1
(analyze.PRIMARY), pooled over the subset (and over its copy-sensitive part: no PhyEditBench-normal).

gap_before = T - base; gap_after = T_editsft - base_editsft (both paired over the same items,
cluster bootstrap CI). closure = 1 - gap_after / gap_before (only meaningful when gap_before < 0).
Also the edit magnitude of every system on the subset.
Writes results/diagnosis/cause2_results.csv, cause2_means.csv   (env: internvlu)
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EVAL, "pipeline"))
import analyze as A  # noqa: E402
from gen import area_size  # noqa: E402

OUT = os.path.join(EVAL, "results", "diagnosis")
ORIG = ["base", "T0", "T2", "T3"]
FT = [m + "_editsft" for m in ORIG]


def edit_mad(m, uids):
    import json
    cfg = json.load(open(os.path.join(EVAL, "pipeline", "config.json")))["inference"]
    out = []
    for u in uids:
        r = m.loc[u]
        src = Image.open(r["inputs"][0]).convert("RGB")
        size = area_size(*src.size, cfg["gen_area_side"], cfg["round_to"])
        a = np.asarray(src.resize(size, Image.LANCZOS), np.float32) / 255
        for mdl in ORIG + FT:
            p = os.path.join(EVAL, "outputs", mdl, r["out_rel"])
            if os.path.exists(p):
                o = np.asarray(Image.open(p).convert("RGB").resize(size, Image.LANCZOS), np.float32) / 255
                out.append((u, mdl, float(np.abs(o - a).mean())))
    return pd.DataFrame(out, columns=["uid", "model", "edit_mad"])


def main():
    uids = open(os.path.join(HERE, "cause2_subset_uids.txt")).read().split()
    m = A.manifest()
    old = (A.MODELS, A.REF)
    A.MODELS, A.REF = ORIG + FT, "__none__"
    df, ps = A.item_scores()
    A.MODELS, A.REF = old
    df = df[df.uid.isin(uids)]
    parts = []
    for bench, (metric, lo, hi) in A.PRIMARY.items():
        s = df[(df.metric == metric) & df.uid.isin(m.index[m.bench == bench])]
        w = s.pivot_table(index="uid", columns="model", values="value", aggfunc="first")
        if len(w):
            parts.append(((w.reindex(columns=ORIG + FT) - lo) / (hi - lo)).assign(bench=bench))
    nz = pd.concat(parts)
    nz = nz.dropna(subset=ORIG + FT)
    meta = m.loc[nz.index]
    em = edit_mad(m, list(nz.index))
    rows, means = [], []
    for pool, idx in (("subset", nz.index), ("subset_copy_sensitive", nz.index[nz.bench != "phyeditbench"]),
                      ("subset_A", nz.index[meta.group == "A"]), ("subset_C", nz.index[meta.group == "C"])):
        sub = nz.loc[idx]
        for mdl in ORIG + FT:
            means.append(dict(pool=pool, model=mdl, norm_mean=float(sub[mdl].mean()), n=len(sub),
                              edit_mad=float(em[(em.model == mdl) & em.uid.isin(idx)].edit_mad.mean())))
        for t in ("T0", "T2", "T3"):
            b = A.paired(sub[t] - sub["base"], meta.loc[idx, "cluster"])
            a_ = A.paired(sub[t + "_editsft"] - sub["base_editsft"], meta.loc[idx, "cluster"])
            diff = A.paired((sub[t + "_editsft"] - sub["base_editsft"]) - (sub[t] - sub["base"]), meta.loc[idx, "cluster"])
            rows.append(dict(pool=pool, model=t, n=len(sub),
                             gap_before=b["delta"], gap_before_lo=b["ci_lo"], gap_before_hi=b["ci_hi"], p_before=b["p"],
                             gap_after=a_["delta"], gap_after_lo=a_["ci_lo"], gap_after_hi=a_["ci_hi"], p_after=a_["p"],
                             change=diff["delta"], change_lo=diff["ci_lo"], change_hi=diff["ci_hi"], p_change=diff["p"],
                             closure=(1 - a_["delta"] / b["delta"]) if b["delta"] < 0 else np.nan))
    # what the fine-tune did to each model on its own, and whether any T+FT reaches the ORIGINAL base
    eff = []
    for pool, idx in (("subset", nz.index), ("subset_copy_sensitive", nz.index[nz.bench != "phyeditbench"])):
        sub = nz.loc[idx]
        for mdl in ORIG:
            r = A.paired(sub[mdl + "_editsft"] - sub[mdl], meta.loc[idx, "cluster"])
            eff.append(dict(pool=pool, comparison=f"{mdl}+FT - {mdl}", **{k: r[k] for k in ("delta", "ci_lo", "ci_hi", "p")}))
        for t in ("T0", "T2", "T3"):
            r = A.paired(sub[t + "_editsft"] - sub["base"], meta.loc[idx, "cluster"])
            eff.append(dict(pool=pool, comparison=f"{t}+FT - base (original)", **{k: r[k] for k in ("delta", "ci_lo", "ci_hi", "p")}))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(eff).to_csv(os.path.join(OUT, "cause2_ft_effect.csv"), index=False)
    print(pd.DataFrame(eff).round(3).to_string())
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cause2_results.csv"), index=False)
    pd.DataFrame(means).to_csv(os.path.join(OUT, "cause2_means.csv"), index=False)
    ps[ps.model.isin(FT)].to_csv(os.path.join(OUT, "cause2_parse_stats.csv"), index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(means).pivot(index="pool", columns="model", values="norm_mean")[ORIG + FT].round(3))
    print(pd.DataFrame(means).pivot(index="pool", columns="model", values="edit_mad")[ORIG + FT].round(3))
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
