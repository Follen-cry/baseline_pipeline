#!/usr/bin/env python
"""Cause 1 (judge unreliability): degenerate baselines + a diagnostic "no edit made" penalty.

Reads the unchanged judge results (results/scores_item.csv, which analyze.py writes, plus the
judge_raw rows of the random / gt baselines) and the edit-magnitude CSVs. Nothing here changes the
judge or the main results; it only re-scores in memory.

1. Degenerate-baseline table: per benchmark (and PhyEditBench type), primary metric normalised to
   0-1 for base, T0/T2/T3, copy, random, gt. `copy_gap` = base - copy: > 0 means the judge
   penalises doing nothing on that benchmark; `random_gap` = base - random.
2. No-edit penalty: an output whose mean |output - input| (0-1 RGB) is below TAU is floored to the
   benchmark's minimum on every judged metric (PhyEditBench/anti dims -> 1, PICABench acc -> 0,
   RISE -> Reasoning 1 & complete 0 & score 1, ImgEdit/UGE/MagicBrush VLM -> 1). Same rule for every
   model. Re-computes the group-level paired deltas and the 4-model ranking for TAU in
   {0.02, 0.03, 0.05}.
Writes results/diagnosis/cause1_baselines.csv, cause1_penalty.csv, cause1_penalty_rates.csv
(env: internvlu)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EVAL, "pipeline"))
import analyze as A  # noqa: E402

OUT = os.path.join(EVAL, "results", "diagnosis")
MODELS = ["base", "T0", "T2", "T3"]
REFS = ["copy", "random", "gt"]
TAUS = [0.02, 0.03, 0.05]
FLOOR = {"phyeditbench": {"overall": 1, "consistency": 1, "instruction_following": 1, "physical_plausibility": 1,
                          "image_quality": 1},
         "phyedit_anti": {"overall": 1, "Instruction_Following": 1, "Physical_Plausibility": 1, "Consistency": 1,
                          "Image_Quality": 1},
         "picabench": {"acc": 0},
         "risebench": {"score": 1, "complete": 0, "Reasoning": 1},
         "imgedit_basic": {"score": 1}, "imgedit_uge": {"score": 1}, "magicbrush": {"vlm_score": 1}}


def load_items():
    m = A.manifest()
    main, _ = A.item_scores()  # base, T0-T3, copy (+ rule metrics)
    # random / gt were judged under the same judge name; item_scores only loops MODELS + [REF]
    old = A.MODELS
    A.MODELS = ["random", "gt"]
    A.REF = "__none__"
    extra, _ = A.item_scores()
    A.MODELS, A.REF = old, "copy"
    extra = extra[extra.model.isin(["random", "gt"]) & ~extra.metric.isin(["psnr_nonedit", "l1", "clip_i", "dino",
                                                                           "edit_mad"])]
    df = pd.concat([main, extra], ignore_index=True)
    em = pd.concat([pd.read_csv(os.path.join(EVAL, "results", "rule", f)) for f in ("editmag.csv", "editmag_ref.csv")])
    em = em[em.metric == "edit_mad"][["uid", "model", "value"]].rename(columns={"value": "edit_mad"})
    return m, df, em


def norm_wide(df, m, bench, metric, lo, hi, models):
    s = df[(df.metric == metric) & df.uid.isin(m.index[m.bench == bench])]
    w = s.pivot_table(index="uid", columns="model", values="value", aggfunc="first")
    return (w.reindex(columns=models) - lo) / (hi - lo)


def baselines_table(m, df):
    rows = []
    cols = MODELS + REFS
    for bench, (metric, lo, hi) in A.PRIMARY.items():
        w = norm_wide(df, m, bench, metric, lo, hi, cols).dropna(subset=MODELS)
        slices = [("all", w.index)]
        if bench == "phyeditbench":
            slices += [(t, w.index[m.loc[w.index, "etype"] == t]) for t in sorted(m.loc[w.index, "etype"].unique())]
        if bench == "risebench":
            slices += [(c, w.index[m.loc[w.index, "category"] == c]) for c in sorted(m.loc[w.index, "category"].unique())]
        for name, idx in slices:
            sub = w.loc[idx]
            r = dict(bench=bench, slice=name, metric=metric, n=len(sub))
            for c in cols:
                r[c] = float(sub[c].mean()) if sub[c].notna().any() else np.nan
                r[f"n_{c}"] = int(sub[c].notna().sum())
            tm = np.nanmean([r[t] for t in ("T0", "T2", "T3")])
            r["trained_mean"] = tm
            r["copy_gap"] = r["base"] - r["copy"]
            r["random_gap"] = r["base"] - r["random"]
            r["copy_beats_base"] = bool(r["copy"] >= r["base"])
            r["copy_beats_trained"] = bool(r["copy"] >= tm)
            r["random_ge_any_model"] = bool(r["random"] >= min(r[x] for x in MODELS))
            rows.append(r)
    return pd.DataFrame(rows)


def penalised(df, em, tau):
    flags = em.assign(flag=em.edit_mad < tau)
    d = df.merge(flags[["uid", "model", "flag"]], on=["uid", "model"], how="left")
    d.loc[d.model == "copy", "flag"] = True  # copy is the input by construction
    d["flag"] = d["flag"].fillna(False).astype(bool)
    d = d.merge(A.manifest()[["bench"]], left_on="uid", right_index=True)
    for bench, fl in FLOOR.items():
        for metric, v in fl.items():
            sel = d.flag & (d.bench == bench) & (d.metric == metric)
            d.loc[sel, "value"] = v
    return d.drop(columns=["bench"]), flags


def ranking(gs):
    t = gs.set_index("model")["norm_mean"].reindex(MODELS)
    return " > ".join(t.sort_values(ascending=False).index)


def main():
    os.makedirs(OUT, exist_ok=True)
    m, df, em = load_items()
    bl = baselines_table(m, df)
    bl.to_csv(os.path.join(OUT, "cause1_baselines.csv"), index=False)
    rows, rates = [], []
    for tau in [None] + TAUS:
        d, flags = (df, None) if tau is None else penalised(df, em, tau)
        gs, tr = A.group_tables(d[d.model.isin(MODELS + ["copy"])], m)
        for g in ("A", "A_copy_sensitive", "B", "C"):
            s = gs[(gs.group == g) & gs.model.isin(["T0", "T2", "T3"])]
            t = tr[tr.group == g]
            r = dict(tau="none" if tau is None else tau, group=g, ranking=ranking(t))
            for _, x in s.iterrows():
                r[f"{x.model}_delta"], r[f"{x.model}_ci_lo"], r[f"{x.model}_ci_hi"], r[f"{x.model}_p_holm"] = \
                    x.delta, x.ci_lo, x.ci_hi, x.p_holm
            for mm in MODELS:
                r[f"{mm}_mean"] = float(t.set_index("model").norm_mean.get(mm, np.nan))
            rows.append(r)
        if flags is not None:
            f = flags.merge(m[["bench", "group"]], left_on="uid", right_index=True)
            for (bench, model), g in f[f.model.isin(MODELS)].groupby(["bench", "model"]):
                rates.append(dict(tau=tau, bench=bench, model=model, flagged=float(g.flag.mean()), n=len(g)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cause1_penalty.csv"), index=False)
    pd.DataFrame(rates).to_csv(os.path.join(OUT, "cause1_penalty_rates.csv"), index=False)
    pd.set_option("display.width", 250)
    print(bl[["bench", "slice", "n"] + MODELS + REFS + ["copy_gap"]].round(3).to_string())
    print(pd.DataFrame(rows)[["tau", "group", "ranking", "T0_delta", "T2_delta", "T3_delta", "T0_p_holm",
                              "T2_p_holm", "T3_p_holm"]].round(4).to_string())


if __name__ == "__main__":
    main()
