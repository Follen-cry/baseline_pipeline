#!/usr/bin/env python
"""Pick the visual examples for the Diagnosis section (results/diagnosis/diag_examples.json).
Image paths only; results/build_artifact.py embeds them. Every score shown comes from the CSVs.

cause1  PhyEditBench items where the unedited copy scores at or above every model (largest copy - base
        first), and items where an unrelated random image still gets >= 0.5 normalised.
cause3  per VLM failure mode (no_edit, partial_edit, instruction_ignored): loss-set items where a
        trained model got that primary label and base was tagged correct; with the VLM's reason.
cause2  subset items with the largest and smallest change in (T - base) after the edit fine-tune.
(env: internvlu)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EVAL, "pipeline"))
import analyze as A  # noqa: E402

RES = os.path.join(EVAL, "results")
DIAG = os.path.join(RES, "diagnosis")
MODELS = ["base", "T0", "T2", "T3"]


def gt_path(r):
    j = r["judge"]
    return j.get("ref") or j.get("gt") or j.get("reference_img")


def out(mdl, r):
    return os.path.join(EVAL, "outputs", mdl, r["out_rel"])


def main():
    man = {json.loads(l)["uid"]: json.loads(l) for l in open(os.path.join(EVAL, "pipeline", "manifest.jsonl"))}
    m = A.manifest()
    old = (A.MODELS, A.REF)
    A.MODELS, A.REF = MODELS + ["copy", "random", "gt"], "__none__"
    df, _ = A.item_scores()
    A.MODELS, A.REF = old
    ex = {"cause1": [], "cause3": [], "cause2": []}

    # ---- cause 1
    s = df[(df.metric == "overall") & df.uid.isin(m.index[m.bench == "phyeditbench"])]
    w = s.pivot_table(index="uid", columns="model", values="value").dropna(subset=MODELS + ["copy", "random"])
    beat = w[w["copy"] >= w[MODELS].max(axis=1)].assign(gap=lambda x: x["copy"] - x["base"]).sort_values(["gap"], ascending=False)
    rnd = w[(w["random"] - 1) / 9 >= 0.5].sort_values("random", ascending=False)
    for kind, sel in (("copy_wins", beat.head(4)), ("random_scores_high", rnd.head(2))):
        for uid, row in sel.iterrows():
            r = man[uid]
            cols = ["input"] + MODELS + ["copy", "random"] + (["gt"] if gt_path(r) else [])
            ex["cause1"].append(dict(kind=kind, uid=uid, bench=r["bench"], etype=r["etype"], instruction=r["instruction"],
                                     imgs={c: (r["inputs"][0] if c == "input" else (gt_path(r) if c == "gt" else out(c, r))) for c in cols},
                                     scores={c: round(float(row[c]), 2) for c in cols if c != "input" and c in row and not pd.isna(row[c])},
                                     metric="PhyEditBench overall (1-10)"))

    # ---- cause 3
    v = pd.DataFrame([json.loads(l) for l in open(os.path.join(DIAG, "cause3_vlm.jsonl"))]).drop_duplicates(["uid", "model"], keep="last")
    base_ok = set(v[(v.model == "base") & (v.primary == "correct")].uid)
    exs = pd.read_csv(os.path.join(RES, "examples.csv")).set_index("uid")
    for mode in ("no_edit", "partial_edit", "instruction_ignored"):
        c = v[(v.model != "base") & (v.primary == mode) & v.uid.isin(base_ok)].copy()
        c["d"] = [exs.loc[u, f"norm_{t}"] - exs.loc[u, "norm_base"] for u, t in zip(c.uid, c.model)]
        c = c.sort_values(["d", "uid"]).drop_duplicates("uid").head(3)
        for _, x in c.iterrows():
            r = man[x.uid]
            cols = ["input", "base", x.model] + (["gt"] if gt_path(r) else [])
            ex["cause3"].append(dict(kind=mode, uid=x.uid, bench=r["bench"], model=x.model, instruction=r["instruction"],
                                     reason=x.reason, base_reason=v[(v.uid == x.uid) & (v.model == "base")].reason.iloc[0],
                                     imgs={cc: (r["inputs"][0] if cc == "input" else (gt_path(r) if cc == "gt" else out(cc, r))) for cc in cols},
                                     scores={cc: round(float(exs.loc[x.uid, f"norm_{cc}"]), 3) for cc in ("base", x.model)},
                                     metric="normalised primary metric (0-1)"))

    # ---- cause 2 (after the fine-tune outputs exist)
    p = os.path.join(DIAG, "cause2_means.csv")
    if os.path.exists(p):
        A.MODELS, A.REF = MODELS + [x + "_editsft" for x in MODELS], "__none__"
        d2, _ = A.item_scores()
        A.MODELS, A.REF = old
        uids = open(os.path.join(HERE, "cause2_subset_uids.txt")).read().split()
        parts = []
        for bench, (metric, lo, hi) in A.PRIMARY.items():
            s = d2[(d2.metric == metric) & d2.uid.isin(m.index[m.bench == bench]) & d2.uid.isin(uids)]
            parts.append((s.pivot_table(index="uid", columns="model", values="value") - lo) / (hi - lo))
        nz = pd.concat(parts).dropna()
        ft = [x + "_editsft" for x in MODELS]
        chg = (nz[[t + "_editsft" for t in ("T0", "T2", "T3")]].mean(1) - nz["base_editsft"]) - (nz[["T0", "T2", "T3"]].mean(1) - nz["base"])
        for kind, sel in (("gap_closes", chg.sort_values(ascending=False).head(3)), ("gap_widens", chg.sort_values().head(2))):
            for uid, val in sel.items():
                r = man[uid]
                cols = ["input"] + MODELS + ft
                ex["cause2"].append(dict(kind=kind, uid=uid, bench=r["bench"], instruction=r["instruction"],
                                         change=round(float(val), 3),
                                         imgs={c: (r["inputs"][0] if c == "input" else out(c, r)) for c in cols},
                                         scores={c: round(float(nz.loc[uid, c]), 3) for c in MODELS + ft},
                                         metric="normalised primary metric (0-1)"))
    json.dump(ex, open(os.path.join(DIAG, "diag_examples.json"), "w"), indent=1)
    print({k: len(v) for k, v in ex.items()})


if __name__ == "__main__":
    main()
