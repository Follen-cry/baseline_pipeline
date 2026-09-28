#!/usr/bin/env python
"""Aggregate judge + rule outputs into result CSVs (env: internvlu).

Inputs : results/judge_raw/<judge>/<bench>/<model>.jsonl, results/rule/*.csv, manifest.jsonl
Outputs (results/):
  scores_item.csv        one row per (uid, model, metric): native per-item values
  table_main.csv         model x benchmark, each benchmark's native aggregate(s)
  table_category.csv     model x benchmark x category (native aggregate)
  table_phyedit_type.csv PhyEditBench by type A-E (weighted overall + 4 dims)
  table_anti.csv         Anti-Physics by rule type (weighted overall + 4 dims)
  paired.csv             trained - base: mean delta, cluster-bootstrap 95% CI, Wilcoxon p, Holm p
  group_summary.csv      per evidence group, normalised (0-1 of metric range) deltas + realised MDE
  trend.csv              normalised group means per model (T0 -> T2 -> T3)
  parse_stats.csv        judge units, first-try parse rate, retries, residual failures
"""
import collections
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
MODELS = ["base", "T0", "T2", "T3"]
TRAINED = ["T0", "T2", "T3"]
REF = "copy"  # input-copy reference: judged like a model, never part of the Holm family
JUDGE = os.environ.get("JUDGE", "qwen3vl30b_fp8")
PHY_W = {"consistency": 0.2, "instruction_following": 0.3, "physical_plausibility": 0.4, "image_quality": 0.1}
ANTI_W = {"Consistency": 0.2, "Instruction_Following": 0.3, "Physical_Plausibility": 0.4, "Image_Quality": 0.1}
N_BOOT = 10000
RNG = np.random.default_rng(0)

# primary per-item metric of each benchmark and its range, used for group-level normalisation
PRIMARY = {"phyeditbench": ("overall", 1, 10), "phyedit_anti": ("overall", 1, 10),
           "picabench": ("acc", 0, 1), "risebench": ("score", 1, 5), "imgedit_basic": ("score", 1, 5),
           "imgedit_uge": ("score", 1, 5), "magicbrush": ("vlm_score", 1, 5)}


def read_jsonl(p):
    """Judge rows, de-duplicated by unit key (last write wins; resumable runs never re-judge a key,
    but an interrupted-then-restarted run could)."""
    if not os.path.exists(p):
        return []
    rows = {}
    for l in open(p):
        r = json.loads(l)
        rows[r.get("key", len(rows))] = r
    return list(rows.values())


def manifest():
    m = pd.DataFrame([json.loads(l) for l in open(os.path.join(HERE, "manifest.jsonl"))])
    # clustering unit: a PhyEditBench trajectory (its 5 types are correlated); every other item alone
    m["cluster"] = np.where(m.bench == "phyeditbench",
                            "phy/" + m.category + "/" + m.subcategory + "/" + m.orig_id.astype(str), m.uid)
    return m.set_index("uid")


def item_scores(judge=JUDGE):
    """-> long DataFrame uid, model, metric, value  (+ parse stats list)."""
    recs, pstats = [], []
    base = os.path.join(RES, "judge_raw", judge)

    def pstat(bench, model, rows):
        n = len(rows)
        first = sum(1 for r in rows if r.get("tries") and r["tries"][0]["ok"])
        ok = sum(1 for r in rows if r.get("parse_ok"))
        if not rows:
            return
        pstats.append(dict(judge=judge, bench=bench, model=model, units=n, first_try_ok=first,
                           ok_after_retry=ok, residual_fail=n - ok))

    for model in MODELS + [REF]:
        rows = read_jsonl(f"{base}/phyeditbench/{model}.jsonl")
        pstat("phyeditbench", model, rows)
        per = collections.defaultdict(dict)
        for r in rows:
            if r.get("score") is not None:
                per[r["uid"]][r["dimension"]] = r["score"]
        for uid, d in per.items():
            for k, v in d.items():
                recs.append((uid, model, k, v))
            if len(d) == 4:
                recs.append((uid, model, "overall", sum(PHY_W[k] * d[k] for k in PHY_W)))

        rows = read_jsonl(f"{base}/phyedit_anti/{model}.jsonl")
        pstat("phyedit_anti", model, rows)
        for r in rows:
            d = {k: r.get(k) for k in ANTI_W}
            for k, v in d.items():
                if v is not None:
                    recs.append((r["uid"], model, k, v))
            if all(v is not None for v in d.values()):
                recs.append((r["uid"], model, "overall", sum(ANTI_W[k] * d[k] for k in ANTI_W)))

        rows = read_jsonl(f"{base}/picabench/{model}.jsonl")
        pstat("picabench", model, rows)
        per = collections.defaultdict(list)
        for r in rows:
            per[r["uid"]].append(bool(r.get("is_correct")))  # unparsed answers count as incorrect (official)
        for uid, v in per.items():
            recs.append((uid, model, "acc", float(np.mean(v))))

        rows = read_jsonl(f"{base}/risebench/{model}.jsonl")
        pstat("risebench", model, rows)
        for r in rows:
            s = r.get("scores")
            if not s:
                continue
            for k in ("score", "complete", "Reasoning", "ApprConsistency", "VisualPlausibility"):
                if s.get(k) is not None and not (isinstance(s[k], float) and math.isnan(s[k])):
                    recs.append((r["uid"], model, k, float(s[k])))

        for b in ("imgedit_basic", "imgedit_uge", "magicbrush"):
            rows = read_jsonl(f"{base}/{b}/{model}.jsonl")
            pstat(b, model, rows)
            for r in rows:
                if r.get("score") is not None:
                    recs.append((r["uid"], model, "vlm_score" if b == "magicbrush" else "score", r["score"]))

    df = pd.DataFrame(recs, columns=["uid", "model", "metric", "value"])
    for part in ("pica", "magicbrush", "editmag"):
        p = os.path.join(RES, "rule", f"{part}.csv")
        if os.path.exists(p) and judge == JUDGE:
            r = pd.read_csv(p)
            r = r[r.value.notna()]
            df = pd.concat([df, r[["uid", "model", "metric", "value"]]], ignore_index=True)
    return df, pd.DataFrame(pstats)


def wide(df, m, bench, metric):
    sub = df[(df.metric == metric) & df.uid.isin(m.index[m.bench == bench])]
    w = sub.pivot_table(index="uid", columns="model", values="value", aggfunc="first")
    return w.dropna(subset=[c for c in MODELS if c in w.columns]) if set(MODELS) <= set(w.columns) else w.iloc[0:0]


def paired(delta, clusters):
    """delta: per-item trained-base; clusters: cluster id per item. Cluster bootstrap CI + Wilcoxon on cluster means."""
    d = pd.DataFrame({"d": delta.values, "c": clusters.values})
    cm = d.groupby("c").d.mean().values
    n = len(cm)
    if n < 3:
        return dict(n_items=len(d), n_clusters=n, delta=float(d.d.mean()), ci_lo=np.nan, ci_hi=np.nan, p=np.nan, mde=np.nan)
    # weight clusters by size so the bootstrap mean equals the item-level mean
    g = d.groupby("c").d.agg(["sum", "count"]).values
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    boot = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = stats.wilcoxon(cm, zero_method="zsplit").pvalue if np.any(cm != 0) else 1.0
    return dict(n_items=len(d), n_clusters=n, delta=float(d.d.mean()), ci_lo=float(lo), ci_hi=float(hi),
                p=float(p), mde=float(2.80 * np.std(cm, ddof=1) / math.sqrt(n)))


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def native_tables(df, m):
    main, cat, typ, anti = [], [], [], []
    for model in MODELS + [REF]:
        d = df[df.model == model].merge(m[["bench", "category", "subcategory", "etype", "group"]],
                                        left_on="uid", right_index=True)
        def agg(sub, bench, metric, how="mean", scale=1.0):
            v = sub[(sub.bench == bench) & (sub.metric == metric)].value
            return (v.mean() * scale if how == "mean" else np.nan), len(v)
        row = dict(model=model)
        # PhyEditBench: official = per-dimension mean, then weighted sum
        def phy_overall(sub):
            dm = {k: sub[(sub.bench == "phyeditbench") & (sub.metric == k)].value.mean() for k in PHY_W}
            return sum(PHY_W[k] * dm[k] for k in PHY_W), dm
        o, dm = phy_overall(d)
        row["PhyEditBench overall (1-10)"] = o
        for k in PHY_W:
            row[f"PhyEditBench {k}"] = dm[k]
        ad = {k: d[(d.bench == "phyedit_anti") & (d.metric == k)].value.mean() for k in ANTI_W}
        row["Anti-Physics overall (1-10)"] = sum(ANTI_W[k] * ad[k] for k in ANTI_W)
        row["PICABench Acc (%)"] = agg(d, "picabench", "acc", scale=100)[0]
        row["PICABench Con PSNR (dB)"] = agg(d, "picabench", "psnr_nonedit")[0]
        for grp, label in (("A", "RISE Temporal+Causal"), ("B", "RISE Spatial"), ("C", "RISE Logical")):
            s = d[(d.bench == "risebench") & (d.group == grp)]
            row[f"{label} Acc (%)"] = s[s.metric == "complete"].value.mean() * 100
            row[f"{label} Score (1-5)"] = s[s.metric == "score"].value.mean()
        row["ImgEdit Basic (1-5)"] = agg(d, "imgedit_basic", "score")[0]
        row["ImgEdit UGE (1-5)"] = agg(d, "imgedit_uge", "score")[0]
        row["MagicBrush VLM (1-5)"] = agg(d, "magicbrush", "vlm_score")[0]
        for k in ("l1", "clip_i", "dino"):
            row[f"MagicBrush {k}"] = agg(d, "magicbrush", k)[0]
        em = d[d.metric == "edit_mad"]
        row["Edit magnitude, all tasks (mean |out-in|, 0-1)"] = em.value.mean() if len(em) else np.nan
        main.append(row)

        # per category, native metric of each benchmark
        for (bench, c, sc), s in d.groupby(["bench", "category", "subcategory"]):
            if bench == "phyeditbench":
                dmc = {k: s[s.metric == k].value.mean() for k in PHY_W}
                cat.append(dict(model=model, bench=bench, category=c, subcategory=sc, metric="overall (1-10)",
                                value=sum(PHY_W[k] * dmc[k] for k in PHY_W), n=s.uid.nunique()))
            elif bench == "picabench":
                cat.append(dict(model=model, bench=bench, category=c, subcategory=sc, metric="Acc (%)",
                                value=s[s.metric == "acc"].value.mean() * 100, n=s.uid.nunique()))
                cat.append(dict(model=model, bench=bench, category=c, subcategory=sc, metric="Con PSNR (dB)",
                                value=s[s.metric == "psnr_nonedit"].value.mean(), n=s.uid.nunique()))
            elif bench == "risebench":
                for mt, lab, sc_ in (("complete", "Acc (%)", 100), ("score", "Score (1-5)", 1)):
                    cat.append(dict(model=model, bench=bench, category=c, subcategory=sc, metric=lab,
                                    value=s[s.metric == mt].value.mean() * sc_, n=s.uid.nunique()))
            elif bench == "phyedit_anti":
                da = {k: s[s.metric == k].value.mean() for k in ANTI_W}
                cat.append(dict(model=model, bench=bench, category=c, subcategory=sc, metric="overall (1-10)",
                                value=sum(ANTI_W[k] * da[k] for k in ANTI_W), n=s.uid.nunique()))
            else:
                mt = "vlm_score" if bench == "magicbrush" else "score"
                cat.append(dict(model=model, bench=bench, category=c, subcategory=sc, metric="score (1-5)",
                                value=s[s.metric == mt].value.mean(), n=s.uid.nunique()))
        # PhyEditBench by type
        for t, s in d[d.bench == "phyeditbench"].groupby("etype"):
            dmt = {k: s[s.metric == k].value.mean() for k in PHY_W}
            typ.append(dict(model=model, type=t, overall=sum(PHY_W[k] * dmt[k] for k in PHY_W), **dmt,
                            n=s.uid.nunique()))
        # PhyEditBench by primary class (official by_primary)
        for c, s in d[d.bench == "phyeditbench"].groupby("category"):
            dmc = {k: s[s.metric == k].value.mean() for k in PHY_W}
            cat.append(dict(model=model, bench="phyeditbench", category=c, subcategory="(all)",
                            metric="overall (1-10)", value=sum(PHY_W[k] * dmc[k] for k in PHY_W), n=s.uid.nunique()))
        for t, s in d[d.bench == "phyedit_anti"].groupby("subcategory"):
            da = {k: s[s.metric == k].value.mean() for k in ANTI_W}
            anti.append(dict(model=model, rule_type=t, overall=sum(ANTI_W[k] * da[k] for k in ANTI_W), **da,
                             n=s.uid.nunique()))
        s = d[d.bench == "phyedit_anti"]
        da = {k: s[s.metric == k].value.mean() for k in ANTI_W}
        anti.append(dict(model=model, rule_type="ALL", overall=sum(ANTI_W[k] * da[k] for k in ANTI_W), **da,
                         n=s.uid.nunique()))
    return pd.DataFrame(main), pd.DataFrame(cat), pd.DataFrame(typ), pd.DataFrame(anti)


def paired_tables(df, m):
    """Paired tests at benchmark, group and category level (native metrics, complete cases only)."""
    out = []
    specs = [("phyeditbench", "overall"), ("phyeditbench", "physical_plausibility"),
             ("phyeditbench", "instruction_following"), ("phyeditbench", "consistency"),
             ("phyeditbench", "image_quality"), ("phyedit_anti", "overall"),
             ("phyedit_anti", "Instruction_Following"), ("phyedit_anti", "Physical_Plausibility"),
             ("picabench", "acc"), ("picabench", "psnr_nonedit"), ("risebench", "score"),
             ("risebench", "complete"), ("imgedit_basic", "score"), ("imgedit_uge", "score"),
             ("magicbrush", "vlm_score"), ("magicbrush", "l1"), ("magicbrush", "clip_i"), ("magicbrush", "dino")]
    specs += [(b, "edit_mad") for b in PRIMARY]  # diagnostic: how much each model changes the input
    for bench, metric in specs:
        w = wide(df, m, bench, metric)
        if w.empty:
            continue
        meta = m.loc[w.index]
        levels = [("benchmark", None, w.index)]
        if bench == "risebench":
            levels = [("group", g, meta.index[meta.group == g]) for g in ("A", "B", "C")]
            levels += [("category", c, meta.index[meta.category == c]) for c in sorted(meta.category.unique())]
            levels += [("subcategory", c, meta.index[meta.subcategory == c]) for c in sorted(meta.subcategory.unique())]
        else:
            levels += [("category", c, meta.index[meta.category == c]) for c in sorted(meta.category.unique())
                       if meta.category.nunique() > 1]
            levels += [("subcategory", c, meta.index[meta.subcategory == c]) for c in sorted(meta.subcategory.unique())
                       if meta.subcategory.nunique() > 1]
            if bench == "phyeditbench":
                levels += [("type", t, meta.index[meta.etype == t]) for t in sorted(meta.etype.unique())]
        for level, name, idx in levels:
            rows = []
            for t in TRAINED:
                r = paired(w.loc[idx, t] - w.loc[idx, "base"], meta.loc[idx, "cluster"])
                rows.append(dict(bench=bench, metric=metric, level=level, name=name or "all", model=t,
                                 base_mean=float(w.loc[idx, "base"].mean()), model_mean=float(w.loc[idx, t].mean()), **r))
            adj = holm([r["p"] if not np.isnan(r["p"]) else 1.0 for r in rows])
            for r, a in zip(rows, adj):
                r["p_holm"] = float(a)
            if REF in w.columns and w.loc[idx, REF].notna().all():
                r = paired(w.loc[idx, REF] - w.loc[idx, "base"], meta.loc[idx, "cluster"])
                rows.append(dict(bench=bench, metric=metric, level=level, name=name or "all", model=REF,
                                 base_mean=float(w.loc[idx, "base"].mean()), model_mean=float(w.loc[idx, REF].mean()),
                                 **r, p_holm=np.nan))
            out += rows
    return pd.DataFrame(out)


def normalised(df, m):
    """Per-item primary metric scaled to [0,1] of its range -> long table uid, model, norm."""
    parts = []
    for bench, (metric, lo, hi) in PRIMARY.items():
        w = wide(df, m, bench, metric)
        if w.empty:
            continue
        parts.append(((w - lo) / (hi - lo)).assign(bench=bench))
    return pd.concat(parts) if parts else pd.DataFrame()


def group_tables(df, m):
    nz = normalised(df, m)
    rows, trend = [], []
    if nz.empty:
        return pd.DataFrame(), pd.DataFrame()
    meta = m.loc[nz.index].copy()
    pools = {g: meta.index[meta.group == g] for g in ("A", "B", "C")}
    # POST-HOC sensitivity pool (not pre-registered): target benchmarks on which the input-copy reference
    # scores clearly below base, i.e. whose judge does not reward inaction. Excludes PhyEditBench normal types.
    pools["A_copy_sensitive"] = meta.index[(meta.group == "A") & (meta.bench != "phyeditbench")]
    for g, idx in pools.items():
        sub = nz.loc[idx]
        for model in MODELS + ([REF] if REF in sub.columns else []):
            trend.append(dict(group=g, model=model, norm_mean=float(sub[model].mean()), n=int(sub[model].notna().sum())))
        res = []
        for t in TRAINED:
            r = paired(sub[t] - sub["base"], meta.loc[idx, "cluster"])
            res.append(dict(group=g, model=t, benches=",".join(sorted(sub.bench.unique())), **r))
        for r, a in zip(res, holm([r["p"] for r in res])):
            r["p_holm"] = float(a)
        if REF in sub.columns:
            ok = sub[REF].notna()
            res.append(dict(group=g, model=REF, benches=",".join(sorted(sub.bench.unique())),
                            **paired((sub[REF] - sub["base"])[ok], meta.loc[idx, "cluster"][ok.values]), p_holm=np.nan))
        rows += res
    return pd.DataFrame(rows), pd.DataFrame(trend)


def main():
    m = manifest()
    df, ps = item_scores()
    os.makedirs(RES, exist_ok=True)
    meta = m[["bench", "group", "category", "subcategory", "etype", "cluster"]]
    df.merge(meta, left_on="uid", right_index=True).to_csv(os.path.join(RES, "scores_item.csv"), index=False)
    ps.to_csv(os.path.join(RES, "parse_stats.csv"), index=False)
    main_t, cat_t, typ_t, anti_t = native_tables(df, m)
    main_t.to_csv(os.path.join(RES, "table_main.csv"), index=False)
    cat_t.to_csv(os.path.join(RES, "table_category.csv"), index=False)
    typ_t.to_csv(os.path.join(RES, "table_phyedit_type.csv"), index=False)
    anti_t.to_csv(os.path.join(RES, "table_anti.csv"), index=False)
    pt = paired_tables(df, m)
    pt.to_csv(os.path.join(RES, "paired.csv"), index=False)
    gs, tr = group_tables(df, m)
    gs.to_csv(os.path.join(RES, "group_summary.csv"), index=False)
    tr.to_csv(os.path.join(RES, "trend.csv"), index=False)
    pd.set_option("display.width", 250)
    print(main_t.T)
    print(gs)
    print(ps.groupby("bench")[["units", "first_try_ok", "ok_after_retry", "residual_fail"]].sum())


if __name__ == "__main__":
    main()
