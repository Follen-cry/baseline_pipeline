#!/usr/bin/env python
"""Examples, image grids and REPORT.md, all from the result CSVs written by analyze.py.

  results/examples.csv    per item: normalised primary metric per model, mean trained-base delta, rank tags
  results/verdicts.csv    per evidence group verdict (pre-registered rule, see SUITE.md)
  results/grids/*.jpg     input | base | T0 | T2 | T3 | reference, instruction + scores underneath
  results/REPORT.md
(env: internvlu)
"""
import json
import os
import textwrap

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(EVAL, "results")
OUT = os.path.join(EVAL, "outputs")
MODELS = ["base", "T0", "T2", "T3"]
TRAINED = ["T0", "T2", "T3"]
PRIMARY = {"phyeditbench": ("overall", 1, 10), "phyedit_anti": ("overall", 1, 10), "picabench": ("acc", 0, 1),
           "risebench": ("score", 1, 5), "imgedit_basic": ("score", 1, 5), "imgedit_uge": ("score", 1, 5),
           "magicbrush": ("vlm_score", 1, 5)}
BENCH_LABEL = {"phyeditbench": "PhyEditBench", "phyedit_anti": "PhyEditBench Anti-Physics", "picabench": "PICABench",
               "risebench": "RISEBench", "imgedit_basic": "ImgEdit Basic", "imgedit_uge": "ImgEdit UGE",
               "magicbrush": "MagicBrush"}
GROUP_LABEL = {"A": "A · Target", "B": "B · Secondary", "C": "C · Controls",
               "A_copy_sensitive": "A* · Target without PhyEditBench-normal (post-hoc)"}
COLS = MODELS + ["copy"]  # copy = input-copy reference, shown for calibration only
# planned MDE (normalised), SUITE.md "Power (planned)"
PLANNED_MDE = {"A": 0.027, "B": 0.090, "C": 0.050}


def manifest():
    return pd.DataFrame([json.loads(l) for l in open(os.path.join(HERE, "manifest.jsonl"))]).set_index("uid")


def score_label(bench, row_scores):
    """Short per-model score string shown under a grid (native metric)."""
    return row_scores


def build_examples(m, items):
    rows = []
    for bench, (metric, lo, hi) in PRIMARY.items():
        s = items[(items.bench == bench) & (items.metric == metric)]
        w = s.pivot_table(index="uid", columns="model", values="value", aggfunc="first").dropna(subset=MODELS)
        if w.empty or not set(MODELS) <= set(w.columns):
            continue
        n = (w[MODELS] - lo) / (hi - lo)
        for uid, r in n.iterrows():
            d = {t: r[t] - r["base"] for t in TRAINED}
            rows.append(dict(uid=uid, bench=bench, group=m.loc[uid, "group"], category=m.loc[uid, "category"],
                             subcategory=m.loc[uid, "subcategory"], etype=m.loc[uid, "etype"], metric=metric,
                             **{f"native_{k}": w.loc[uid, k] for k in MODELS},
                             **{f"norm_{k}": r[k] for k in MODELS},
                             mean_delta=float(np.mean(list(d.values()))),
                             n_trained_better=int(sum(v > 0 for v in d.values())),
                             n_trained_worse=int(sum(v < 0 for v in d.values()))))
    ex = pd.DataFrame(rows)
    # rank tags: "win" = all three trained models beat base, ranked by mean delta; "loss" = all three worse
    ex["tag"] = ""
    ex.loc[ex.n_trained_better == 3, "tag"] = "win"
    ex.loc[ex.n_trained_worse == 3, "tag"] = "loss"
    return ex


def pick(ex, by, k):
    """Top-k clearest wins and losses within each `by` group (ties broken by uid for determinism)."""
    out = []
    for key, g in ex.groupby(by):
        w = g[g.tag == "win"].sort_values(["mean_delta", "uid"], ascending=[False, True]).head(k)
        l = g[g.tag == "loss"].sort_values(["mean_delta", "uid"], ascending=[True, True]).head(k)
        out += [w.assign(kind="win"), l.assign(kind="loss")]
    return pd.concat(out) if out else ex.iloc[0:0]


def ref_path(r):
    j = r["judge"]
    if r["bench"] == "phyeditbench":
        return j["ref"]
    if r["bench"] == "magicbrush":
        return j["gt"]
    if r["bench"] == "risebench" and j.get("reference_img"):
        return j["reference_img"]
    return None


def detail_scores(items, uid, bench):
    """Per-model detail string: native primary + key sub-scores."""
    s = items[items.uid == uid]
    out = {}
    for mdl in MODELS:
        v = s[s.model == mdl].set_index("metric").value
        if bench == "phyeditbench":
            txt = f"overall {v.get('overall', np.nan):.1f} · phys {v.get('physical_plausibility', np.nan):.0f} · instr {v.get('instruction_following', np.nan):.0f}"
        elif bench == "phyedit_anti":
            txt = f"overall {v.get('overall', np.nan):.1f} · IF {v.get('Instruction_Following', np.nan):.0f} · PP {v.get('Physical_Plausibility', np.nan):.0f}"
        elif bench == "picabench":
            txt = f"acc {100 * v.get('acc', np.nan):.0f}% · PSNR {v.get('psnr_nonedit', np.nan):.1f}"
        elif bench == "risebench":
            txt = (f"R{v.get('Reasoning', np.nan):.0f} C{v.get('ApprConsistency', np.nan):.0f} "
                   f"V{v.get('VisualPlausibility', np.nan):.0f} · {'✓' if v.get('complete', 0) == 1 else '✗'}")
        elif bench == "magicbrush":
            txt = f"VLM {v.get('vlm_score', np.nan):.2f} · DINO {v.get('dino', np.nan):.2f}"
        else:
            txt = f"score {v.get('score', np.nan):.2f}"
        out[mdl] = txt.replace("nan", "–")
    return out


def grid(r, scores, path, cell=256):
    ims = [("input", r["inputs"][0])] + [(mdl, os.path.join(OUT, mdl, r["out_rel"])) for mdl in MODELS]
    rp = ref_path(r)
    if rp:
        ims.append(("reference", rp))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    text = textwrap.wrap(r["instruction"].replace("\n", " "), width=int(len(ims) * cell / 7.2))[:4]
    H = cell + 22 + 18 + 16 * len(text) + 8
    g = Image.new("RGB", (cell * len(ims), H), "white")
    d = ImageDraw.Draw(g)
    for i, (lab, p) in enumerate(ims):
        im = Image.open(p).convert("RGB")
        im.thumbnail((cell, cell))
        g.paste(im, (i * cell + (cell - im.width) // 2, 18 + (cell - im.height) // 2))
        d.text((i * cell + 4, 2), lab, fill="black", font=font)
        if lab in scores:
            d.text((i * cell + 4, 20 + cell), scores[lab], fill="black", font=font)
    for k, line in enumerate(text):
        d.text((4, cell + 42 + 16 * k), line, fill=(60, 60, 60), font=font)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    g.save(path, quality=88)


def fmt(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def resolve(text, gs, pt, main_t, ag):
    """Same {{kind:...}} placeholders as the artifact page: every number comes from the CSVs."""
    import re

    def sg(x, d=3):
        return "–" if x is None or pd.isna(x) else f"{x:+.{d}f}"

    def rep(mt):
        kind, a = mt.group(1), mt.group(2).split(":")
        if kind == "g":
            r = gs[(gs.group == a[0]) & (gs.model == a[1])]
            if r.empty:
                return "?"
            r = r.iloc[0]
            return f"[{sg(r.ci_lo)}, {sg(r.ci_hi)}]" if a[2] == "ci" else (fmt(r.p_holm, 3) if a[2] == "p" else sg(r[a[2]]))
        if kind == "p":
            r = pt[(pt.bench == a[0]) & (pt.metric == a[1]) & (pt.level == a[2]) & (pt["name"].astype(str) == a[3]) & (pt.model == a[4])]
            if r.empty:
                return "?"
            r, d = r.iloc[0], int(a[6]) if len(a) > 6 else 2
            if a[5] == "ci":
                return f"[{sg(r.ci_lo, d)}, {sg(r.ci_hi, d)}]"
            return fmt(r.p_holm, 3) if a[5] == "p" else (sg(r.delta, d) if a[5] == "delta" else fmt(r[a[5]], d))
        if kind == "a":
            r = ag[(ag.other_judge == a[0]) & (ag.bench == a[1])] if ag is not None else pd.DataFrame()
            return fmt(r.iloc[0][a[2]], 2) if len(r) else "?"
        if kind == "m":
            col = ":".join(a[1:])
            return fmt(main_t.loc[a[0], col], 2) if col in main_t.columns else "?"
        return mt.group(0)
    return re.sub(r"\{\{(\w+):([^}]+)\}\}", rep, text)


def verdicts(gs):
    rows = []
    for g in ("A", "B", "C"):
        s = gs[(gs.group == g) & gs.model.isin(TRAINED)]
        if s.empty:
            continue
        sig_pos = s[(s.p_holm < 0.05) & (s.ci_lo > 0)]
        sig_neg = s[(s.p_holm < 0.05) & (s.ci_hi < 0)]
        mde = PLANNED_MDE[g]
        if g == "C":
            if len(sig_neg):
                v = "harm: training degraded general editing"
            elif (s.ci_lo > -mde).all():
                v = "no harm: all CIs exclude a loss larger than the planned MDE"
            else:
                v = "inconclusive: no significant loss, but a loss of MDE size is not excluded"
        else:
            if len(sig_pos) and not len(sig_neg):
                v = "supports: " + ", ".join(sig_pos.model) + " significantly above base"
            elif len(sig_neg) and not len(sig_pos):
                v = "refutes: " + ", ".join(sig_neg.model) + " significantly below base"
            elif (s.ci_hi < mde).all():
                v = "refutes (no effect): every CI excludes a gain of planned-MDE size"
            else:
                v = "inconclusive"
        rows.append(dict(group=g, verdict=v))
    return pd.DataFrame(rows)


def main():
    m = manifest()
    items = pd.read_csv(os.path.join(RES, "scores_item.csv"))
    main_t = pd.read_csv(os.path.join(RES, "table_main.csv")).set_index("model")
    cat = pd.read_csv(os.path.join(RES, "table_category.csv"))
    typ = pd.read_csv(os.path.join(RES, "table_phyedit_type.csv"))
    anti = pd.read_csv(os.path.join(RES, "table_anti.csv"))
    pt = pd.read_csv(os.path.join(RES, "paired.csv"))
    gs = pd.read_csv(os.path.join(RES, "group_summary.csv"))
    tr = pd.read_csv(os.path.join(RES, "trend.csv"))
    ps = pd.read_csv(os.path.join(RES, "parse_stats.csv"))
    ag = pd.read_csv(os.path.join(RES, "judge_agreement.csv")) if os.path.exists(os.path.join(RES, "judge_agreement.csv")) else None
    cfg = json.load(open(os.path.join(HERE, "config.json")))

    # suite description CSVs (the artifact reads its numbers from CSVs only)
    m.reset_index().groupby(["group", "bench", "category", "subcategory"]).size().rename("n").reset_index() \
        .to_csv(os.path.join(RES, "suite_counts.csv"), index=False)
    pd.DataFrame([  # SUITE.md "Power (planned)": MDE = 2.8 * SD(delta) / sqrt(n_eff), normalised units
        dict(scope="A total", n=950, n_eff=720, sd_delta=0.26, mde_norm=0.027, mde_native=""),
        dict(scope="A · PhyEditBench overall", n=480, n_eff=250, sd_delta=0.25, mde_norm=0.044, mde_native="0.40 pt (1–10)"),
        dict(scope="A · PICABench Acc", n=260, n_eff=260, sd_delta=0.30, mde_norm=0.052, mde_native="5.2 pp"),
        dict(scope="A · RISE Temporal+Causal score", n=175, n_eff=175, sd_delta=0.25, mde_norm=0.053, mde_native="0.21 (1–5)"),
        dict(scope="A · Anti-Physics", n=35, n_eff=35, sd_delta=0.25, mde_norm=0.12, mde_native="1.1 pt (1–10)"),
        dict(scope="B RISE Spatial", n=60, n_eff=60, sd_delta=0.25, mde_norm=0.090, mde_native="0.36 (1–5)"),
        dict(scope="C total", n=200, n_eff=200, sd_delta=0.25, mde_norm=0.050, mde_native=""),
        dict(scope="C · ImgEdit Basic", n=108, n_eff=108, sd_delta=0.25, mde_norm=0.067, mde_native="0.27 (1–5)"),
    ]).to_csv(os.path.join(RES, "power_planned.csv"), index=False)
    pd.DataFrame([
        dict(benchmark="PhyEditBench (+Anti-Physics)", decision="included", group="A",
             why="Real 4-state physical trajectories with GT targets; step-wise (A–C), chained (D) and global (E) types; counterfactual rules test prior vs instruction."),
        dict(benchmark="PICABench", decision="included", group="A",
             why="8 physics laws with region-grounded yes/no QA on annotated ROIs; Mechanics/State weighted up."),
        dict(benchmark="RISEBench", decision="included", group="A/B/C",
             why="Temporal+Causal are target, Spatial secondary, Logical a control; 1–5 rubric and strict all-dims accuracy."),
        dict(benchmark="ImgEdit (Basic + UGE)", decision="included", group="C",
             why="9 standard edit types plus hard UGE edits: general editing ability should not drop."),
        dict(benchmark="MagicBrush", decision="included", group="C",
             why="GT targets allow judge-free L1 / CLIP-I / DINO checks."),
        dict(benchmark="KRIS-Bench", decision="skipped", group="",
             why="Knowledge-based editing; its physics/temporal subsets overlap RISE Temporal/Causal and PICABench."),
        dict(benchmark="GEdit-Bench", decision="skipped", group="",
             why="General real-user edits; duplicates ImgEdit's role as a control."),
        dict(benchmark="PBench-Edit (ChronoEdit)", decision="skipped", group="",
             why="Video-derived physical edits; overlaps PhyEditBench, which has GT targets and step-wise types."),
        dict(benchmark="AURORA-Bench", decision="skipped", group="",
             why="Action edits from video incl. SSv2, which is in the T training pool: confounds transfer with domain overlap. Candidate positive control."),
    ]).to_csv(os.path.join(RES, "benchmarks_considered.csv"), index=False)

    ex = build_examples(m, items)
    ex.to_csv(os.path.join(RES, "examples.csv"), index=False)
    vd = verdicts(gs)
    vd.to_csv(os.path.join(RES, "verdicts.csv"), index=False)

    # REPORT grids: 4 clearest wins + 4 clearest losses per evidence group
    sel = pick(ex, "group", 4)
    grid_md = {}
    for _, e in sel.iterrows():
        r = m.loc[e.uid].to_dict()
        r["uid"] = e.uid
        fn = f"grids/{e.group}_{e.kind}_{e.uid.replace('/', '_').replace('&', 'and')}.jpg"
        grid(r, detail_scores(items, e.uid, e.bench), os.path.join(RES, fn))
        grid_md.setdefault((e.group, e.kind), []).append((fn, e))

    L = []
    L.append("# Video-SSL physics-editing evaluation: base vs T0 / T2 / T3\n")
    L.append("Suite design, selection and planned power: [`../SUITE.md`](../SUITE.md). Every number below is read "
             "from the CSVs in this folder (`pipeline/analyze.py` → `report.py`).\n")
    L.append("> **Judge caveat.** All VLM scores come from Qwen3-VL-30B-A3B-Instruct-FP8 (vLLM, temperature 0) running "
             "each benchmark's official judge prompts and parsers. They are **not comparable to published "
             "GPT-4o/4.1-judged leaderboard numbers**. Compare only across our four models.\n")
    L.append("## Setup\n")
    inf = cfg["inference"]
    L.append(f"- Resolution: {inf['resolution']}\n- Sampler: {inf['sampler']}, {inf['num_inference_steps']} steps, "
             f"all_cfg_scale {inf['all_cfg_scale']}, part_cfg_scale {inf['part_cfg_scale']}\n- Seed: {inf['seed']} "
             f"({inf['seed_rule']})\n- Prompt: {inf['prompt_template']}\n")
    L.append("- Checkpoints: " + "; ".join(f"**{k}** `{v}`" for k, v in cfg["models"].items()) + "\n")
    L.append("- Per-benchmark judging: PhyEditBench `utils.score_one_dimension` (4 dims, 1–10; overall = "
             "0.4·Phys+0.3·Instr+0.2·Cons+0.1·Qual); Anti-Physics `gpt_eval_anti.vlm_judge` + official checklists; "
             "PICABench `PicaEval_qwen` ROI-crop yes/no QA (Acc = mean per-sample QA accuracy) + official masked PSNR "
             "(Con, 512); RISEBench `gpt_eval.eval_vanilla` (Reasoning/Consistency/Plausibility 1–5; Acc = all "
             "applicable dims = 5; Score = official weighted 1–5); ImgEdit `basic_bench` type-specific prompts and "
             "`step1` parser; UGE `UGE_bench`; MagicBrush L1/CLIP-I/DINO vs GT + the UGE (type-agnostic ImgEdit) rubric.\n")
    L.append("- Parse policy: official parser first; on failure the call is retried up to 3× (temperature 0.7, "
             "seeded) and every attempt is logged; residual failures are kept and shown below. UGE/MagicBrush add "
             "a documented `Score: N` fallback because the official UGE prompt defines no score line.\n")

    L.append("## Evidence-group summary\n")
    L.append("Δ = trained − base on the benchmark's primary per-item metric scaled to 0–1 of its range (PhyEditBench "
             "overall, Anti overall, PICABench Acc, RISE Score, ImgEdit/UGE/MagicBrush VLM score), pooled over all "
             "items in the group. The CI is a cluster bootstrap (10k resamples; a PhyEditBench trajectory is one "
             "cluster). p is a Wilcoxon signed-rank test on cluster means, Holm-adjusted over T0/T2/T3; "
             "* p<0.05, ** p<0.01. Realised MDE = 2.8·SD(Δ)/√n.\n")
    L.append("| group | model | n | Δ (norm.) | 95% CI | p (Holm) | realised MDE |")
    L.append("|---|---|---:|---:|---|---:|---:|")
    for _, r in gs.iterrows():
        L.append(f"| {GROUP_LABEL[r.group]} | {r.model} | {r.n_items} | {r.delta:+.3f}{star(r.p_holm)} | "
                 f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {fmt(r.p_holm, 3)} | {r.mde:.3f} |")
    L.append("\nGroup means per model (normalised), T0 → T2 → T3 trend:\n")
    L.append("| group | " + " | ".join(COLS) + " |\n|---|" + "---:|" * len(COLS))
    for g, s in tr.groupby("group"):
        L.append(f"| {GROUP_LABEL[g]} | " + " | ".join(fmt(s.set_index('model').norm_mean.get(mm), 3) for mm in COLS) + " |")
    L.append("\n**Verdicts** (rule fixed before results: *supports* = a trained model significantly above base "
             "(Holm p<0.05, CI>0) and none significantly below; *refutes (no effect)* = every CI upper bound below "
             "the planned MDE; controls: *no harm* = every CI lower bound above −planned MDE):\n")
    for _, r in vd.iterrows():
        L.append(f"- **{GROUP_LABEL[r.group]}**: {r.verdict}")

    L.append("\n## Main table (native metrics)\n")
    t = main_t.T
    L.append("| metric | " + " | ".join(COLS) + " |\n|---|" + "---:|" * len(COLS))
    for k, r in t.iterrows():
        L.append(f"| {k} | " + " | ".join(fmt(r.get(mm)) for mm in COLS) + " |")

    L.append("\n## Paired comparisons vs base (benchmark level)\n")
    L.append("| benchmark | metric | level | model | n | base | model | Δ | 95% CI | p (Holm) |")
    L.append("|---|---|---|---|---:|---:|---:|---:|---|---:|")
    for _, r in pt[pt.level.isin(["benchmark", "group"])].iterrows():
        L.append(f"| {BENCH_LABEL[r.bench]} | {r.metric} | {r.level}:{r['name']} | {r.model} | {r.n_items} | "
                 f"{r.base_mean:.3f} | {r.model_mean:.3f} | {r.delta:+.3f}{star(r.p_holm)} | "
                 f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {fmt(r.p_holm, 3)} |")

    L.append("\n## PhyEditBench by type (A–E)\n")
    L.append("| type | model | n | overall (1–10) | phys | instr | cons | qual |\n|---|---|---:|---:|---:|---:|---:|---:|")
    typ = typ[typ.model.isin(COLS)]
    for _, r in typ.sort_values(["type", "model"]).iterrows():
        L.append(f"| {r.type} | {r.model} | {r.n} | {r.overall:.2f} | {r.physical_plausibility:.2f} | "
                 f"{r.instruction_following:.2f} | {r.consistency:.2f} | {r.image_quality:.2f} |")
    L.append("\nPaired Δ by type (overall):\n")
    L.append("| type | model | Δ | 95% CI | p (Holm) |\n|---|---|---:|---|---:|")
    for _, r in pt[(pt.bench == "phyeditbench") & (pt.metric == "overall") & (pt.level == "type")].iterrows():
        L.append(f"| {r['name']} | {r.model} | {r.delta:+.2f}{star(r.p_holm)} | [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] | {r.p_holm:.3g} |")

    L.append("\n## Anti-Physics (counterfactual rules)\n")
    L.append("| rule type | model | n | overall | Instr. Following | Phys. Plausibility | Consistency | Image Quality |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in anti.iterrows():
        L.append(f"| {r.rule_type} | {r.model} | {r.n} | {r.overall:.2f} | {r.Instruction_Following:.2f} | "
                 f"{r.Physical_Plausibility:.2f} | {r.Consistency:.2f} | {r.Image_Quality:.2f} |")

    L.append("\n## Per-category tables\n")
    for bench in PRIMARY:
        s = cat[cat.bench == bench]
        if s.empty:
            continue
        L.append(f"\n### {BENCH_LABEL[bench]}\n")
        piv = s.pivot_table(index=["category", "subcategory", "metric", "n"], columns="model", values="value").reset_index()
        L.append("| category | subcategory | metric | n | " + " | ".join(COLS) + " |\n|---|---|---|---:|" + "---:|" * len(COLS))
        for _, r in piv.iterrows():
            L.append(f"| {r.category} | {r.subcategory} | {r.metric} | {r.n} | " + " | ".join(fmt(r.get(mm)) for mm in COLS) + " |")

    L.append("\n## Judge reliability\n")
    L.append("Parse statistics (all judged units):\n")
    agg = ps.groupby("bench")[["units", "first_try_ok", "ok_after_retry", "residual_fail"]].sum().reset_index()
    L.append("| benchmark | units | parsed first try | parsed after retry | residual failures |\n|---|---:|---:|---:|---:|")
    for _, r in agg.iterrows():
        L.append(f"| {BENCH_LABEL.get(r.bench, r.bench)} | {r.units} | {r.first_try_ok} | {r.ok_after_retry} | {r.residual_fail} |")
    if ag is not None and len(ag):
        L.append("\nSanity re-judge on a 5% stratified subset (all 4 models):\n")
        L.append("| comparison | benchmark | units | exact agree | within ±1 | Spearman ρ / κ | model-rank Kendall τ |")
        L.append("|---|---|---:|---:|---:|---:|---:|")
        for _, r in ag.iterrows():
            rho = r.get("kappa") if r.bench == "picabench" else r.get("spearman")
            L.append(f"| {r.comparison} | {BENCH_LABEL[r.bench]} | {r.n_units} | {r.exact_agree:.2f} | "
                     f"{fmt(r.get('within1_agree'))} | {fmt(rho)} | {fmt(r.model_rank_kendall_tau)} |")

    L.append("\n## Qualitative examples\n")
    L.append("Wins = all three trained models beat base on the primary metric, ranked by mean Δ; losses = all three below "
             "base. Columns: input | base | T0 | T2 | T3 | reference (when the benchmark has one). The line under each "
             "column is that model's judge scores for this item.\n")
    for g in ("A", "B", "C"):
        for kind in ("win", "loss"):
            lst = grid_md.get((g, kind), [])
            L.append(f"\n### {GROUP_LABEL[g]}: clearest {kind}s\n")
            if not lst:
                L.append("_none: no item where all three trained models " + ("beat" if kind == "win" else "fell below") + " base_\n")
            for fn, e in lst:
                L.append(f"**{BENCH_LABEL[e.bench]} · {e.subcategory}** (`{e.uid}`, mean Δ {e.mean_delta:+.2f})\n\n![]({fn})\n")
    cj = os.path.join(RES, "conclusions.json")
    if os.path.exists(cj):
        C = json.load(open(cj))
        res = lambda t: resolve(t, gs, pt, main_t, ag)
        L.append("\n## Conclusions\n")
        for g in C.get("groups", []):
            v = vd[vd.group == g["group"]].verdict
            L.append(f"### {GROUP_LABEL[g['group']]}: {v.iloc[0] if len(v) else ''}\n")
            L += [res(p) + "\n" for p in g["paras"]]
        L.append("### Main caveats\n")
        L += ["- " + res(c) for c in C.get("caveats", [])]
        L.append("\n### Next steps\n")
        L += ["- " + res(c) for c in C.get("next", [])]
    open(os.path.join(RES, "REPORT.md"), "w").write("\n".join(L) + "\n")
    print("REPORT.md written;", len(sel), "grids")


if __name__ == "__main__":
    main()
