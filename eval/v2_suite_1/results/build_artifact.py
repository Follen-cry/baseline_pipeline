#!/usr/bin/env python
"""Build the shareable single-file HTML report (results/eval_report.html) from the result CSVs.

Every number on the page is read from a CSV in this folder (or from ../pipeline/config.json for
the inference settings). Images: input / base / T0 / T2 / T3 / reference, downscaled to 384 px on the long
side, WebP, embedded as data URIs. If the page would exceed MAX_BYTES, the number of examples per category
is reduced (every category keeps at least one).

  python results/build_artifact.py      (env: internvlu)
"""
import base64
import html
import io
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

RES = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(RES)
SUITE = os.path.join(EVAL, "pipeline")
OUT = os.path.join(EVAL, "outputs")
MODELS = ["base", "T0", "T2", "T3"]
MAX_BYTES = 15_300_000  # page limit is 16 MB
LONG_SIDE = 384


def csv(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def records(df):
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))


_cache = {}


def img_uri(path, q=62, side=LONG_SIDE):
    if (path, side) in _cache:
        return _cache[(path, side)]
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=q, method=6)
    _cache[(path, side)] = "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()
    return _cache[(path, side)]


def ref_path(r):
    j = r["judge"]
    if r["bench"] == "phyeditbench":
        return j["ref"]
    if r["bench"] == "magicbrush":
        return j["gt"]
    if r["bench"] == "risebench" and j.get("reference_img"):
        return j["reference_img"]
    return None


def choose_examples(ex, per_cat):
    """Per (bench, subcategory): the `per_cat` highest and lowest mean-delta items (clearest wins / losses)."""
    out = []
    for (b, sc), g in ex.groupby(["bench", "subcategory"]):
        g = g.sort_values(["mean_delta", "uid"], ascending=[False, True])
        top, bot = g.head(per_cat), g.tail(per_cat).iloc[::-1]
        bot = bot[~bot.uid.isin(top.uid)]
        out += [top.assign(kind=np.where(top.tag == "win", "win", "best")),
                bot.assign(kind=np.where(bot.tag == "loss", "loss", "worst"))]
    return pd.concat(out)


def item_detail(items, uid):
    s = items[items.uid == uid]
    return {mdl: {r.metric: (None if pd.isna(r.value) else round(float(r.value), 3))
                  for r in s[s.model == mdl].itertuples()} for mdl in MODELS}


def diag_data():
    """Diagnosis section: CSVs under results/diagnosis/ + diagnosis.json text + embedded example images."""
    d = os.path.join(RES, "diagnosis")
    if not os.path.isdir(d):
        return None
    rd = lambda n: records(pd.read_csv(os.path.join(d, n))) if os.path.exists(os.path.join(d, n)) else []
    ex = json.load(open(os.path.join(d, "diag_examples.json"))) if os.path.exists(os.path.join(d, "diag_examples.json")) else {}
    for k in ex:
        for e in ex[k]:
            e["imgs"] = {c: img_uri(p, q=55, side=320) for c, p in e["imgs"].items()}
    txt = json.load(open(os.path.join(d, "diagnosis.json"))) if os.path.exists(os.path.join(d, "diagnosis.json")) else {}
    return dict(c1=rd("cause1_baselines.csv"), c1p=rd("cause1_penalty.csv"), c2=rd("cause2_results.csv"),
                c2m=rd("cause2_means.csv"), c3=rd("cause3_modes.csv"), c3t=rd("cause3_target_change.csv"),
                c3r=rd("cause3_code_review.csv"), ex=ex, ranking=txt.get("ranking", []), summary=txt.get("summary", []),
                sections=txt.get("sections", {}), c2e=rd("cause2_ft_effect.csv"), c2d=rd("cause2_ftdata_editmag.csv"),
                ctl=rd("control_base_ft.csv"), ctn=rd("control_null_pixels.csv"), wd=rd("weight_diff.csv"))


def build(per_cat):
    man = {json.loads(l)["uid"]: json.loads(l) for l in open(os.path.join(SUITE, "manifest.jsonl"))}
    items = csv("scores_item.csv")
    ex = csv("examples.csv")
    sel = choose_examples(ex, per_cat)
    exs = []
    for e in sel.itertuples():
        r = man[e.uid]
        imgs = {"input": img_uri(r["inputs"][0])}
        for mdl in MODELS:
            imgs[mdl] = img_uri(os.path.join(OUT, mdl, r["out_rel"]))
        rp = ref_path(r)
        if rp:
            imgs["reference"] = img_uri(rp)
        exs.append(dict(uid=e.uid, bench=e.bench, group=e.group, category=e.category, subcategory=e.subcategory,
                        etype=e.etype, kind=e.kind, mean_delta=round(float(e.mean_delta), 4),
                        instruction=r["instruction"], imgs=imgs, scores=item_detail(items, e.uid)))
    cfg = json.load(open(os.path.join(SUITE, "config.json")))
    data = dict(
        config=cfg,
        suite_counts=records(csv("suite_counts.csv")),
        power=records(csv("power_planned.csv")),
        main=records(csv("table_main.csv")),
        category=records(csv("table_category.csv")),
        phytype=records(csv("table_phyedit_type.csv")),
        anti=records(csv("table_anti.csv")),
        paired=records(csv("paired.csv")),
        groups=records(csv("group_summary.csv")),
        trend=records(csv("trend.csv")),
        parse=records(csv("parse_stats.csv")),
        agree=records(csv("judge_agreement.csv")),
        verdicts=records(csv("verdicts.csv")),
        skipped=records(csv("benchmarks_considered.csv")),
        conclusions=json.load(open(os.path.join(RES, "conclusions.json"))) if os.path.exists(
            os.path.join(RES, "conclusions.json")) else {},
        examples=exs,
        diag=diag_data(),
    )
    page = open(os.path.join(RES, "artifact_template.html")).read()
    return page.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))


def main():
    for per_cat in (3, 2, 1):
        page = build(per_cat)
        if len(page.encode()) <= MAX_BYTES:
            break
    path = os.path.join(RES, "eval_report.html")
    open(path, "w").write(page)
    print(f"{path}: {len(page.encode()) / 1e6:.2f} MB, {per_cat} example(s) per side per category")


if __name__ == "__main__":
    main()
