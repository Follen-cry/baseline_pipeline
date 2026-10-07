#!/usr/bin/env python
"""Build the shareable page results/report.html from the result files (CPU only; env: internvlu).

Every number on the page is read here from results/table.csv, results/scores_item.csv, results/coverage.csv,
results/scoring_rules.json, manifest.jsonl and config.json (run score.py and scoring_rules.py first); scripts/report_template.html holds only layout and code.
Gallery: config.gallery.per_bench datapoints per benchmark, drawn with random.Random(f"{seed}:gallery:{bench}")
from the manifest; images are embedded as JPEG data URIs (config.gallery.thumb_side, jpeg_quality).
Systems shown: config.report_models (rows + gallery columns) and config.references (table reference rows,
per-item reference scores under each grid); `gt` images where OUT/gt/<out_rel> exists.
"""
import base64
import io
import json
import math
import os
import random

import pandas as pd
from PIL import Image

from common import BENCHES, CFG, OUT, PHY_W, RES, manifest

HERE = os.path.dirname(os.path.abspath(__file__))
G = CFG["gallery"]
MODELS = CFG["report_models"]
REFS = CFG["references"]
LABEL = {"phyeditbench": "PhyEditBench", "picabench": "PICABench", "risebench": "RISEBench", "imgedit_basic": "ImgEdit"}


def img_uri(p):
    im = Image.open(p).convert("RGB")
    im.thumbnail((G["thumb_side"], G["thumb_side"]), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=G["jpeg_quality"], optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def clean(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def main():
    rows = manifest()
    table = pd.read_csv(os.path.join(RES, "table.csv"))
    cov = pd.read_csv(os.path.join(RES, "coverage.csv"))
    sc = pd.read_csv(os.path.join(RES, "scores_item.csv"))
    missing = pd.read_csv(os.path.join(RES, "missing.csv"))
    S = {(r.uid, r.system, r.metric): r.value for r in sc.itertuples()}

    def item_score(r, s):
        u, b = r["uid"], r["bench"]
        if b == "phyeditbench":
            d = {k: S.get((u, s, k)) for k in PHY_W}
            return dict(v=S.get((u, s, "overall")), dims={k: clean(v) for k, v in d.items()})
        if b == "picabench":
            n = len(r["judge"]["annotated_qa_pairs"])
            a = S.get((u, s, "acc"))
            return dict(v=a, n_qa=n, n_ok=None if a is None else round(a * n))
        if b == "risebench":
            return dict(v=S.get((u, s, "score")), complete=clean(S.get((u, s, "complete"))),
                        dims={k: clean(S.get((u, s, k))) for k in ("Reasoning", "ApprConsistency", "VisualPlausibility")})
        return dict(v=S.get((u, s, "score")))

    gallery = []
    for b in BENCHES:
        uids = sorted(r["uid"] for r in rows if r["bench"] == b)
        pick = set(random.Random(f"{G['seed']}:gallery:{b}").sample(uids, min(G["per_bench"][b], len(uids))))
        for r in rows:
            if r["uid"] not in pick:
                continue
            j = r["judge"]
            gt_img = os.path.join(OUT, "gt", r["out_rel"])
            ref = dict(img=img_uri(gt_img) if os.path.exists(gt_img) else None)
            if b == "phyeditbench":
                ref.update(kind="GT next state", text=j.get("explain"))
            elif b == "risebench":
                ref.update(kind="Reference (text)", text=j.get("reference"))
            elif b == "picabench":
                ref.update(kind="Judge checklist", qa=[[q["question"], q["answer"]] for q in j["annotated_qa_pairs"]])
            gallery.append(dict(
                uid=r["uid"], bench=b, category=r["category"], subcategory=r["subcategory"], etype=r["etype"],
                instruction=r["instruction"], input=img_uri(r["inputs"][0]), ref=ref,
                outs={s: img_uri(os.path.join(OUT, s, r["out_rel"])) for s in MODELS},
                scores={s: {k: clean(v) if not isinstance(v, dict) else v for k, v in item_score(r, s).items()}
                        for s in MODELS + REFS}))

    counts = {b: sum(r["bench"] == b for r in rows) for b in BENCHES}
    pools = {p: dict(CFG["sampling"]["pools"][p], n=n) for p, n in CFG["sampling"]["counts"].items()}
    data = dict(
        models=MODELS, refs=REFS, labels=LABEL, counts=counts, n_items=len(rows), pools=pools,
        seed=CFG["sampling"]["seed"], gallery_seed=G["seed"], judge=CFG["judge"]["served_name"],
        table=[{k: clean(v) for k, v in rec.items()} for rec in table.to_dict("records")],
        coverage=[{k: clean(v) for k, v in rec.items()} for rec in cov.to_dict("records")],
        n_missing=len(missing), missing=missing.to_dict("records"),
        gallery=gallery,
        rules=json.load(open(os.path.join(RES, "scoring_rules.json"))))
    html = open(os.path.join(HERE, "report_template.html")).read()
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    out = os.path.join(RES, "report.html")
    open(out, "w").write(html)
    per = ", ".join(f"{LABEL[b]} {sum(g['bench'] == b for g in gallery)}" for b in BENCHES)
    print(f"wrote {out}: {len(html) / 1e6:.1f} MB, {len(gallery)} gallery items "
          f"({per})")


if __name__ == "__main__":
    main()
