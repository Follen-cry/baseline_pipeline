#!/usr/bin/env python
"""Judge rows -> per-item scores, benchmark table and coverage check (CPU only, seconds; env: internvlu).

Systems: config.models + config.references, then any other <model>.jsonl found under results/judge_raw (a new
checkpoint appears automatically once judged). `gt` is image-only and never scored.
Per-item metrics (each benchmark's official per-sample rule, as in v2_suite_1's analyze.py):
  phyeditbench  4 dimensions (1-10) and overall = 0.2 consistency + 0.3 instruction_following
                + 0.4 physical_plausibility + 0.1 image_quality (needs all 4)
  picabench     acc = share of the item's QA pairs answered correctly (unparsed answer = incorrect, official);
                needs every QA pair judged
  risebench     score (1-5) and complete (0/1, RISE "accuracy"); a row whose 3 judge calls did not parse = missing
  imgedit_basic score (1-5, mean of the rubric's sub-scores)
Writes results/:
  scores_item.csv  uid, system, bench, metric, value
  table.csv        system x benchmark native aggregates + Overall (0-1)
  coverage.csv     system x benchmark: items, images present, primary scores present, parse failures kept
  missing.csv      one row per (system, uid) without an image or primary score (empty when complete)
Overall = mean over all 200 items of the primary metric rescaled to 0-1 of its range (PhyEditBench 1-10,
PICABench 0-1, RISE 1-5, ImgEdit 1-5), i.e. benchmarks weighted by item count 60:60:60:20. It is only
reported for a system that has every primary score.
"""
import collections
import json
import math
import os

import pandas as pd

from common import BENCHES, JUDGE_RAW, OUT, PHY_W, PRIMARY, RES, manifest, systems


def read_jsonl(p):
    rows = {}
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            rows[r["key"]] = r  # de-duplicate by unit key, last write wins
    return list(rows.values())


def all_systems():
    s = systems()
    found = {f[:-6] for b in BENCHES if os.path.isdir(os.path.join(JUDGE_RAW, b))
             for f in os.listdir(os.path.join(JUDGE_RAW, b)) if f.endswith(".jsonl")}
    return s + sorted(found - set(s) - {"gt"})


def item_scores(sysname, man):
    recs, fails = [], collections.Counter()
    n_qa = {r["uid"]: len(r["judge"]["annotated_qa_pairs"]) for r in man if r["bench"] == "picabench"}

    rows = read_jsonl(f"{JUDGE_RAW}/phyeditbench/{sysname}.jsonl")
    per = collections.defaultdict(dict)
    for r in rows:
        fails["phyeditbench"] += not r.get("parse_ok")
        if r.get("score") is not None:
            per[r["uid"]][r["dimension"]] = r["score"]
    for uid, d in per.items():
        recs += [(uid, "phyeditbench", k, v) for k, v in d.items()]
        if len(d) == 4:
            recs.append((uid, "phyeditbench", "overall", sum(PHY_W[k] * d[k] for k in PHY_W)))

    per = collections.defaultdict(list)
    for r in read_jsonl(f"{JUDGE_RAW}/picabench/{sysname}.jsonl"):
        fails["picabench"] += not r.get("parse_ok")
        per[r["uid"]].append(bool(r.get("is_correct")))
    recs += [(uid, "picabench", "acc", sum(v) / len(v)) for uid, v in per.items() if len(v) == n_qa[uid]]

    for r in read_jsonl(f"{JUDGE_RAW}/risebench/{sysname}.jsonl"):
        fails["risebench"] += not r.get("parse_ok")
        s = r.get("scores") or {}
        for k in ("score", "complete", "Reasoning", "ApprConsistency", "VisualPlausibility"):
            if s.get(k) is not None and not (isinstance(s[k], float) and math.isnan(s[k])):
                recs.append((r["uid"], "risebench", k, float(s[k])))

    for r in read_jsonl(f"{JUDGE_RAW}/imgedit_basic/{sysname}.jsonl"):
        fails["imgedit_basic"] += not r.get("parse_ok")
        if r.get("score") is not None:
            recs.append((r["uid"], "imgedit_basic", "score", float(r["score"])))
    df = pd.DataFrame(recs, columns=["uid", "bench", "metric", "value"])
    df.insert(1, "system", sysname)
    return df, fails


def main():
    man = manifest()
    mdf = pd.DataFrame(man).set_index("uid")
    uids = set(mdf.index)
    frames, cov, missing, table = [], [], [], []
    for s in all_systems():
        df, fails = item_scores(s, man)
        df = df[df.uid.isin(uids)]
        frames.append(df)
        for bench in BENCHES:
            metric, lo, hi = PRIMARY[bench]
            bu = list(mdf.index[mdf.bench == bench])
            have = set(df[(df.bench == bench) & (df.metric == metric)].uid)
            imgs = {u for u in bu if os.path.exists(os.path.join(OUT, s, mdf.loc[u, "out_rel"]))}
            cov.append(dict(system=s, bench=bench, items=len(bu), images=len(imgs), scored=len(have & set(bu)),
                            parse_fail_rows_kept=fails[bench]))
            for u in bu:
                if u not in imgs or u not in have:
                    missing.append(dict(system=s, uid=u, bench=bench, image=u in imgs, primary_score=u in have))

        def m(bench, metric):
            v = df[(df.bench == bench) & (df.metric == metric)].value
            return v.mean() if len(v) else float("nan")
        row = {"system": s,
               # official PhyEditBench aggregate: per-dimension means, then the weighted sum
               "PhyEditBench overall (1-10)": sum(PHY_W[k] * m("phyeditbench", k) for k in PHY_W),
               "PICABench Acc (%)": 100 * m("picabench", "acc"),
               "RISEBench Score (1-5)": m("risebench", "score"),
               "RISEBench Acc (%)": 100 * m("risebench", "complete"),
               "ImgEdit (1-5)": m("imgedit_basic", "score")}
        norm = [(v - PRIMARY[b][1]) / (PRIMARY[b][2] - PRIMARY[b][1])
                for b in BENCHES for v in df[(df.bench == b) & (df.metric == PRIMARY[b][0])].value]
        row["Overall (0-1)"] = sum(norm) / len(norm) if len(norm) == len(uids) else float("nan")
        row["n_scored"] = len(norm)
        table.append(row)

    os.makedirs(RES, exist_ok=True)
    pd.concat(frames).to_csv(os.path.join(RES, "scores_item.csv"), index=False)
    pd.DataFrame(table).to_csv(os.path.join(RES, "table.csv"), index=False)
    pd.DataFrame(cov).to_csv(os.path.join(RES, "coverage.csv"), index=False)
    pd.DataFrame(missing, columns=["system", "uid", "bench", "image", "primary_score"]).to_csv(
        os.path.join(RES, "missing.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(table).round(3).to_string(index=False))
        print(pd.DataFrame(cov).to_string(index=False))
    print(f"missing (system, uid) pairs: {len(missing)}")
    for r in missing:
        print("  MISSING", r)


if __name__ == "__main__":
    main()
