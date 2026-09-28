#!/usr/bin/env python
"""Cause 3 evidence from the training data itself: how much does each v2 training target differ from
the frame the trainer feeds as the VAE pixel condition (cond_image = nearest shown frame,
derive_settings.py:23-24)? Uses the motion scores merge_pools.py stored per window
(fraction of pixels with |diff| > 25 after 256-px gray blur, window_motion.py).

T0/T1/T3/T4-A: target F3, cond F2 -> change_adjacent[2].
T2: target Fk, cond = nearest shown -> F0->F1 (k=1, cond F0), F1->F2 (k=2, cond F1), F2->F3 (k=3, cond F2);
    here the average over k of the three adjacent changes is reported.
Writes results/diagnosis/cause3_target_change.csv   (any python3)
"""
import collections
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BP = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # baseline_pipeline repo root
POOL = os.path.join(BP, "data", "v2", "datasets", "temporal_ssl", "pools", "main")
OUT = os.path.join(HERE, "..", "results", "diagnosis", "cause3_target_change.csv")


def summ(xs):
    xs = sorted(xs)
    n = len(xs)
    return dict(n=n, median=xs[n // 2], lt_1pct=sum(x < 0.01 for x in xs) / n,
                lt_5pct=sum(x < 0.05 for x in xs) / n, lt_10pct=sum(x < 0.10 for x in xs) / n)


rows = []
by = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(os.path.join(POOL, "*_train.jsonl"))):
    for l in open(f):
        w = json.loads(l)
        a = w["motion"]["change_adjacent"]
        for src in (w["source"], "ALL"):
            by[src]["T0_T3 (F2->F3)"].append(a[2])
            by[src]["T2 (avg over missing k)"] += a
            by[src]["stalled"].append(1.0 if w["motion"]["stalled"] else 0.0)
for src in sorted(by, key=lambda s: (s != "ALL", s)):
    for setting in ("T0_T3 (F2->F3)", "T2 (avg over missing k)"):
        rows.append(dict(source=src, setting=setting, **summ(by[src][setting]),
                         stalled_frac=sum(by[src]["stalled"]) / len(by[src]["stalled"])))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
