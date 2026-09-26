#!/usr/bin/env python3
"""Pick a small, fixed validation subset of the T0-T4 eval split (for in-training eval).

Input : datasets/temporal_ssl/settings/<run>/T{0..4}_eval.jsonl (derive_settings.py)
Output: datasets/temporal_ssl/settings/<run>/T{0..4}_evalmini.jsonl + evalmini_ids.json
        meta/<run>/T{0..4}_evalmini_meta.json

The same windows are used in every setting, so the T0-T4 validation curves are comparable: pick
`--per-source` windows per source (default 13 -> 78 rows), stratified by GAP within each source
(largest-remainder allocation, seeded), then keep exactly those windows' rows of each T*_eval.
Rows are copied unchanged, so prompts, answers, cond_image and the T2 / T4-B missing position
and shuffle are the ones derive_settings.py fixed.

T4 eval rows are all T4-B; a T4 model's T4-A validation is T0_evalmini.

Usage:
    python make_eval_mini.py [--run main] [--per-source 13]
"""
import argparse, collections, json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.normpath(os.path.join(HERE, "../.."))
SETS = os.path.join(V2, "datasets", "temporal_ssl", "settings")
META = os.path.join(V2, "meta")
SEED = 42
SETTINGS = ("T0", "T1", "T2", "T3", "T4")
MAX_DYNAMIC_PATCH = 3


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def allocate(counts, n):
    """Split n over strata proportionally to counts (largest remainder, ties by stratum key)."""
    total = sum(counts.values())
    quota = {k: n * c / total for k, c in counts.items()}
    alloc = {k: min(counts[k], int(q)) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - int(quota[k])), str(k))):
        if sum(alloc.values()) >= n:
            break
        if alloc[k] < counts[k]:
            alloc[k] += 1
    return alloc


def pick_windows(rows, per_source):
    by_src = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_src[r["source"]][r["gap_s"]].append(r["window_id"])
    picked = []
    for src in sorted(by_src):
        strata = by_src[src]
        alloc = allocate({g: len(v) for g, v in strata.items()}, per_source)
        for g in sorted(strata):
            ids = sorted(strata[g])
            random.Random(f"{SEED}:evalmini:{src}:{g}").shuffle(ids)
            picked += ids[: alloc[g]]
    return sorted(picked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--per-source", type=int, default=13)
    a = ap.parse_args()
    sdir, mdir = os.path.join(SETS, a.run), os.path.join(META, a.run)
    ids = pick_windows(read_jsonl(os.path.join(sdir, "T0_eval.jsonl")), a.per_source)
    keep = set(ids)
    report = {"run": a.run, "seed": SEED, "per_source": a.per_source, "windows": len(ids), "settings": {}}
    for st in SETTINGS:
        rows = [r for r in read_jsonl(os.path.join(sdir, f"{st}_eval.jsonl")) if r["window_id"] in keep]
        if len(rows) != len(ids):
            raise SystemExit(f"{st}_eval: {len(rows)} rows for {len(ids)} windows")
        path = os.path.join(sdir, f"{st}_evalmini.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        name = f"v2_{a.run}_{st}_evalmini"
        with open(os.path.join(mdir, f"{st}_evalmini_meta.json"), "w") as f:
            json.dump({name: {"root": "/", "annotation": path, "data_augment": False,
                              "max_dynamic_patch": MAX_DYNAMIC_PATCH, "repeat_time": 1,
                              "length": len(rows), "task_type": "imgen"}}, f, indent=2)
        c = collections.Counter
        report["settings"][st] = {
            "rows": len(rows), "by_source": dict(sorted(c(r["source"] for r in rows).items())),
            "gap_s": {str(k): v for k, v in sorted(c(r["gap_s"] for r in rows).items())},
            "target": dict(sorted(c(r["layout"]["target"] for r in rows).items())),
            "stalled": sum(r["stalled"] for r in rows)}
        print(f"[evalmini] {a.run} {st}: {len(rows)} rows -> {path}")
    with open(os.path.join(sdir, "evalmini_ids.json"), "w") as f:
        json.dump({**report, "window_ids": ids}, f, indent=1)
    print(json.dumps(report["settings"], indent=1))


if __name__ == "__main__":
    main()
