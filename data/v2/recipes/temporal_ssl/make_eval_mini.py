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

Extra settings outside T0-T4 (e.g. T2.1, T5, T6) don't share T0's window ids -- they pick their
own windows independently from their own *_eval.jsonl (stratified the same way, by source x gap),
since they don't need to be comparable window-for-window with T0-T4. Pass them via --settings,
e.g. --settings T2.1 (run main) or --settings T5,T6 (run phystran_g1). With a small single-source
eval pool (e.g. T5/T6's 125 rows) and --per-source >= that pool size, the "mini" is the full eval
set -- there is nothing left to subsample.

Usage:
    python make_eval_mini.py [--run main] [--per-source 13]
    python make_eval_mini.py --run main --settings T2.1 --per-source 13
    python make_eval_mini.py --run phystran_g1 --settings T5,T6 --per-source 125
"""
import argparse, collections, json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.normpath(os.path.join(HERE, "../.."))
SETS = os.path.join(V2, "datasets", "temporal_ssl", "settings")
META = os.path.join(V2, "meta")
SEED = 42
SETTINGS = ("T0", "T1", "T2", "T3", "T4")  # the shared-window group
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


def write_mini(sdir, mdir, run, st, rows):
    path = os.path.join(sdir, f"{st}_evalmini.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    name = f"v2_{run}_{st}_evalmini"
    with open(os.path.join(mdir, f"{st}_evalmini_meta.json"), "w") as f:
        json.dump({name: {"root": "/", "annotation": path, "data_augment": False,
                          "max_dynamic_patch": MAX_DYNAMIC_PATCH, "repeat_time": 1,
                          "length": len(rows), "task_type": "imgen"}}, f, indent=2)
    c = collections.Counter
    stat = {"rows": len(rows), "by_source": dict(sorted(c(r["source"] for r in rows).items())),
            "gap_s": {str(k): v for k, v in sorted(c(r["gap_s"] for r in rows).items())},
            "target": dict(sorted(c(r["layout"]["target"] for r in rows).items())),
            "stalled": sum(r["stalled"] for r in rows)}
    print(f"[evalmini] {run} {st}: {len(rows)} rows -> {path}")
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--per-source", type=int, default=13)
    ap.add_argument("--settings", default=",".join(SETTINGS),
                    help="comma list; settings in T0-T4 share one window pick (T0_eval's), "
                         "others each pick their own windows from their own *_eval.jsonl")
    a = ap.parse_args()
    sdir, mdir = os.path.join(SETS, a.run), os.path.join(META, a.run)
    settings = a.settings.split(",")
    shared = [s for s in settings if s in SETTINGS]
    extra = [s for s in settings if s not in SETTINGS]
    report = {"run": a.run, "seed": SEED, "per_source": a.per_source, "settings": {}}

    if shared:
        ids = pick_windows(read_jsonl(os.path.join(sdir, "T0_eval.jsonl")), a.per_source)
        keep = set(ids)
        report["windows"] = len(ids)
        for st in shared:
            rows = [r for r in read_jsonl(os.path.join(sdir, f"{st}_eval.jsonl")) if r["window_id"] in keep]
            if len(rows) != len(ids):
                raise SystemExit(f"{st}_eval: {len(rows)} rows for {len(ids)} windows")
            report["settings"][st] = write_mini(sdir, mdir, a.run, st, rows)
        with open(os.path.join(sdir, "evalmini_ids.json"), "w") as f:
            json.dump({**report, "window_ids": ids}, f, indent=1)

    for st in extra:
        rows_all = read_jsonl(os.path.join(sdir, f"{st}_eval.jsonl"))
        ids = pick_windows(rows_all, a.per_source)
        keep = set(ids)
        rows = [r for r in rows_all if r["window_id"] in keep]
        report["settings"][st] = {**write_mini(sdir, mdir, a.run, st, rows), "windows": len(ids)}

    print(json.dumps(report["settings"], indent=1))


if __name__ == "__main__":
    main()
