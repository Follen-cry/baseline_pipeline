#!/usr/bin/env python3
"""Trim a source's oversampled selection to exactly N clips using clip_qc.py metrics.

1. Hard filters (all thresholds are CLI args, recorded in the output rows' `qc` field):
   decode_ok, duration >= --min-duration, n_cuts <= --max-cuts,
   cam_motion <= --max-cam (camera ~ fixed), obj_motion_max >= --min-obj (something moves).
   With --vlm (vlm_check.py output): also require shows_label and real_footage; within a
   group, clips without heavy_text are taken first.
2. Water-fill N over --group-key (equal share per group; groups with fewer passing clips give
   all they have), random within a group (seed 42).
3. Write <out> (manifest rows + qc metrics). With --delete-rest, delete the videos of the
   manifest rows that were not kept (the manifest/qc files are kept, so they can be re-fetched).

Usage:
    python trim_selection.py --manifest raw/<src>/selection_10k.jsonl --qc raw/<src>/qc.jsonl \
        --group-key class --out raw/<src>/selection_5k.jsonl [--n 5000] [--delete-rest]
"""
import argparse, json, os, random
from collections import Counter, defaultdict

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--qc", required=True)
    ap.add_argument("--group-key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--min-duration", type=float, default=2.9)
    ap.add_argument("--max-cuts", type=int, default=0)
    ap.add_argument("--max-cam", type=float, default=0.05)
    ap.add_argument("--min-obj", type=float, default=0.02)
    ap.add_argument("--vlm", help="vlm_check.py output; require shows_label + real_footage")
    ap.add_argument("--delete-rest", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.manifest)]
    qc = {}
    for l in open(a.qc):
        q = json.loads(l)
        qc[q["id"]] = q
    vlm = {json.loads(l)["id"]: json.loads(l) for l in open(a.vlm)} if a.vlm else None
    reasons, groups = Counter(), defaultdict(list)
    for r in rows:
        q = qc.get(r["id"])
        why = ("no_qc" if q is None else "decode" if not q["decode_ok"] or "cam_motion" not in q
               else "duration" if q["duration"] < a.min_duration
               else "cuts" if q["n_cuts"] > a.max_cuts
               else "camera" if q["cam_motion"] > a.max_cam
               else "static" if q["obj_motion_max"] < a.min_obj
               else None)
        if why is None and vlm is not None:
            v = vlm.get(r["id"], {})
            why = ("no_vlm" if "shows_label" not in v else "vlm_label" if not v["shows_label"]
                   else "vlm_not_real" if not v.get("real_footage", True) else None)
        reasons[why or "pass"] += 1
        if why is None:
            groups[r[a.group_key]].append(r)
    print(f"[trim] {len(rows)} rows:", dict(reasons))

    rng = random.Random(SEED)
    for g in groups:
        groups[g].sort(key=lambda r: r["id"])
        rng.shuffle(groups[g])
        if vlm is not None:                         # stable: random order within text / no-text
            groups[g].sort(key=lambda r: bool(vlm[r["id"]].get("heavy_text")))
    quota, left = {g: 0 for g in groups}, a.n
    open_g = sorted(groups)
    while left > 0 and open_g:
        share = max(1, left // len(open_g))
        for g in list(open_g):
            take = min(share, len(groups[g]) - quota[g], left)
            quota[g] += take
            left -= take
            if quota[g] == len(groups[g]):
                open_g.remove(g)
            if left == 0:
                break
    kept = [r for g in sorted(groups) for r in groups[g][:quota[g]]]
    if len(kept) < a.n:
        print(f"[trim] WARNING only {len(kept)} clips pass (< {a.n})")
    params = {k: getattr(a, k) for k in ("min_duration", "max_cuts", "max_cam", "min_obj")}
    with open(a.out + ".tmp", "w") as f:
        for r in kept:
            extra = {"vlm": {k: v for k, v in vlm[r["id"]].items() if k not in ("id", "raw")}} if vlm else {}
            f.write(json.dumps({**r, "qc": {**qc[r["id"]], "thresholds": params}, **extra}) + "\n")
    os.replace(a.out + ".tmp", a.out)
    print(f"[trim] kept {len(kept)} -> {a.out}; per group:", dict(sorted(quota.items())))

    if a.delete_rest:
        keep_ids = {r["id"] for r in kept}
        n = freed = 0
        for r in rows:
            if r["id"] not in keep_ids and os.path.exists(r["video"]):
                freed += os.path.getsize(r["video"])
                os.remove(r["video"])
                n += 1
        print(f"[trim] deleted {n} unselected videos ({freed / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
