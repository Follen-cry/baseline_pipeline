#!/usr/bin/env python3
"""Select 10K physics clips from Something-Something v2 and extract them from the local archive.

Inputs (see README.md):
  labels  : /scratch/network/ssd/junlin/raw/ssv2/labels/{train,validation}.json
  archive : /scratch/local/ssd/junlin/data/mvp/videos/20bn-something-something-v2-{00,01}
            (split gzip tar of all 220,847 .webm; torrnode11-local, so run on torrnode11)

Selection (seed 42): 24 physics templates (TEMPLATES below: the 15 core ones + 9 add-ons).
Target N_TOTAL clips, water-filled across templates: every template gets an equal share,
templates with fewer clips than their share contribute everything they have, and the
remainder is spread over the larger ones. Within a template, clips are drawn uniformly at
random from train + validation (the test split has no labels).

Output:
  <OUT>/selection_10k.jsonl : id, split, template, label, placeholders, video
  <OUT>/videos/<id>.webm    : extracted clips (12 fps, mostly 240p, as released)

Usage:
    python select_extract_ssv2.py --dry-run   # selection + manifest only
    python select_extract_ssv2.py             # + one streaming pass over the archive (resumable)
"""
import argparse, json, os, random, subprocess, sys, time
from collections import Counter, defaultdict

LABELS = "/scratch/network/ssd/junlin/raw/ssv2/labels"
ARCHIVE = ["/scratch/local/ssd/junlin/data/mvp/videos/20bn-something-something-v2-00",
           "/scratch/local/ssd/junlin/data/mvp/videos/20bn-something-something-v2-01"]
OUT = "/scratch/network/ssd/junlin/raw/ssv2"
N_TOTAL = 10_000
SEED = 42

# (group, template with [brackets] removed); groups are for reporting only
TEMPLATES = [
    # core 15
    ("fall", "Pushing something so that it falls off the table"),
    ("fall", "Something falling like a rock"),
    ("fall_air_resistance", "Something falling like a feather or paper"),
    ("fall", "Lifting something up completely, then letting it drop down"),
    ("fall_tilt", "Tilting something with something on it until it falls off"),
    ("fall_unsupported", "Putting something on the edge of something so it is not supported and falls down"),
    ("roll", "Letting something roll along a flat surface"),
    ("roll_slope", "Letting something roll down a slanted surface"),
    ("slide_slope", "Putting something that can't roll onto a slanted surface, so it slides down"),
    ("topple", "Poking something so that it falls over"),
    ("topple", "Putting something that cannot actually stand upright upright on the table, so it falls on its side"),
    ("spin", "Spinning something that quickly stops spinning"),
    ("projectile", "Throwing something in the air and letting it fall"),
    ("collision", "Moving something and something so they collide with each other"),
    ("fluid", "Pouring something into something"),
    # add-ons
    ("collision", "Throwing something against something"),
    ("spin", "Pushing something so it spins"),
    ("topple", "Tipping something over"),
    ("roll_slope", "Letting something roll up a slanted surface, so it rolls back down"),
    ("slide_tilt", "Lifting a surface with something on it until it starts sliding down"),
    ("stack_collapse", "Poking a stack of something so the stack collapses"),
    ("fluid", "Spilling something onto something"),
    ("fluid", "Pouring something out of something"),
    ("fall", "Dropping something onto something"),
]


def norm(t):
    return t.replace("[", "").replace("]", "")


def select():
    rows = []
    for split, fn in (("train", "train.json"), ("validation", "validation.json")):
        for r in json.load(open(os.path.join(LABELS, fn))):
            rows.append(dict(r, split=split))
    group = {t: g for g, t in TEMPLATES}
    by_t = defaultdict(list)
    for r in rows:
        t = norm(r["template"])
        if t in group:
            by_t[t].append(r)
    missing = [t for _, t in TEMPLATES if t not in by_t]
    assert not missing, f"templates not found in labels: {missing}"
    rng = random.Random(SEED)
    for t in by_t:
        by_t[t].sort(key=lambda r: int(r["id"]))
        rng.shuffle(by_t[t])
    # water-fill quotas
    quota = {t: 0 for t in by_t}
    left = N_TOTAL
    open_t = sorted(by_t, key=lambda t: len(by_t[t]))
    while left > 0 and open_t:
        share = max(1, left // len(open_t))
        for t in list(open_t):
            give = min(share, len(by_t[t]) - quota[t], left)
            quota[t] += give
            left -= give
            if quota[t] == len(by_t[t]):
                open_t.remove(t)
            if left == 0:
                break
    sel = []
    for g, t in TEMPLATES:
        for r in by_t[t][:quota[t]]:
            sel.append(dict(id=r["id"], split=r["split"], group=g, template=t,
                            template_raw=r["template"], label=r["label"],
                            placeholders=r.get("placeholders", []),
                            video=os.path.join(OUT, "videos", f"{r['id']}.webm")))
    return sel, quota, {t: len(v) for t, v in by_t.items()}


def extract(sel):
    vdir = os.path.join(OUT, "videos")
    os.makedirs(vdir, exist_ok=True)
    todo = [r for r in sel if not (os.path.exists(r["video"]) and os.path.getsize(r["video"]) > 0)]
    print(f"[extract] {len(sel) - len(todo)} already present, {len(todo)} to extract", flush=True)
    if not todo:
        return 0
    lst = os.path.join(OUT, "extract_list.txt")
    with open(lst, "w") as f:
        for r in todo:
            f.write(f"20bn-something-something-v2/{r['id']}.webm\n")
    t0 = time.time()
    # one streaming pass: cat the split parts | tar -xz only the listed members; strip the top dir
    cmd = (f"cat {' '.join(ARCHIVE)} | tar -xzf - -C {vdir} --strip-components=1 -T {lst}")
    res = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True)
    print(f"[extract] tar exit={res.returncode} in {time.time() - t0:.0f}s", flush=True)
    if res.stderr.strip():
        print(res.stderr.strip()[:1000], flush=True)
    miss = [r["id"] for r in sel if not os.path.exists(r["video"])]
    print(f"[extract] present {len(sel) - len(miss)}/{len(sel)}; missing {len(miss)} {miss[:10]}", flush=True)
    return 1 if miss else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    man = os.path.join(OUT, "selection_10k.jsonl")
    if os.path.exists(man):
        sel = [json.loads(l) for l in open(man)]
        print(f"[select] reusing {man}: {len(sel)} clips", flush=True)
    else:
        sel, quota, avail = select()
        with open(man, "w") as f:
            for r in sel:
                f.write(json.dumps(r) + "\n")
        for g, t in TEMPLATES:
            print(f"  {quota[t]:4d}/{avail[t]:5d}  [{g}] {t}", flush=True)
        print(f"[select] {len(sel)} clips over {len(TEMPLATES)} templates; "
              f"splits {dict(Counter(r['split'] for r in sel))} -> {man}", flush=True)
    if args.dry_run:
        return 0
    return extract(sel)


if __name__ == "__main__":
    sys.exit(main())
