#!/usr/bin/env python
"""One-off import of existing v2_suite_1 results for the 200 subset uids (already run; no inference, no judging).

For every system in config.provenance.systems (base, T0, T2, T3, copy, random, gt):
  images      source_outputs/<system>/<out_rel>          -> OUT/<system>/<out_rel>   (real copies, not links)
  judge rows  source_judge_raw/<bench>/<system>.jsonl     -> results/judge_raw/<judge>/<bench>/<system>.jsonl
              (rows whose uid is in the subset; de-duplicated by unit key, last write wins, as v2_suite_1's analyze.py)
`gt` is an image-only reference (PhyEditBench GT state, a few RISE reference images); it is shown, not scored.
Writes results/import_log.csv (per system x bench: images copied / missing, judge rows kept).
"""
import csv
import json
import os
import shutil

from common import BENCHES, CFG, JUDGE_RAW, OUT, RES, manifest

P = CFG["provenance"]
rows = manifest()
uids = {r["uid"] for r in rows}
log = []
for sysname in P["systems"]:
    for bench in BENCHES:
        br = [r for r in rows if r["bench"] == bench]
        n_img = n_miss = 0
        for r in br:
            src = os.path.join(P["source_outputs"], sysname, r["out_rel"])
            dst = os.path.join(OUT, sysname, r["out_rel"])
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                n_img += 1
            else:
                n_miss += 1
        n_rows = 0
        jsrc = os.path.join(P["source_judge_raw"], bench, f"{sysname}.jsonl")
        if os.path.exists(jsrc):
            keep = {}
            for l in open(jsrc):
                j = json.loads(l)
                if j["uid"] in uids:
                    keep[j["key"]] = j
            os.makedirs(os.path.join(JUDGE_RAW, bench), exist_ok=True)
            with open(os.path.join(JUDGE_RAW, bench, f"{sysname}.jsonl"), "w") as f:
                for j in keep.values():
                    f.write(json.dumps(j, ensure_ascii=False) + "\n")
            n_rows = len(keep)
        log.append(dict(system=sysname, bench=bench, items=len(br), images_copied=n_img, images_missing=n_miss,
                        judge_rows=n_rows, judge_source=jsrc if os.path.exists(jsrc) else ""))
        print(log[-1])
with open(os.path.join(RES, "import_log.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(log[0]))
    w.writeheader()
    w.writerows(log)
