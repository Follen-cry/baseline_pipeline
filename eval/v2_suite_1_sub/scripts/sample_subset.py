#!/usr/bin/env python
"""Build manifest.jsonl: a fixed-seed random sample of v2_suite_1 datapoints (already run; kept for provenance).

Reads the source manifest as a data file (config.provenance.source_manifest); rows are copied verbatim,
so every field and every uid is identical to v2_suite_1. Pools and counts: config.sampling.
Each pool has its own RNG stream, random.Random(f"{seed}:{pool}"), over the pool's uids sorted.
Output rows keep the source manifest's order.
"""
import json
import random

from common import CFG, MANIFEST

S = CFG["sampling"]
src = [json.loads(l) for l in open(CFG["provenance"]["source_manifest"])]
keep = set()
for pool, n in S["counts"].items():
    spec = S["pools"][pool]
    uids = sorted(r["uid"] for r in src if r["bench"] == spec["bench"] and r["group"] == spec["group"])
    pick = random.Random(f"{S['seed']}:{pool}").sample(uids, n)
    keep |= set(pick)
    print(f"{pool:14s} pool {len(uids):4d} -> {n}")
rows = [r for r in src if r["uid"] in keep]
assert len(rows) == sum(S["counts"].values()) == len(keep)
with open(MANIFEST, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} rows -> {MANIFEST}")
