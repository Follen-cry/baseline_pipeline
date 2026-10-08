#!/usr/bin/env python
"""Pick the 5 PhysicTran38K datapoints (3 input frames F0-F2 -> target F3) and write datapoints.jsonl (one-off, already run).

Source: T5 evalmini (phystran_g1, 125 rows, gap 1.0 s). One row per distinct category so the 5 cover different physics.
Only the fields the experiment needs are kept; frame paths are checked to exist.
"""
import json
import os
import random

from common import CFG, HERE

sel = CFG["selection"]
rows = [json.loads(l) for l in open(CFG["source_jsonl"]) if l.strip()]
by_cat = {}
for r in rows:
    cat = r["clip_id"].split("__")[1]  # clip_id = physictran38k__<category>__<n>
    by_cat.setdefault(cat, []).append(r)
cats = sorted(by_cat)
picked = random.Random(f"{sel['seed']}:phystran_g1").sample(cats, sel["n"])
out = []
for c in picked:
    r = sorted(by_cat[c], key=lambda x: x["id"])[0]
    assert len(r["image"]) == 3 and all(os.path.exists(p) for p in r["image"] + [r["target_image"]]), r["id"]
    out.append({
        "id": r["id"], "category": c, "caption": r["caption"], "gap_s": r["gap_s"],
        "frames": r["image"], "target_image": r["target_image"],
        "prompt": r["conversations"][0]["value"],  # T5's exact user turn: caption + F0/F1/F2 <image> slots + GAP
    })
with open(os.path.join(HERE, CFG["datapoints"]), "w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(f"{len(rows)} rows, {len(cats)} categories -> {len(out)} datapoints:")
for o in out:
    print(" ", o["id"])
