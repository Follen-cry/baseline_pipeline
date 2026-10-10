#!/usr/bin/env python
"""Pick the PhysicTran38K datapoints (3 input frames F0-F2 -> target F3), copy their images here and write datapoints.jsonl.

Source: T5 evalmini (phystran_g1, 125 rows, gap 1.0 s). One row per distinct category (46 exist).
  - initial 5: the first run's pick (kept as rows 0-4 so existing outputs stay valid)
  - 35 more: random.Random(f"{seed}:ext35").sample(sorted(remaining categories), 35)
Images are copied to data/images/<id>/{F0,F1,F2,F3}.jpg and datapoints.jsonl stores those relative paths, so the experiment
no longer depends on /scratch/network/ssd/junlin/v2_frames. Re-running is idempotent.
"""
import json
import os
import random
import shutil

from common import CFG, HERE

sel = CFG["selection"]
rows = [json.loads(l) for l in open(CFG["source_jsonl"]) if l.strip()]
by_cat = {}
for r in rows:
    by_cat.setdefault(r["clip_id"].split("__")[1], []).append(r)  # clip_id = physictran38k__<category>__<n>
cats = sorted(by_cat)
first = random.Random(f"{sel['seed']}:phystran_g1").sample(cats, sel["n_initial"])
rest = random.Random(f"{sel['seed']}:ext35").sample([c for c in cats if c not in first], sel["n"] - sel["n_initial"])
picked = first + rest
out = []
for c in picked:
    r = sorted(by_cat[c], key=lambda x: x["id"])[0]
    srcs = r["image"] + [r["target_image"]]
    assert len(r["image"]) == 3 and all(os.path.exists(p) for p in srcs), r["id"]
    rel = [os.path.join(CFG["images_dir"], r["id"], f"F{i}.jpg") for i in range(4)]
    os.makedirs(os.path.join(HERE, CFG["images_dir"], r["id"]), exist_ok=True)
    for s, d in zip(srcs, rel):
        shutil.copyfile(s, os.path.join(HERE, d))
    out.append({
        "id": r["id"], "category": c, "caption": r["caption"], "gap_s": r["gap_s"],
        "frames": rel[:3], "target_image": rel[3], "source_frames": srcs,
        "prompt": r["conversations"][0]["value"],  # T5's exact user turn: caption + F0/F1/F2 <image> slots + GAP
    })
with open(os.path.join(HERE, CFG["datapoints"]), "w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(f"{len(rows)} rows, {len(cats)} categories -> {len(out)} datapoints ({len(first)} initial + {len(rest)} new):")
for o in out:
    print(" ", o["id"])
