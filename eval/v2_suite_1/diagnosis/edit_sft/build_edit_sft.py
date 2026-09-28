#!/usr/bin/env python
"""Cause-2 probe data: a small instruction-editing SFT set, identical for base/T0/T2/T3.

640 MagicBrush *train* turns (4 shards of osunlp/MagicBrush; COCO images, official train split,
disjoint from the test split the suite evaluates) + 640 PICA-100K pairs (1 shard of
Andrew613/PICA-100K; synthetic video-derived physics edits, superficial prompt) = 1,280 rows.

No-overlap check: 64-bit difference hash of every training source AND target image vs every
input/reference image in ../../pipeline/manifest.jsonl; rows within Hamming distance
<= 4 of any eval image are dropped (and counted in the report). MagicBrush img_ids used by the
eval's MagicBrush test slice are also excluded by id.

Writes rows in the trainer's edit format (see sft/v1/run_reasoning_edit_sft.sh):
  {"id", "task_type": "imgen", "image": [src], "target_image": tgt,
   "conversations": [{"from":"human","value":"<image>\n<instruction>"},
                     {"from":"gpt","value":"Here is the edited image: <img>"}]}
-> DATA/train.jsonl, DATA/meta.json, DATA/overlap_report.json      (env: internvlu)
"""
import io
import json
import os
import random

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.abspath(os.path.join(HERE, "..", "..", "pipeline"))
RAW = "/scratch/network/ssd/junlin/ssl_eval/diag_sft/raw"
DATA = "/scratch/network/ssd/junlin/ssl_eval/diag_sft/data"
N_PER_SOURCE = 640
SEED = 20260928
HAMMING_MAX = 4


def dhash(im, size=8):
    g = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def ham(a, b):
    return bin(a ^ b).count("1")


def eval_hashes():
    hs = []
    for l in open(os.path.join(SUITE, "manifest.jsonl")):
        r = json.loads(l)
        paths = list(r["inputs"])
        j = r["judge"]
        for k in ("ref", "gt", "reference_img"):
            if isinstance(j.get(k), str) and os.path.exists(j[k]):
                paths.append(j[k])
        for p in paths:
            hs.append((dhash(Image.open(p)), r["uid"]))
    return hs


def main():
    os.makedirs(os.path.join(DATA, "images"), exist_ok=True)
    rng = random.Random(SEED)
    ev = eval_hashes()
    eval_mb_ids = {json.loads(l)["orig_id"] for l in open(os.path.join(SUITE, "manifest.jsonl"))
                   if json.loads(l)["bench"] == "magicbrush"}
    cand = []
    for f in sorted(os.listdir(RAW)):
        t = pq.read_table(os.path.join(RAW, f)).to_pylist()
        if f.startswith("mb_"):
            cand += [("magicbrush", f"mb_{r['img_id']}_t{r['turn_index']}", r["source_img"]["bytes"],
                      r["target_img"]["bytes"], r["instruction"], r["img_id"]) for r in t]
        else:
            cand += [("pica100k", f"pica_{f[-13:-8]}_{i}", r["src_img"]["bytes"], r["tgt_img"]["bytes"],
                      r["superficial_prompt"], None) for i, r in enumerate(t)]
    rows, dropped = [], []
    for src_name in ("magicbrush", "pica100k"):
        pool = [c for c in cand if c[0] == src_name]
        rng.shuffle(pool)
        kept = 0
        for s, rid, sb, tb, ins, img_id in pool:
            if kept >= N_PER_SOURCE:
                break
            if img_id is not None and img_id in eval_mb_ids:
                dropped.append(dict(id=rid, reason="magicbrush img_id in eval"))
                continue
            si, ti = Image.open(io.BytesIO(sb)).convert("RGB"), Image.open(io.BytesIO(tb)).convert("RGB")
            hs_, ht_ = dhash(si), dhash(ti)
            hit = next(((u, d) for h, u in ev for d in [min(ham(h, hs_), ham(h, ht_))] if d <= HAMMING_MAX), None)
            if hit:
                dropped.append(dict(id=rid, reason=f"near-duplicate of eval {hit[0]} (hamming {hit[1]})"))
                continue
            sp, tp = os.path.join(DATA, "images", rid + "_src.png"), os.path.join(DATA, "images", rid + "_tgt.png")
            si.save(sp)
            ti.save(tp)
            rows.append({"id": rid, "task_type": "imgen", "image": [sp], "target_image": tp,
                         "conversations": [{"from": "human", "value": "<image>\n" + ins.strip()},
                                           {"from": "gpt", "value": "Here is the edited image: <img>"}]})
            kept += 1
    rng.shuffle(rows)
    with open(os.path.join(DATA, "train.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"edit_sft_probe": {"root": "/", "annotation": os.path.join(DATA, "train.jsonl"),
                                  "data_augment": False, "max_dynamic_patch": 4, "repeat_time": 1,
                                  "length": len(rows), "task_type": "imgen"}},
              open(os.path.join(DATA, "meta.json"), "w"), indent=2)
    json.dump(dict(rows=len(rows), per_source={s: sum(r["id"].startswith(p) for r in rows) for s, p in (("magicbrush", "mb_"), ("pica100k", "pica_"))},
                   eval_images_checked=len(ev), hamming_max=HAMMING_MAX, dropped=dropped),
              open(os.path.join(DATA, "overlap_report.json"), "w"), indent=2)
    print(len(rows), "rows;", len(dropped), "dropped for overlap")


if __name__ == "__main__":
    main()
