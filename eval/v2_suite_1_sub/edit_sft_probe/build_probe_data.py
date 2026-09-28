#!/usr/bin/env python
"""Lightweight edit-SFT probe data: a small instruction-editing SFT set, identical for base/T0/T2/T3,
paired with v2_suite_1_sub's 200-item manifest (not the full 1,210-item v2_suite_1 suite).

Modelled on ../../v2_suite_1/diagnosis/edit_sft/build_edit_sft.py, but lighter:
  - ~800 rows instead of 1,280 (400 MagicBrush *train* turns + 400 PICA-100K pairs, same 1:1
    ratio as the original 640/640 split).
  - Overlap check runs against THIS suite's ../manifest.jsonl (200 items) instead of v2_suite_1's
    1,210-item pipeline/manifest.jsonl. v2_suite_1_sub's manifest happens to contain no
    bench=="magicbrush" rows (that bench wasn't sampled into the 200-item subset), so the
    MagicBrush-img_id exclusion below is a no-op here; it is kept for parity/safety in case the
    subset is ever resampled to include one.
  - A different fixed seed (20260929, one day after the subset's own 20260928) so this is an
    independent draw from the original 1,280-row probe, not a truncation of it.

Same source data as the original probe (osunlp/MagicBrush official train split; Andrew613/PICA-100K),
already downloaded to RAW by the original probe's setup.

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
SUITE = os.path.abspath(os.path.join(HERE, ".."))  # v2_suite_1_sub/
RAW = "/scratch/network/ssd/junlin/ssl_eval/diag_sft/raw"  # reused, unchanged, from the original probe
DATA = "/scratch/network/ssd/junlin/ssl_eval/v2_eval_suite_1_sub/edit_sft_probe/data"
N_PER_SOURCE = 400
SEED = 20260929
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
    json.dump({"edit_sft_probe_sub": {"root": "/", "annotation": os.path.join(DATA, "train.jsonl"),
                                       "data_augment": False, "max_dynamic_patch": 4, "repeat_time": 1,
                                       "length": len(rows), "task_type": "imgen"}},
              open(os.path.join(DATA, "meta.json"), "w"), indent=2)
    json.dump(dict(rows=len(rows), per_source={s: sum(r["id"].startswith(p) for r in rows) for s, p in (("magicbrush", "mb_"), ("pica100k", "pica_"))},
                   eval_images_checked=len(ev), hamming_max=HAMMING_MAX, dropped=dropped),
              open(os.path.join(DATA, "overlap_report.json"), "w"), indent=2)
    print(len(rows), "rows;", len(dropped), "dropped for overlap")


if __name__ == "__main__":
    main()
