#!/usr/bin/env python
"""Build the compact video-SSL physics-editing suite manifest.

Stratified random selection (fixed seed) from PhyEditBench, PICABench, RISEBench,
ImgEdit (Basic + UGE) and MagicBrush; see ../SUITE.md for the rationale. Every
selected input / reference image is copied to STAGE (network storage, readable
from all nodes) and the manifest points there.

Output: pipeline/manifest.jsonl, one row per generation task:
  uid, bench, orig_id, category, subcategory, group, instruction,
  inputs [abs paths], out_rel (path under outputs/<model>/), judge {...}
Run (env: internvlu, needs pandas):  python build_manifest.py
"""
import collections
import glob
import io
import json
import math
import os
import random
import shutil

import pandas as pd
from PIL import Image

SEED = 20260927
HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = "/scratch/network/ssd/junlin/ssl_eval/staged"
SRC = "/scratch/network/ssd/junlin/ssl_eval/src"
PHY = "/scratch/local/ssd/junlin/data/PhyEditBench/bench"
PICA = "/scratch/network/ssd/junlin/ssl_eval/data/PICABench/picabench.parquet"
RISE = "/scratch/local/ssd/junlin/data/RISEBench"
IMGE = "/scratch/local/ssd/junlin/data/ImgEdit/Benchmark"
MB = "/scratch/local/ssd/junlin/data/MagicBrush/test"

# ---- allocation (see SUITE.md) ----
PHY_TRAJ_PER_SUB = 8          # x12 subclasses x5 types = 480 tasks
PICA_PER_LAW = {"Causality": 45, "Deformation": 45, "Global": 45, "Local": 45,
                "Light_Propagation": 20, "Light_Source_Effects": 20, "Reflection": 20, "Refraction": 20}
RISE_SPATIAL_PER_SUB = 12     # 5 subtasks -> 60
RISE_LOGICAL_PER_SUB = 12     # 3 subtasks -> 36
IMGEDIT_PER_TYPE = 12         # 9 types -> 108
UGE_N = 16
MB_N = 40
TYPES = ["TypeA", "TypeB", "TypeC", "TypeD", "TypeE"]


def stage(src, rel):
    dst = os.path.join(STAGE, rel)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return dst


def _unused_sample(rng, xs, k):
    xs = sorted(xs, key=str)
    return xs if k >= len(xs) else sorted(rng.sample(xs, k), key=str)


def phyedit(rng, rows):
    import sys
    sys.path.insert(0, os.path.join(SRC, "PhyEditBench"))
    from gpt_eval import build_type_fields  # official A-E expansion, verbatim
    for meta in sorted(glob.glob(f"{PHY}/*/*/meta.json")):
        primary, sub = meta.split("/")[-3:-1]
        dps = json.load(open(meta))
        for dp in sorted(rng.sample(dps, min(PHY_TRAJ_PER_SUB, len(dps))), key=lambda d: int(d["id"])):
            for f in dp["frames"].values():
                stage(f"{PHY}/{primary}/{sub}/{f}", f"phyeditbench/bench/{primary}/{sub}/{f}")
            for t in TYPES:
                tf = build_type_fields(dp, t)
                rows.append(dict(
                    uid=f"phy/{primary}/{sub}/{t}/{dp['id']}", bench="phyeditbench", orig_id=dp["id"],
                    category=primary, subcategory=sub, etype=t, group="A",
                    instruction=tf["instruction"],
                    inputs=[f"{STAGE}/phyeditbench/bench/{primary}/{sub}/{tf['input_frame']}"],
                    out_rel=f"phyeditbench/{primary}/{sub}/{t}/{dp['id']}.png",
                    judge=dict(ref=f"{STAGE}/phyeditbench/bench/{primary}/{sub}/{tf['ref_frame']}",
                               explain=tf["explain"], invariants=tf["invariants"])))
    # Anti-Physics: all 35 (7 per rule type)
    anti = [json.loads(l) for l in open(f"{PHY}/anti-physic/meta.jsonl")]
    cl = {json.loads(l)["data_id"]: json.loads(l)["checklist"] for l in open(f"{PHY}/anti-physic/checklists.jsonl")}
    for a in anti:
        inp = stage(f"{PHY}/anti-physic/input_data/data_{a['data_id']}.png",
                    f"phyeditbench/bench/anti-physic/input_data/data_{a['data_id']}.png")
        rows.append(dict(
            uid=f"phyanti/{a['data_id']}", bench="phyedit_anti", orig_id=str(a["data_id"]),
            category="Anti-Physics", subcategory=a["data_type"], etype="anti", group="A",
            instruction=a["edit_prompt"], inputs=[inp],
            out_rel=f"phyeditbench/anti-physic/{a['data_id']}.png",
            judge=dict(checklist=cl[a["data_id"]], expected_phenomenon=a["expected_phenomenon"])))


def picabench(rng, rows):
    d = pd.read_parquet(PICA)
    d["index"] = range(len(d))  # prepare_meta_info.py uses the HF row order as `index`
    for law, k in PICA_PER_LAW.items():
        sub = d[d.physics_law == law]
        for idx in sorted(rng.sample(list(sub["index"]), k)):
            r = d.iloc[idx]
            p = f"{STAGE}/picabench/input_img/{idx}.jpg"
            if not os.path.exists(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                # identical to prepare_meta_info.save_input_image
                Image.open(io.BytesIO(r.input_image["bytes"])).convert("RGB").save(p, quality=95)
            qa = [dict(question=q["question"], answer=q["answer"], box={kk: float(v) for kk, v in q["box"].items()})
                  for q in r.annotated_qa_pairs]
            ea = r.edit_area
            if ea is None:  # upstream: falsy edit_area -> whole-image PSNR
                ea = []
            elif not isinstance(ea, str):
                ea = [{kk: (float(v) if kk not in ("id", "order") else (int(v) if kk == "order" else v))
                       for kk, v in e.items()} for e in ea]
            rows.append(dict(
                uid=f"pica/{idx}", bench="picabench", orig_id=str(idx),
                category=r.physics_category, subcategory=law, etype=r.edit_operation, group="A",
                instruction=r.superficial_prompt, inputs=[p], out_rel=f"picabench/{idx}.png",
                judge=dict(annotated_qa_pairs=qa, edit_area=ea, image_path=r.image_path)))


def risebench(rng, rows):
    data = json.load(open(f"{RISE}/datav2_total_w_subtask.json"))
    by = collections.defaultdict(list)
    for x in data:
        by[(x["category"], x["subtask"])].append(x)
    for (cat, st), xs in sorted(by.items()):
        if cat in ("temporal_reasoning", "causal_reasoning"):
            pick, grp = xs, "A"
        elif cat == "spatial_reasoning":
            pick, grp = sorted(rng.sample(xs, min(RISE_SPATIAL_PER_SUB, len(xs))), key=lambda x: x["index"]), "B"
        else:
            pick, grp = sorted(rng.sample(xs, min(RISE_LOGICAL_PER_SUB, len(xs))), key=lambda x: x["index"]), "C"
        for x in pick:
            j = {k: v for k, v in x.items() if not (isinstance(v, float) and math.isnan(v))}
            for k in ("image", "reference_img"):
                if j.get(k):
                    j[k] = stage(f"{RISE}/data/{j[k]}", f"risebench/data/{j[k]}")
            rows.append(dict(
                uid=f"rise/{x['index']}", bench="risebench", orig_id=x["index"],
                category=cat, subcategory=st, etype=cat, group=grp,
                instruction=x["instruction"], inputs=[j["image"]],
                out_rel=f"risebench/images/{cat}/{x['index']}.png", judge=j))


def imgedit(rng, rows):
    ann = json.load(open(f"{IMGE}/singleturn/singleturn.json"))
    by = collections.defaultdict(list)
    for k, v in ann.items():
        by[v["edit_type"]].append(k)
    for et, keys in sorted(by.items()):
        for k in sorted(rng.sample(sorted(keys), IMGEDIT_PER_TYPE), key=int):
            v = ann[k]
            inp = stage(f"{IMGE}/singleturn/{v['id']}", f"imgedit/singleturn/{v['id']}")
            rows.append(dict(uid=f"imgedit/{k}", bench="imgedit_basic", orig_id=k, category="basic",
                             subcategory=et, etype=et, group="C", instruction=v["prompt"], inputs=[inp],
                             out_rel=f"imgedit/basic/{k}.png", judge=dict(edit_type=et, id=v["id"])))
    uge = json.load(open(f"{SRC}/ImgEdit/Benchmark/UGE/UGE_edit.json"))
    for k in sorted(rng.sample(sorted(uge), UGE_N), key=int):
        v = uge[k]
        inp = stage(f"{IMGE}/hard/{v['id']}", f"imgedit/hard/{v['id']}")
        rows.append(dict(uid=f"uge/{k}", bench="imgedit_uge", orig_id=k, category="uge", subcategory="uge",
                         etype="uge", group="C", instruction=v["prompt"], inputs=[inp],
                         out_rel=f"imgedit/uge/{k}.png", judge=dict(id=v["id"])))


def magicbrush(rng, rows):
    sess = json.load(open(f"{MB}/edit_sessions.json"))
    for img_id in sorted(rng.sample(sorted(sess), MB_N)):
        t = sess[img_id][0]  # turn 1, single-turn protocol
        inp = stage(f"{MB}/images/{img_id}/{t['input']}", f"magicbrush/{img_id}/{t['input']}")
        gt = stage(f"{MB}/images/{img_id}/{t['output']}", f"magicbrush/{img_id}/{t['output']}")
        rows.append(dict(uid=f"mb/{img_id}", bench="magicbrush", orig_id=img_id, category="magicbrush",
                         subcategory="turn1", etype="turn1", group="C", instruction=t["instruction"],
                         inputs=[inp], out_rel=f"magicbrush/{img_id}_1.png", judge=dict(gt=gt)))


def main():
    rows = []
    # one independent RNG stream per source so changing one allocation never reshuffles another
    phyedit(random.Random(f"{SEED}:phy"), rows)
    picabench(random.Random(f"{SEED}:pica"), rows)
    risebench(random.Random(f"{SEED}:rise"), rows)
    imgedit(random.Random(f"{SEED}:imgedit"), rows)
    magicbrush(random.Random(f"{SEED}:mb"), rows)
    assert len({r["uid"] for r in rows}) == len(rows)
    for r in rows:
        for p in r["inputs"]:
            assert os.path.exists(p), p
    with open(os.path.join(HERE, "manifest.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter((r["group"], r["bench"]) for r in rows)
    for k in sorted(c):
        print(k, c[k])
    print("total", len(rows))


if __name__ == "__main__":
    main()
