#!/usr/bin/env python
"""Build a self-contained page describing the Cause-2 edit-SFT probe dataset.

Reads the exact training rows (diag_sft/data/train.jsonl), the overlap report and the source parquet
shards (for PICA-100K law / operation / all three prompt tiers, and MagicBrush masks-free metadata),
computes statistics over all 1,280 rows, draws 48 examples with a fixed seed (24 MagicBrush: 8 per
turn; 24 PICA-100K: spread over edit operations) and writes one HTML file with everything embedded.

  python build_dataset_page.py      -> ../../results/diagnosis/ft_dataset_page.html   (env: internvlu)
"""
import base64
import collections
import io
import json
import os
import random

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
D = "/scratch/network/ssd/junlin/ssl_eval/diag_sft"
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "results", "diagnosis", "ft_dataset_page.html"))
SEED = "20260928:ft-dataset-page"
SIDE = 384
MB_PER_TURN = 8     # 3 turns -> 24 MagicBrush examples
N_PICA = 24         # spread round-robin over edit operations


def uri(im, q=70):
    im = im.copy()
    im.thumbnail((SIDE, SIDE), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "WEBP", quality=q, method=6)
    return "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()


def diff_map(s, t):
    """|target - source| as a heat map (0 -> dark, >= 0.35 -> bright), on the target's grid."""
    s = np.asarray(s.convert("RGB").resize(t.size, Image.LANCZOS), np.float32) / 255
    t = np.asarray(t.convert("RGB"), np.float32) / 255
    d = np.clip(np.abs(t - s).mean(2) / 0.35, 0, 1)
    # simple perceptual ramp: black -> deep blue -> orange -> pale yellow
    stops = np.array([[0.05, 0.05, 0.08], [0.12, 0.24, 0.55], [0.93, 0.41, 0.20], [1.0, 0.95, 0.72]])
    x = d * (len(stops) - 1)
    i = np.clip(x.astype(int), 0, len(stops) - 2)
    f = (x - i)[..., None]
    rgb = stops[i] * (1 - f) + stops[i + 1] * f
    return Image.fromarray((rgb * 255).astype(np.uint8))


def mad(s, t):
    s = np.asarray(s.convert("RGB").resize((512, 512)), np.float32) / 255
    t = np.asarray(t.convert("RGB").resize((512, 512)), np.float32) / 255
    return float(np.abs(t - s).mean())


def main():
    rows = [json.loads(l) for l in open(os.path.join(D, "data", "train.jsonl"))]
    overlap = json.load(open(os.path.join(D, "data", "overlap_report.json")))
    meta_json = json.load(open(os.path.join(D, "data", "meta.json")))
    pica = pq.read_table(os.path.join(D, "raw", "pica100k_00000.parquet"),
                         columns=["superficial_prompt", "intermediate_prompt", "explicit_prompt", "law", "op"]).to_pylist()

    recs = []
    for r in rows:
        src = "magicbrush" if r["id"].startswith("mb_") else "pica100k"
        s, t = Image.open(r["image"][0]), Image.open(r["target_image"])
        prompt = r["conversations"][0]["value"]
        rec = dict(id=r["id"], source=src, instruction=prompt.split("\n", 1)[1], human=prompt,
                   gpt=r["conversations"][1]["value"], src_size=list(s.size), tgt_size=list(t.size),
                   edit=mad(s, t), words=len(prompt.split("\n", 1)[1].split()),
                   src_path=r["image"][0], tgt_path=r["target_image"])
        if src == "magicbrush":
            img_id, turn = r["id"][3:].rsplit("_t", 1)
            rec.update(coco_id=img_id, turn=int(turn))
        else:
            p = pica[int(r["id"].rsplit("_", 1)[1])]
            rec.update(law=p["law"], op=p["op"], intermediate=p["intermediate_prompt"], explicit=p["explicit_prompt"],
                       row=int(r["id"].rsplit("_", 1)[1]))
        recs.append(rec)

    # ---- stats over all rows
    stats = {}
    for src in ("magicbrush", "pica100k"):
        rs = [x for x in recs if x["source"] == src]
        e = np.array([x["edit"] for x in rs])
        w = np.array([x["words"] for x in rs])
        sizes = collections.Counter(f"{x['src_size'][0]}×{x['src_size'][1]}" for x in rs)
        stats[src] = dict(n=len(rs), edit_median=float(np.median(e)), edit_mean=float(e.mean()),
                          below_003=float((e < 0.03).mean()), words_median=float(np.median(w)),
                          words_min=int(w.min()), words_max=int(w.max()),
                          hist=np.histogram(np.clip(e, 0, 0.4), bins=20, range=(0, 0.4))[0].tolist(),
                          sizes=sizes.most_common(4))
    stats["magicbrush"]["turns"] = sorted(collections.Counter(x["turn"] for x in recs if x["source"] == "magicbrush").items())
    stats["pica100k"]["laws"] = collections.Counter(x["law"] for x in recs if x["source"] == "pica100k").most_common()
    stats["pica100k"]["ops"] = collections.Counter(x["op"] for x in recs if x["source"] == "pica100k").most_common()

    # ---- examples: fixed seed, stratified (PICA by edit operation: the shard holds one physics law only)
    rng = random.Random(SEED)
    ex = []
    for turn in (1, 2, 3):
        pool = sorted([x for x in recs if x["source"] == "magicbrush" and x["turn"] == turn], key=lambda x: x["id"])
        ex += rng.sample(pool, MB_PER_TURN)
    laws = [l for l, _ in collections.Counter(x["op"] for x in recs if x["source"] == "pica100k").most_common()]
    per = {l: sorted([x for x in recs if x["source"] == "pica100k" and x["op"] == l], key=lambda x: x["id"]) for l in laws}
    picked = []
    while len(picked) < N_PICA:
        for l in laws:
            if len(picked) < N_PICA and per[l]:
                picked.append(per[l].pop(rng.randrange(len(per[l]))))
    ex += picked
    for x in ex:
        s, t = Image.open(x["src_path"]).convert("RGB"), Image.open(x["tgt_path"]).convert("RGB")
        x["img_src"], x["img_tgt"], x["img_diff"] = uri(s), uri(t), uri(diff_map(s, t))

    example_row = json.loads(open(os.path.join(D, "data", "train.jsonl")).readline())
    data = dict(mb_per_turn=MB_PER_TURN, stats=stats, examples=[{k: v for k, v in x.items() if k not in ("src_path", "tgt_path")} for x in ex],
                overlap=dict(checked=overlap["eval_images_checked"], hamming=overlap["hamming_max"],
                             dropped=collections.Counter(d["reason"].split(" (")[0].replace("near-duplicate of eval ", "")
                                                         for d in overlap["dropped"]).most_common(),
                             n_dropped=len(overlap["dropped"])),
                meta_json=meta_json, example_row=example_row, n=len(recs))
    page = open(os.path.join(HERE, "dataset_page_template.html")).read()
    page = page.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(page)
    print(f"{OUT}: {len(page.encode()) / 1e6:.2f} MB, {len(ex)} examples")
    for src in stats:
        print(src, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats[src].items() if k != "hist"})


if __name__ == "__main__":
    main()
