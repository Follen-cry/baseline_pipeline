#!/usr/bin/env python
"""Build artifact/index.html + artifact/img/** (thumbnails, referenced by relative URL) from the 40-datapoint run.

  python build_artifact.py         (any env with Pillow + numpy)
Inputs: datapoints.jsonl, outputs/{none,first,all}{,_688x400,_832x480}/, outputs/metrics.csv (metrics.py), tags.json (make_tags.py),
outputs/preproc_tension.png (inspect_preproc.py), artifact_template.html.
"""
import csv
import html
import json
import os

import numpy as np
from PIL import Image

from common import CONDS, HERE, SIZES, datapoints, path, size_tag

OUT = path("outputs_root")
ART = os.path.join(HERE, "artifact")
W = 420  # thumbnail width
DEFAULT_SIZE = "688x400"
TAGS = json.load(open(os.path.join(HERE, "tags.json")))["tags"]
LABEL = {"o": "coherent", "L": "layout artifact", "D": "duplicated object", "F": "object missing / wrong"}
SIZE_NOTE = {"832x480": "native frame size", "688x400": "= VAE condition size of the 672×384 run", "672x384": "first run (area-512 rule)"}

rows = list(csv.DictReader(open(os.path.join(OUT, "metrics.csv"))))
M = {(r["id"], r["size"], r["condition"]): r for r in rows}
dps = datapoints()
n = len(dps)


def thumb(src, dst, w=W, q=74):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im = Image.open(src).convert("RGB")
    im.resize((w, round(im.height * w / im.width)), Image.LANCZOS).save(dst, "JPEG", quality=q)


def mean_se(vals):
    v = np.array(vals, dtype=float)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


# ---- thumbnails (published as files) ----
for d in dps:
    base = os.path.join(ART, "img", d["id"])
    for i, p in enumerate(d["frames"] + [d["target_image"]]):
        thumb(p, os.path.join(base, f"F{i}.jpg"))
    for s in SIZES:
        for c in CONDS:
            thumb(os.path.join(OUT, c + size_tag(s), d["id"] + ".png"), os.path.join(base, f"{c}_{s}.jpg"))
thumb(os.path.join(OUT, "preproc_tension.png"), os.path.join(ART, "img", "preproc_tension.jpg"), w=1400, q=85)

# ---- table 1: similarity metrics, mean ± s.e. over the 40 datapoints ----
def cell(vals, fmt):
    m, se = mean_se(vals)
    return f'<td class="num">{m:{fmt}}<small> ±{se:{fmt}}</small></td>'


t1 = ""
for s in SIZES:
    for j, c in enumerate(CONDS):
        r = [M[(d["id"], s, c)] for d in dps]
        t1 += (f'<tr class="c-{c}">' + (f'<th rowspan="3" class="sizecell">{s}<small>{SIZE_NOTE[s]}</small></th>' if j == 0 else "") +
               f'<td><i class="dot"></i>{c}</td>' + cell([float(x["clip_gt"]) for x in r], ".3f") + cell([float(x["clip_f2"]) for x in r], ".3f") +
               cell([float(x["mad_gt"]) for x in r], ".1f") + cell([float(x["mad_f2"]) for x in r], ".1f") + "</tr>")
ref = [M[(d["id"], "-", "copy_F2")] for d in dps]
t1 += ('<tr class="ref"><th class="sizecell">reference</th><td>copy F2</td>' + cell([float(x["clip_gt"]) for x in ref], ".3f") +
       '<td class="num">1.000</td>' + cell([float(x["mad_gt"]) for x in ref], ".1f") + '<td class="num">0.0</td></tr>')

# ---- table 2: paired first vs all (CLIP to GT) ----
t2 = ""
for s in SIZES:
    diffs = [float(M[(d["id"], s, "all")]["clip_gt"]) - float(M[(d["id"], s, "first")]["clip_gt"]) for d in dps]
    m, se = mean_se(diffs)
    wins = sum(x > 0 for x in diffs)
    madd = [float(M[(d["id"], s, "first")]["mad_gt"]) - float(M[(d["id"], s, "all")]["mad_gt"]) for d in dps]
    t2 += (f'<tr><th class="sizecell">{s}</th><td class="num">{m:+.3f}<small> ±{se:.3f}</small></td><td class="num">{wins} / {n}</td>'
           f'<td class="num">{sum(x > 0 for x in madd)} / {n}</td></tr>')

# ---- table 3: hand tags ----
t3 = ""
for s in SIZES:
    for j, c in enumerate(["first", "all"]):
        tg = TAGS[s][c]
        cnt = {k: tg.count(k) for k in "oLDF"}
        t3 += (f'<tr class="c-{c}">' + (f'<th rowspan="2" class="sizecell">{s}<small>{SIZE_NOTE[s]}</small></th>' if j == 0 else "") +
               f'<td><i class="dot"></i>{c}</td>'
               f'<td class="barcell"><span class="bar"><i style="width:{100 * cnt["o"] / n:.0f}%"></i></span><b>{cnt["o"]}</b> / {n}</td>'
               f'<td class="num">{cnt["L"]}</td><td class="num">{cnt["D"]}</td><td class="num">{cnt["F"]}</td></tr>')

# ---- gallery cards ----
cards = []
for k, d in enumerate(dps):
    base = f"img/{d['id']}"
    ins = "".join(
        f'<figure><img src="{base}/F{i}.jpg" alt="F{i}" loading="lazy"><figcaption><b>F{i}</b><span>VAE in: {["first, all", "all", "all"][i]}</span></figcaption></figure>'
        for i in range(3))
    ins += f'<figure class="gt"><img src="{base}/F3.jpg" alt="ground truth F3" loading="lazy"><figcaption><b>GT F3</b><span>not shown to the model</span></figcaption></figure>'
    outs, flags = "", []
    for c in CONDS:
        blocks = ""
        for s in SIZES:
            r = M[(d["id"], s, c)]
            tag = "scene lost" if c == "none" else LABEL[TAGS[s][c][k]]
            chip_cls = "chip ok" if (c != "none" and TAGS[s][c][k] == "o") else "chip bad"
            blocks += (f'<div class="sz sz-{s}"><img src="{base}/{c}_{s}.jpg" alt="{c} {s}" loading="lazy">'
                       f'<figcaption><b><i class="dot"></i>{c}</b><span class="{chip_cls}">{tag}</span></figcaption>'
                       f'<p>CLIP→GT {float(r["clip_gt"]):.3f} · MAD→GT {float(r["mad_gt"]):.1f}</p></div>')
        outs += f'<figure class="out c-{c}">{blocks}</figure>'
    art = " ".join(f'data-art-{s}="{int(any(TAGS[s][c][k] != "o" for c in ("first", "all")))}"' for s in SIZES)
    cards.append(f"""
<section class="dp" id="dp{k}" {art}>
  <h3><span class="idx">{k}</span>{d['category'].replace('_', ' ')}<small>{html.escape(d['id'].split('__')[-3])} · gap {d['gap_s']} s</small></h3>
  <details><summary>Caption fed to the model</summary><p class="cap">{html.escape(d['caption'])}</p></details>
  <div class="row ins">{ins}</div>
  <div class="row outs">{outs}</div>
</section>""")

doc = open(os.path.join(HERE, "artifact_template.html")).read()
for key, val in {"{{CARDS}}": "\n".join(cards), "{{T1}}": t1, "{{T2}}": t2, "{{T3}}": t3, "{{N}}": str(n), "{{DEFAULT}}": DEFAULT_SIZE,
                 "{{INF}}": html.escape(", ".join(f"{a} = {b}" for a, b in __import__("common").CFG["inference"].items()))}.items():
    doc = doc.replace(key, val)
import base64
import re


def datauri(rel):
    return "data:image/jpeg;base64," + base64.b64encode(open(os.path.join(ART, rel), "rb").read()).decode()


# inline every thumbnail: one self-contained page (the 16 MB page limit is checked below)
doc = re.sub(r'src="(img/[^"]+\.jpg)"', lambda m: f'src="{datauri(m.group(1))}"', doc)
os.makedirs(ART, exist_ok=True)
p = os.path.join(ART, "index.html")
open(p, "w").write(doc)
nf = sum(len(f) for _, _, f in os.walk(os.path.join(ART, "img")))
sz = sum(os.path.getsize(os.path.join(a, x)) for a, _, f in os.walk(os.path.join(ART, "img")) for x in f)
print(p, f"{os.path.getsize(p) / 1e3:.0f} KB html;", nf, "image files", f"{sz / 1e6:.1f} MB")
for s in SIZES:
    print(s, {c: {k: TAGS[s][c].count(k) for k in "oLDF"} for c in ("first", "all")})
