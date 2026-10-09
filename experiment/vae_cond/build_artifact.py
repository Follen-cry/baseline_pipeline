#!/usr/bin/env python
"""Build artifact/index.html (self-contained, images embedded as base64 JPEG) from datapoints.jsonl + outputs/.

  python build_artifact.py         (any env with Pillow)
Needs outputs/{none,first,all}{,_688x400,_832x480}/, outputs/size_test_diff.csv (size_sheet.py) and outputs/preproc_tension.png
(inspect_preproc.py). Notes below are hand-written from looking at the generated images; numbers come from the CSV.
"""
import base64
import csv
import html
import io
import os

from PIL import Image

from common import CFG, HERE, datapoints, path

OUT = path("outputs_root")
SIZES = [("832x480", "832x480", "native frame size"), ("688x400", "688x400", "= VAE condition size"), ("672x384", "672x384", "first run, area-512 rule")]
TAG = {"672x384": "", "688x400": "_688x400", "832x480": "_832x480"}
CONDS = ["none", "first", "all"]


def b64(p, w=560, q=82):
    im = Image.open(p).convert("RGB")
    im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


diff = {(r["id"], r["size"], r["condition"]): r for r in csv.DictReader(open(os.path.join(OUT, "size_test_diff.csv")))}

# hand-written notes per (category, size, condition)
N = {
 "disease_to_health": {
  "672x384": ("Empty brown frame with a block of text.", "Lizard on the rock, a copy of F0 with two stray digits along the bottom edge.", "Same scene, but the lizard is a ghosted blend of several poses."),
  "688x400": ("Two overlapping panels of a lizard-like shape with garbled digits.", "Clean lizard on the rock under the lamp.", "Clean lizard, head raised and slightly stretched."),
  "832x480": ("An unrelated cartoon-style face.", "Clean lizard on the rock under the lamp.", "Clean lizard on the rock under the lamp."),
 },
 "evaporation": {
  "672x384": ("Two grayscale portraits and garbled text.", "Clean wet cobblestones, close to F0/F2. A small red speck near the water.", "Clean wet cobblestones, close to F0/F2."),
  "688x400": ("Gray paving-stone pattern, no water, no color.", "Clean wet cobblestones.", "Clean wet cobblestones."),
  "832x480": ("Flat gray noise.", "Clean wet cobblestones.", "Clean wet cobblestones."),
 },
 "mature_to_flower": {
  "672x384": ("A three-panel contact sheet of the plant with garbled captions.", "The plant is shrunk into a gray-bordered panel with a caption line underneath.", "The plant, now with a few yellow flowers, sits as an inset in a gray frame with garbled text."),
  "688x400": ("A yellow flower cluster in a meadow, wrong framing.", "Full-frame meadow and plant, few flowers, a faint watermark-like mark at the bottom right.", "Full-frame meadow and plant with a few yellow flowers."),
  "832x480": ("A single yellow flower head on a blurred green background.", "Full-frame meadow and plant, almost no flowers.", "Full-frame meadow and plant with a few yellow flowers, closest to the ground truth."),
 },
 "fruit_ripening": {
  "672x384": ("A blurry yellow fruit on a gray plate.", "Clean frame, green-yellow fruit under the grow light, looks like F2.", "Clean frame, nearly identical to first."),
  "688x400": ("A pale yellow fruit on a small white plate, gray background.", "Clean frame, green-yellow fruit, looks like F2.", "Clean frame, nearly identical to first."),
  "832x480": ("A smooth yellow fruit on a white table, no grow light.", "Clean frame, brighter and more yellow than F2, with a warm glow.", "Same as first, slightly more glow."),
 },
 "tension": {
  "672x384": ("Flat gray field.", "Teal background with a strip of film at the bottom edge. Hands are gone.", "Same teal background, with a streaky band of fragments at the bottom."),
  "688x400": ("Flat gray-green field.", "Blurred glass scene with a faint strip, hands missing.", "Hands holding the stretched strip, as in the input frames."),
  "832x480": ("Flat gray-green field.", "Hands holding the stretched strip, as in the input frames.", "Hands holding the stretched strip, as in the input frames."),
 },
}

dps = datapoints()
cards = []
for d in dps:
    cat = d["category"]
    ins = "".join(
        f'<figure><img src="{b64(p)}" alt="F{i}" loading="lazy"><figcaption><b>F{i}</b><span>VAE in: {["first, all", "all", "all"][i]}</span></figcaption></figure>'
        for i, p in enumerate(d["frames"]))
    ins += (f'<figure class="gt"><img src="{b64(d["target_image"])}" alt="ground truth F3" loading="lazy">'
            f'<figcaption><b>GT F3</b><span>not shown to the model</span></figcaption></figure>')
    outs = ""
    for ci, c in enumerate(CONDS):
        blocks = ""
        for s, _, _ in SIZES:
            r = diff[(d["id"], s, c)]
            blocks += (f'<div class="sz sz-{s}"><img src="{b64(os.path.join(OUT, c + TAG[s], d["id"] + ".png"))}" alt="{c} {s}" loading="lazy">'
                       f'<figcaption><b><i class="dot"></i>{c}</b><span>MAD GT {r["mad_to_gt_F3"]} · F2 {r["mad_to_F2"]}</span></figcaption>'
                       f'<p>{html.escape(N[cat][s][ci])}</p></div>')
        outs += f'<figure class="out c-{c}">{blocks}</figure>'
    cards.append(f"""
<section class="dp" id="{cat}">
  <h3>{cat.replace('_', ' ')}<small>{html.escape(d['id'].split('__')[-3])} · gap {d['gap_s']} s</small></h3>
  <details><summary>Caption fed to the model</summary><p class="cap">{html.escape(d['caption'])}</p></details>
  <div class="row ins">{ins}</div>
  <div class="row outs">{outs}</div>
</section>""")


def mean(s, c, k):
    v = [float(diff[(d["id"], s, c)][k]) for d in dps]
    return sum(v) / len(v)


rows = ""
for s, _, note in SIZES:
    for j, c in enumerate(CONDS):
        rows += (f'<tr class="c-{c}">' + (f'<th rowspan="3" class="sizecell">{s}<small>{note}</small></th>' if j == 0 else "") +
                 f'<td><i class="dot"></i>{c}</td><td class="num">{mean(s, c, "mad_to_gt_F3"):.1f}</td><td class="num">{mean(s, c, "mad_to_F2"):.1f}</td></tr>')

doc = open(os.path.join(HERE, "artifact_template.html")).read()
doc = doc.replace("{{CARDS}}", "\n".join(cards)).replace("{{SUMMARY}}", rows)
doc = doc.replace("{{PREPROC}}", b64(os.path.join(OUT, "preproc_tension.png"), w=1400, q=85))
doc = doc.replace("{{INF}}", html.escape(", ".join(f"{k} = {v}" for k, v in CFG["inference"].items())))
os.makedirs(os.path.join(HERE, "artifact"), exist_ok=True)
p = os.path.join(HERE, "artifact", "index.html")
open(p, "w").write(doc)
print(p, f"{os.path.getsize(p) / 1e6:.2f} MB")
