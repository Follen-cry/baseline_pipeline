#!/usr/bin/env python
"""Build artifact/index.html (self-contained, images embedded as base64 JPEG) from datapoints.jsonl + outputs/.

  python build_artifact.py         (any env with Pillow)
The findings text below is written by hand from looking at outputs/comparison.png; numbers come from outputs/pixel_diff.csv.
"""
import base64
import csv
import html
import io
import os

from PIL import Image

from common import CFG, HERE, datapoints, path

OUT = path("outputs_root")
W = 560  # embedded image width


def b64(p):
    im = Image.open(p).convert("RGB")
    im = im.resize((W, round(im.height * W / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


diff = {(r["id"], r["condition"]): r for r in csv.DictReader(open(os.path.join(OUT, "pixel_diff.csv")))}

# hand-written per-datapoint notes (from viewing the sheet), keyed by category
NOTES = {
    "disease_to_health": {"none": "Empty brown frame with a block of text. Nothing of the scene survives.",
                          "first": "Lizard on the rock, a copy of F0 with two stray digits along the bottom edge.",
                          "all": "Same scene, but the lizard is a ghosted blend of several poses."},
    "evaporation": {"none": "Two grayscale portraits and garbled text. Unrelated to the clip.",
                    "first": "Clean wet cobblestones, close to F0/F2. A small red speck near the water.",
                    "all": "Clean wet cobblestones, close to F0/F2."},
    "mature_to_flower": {"none": "A three-panel contact sheet of the plant with garbled captions. Content is on topic, layout is not.",
                         "first": "The plant is shrunk into a gray-bordered panel with a caption line underneath.",
                         "all": "The plant, now with a few yellow flowers, sits as an inset picture in a gray frame with garbled text."},
    "fruit_ripening": {"none": "A blurry yellow fruit on a gray plate. Right object, wrong scene and camera.",
                       "first": "Clean frame, green-yellow fruit under the grow light, looks like F2.",
                       "all": "Clean frame, nearly identical to `first`."},
    "tension": {"none": "Flat gray field. Nothing generated.",
                "first": "Teal background with a strip of film at the bottom edge. Hands are gone.",
                "all": "Same teal background, with a streaky band of fragments at the bottom."},
}

dps = datapoints()
cards = []
for d in dps:
    cat = d["category"]
    ins = "".join(
        f'<figure><img src="{b64(p)}" alt="F{i}" loading="lazy"><figcaption><b>F{i}</b>'
        f'<span>{["ViT only (also VAE in first, all)", "ViT only (also VAE in all)", "ViT; VAE only in all"][i] if False else ["VAE: first, all", "VAE: all", "VAE: all"][i]}</span></figcaption></figure>'
        for i, p in enumerate(d["frames"]))
    gt = f'<figure class="gt"><img src="{b64(d["target_image"])}" alt="ground truth F3" loading="lazy"><figcaption><b>GT F3</b><span>target, not shown to the model</span></figcaption></figure>'
    outs = ""
    for c in ["none", "first", "all"]:
        r = diff[(d["id"], c)]
        outs += (f'<figure class="out c-{c}"><img src="{b64(os.path.join(OUT, c, d["id"] + ".png"))}" alt="{c}" loading="lazy">'
                 f'<figcaption><b><i class="dot"></i>{c}</b><span>MAD to GT {r["mad_to_gt_F3"]} · to F2 {r["mad_to_F2"]}</span></figcaption>'
                 f'<p>{html.escape(NOTES[cat][c])}</p></figure>')
    cards.append(f"""
<section class="dp" id="{cat}">
  <h3>{cat.replace('_', ' ')}<small>{html.escape(d['id'].split('__')[-3])} · gap {d['gap_s']} s</small></h3>
  <details><summary>Caption fed to the model</summary><p class="cap">{html.escape(d['caption'])}</p></details>
  <div class="row ins">{ins}{gt}</div>
  <div class="row outs">{outs}</div>
</section>""")

# summary table (mean over the 5)
def mean(c, k):
    v = [float(diff[(d["id"], c)][k]) for d in dps]
    return sum(v) / len(v)

summary = "".join(
    f'<tr class="c-{c}"><th><i class="dot"></i>{c}</th><td>{n}</td><td>{mean(c, "mad_to_gt_F3"):.1f}</td><td>{mean(c, "mad_to_F2"):.1f}</td></tr>'
    for c, n in [("none", "0"), ("first", "1 (F0)"), ("all", "3 (F0, F1, F2)")])

doc = open(os.path.join(HERE, "artifact_template.html")).read()
doc = doc.replace("{{CARDS}}", "\n".join(cards)).replace("{{SUMMARY}}", summary)
doc = doc.replace("{{INF}}", html.escape(", ".join(f"{k} = {v}" for k, v in CFG["inference"].items())))
os.makedirs(os.path.join(HERE, "artifact"), exist_ok=True)
p = os.path.join(HERE, "artifact", "index.html")
open(p, "w").write(doc)
print(p, f"{os.path.getsize(p) / 1e6:.2f} MB")
