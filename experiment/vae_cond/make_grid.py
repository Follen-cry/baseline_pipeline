#!/usr/bin/env python
"""Build the side-by-side comparison from outputs/: one row per datapoint, columns
F0 | F1 | F2 | GT F3 | none | first | all  ->  outputs/comparison.png (+ per-datapoint rows in outputs/rows/).

  python make_grid.py        (any env with Pillow)
Also writes outputs/pixel_diff.csv: mean abs pixel difference of each condition's image to GT F3 and to F2
(a crude "does it just copy the last frame" signal; the judgement is the visual comparison, not this number).
"""
import csv
import os

from PIL import Image, ImageDraw

from common import datapoints, path

CONDS = ["none", "first", "all"]
T = 300  # thumbnail height
out_root = path("outputs_root")


def fit(im, h=T):
    return im.convert("RGB").resize((round(im.width * h / im.height), h), Image.LANCZOS)


def mad(a, b):
    import numpy as np
    b = b.resize(a.size, Image.LANCZOS)
    return float(np.abs(np.asarray(a, dtype="f4") - np.asarray(b, dtype="f4")).mean())


rows, stats = [], []
for d in datapoints():
    cells = [("F0 (VAE only in 'first'/'all')", Image.open(d["frames"][0])), ("F1 (VAE only in 'all')", Image.open(d["frames"][1])),
             ("F2 (VAE only in 'all')", Image.open(d["frames"][2])), ("GT F3", Image.open(d["target_image"]))]
    gen = {}
    for c in CONDS:
        p = os.path.join(out_root, c, d["id"] + ".png")
        if os.path.exists(p):
            gen[c] = Image.open(p)
            cells.append((f"VAE cond: {c}", gen[c]))
    thumbs = [(t, fit(im)) for t, im in cells]
    pad, lab = 6, 22
    W = sum(im.width for _, im in thumbs) + pad * (len(thumbs) + 1)
    row = Image.new("RGB", (W, T + lab + pad), "white")
    dr, x = ImageDraw.Draw(row), pad
    for t, im in thumbs:
        row.paste(im, (x, lab))
        dr.text((x + 2, 4), t, fill="black")
        x += im.width + pad
    cap = Image.new("RGB", (W, 18), "white")
    ImageDraw.Draw(cap).text((pad, 2), d["category"], fill=(90, 90, 90))
    rows.append((cap, row))
    gt, f2 = Image.open(d["target_image"]).convert("RGB"), Image.open(d["frames"][2]).convert("RGB")
    for c, im in gen.items():
        im = im.convert("RGB")
        stats.append({"id": d["id"], "category": d["category"], "condition": c,
                      "mad_to_gt_F3": round(mad(im, gt), 2), "mad_to_F2": round(mad(im, f2), 2)})

if not rows:
    raise SystemExit("no datapoints")
os.makedirs(os.path.join(out_root, "rows"), exist_ok=True)
W = max(r.width for _, r in rows)
H = sum(c.height + r.height for c, r in rows)
sheet, y = Image.new("RGB", (W, H), "white"), 0
for (cap, row), d in zip(rows, datapoints()):
    row.save(os.path.join(out_root, "rows", d["id"] + ".jpg"), quality=90)
    sheet.paste(cap, (0, y)); y += cap.height
    sheet.paste(row, (0, y)); y += row.height
sheet.save(os.path.join(out_root, "comparison.png"))
with open(os.path.join(out_root, "pixel_diff.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "category", "condition", "mad_to_gt_F3", "mad_to_F2"])
    w.writeheader()
    w.writerows(stats)
print(f"wrote {out_root}/comparison.png ({W}x{H}), pixel_diff.csv ({len(stats)} rows)")
