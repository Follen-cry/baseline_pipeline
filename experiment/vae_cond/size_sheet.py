#!/usr/bin/env python
"""Contact sheet for the size test: per datapoint, rows = output size (672x384 default, 688x400 = VAE cond size, 832x480 native),
columns = none | first | all.  -> outputs/size_test.png   (any env with Pillow)"""
import os

from PIL import Image, ImageDraw

from common import datapoints, path

OUT = path("outputs_root")
SIZES = [("", "672x384 (default)"), ("_688x400", "688x400 (= VAE cond size)"), ("_832x480", "832x480 (native)")]
T = 200
tiles = []
for d in datapoints():
    for tag, lab in SIZES:
        row = []
        for c in ["none", "first", "all"]:
            im = Image.open(os.path.join(OUT, c + tag, d["id"] + ".png")).convert("RGB")
            row.append(im.resize((round(im.width * T / im.height), T), Image.LANCZOS))
        tiles.append((d["category"], lab, row))
cw = max(im.width for _, _, r in tiles for im in r)
W, H = 190 + 3 * (cw + 6), len(tiles) * (T + 4) + 30
s = Image.new("RGB", (W, H), "white")
dr = ImageDraw.Draw(s)
for j, c in enumerate(["none", "first", "all"]):
    dr.text((190 + j * (cw + 6), 8), c, fill="black")
for i, (cat, lab, row) in enumerate(tiles):
    y = 30 + i * (T + 4)
    dr.text((4, y + 4), cat if lab.startswith("672") else "", fill="black")
    dr.text((4, y + 20), lab, fill=(90, 90, 90))
    for j, im in enumerate(row):
        s.paste(im, (190 + j * (cw + 6), y))
s.save(os.path.join(OUT, "size_test.png"))
print("wrote", os.path.join(OUT, "size_test.png"), s.size)

# pixel distances (same MAD as make_grid.py) for every size/condition -> outputs/size_test_diff.csv
import csv

import numpy as np


def mad(a, b):
    b = b.resize(a.size, Image.LANCZOS)
    return float(np.abs(np.asarray(a, dtype="f4") - np.asarray(b, dtype="f4")).mean())


rows = []
for d in datapoints():
    gt, f2 = (Image.open(p).convert("RGB") for p in (d["target_image"], d["frames"][2]))
    for tag, lab in SIZES:
        for c in ["none", "first", "all"]:
            im = Image.open(os.path.join(OUT, c + tag, d["id"] + ".png")).convert("RGB")
            rows.append({"id": d["id"], "category": d["category"], "size": tag.lstrip("_") or "672x384", "condition": c,
                         "mad_to_gt_F3": round(mad(im, gt), 2), "mad_to_F2": round(mad(im, f2), 2)})
with open(os.path.join(OUT, "size_test_diff.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for sz in ["672x384", "688x400", "832x480"]:
    for c in ["none", "first", "all"]:
        r = [x for x in rows if x["size"] == sz and x["condition"] == c]
        print(sz, c, "GT %.1f  F2 %.1f" % (np.mean([x["mad_to_gt_F3"] for x in r]), np.mean([x["mad_to_F2"] for x in r])))
