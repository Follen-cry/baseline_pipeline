#!/usr/bin/env python
"""Contact sheets for eyeballing all datapoints: outputs/review_<size>_<k>.png, 10 datapoints per sheet.
Columns: F2 (last input) | GT F3 | none | first | all.   (any env with Pillow)"""
import os

from PIL import Image, ImageDraw

from common import CONDS, SIZES, datapoints, path, size_tag

OUT = path("outputs_root")
T, PER = 170, 10
dps = datapoints()
for s in SIZES:
    for k in range(0, len(dps), PER):
        chunk = dps[k : k + PER]
        rows = []
        for d in chunk:
            ims = [Image.open(d["frames"][2]), Image.open(d["target_image"])] + [
                Image.open(os.path.join(OUT, c + size_tag(s), d["id"] + ".png")) for c in CONDS]
            rows.append((d["category"], [im.convert("RGB").resize((round(im.width * T / im.height), T), Image.LANCZOS) for im in ims]))
        cw = max(im.width for _, r in rows for im in r)
        sheet = Image.new("RGB", (150 + 5 * (cw + 4), 20 + len(rows) * (T + 4)), "white")
        dr = ImageDraw.Draw(sheet)
        for j, lab in enumerate(["F2 (input)", "GT F3"] + CONDS):
            dr.text((150 + j * (cw + 4), 4), lab, fill="black")
        for i, (cat, r) in enumerate(rows):
            y = 20 + i * (T + 4)
            dr.text((4, y + 4), f"{k + i}: {cat}", fill="black")
            for j, im in enumerate(r):
                sheet.paste(im, (150 + j * (cw + 4), y))
        p = os.path.join(OUT, f"review_{s}_{k // PER}.png")
        sheet.save(p)
        print("wrote", p, sheet.size)
