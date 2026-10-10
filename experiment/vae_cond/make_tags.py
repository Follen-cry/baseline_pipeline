#!/usr/bin/env python
"""Writes tags.json from the hand tags below (10 characters per block of 10 datapoints: 0-9, 10-19, 20-29, 30-39).
Kept as a script so the blocks stay readable; see the _doc field of tags.json for the code meanings."""
import json
import os

from common import HERE

B = {
    ("832x480", "first"): ["oooooFoooo", "ooLooooDoo", "ooooooDDoo", "FooFooDooD"],
    ("832x480", "all"): ["oooooFoDoo", "oFoooLooFo", "ooooFoFDoo", "oooFFooooD"],
    ("688x400", "first"): ["ooooFFoooo", "oooooLoooo", "ooooooFLoo", "oooFFooooo"],
    ("688x400", "all"): ["oooooFFFoo", "oFoooLoooo", "ooooFoFooo", "oLoFFooooL"],
    ("672x384", "first"): ["LoLoLFoooo", "oooooLoLFo", "oooLooFLLo", "ooLFFoLooo"],
    ("672x384", "all"): ["FoLoLFoooL", "oFoooLoLoo", "oooLFoLLoo", "ooLFFoLooo"],
}
doc = {
    "_doc": ("Hand tags of the generated frames, one character per datapoint in datapoints.jsonl order (40). Read from "
             "outputs/review_<size>_<k>.png contact sheets (170 px tall thumbnails) by one unblinded reviewer (Claude); approximate. "
             "Codes: o = coherent frame of the input scene (no layout problem); L = layout artifact (panel/border, caption text strip, "
             "letterbox bars, shifted/zoomed/cropped content, inset picture); D = object duplicated side by side; F = main object "
             "missing, wrong, garbled or fragmented. 'none' is not tagged: it reproduces the input scene in 0 of 40 at every size "
             "(gray fields, portraits, text posters, a different object or room)."),
    "codes": {"o": "coherent", "L": "layout artifact", "D": "duplicated object", "F": "object missing / wrong / fragmented"},
    "tags": {},
}
for (size, cond), blocks in B.items():
    s = "".join(blocks)
    assert len(s) == 40 and set(s) <= set("oLDF"), (size, cond, len(s))
    doc["tags"].setdefault(size, {})[cond] = s
json.dump(doc, open(os.path.join(HERE, "tags.json"), "w"), indent=1)
for size, d in doc["tags"].items():
    for cond, s in d.items():
        print(size, cond, {c: s.count(c) for c in "oLDF"})
