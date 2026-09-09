"""AURORA-Bench loader (Learning Action and Reasoning-Centric Image Editing
from Videos and Simulations, NeurIPS'24 D&B).

Official data (HF `McGill-NLP/aurora-bench`, 400-row `test` split): each row
has an ``input`` image, an ``instruction``, and a ``source`` tag (one of 8
origin datasets: magicbrush, ag, something, clevr, whatsup, emu, epic,
kubric; 50 items each). The HF parquet embeds images as raw bytes with no
stable per-row id (the original filename in ``input.path``, e.g.
``368667-output1.png``, is reused across multiple rows since AURORA chains
edits — an earlier pair's *output* becomes a later pair's *input*), so this
loader instead expects the parquet already extracted to a json + images/
layout (source-scoped running index as the id) via
``eval/suites/aurorabench/extract_from_hf.py``.

There is no ground-truth output image in this benchmark split — AURORA-Bench
is scored by an LMM judge / human raters on the *generated* edit, not by
comparison to a reference image.
"""
import json
import os
from dataclasses import dataclass
from typing import List


@dataclass
class AuroraBenchItem:
    key: str  # "<source>_<idx:03d>", e.g. "magicbrush_007"
    source: str
    input_path: str
    instruction: str
    gen_filename: str  # "<source>/<key>.png"


def load_items(data_root: str) -> List[AuroraBenchItem]:
    """data_root must contain test.json and images/ (see extract_from_hf.py)."""
    ann = json.load(open(os.path.join(data_root, "test.json")))
    items = []
    for it in ann:
        items.append(AuroraBenchItem(
            key=it["key"],
            source=it["source"],
            input_path=os.path.join(data_root, it["input"]),
            instruction=it["instruction"],
            gen_filename=f"{it['source']}/{it['key']}.png",
        ))
    return items


def item_key(item: AuroraBenchItem) -> str:
    return item.key


def filter_missing(items: List[AuroraBenchItem]) -> List[AuroraBenchItem]:
    """Drop items whose source image isn't present (defensive; mirror data can be partial)."""
    return [it for it in items if os.path.exists(it.input_path)]
