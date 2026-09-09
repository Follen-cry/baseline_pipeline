"""ImgEdit-Bench (Basic-Bench) loader (ImgEdit: A Unified Image Editing
Dataset and Benchmark, arXiv:2505.20275).

Official data (HF `sysuyy/ImgEdit`, `Benchmark.tar` -> `Benchmark/singleturn/`):
``singleturn.json`` is a flat dict of 737 single-input-image items keyed by a
numeric string id, each with an ``id`` (image path relative to
``singleturn/``), a ``prompt`` (edit instruction), and an ``edit_type`` (one
of 9 categories: action/add/adjust/background/compose/extract/remove/
replace/style). ``judge_prompt.json`` (upstream: ``prompts.json``) carries
the per-edit_type GPT-judge rubric, keyed the same way.

Most categories address one source image directly (e.g. ``animal/<id>.jpg``);
``compose`` addresses a subject/reference pair via a nested path
(``compose/<human|objects|animal>/<n>.jpg``) but only one path is present per
item — this loader treats it like any other single ``input_path``, same as
upstream's own ``origin_img_root``-relative lookup.

Generated-image naming: ``<edit_type>/<key>.png`` (key = the item's dict
key, e.g. ``"1082"``) — same nesting convention as
``suites/risebench/dataset.py`` / ``suites/aurorabench/dataset.py``, not
upstream's own flat ``<key>.png`` (upstream's ``basic_bench.py`` doesn't care
about nesting, only that the filename prefix matches the key).

ImgEdit-Bench also ships UGE-Bench (``Benchmark/hard/`` + GitHub-hosted
``Benchmark/UGE/UGE_edit.json``, 47 items, complex multi-object spatial
reasoning) and Multiturn-Bench (``Benchmark/multiturn/{content_memory,
content_understand,version_backtrace}/annotation.json``, chained edits) —
both already present in the same ``Benchmark.tar`` this loader reads, but
neither is wired up here. Add sibling loaders following this shape if/when
those get ported (Basic-Bench is the subset most papers cite as
"ImgEdit-Bench").
"""
import json
import os
from dataclasses import dataclass
from typing import List


@dataclass
class ImgEditItem:
    key: str  # e.g. "1082"
    edit_type: str  # action | add | adjust | background | compose | extract | remove | replace | style
    input_path: str
    instruction: str
    gen_filename: str  # "<edit_type>/<key>.png"


def load_items(data_root: str) -> List[ImgEditItem]:
    """data_root must contain singleturn.json and the per-category image dirs
    (i.e. point at the extracted Benchmark/singleturn/)."""
    ann = json.load(open(os.path.join(data_root, "singleturn.json")))
    items = []
    for key, it in ann.items():
        items.append(ImgEditItem(
            key=key,
            edit_type=it["edit_type"],
            input_path=os.path.join(data_root, it["id"]),
            instruction=it["prompt"],
            gen_filename=f"{it['edit_type']}/{key}.png",
        ))
    return items


def item_key(item: ImgEditItem) -> str:
    return item.key


def filter_missing(items: List[ImgEditItem]) -> List[ImgEditItem]:
    """Drop items whose source image isn't present (defensive; mirror data can be partial)."""
    return [it for it in items if os.path.exists(it.input_path)]
