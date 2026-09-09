"""RISEBench loader (Envisioning Beyond the Pixels: Benchmarking
Reasoning-Informed Visual Editing, NeurIPS'25 DB Oral).

Official data (HF `PhoenixZ/RISEBench`, GitHub `PhoenixZ810/RISEBench`):
``datav2_total_w_subtask.json`` is a flat list of 360 single-input-image
items across 4 reasoning categories (temporal/causal/spatial/logical), each
with an ``instruction`` and a free-text ``reference`` describing the
expected outcome (used as a judge hint, not a pixel-level ground truth —
RISEBench has no GT image). ``image`` paths are relative to a ``data/``
subdirectory shipped alongside the json (from the repo's ``data.zip``).

Generated-image naming: ``<category>/<index>.png`` (index = the item's
``index`` field, e.g. ``temporal_reasoning_1``), one file per item — there
is no multi-turn chaining like MagicBrush, every item is independent.

Per-item rows are sparse: each row only carries the keys relevant to its
own category/subtask (missing keys are simply absent, not null), matching
upstream's own ``pd.isna(item[...])``-gated branches in ``gpt_eval.py``.
The extra optional fields below (``consistency_free``, ``reasoning_img``,
``reference_img``, ``reference_txt``, ``reasoning_wo_ins``) feed the GPT
judge's per-category prompt selection in ``evaluators/eval_risebench.py``
— see that module's docstring for how each one is used.
"""
import json
import os
from dataclasses import dataclass
from typing import List, Optional


def _present(it: dict, key: str) -> bool:
    """Mirror upstream's `key in item and not pd.isna(item[key])` presence
    check — these fields are flag-like ('True' as a string, or a path);
    what matters is whether the key is set at all, not its value."""
    return key in it and it[key] is not None and it[key] != ""


@dataclass
class RiseBenchItem:
    index: str  # e.g. "temporal_reasoning_1"
    category: str  # temporal_reasoning | causal_reasoning | spatial_reasoning | logical_reasoning
    subtask: str
    input_path: str
    instruction: str
    reference: str
    gen_filename: str  # "<category>/<index>.png"
    # Judge-only fields (see evaluators/eval_risebench.py):
    consistency_free: bool = False  # skip the ApprConsistency judge call entirely
    reasoning_img: bool = False  # reasoning judge also needs the original input image
    reasoning_img_op: bool = False  # unused by upstream eval_vanilla; kept for fidelity
    reference_img_path: Optional[str] = None  # ground-truth answer image (spatial/logical)
    reference_txt: Optional[str] = None  # ground-truth answer text (logical)
    reasoning_wo_ins: bool = False  # logical img-vs-img judge doesn't need the instruction text


def load_items(data_root: str) -> List[RiseBenchItem]:
    """data_root must contain datav2_total_w_subtask.json and data/."""
    ann = json.load(open(os.path.join(data_root, "datav2_total_w_subtask.json")))
    items = []
    for it in ann:
        items.append(RiseBenchItem(
            index=it["index"],
            category=it["category"],
            subtask=it.get("subtask", ""),
            input_path=os.path.join(data_root, "data", it["image"]),
            instruction=it["instruction"],
            reference=it.get("reference", ""),
            gen_filename=f"{it['category']}/{it['index']}.png",
            consistency_free=_present(it, "consistency_free"),
            reasoning_img=_present(it, "reasoning_img"),
            reasoning_img_op=_present(it, "reasoning_img_op"),
            reference_img_path=(os.path.join(data_root, "data", it["reference_img"])
                                 if _present(it, "reference_img") else None),
            reference_txt=it.get("reference_txt") if _present(it, "reference_txt") else None,
            reasoning_wo_ins=_present(it, "reasoning_wo_ins"),
        ))
    return items


def item_key(item: RiseBenchItem) -> str:
    return item.index


def filter_missing(items: List[RiseBenchItem]) -> List[RiseBenchItem]:
    """Drop items whose source image isn't present (defensive; mirror data can be partial)."""
    return [it for it in items if os.path.exists(it.input_path)]
