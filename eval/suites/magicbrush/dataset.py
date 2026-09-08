"""MagicBrush test-split loader (single-turn / "independent editing" protocol).

Test data (source: OSU-NLP-Group/MagicBrush's withheld official test set,
mirrored at HF `akshayg08/MagicBrushTest` since the upstream release requires a
manual form): `edit_sessions.json` maps img_id -> ordered list of turns
``{input, mask, output, instruction}``. For turn i>0, ``input`` is already the
*ground-truth* output of turn i-1 (verified against the raw JSON), so reading
each turn's own ``input``/``output`` fields directly gives single-turn (a.k.a.
MagicBrush's "independent editing") samples with no iterative chaining through
a model's own prior generations.

Generated-image naming mirrors the official ``evaluation/image_eval.py``
convention exactly (turn 1 -> ``<img_id>_1.png``, turn n>1 ->
``<img_id>_inde_{n}.png``) so its GT-pairing loader logic can consume our
outputs directly.
"""
import json
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MagicBrushItem:
    img_id: str
    turn: int  # 1-indexed
    input_path: str
    mask_path: str
    gt_output_path: str
    instruction: str
    target_caption: str
    gen_filename: str  # official naming for this turn's generated image


def load_items(data_root: str) -> List[MagicBrushItem]:
    """data_root must contain edit_sessions.json, global_descriptions.json, images/."""
    sessions = json.load(open(os.path.join(data_root, "edit_sessions.json")))
    captions = json.load(open(os.path.join(data_root, "global_descriptions.json")))
    img_dir = os.path.join(data_root, "images")

    items = []
    for img_id, turns in sessions.items():
        cap_map = captions.get(img_id, {})
        for i, t in enumerate(turns):
            turn = i + 1
            gen_filename = f"{img_id}_1.png" if turn == 1 else f"{img_id}_inde_{turn}.png"
            items.append(MagicBrushItem(
                img_id=img_id,
                turn=turn,
                input_path=os.path.join(img_dir, img_id, t["input"]),
                mask_path=os.path.join(img_dir, img_id, t["mask"]),
                gt_output_path=os.path.join(img_dir, img_id, t["output"]),
                instruction=t["instruction"],
                target_caption=cap_map.get(t["output"], ""),
                gen_filename=gen_filename,
            ))
    return items


def item_key(item: MagicBrushItem) -> str:
    return f"{item.img_id}_{item.turn}"


def filter_missing(items: List[MagicBrushItem]) -> List[MagicBrushItem]:
    """Drop items whose source assets aren't present (defensive; mirror data can be partial)."""
    return [it for it in items if os.path.exists(it.input_path) and os.path.exists(it.gt_output_path)]
