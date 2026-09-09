"""PhyEditBench loader (PhyEditBench: A Real-World Multi-Stage Benchmark for
Physics-Aware Image Editing, ECCV'26, arXiv:2606.26551).

Official data (GitHub `Previsior/PhyEditBench`, HF
`Previsior-PhyEdit/phyedit-bench`): ``bench/<PrimaryClass>/<SubClass>/meta.json``
holds 238 real-world four-state physical trajectories (``input`` ->
``intermediate_1`` -> ``intermediate_2`` -> ``output``) across 4 primary
physical categories / 12 subclasses, each with 3 step-wise instructions and
1 global instruction. ``bench/anti-physic/meta.jsonl`` holds 35 synthetic
counterfactual-physics single-image items (no output/GT state — scored
against a free-text ``expected_phenomenon``/checklist, not a target image).

Upstream's official protocol (``gpt_eval.py::build_type_fields``) turns each
real trajectory into 5 independent generation tasks (README's Type A-E
table):
  TypeA: input -> intermediate_1           (step 1)
  TypeB: intermediate_1 -> intermediate_2  (step 2)
  TypeC: intermediate_2 -> output          (step 3)
  TypeD: input -> output                   (steps 1-3, packed sequentially)
  TypeE: input -> output                   (global instruction)
This loader mirrors that exactly (238*5 = 1190 real items) plus one item per
anti-physic row (35, ``anti-physic``), matching upstream's own
``resolve_pred_path`` filename convention verbatim
(``<PrimaryClass>/<SubClass>/<Type>/<id>.png`` and ``anti-physic/<data_id>.png``,
see ``gpt_eval.py``/``gpt_eval_anti.py``) so a future GPT-judge port of
``evaluators/`` could read this suite's output tree without renaming
anything.
"""
import json
import os
from dataclasses import dataclass
from typing import List, Optional

# Order/spelling matches upstream gpt_eval.py::PRIMARY_CLASSES exactly.
PRIMARY_CLASSES = [
    "Rigid_Body_&_Interaction",
    "Deformation_&_Fracture",
    "Fluid_Dynamics",
    "State_Change_&_Environment",
]

TYPE_ORDER = ["TypeA", "TypeB", "TypeC", "TypeD", "TypeE"]


@dataclass
class PhyEditItem:
    key: str  # e.g. "Deformation_&_Fracture/brittle_fracture/TypeA/12" or "anti-physic/7"
    edit_type: str  # TypeA | TypeB | TypeC | TypeD | TypeE | anti-physic
    primary: Optional[str]
    sub: Optional[str]
    dp_id: str
    input_path: str
    instruction: str
    gen_filename: str


def _packed_typeD_instruction(steps: List[str]) -> str:
    # Verbatim upstream gpt_eval.py::build_type_fields TypeD prompt wrapper --
    # kept identical so a future judge scores what upstream's own README
    # documents, not a paraphrase of it.
    return (
        "Apply the following steps sequentially to reach the FINAL output state.\n"
        "Do NOT produce intermediate outputs; only the final state matters.\n\n"
        f"Step 1:\n{steps[0]}\n\nStep 2:\n{steps[1]}\n\nStep 3:\n{steps[2]}"
    )


def _load_real_items(bench_root: str) -> List[PhyEditItem]:
    items = []
    for primary in PRIMARY_CLASSES:
        primary_dir = os.path.join(bench_root, primary)
        if not os.path.isdir(primary_dir):
            continue
        for sub in sorted(os.listdir(primary_dir)):
            sub_dir = os.path.join(primary_dir, sub)
            meta_path = os.path.join(sub_dir, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            for dp in json.load(open(meta_path)):
                dp_id = dp["id"]
                frames = dp["frames"]
                steps = dp["instruction"]["steps"]
                global_inst = dp["instruction"]["global"]

                def add(edit_type: str, src_rel: str, instruction: str) -> None:
                    items.append(PhyEditItem(
                        key=f"{primary}/{sub}/{edit_type}/{dp_id}",
                        edit_type=edit_type,
                        primary=primary,
                        sub=sub,
                        dp_id=dp_id,
                        input_path=os.path.join(sub_dir, src_rel),
                        instruction=instruction,
                        gen_filename=f"{primary}/{sub}/{edit_type}/{dp_id}.png",
                    ))

                add("TypeA", frames["input"], steps[0])
                add("TypeB", frames["intermediate_1"], steps[1])
                add("TypeC", frames["intermediate_2"], steps[2])
                add("TypeD", frames["input"], _packed_typeD_instruction(steps))
                add("TypeE", frames["input"], global_inst)
    return items


def _load_anti_physic_items(bench_root: str) -> List[PhyEditItem]:
    anti_dir = os.path.join(bench_root, "anti-physic")
    meta_path = os.path.join(anti_dir, "meta.jsonl")
    if not os.path.isfile(meta_path):
        return []
    items = []
    for line in open(meta_path):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        data_id = row["data_id"]
        items.append(PhyEditItem(
            key=f"anti-physic/{data_id}",
            edit_type="anti-physic",
            primary=None,
            sub=None,
            dp_id=str(data_id),
            input_path=os.path.join(anti_dir, "input_data", f"data_{data_id}.png"),
            instruction=row["edit_prompt"],
            gen_filename=f"anti-physic/{data_id}.png",
        ))
    return items


def load_items(data_root: str) -> List[PhyEditItem]:
    """data_root must contain bench/ (the PhyEditBench GitHub repo's data
    dir), e.g. /scratch/local/ssd/junlin/data/PhyEditBench."""
    bench_root = os.path.join(data_root, "bench")
    return _load_real_items(bench_root) + _load_anti_physic_items(bench_root)


def item_key(item: PhyEditItem) -> str:
    return item.key


def filter_missing(items: List[PhyEditItem]) -> List[PhyEditItem]:
    """Drop items whose source image isn't present (defensive; mirror data can be partial)."""
    return [it for it in items if os.path.exists(it.input_path)]
