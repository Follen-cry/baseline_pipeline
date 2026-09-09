# PhyEditBench — physics-aware, multi-stage image editing

[Previsior/PhyEditBench](https://github.com/Previsior/PhyEditBench)
("PhyEditBench: A Real-World Multi-Stage Benchmark for Physics-Aware Image
Editing", ECCV'26, [arXiv:2606.26551](https://arxiv.org/abs/2606.26551)).

Follows the `risebench`/`aurorabench` pattern: **generation only, InternVL-U**
— the official scorer is a GPT-4o judge (`gpt_eval.py`/`gpt_eval_anti.py`),
documented below but not ported (same reasoning as those two suites; unlike
`imgedit`, PhyEditBench doesn't ship a redistributable local judge
checkpoint).

## Data

238 real-world four-state physical trajectories (`input` -> `intermediate_1`
-> `intermediate_2` -> `output`) across 4 primary physical categories / 12
subclasses — `Deformation_&_Fracture` (brittle_fracture, elasticity,
plastic_deformation), `Fluid_Dynamics` (Buoyancy_&_Tension, Pouring_&_Flow,
Splashing_&_Impact), `Rigid_Body_&_Interaction` (Collision_&_Chain,
Gravity_&_Fall, Rotation_&_Rolling, Stability_&_Balance),
`State_Change_&_Environment` (Diffusion_&_Aerodynamics, Phase_Changes) — each
with 3 step-wise instructions (one per state transition), a global
instruction (input -> final output directly), physical `explain` text, and
`invariants` (what must stay unchanged). Plus 35 synthetic **Anti-Physics**
items (`bench/anti-physic/`) testing counterfactual physical rules — single
input image, one `edit_prompt`, **no output/GT state** (scored against a
free-text `expected_phenomenon` + `checklist`, not a target image).

Source: GitHub `Previsior/PhyEditBench` (plain git clone, not Xet-backed —
no HF download workaround needed here, unlike `risebench`/`aurorabench`/
`imgedit`). **Known-good copy**, already staged (1.3G, whole repo `bench/`
dir plus upstream's own eval scripts for reference):
```
/scratch/local/ssd/junlin/data/PhyEditBench/
  bench/
    Deformation_&_Fracture/{brittle_fracture,elasticity,plastic_deformation}/
      meta.json                      # [{id, class, frames, instruction, explain, invariants}, ...]
      {input,intermediate_1,intermediate_2,output}/data_<id>.png
    Fluid_Dynamics/{Buoyancy_&_Tension,Pouring_&_Flow,Splashing_&_Impact}/...
    Rigid_Body_&_Interaction/{Collision_&_Chain,Gravity_&_Fall,Rotation_&_Rolling,Stability_&_Balance}/...
    State_Change_&_Environment/{Diffusion_&_Aerodynamics,Phase_Changes}/...
    anti-physic/
      meta.jsonl                     # {data_id, data_type, sub_id, edit_prompt, expected_phenomenon}
      checklists.jsonl               # {data_id, data_type, sub_id, checklist}
      input_data/data_<id>.png
  gpt_eval.py / gpt_eval_anti.py / utils.py   # upstream's official GPT-judge scorers, for reference
```
Not vendored into this repo (third-party licensed data, 1.3G) — point
`--data_root` at the path above, or re-clone
`https://github.com/Previsior/PhyEditBench` if working from a different
machine (whole repo is only ~2.7G including `.git`, small enough for a plain
`git clone --depth 1`).

## Protocol

Upstream's official protocol (`gpt_eval.py::build_type_fields`) turns each of
the 238 real trajectories into **5 independent generation tasks**:

| Type | Input state | Target state | Instruction |
|:---:|:---|:---|:---|
| TypeA | input | intermediate_1 | step 1 |
| TypeB | intermediate_1 | intermediate_2 | step 2 |
| TypeC | intermediate_2 | output | step 3 |
| TypeD | input | output | steps 1-3 packed sequentially (verbatim upstream wrapper text) |
| TypeE | input | output | global instruction |

`dataset.py` mirrors this exactly (238 * 5 = 1190 real items) plus one item
per anti-physic row (35, `edit_type="anti-physic"`) — **1225 items total**.
Generated-image naming matches upstream's own `resolve_pred_path` /
`gpt_eval_anti.py::resolve_pred_path` verbatim:
`<PrimaryClass>/<SubClass>/<TypeA-E>/<id>.png` and `anti-physic/<data_id>.png`
— not a suite-local convention like `risebench`/`imgedit` use, so this
suite's output tree could be scored directly by upstream's own
`gpt_eval.py --generated_root <out>/images --model_name .` without renaming
anything, if a judge port isn't written.

## Generation

```bash
CUDA_VISIBLE_DEVICES=<g> python inference/gen_phyeditbench_internvlu.py \
    --data_root /scratch/local/ssd/junlin/data/PhyEditBench \
    --out /scratch/local/ssd/junlin/results/PhyEditBench/<Model> --limit 0
```

`--model_path` defaults to the base InternVL-U checkpoint snapshot; point it
at a merged fine-tuned checkpoint dir to evaluate a variant instead. Output:
`<out>/images/<gen_filename>` + resumable `<out>/gen_manifest*.json`. Same
`--shard_idx`/`--num_shards` sharding convention as
`suites/magicbrush/inference/gen_magicbrush_internvlu.py`.

PhyEditBench ships some large source images (up to 8.85MP,
`Rigid_Body_&_Interaction/Collision_&_Chain/input/data_130.png`) that exceed
the InternVL-U pipeline's VAE-decode capacity — the generation script reuses
`suites/risebench/`'s measured 1.8MP-safe downscale cap, and it's
load-bearing here (checked: 987 source images scanned, max 8.85MP, well over
the ~2.1MP crash boundary).

**Known dependency gap** (same as `suites/magicbrush/`): the generation
script imports `internvlu` (`InternVLUPipeline`) from
`Model_Related/InternVLU/InternVL-U` in the old `ssl_mllm` tree via a
hardcoded `PKG` path — not part of this repo's `training/models/internvl-u`
submodule (training fork only). Not fixed by this migration.

## Scoring

Not ported. Upstream's `gpt_eval.py` (normal, 5 types) and `gpt_eval_anti.py`
(Anti-Physics) call GPT-4o (`OPENAI_API_KEY`) to rate each edit 1-10 across 4
dimensions — Consistency, Instruction Following, Physical Plausibility
(highest weight), Image Quality — writing `<ModelName>_normal_scores.jsonl` +
`_normal_summary.json` (`overall`, `by_dimension`, `by_type`, `by_primary`,
`by_primary_sub`) and `<ModelName>_anti_scores.jsonl` + `_anti_summary.json`
(`overall`, `by_dimension`, `by_data_type`) respectively. The staged copy at
`/scratch/local/ssd/junlin/data/PhyEditBench/{gpt_eval.py,gpt_eval_anti.py,
utils.py}` can be run directly against this suite's `<out>/images/` tree
(see Protocol above for why no renaming is needed) once a valid
`OPENAI_API_KEY` is available — see `suites/risebench/README.md`'s note on
this cluster's other project `.env` keys being revoked as of 2026-09-09.

## Validate-then-run

Use `--limit N` on the generation script to smoke-test a handful of items
end-to-end before committing to the full 1225-item run.
