# InternVL-U training shell scripts

29 scripts, split by role. All are invoked as `bash shell/internvlu/<subdir>/<script>.sh`
from the `internvl_chat/` project root (paths inside these scripts are relative to that
root, not to the script's own location) — see [`CONVENTIONS.md`](../../../../CONVENTIONS.md)
for naming conventions on new scripts.

## `engine/` — shared trainer launchers

The actual `torchrun`/deepspeed entry points. Called *by* the `sft/`/`dualssl/` scripts
below (via `bash shell/internvlu/engine/<name>.sh` with env vars) rather than run directly
in normal use.

| Script | What it trains |
|---|---|
| `internvlu_4b_sft_full_unified.sh` | **The one nearly every current `sft/`/`dualssl/` script calls into.** Full unified SFT: understanding side (text-CE, LoRA on LLM by default) + generation side (flow-matching MSE) jointly, VAE frozen. |
| `internvlu_4b_sft_full_unified_lora.sh` | Unified SFT with LoRA on both the LLM (r16) and generation_decoder (r16) — lighter-memory variant. |
| `internvlu_4b_sft_full.sh` | Full SFT, text-only CE on the understanding side (no LoRA). |
| `internvlu_4b_sft_lora.sh` | LoRA SFT, text-only CE on the understanding side. |

## `sft/v1/` — v1 per-dataset next-frame-generation SFT drivers

The v1 pipeline (S0-S4 and everything built on it; moved here from `sft/` when v1/v2 were split): each script trains on one dataset/stage, then fixes
the saved processor and merges the LoRA into an inference-ready pipeline. All call into
`engine/internvlu_4b_sft_full_unified.sh`.

| Script | Dataset / task |
|---|---|
| `run_s0_gen_sft.sh` | S0 (V2I-F): 3 context frames -> fixed next frame, no caption |
| `run_s1_gen_sft.sh` | S1 (V2I-V): 3 context frames -> next OR preceding frame (stated in prompt) |
| `run_s2_gen_sft.sh` | S2 (VC2I-F): 3 context frames + caption -> fixed next frame |
| `run_s3_gen_sft.sh` | S3 (VC2IA-F): 3 context frames + caption + 4 MCQ candidates -> next frame + answer letter |
| `run_vbvr_gen_sft.sh` | VBVR next-frame split, 10 in-domain tasks x 10k windows (100k rows) |
| `run_target_pred_mop_sft.sh` | VBVR target-frame-prediction on `multi_object_placement` (1-in -> 1-out) |
| `run_epic_gen_sft.sh` | EPIC-KITCHENS-100 `epic_gen_noaction` (4000 train) |
| `run_intphys2_gen_sft.sh` | IntPhys2 qwen-filtered c={1,1.5,2}s (3000 train) |
| `run_maze_gen_sft.sh` | 5x5 maze dataset (`data/datasets/maze_dataset`, 9600 clips), 3 ctx frames -> 4th |
| `run_maze_route_sft.sh` | Maze route-drawing, edit-mode/VAE-conditioned (plain maze -> maze with path) |
| `run_panda3dsr_gen_sft.sh` | Panda-70M `panda3dsr_gen_noaction` (1500 train) |
| `run_reasoning_edit_sft.sh` | Reasoning-edit SFT: KRIS-Bench + RE-Edit + RISE-Bench (948 samples) |

## `dualssl/` — dual-supervision SSL variants

Same datasets as some `sft/` scripts, but each record carries both a diffusion target
(generate the true next frame) AND an understanding letter-CE (pick the true next frame
among 4 options). Also call into `engine/internvlu_4b_sft_full_unified.sh`.

| Script | Dataset |
|---|---|
| `run_epic_dualssl_sft.sh` | EPIC-KITCHENS-100 `epic_dual_noaction` |
| `run_intphys2_dualssl_sft.sh` | IntPhys2 Exp 3 / Exp 4 |
| `run_panda3dsr_dualssl_sft.sh` | Panda-70M `panda3dsr_dual_noaction` |

## `orchestration/v1/` — v1 multi-run drivers

| Script | Role |
|---|---|
| `orchestrate_s0s3_torrnode8.sh` | Runs the 4 S0-S3 `sft/v1/` trainings across one node's free GPUs, launching the next as GPUs free up |
| `run_plausibility_all.sh` | IntPhys2 plausibility (Yes/No) LoRA SFT across 3 base models x {8f,16f}: zero-shot eval -> finetune -> merge -> eval |
| `run_2d_geo_trans_composed_intermediate_sft_queue.sh` | Composed-intermediate (2x2 grid) SFT on `2d_geo_trans_Composed_intermediate` from base + S0-S4 merged checkpoints, queued over free GPUs (stage-2 recipe: `sft/v1/run_target_pred_mop_sft.sh`) |
| `run_composed_intermediate_sft_queue_3tasks.sh` | Same composed-intermediate SFT for `multi_object_placement` / `rotation_puzzle` / `shape_color_then_move` |
| `run_mop_cross_sft.sh` | `multi_object_placement_Composed_intermediate_cross` (3000-row) SFT, one checkpoint per GPU in parallel |

`orchestrate_s0s3_torrnode8.sh` and the 3 composed-intermediate drivers keep `REPO_ROOT`
pointed at the original `ssl_mllm/Model_Related/InternVLU/InternVL/internvl_chat` checkout
they actually ran from, so they still call `shell/internvlu/sft/<script>.sh` (that checkout
was not restructured; its `run_target_pred_mop_sft.sh` and training code are identical to
`sft/v1/` here). The two `run_wm_sft_*` drivers run from this checkout and call `sft/v1/`.

## `legacy/` — earlier/exploratory lineages, not part of the current `sft/`/`dualssl/` pipeline

Kept for reproducibility of past results; don't build new work on these unless you're
specifically extending that lineage.

| Script | What it is |
|---|---|
| `run_intphys2_gen_sft_vae.sh`, `run_intphys2_gen_sft_vae_vit.sh`, `run_intphys2_gen_sft_vit.sh` | Variants of `sft/v1/run_intphys2_gen_sft.sh` that additionally unfreeze the VAE and/or ViT backbone |
| `run_intphys2_implicitleak_sft.sh` | IntPhys2 "implicit-leak" content control (GT next frame appended as unflagged Frame 4) |
| `run_intphys2_implicitleak_plausibility_ft.sh` | Plausibility finetune on top of the implicit-leak control model |
| `train_plausibility.sh`, `train_ffs_v2_1ep.sh` | Self-contained FFS-v2 LoRA SFT engines called by the implicit-leak/plausibility scripts above |
| `train_v2_pipeline.sh` | Orchestrates gen-SFT then ffs-SFT sequentially for the v1-comparable v2 recipe |

## Versions

- **v1** (S0-S4 and the experiments built on it): `sft/v1/`, `orchestration/v1/`. Tag `v1`
  pins the pre-split layout, where these scripts sat directly in `sft/` and `orchestration/`.
- **v2** (temporal SSL, T0-T4): `sft/v2/`, `orchestration/v2/`.
- `engine/`, `dualssl/`, and `legacy/` are shared or pre-v1 and not versioned.

See `baseline_pipeline/docs/VERSIONS.md`.
