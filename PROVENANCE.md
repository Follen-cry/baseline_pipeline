# S0-S3 InternVL-U baseline pipeline — provenance manifest

Traced from the current `ssl_mllm` working tree on 2026-09-05, before any migration into the
new generalized `baseline_pipeline` repo structure. This document records **exactly which
real files/scripts/data paths were used** for the actual S0-S3 run, so migration copies real
provenance rather than idealized equivalents (filenames may still change to fit the new
structure — see repo-structure plan). §6's original ambiguities were resolved in a 2026-09-05
follow-up pass (full reads, not grep) — see §6/§7 for resolutions and the added real-world-
source registry. Stage-1 (S0-S3) training data mixes VBVR with 4 natural-video sources
(`epic_kitchens`, `nwm`, `panda70m_epic`, `panda70m_v1`) — see §7 for full per-source detail.

All paths below are relative to `/scratch/network/ssd2/junlin/ssl_mllm/` unless marked otherwise.

> **2026-09-24 layout note:** when the repo was split into v1/v2 (`docs/VERSIONS.md`),
> everything this document calls `baseline_pipeline/data/<layer>/...` moved to
> `baseline_pipeline/data/v1/<layer>/...`, and the submodule launchers moved from
> `shell/internvlu/{sft,orchestration}/` to `.../{sft,orchestration}/v1/`. Repo-internal
> references below have been updated. Paths into the original `ssl_mllm/` tree
> (`data/datasets/epic_ssl`, `Model_Related/...`) are unchanged, because those locations did not move.

## 1. Data generation (raw) — which VBVR-DataFactory generators feed S0-S3

Only **5 of the 16** `VBVR-DataFactory/*` generators feed the S0-S3 merged dataset — the
`"pilot5"` preset in `VBVR-DataGeneration/vbvr_task_presets.py:25-31`, driven by
`VBVR-DataGeneration/next_frame/vbvr_next_frame_generate.py`:

| VBVR task | Generator dir (under `VBVR-DataFactory/`) | Category |
|---|---|---|
| `multi_object_placement` | `G-5_multi_object_placement_data-generator` | Perception |
| `shape_color_then_move` | `O-11_shape_color_then_move_data-generator` | Abstraction |
| `shape_outline_then_move` | `O-13_shape_outline_then_move_data-generator` | Abstraction |
| `rotation_puzzle` | `O-44_rotation_puzzle_data-generator` | Transformation |
| `2d_geometric_transformation` | `O-6_2d_geometric_transformation_data-generator` | Transformation |

The other 11 generators (`ball_bounces_given_time`, `mirror_reflection`,
`shape_color_then_scale`, `grid_shortest_path`, `key_door_matching`, `grid_shift`,
`stable_sort`, `maze`, `animal_size_sorting`, plus `glass_refraction`/`gravity_physics` which
have no train jsonl) feed a **separate, unrelated** "10 canonical + 4 OOD-ext" main training
set (`data/datasets/vbvr_next_frame/next_frame_train_no_ce.jsonl`, 140k rows, used by
`run_vbvr_gen_sft.sh`) — confirmed by `data/datasets/vbvr_next_frame/README.md:10-43`. **Not**
part of S0-S3; exclude from the S0-S3 provenance when migrating.

Confirmed pool size: **5 tasks × 5,000 windows = 25,000 rows**, shared verbatim across
S0/S1/S2/S3, per `data/datasets/vbvr_next_frame/sampling_summary.json` and its README.

`data/v1/meta/final_S{0,1,2,3}_meta.json` (+ `final_S3_eval_meta.json`) each point to one
`data/v1/datasets/final_s0s3/S{N}_{train,eval}.jsonl` file (`task_type: "imgen"`), row counts:
49,113 / 49,113 / 49,113 / 47,770 (train) + 2,075 (S3 eval).

- `data/scripts/filter_final_s0s3.py` — post-merge cleanup: drops rows in
  `final_s0s3/*.jsonl` referencing missing image files (~2.7% loss, from
  `build_panda70m_epic_ssl.py`'s cv2 seek failures), rewrites files in place with a `.bak`
  backup, updates the `length` field in each `data/v1/meta/final_S{N}*_meta.json`.
- `data/scripts/merge_final_s0s3.py` — the actual dataset assembler. Merges VBVR + 4
  real-world sources per setting:

  | Setting | vbvr file | epic (`build_epic_ssl.py`) | nwm (`build_nwm_ssl.py`) | panda70m_epic (`build_panda70m_epic_ssl.py`) | panda70m_v1 (`build_panda70m_ssl_v1_driver.py` → `build_panda70m_ssl.py`) |
  |---|---|---|---|---|---|
  | S0 | `vbvr_next_frame/vbvr_S0_train_nocap.jsonl` | `epic_ssl/epic_gen_noaction/*_train.jsonl` | `nwm_ssl/nwm_gen_noaction/*_train.jsonl` | `panda70m_epic_ssl/panda70m_gen_noaction/*_train.jsonl` | `panda70m_ssl_v1/panda70m_v1_gen_S0/*_train.jsonl` |
  | S1 | `vbvr_S1_train_mixed_nocap.jsonl` | `epic_gen_S1/*_train.jsonl` | `nwm_gen_S1/*_train.jsonl` | `panda70m_gen_S1/*_train.jsonl` | `panda70m_v1_gen_S1/*_train.jsonl` |
  | S2 | `vbvr_S2_train.jsonl` | `epic_gen_action/*_train.jsonl` | `nwm_gen_action/*_train.jsonl` | `panda70m_gen_S2/*_train.jsonl` | `panda70m_v1_gen_S2/*_train.jsonl` |
  | S3 train/eval | `vbvr_S3_{train,eval}.jsonl` | `epic_dual_action/*_{train,eval}.jsonl` | `nwm_dual_action/*_{train,eval}.jsonl` | `panda70m_dual_noaction/*_{train,eval}.jsonl` | `panda3dsr_dual_noaction/*_{train,eval}.jsonl` |

  All paths rooted at `data/v1/datasets/`. Row `source`/`orig_id` stamped by
  `load_and_tag`/`write_merged`. `max_dynamic_patch=3` for S0/S1/S2, `7` for S3 (3 ctx + 4 MCQ
  options).

## 2. Data processing (`VBVR-DataGeneration/next_frame/*`)

- `vbvr_next_frame_generate.py` — drives each generator via `examples/generate.py`
  subprocess, chunked (`DEFAULT_CHUNK_SIZE=500`) to avoid OOM. Output raw videos to
  `/scratch/network/ssd/junlin/vbvr_next_frame/raw/<task>/chunk_NNNN/...` — **on a different
  volume than this repo**, so migration needs an explicit copy/restructure step, not a
  relative-path move. `RAW_ROOT = /scratch/network/ssd/junlin/vbvr_next_frame/raw`,
  `FACTORY_ROOT = VBVR-DataFactory/`. All 5 S0-S3 tasks use 11,000 raw videos each.
- `vbvr_next_frame_sample.py` — duration-adaptive 3-in-1-out window sampling from raw videos
  (`rglob("ground_truth.*")`), writes to
  `SAMPLES_ROOT = /scratch/network/ssd/junlin/vbvr_next_frame/samples/<task>/<id>/`
  (`frame_0/1/2.png`, `target.png`, `window_meta.json`). Supports `--no-ce`,
  `--direction {forward,backward,mixed}`, `--reuse-samples` (cheap rebuild without
  re-decoding — used for S1's mixed direction and other wrapping variants).
- `vbvr_next_frame_prompt_adapt.py` — per-task "predict the next frame" prompt text from
  `metadata.json` params (production copy of `data/scripts/vbvr_prompt_adapt.py`, logic
  unchanged).
- `vbvr_next_frame_make_dual.py` — builds S3's dual MCQ+generation rows (4-option MCQ: 1
  ground truth + 3 distractors from other windows of the same task). **This is also the real
  train/eval split point** — see §3.
- `VBVR-DataGeneration/vbvr_task_presets.py` — defines `TASK_PRESETS["pilot5"]`, the
  authoritative 5-task list for S0-S3's shared VBVR pool (`resolve_tasks()`), more specific
  than any per-script `--tasks` default.
- Earlier smoke-test versions `data/scripts/vbvr_generate_raw.py`, `vbvr_prompt_adapt.py`,
  `vbvr_sample_windows.py` — left in place as reference, confirmed **not** used for the
  production S0-S3 run (`VBVR-DataGeneration/README.md:10-12`). Do not migrate as if they were
  load-bearing.
- Real-world source builders (each independently invoked, not chained through
  VBVR-DataFactory): `data/scripts/build_epic_ssl.py` (EPIC-KITCHENS-100, 116 local MP4s),
  `build_nwm_ssl.py`, `build_panda70m_epic_ssl.py`
  (`FRAMES_ROOT=/scratch/network/ssd/junlin/panda70m_epic_ssl_frames`),
  `build_panda70m_ssl_v1_driver.py` (thin CLI wrapper around `build_panda70m_ssl.py`'s
  `splits`/`anchors`/`ffs`/`dual`/`gen` stages).
- **`data/scripts/build_real_video_extra_settings.py`** — a 6th, generic script that is the
  *actual* producer of 4 of the 20 cells in the §1 merge table: `epic_gen_S1`, `nwm_gen_S1`,
  `panda70m_gen_S1`, and `panda70m_gen_S2`. None of the 4 source-specific builders above
  contain their own S1 (`mixed` direction) logic or, for panda70m_epic, S2 (`--with-caption`)
  logic — this script fills that gap generically across sources. Load-bearing; do not migrate
  the 4 source builders alone and assume they're self-sufficient for S0-S3.

## 3. Train/eval split

- **S0/S1/S2**: no held-out eval, by project convention — `merge_final_s0s3.py`'s module
  docstring states this explicitly ("S0/S1/S2 = train only, no eval holdout by project
  convention"); `run_s0_gen_sft.sh`/`run_s1_gen_sft.sh` headers confirm "No held-out eval
  wired in for S0/S1 (none exists for S0-S2)."
- **S3 only**: split happens *upstream*, inside `vbvr_next_frame_make_dual.py`'s
  `split_window_dirs()` — a **window-level** (not row-level) random split
  (`rng = random.Random(f"{seed}:holdout")`), `EVAL_COUNT_PER_TASK = 200` windows/task held
  out. Eval-holdout distractors are drawn **only from other eval windows**, never mixing with
  train, so no eval image ever leaks into a training row (even as a wrong MCQ option). The
  same convention (paired `_train.jsonl`/`_eval.jsonl`) applies to the 4 real-world sources'
  `_dual_action`/`_dual_noaction` builders for S3.
- Confirmed on-disk contents of `data/v1/datasets/final_s0s3/`: `S0_train.jsonl` (49,113 rows),
  `S1_train.jsonl` (49,113), `S2_train.jsonl` (49,113), `S3_train.jsonl` (47,770),
  `S3_eval.jsonl` (2,075) — counts already reflect `filter_final_s0s3.py`'s post-merge drop
  (~2.7% loss concentrated in `panda70m_epic_ssl` rows).

## 4. Training stage 1 (S0-S3 gen-SFT)

All 4 scripts share one template
(`Model_Related/InternVLU/InternVL/internvl_chat/shell/internvlu/engine/internvlu_4b_sft_full_unified.sh`)
and are **independent** — each defaults `INTERNVLU_CKPT` to the same `BASE_SNAPSHOT`; none
chains from another stage's output (confirmed: the orchestrator never overrides
`INTERNVLU_CKPT`).

- `BASE_SNAPSHOT` (all 4):
  `/scratch/network/ssd2/junlin/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01`
- `run_s0_gen_sft.sh`: `META_PATH=data/v1/meta/final_S0_meta.json`,
  `OUTPUT_DIR=/scratch/network/ssd2/junlin/models/internvlu-s0-gen` (+`-merged`),
  `LM_LOSS_WEIGHT=0.0` (no-caption → text-CE carries no signal), `GEN_LOSS_WEIGHT=0.5`
  (warmup 20), `IMGEN_RATIO=1.0`, `GEN_IMAGE_SIZE=512`, `LORA_RANK=32`, `BATCH_SIZE=16`,
  `EPOCHS=1`, `LR=1e-5`, `INTERNVLU_VAE_COND=1`.
- `run_s1_gen_sft.sh`: same recipe, `META_PATH=data/v1/meta/final_S1_meta.json`,
  `OUTPUT_DIR=.../internvlu-s1-gen`, `LM_LOSS_WEIGHT=0.0`.
- `run_s2_gen_sft.sh`: `META_PATH=data/v1/meta/final_S2_meta.json`,
  `OUTPUT_DIR=.../internvlu-s2-gen`, **`LM_LOSS_WEIGHT=0.0`** (confirmed by full read — same
  default as S0/S1; S2's caption sits in the masked human turn, the supervised GPT turn is
  still the fixed phrase, so 0.0 is correct despite S2 having real caption text). All other
  values (GPUS=8, EPOCHS=1, LR=1e-5, LORA_RANK=32, BATCH_SIZE=16, GEN_IMAGE_SIZE=512,
  GEN_LOSS_WEIGHT=0.5/warmup 20, IMGEN_RATIO=1.0, INTERNVLU_VAE_COND=1) are identical across
  S0/S1/S2.
- `run_s3_gen_sft.sh`: `META_PATH=data/v1/meta/final_S3_meta.json`,
  `OUTPUT_DIR=.../internvlu-s3-gen`, **`LM_LOSS_WEIGHT=0.5`** (S3's MCQ answer-letter CE is
  real supervision, unlike S0-S2). **Doc/code mismatch, flag for the new repo**: S0/S1/S2's
  own script comments claim "S3 keeps LM_LOSS_WEIGHT=1.0"; the actual `run_s3_gen_sft.sh`
  default and the real run log (`s3_blockA_manual_20260901_163947.log`, `lm_loss_weight=0.5`)
  both confirm **0.5** is what really ran. Trust the code+log, not the S0-S2 comments — don't
  let a recipe doc in the new repo silently inherit the wrong value.
- All 4: post-train steps replace the degenerate saved processor with the base snapshot's,
  then `tools/merge_lora_u_full.py` merges LoRA into `<OUTPUT_DIR>-merged`, verified 0 LoRA
  keys remain.

`Model_Related/InternVLU/InternVL/internvl_chat/shell/internvlu/orchestration/orchestrate_s0s3_torrnode8.sh`
— polling launcher across torrnode8's 8 A40 GPUs (Block A=`0,2,3,4`, Block B=`1,5,6,7`), 180s
poll interval, launches `run_${setting}_gen_sft.sh` with `CUDA_VISIBLE_DEVICES`/`GPUS` set but
**no** `INTERNVLU_CKPT` override.

**Actual run history** (log-cited — diverges from a naive "one clean orchestrator run"
reading of the script; per
`Model_Related/InternVLU/InternVL/internvl_chat/logs/s0s3_torrnode8/orchestrator.log` and
per-job logs):

- **S0**: real successful run = `s0_blockA_20260831_143625.log` (started 2026-08-31 14:36,
  ran through 23:58, ~9.5h). An earlier same-day attempt at 14:31 was a false start
  (`job_alive()` bug, documented in the script's own comments, already fixed by the time of
  the real run).
- **S1/S2/S3**: a first orchestrator pass on 2026-09-01 00:02–00:16 launched all three
  back-to-back and **all failed within ~5 min each** (`exit=1`, torchrun `ChildFailedError` at
  rank 2 — likely stale GPU state right after S0 finished at 23:58). Logs:
  `s1_blockA_20260901_000248.log`, `s2_blockA_20260901_000733.log`,
  `s3_blockA_20260901_001205.log`.
- Real successful runs, all launched later that same day:
  `s1_blockA_20260901_073959.log` (orchestrator-launched, exit=0, ~9h),
  `s2_blockB_manual_20260901_075110.log` (launched **manually**, not by the orchestrator, in
  parallel on Block B), `s3_blockA_manual_20260901_163947.log` (launched **manually** on Block
  A after S1 freed it, 16:39 → 2026-09-02 02:20).
- All 4 final merged checkpoints confirmed via
  `[verify] merged vlm: 0 lora keys / 658 total -> OK` and exist on disk:
  `/scratch/network/ssd2/junlin/models/internvlu-s{0,1,2,3}-gen-merged/`.

## 5. Downstream stage 2 + eval

- `Model_Related/InternVLU/InternVL/internvl_chat/shell/internvlu/sft/run_target_pred_mop_sft.sh`
  — stage-2 continuation SFT. Default
  `META_PATH=data/meta/vbvr_target_pred_id_mop_meta.json`, but per `CHECKPOINTS.md` the actual
  stage-2 4-task runs override
  `META_PATH=data/v1/meta/vbvr_target_pred_4task_meta.json` (→
  `data/v1/datasets/vbvr_target_pred_4task/target_pred_4task_train_no_ce.jsonl`, 2,000 rows = 4
  tasks × 500). `INTERNVLU_CKPT` set per-run to each `internvlu-s{N}-gen-merged` (read-only,
  script refuses if `OUTPUT_DIR == INTERNVLU_CKPT`). `LM_LOSS_WEIGHT=0.0`, `LORA_RANK=32`,
  `LR=1e-5`, `BATCH_SIZE=16`, 1 epoch.
- Confirmed checkpoints on disk: `internvlu-{base,s0,s1,s2,s3}-4task500sft(-merged)` under
  **`/scratch/network/ssd/junlin/models/`** (note: no "2" — a different volume than stage-1's
  checkpoints, which live on `ssd2`; do not conflate the two when writing a checkpoint index
  for the new repo).
- **Exact training invocations now confirmed** (previously unlogged — found at
  `results/vbvr_9task_sft_logs/s0s3_followup/`, a full second orchestrator run
  (`pipeline.log`/`orchestrator.log`, tag `[followup]`/`[base-followup]`) with per-stage
  `{base,s0,s1,s2,s3}_{sft,infer,score_rule,score_judge}.log`). Confirmed pattern: each
  `run_target_pred_mop_sft.sh` invocation sets `model_name_or_path=<internvlu-s{N}-gen-merged>`
  (or the raw `BASE_SNAPSHOT` for `base`, i.e. no stage-1 pretrain), `output_dir` the matching
  `internvlu-s{N}-4task500sft`, `meta_path=data/v1/meta/vbvr_target_pred_4task_meta.json`,
  `num_train_epochs=1`, `learning_rate=1e-5`, `use_llm_lora=32` — otherwise identical across
  all 5 runs. `pipeline.log` also documents real hiccups worth knowing about if this path is
  re-run: an `s1` pretrain-dependency wait (blocked on the buggy first stage-1 orchestrator
  attempt), an `s1_infer` crash requiring a restart, and a hand-applied `model_index.json`
  "missing processor entry" patch to the s2/s3 merged dirs before inference. None of this
  affects the final scored numbers — all 5 stages reached `COMPLETE`.
- `Evaluation/VBVR-CustomEval/artifacts/gen_s0s3_4task_comparison.py` — the actual comparison
  script (name unchanged by the recent "generalize judge scoring" commit). Reads, per model,
  `results/vbvr_target_pred_eval/{base,s0,s1,s2,s3}_4task500sft/scored_new.json` (rule-based)
  and `judge_scored.json` (judge-based); `RULE_FILE = "scored_new.json"` — the 2026-09-03
  rewritten scorers; the old `scored.json` is explicitly noted elsewhere as superseded /
  unreliable, don't migrate as if current. `rotation_puzzle` uses a separate yes/no checklist
  judge (`judge_scored_yesno.json`) instead of the weighted rubric judge (which had saturated
  on it).
- Scorer code paths: `Evaluation/VBVR-CustomEval/validation/score_target_pred_eval_v2.py`
  (driver) → `Evaluation/VBVR-CustomEval/scorers/{rotation_puzzle,multi_object_placement,
  shape_color_then_move,2d_geometric_transformation}.py` (+ shared `cvlib.py`, `harness.py`).
- Judge code paths: `Evaluation/VBVR-CustomEval/judge_eval/score_all_with_judge.py
  --run-dir <dir>` → imports `llm_judge_full.FullVLMJudge`/`TASK_CRITERIA` from
  `Evaluation/VBVR-CustomEval/evaluators/llm_judge_full.py`, and `judge_yesno.py` (same dir)
  for the rotation_puzzle checklist path.
- Output artifact: `Evaluation/VBVR-CustomEval/artifacts/s0s3_4task_comparison.html`.

## 6. Ambiguities — resolved (2026-09-05 follow-up pass, full reads not just grep)

1. **Stage-2 4-task-500 training logs: FOUND.** `results/vbvr_9task_sft_logs/s0s3_followup/`
   (missed in the first pass — under `results/`, not `logs/` or the checkpoint dirs). Exact
   invocations now confirmed — see §5 above.
2. **S2's `LM_LOSS_WEIGHT`: confirmed `0.0`** by full read of `run_s2_gen_sft.sh` — see §4
   above (also surfaced a real S3 doc/code mismatch, flagged there).
3. **`next_frame_train.jsonl`: confirmed truly unreferenced** by any executable path.
   Repo-wide grep (all extensions) found only: its own README (describes it as unused);
   `vbvr_next_frame_sample.py`'s docstring (documents it as that script's own *default write
   target*, not a read-reference); `vbvr_target_pred_merge_9task.py`'s docstring (mentions it
   only for a superseded diffing method); and one **stale** mention in
   `Evaluation/VBVR-CustomEval/artifacts/gen_9task_sft_comparison.py:703` (generated-HTML
   prose describing "10 tasks × 10,000 rows," a row count that actually matches
   `next_frame_train_no_ce.jsonl` — 100,000 rows on disk — not this file — confirmed 25,000
   rows, 5-task pool). The path was evidently reused/overwritten by the S0-S3 pilot5 pool
   after that report's prose was written and the prose was never updated. No code anywhere
   `open()`s this file today. **Confirmed excludable** from `data/v1/sources/vbvr/`.
4. **`.bak` mtime forensics: high-confidence single run.** Mtimes cluster in one tight,
   monotonic ~20-second window on 2026-08-31 (`S0_train.jsonl.bak` 12:00:28.30 →
   `S0_train.jsonl` 12:00:40.99 → `S1...` → `S2...` → `S3_train...` → `S3_eval...`,
   ending 12:00:48.84), exactly matching `filter_final_s0s3.py`'s `FILES` dict iteration
   order — the signature of one sequential execution, not multiple re-runs (which would show
   non-monotonic or widely-separated timestamps). Not provable from logs/git (`data/` is
   entirely untracked in this repo — `git status` shows it all as `??`), but the mtime
   evidence is strong. **Conclusion: ran exactly once**, treat current on-disk
   `S{0,1,2,3}_train.jsonl`/`S3_eval.jsonl` as the single, final filtered output.
5. **Real-world source builders: all 5 scripts fully read** (including a 6th script missed in
   the first pass — see the `build_real_video_extra_settings.py` note added to §2). Full
   per-source detail — raw data locations, stage structure, train/eval split mechanism, output
   files, sizes — now in §7 below.

## 7. Real-world source registry (`data/v1/sources/` candidates beyond `vbvr`)

Stage-1 (S0-S3) training data is **not VBVR-only** — every one of `merge_final_s0s3.py`'s
S0/S1/S2/S3 settings mixes the VBVR pilot5 pool with 4 natural-video sources. These are
first-class citizens of the new repo's `data/v1/sources/` registry, same detail level as VBVR.

### `epic_kitchens` — `data/scripts/build_epic_ssl.py`
- **Raw data**: source MP4s (116 videos, EPIC-KITCHENS-100 validation split) live only on
  `torrnode11`'s local SSD:
  `/scratch/local/ssd/junlin/data/worldprediction/epic-kitchen/<P##>/<P##_##>.MP4`
  (downloaded by `Evaluation/WorldPrediction/download_epic_kitchen.sh`; the `anchors` stage
  asserts it's running on that node via `_assert_on_source_node()`). Extracted frame JPGs go
  to network storage `/scratch/network/ssd/junlin/epic_ssl_frames/` (readable from any node —
  **this is what the training jsonls actually reference**, not the raw MP4s). Measured:
  `epic_ssl_frames` = 22G; local raw MP4 root = 167G.
- **Stages**: `splits → anchors → ffs → gen → dual`. Only `anchors` touches video (must run on
  torrnode11); `ffs`/`gen`/`dual` only touch jsonls (portable). Narration matched from
  `EPIC_100_validation_local116.csv` (real per-frame verb+noun annotations — unlike the other
  3 sources, which have no native narration).
- **Train/eval split**: **video-level**. `build_trajectory_splits()`:
  `random.Random(SPLIT_SEED=42).shuffle(video_ids)`, `EVAL_HOLDOUT_VIDEOS=20` of 116 held out
  whole (assert no overlap); anchors are then compiled and subsampled independently within
  each pool (`TARGET_TRAIN=7000`, `TARGET_EVAL=300`, per-video caps 100/25, `ANCHOR_SEED=0`) —
  eval rows can never share a source video with train rows.
- **S0-S3 outputs**: `epic_gen_noaction/{train,eval}.jsonl` (S0, 7000 train, native
  `build_epic_ssl.py --stage gen`); `epic_gen_S1/epic_gen_S1_train.jsonl` (S1, **via
  `build_real_video_extra_settings.py`**, 7000 rows, `direction: mixed`); `epic_gen_action/
  {train,eval}.jsonl` (S2, real narration text as caption, native `--stage gen`);
  `epic_dual_action/{train,eval}.jsonl` (S3, 7000/300, via `make_nwm_dual_ssl.py` converter
  reused unmodified from NWM).

### `nwm` (RECON) — `data/scripts/build_nwm_ssl.py`
- **Raw data**: `RECON_DATA = /scratch/network/ssd/junlin/nwm_data/recon` (network storage,
  26G — images referenced in place, never copied). Trajectory name lists come from a sibling
  repo, `Evaluation/nwm/data_splits/recon/{train,test}/traj_names.txt` (9468 train / 2367 test
  trajectories), imported via a deliberate one-off `sys.path` insert into
  `Evaluation/nwm/vlm_benchmark/core.py` (its own docstring flags this as a narrow, one-off
  exception to the usual `Evaluation/` depends-on-`data/` direction — worth deciding how to
  handle in the new repo rather than silently carrying the reverse dependency over).
- **Stages**: `splits → anchors → ffs → gen → dual`, plus an extra `edit` stage
  (single-image-editing reframe, `nwm_edit_action`) that is **not** consumed by
  `merge_final_s0s3.py` — exclude from S0-S3 provenance.
- **Train/eval split**: **trajectory-level**, only from RECON's official **train** list
  (asserts zero overlap with RECON's official test list, reserved for the separate NWM eval
  benchmark). `SPLIT_SEED=42` shuffle, `EVAL_HOLDOUT_TRAJ=500`. `TARGET_TRAIN=5284`
  (deliberately capped well below RECON's ~22k ceiling — comment cites the exact real-world
  mix budget: v1=7016, EPIC=7000, panda70m_epic=5700, NWM=5284 ≈ 25k total, matching VBVR's
  5-task × 5000), `TARGET_EVAL=300`, `PER_TRAJ_CAP=2`.
- **Known leak caveat** (documented in the script, carry into new repo's docs): NWM's
  `horizon==stride==8` chains consecutive sliding windows, so ~89% of `nwm_dual_*` rows have
  `cond_image` pixel-duplicating a wrong MCQ *option* (never the ground-truth answer — never
  equal to `target_image` — but not pixel-disjoint from every option, unlike EPIC/panda70m).
- **S0-S3 outputs**: `nwm_gen_noaction/{train,eval}.jsonl` (S0, 5284 train);
  `nwm_gen_S1/nwm_gen_S1_train.jsonl` (S1, **via `build_real_video_extra_settings.py`**, 5284
  rows); `nwm_gen_action/{train,eval}.jsonl` (S2, pose-derived action text, reused from the
  NWM eval benchmark's validated pose→action math); `nwm_dual_action/{train,eval}.jsonl` (S3,
  5284/300).

### `panda70m_epic` — `data/scripts/build_panda70m_epic_ssl.py`
- **Raw data**: clips downloaded via `yt-dlp` (stream-copy,
  `--extractor-args youtube:player_client=android` to dodge bot-detection) to node-local SSD
  `CLIPS_ROOT = /scratch/local/ssd/junlin/data/datasets/panda70m_epic_ssl/clips` (node-pinned,
  not reachable from this session). Extracted frames → network storage
  `FRAMES_ROOT = /scratch/network/ssd/junlin/panda70m_epic_ssl_frames` (1.7G). Source pool:
  `data/datasets/panda70m/epickitchen_filter/stage2_judged.jsonl` (Panda-70M training_10m
  clips, regex + Qwen3-VL-30B-AWQ-judge-filtered to kitchen+manipulation,
  `desirable_filtering=='desirable'`, `N_DOWNLOAD_SAMPLE=700` sampled). **Honesty caveat in the
  script itself**: despite the "epic" name this is **not actually egocentric** footage — a
  manual spot check found only ~25-30% of clips have hands/object-dominant framing; prompts
  deliberately avoid claiming "first-person"/"egocentric."
- **Stages**: `download → anchors → gen → dual` — **no standalone `splits`/`ffs` stage**
  (unlike EPIC/NWM); an FFS-shaped intermediate is written to `_ffs_internal/` purely so
  `dual` can convert it, never registered/shipped as its own task. `anchors` does NWM-style
  dense sliding-window extraction per clip (`STRIDE=4`, `PER_CLIP_CAP=20`), unlike EPIC's
  single-transitions-per-long-video approach.
- **Train/eval split**: **clip-level**, decided inside `build_anchors()` itself (no separate
  splits file, unlike EPIC/NWM): `rng_split = random.Random(SPLIT_SEED=42)` shuffles eligible
  clip IDs, `n_eval_clips = round(0.15 * len(clips))` (a **fraction**, not a fixed count —
  unique among the 4 sources), rest to train, assert disjoint. `TARGET_TRAIN=5700`,
  `TARGET_EVAL=300`.
- **S0-S3 outputs**: `panda70m_gen_noaction/{train,eval}.jsonl` (S0, 5700 train, native
  `--stage gen`); `panda70m_gen_S1/panda70m_gen_S1_train.jsonl` (S1, **via
  `build_real_video_extra_settings.py`**, 5700 rows); `panda70m_gen_S2/
  panda70m_gen_S2_train.jsonl` (S2, **also via `build_real_video_extra_settings.py`**, with
  `--caption-metadata-key caption` — this source has no native action/caption gen variant of
  its own, exactly why the generic script exists); `panda70m_dual_noaction/{train,eval}.jsonl`
  (S3, 5700/300).

### `panda70m_v1` — `data/scripts/build_panda70m_ssl_v1_driver.py` (thin config-override
wrapper) → `data/scripts/build_panda70m_ssl.py` (shared module, monkey-patched via
`m.OUT`/`m.LOCAL_CLIP_ROOT`/`m.FRAMES_ROOT` reassignment)
- **Raw data**: `V1_CLIP_ROOT = /scratch/local/ssd/junlin/data/datasets/panda70m/
  3dsrbench_sample5000` (node-local, video2dataset sharded format, 1,073 COCO/3DSRBench-
  filtered clips — not reachable from this session). `V1_FRAMES_ROOT = /scratch/network/ssd/
  junlin/panda70m_ssl_v1_frames` (1.7G). Reuses an existing `clip_manifest.json` from the
  older `data/datasets/panda70m_ssl/` pipeline (copied in, never regenerated —
  `build_clip_manifest()` is never called for v1, since it now expects a different,
  incompatible sidecar format).
- **Stages**: `splits → anchors → ffs → dual → gen` — note **`gen` comes last** here (unlike
  the other 3 sources), invoked repeatedly with different flags per setting.
- **Train/eval split**: **clip-level**, in the shared module's `build_clip_splits()`:
  `random.Random(SPLIT_SEED=42).shuffle`, `EVAL_HOLDOUT_CLIPS` (v1 override: 130, ~12% of
  ~1068 valid clips) held out whole. Per the module's own comment: **S0/S1/S2 (`gen`) take NO
  eval holdout at all**, project-wide convention — `build_gen()` unconditionally reads only
  `anchors_train.jsonl`, never `anchors_eval.jsonl`; the held-out clips exist solely to
  populate S3's MCQ eval pool.
- **Direction/caption handled natively** (unlike panda70m_epic, which needed the generic
  script): `build_panda70m_ssl.py`'s own `build_gen(direction, has_caption, setting_label,
  out_stem, seed)` supports all 3 settings directly; the v1 driver's CLI flags pass straight
  through: `--stage gen --direction forward --setting-label S0 --out-stem
  panda70m_v1_gen_S0` (S0, confirmed); S1/S2 inferred from the pattern + on-disk headers
  (`--direction mixed`/`--with-caption`), not separately log-confirmed.
- **S0-S3 outputs** (`TARGET_TRAIN=7016` ceiling, but **actual on-disk count is 6254** for all
  of S0/S1/S2 — the achieved count fell short of the stated ceiling, consistent across all
  three since they share one `anchors_train.jsonl`): `panda70m_v1_gen_S0/..._train.jsonl`,
  `panda70m_v1_gen_S1/..._train.jsonl`, `panda70m_v1_gen_S2/..._train.jsonl` (6254 each);
  `panda3dsr_dual_noaction/{train,eval}.jsonl` (S3, 6254/200 — **note this dir/file is named
  `panda3dsr_*`, not `panda70m_v1_*`**, an inconsistent naming artifact carried from the older
  v1 pipeline; migration should be aware the S3 pair breaks the `panda70m_v1_gen_S{N}` naming
  pattern the other 3 settings use).

### Cross-source split-mechanism summary (for the new repo's `data/splits/` design)
All 4 real-world sources use a **seeded random split at a coarser-than-row granularity**
(video-level for EPIC, trajectory-level for NWM, clip-level for both panda70m sources), always
`SPLIT_SEED=42`, always producing a disjoint train/eval pool *before* anchors/rows are
generated — structurally the same shape as VBVR's own window-level split (§3), just at a
different granularity name. A generic `data/splits/` layer can plausibly handle all 5 sources
uniformly if parameterized on "disjointness unit" (window/video/trajectory/clip) rather than
hardcoded to one — **it should never assume row-level splitting**, since none of the 5 sources
split at that granularity.
