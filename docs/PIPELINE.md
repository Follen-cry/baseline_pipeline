# S0-S3 baseline pipeline — narrative walkthrough

For exact file paths, config values, and confirmed-vs-open items, see `../PROVENANCE.md` — this
document is the story; that one is the evidence.

## 1. Data generation

Five VBVR synthetic-task generators (`data/sources/vbvr/generators/`) each produce raw videos.
In parallel, four real-world sources — `epic_kitchens`, `nwm`, `panda70m_epic`, `panda70m_v1` —
each pull from their own raw footage (three of them via their own `anchors`/`splits`-style
stage, panda70m_v1 sharing a module with panda70m_epic's sibling driver).

## 2. Per-source processing + per-source train/eval split

Every source turns its raw footage into rows of a specific shape (context frames + target frame
+ prompt text), **and decides its own train/eval split at this stage** — VBVR splits by sample
window, epic_kitchens by video, nwm by trajectory, both panda70m sources by clip. All five use a
seeded RNG and hold out a disjoint pool before any row-level construction happens, so eval rows
never share source footage with train rows. `data/common/build_real_video_extra_settings.py`
fills in some of the real-world sources' settings (S1 for all three, plus panda70m_epic's S2)
that their own source-specific builders don't natively produce.

Four settings come out the other side per source: S0 (no caption, forward), S1 (no caption,
mixed direction), S2 (real/derived caption), S3 (dual generation + 4-option MCQ, with its own
paired train/eval files).

## 3. Recipe assembly

`data/recipes/s0s3_baseline/merge_final_s0s3.py` takes the pre-split per-source files for a
given setting and interleaves them into one jsonl per setting, tagging each row with its
`source` and `orig_id`. `filter_final_s0s3.py` then drops rows whose referenced images turned
out missing (a real, if small, gap in the raw pipeline). Output: `data/datasets/final_s0s3/`.

## 4. Training — stage 1 (S0-S3 gen-SFT)

Four independent LoRA SFT runs (`training/models/internvl-u/internvl_chat/shell/internvlu/sft/
run_s{0,1,2,3}_gen_sft.sh`), all starting from the same base InternVL-U snapshot — S1/S2/S3 do
**not** continue from S0's checkpoint, despite the naming suggesting a curriculum. Each merges
its LoRA into a `-merged` checkpoint when done.

## 5. Training — stage 2 (target_pred continuation SFT)

Each stage-1 merged checkpoint (plus a no-pretrain `base` control) gets a second SFT pass
(`run_target_pred_mop_sft.sh`) on a small 4-task, 500-example curated set
(`data/datasets/vbvr_target_pred_4task/`). This is the actual object of comparison — stage-1
checkpoints are not separately evaluated.

## 6. Evaluation

`eval/suites/vbvr/` scores each stage-2 checkpoint two ways: deterministic rule-based scorers
(`scorers/`, one per task) and an LLM judge (`judge_eval/`, rubric-based except for
`rotation_puzzle`'s yes/no checklist, which replaced the rubric judge after it saturated).
`eval/suites/vbvr/artifacts/gen_s0s3_4task_comparison.py` pulls both scoring outputs across all
5 checkpoints into one comparison report.
