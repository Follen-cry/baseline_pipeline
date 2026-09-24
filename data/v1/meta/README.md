# `data/v1/meta/`

InternVL-trainer-format meta jsons — each one maps a dataset name to an `annotation` jsonl
path + a few loader flags (`max_dynamic_patch`, `task_type`, etc). These are what a training
script's `META_PATH` env var points at.

| File | Points at | Used by |
|---|---|---|
| `final_S0_meta.json` / `final_S1_meta.json` / `final_S2_meta.json` | `../datasets/final_s0s3/S{N}_train.jsonl` | `run_s{0,1,2}_gen_sft.sh` |
| `final_S3_meta.json` / `final_S3_eval_meta.json` | `../datasets/final_s0s3/S3_{train,eval}.jsonl` | `run_s3_gen_sft.sh` |
| `vbvr_target_pred_4task_meta.json` | `../datasets/vbvr_target_pred_4task/target_pred_4task_train_no_ce.jsonl` | `run_target_pred_mop_sft.sh` (stage 2, overriding its own default `META_PATH`) |

Full config detail (loss weights, LoRA rank, etc. per stage): `../../PROVENANCE.md` §4-§5.
