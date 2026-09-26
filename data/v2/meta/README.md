# `data/v2/meta/` — v2 meta jsons

InternVL-trainer meta files for the v2 launchers (`sft/v2/run_t{N}_gen_sft.sh`):
`<run>/T{0..4}_{train,eval}_meta.json` (written by `recipes/temporal_ssl/derive_settings.py`), each
pointing at `../datasets/temporal_ssl/settings/<run>/T{N}_{split}.jsonl` (`task_type: imgen`,
`max_dynamic_patch: 3`).
