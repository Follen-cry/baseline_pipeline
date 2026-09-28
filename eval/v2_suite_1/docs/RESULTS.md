# Results guide

Every file under `results/`, what it holds and which script writes it. Scores are per the benchmark's
native metric unless a column says *norm*: the benchmark's primary metric rescaled to 0–1 of its range
(PhyEditBench / Anti-Physics overall 1–10, PICABench Acc 0–1, RISE Score 1–5, ImgEdit / UGE /
MagicBrush VLM 1–5). "Models" are base, T0, T2, T3; reference outputs are `copy` (input unchanged),
`random` (an unrelated image) and `gt` (ground truth), which are scored like models but never counted
as models. Diagnosis-only systems are `<model>_editsft` (after the probe fine-tune) and `base_nullft`
(learning-rate-0 control).

## Reports

| File | Written by | Content |
|---|---|---|
| `REPORT.md` | `pipeline/report.py` | Main evaluation report: setup, group summary, verdicts, native tables, paired tests, PhyEditBench by type, Anti-Physics, per-category tables, judge reliability, example grids, conclusions |
| `DIAGNOSIS.md` | `diagnosis/build_diagnosis.py` | Why the trained models score below base: ranked recommendation, control check, evidence for/against each cause, code review |
| `eval_report.html` (git-ignored) | `results/build_artifact.py` | Self-contained shareable page (both reports, charts, filterable examples); ~15 MB |
| `artifact_template.html`, `conclusions.json` | hand-written | Page template and the conclusions text (with `{{...}}` placeholders resolved from the CSVs) |
| `grids/*.jpg` | `pipeline/report.py` | Input / base / T0 / T2 / T3 / reference grids with judge scores, used by REPORT.md |

## Main evaluation tables (`pipeline/analyze.py` unless noted)

| File | Rows | Content |
|---|---|---|
| `scores_item.csv` | one per (task, model, metric) | Every per-item score incl. rule metrics and edit size; columns `uid, model, metric, value, bench, group, category, subcategory, etype, cluster`. The source for everything else |
| `table_main.csv` | one per model (+ copy) | Model × benchmark native metrics, plus mean edit size |
| `table_category.csv` | model × benchmark × category | Native metric per category / subcategory |
| `table_phyedit_type.csv` | model × Type A–E | PhyEditBench weighted overall and the 4 dimensions |
| `table_anti.csv` | model × rule type | Anti-Physics overall and 4 dimensions per counterfactual rule |
| `paired.csv` | benchmark × metric × level × model | trained (or copy) − base: mean delta, cluster-bootstrap 95% CI, Wilcoxon p, Holm p, realised MDE, at benchmark / group / category / subcategory / type level |
| `group_summary.csv` | evidence group × model | Pooled normalised delta per group (A, B, C, post-hoc A_copy_sensitive) with CI and Holm p |
| `trend.csv` | group × model | Normalised group means (T0 → T2 → T3 trend, base and copy for reference) |
| `parse_stats.csv` | judge × benchmark × model | Judged units, parsed first try, parsed after retry, residual failures |
| `judge_agreement.csv` | comparison × benchmark | 5% sanity subset: same judge re-run and Qwen3-VL-8B vs the main judge (exact / ±1 agreement, Spearman or κ, model-rank Kendall τ); written by `pipeline/agreement.py` |
| `examples.csv` | one per task | Normalised primary metric per model, mean trained − base delta, win/loss tag; `pipeline/report.py` |
| `verdicts.csv` | one per group | Pre-registered verdict (supports / refutes / inconclusive, controls: no harm / harm); `pipeline/report.py` |
| `suite_counts.csv`, `power_planned.csv`, `benchmarks_considered.csv` | – | Suite composition, planned MDEs and the benchmark include/skip list for the page; `pipeline/report.py` |

## Raw judge outputs: `judge_raw/<judge>/<benchmark>/<model>.jsonl`

`judge.py` writes one row per judged unit (a PhyEditBench dimension, a PICABench QA, a RISE item, …)
with the raw response, parsed score(s), `parse_ok` and every attempt in `tries`. Re-running skips keys
already present. Judges: `qwen3vl30b_fp8` (main; all models, copy/random/gt, `*_editsft`),
`qwen3vl30b_fp8_rerun` and `qwen3vl8b` (5% sanity subset only).

## Rule metrics: `rule/`

| File | Written by | Content |
|---|---|---|
| `pica.csv` | `pipeline/rule_metrics.py --part pica` | PICABench non-edited-region PSNR (official `PicaEval_consistency`, 512); items whose edit area covers the whole image have no value (official behaviour) |
| `magicbrush.csv` | `rule_metrics.py --part magicbrush` | MagicBrush L1, CLIP-I (ViT-B/32), DINO (vits16) against GT |
| `editmag.csv` | `pipeline/make_copy_reference.py` | Edit size: mean \|output − input\| (0–1 RGB) for base / T0 / T2 / T3 |
| `editmag_ref.csv` | `diagnosis/make_degenerate_baselines.py` | Same for the random and gt references |

## Diagnosis: `diagnosis/`

| File | Written by | Content |
|---|---|---|
| `cause1_baselines.csv` | `diagnosis/cause1_rescore.py` | Normalised primary metric per benchmark (and PhyEditBench type, RISE category) for models, copy, random, gt; copy/random gaps |
| `cause1_penalty.csv` | `cause1_rescore.py` | Group deltas, Holm p and model ranking with no penalty and with the no-edit penalty at τ = 0.02 / 0.03 / 0.05 |
| `cause1_penalty_rates.csv` | `cause1_rescore.py` | Share of outputs flagged as no-edit per τ, benchmark and model |
| `cause2_means.csv` | `diagnosis/cause2_analyze.py` | Normalised mean and edit size on the 150-item subset for the 4 models before and after the probe fine-tune |
| `cause2_results.csv` | `cause2_analyze.py` | Gap to base before / after the fine-tune, change, closure, with CIs |
| `cause2_ft_effect.csv` | `cause2_analyze.py` | What the fine-tune did to each model on its own, and T+FT vs the original base |
| `cause2_ftdata_editmag.csv` | `diagnosis/ftdata_editmag.py` | Edit size of the probe's own training pairs (MagicBrush-train vs PICA-100K) |
| `cause2_parse_stats.csv` | `cause2_analyze.py` | Judge parse statistics for the `*_editsft` runs |
| `cause3_target_change.csv` | `diagnosis/pool_target_change.py` | How much v2 training targets differ from the VAE-condition frame, per source and setting |
| `cause3_auto.csv` | `diagnosis/failure_modes.py` | Per task and model: edit size, colour shift, sharpness ratio, no-edit / drift / blur flags |
| `cause3_vlm.jsonl` | `failure_modes.py` | VLM failure-mode tags (primary + secondary + reason) for all 4 models on the 67-item loss set |
| `cause3_modes.csv` | `failure_modes.py` | Failure-mode rates per model on the loss set (VLM + automatic) and on all items (automatic) |
| `cause3_code_review.csv` | hand-curated | Training-code findings with file:line locations and severity (high / medium / low / cleared) |
| `weight_diff.csv` | `diagnosis/weight_diff.py` | Tensor-level comparison vs the original base per component (VLM, decoder, VAE): missing keys, dtype changes, relative change |
| `control_null_pixels.csv` | `diagnosis/control_check.py` | Null fine-tune vs original base: pixel-identical images on the 150-item subset |
| `control_base_ft.csv` | `control_check.py` | Base before / after the real probe fine-tune: per-benchmark scores, consistency, reasoning, no-edit rate, edit size |
| `diag_examples.json` | `diagnosis/select_examples.py` | Image paths and scores for the Diagnosis section's examples |
| `diagnosis.json` | `diagnosis/build_diagnosis.py` | Ranking, summary and per-cause text for the page (placeholders resolved client-side) |
