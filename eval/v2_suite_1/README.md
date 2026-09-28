# v2_suite_1: does video SSL improve (physics-aware) image editing?

A compact, evidence-driven evaluation of the v2 temporal-SSL checkpoints of InternVL-U, plus a
diagnosis of the result. Everything this suite produced lives in this directory: design, code,
judge outputs, result tables, reports and the builder for the shareable page.

| Model | Checkpoint | Training |
|---|---|---|
| base | HF `InternVL-U/InternVL-U` snapshot `f012d76…` | none (starting point) |
| T0 | `/scratch/network/ssd/junlin/models/internvlu-v2-t0-gen-merged` | ordered F0–F2 + Δt → F3 |
| T2 | `…/internvlu-v2-t2-gen-merged` | 3 of 4 frames + missing index → missing frame |
| T3 | `…/internvlu-v2-t3-gen-merged` | shuffled F0–F2 → ORDER text, then F3 |

## Findings

**Evaluation** ([`results/REPORT.md`](results/REPORT.md)). On 1,210 tasks (target / secondary /
control evidence groups), all three trained models score **below base** on the target group
(physics, state change, time); the pre-registered verdict is *refutes*. The trained models edit
about 35% less than base. PhyEditBench's apparent gain is an artifact: with this judge, returning
the input unchanged scores above every model.

**Diagnosis** ([`results/DIAGNOSIS.md`](results/DIAGNOSIS.md)). Ranked causes:
1. **Training-induced (work on first).** The trained models mostly fail by not editing. The v2
   supervision conditions the decoder on the frame nearest the target, and about half of the targets
   barely differ from it. A control shows the same copy-like failure appears on base within 160 steps
   of any fine-tune whose targets sit close to the VAE condition.
2. **Judge.** Real flaw on PhyEditBench (rewards copies and even random images), but penalising
   no-edit outputs keeps base on top, so it does not explain the result. Fix it so progress is measurable.
3. **Format mismatch.** Not supported: an identical editing fine-tune does not lift T0/T2/T3.

Shareable page (private until shared): https://claude.ai/artifact/DDBbysKBGP8UKhxkApMuta, built by
[`results/build_artifact.py`](results/build_artifact.py).

**Scope notes.** All generation used `generation_mode="image"` (not `text_image`), which matches how
T0/T2 were trained but not T3 (T3 was trained to write an ORDER line before the image). Scores come
from a local Qwen3-VL-30B-A3B-FP8 judge running each benchmark's official judge code; they are not
comparable to published GPT-judged leaderboards.

## Layout

```
v2_suite_1/
├── README.md            this file
├── SUITE.md             suite design: hypothesis, evidence groups, selection, planned power, verdict rule
├── docs/
│   ├── RUNBOOK.md       how to reproduce every step (envs, nodes, judge servers, commands, gotchas)
│   └── RESULTS.md       every file under results/: what it holds and which script writes it
├── pipeline/            the evaluation itself
│   ├── build_manifest.py → manifest.jsonl (1,210 tasks), config.json (models + inference settings)
│   ├── gen.py, launch_gen.sh                      generation (resumable, sharded over nodes)
│   ├── judge.py, judge_loop.sh, judge_copy.sh     official judge modules, API client swapped
│   ├── rule_metrics.py, make_copy_reference.py    PSNR / L1 / CLIP-I / DINO, copy reference, edit size
│   ├── sanity.sh, sanity_uids.txt, agreement.py   5% re-judge (same judge + Qwen3-VL-8B)
│   └── analyze.py, report.py                      tables, paired stats, REPORT.md, grids
├── diagnosis/           probes for the three candidate causes + the control check
│   ├── make_degenerate_baselines.py, judge_degenerate.sh, cause1_rescore.py     Cause 1 (judge)
│   ├── edit_sft/, cause2_*.{py,sh}, ftdata_editmag.py, cause2_subset_uids.txt   Cause 2 (format)
│   ├── failure_modes.py, pool_target_change.py                                   Cause 3 (training)
│   ├── weight_diff.py, control_check.py                                          control check
│   └── select_examples.py, build_diagnosis.py                                    DIAGNOSIS.md + page data
├── results/             all outputs that are not images (see docs/RESULTS.md)
│   ├── REPORT.md, DIAGNOSIS.md
│   ├── *.csv            main result tables
│   ├── judge_raw/       every judge call (3 judges x benchmark x model), resumable JSONL
│   ├── rule/            rule-based metrics
│   ├── diagnosis/       diagnosis tables and page data
│   ├── grids/           example image grids used by REPORT.md
│   └── build_artifact.py, artifact_template.html, conclusions.json   the shareable page
└── outputs -> /scratch/network/ssd/junlin/ssl_eval/outputs   generated images (not in git)
```

Git-ignored but kept on disk: `outputs/` (symlink), `results/judge_logs/`, `results/pica_viz/`
(PICABench ROI crops), `results/eval_report.html` (the built 15 MB page).

## Quick start

Rebuild every table, report and the page from the stored judge outputs (CPU only, a few minutes;
`internvlu` env):

```bash
cd eval/v2_suite_1
PY=~/miniconda3/envs/internvlu/bin/python
(cd pipeline && $PY analyze.py && $PY agreement.py && $PY report.py)
(cd diagnosis && python3 pool_target_change.py && $PY cause1_rescore.py && $PY cause2_analyze.py \
   && $PY control_check.py && $PY select_examples.py && $PY build_diagnosis.py)
$PY results/build_artifact.py        # -> results/eval_report.html
```

To evaluate a new checkpoint on the same 1,210 tasks, add it to `pipeline/config.json`, generate with
`launch_gen.sh`, start judge servers and run `judge_loop.sh` (all resumable). See
[`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Data outside the repo

| What | Where |
|---|---|
| Official benchmark repos (judge code imported from here) | `/scratch/network/ssd/junlin/ssl_eval/src/{PICABench,PhyEditBench,RISEBench,ImgEdit}` |
| Staged inputs / references for the 1,210 tasks (all nodes can read) | `/scratch/network/ssd/junlin/ssl_eval/staged/` |
| PICABench parquet | `/scratch/network/ssd/junlin/ssl_eval/data/PICABench/` |
| Generated images (`outputs/<model>/…`) | `/scratch/network/ssd/junlin/ssl_eval/outputs/` |
| Judge weights (network copy) | `/scratch/network/ssd/junlin/models/judges/Qwen3-VL-30B-A3B-Instruct-FP8` |
| Cause-2 fine-tune data | `/scratch/network/ssd/junlin/ssl_eval/diag_sft/` |
| Cause-2 / control checkpoints (65 GB, deletable) | `/scratch/network/ssd/junlin/models/diag_editsft/` |
