# Video-SSL → physics-aware editing suite

**Hypothesis.** v2 temporal self-supervised training on video (T0 forecasting, T2 masked-frame
interpolation/forecasting, T3 order-recovery + forecasting; see `docs/v2.md`) improves
instruction-based image editing, most of all edits that need physical dynamics, state change,
causality or time evolution.

**Models** (identical inference settings, `pipeline/config.json`):

| model | checkpoint | what it is |
|---|---|---|
| base | HF `InternVL-U/InternVL-U` snapshot `f012d76…` | untrained starting point |
| T0 | `/scratch/network/ssd/junlin/models/internvlu-v2-t0-gen-merged` | VC2I-F: ordered F0-F2 + caption + Δt → generate F3 |
| T2 | `…/internvlu-v2-t2-gen-merged` | VC2I-M: 3 of 4 frames + Δt + missing index → generate the missing frame |
| T3 | `…/internvlu-v2-t3-gen-merged` | VC2I-OF: shuffled F0-F2 → predict ORDER text, then generate F3 |

All three: 3,750 steps on the same 60K-window pool (PhysicTran38K, PhysInOne, VBVR, BAAI
physics, MiT physics, SSv2), area-512 generation targets.

Code: `pipeline/` (`build_manifest.py` → `manifest.jsonl`, `gen.py`, `judge.py`,
`rule_metrics.py`, `analyze.py`); outputs `outputs/<model>/…` (symlink to network storage);
results `results/`. Layout and how to run: [`README.md`](README.md), [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Evidence groups and expected pattern

| group | what | supports the hypothesis | refutes it |
|---|---|---|---|
| **A. Target** | PhyEditBench (Types A–E + Anti-Physics), PICABench (8 laws, superficial prompts), RISEBench Temporal + Causal | trained − base > 0 with CI excluding 0; largest on PhyEditBench *Physical Plausibility*, step-wise Types A–C (closest to next-frame prediction), PICABench Mechanics/State, RISE Temporal | Δ ≤ 0, or Δ in A no larger than in C (a generic quality shift, not physics) |
| **B. Secondary** | RISEBench Spatial | small positive or neutral | clear loss |
| **C. Controls** | RISEBench Logical, ImgEdit Basic (9 types), ImgEdit UGE, MagicBrush turn-1 | Δ ≈ 0 (within ±MDE): the training did not break general editing | clear loss (forgetting) → any A gain must be read net of it |

**Verdict rule (fixed before any results were read; implemented in `report.py::verdicts`)**, on
each group's pooled normalised Δ:
- A/B *supports*: at least one trained model with Holm p < 0.05 and CI > 0, and none significantly below base.
- A/B *refutes*: a trained model significantly below base (and none above), or *refutes (no effect)*
  when every model's CI upper bound is below the planned MDE.
- A/B *inconclusive*: anything else.
- C *no harm*: every CI lower bound above −planned MDE. *Harm*: any model significantly below base.

Anti-Physics is read separately. Counterfactual instructions ("the ball is harder than the
stone") test whether the model follows the text or falls back on learned physics. If video SSL
strengthens real-world priors, we expect Physical-Plausibility-to-the-counterfactual and
Instruction-Following on Anti-Physics to *drop* while normal PhyEditBench rises.

Trend expectation T0 → T2 → T3: if richer temporal objectives matter, T2 (interpolation +
forecasting) and T3 (explicit temporal order) ≥ T0 in group A, with no extra cost in C.

Prior evidence to keep in mind: on the in-domain evalmini none of T0/T2/T3 beat the copy-input
baseline in PSNR (`project_v2_t0t2t3_training`), so a null result in A is plausible a priori.

## Selection

Stratified random sampling. Each source has its own RNG stream (`random.Random("20260927:<src>")`),
so changing one allocation never reshuffles another. **No selection on model outputs**: picking
items where base is weak or strong would bias paired comparisons (regression to the mean).
"Most informative" is instead achieved through *allocation*: most items go to target
categories, and every stratum gets enough items for a per-category read.

**Per-stratum minimum: 7 datapoints** (the smallest RISE subtask, Societal Transformation,
only has 7; every other stratum has ≥ 12, PhyEditBench subclasses 40 tasks each).

| group | benchmark | selection | tasks |
|---|---|---|---:|
| A | PhyEditBench | 8 trajectories × 12 subclasses (Stability_&_Balance has exactly 8) × 5 official types A–E (`gpt_eval.build_type_fields`, verbatim) | 480 |
| A | PhyEditBench Anti-Physics | all 35 (7 × 5 rule types; official `checklists.jsonl`) | 35 |
| A | PICABench (superficial prompt) | 45 × {Causality, Deformation, Global, Local} + 20 × {Light_Propagation, Light_Source_Effects, Reflection, Refraction}; Mechanics + State weighted 2.25× | 260 |
| A | RISEBench Temporal + Causal | all (85 + 90) | 175 |
| B | RISEBench Spatial | 12 × 5 subtasks | 60 |
| C | RISEBench Logical | 12 × 3 subtasks | 36 |
| C | ImgEdit Basic | 12 × 9 edit types (add, remove, replace, adjust, style, background, action, compose, extract) | 108 |
| C | ImgEdit UGE | 16 of 47 | 16 |
| C | MagicBrush test, turn 1 | 40 sessions | 40 |
| | **total** | | **1,210** (× 4 models = 4,840 generations) |

Per-category counts (A): PhyEditBench 40 per subclass (8 traj × 5 types), 96 per type;
Anti-Physics 7 per rule type; PICABench 45/45/45/45/20/20/20/20; RISE Temporal: Material 46,
Life 19, Environmental 13, Societal 7; Causal: Structural Deformation 36, State Transition 25,
Chem/Bio 16, Physics Manifestation 13. B: 12 per Spatial subtask. C: 12 per Logical subtask,
12 per ImgEdit type.

## Power (planned)

Paired design: every model is scored on the same items, and we report Δ = trained − base per
item. For a two-sided paired test at α = 0.05 with 80% power, the minimum detectable mean
difference is **MDE ≈ 2.80 · SD(Δ) / √n_eff**. SD(Δ) was taken from our own earlier ImgEdit
runs (base vs S0/S2/S4, same local judge): SD(Δ) ≈ 1.0 on a 1–5 scale, ≈ 0.9 × the per-item SD.
Generation noise dominates, so pairing buys little. We therefore assume SD(Δ) ≈ 0.25 of the
scale range for Likert metrics, and 0.30 for PICABench per-sample accuracy. PhyEditBench's 5
types share a trajectory, so n_eff is taken as ≈ 250 of 480. On the common 0–1 normalised
scale, Δ counts as a fraction of the metric range.

| group / benchmark | n | n_eff | assumed SD(Δ) | MDE (normalised) | MDE (native) |
|---|---:|---:|---:|---:|---:|
| **A total** | 950 | ~720 | 0.26 | **0.027** | — |
| A · PhyEditBench overall | 480 | ~250 | 0.25 | 0.044 | 0.40 pt on 1–10 |
| A · PICABench Acc | 260 | 260 | 0.30 | 0.052 | 5.2 pp |
| A · RISE T+C score | 175 | 175 | 0.25 | 0.053 | 0.21 on 1–5 |
| A · Anti-Physics | 35 | 35 | 0.25 | 0.12 | 1.1 pt on 1–10 |
| **B** RISE Spatial | 60 | 60 | 0.25 | **0.090** | 0.36 on 1–5 |
| **C total** | 200 | 200 | 0.25 | **0.050** | — |
| C · ImgEdit Basic | 108 | 108 | 0.25 | 0.067 | 0.27 on 1–5 |

A has the most power by design (≈ 4.7× C's n, ≈ 16× B's). Per-category reads (40 PhyEditBench
tasks, 45 PICABench items) can only detect ≈ 0.11–0.13 normalised, so they are descriptive, and
only group- and benchmark-level tests are confirmatory. `analyze.py` also reports the
*realised* MDE from the observed SD(Δ).

## Benchmarks considered and skipped

- **KRIS-Bench**: knowledge-based editing (factual/conceptual/procedural). Its physics,
  chemistry and temporal subsets overlap RISEBench Temporal/Causal and PICABench, and it adds no
  distinct axis.
- **GEdit-Bench**: general real-user edits. It duplicates ImgEdit Basic's role as a control.
- **PBench-Edit (ChronoEdit)**: video-frame-derived physical-consistency edits (human, robot,
  driving). It overlaps PhyEditBench, which is stronger here because it ships GT target states
  and step-wise/global types.
- **AURORA-Bench** (staged in `suites/aurorabench`): action edits from video. Its SSv2 slice
  shares a domain with the T-training pool (SSv2 is 11.8% of it), which would confound transfer
  with in-domain overlap. It is a candidate *positive control* in a follow-up.
- **MagicBrush**: included as a small control because it has GT targets, which enables
  rule-based L1/CLIP-I/DINO scores independent of the VLM judge.

## Judging (summary; details in `results/REPORT.md`)

Judge: Qwen3-VL-30B-A3B-Instruct-FP8 on vLLM, temperature 0. Each benchmark's official judge
module is imported from the upstream repo and only the API client is swapped. Rule-based:
PICABench Con (masked PSNR, official `PicaEval_consistency.py`, 512), MagicBrush L1/CLIP-I/DINO.
**Scores are not comparable to published GPT-judged leaderboards**; compare only across our 4
models.

**Added after the first results (post-hoc, diagnostic only):** an *input-copy reference*
(`make_copy_reference.py`: the input returned unchanged at the generation size, judged like a model)
and per-item *edit magnitude* (mean |output − input|). Together they showed that the PhyEditBench judge
rewards inaction: copy scores above every model. See `results/REPORT.md` → Conclusions.
