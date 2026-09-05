# Rule-based scorer rewrite — summary

Four VBVR task scorers rebuilt from scratch as deterministic OpenCV/numpy pipelines.
No learned models, no VLM calls, no network; each runs standalone on a single
(input, candidate, ground_truth) triple.

```
scorers/cvlib.py                        shared CV primitives
scorers/rotation_puzzle.py              score() / debug() / overall()
scorers/multi_object_placement.py
scorers/shape_color_then_move.py
scorers/2d_geometric_transformation.py
scorers/samples.py  montage.py  harness.py  scan.py     evaluation harness
tests/<task>_test.py                    68 synthetic unit tests, all passing
<task>_notes.md                         per-task rationale, thresholds, failure modes
```

The old scorers are untouched — `evaluators/image_evaluator.py` and the vendored
`VBVR-EvalKit` evaluators still run, so before/after comparison is possible.

Run one triple:

```bash
python scorers/rotation_puzzle.py IN.png CAND.png GT.png --debug-dir /tmp/dbg
```

Reproduce the tables below: `python scorers/scan.py <task> <task>`.
Reproduce a review batch: `python scorers/harness.py <task> <task> --which calib|holdout`.

## Scores per model variant

Mean over 100 eval samples per variant. **old** = existing rule-based scorer,
**new** = this rewrite, **judge** = the VLM judge already in the repo.

### rotation_puzzle

| variant | old | **new** | judge |
|---|---|---|---|
| base | 0.526 | **0.899** | 0.904 |
| s0 | 0.552 | **0.942** | 0.933 |
| s1 | 0.547 | **0.893** | 0.918 |
| s2 | 0.550 | **0.921** | 0.927 |
| s3 | 0.549 | **0.934** | 0.936 |

### multi_object_placement

| variant | old | **new** | judge |
|---|---|---|---|
| base | 0.335 | **0.429** | 0.105 |
| s0 | 0.490 | **0.738** | 0.274 |
| s1 | 0.373 | **0.534** | 0.194 |
| s2 | 0.512 | **0.755** | 0.320 |
| s3 | 0.444 | **0.691** | 0.209 |

### shape_color_then_move

| variant | old | **new** | judge |
|---|---|---|---|
| base | 0.675 | **0.694** | 0.799 |
| s0 | 0.806 | **0.910** | 0.947 |
| s1 | 0.785 | **0.883** | 0.929 |
| s2 | 0.795 | **0.902** | 0.936 |
| s3 | 0.814 | **0.898** | 0.950 |

### 2d_geometric_transformation

| variant | old | **new** | judge |
|---|---|---|---|
| base | 0.482 | **0.363** | 0.451 |
| s0 | 0.464 | **0.511** | 0.805 |
| s1 | 0.480 | **0.510** | 0.798 |
| s2 | 0.448 | **0.597** | 0.838 |
| s3 | 0.453 | **0.496** | 0.848 |

## Agreement and discrimination

`held-out` is the fraction of a fresh, never-tuned-against batch of 10
(sample, variant) pairs — striding the whole split, cycling base/s0/s1/s2/s3 — on which
my own per-sub-criterion visual verdict matched the scorer's output. `spread` is
max − min of the per-variant means: how much the metric separates the five models at all.

| task | calibration | **held-out** | Pearson vs judge (old → new) | Spearman (old → new) | spread (old → new) |
|---|---|---|---|---|---|
| rotation_puzzle | 10/10 | **10/10** | +0.266 → **+0.691** | +0.172 → **+0.609** | 0.026 → 0.049 |
| multi_object_placement | 10/10 | **10/10** | +0.225 → **+0.462** | +0.266 → **+0.513** | 0.177 → 0.325 |
| shape_color_then_move | 10/10 | **10/10** | +0.272 → **+0.569** | +0.323 → **+0.579** | 0.140 → 0.215 |
| 2d_geometric_transformation | 10/10 | **10/10** | **−0.136** → **+0.364** | −0.140 → +0.370 | 0.034 → 0.234 |

Correlation with the judge improved on every task, and the old
2d_geometric_transformation scorer was mildly *anti*-correlated with correctness — it
also ranked the untrained base model **highest** of the five variants.

## What was actually wrong with the old scorers

Every one of the four turned out to be dominated by a detection failure that silently
routed into a constant fallback, rather than by a scoring-design problem:

- **rotation_puzzle** — all four sub-criteria read the pipes through
  `_detect_blue_pipes`, a hardcoded HSV window `[100,100,100]..[130,255,255]`. The pipes
  are rendered in a per-sample colour (orange, yellow, magenta, cyan, green, red all
  appear). For every non-blue sample the mask was empty and the code returned its `0.5`
  constants. **That is the origin of the near-identical ~0.55 scores** — and of the flat
  0.526–0.552 across all five variants, i.e. the old scorer was very nearly a constant
  function of its input.
- **multi_object_placement** — `_detect_colored_objects` enumerated four hardcoded HSV
  boxes (red/blue/green/yellow); magenta, cyan and orange objects were invisible and
  every sub-criterion fell through to its `0.2`/`0.3` "detection failed" constant. It
  also used a fixed 30 px match radius for objects 40–57 px wide, allowed one candidate
  to satisfy several GT objects, and scored `star_invariance` against the *input* stars —
  so a correct solve, which covers every star, scored the failure constant.
- **shape_color_then_move** — `_evaluate_second_row`, 35% of the weight, is literally
  `if len(final_bottom) >= 3: return 1.0`. Three blobs of any colour or shape anywhere in
  the bottom half scored a perfect 1.0. `_evaluate_first_row_preservation`, another 40%,
  compared *sorted hue lists*, so right colours in wrong cells also scored 1.0.
- **2d_geometric_transformation** — no sub-criterion looked at the ground-truth final
  frame at all, and `_detect_main_shape`'s `hsv[:,:,1] > 50` returns **zero objects** for
  this deliberately muted palette. `_evaluate_position` also had a 0.2 floor, so a shape
  left completely untouched still scored 0.2.

The rewrites share one structural change: **match each GT object to a candidate
counterpart, with colour as a hard gate, and score position/pose/shape per object** —
so an element present somewhere in the image earns no positional credit.

## Where the new scorers disagree with the judge

Two measurable judge defects, both documented in the per-task notes:

- **multi_object_placement**: the judge marks `star_invariance` **wrong on 91% of
  samples** (455/500) — but GT contains no visible star markers in any of the 100 GT
  frames, because a correct solve covers them. Its own reasoning says so: *"no star
  markers, so objects cannot be on matching stars, violating … star_invariance."* It also
  returns **zero `partial` verdicts** across all 500 × 4 sub-criteria, so two-of-three
  correct placements score 0. This is why the judge's mean is far below the new scorer's;
  the new scorer nevertheless reproduces the judge's variant ordering exactly
  (base < s1 < s3 < s0 < s2).
- **shape_color_then_move**: the judge is blind to global scene corruption — 00009/base
  and 00092/base render the whole answer on a black and an orange field and are both
  scored **1.00**.

Conversely the judge is right, and the new scorer conservative, on
**2d_geometric_transformation**, where the remaining gap is real (see below).

## Sub-criteria I still consider unreliable

| task · sub-criterion | weight | why |
|---|---|---|
| **2d_geo · `shape_fidelity`** | 0.14 | Silhouette overlap saturates for compact shapes: a circle substituted for the 5-gon scores 0.778 and reads as only mildly wrong. Cannot separate "same shape, ragged edges" from "different but similarly compact shape". |
| **2d_geo · `rotation_angle`** | 0.50 | Sound as measured, but it carries half the weight and is where the whole judge gap lives: 38% of candidates are ≥45° out. Also, an exhaustive best fit can never score a 4-fold-symmetric shape below the 90° band — none of the sampled shapes are strongly symmetric, so that path is untested rather than known-broken. |
| **mop · `fidelity`** | 0.19 | Shape class is a hard 0/1 on a three-way classifier, so a marginal circle/quad call swings half an object's fidelity. Measured against the input rather than GT (right in principle), so a candidate that redraws everything slightly larger loses fidelity uniformly. |
| **sctm · `color_accuracy`** | 0.20 | Binary over exactly 2 cells, so only 0, 0.5, 1.0 are possible — coarse on a single sample, and a near-miss hue (00075/s1's pale seafoam for dark green) falls off a cliff instead of reading as partial. |
| **sctm · `shape_count`** | 0.05 | Almost always 1.0; carries essentially no information. Retained only because the rubric names it. |
| **rotation_puzzle · `alignment_precision`** | 0.10 | Only 4 junctions with a 3→14 px ramp, so it is effectively quantised to {0, 0.25, 0.5, 0.75, 1}. Fine in aggregate, not readable on one sample. |
| **mop · surplus-object penalty** | — | The one threshold in the whole set with no measurement behind it (0.25 per surplus object, capped at 0.5). |

Two further limits worth stating plainly:

- **rotation_puzzle's GT is the same closed ring in all 100 samples.** The scorer derives
  the target arm configuration from GT rather than hardcoding it, but that generality has
  never been exercised.
- **shape_color_then_move never checks the analogy as an analogy.** It asks whether E and
  F match GT, not whether the candidate applied the same recolour-then-move that A→B→C
  demonstrates. Those coincide because GT is correct, but a scorer that verified the
  transformation itself would be a stronger test of reasoning.

## Method note

Each task went through at least three refine passes driven by looking at the input,
candidate, GT and debug visualisations for every instance in the batch — not by reading
numbers. Every fix in the per-task logs was traced to a specific detection or logic
defect; where a threshold moved, the note records the measurement it was derived from and
the check that it was not fitted to the batch. Three changes were **reverted** after
measurement contradicted them, and four unit-test expectations were corrected rather than
loosening a scorer to pass them; both are recorded in the notes and in the source.

One case is deliberately left disagreeing with the judge: on rotation_puzzle 00081/s0 and
00032/s3 the judge says 1.00 and the scorer says 0.82, because the candidate's pipe arm
stops **36 px and 52 px** short of a 110-px tile edge where GT stops within 2–4 px. The
measurement is unambiguous, so the threshold was not moved to match the judge.
