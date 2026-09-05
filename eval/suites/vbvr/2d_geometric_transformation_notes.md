# 2d_geometric_transformation — rule-based scorer rewrite

`scorers/2d_geometric_transformation.py`, replacing `GeometricTransformationEvaluator`
(VBVR-EvalKit `vbvr_bench/evaluators/Out_of_Domain_50_part4.py:282`) and its image-only
override at `evaluators/image_evaluator.py:659`. The old scorers are untouched.

## The scene

A light grey field (BGR 240,240,240) holding one coloured polygon at its start pose, a
thin **dashed grey target silhouette** showing where it must end up, and a small pivot
marker (a white dot with a dark ring). The task is to rotate the polygon about the pivot
so it lands on the target silhouette.

Measured across the split:

- The GT final frame contains **exactly one** foreground component (30/30 sampled): the
  polygon at its target pose. The dashed target outline is gone; the pivot marker
  remains (it is only ~57 px).
- Required displacement start → target: p5 21 px, median 49 px, p95 74 px, **minimum
  17 px** — against a polygon whose equivalent side is ~41 px, so the move is always
  substantial.
- Required rotation spans the full circle (best-fit values from −168° to +178°), and the
  polygon is rigid: best-fit IoU at the correct angle is p50 0.924, p5 0.770.

## What was wrong with the old logic

**None of the sub-criteria looked at the ground-truth final frame at all** — every one
compared the generated frame against the *input*.

1. **`_detect_main_shape` selects contours by `hsv[:,:,1] > 50`.** This palette is
   deliberately muted — (135,160,132), (135,135,171) — and the saturation mask returns
   **zero objects** for many samples. Every downstream sub-criterion then hit its
   constant fallback, so the scorer was frequently not measuring the image at all.
2. **`_evaluate_rotation_angle`** compared a single `_get_contour_angle` of the input's
   target outline against the final shape. One contour angle (minAreaRect / principal
   axis) is ambiguous under symmetry and unstable for a near-square shape, and it
   returns a flat `0.5` whenever either detection fails.
3. **`_evaluate_position`** measured distance to the *input's* outline centre on a fixed
   ladder (`<30 → 1.0`, `<60 → 0.7`, `<100 → 0.4`, else **0.2**). The 0.2 floor means a
   shape left completely untouched still scores 0.2, and a missing centre returns 0.5.
4. **`_evaluate_shape_fidelity`** compared areas only, with the same `0.5` fallback.

The result is visible in the aggregate: the old scorer gave **base the *highest* score
of all five variants** (0.482 vs 0.448–0.480), and its per-sample correlation with the
VLM judge is **−0.136** — it was mildly *anti*-correlated with correctness.

`rotation_center` stays dropped (it needs ≥3 sampled frames to fit an arc; a single
generated frame draws no trail) and the remaining three weights stay renormalised,
exactly as the existing image-only override does.

## What the new scorer measures

Weights unchanged: `rotation_angle` 0.50, `position_alignment` 0.357, `shape_fidelity`
0.143 (0.35/0.25/0.10 renormalised over 0.70).

**Layer 1 — global gates.** Foreground ratio vs GT and a CIELAB background-consistency
gate.

**Layer 2 — detection and colour.** Objects are found **relative to the detected
background** (`foreground_mask`), not by saturation — this is what makes the muted
palette visible at all. The candidate polygon's colour is gated against GT's in
L-weighted CIELAB.

**Layer 3 — pose, against GT.**

- `position_alignment` — centroid error against GT, ramped 1→0 over
  `max(0.10·side, 3px)` → **`max(required, 0.30·side)`**, where `required` is the
  distance the task actually asked the shape to travel. A candidate that leaves the
  shape where it started is `required` px away and therefore scores **0 by
  construction** — no credit for the right element in the wrong place.
- `rotation_angle` — the rotation that best maps the candidate's silhouette onto GT's,
  found by **exhaustive search over 360°** maximising centred IoU (`cvlib.best_rotation`,
  4° coarse then 1° fine). This is used instead of a single contour angle because an
  exhaustive best fit is unambiguous under symmetry and reports a number that can be
  stated directly ("best alignment at 12°").
- `shape_fidelity` — rigid-body check against the **input** polygon (best-fit IoU and
  area ratio, 0.5 each), so a candidate that reaches the target by *deforming* the shape
  is caught.

Two additional gates, both multiplicative and both documented in the source:

- **Object identity.** A candidate that erased the polygon and left only the hollow
  target outline produces a component whose centroid sits exactly on GT's, and would
  otherwise collect near-full position and rotation credit for drawing no shape. The
  separating quantity is best-fit IoU, not area: outline-only scores **0.134** where a
  0.6×-scaled shape scores 0.377, a substituted circle 0.778 and the true shape 1.000 —
  their *areas* overlap (0.239 vs 0.374) but their silhouettes do not.
- **Residual target outline.** The task is complete only once the dashed silhouette is
  gone. Measured as *neutral grey ink* — foreground pixels that are unsaturated and
  darker than the background — expressed as the fraction of the outline ink that should
  have been erased and was not, so it reads 0 for a clean solve however much grey the
  polygon itself carries. Applied as a mild ×(1 − 0.25·retained), since it is not one of
  the three sub-criteria the rubric names.

## Thresholds and where they come from

| Threshold | Value | Derivation |
|---|---|---|
| `FG_TOL` | 20 | L2 BGR from the grey background; flat-region jitter measures < 8. |
| `MIN_SHAPE_AREA` | 300 px² | The polygon is 1400–3000 px²; outline dashes are < 100 px². |
| position `good` | max(0.10·side, 3px) | Rendering jitter on a ~41 px shape. |
| position `bad` | max(required, 0.30·side) | The *starting* error — zero progress must score zero. |
| `ROT_GOOD_DEG` | 8 | GT's own best-fit residual is ≤ 2°; below ~8° the poses of a 41-px shape are visually indistinguishable. |
| `ROT_BAD_DEG` | 45 | A quarter turn — unambiguously the wrong pose. Consistent with the observed error distribution: 25% of candidates are within 8°, 42% within 20°, and 38% at or beyond 45°. |
| `FID_GOOD_IOU` / `FID_BAD_IOU` | 0.80 / 0.55 | The same rigid polygon at its best-fit angle scores p50 0.924, p5 0.770. |
| identity gate | IoU 0.45 → 0.20 | Far below the p5 (0.770) of genuinely-identical objects, far above outline-only (0.134). |
| residual multiplier | ×(1 − 0.25·retained) | Judgement call — deliberately mild, as noted above. |

## Test-and-refine log

Calibration batch: slots 0,10,…,90 cycled across base/s0/s1/s2/s3.

| Pass | Mismatch found by looking at the images | Diagnosis | Fix |
|---|---|---|---|
| 1 | 00010/s0: the candidate leaves the shape **exactly where it started** (err 17 px == required 17 px) but scored `position_alignment` 0.34 | The ramp's far end was `max(0.60·side, 0.55·required)`, dominated by shape size whenever the required move was short | `bad` is now the starting error itself, so zero progress scores zero. 00010 → 0.01 |
| 2 | 00030/s2: the dashed target outline is plainly still drawn, but the residual check reported 0 | Counting connected components cannot see it — once the shape lands on the outline the two touch and **merge into one component**. Total foreground area cannot either: the outline is a thin stroke worth only ~15% of it (input/GT area ratio 1.09–1.21) | Residual measured as normalised excess **grey ink**. 00030/s2 → 0.53 retained; clean solves → 0.00 |
| 3 (unit tests) | A candidate containing **only the outline** scored 0.45 on both position and rotation | The hollow outline closes into a component centred on GT's shape, so it inherited full pose credit for an object that is not the shape | Added the best-fit-IoU **object-identity gate** on the pose criteria |
| 3 | A correctly placed shape rendered at 0.6× was capped by the global blankness ceiling as though the frame were near-blank | `cvlib.blankness`'s 0.45 boundary was arbitrary and conflated *smaller object* with *empty image* — a size error the fidelity criteria already measure | Boundaries lowered to 0.05 / 0.15 / 0.30. Re-verified: all four tasks' scans and all 68 unit tests unchanged |

Calibration agreement: **10/10**. Held-out agreement (slots 5,15,…,95): **10/10**.

Unit tests: `tests/2d_geometric_transformation_test.py`, 20 synthetic cases including
blank, outline-only, perfect match, shape left at start (no position credit), a short
required move (the 00010 regression), right-place-wrong-rotation, small rotation
tolerated, rotation monotonicity, a 180° flip, position monotonicity, scaled and
deformed shapes, leftover outline detection, wrong colour, repainted background, and an
explicit check that the muted palette is detected at all. All pass.

Two test expectations were corrected rather than loosening the scorer: a circle
substituted for the polygon overlaps it by 0.778 at best fit — genuinely similar — so it
is asserted as *partial*, and recorded below as a known limit.

## Results

| variant | old rule | **new rule** | judge |
|---|---|---|---|
| base | 0.482 | **0.363** | 0.451 |
| s0 | 0.464 | **0.511** | 0.805 |
| s1 | 0.480 | **0.510** | 0.798 |
| s2 | 0.448 | **0.597** | 0.838 |
| s3 | 0.453 | **0.496** | 0.848 |

Per-sample correlation with the judge: Pearson **−0.136 → +0.364**, Spearman
**−0.140 → +0.370**. Variant spread (max − min of the per-variant means, i.e. how much
the metric separates the models at all) goes from **0.034 → 0.234**; the judge's is 0.397.

The new scorer is still well below the judge in absolute level. That gap is honest and
is concentrated in `rotation_angle`, which carries half the weight: measured over 142
(sample, variant) pairs, only 25% of candidates land within 8° of GT's pose and **38%
are 45° or more out**, which the debug overlays confirm on inspection — the models
routinely place the shape in roughly the right spot at plainly the wrong orientation.
The judge scores many of those as correct. On the calibration and held-out batches my own
visual verdict agreed with the scorer, not the judge, on every such case.

## Known failure modes

- **`shape_fidelity` (14% weight) is the least reliable sub-criterion.** It is a
  silhouette-overlap measure, and silhouette overlap saturates for compact shapes: a
  circle substituted for this 5-gon scores 0.778 and therefore reads as only mildly
  wrong. It cannot distinguish "same shape, slightly ragged edges" from "different but
  similarly compact shape".
- **`rotation_angle` is unreliable for near-symmetric polygons.** The exhaustive best fit
  correctly returns the *smallest* error under symmetry, which is the right convention,
  but it means a shape with 4-fold symmetry can never score below the 90° band however
  it is oriented. None of the sampled shapes are strongly symmetric, so this is untested
  rather than known-broken.
- **The pivot is never used.** The rubric's original intent was "rotate *about the marked
  point*", and the scorer only checks the end pose. A candidate that reaches the correct
  final pose by an incorrect rotation centre is indistinguishable — but with one frame
  there is no pixel evidence of the path, which is exactly why `rotation_center` was
  dropped in the first place.
- **The residual-outline multiplier is a judgement call** (×0.25 at most) and its grey-ink
  measure will misread a candidate whose polygon is itself rendered in a desaturated
  grey.
- `_main_shape` takes the largest colour-matching component, so a candidate that splits
  the polygon into two pieces is scored on the larger fragment alone.
