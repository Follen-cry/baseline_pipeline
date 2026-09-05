# shape_color_then_move — rule-based scorer rewrite

`scorers/shape_color_then_move.py`, replacing `ShapeColorThenMoveEvaluator`
(VBVR-EvalKit `vbvr_bench/evaluators/Out_of_Domain_50_part4.py:704`). The old scorer
is untouched.

## The scene

A fixed 2×3 grid of glyphs on white. The top row is a worked example A → B → C; the
bottom row is D → ? → ?, and the model must produce E and F by applying the same
transformation.

The transformation is exactly what the task name says — **colour, then move**:

    A --recolour--> B --translate--> C

Measured across all 100 GT frames:

- GT always holds exactly 6 objects; the input frame always exactly 4.
- Column centres are fixed at x = 95.6 / 239.6 / 383.6 (sd 4.2) in the 512-px working
  frame; rows split cleanly at y = 256 (top ≤ 205, bottom ≥ 252).
- The B→C displacement and the E→F displacement are **identical to within 1.0 px** in
  every sample. dx is always +144 (the column pitch); dy varies per sample over
  0, ±12, ±20, ±30, ±40, ±50.

Worth stating because it is easy to misread from a thumbnail: **the third column is
displaced, not enlarged.** B and C have identical area and bounding-box size (e.g.
79×79 for both), differing only in y. My own first reading of these frames as "the
shape grows" was wrong, and the measurement corrected it.

## What was wrong with the old logic

1. **`_evaluate_second_row` — 35% of the weight — is
   `if len(final_bottom) >= 3: return 1.0`.** It counts connected components in the
   bottom half of the frame and nothing else. Three blobs of any colour, any shape,
   anywhere in the bottom half scored a perfect 1.0.
2. **`_evaluate_first_row_preservation` — 40% of the weight — compares *sorted hue
   lists*.** A top row with the right colours in the wrong cells scores 1.0, and so
   does one whose glyphs have been replaced with entirely different shapes.
3. **`_evaluate_color_accuracy`** reads B's hue from whatever shape happens to lie in
   the middle third of the input's top half, then compares raw HSV hue with a ±20
   window — unstable for the dark, low-saturation colours in this palette.
4. **`_detect_shapes_with_info` records only centroid, hue and area** — no shape
   identity at all, so a glyph substitution is invisible.

Together, 75% of the weight could be earned without the candidate's bottom row being
in the right places, the right shapes, or in the right cells.

## What the new scorer measures

Weights unchanged: `first_row_preservation` 0.40, `second_row_completion` 0.35,
`color_accuracy` 0.20, `shape_count` 0.05.

**Layer 1 — global gates.** Foreground ratio vs GT, plus a CIELAB background-consistency
gate (ceiling 0.15).

**Layer 2 — detection.** Objects are found **relative to the detected background**, not
by saturation. See the refine log: a saturation mask collapses entirely when a candidate
repaints the scene.

**Layer 3 — per-cell match against GT.** GT's six objects are assigned to named cells
A–F by row then column order, and each is matched to the nearest candidate object within
`POS_BAD_PX`. Each cell is then scored as a **product** of three independent conditions,
so one hard failure cannot be averaged away:

    cell = position_band × glyph_band × (colour within ΔE)

- `first_row_preservation` — mean cell score over A, B, C.
- `second_row_completion` — mean cell score over D, E, F.
- `color_accuracy` — E and F carry B's colour, which is what GT's E and F already
  encode, so it is scored directly against GT's cell colours.
- `shape_count` — candidate object count vs the expected 6.

Glyph identity is a **centred mask IoU** (`cvlib.centred_iou`), not a shape class: these
frames render crosses, T-glyphs, arrows, stars, L-shapes and bars, which a coarse
three-way classifier collapses together.

## Thresholds and where they come from

| Threshold | Value | Derivation |
|---|---|---|
| `POS_GOOD_PX` | 10 | Column centres have sd 4.2 px, so 10 px is inside rendering jitter. |
| `POS_BAD_PX` | 45 | Half the 144-px column pitch minus a glyph half-width — beyond it, the object is nearer a different cell. |
| `IOU_GOOD` | 0.85 | **p5** of candidate glyphs that are positionally and chromatically correct (n=1254; p25 0.954, p50 0.980). The identical rendered glyph in input vs GT scores exactly 1.000 (n=300, min 1.000). |
| `IOU_BAD` | 0.60 | Two genuinely different glyphs in the same GT frame score p50 0.448, p95 0.750 (n=200). The ramp 0.85→0.60 spans where the two populations actually separate. |
| `MIN_AREA` | 200 px² | Real objects have p5 area 1497; the connecting arrow glyphs measure 24–26 px². |
| `FG_TOL` | 25 | L2 BGR distance from background; flat-region jitter measures < 8. |
| `SAME_COLOR_DE` | 25 | See `cvlib.dE` — L-weighted, validated in `multi_object_placement_notes.md`. |
| `analogy_gate` | ramps below `first_row = 0.5` | Preserves the original evaluator's cliff at 0.5 without charging ordinary top-row loss twice. |

## Test-and-refine log

Calibration batch: slots 0,10,…,90 cycled across base/s0/s1/s2/s3.

| Pass | Mismatch found by looking at the images | Diagnosis | Fix |
|---|---|---|---|
| 1 | 00040/s3: bottom row is visually **correct**, but `second_row_completion` = 0.73 | My `analogy_gate` was continuous (`0.25 + 0.75·first_row`), so a top row that merely dropped glyph C — an error `first_row_preservation` already scores at 0.67 — docked the bottom row as well | Gate holds at 1.0 down to `first_row = 0.5` (the original evaluator's cliff) and ramps to 0 only below that. 00040 → 0.98 |
| 2 | 00009/base, 00092/base, 00089/s0 all scored 0.00 with candidate object counts of **18, 1 and 12** | All three repaint the background (black, orange, black). `chroma_mask` flags the background itself as foreground on an orange field, and returns pure noise on a black one. The global gate produced the right final number but the per-cell detail was garbage, so the scorer was not measuring what it claimed | Detection switched to **background-relative** (`foreground_mask`). Both masks give exactly 6 objects on all 100 GT frames; on the pathological frames the new one gives a sane 5 |
| 3 | Unit test: a circle inscribed in a square scored 0.84 as a glyph *match* | `IOU_BAD = 0.45` was asserted in a comment, never measured — and an inscribed circle overlaps its square by π/4 = 0.785 | Measured both populations on the split (above) and set `IOU_BAD = 0.60`; corrected the docstring's unvalidated claim. Calibration and held-out were re-run and both held |

Calibration agreement: **10/10**. Held-out agreement (slots 5,15,…,95): **10/10**.

Unit tests: `tests/shape_color_then_move_test.py`, 16 synthetic cases including blank,
input echoed back unchanged (no completion credit), perfect match, one top cell missing
(bottom row *not* docked twice), destroyed top row invalidating the analogy, one bottom
cell missing, wrong colours in E/F, the sequence-shifted-by-one failure seen in
00000/base, the move step not applied to F, jitter tolerance, glyph substitution
(asserted per cell), size change, repainted background, and shape count. All pass.

Two test expectations were corrected during this work rather than loosening the scorer:
both asserted on the D/E/F **mean**, which by design dilutes a single bad cell to ~0.14
— that is the behaviour verified against the real samples (00019: E miscoloured → 0.64),
so the assertions were retargeted at the per-cell score.

## Results

| variant | old rule | **new rule** | judge |
|---|---|---|---|
| base | 0.675 | **0.694** | 0.799 |
| s0 | 0.806 | **0.910** | 0.947 |
| s1 | 0.785 | **0.883** | 0.929 |
| s2 | 0.795 | **0.902** | 0.936 |
| s3 | 0.814 | **0.898** | 0.950 |

The new scorer sits between the old rule and the judge and separates base from the
pretrained variants much more sharply (base is 0.19–0.22 below s0–s3, against 0.11–0.14
for the old rule).

Where it disagrees with the judge, inspection favours the scorer on both sides:

- **The judge is blind to global scene corruption.** 00009/base and 00092/base render
  the whole answer on a black and an orange field respectively; the judge scores both
  **1.00**. The rewrite floors them.
- **The judge is all-or-nothing.** 00019 (top row perfect, two of three bottom cells
  correct, one miscoloured) is scored **0.00** by the judge; the rewrite gives 0.73–0.77,
  which is what the image shows.

## Known failure modes

- **`shape_count` (5% weight) is nearly always 1.0** and carries almost no information;
  it is retained only because the rubric names it.
- **The analogy is never checked as an analogy.** The scorer asks whether E and F match
  *GT*, not whether the candidate applied the same recolour-then-move that A→B→C shows.
  Those coincide here because GT is correct, but a scorer that verified the
  transformation itself would be a stronger test of reasoning.
- **Cell assignment is by rank order within a row**, so if a candidate draws two glyphs
  in one column and none in another, the columns shift and several cells can be scored
  against the wrong GT object at once.
- **`color_accuracy` is binary per cell** (2 cells → only 0, 0.5, 1.0 possible), so it
  is coarse on a single sample. A near-miss hue (00075/s1's pale seafoam for dark green)
  falls off a cliff rather than reading as partial.
- Glyphs that touch or overlap merge into one component and are scored as a single
  mis-shaped object.
