# multi_object_placement — rule-based scorer rewrite

`scorers/multi_object_placement.py`, replacing `MultiObjectPlacementEvaluator`
(VBVR-EvalKit `vbvr_bench/evaluators/In_Domain_50_part1.py:191`) and its image-only
override at `evaluators/image_evaluator.py:502`. The old scorers are untouched.

## The scene

A white field holding N coloured shapes (N ∈ {2,3,4}; circle / square / triangle,
1591–3245 px² in the 512-px working frame) and N small star markers (a constant
137–138 px²) whose colours match the shapes one-for-one. The task is to move each
shape onto the star of its own colour.

Two facts measured across all 100 GT frames, which the rewrite is built on:

- **GT contains zero visible star markers.** Every star ends up completely covered
  by its shape.
- **Every GT shape centroid lies within 10 px (median 0.6 px) of the matching input
  star.** So GT is a direct, per-object positional reference for "on the right star".

## What was wrong with the old logic

1. **`_detect_colored_objects` enumerated four hardcoded HSV boxes** — red, blue,
   green, yellow. The split also uses magenta, cyan and orange, which fall in no box.
   Objects in those colours were invisible, and every sub-criterion then fell through
   to its `0.2` / `0.3` *"detection failed"* constant rather than reporting a miss.
2. **No connected-component separation and no shape typing.** It ran `findContours` on
   a per-colour mask and kept only centroid and area, so a circle rendered as a square
   scored as a perfect placement.
3. **`color_matching` used a fixed 30 px radius** for objects whose width ranges 40–57 px,
   and let many candidate objects match the same GT object (the `break` exits only the
   inner loop), so one well-placed object could satisfy several GT objects at once.
4. **`star_invariance` compared candidate stars to *input* stars and returned 0.3 when
   either set was empty.** Since a correct solve covers every star, the correct answer
   scored the "detection failed" constant.
5. `path` (20% of the original weight) measured motion variance across video frames and
   is undefined for a single generated frame. It stays dropped, and the remaining four
   weights stay renormalised, exactly as the existing image-only override does.

## What the new scorer measures

Weights unchanged: `color_matching` 0.375, `alignment` 0.3125, `fidelity` 0.1875,
`star_invariance` 0.125 (0.30/0.25/0.15/0.10 renormalised over 0.80).

**Layer 1 — global gates.** Foreground-pixel ratio vs GT (`cvlib.blankness`) and a
CIELAB background-consistency check, both able to floor the score before any
per-object work runs.

**Layer 2 — colour.** Shapes and markers are separated by **area** (threshold 600,
sitting in an empty gap an order of magnitude wide: shapes ≥1591, markers ≈137), and
the distinct-colour multisets of candidate and GT are compared. This is reported in
the debug output and drives the surplus-object penalty.

**Layer 3 — per-object structural matching.** `cvlib.match` assigns each GT object to
at most one candidate object, with colour as a **hard gate** — an object of the wrong
colour is never a match, so a candidate cannot earn positional credit by putting *some*
object in the right place. Then:

- `color_matching` — fraction of GT objects with a colour-matched candidate within
  `0.60 × side`, minus a surplus penalty.
- `alignment` — centroid distance ramped 1→0 over `0.20 × side` → `0.80 × side`.
- `fidelity` — shape class *and* area, compared against the **input** object of that
  colour (0.5 each).
- `star_invariance` — GT shows no markers, so a candidate showing none scores 1.0. A
  marker the candidate *does* draw is acceptable only where the input had one; a marker
  anywhere else is a hallucinated element.

All placement tolerances are fractions of the GT object's own equivalent side length
`√area` (40–57 px here), not fixed pixel radii, so the same rule applies to a small
triangle and a large square.

## Thresholds and where they come from

| Threshold | Value | Derivation |
|---|---|---|
| shape / marker split | 600 px² | Measured: shapes 1591–3245 px², markers 137–138 px². 600 is an order of magnitude clear of both. |
| `ON_TARGET_FRAC` | 0.60 × side | The shape must substantially overlap the star it is meant to cover. GT sits at 0.01 × side. |
| `ALIGN_GOOD/BAD_FRAC` | 0.20 / 0.80 × side | 0.20 ≈ 11 px, within rendering jitter; at 0.80 the object no longer overlaps the target at all. |
| `SAME_COLOR_DE` | 25 | See the L-weighting entry below. |
| `L_WEIGHT` (cvlib) | 0.5 | See below. |
| `STAR_TOL_PX` | 14 | About one marker width; unmoved markers measure ≤ 6 px off. |
| surplus penalty | 0.25/object, cap 0.5 | Judgement call, not derived. Only fires on genuine surplus. |

## Test-and-refine log

Calibration batch: slots 0,10,…,90 cycled across base/s0/s1/s2/s3.

| Pass | Mismatch found by looking at the images | Diagnosis | Fix |
|---|---|---|---|
| 1 | 00030/s2: candidate turns a blue **square** into a **circle**, yet `fidelity` = 0.98 | `cvlib.shape_type` tested `circ > 0.80` *before* the vertex count. A perfect square has circularity π/4 = 0.785, and rasterising pushes it to 0.80–0.81 — so **every square was labelled a circle**. Separately, the same rendered object came back 'rect' in one frame and 'square' in another | Vertex count now precedes circularity; 'square'/'rect' merged into 'quad' (callers compare aspect ratio numerically). Validated: input and GT contain the same objects, and label disagreement went to **0/100 samples** |
| 2 | 00060/s0 (one of two objects correct) scored 0.25 where inspection says 0.50; 00080/s2 (two of three) scored 0.44 where it should be 0.67 | `color_matching` multiplied `mean(on_target)` by palette coverage. Since `match` already hard-gates on colour, a missing colour was **penalised twice** | Dropped the multiplication; palette coverage is now diagnostic only |
| 2b | *(tried and reverted)* charging surplus per **unmatched candidate** shape instead of per count difference | It re-introduced exactly the same double-count for *substitutions* — 00060/s0 and 00070/s1 each swap one object for another, which is one error, not two. Rows 06/08 regressed to 0.42/0.58 | Reverted to the count difference; both the attempt and the reason are recorded in the source |
| 3 (held-out, then re-opened) | 00085/s2 `star_invariance` = 0.00, but the candidate's blue marker is **2.4 px** from its input star — plainly unmoved | Its dE was 33.8. Markers are ~13 px across, so they are mostly anti-aliased edge; the generated marker is the same hue rendered *darker* — Lab (90,179,48) vs (130,172,51), a 40-unit **L** shift with a,b within 7 | Down-weighted lightness in `cvlib.dE` (`L_WEIGHT = 0.5`). Validated on the whole split: verified-unmoved markers (n=72) p95 **34.3 → 19.7**, while genuinely different palette colours stay **≥ 39.4** apart. `SAME_COLOR_DE = 25` now sits in a ~20-wide gap on both sides |

Pass 3 was found on the held-out batch, so under the protocol the batch was re-opened
and both batches re-verified together: calibration was **unchanged** by the fix, and the
change is a measurement correction validated against all 500 (sample, variant) pairs
rather than against the two batches.

Calibration agreement: **10/10**. Held-out agreement: **10/10** (9/10 before pass 3).

Unit tests: `tests/multi_object_placement_test.py`, 18 synthetic cases including blank,
perfect match, nothing-moved (right elements in the wrong place → 0 positional credit),
one object missing, colour-swapped, substitution-charged-once, surplus object,
shape substitution, size change, small offset tolerated vs one-object-width shift
penalised, alignment monotonicity, covered stars, unmoved star left visible, a star
drawn somewhere new, darker rendering still matching, and a clearly different colour
still rejected. All pass.

## Results

| variant | old rule | **new rule** | judge |
|---|---|---|---|
| base | 0.335 | **0.425** | 0.105 |
| s0 | 0.490 | **0.738** | 0.274 |
| s1 | 0.373 | **0.524** | 0.194 |
| s2 | 0.512 | **0.755** | 0.320 |
| s3 | 0.444 | **0.687** | 0.209 |

The new scorer reproduces the judge's **variant ordering exactly**
(base < s1 < s3 < s0 < s2) at a much higher absolute level. On inspection the level
difference is the judge's fault, not the scorer's, for two measurable reasons:

- **The judge marks `star_invariance` wrong on 91% of samples** (455/500) — but GT
  contains no visible stars in any of the 100 samples, so it is penalising the
  *correct* final state. Its own reasoning says so: *"no star markers, so objects
  cannot be on matching stars, violating … star_invariance."* That is a flat 0.125 of
  weight removed from nearly every sample.
- **The judge never returns `partial`** — 0 partials across all 500 × 4 sub-criteria.
  A candidate with two of three objects correctly placed scores 0 on `color_matching`,
  not 0.67.

For this task I trust the new scorer over the judge, and used my own visual
verdicts (not the judge's) as the agreement reference throughout.

## Known failure modes

- **`fidelity` is the weakest sub-criterion.** Shape class is a hard 0/1 on a
  three-way classifier, so a marginal circle/quad call swings 0.5 of an object's
  fidelity. It is also measured against the *input* object rather than GT, which is
  right in principle (the object should be unchanged by the move) but means a
  candidate that redraws every object slightly larger loses fidelity uniformly.
- **The surplus-object penalty (0.25 each, cap 0.5) is a judgement call**, not derived
  from data — it is the one threshold here with no measurement behind it.
- **Objects that shrink to marker size are reclassified as markers** (00050/base), so
  they vanish from `fidelity` and instead depress `star_invariance`. The total lands in
  the right place but the per-criterion attribution is misleading.
- **Colour matching assumes a flat, well-separated palette.** It is validated on this
  renderer's six colours (min separation 39.4 dE); a pastel or near-neighbour palette
  would need `SAME_COLOR_DE` re-derived.
- Heavily overlapping objects merge into one connected component and are then scored
  as a single mis-shaped object.
