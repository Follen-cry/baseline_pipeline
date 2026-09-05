# rotation_puzzle — rule-based scorer rewrite

`scorers/rotation_puzzle.py`, replacing `RotationPuzzleEvaluator._evaluate_task_specific`
(VBVR-EvalKit `vbvr_bench/evaluators/In_Domain_50_part5.py:1006`). The old scorer is
untouched and still runnable for before/after comparison.

## The scene

A 2×2 grid of white tiles on a near-white background (BGR ≈ 252,250,248). Each tile
carries one coloured L-shaped ("elbow") pipe. The task is to rotate each elbow by a
multiple of 90° so the four arms join into a single closed ring. In the 512-px working
frame the grid is fixed: tiles at (138,138), (263,138), (138,263), (263,263), each
110×110, with a 15 px gutter between neighbours and a ~6 px pipe stroke.

Ground truth is always the same arm configuration: `TL={E,S}, TR={S,W}, BL={E,N}, BR={N,W}`.

## What was wrong with the old logic

**Every one of the four sub-criteria read the pipes through `_detect_blue_pipes`, a
hardcoded HSV window `[100,100,100]..[130,255,255]`.** The pipes in this task are
rendered in a per-sample colour — orange, yellow, magenta, cyan, green, red all appear
in the eval split. For any non-blue sample the mask came back completely empty and the
code fell through to its constants:

- `_evaluate_path_connection` → `if np.sum(gt_blue) == 0: return 0.5`
- `_evaluate_rotation_accuracy` → `return 0.5` (empty `quadrant_scores`)
- `_evaluate_alignment_precision` → `0/max(1, 0)` → 0.0

so most samples landed on the same weighted constant. That is the origin of the
"suspiciously identical ~0.55" scores: they were not measurements at all. It also
explains the flat 0.526–0.552 spread across all five model variants — the old scorer
was very nearly a constant function of its input.

Two further problems, independent of the colour bug:

- `path_connection` and `rotation_accuracy` were both plain **IoU of the pipe mask**
  (globally, then per quadrant). IoU is a similarity metric, not a connectivity test:
  it cannot distinguish "ring closed" from "ring with one arm 30 px short", and it
  rewards any candidate whose ink happens to overlap GT's ink.
- `position_preservation` compared 10-px-tall strips through the centre of the
  *generated* frame against the *input* frame with a mean-absolute-difference ladder
  (`<20 → 1.0`, `<50 → 0.7`). Because both frames are ~97% white background, that
  difference is tiny almost regardless of content, so this 20%-weight criterion sat
  near 1.0 for nearly every candidate.

## What the new scorer measures

Sub-criterion names and weights are unchanged (`path_connection` 0.40,
`rotation_accuracy` 0.30, `position_preservation` 0.20, `alignment_precision` 0.10).

**Layer 1 — global gates (cheap, can floor the score).**
`cvlib.blankness` compares candidate foreground pixel count to GT's; a pipe-pixel ratio
below 0.15 caps everything at 0.10, below 0.35 at 0.50. A separate *background* gate
compares the candidate's border colour to GT's in CIELAB — a candidate that repaints
the scene (sample 00058/base renders the ring on solid black) has abandoned the task
even though its foreground ratio looks healthy, and is capped at 0.15.

**Layer 2 — colour.** The dominant pipe colour of candidate and GT are compared with
CIE76 ΔE. A structurally correct ring in the wrong colour keeps only half credit on
`path_connection`, `rotation_accuracy` and `alignment_precision`.

**Layer 3 — per-cell structure.** The tile grid is recovered from the *GT* frame
(bright, unsaturated, square, solid connected components; falls back to the measured
constants if that finds ≠ 4 tiles) and the *same* cells are applied to the candidate,
so an object can only earn credit where GT says it belongs. Each cell is then reduced
to a discrete, exact state: **which of its four edge midpoints an arm actually reaches**,
a 2-of-{N,E,S,W} set. From that:

- `rotation_accuracy` — fraction of cells whose arm set exactly equals GT's; a cell
  sharing some but not all arms gets `0.5 · |A∩B| / |A∪B|`.
- `path_connection` — of the four internal junctions GT requires, the fraction joined
  on *both* sides in the candidate, minus 0.25 per dangling arm pointing at the outer
  border (a ring cannot have loose ends).
- `position_preservation` — per cell: tile still present *and* holding a pipe → 1.0;
  pipe but no tile → 0.6; tile whose pipe was erased → 0.5.
- `alignment_precision` — the physical offset, in px, between the centres of mass of
  the two arms meeting at each junction, ramped 1→0 over 3→14 px.

## Thresholds and where they come from

| Threshold | Value | Derivation |
|---|---|---|
| `ARM_GAP_PX` | 8 px | An arm counts as present if the pipe comes within 8 px of its tile edge. Measured over 2400 candidate arms (5 variants × 30 samples × 4 cells × 4 directions): **1197 arms land at 0–6 px, 1183 at 40–60 px, and only 19 (0.8%) fall anywhere between.** GT arms reach within 2–4 px; a GT non-arm sits 51 px away. The gutter is 15 px and the stroke ~6 px, so an arm must come within roughly one stroke width of its own edge for two ends to meet. Anything in 6–40 px would give the same answer on this data. |
| `ARM_FRAC` | 0.02 | Noise guard only — the arm band must contain *some* pixels. Not load-bearing. |
| `SAME_COLOR_DE` | 25 (ΔE) | Within-object colour spread across a candidate/GT pair measures < 12; distinct palette entries are > 45 apart. 25 sits in that gap. |
| background gate | ΔE > 40 | Same-scene backgrounds differ by < 5 ΔE (generation cast); the black-background failure differs by > 150. |
| alignment ramp | 3 → 14 px | GT junction offsets measure ≤ 3 px. 14 px is one gutter width — beyond that the arms visibly miss each other. |
| foreground ratio | 0.15 / 0.35 | Principle-1 ceilings, chosen to be obviously-blank / obviously-sparse rather than fitted. |

**The one threshold I deliberately did *not* move.** Two calibration cases (00081/s0,
00032/s3) scored 0.82 where the VLM judge said 1.0. Measuring rather than tuning: the
candidate's arm stops **36 px and 52 px** short of a 110-px tile edge, where GT stops
within 2–4 px. Those are real gaps of roughly half a tile, clearly visible on inspection.
The judge is eyeballing "looks like a ring"; the scorer is right and the threshold stays.

## Test-and-refine log

Calibration batch: sample slots 0,10,…,90 cycled across base/s0/s1/s2/s3.
(The first attempt used slots 0–9, which turned out to be *all* near-perfect
generations — no failures to calibrate against — so the batch was restratified to
stride the whole split.)

| Pass | Mismatch found by looking at the images | Diagnosis | Fix |
|---|---|---|---|
| 1 | 00058/base scored 0.29; the candidate draws the ring on a **black background** with no tiles — visually a total failure | Foreground-*ratio* gating cannot see an inverted background; the pipe mask still looked healthy | Added the CIELAB background-consistency gate (ceiling 0.15) |
| 1 | 00049/base: tiles present, all pipes gone. `position_preservation` returned 0.0 although the criterion is about tiles, which *are* in place | Sub-score was conflating "no pipe" with "no tile", i.e. doing the global gate's job badly | Per-cell credit now distinguishes tile-with-pipe / pipe-only / tile-only; the global ceiling does the flooring |
| 2 | 00020/s1: `BL.N` coverage 0.052 vs a 0.06 threshold — a coin-flip that no measurement could justify | Coverage *fraction* conflates a thin stroke with a short arm | Replaced the coverage test with a **physical reach-to-edge measurement in px**. Verified the reformulation is not a tuning hack: split means moved 0.905 → 0.899 |
| 3 | none | — | — |

Calibration agreement after pass 2: **10/10**.
Held-out batch (slots 5,15,…,95, same variant cycle, no tuning against it):
**10/10 agreement**, including correctly flagging 00085/s2 at 0.60 — a ring corrupted by
a spurious thick vertical bar.

Unit tests: `tests/rotation_puzzle_test.py`, 14 synthetic cases (blank, perfect match,
single pipe missing, one/all tiles mis-rotated, colour swap, repainted background,
3-px jitter tolerated vs half-tile shift penalised, short-arm connectivity both ways,
and a monotonicity check that the score never rises as more cells go wrong). All pass.

## Results

| variant | old rule | **new rule** | judge |
|---|---|---|---|
| base | 0.526 | **0.899** | 0.904 |
| s0 | 0.552 | **0.942** | 0.933 |
| s1 | 0.547 | **0.893** | 0.918 |
| s2 | 0.550 | **0.917** | 0.927 |
| s3 | 0.549 | **0.934** | 0.936 |

Per-sample correlation with the judge: Pearson **0.71** (old: 0.27), Spearman **0.64**
(old: 0.17). Score spread went from ~0.02 between variants (a near-constant) to a
distribution with sd 0.15–0.22 and 3–9 sub-0.5 samples per variant.

## Known failure modes

- **`alignment_precision` is the weakest sub-criterion.** With only 4 junctions and a
  3→14 px ramp it is coarse and quantised (values are effectively 0, 0.25, 0.5, 0.75, 1).
  It carries only 10% weight, but I would not read it on a single sample.
- **The GT arm configuration is constant across the whole split**, so the scorer has
  never been exercised against a GT that is *not* the closed ring. The logic derives
  everything from GT rather than hardcoding the ring, but that generality is untested.
- **A candidate that draws one continuous rounded square** (rather than four elbows with
  gutters) scores 1.0. I judged that correct on inspection — the ring is formed — but it
  is a rendering the task designer arguably did not intend.
- **Tile detection needs tiles brighter than 253 and unsaturated.** Candidates that tint
  the tiles (e.g. the green cast in 00039/base) fall back to the constant grid. That is
  the right behaviour here, but it means `position_preservation` under-reports tile loss
  on heavily colour-cast generations.
- The scorer assumes a 2×2 grid throughout (`JUNCTIONS` is hardcoded). It would need
  generalising for any other grid size.
