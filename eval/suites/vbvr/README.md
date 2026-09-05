# VBVR-CustomEval

Our own scoring layer for the VBVR tasks used to train/eval InternVL-U, built
on top of the vendored [`VBVR-EvalKit`](../VBVR-EvalKit) clone but kept
separate from it so upstream `git pull`s never conflict with our changes.

`image_evaluator.py` implements image-only evaluators for **15** tasks, but
the **locked set actually used for training/eval is 10** (`LOCKED_TASKS` in
that file) -- not a uniform per-category ratio. Selected 2026-08-25 by
dropping, per category, whichever ID task(s) were either redundant with
another ID task in the same category or too eval-problematic to trust:

| Category | ID kept | OOD (all kept) |
|---|---|---|
| Abstraction | *(none)* | `shape_color_then_move` |
| Knowledge | `ball_bounces_given_time` | `glass_refraction` |
| Perception | `stable_sort`, `multi_object_placement` | `animal_size_sorting` |
| Spatiality | *(none)* | `maze` |
| Transformation | `grid_shift`, `rotation_puzzle` | `2d_geometric_transformation` |

Dropped: `shape_color_then_scale` + `shape_outline_then_move` (redundant with
each other), `mirror_reflection` (kept `ball_bounces_given_time` instead
despite its Known Issue below -- deliberate choice, not an oversight),
`grid_shortest_path` + `key_door_matching` (the two most eval-problematic
tasks in the set). All 15 evaluator classes stay implemented either way --
re-including a dropped task later is just adding it back to `LOCKED_TASKS`.

## Why this exists

`../VBVR-EvalKit/vbvr_bench/evaluators/` is the official repo, cloned as-is.
Its evaluators are written for I2V (image-to-video) models — they score a full
generated video against ground truth. InternVL-U only produces a single
`final_frame`, so none of those evaluators can be used unmodified. This
directory holds the adaptation layer, kept out of the vendored tree on
purpose:

- **`../VBVR-EvalKit/`** — do not edit. Official source, safe to `git pull`.
- **`VBVR-CustomEval/`** (here) — everything project-specific: the image-only
  adaptation, hand-tuned scoring overrides, validation, and the two
  reference artifacts explaining all of it.

## Layout

```
evaluators/
├── image_evaluator.py      15 image-only evaluator classes (ImageOnlyEvalMixin
│                            + per-task subclasses), registered in
│                            TASK_IMAGE_EVALUATOR_MAP / get_image_evaluator()
├── llm_judge.py             VLM-judge scoring (talks to a local OpenAI-compatible
│                            vision server, e.g. the qwen3-vl-30b-fp8 vLLM
│                            instance on localhost:8000) for the tasks where
│                            rule-based scoring is known weak. Supplements
│                            image_evaluator.py's scores, doesn't replace them.
└── custom_rules/            Put hand-tuned per-task scoring overrides here.
                              Empty for now — see "Customizing a task's rules"
                              below for the pattern to follow.

validation/
├── test_harness.py                  Builds 20 test pairs per task (5 correct +
│                                     15 deliberately mismatched, from the 5
│                                     downloaded GT samples) and runs both the
│                                     original video evaluator and our image
│                                     evaluator on each pair.
├── validation_results.json          Full per-pair results from the last run
│                                     (300 evaluations across the original 15
│                                     tasks, 0 errors, 2026-08-24; filtered to
│                                     the 10 in LOCKED_TASKS on 2026-08-25).
├── judge_harness.py                 Same 20-pair methodology, but through
│                                     llm_judge.py instead of the rule-based
│                                     evaluators -- only for the tasks the
│                                     judge is meant to supplement.
└── judge_validation_results.json    Results from the last judge_harness.py run.

artifacts/
├── gen_artifact.py                 Builds the "VBVR Image-Only Evaluation" page
├── gen_rules_artifact.py           Builds the "VBVR Scoring Rules" page
├── vbvr_task_examples.json         Cached sample-00000 frames+prompts (base64),
│                                    reused by both generators so they don't
│                                    need to touch VBVR-Bench/ every rebuild
├── vbvr_image_eval_strategy.html   Published: https://claude.ai/code/artifact/8ceaf7f8-4746-4c73-8633-3b631864b063
└── vbvr_scoring_rules.html         Published: https://claude.ai/code/artifact/11f5c502-3153-47b1-924e-3a72471ce293
```

## How the 15 tasks are classified

- **Tier 1** (9 tasks): `_evaluate_task_specific` in the original evaluator
  only ever reads `video_frames[0]` / `video_frames[-1]`. `ImageOnlyEvalMixin`
  feeds it a synthetic 2-frame "video" and the original class is used
  unmodified — verified to reproduce the video evaluator's score exactly.
- **Tier 2** (4 tasks): mostly first/final-frame logic, but one sub-metric
  needed ≥3 sampled frames. Dropped and renormalized in an overridden
  `_evaluate_task_specific`.
- **Tier 3** (2 tasks): the majority of the original weight was
  trajectory-dependent. Redefined more substantially, reusing the original
  class's own detector helpers (e.g. `_detect_keys`, `_track_ball_positions`)
  but on 2 frames instead of a full video.

Full task-by-task reasoning, weight breakdowns, and validation numbers are in
the two artifacts above (`gen_artifact.py` / `gen_rules_artifact.py` build
them; re-run after editing task metadata in those files).

## Known issues

- **`O-15_ball_bounces_given_time`: two pivots on the same day (2026-08-25),
  now simplified to a single deterministic check.** First pivot: noticed
  `final_frame.png` in VBVR-Bench's frozen GT renders the entire bounce path
  as a static drawn orange polyline, so bounce_count/physics/trajectory
  looked jointly recoverable from one frame. Precise vertex/corner extraction
  was tried first (Hough line-segment clustering, then
  `cv2.goodFeaturesToTrack`) and was too noisy on these images. Replaced with
  a more robust proxy: `path_shape` (70%) = IoU between the generated and GT
  orange-path masks, validated at correct=1.000 / wrong=0.004-0.009 across
  the 5 downloaded samples (~0.99 gap); `final_position` (30%) kept the
  previously-working ball-position check. Full 20-pair re-validation at that
  point: gap improved from 0.75 to 0.92.

  Second pivot, same day: generating 10 fresh samples via this task's own
  DataFactory generator (for the `eval_samples/` test pipeline, unrelated
  work) surfaced that the generator's visual style has since changed
  upstream -- `final_frame.png` from the live generator no longer draws any
  bounce path, only the ball resting at its final position. Whatever
  `path_shape` was validated against no longer reflects what this task's
  data actually looks like going forward, and a metric that's silently wrong
  is worse than no metric. Removed `path_shape` entirely; `final_position`
  (unchanged) is now the sole task_specific score at 100% weight. Re-
  validated: gap is 0.75 (identical to the pre-path_shape number, since
  final_position was untouched both times). No VLM judge is used for this
  task -- final_position is already a plain deterministic check.

  This is still the deliberately-chosen Knowledge ID task in `LOCKED_TASKS`
  (over the cleaner Tier-1 `mirror_reflection`) -- kept specifically to track
  this kind of live-generator-vs-frozen-benchmark drift, which is a real,
  general risk of this project's setup (VBVR-DataFactory generator repos are
  independently versioned from the frozen VBVR-Bench GT snapshot they
  originally produced) and could in principle affect other locked tasks too,
  though only this one has been checked and confirmed so far. Separately,
  the earlier lesson about not assuming a dropped sub-metric is unrecoverable
  without checking the actual final_frame still holds (multi_object_placement's
  `path`, `grid_shift`'s `synchronization`, and `2d_geometric_transformation`'s
  `rotation_center` were all checked the same way on 2026-08-25 and confirmed
  to have no recoverable signal -- no drawn trail/arc in any of their final
  frames).

- **`O-62_gravity_physics` was dropped from the locked 15-task list (2026-08-25),
  replaced by `O-18_glass_refraction`.** Its official generator doesn't clamp
  the falling object at ground level: for any parameter draw where the object
  would hit the ground before the stated duration elapses, the simulated
  height goes negative and it's rendered off-canvas. Confirmed by computing
  h(3) = h0 + v0*3 - 0.5*g*9 for all 5 downloaded GT samples -- negative in
  exactly the 4 samples whose `final_frame.png` is empty. This was
  independently caught two ways: (1) direct visual inspection, and (2) the
  VLM judge (see below) scoring GT-vs-itself near 0 because it correctly
  noticed no object was visible. Every scoring method inherits this upstream
  defect, so it wasn't fixable by changing the evaluator. `glass_refraction`
  has no equivalent failure mode and is Tier 1 (no image-only adaptation
  needed at all). It ships in VBVR-Bench's own In-Domain_50 folder, but our
  ID/OOD split is our own choice, not VBVR's -- we simply exclude
  glass_refraction's data from training and treat it as our OOD task, so
  Knowledge stays a clean 2 ID / 1 OOD like every other category (10 ID / 5
  OOD overall). See `GlassRefractionImageEvaluator`'s docstring in
  `image_evaluator.py`.

- **`G-45_key_door_matching`'s official GT data is frequently ambiguous, not
  just its evaluator.** Cross-validated two independent ways: the rule-based
  key-disappearance detector (`_detect_keys` diffed across frames) found no
  key ever collected in 4 of 5 downloaded samples, and the VLM judge
  independently reached the same conclusion by just looking at the frames
  (3 of 5 scored "agent did not move / key not collected" on GT-vs-itself).
  Any scoring method built on this task's GT will be measuring "is this GT
  sample legible" as much as "is the model's answer correct" unless the
  degenerate samples are filtered first. Resolved by dropping the task
  entirely from `LOCKED_TASKS` (2026-08-25) rather than filtering samples --
  the evaluator code and rubric stay in place if it's ever worth revisiting
  with a cleaner GT subset.

## Running things

```bash
# Re-run the 20-pair validation for all 15 tasks (~15-20 min, mostly video decode)
cd validation && python3 test_harness.py

# Rebuild both artifacts after editing task metadata
cd artifacts && python3 gen_artifact.py && python3 gen_rules_artifact.py
# then re-publish each .html to its existing artifact URL (see file headers above)
```

`image_evaluator.py` adds `../VBVR-EvalKit` to `sys.path` at import time (no
`pip install` needed — this repo's Python env is a shared base conda env, so
we avoid installing into it).

## Customizing a task's rules

To hand-tune scoring for a task instead of using the default Tier 1/2/3
adaptation in `image_evaluator.py`:

1. Add a file under `evaluators/custom_rules/`, e.g. `custom_rules/key_door_matching.py`.
2. Subclass the existing image evaluator (or `ImageOnlyEvalMixin` + the
   original vendored class directly) and override `_evaluate_task_specific`.
3. Point `TASK_IMAGE_EVALUATOR_MAP['G-45_key_door_matching_data-generator']`
   at your new class instead of the default one in `image_evaluator.py`.
4. Re-run `validation/test_harness.py` for that task to confirm the change
   didn't silently break the discriminative gap or introduce errors.
