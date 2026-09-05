"""
Image-only evaluation wrappers for VBVR-Bench, covering the 15 locked
InternVL-U pilot tasks.

VBVR-Bench's evaluators are written against I2V (image-to-video) model output:
BaseEvaluator.evaluate() expects a `video_path` and scores first_frame_consistency /
final_frame_accuracy / temporal_smoothness / visual_quality / task_specific from
decoded video frames.

InternVL-U (and other single-image generation models) only produce a
first_frame + final_frame pair, with no intermediate trajectory. Two strategies
are used here, per task, decided by reading each task's `_evaluate_task_specific`
source directly (see Evaluation/VBVR-CustomEval/artifacts for the full writeup):

  Tier 1 (direct reuse): the task_specific logic already only reads
  video_frames[0] / video_frames[-1]. `ImageOnlyEvalMixin` feeds it a synthetic
  2-frame "video" [gen_first_frame, gen_final_frame] and the original,
  unmodified evaluator class is used as-is.

  Tier 2 / Tier 3 (redefined): the original logic depends on tracking an
  object across many sampled frames for one or more sub-metrics. For these,
  `_evaluate_task_specific` is overridden: sub-metrics that are still
  computable from [first_frame, final_frame] are reused verbatim (calling the
  original class's own helper methods), and sub-metrics that are fundamentally
  unmeasurable from a 2-frame endpoint pair are either replaced with an
  honest 2-frame equivalent (documented per class) or dropped, with the
  remaining TASK_WEIGHTS renormalized to sum to 1.0. Never left as a silent
  constant fallback that looks like a real measurement.

This module lives in VBVR-CustomEval, a sibling of the vendored VBVR-EvalKit
clone, and imports vbvr_bench from there via sys.path rather than editing the
vendored package in place -- keeps VBVR-EvalKit clean for future `git pull`s
and keeps all project-specific scoring logic (including hand-tuned rules in
`custom_rules/`) in one place that's tracked by the main repo.
"""

import os
import sys
from typing import Any, Dict, Optional

import cv2
import numpy as np

_VBVR_EVALKIT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "VBVR-EvalKit")
)
if _VBVR_EVALKIT_ROOT not in sys.path:
    sys.path.insert(0, _VBVR_EVALKIT_ROOT)

from vbvr_bench.utils import load_image, normalize_frame_size, safe_distance


def _visual_coherence_penalty(gen_frame: np.ndarray, gt_frame: np.ndarray,
                               thresh: float = 0.08, floor: float = 0.05, ceil: float = 0.30,
                               quant: int = 16, bg_dist: float = 18.0, palette_dist: float = 40.0,
                               min_palette_frac: float = 0.003) -> Dict[str, float]:
    """
    Cheap, generic sanity gate applied to every task's task_specific score.

    Found by inspecting real (untrained-base-model) generations 2026-08-26: every
    per-task detector below (color-blob detection, contour/shape matching, etc.)
    was written and validated (validation/test_harness.py) against clean synthetic
    "wrong" pairs -- a different sample's clean GT frame standing in for a bad
    generation. Real generations are not like that: they're often visually
    incoherent (blurry gradients, stray hues, text-like watermark artifacts baked
    into a shape) in ways that can still coincidentally trip a detector's loose
    color/position thresholds. Concretely: a task_specific score of 0.995 for a
    ball_bounces_given_time sample whose "ball" was actually a thin orange ring of
    dots with no resemblance to the GT's solid pink circle; a 1.0 for a
    shape_color_then_move sample where the shapes had garbled text baked into them
    and the wrong color entirely. A VLM judge shown the same image immediately
    called both wrong -- confirmed by eye, see judge reasoning in
    judge_eval/judge_scored_900.json for these ids.

    This computes what fraction of the generated final frame's non-background
    pixels have a color that doesn't resemble ANY color actually present in this
    sample's own GT final frame (the palette is per-sample, not per-task-hardcoded,
    since e.g. animal_size_sorting's palette differs sample to sample). A high
    off-palette fraction is a proxy for "this doesn't even look like a clean
    rendering in this task's visual style" -- not a replacement for the structural
    per-task checks, just a floor under them so a detector's false positive on
    incoherent output can't reach a high score.

    Two off-palette signals are combined (final off_frac = max of both), because
    a pure pixel-area fraction has a dilution blind spot: a big shared/static
    element (e.g. ball_bounces_given_time's square border, identical in every
    sample of that task) can dominate the foreground pixel count and hide a
    small-but-completely-wrong object (the actual scored ball) inside the noise.
    So alongside off_frac_area (pixel-count based), off_frac_blob runs
    cv2.connectedComponentsWithStats on the foreground mask and checks what
    FRACTION OF DISTINCT BLOBS (not pixels) are off-palette -- a small wrong
    object is then its own blob and can't hide behind a big correct one. Still
    imperfect: a thin, heavily anti-aliased shape (e.g. a 1-pixel-wide ring at
    200px working resolution) can have most of its pixels blend toward
    in-palette background/border tones, understating both signals -- a known
    residual gap, not chased further past this fix (diminishing returns; see
    validation notes in judge_eval/judge_scored_900.json for the specific case).

    Calibrated (thresh/floor/ceil) by grid search against the Qwen judge's
    scores on the same 900 real generations (see judge_eval/judge_scored_900.json):
    raised overall rule-vs-judge Spearman correlation from 0.14 (no gate) to
    0.27 with these settings, improving 7/9 active tasks and not hurting the
    other 2 (grid_shift ~unchanged, ball_bounces_given_time already dominated
    by judge-side floor effect -- 97/100 samples judged "wrong" regardless, too
    little judge-side variance for any rule-side change to correlate against).

    Returns {'off_frac': ..., 'off_frac_area': ..., 'off_frac_blob': ...,
    'penalty': ...}; penalty is a 0..1 multiplier, 1.0 = no penalty
    (off_frac <= thresh), floor at/above ceil, linear between.
    """
    if gt_frame is None or gen_frame is None:
        return {'off_frac': 0.0, 'penalty': 1.0}

    def _shrink(img, max_side=200):
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                              interpolation=cv2.INTER_AREA)
        return img

    gt = _shrink(gt_frame).astype(np.float32)
    gen = _shrink(gen_frame, max_side=max(gt.shape[:2])).astype(np.float32)
    if gen.shape[:2] != gt.shape[:2]:
        gen = cv2.resize(gen, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)

    corners = np.concatenate([
        gt[0, 0:5].reshape(-1, 3), gt[0, -5:].reshape(-1, 3),
        gt[-1, 0:5].reshape(-1, 3), gt[-1, -5:].reshape(-1, 3),
    ])
    bg = np.median(corners, axis=0)

    def fg_mask(img):
        return np.linalg.norm(img - bg[None, None, :], axis=2) > bg_dist

    gt_fg = gt.reshape(-1, 3)[fg_mask(gt).reshape(-1)]
    if len(gt_fg) < 20:
        return {'off_frac': 0.0, 'penalty': 1.0}  # GT itself is ~blank; nothing to gate against

    q = (gt_fg // quant * quant).astype(np.int32)
    uniq, counts = np.unique(q, axis=0, return_counts=True)
    keep = counts / len(gt_fg) >= min_palette_frac
    palette = uniq[keep].astype(np.float32)
    if len(palette) == 0:
        palette = uniq.astype(np.float32)

    gen_mask = fg_mask(gen)
    gen_fg = gen.reshape(-1, 3)[gen_mask.reshape(-1)]
    if len(gen_fg) == 0:
        return {'off_frac': 0.0, 'penalty': 1.0}

    dists = np.linalg.norm(gen_fg[:, None, :] - palette[None, :, :], axis=2)
    is_off = dists.min(axis=1) > palette_dist
    off_frac_area = float(is_off.sum() / len(gen_fg))

    # Area-weighted off_frac alone is dominated by whichever element covers the
    # most pixels -- for tasks like ball_bounces_given_time where a big static
    # border occupies nearly all foreground pixels and the actual scored object
    # (a small ball) is a tiny fraction, a completely-wrong small object gets
    # diluted to near-zero off_frac_area even though it's 100% wrong. Also check
    # per-connected-component: what fraction of distinct foreground BLOBS (not
    # pixels) are off-palette, so a small-but-wrong object can't hide behind a
    # large shared/correct one. Final off_frac is whichever signal is stronger.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        gen_mask.astype(np.uint8), connectivity=8)
    off_frac_blob = 0.0
    if n_labels > 1:
        off_mask_full = np.zeros(gen_mask.shape, dtype=bool)
        off_mask_full[gen_mask] = is_off
        blob_ids = [i for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= 4]
        if blob_ids:
            off_blobs = sum(1 for i in blob_ids if off_mask_full[labels == i].mean() > 0.5)
            off_frac_blob = off_blobs / len(blob_ids)

    off_frac = max(off_frac_area, off_frac_blob)

    if off_frac <= thresh:
        penalty = 1.0
    elif off_frac >= ceil:
        penalty = floor
    else:
        frac = (off_frac - thresh) / (ceil - thresh)
        penalty = 1.0 - frac * (1.0 - floor)

    return {'off_frac': off_frac, 'off_frac_area': off_frac_area, 'off_frac_blob': off_frac_blob,
            'penalty': penalty}


class ImageOnlyEvalMixin:
    """
    Mixin that overrides BaseEvaluator.evaluate() to score a first_frame/final_frame
    image pair instead of a video file. Mix in *before* the concrete video evaluator
    class so this evaluate() takes priority over BaseEvaluator's:

        class FooImageEvaluator(ImageOnlyEvalMixin, FooEvaluator): pass

    Expected eval_info keys (in addition to the usual gt_* keys):
        - final_frame_path: path to the model-generated final frame (required)
        - first_frame_path: path to the model-generated first frame (optional;
          if absent, falls back to the GT first frame -- valid for I2V-style
          models that are expected to reproduce the condition image verbatim)

    temporal_smoothness is intentionally omitted from `dimensions` (undefined for
    a 2-frame sequence). BaseEvaluator._calculate_overall_score / weighted_average
    only sums weights for keys present in `dimensions`, so the remaining 4 weights
    (first_frame_consistency 0.15, final_frame_accuracy 0.35, visual_quality 0.10,
    task_specific 0.25 -> total 0.85) are renormalized automatically, no override
    needed.
    """

    def evaluate(self, eval_info: Dict, **kwargs) -> Dict[str, Any]:
        result = {'score': 0.0, 'dimensions': {}, 'details': {'image_only_mode': True}}

        try:
            gen_final_frame = self._load_gen_frame(eval_info, 'final_frame_path')
            if gen_final_frame is None:
                raise ValueError(
                    f"final_frame_path missing or unreadable: {eval_info.get('final_frame_path')}"
                )

            gen_first_frame = self._load_gen_frame(eval_info, 'first_frame_path')
            gt_first_frame = self._load_gt_first_frame(eval_info)
            gt_final_frame = self._load_gt_final_frame(eval_info)
            gt_frames = self._load_gt_video_frames(eval_info)

            if gen_first_frame is None:
                # I2V-style assumption: the model is expected to reproduce the
                # condition frame verbatim, so fall back to GT's first frame.
                gen_first_frame = gt_first_frame if gt_first_frame is not None else gen_final_frame
                result['details']['first_frame_fallback'] = True

            # Normalize sizes the same way BaseEvaluator.evaluate() does.
            target_frame = gt_first_frame if gt_first_frame is not None else gt_final_frame
            if target_frame is not None:
                if gen_first_frame.shape != target_frame.shape:
                    gen_first_frame = normalize_frame_size(gen_first_frame, target_frame)
                if gen_final_frame.shape != target_frame.shape:
                    gen_final_frame = normalize_frame_size(gen_final_frame, target_frame)

            video_frames = [gen_first_frame, gen_final_frame]
            result['details']['video_frame_count'] = len(video_frames)

            dimensions = {}

            if gt_first_frame is not None:
                dimensions['first_frame_consistency'] = self._evaluate_first_frame(
                    gen_first_frame, gt_first_frame
                )

            if gt_final_frame is not None:
                dimensions['final_frame_accuracy'] = self._evaluate_final_frame(
                    gen_final_frame, gt_final_frame
                )
            else:
                dimensions['final_frame_accuracy'] = 0.0

            dimensions['visual_quality'] = self._evaluate_visual_quality([gen_final_frame])

            task_score = self._evaluate_task_specific(
                video_frames, gt_frames, gt_first_frame, gt_final_frame, eval_info
            )
            task_score = max(0.0, min(1.0, task_score))

            coherence = _visual_coherence_penalty(gen_final_frame, gt_final_frame)
            result['details']['visual_coherence'] = coherence
            task_score = task_score * coherence['penalty']
            dimensions['task_specific'] = task_score

            if hasattr(self, '_last_task_details'):
                result['details']['task_specific_details'] = self._last_task_details

            if kwargs.get('task_specific_only'):
                result['dimensions'] = {'task_specific': task_score}
                result['score'] = task_score
            else:
                result['dimensions'] = dimensions
                result['score'] = self._calculate_overall_score(dimensions)

        except Exception as e:
            result['error'] = str(e)
            result['score'] = 0.0

        return result

    @staticmethod
    def _load_gen_frame(eval_info: Dict, key: str) -> Optional[np.ndarray]:
        path = eval_info.get(key)
        if path and os.path.exists(path):
            return load_image(path)
        return None


# =============================================================================
# Tier 1 -- direct reuse (task_specific already reads only first/final frame)
# =============================================================================

from vbvr_bench.evaluators.In_Domain_50_part1 import (  # noqa: E402
    StableSortEvaluator,
    GridShortestPathEvaluator,
)
from vbvr_bench.evaluators.In_Domain_50_part2 import KeyDoorMatchingEvaluator  # noqa: E402
from vbvr_bench.evaluators.In_Domain_50_part3 import (  # noqa: E402
    ShapeColorThenScaleEvaluator,
    ShapeOutlineThenMoveEvaluator,
    BallBounceEvaluator,
    MirrorReflectionEvaluator,
    GlassRefractionEvaluator,
)
from vbvr_bench.evaluators.In_Domain_50_part5 import (  # noqa: E402
    GridShiftEvaluator,
    RotationPuzzleEvaluator,
)
from vbvr_bench.evaluators.Out_of_Domain_50_part4 import (  # noqa: E402
    GeometricTransformationEvaluator,
    ShapeColorThenMoveEvaluator,
    MazePathfindingEvaluator,
)
from vbvr_bench.evaluators.Out_of_Domain_50_part5 import (  # noqa: E402
    AnimalSizeSortingEvaluator,
)
from vbvr_bench.evaluators.In_Domain_50_part1 import MultiObjectPlacementEvaluator  # noqa: E402


class ShapeColorThenScaleImageEvaluator(ImageOnlyEvalMixin, ShapeColorThenScaleEvaluator):
    """O-12_shape_color_then_scale. Pilot task -- validated against the video
    evaluator on matched (identical scores) and deliberately mismatched
    (identical scores) GT pairs."""
    pass


class ShapeOutlineThenMoveImageEvaluator(ImageOnlyEvalMixin, ShapeOutlineThenMoveEvaluator):
    """O-13_shape_outline_then_move. Centroid comparisons only touch
    video_frames[0] / video_frames[-1]."""
    pass


class MirrorReflectionImageEvaluator(ImageOnlyEvalMixin, MirrorReflectionEvaluator):
    """O-19_mirror_reflection. Ray/angle detection runs on the final frame alone."""
    pass


class RotationPuzzleImageEvaluator(ImageOnlyEvalMixin, RotationPuzzleEvaluator):
    """O-44_rotation_puzzle. Pipe connectivity is a property of the final tile
    arrangement; evaluator diffs frame 0 vs frame -1 only."""
    pass


class StableSortImageEvaluator(ImageOnlyEvalMixin, StableSortEvaluator):
    """G-3_stable_sort. Sort correctness reads off the final arrangement only."""
    pass


class ShapeColorThenMoveImageEvaluator(ImageOnlyEvalMixin, ShapeColorThenMoveEvaluator):
    """O-11_shape_color_then_move (OOD). Shape/color detection on frame 0 and
    frame -1 only."""
    pass


class MazePathfindingImageEvaluator(ImageOnlyEvalMixin, MazePathfindingEvaluator):
    """
    O-39_maze (OOD). Every sub-score compares the maze image in frame 0
    against frame -1 (wall/marker detection), not a replayed solve animation.

    REVISED 2026-08-26 after auditing a fresh random sample of real
    generations against the vendored evaluator's source (Out_of_Domain_50_part4.py):
    3 of the 4 sub-metrics are structurally blind to path LENGTH, only to
    local properties of whatever path pixels exist --
      - path_validity: violation_ratio = wall-overlap-pixels / path-pixels.
        A 5-pixel scribble that touches no wall scores 1.0, same as a full
        solve.
      - navigation_accuracy: connected-component count == 1 blob -> 1.0. A
        tiny isolated stub is exactly as "continuous" as a full path.
      - path_completeness: only checks proximity to start/end, not the path
        connecting them -- a stub sitting right next to the start marker
        alone already scores 0.6.
    Confirmed by inspection: multiple real samples with only a short
    scribble near the start marker (visibly nowhere near solving the maze)
    scored 0.4-0.71 from path_validity=1.0 + navigation=1.0 + a partial
    completeness credit, none of which required the path to actually go
    anywhere. Fixed the same way as GridShiftImageEvaluator gates on
    block_preserved: compute how many path-marker pixels the candidate
    drew relative to how many the GT's own full-solve path uses (its own
    detector, no new detection logic), and gate the whole task_specific
    score by that coverage ratio -- a full, GT-length solve is unaffected
    (gate ~1.0), a short stub is heavily discounted regardless of how
    "clean" its local pixels look.
    """

    def _evaluate_task_specific(self, video_frames, gt_frames, gt_first_frame, gt_final_frame, eval_info):
        base_score = super()._evaluate_task_specific(
            video_frames, gt_frames, gt_first_frame, gt_final_frame, eval_info
        )
        if gt_final_frame is None or not video_frames:
            return base_score

        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        if gen_final.shape != gt_final.shape:
            gt_final = normalize_frame_size(gt_final, gen_final)

        gen_path = self._detect_path_markers(gen_final)
        gt_path = self._detect_path_markers(gt_final)
        gen_px = int((gen_path > 0).sum()) if gen_path is not None else 0
        gt_px = int((gt_path > 0).sum()) if gt_path is not None else 0

        if gt_px < 20:
            coverage = 1.0  # GT itself has no meaningful path pixel count to compare against
        else:
            ratio = gen_px / gt_px
            coverage = min(1.0, ratio)  # over-drawing isn't rewarded further, only under-drawing penalized

        low, high, floor = 0.15, 0.6, 0.1
        if coverage >= high:
            gate = 1.0
        elif coverage <= low:
            gate = floor
        else:
            frac = (coverage - low) / (high - low)
            gate = floor + frac * (1.0 - floor)

        details = getattr(self, '_last_task_details', {})
        details['path_coverage_gate'] = {'gen_path_px': gen_px, 'gt_path_px': gt_px,
                                          'coverage': coverage, 'gate': gate}
        self._last_task_details = details
        return base_score * gate


class AnimalSizeSortingImageEvaluator(ImageOnlyEvalMixin, AnimalSizeSortingEvaluator):
    """O-65_animal_size_sorting (OOD). Final-arrangement check, same pattern
    as stable_sort."""
    pass


# =============================================================================
# Tier 2 -- reuse with one sub-metric dropped + renormalized
# =============================================================================

class GridShiftImageEvaluator(ImageOnlyEvalMixin, GridShiftEvaluator):
    """
    O-36_grid_shift, image-only mode.

    `synchronization` (20% weight in the original) needs >=3 sampled frames to
    check that all blocks move in lockstep over time. CONFIRMED unrecoverable
    (2026-08-25), not just deferred: unlike ball_bounces_given_time,
    final_frame.png here shows only the blocks' resting positions, no motion
    trail or timing cue of any kind -- there is no pixel in a single frame
    that carries "did these move in sync." Deliberately dropped, not
    replaced with a judge guess (a judge shown one frame has exactly as
    little signal as the rule-based code does; asking it anyway would
    fabricate a number, not measure anything). Renormalized across the other
    4 sub-metrics, which were already first/final-frame only in the
    original code.
    """

    IMAGE_TASK_WEIGHTS = {
        'direction_correctness': 0.30 / 0.80,
        'step_accuracy': 0.30 / 0.80,
        'position_precision': 0.15 / 0.80,
        'completeness': 0.05 / 0.80,
    }

    def _evaluate_task_specific(self, video_frames, gt_frames, gt_first_frame, gt_final_frame, eval_info):
        if not video_frames or gt_final_frame is None:
            return 0.0

        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        if gen_final.shape != gt_final.shape:
            gt_final = normalize_frame_size(gt_final, gen_final)

        first_blocks = self._detect_colored_blocks(first_frame)
        gen_final_blocks = self._detect_colored_blocks(gen_final)
        gt_final_blocks = self._detect_colored_blocks(gt_final)

        completeness_score = self._evaluate_completeness(first_blocks, gen_final_blocks)
        pattern_score = self._evaluate_block_pattern_preservation(
            first_frame, gen_final, first_blocks, gen_final_blocks
        )
        block_preserved = min(completeness_score, pattern_score) > 0.5

        scores = {'completeness': min(completeness_score, pattern_score)}
        if not block_preserved:
            scores['direction_correctness'] = 0.0
            scores['step_accuracy'] = 0.0
            scores['position_precision'] = 0.0
        else:
            scores['direction_correctness'] = self._evaluate_direction(
                first_blocks, gen_final_blocks, gt_final_blocks
            )
            scores['step_accuracy'] = self._evaluate_step_accuracy(
                first_blocks, gen_final_blocks, gt_final_blocks, gen_final
            )
            scores['position_precision'] = self._evaluate_position_precision(
                gen_final_blocks, gt_final_blocks
            )

        self._last_task_details = scores
        return sum(scores[k] * self.IMAGE_TASK_WEIGHTS[k] for k in self.IMAGE_TASK_WEIGHTS)


class MultiObjectPlacementImageEvaluator(ImageOnlyEvalMixin, MultiObjectPlacementEvaluator):
    """
    G-5_multi_object_placement, image-only mode.

    `path` (20% weight) measures motion-smoothness variance across up to 10
    sampled frames. With only 2 points it isn't just noisy, it's structurally
    trivial (a straight line has zero variance by construction), so keeping it
    would silently always score near 1.0 rather than measuring anything.
    CONFIRMED unrecoverable (2026-08-25): checked final_frame.png for any
    drawn motion trail the way ball_bounces_given_time has -- there is none,
    objects just sit at their resting position. Deliberately dropped rather
    than replaced with a judge guess (no visual signal for either method to
    read). Renormalized across color_matching/alignment/fidelity/star_invariance.
    """

    IMAGE_TASK_WEIGHTS = {
        'color_matching': 0.30 / 0.80,
        'alignment': 0.25 / 0.80,
        'fidelity': 0.15 / 0.80,
        'star_invariance': 0.10 / 0.80,
    }

    def _evaluate_task_specific(self, video_frames, gt_frames, gt_first_frame, gt_final_frame, eval_info):
        if len(video_frames) < 2 or gt_final_frame is None:
            return 0.0

        scores = {}
        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        gen_objects = self._detect_colored_objects(last_frame)
        gt_objects = self._detect_colored_objects(gt_final_frame)

        if gen_objects and gt_objects:
            matched = 0
            for gen_obj in gen_objects:
                for gt_obj in gt_objects:
                    if gen_obj['color'] == gt_obj['color']:
                        if safe_distance(gen_obj['center'], gt_obj['center']) < 30:
                            matched += 1
                            break
            scores['color_matching'] = matched / max(len(gt_objects), 1)
        else:
            scores['color_matching'] = 0.2

        if gen_objects and gt_objects:
            total_dist, count = 0.0, 0
            for gen_obj in gen_objects:
                min_dist = float('inf')
                for gt_obj in gt_objects:
                    if gen_obj['color'] == gt_obj['color']:
                        min_dist = min(min_dist, safe_distance(gen_obj['center'], gt_obj['center']))
                if min_dist < float('inf'):
                    total_dist += min_dist
                    count += 1
            avg_dist = total_dist / count if count > 0 else 100
            scores['alignment'] = max(0, 1.0 - avg_dist / 50.0)
        else:
            scores['alignment'] = 0.2

        first_objects = self._detect_colored_objects(first_frame)
        if first_objects and gen_objects:
            count_ratio = min(len(gen_objects), len(first_objects)) / max(len(gen_objects), len(first_objects), 1)
            first_area = sum(o['area'] for o in first_objects)
            gen_area = sum(o['area'] for o in gen_objects)
            area_ratio = min(first_area, gen_area) / max(first_area, gen_area, 1)
            scores['fidelity'] = 0.5 * count_ratio + 0.5 * area_ratio
        else:
            scores['fidelity'] = 0.2

        first_stars = self._detect_star_markers(first_frame)
        final_stars = self._detect_star_markers(last_frame)
        if first_stars and final_stars:
            preserved = 0
            for fs in first_stars:
                for ls in final_stars:
                    if fs['color'] == ls['color']:
                        if safe_distance(fs['center'], ls['center']) < 20:
                            preserved += 1
                            break
            scores['star_invariance'] = preserved / max(len(first_stars), 1)
        else:
            scores['star_invariance'] = 0.3

        self._last_task_details = scores
        return sum(scores[k] * self.IMAGE_TASK_WEIGHTS[k] for k in self.IMAGE_TASK_WEIGHTS)


class GridShortestPathImageEvaluator(ImageOnlyEvalMixin, GridShortestPathEvaluator):
    """
    G-18_grid_shortest_path, image-only mode.

    RECLASSIFIED after empirical validation caught this: `path_optimal` (50%
    weight!) sums consecutive-position distances across *every* sampled frame
    of both the generated and GT video (`for frame in video_frames: ...` /
    `for frame in gt_frames: ...`) and compares total path length. This was
    missed by manual code review (grep for `video_frames[` and
    `(video_frames)` both miss a bare `for frame in video_frames:` loop) and
    only surfaced because the 20-pair validation run showed only 20% exact
    agreement with the video evaluator even on *correctly paired* samples --
    with only 2 frames, gen_path_len collapses to the straight-line
    start-to-end distance while gt_path_len (computed from the real,
    un-truncated gt_frames) stays the true winding path length, so the ratio
    is structurally biased low. Dropped and renormalized across
    completion/movement/fidelity, which are still first/final-frame-only
    (movement's diagonal-move check degrades to checking a single start->end
    segment, weaker signal but not fabricated).
    """

    IMAGE_TASK_WEIGHTS = {
        'completion': 0.25 / 0.50,
        'movement': 0.15 / 0.50,
        'fidelity': 0.10 / 0.50,
    }

    def _evaluate_task_specific(self, video_frames, gt_frames, gt_first_frame, gt_final_frame, eval_info):
        if len(video_frames) < 1 or gt_final_frame is None:
            return 0.0

        last_frame = video_frames[-1]
        agent_positions = []
        for frame in video_frames:
            pos = self._detect_agent(frame)
            if pos is not None:
                agent_positions.append(pos)

        endpoint = self._detect_endpoint(last_frame)
        final_agent = self._detect_agent(last_frame)
        gt_final_agent = self._detect_agent(gt_final_frame)

        scores = {}
        if final_agent is not None and gt_final_agent is not None:
            dist = np.sqrt((final_agent[0] - gt_final_agent[0]) ** 2 + (final_agent[1] - gt_final_agent[1]) ** 2)
            scores['completion'] = max(0, 1.0 - dist / 50.0)
        elif endpoint is not None and final_agent is not None:
            dist = np.sqrt((endpoint[0] - final_agent[0]) ** 2 + (endpoint[1] - final_agent[1]) ** 2)
            scores['completion'] = 1.0 if dist < 50 else max(0, 1.0 - dist / 100.0)
        else:
            scores['completion'] = 0.2

        if len(agent_positions) >= 2:
            diagonal_count = 0
            for i in range(1, len(agent_positions)):
                dx = abs(agent_positions[i][0] - agent_positions[i - 1][0])
                dy = abs(agent_positions[i][1] - agent_positions[i - 1][1])
                if dx > 10 and dy > 10:
                    diagonal_count += 1
            scores['movement'] = max(0, 1.0 - diagonal_count * 0.2)
        else:
            scores['movement'] = 0.2

        scores['fidelity'] = 1.0 if (final_agent is not None or agent_positions) else 0.2

        self._last_task_details = scores
        return sum(scores[k] * self.IMAGE_TASK_WEIGHTS[k] for k in self.IMAGE_TASK_WEIGHTS)


class GeometricTransformationImageEvaluator(ImageOnlyEvalMixin, GeometricTransformationEvaluator):
    """
    O-6_2d_geometric_transformation (OOD), image-only mode.

    `rotation_center` (30% weight) fits a circular arc to >=3 sampled shape
    centers; with 2 frames the original code hard-returns 0.0 (a fixed
    penalty, not a neutral default). CONFIRMED unrecoverable (2026-08-25):
    final_frame.png shows only the shape's end orientation plus a static
    pivot-point marker dot -- no arc, sweep, or trail is drawn, so there's no
    pixel evidence of *how* it got there, only where it ended up (which
    rotation_angle/position_alignment already score). Deliberately dropped
    rather than replaced with a judge guess. Renormalized across
    rotation_angle/position_alignment/shape_fidelity.
    """

    IMAGE_TASK_WEIGHTS = {
        'rotation_angle': 0.35 / 0.70,
        'position_alignment': 0.25 / 0.70,
        'shape_fidelity': 0.10 / 0.70,
    }

    def _evaluate_task_specific(self, video_frames, gt_frames, gt_first_frame, gt_final_frame, eval_info):
        if len(video_frames) < 2:
            return 0.0
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        scores = {
            'rotation_angle': self._evaluate_rotation_angle(first_frame, final_frame),
            'position_alignment': self._evaluate_position(first_frame, final_frame),
            'shape_fidelity': self._evaluate_shape_fidelity(first_frame, final_frame),
        }
        self._last_task_details = scores
        return sum(scores[k] * self.IMAGE_TASK_WEIGHTS[k] for k in self.IMAGE_TASK_WEIGHTS)


# =============================================================================
# Tier 3 -- majority of the original weight is trajectory-dependent; redefined
# =============================================================================

class BallBounceImageEvaluator(ImageOnlyEvalMixin, BallBounceEvaluator):
    """
    O-15_ball_bounces_given_time, image-only mode.

    REVISED (2026-08-25): a `path_shape` (IoU of drawn orange-path masks)
    criterion briefly lived here, validated against VBVR-Bench's official GT
    samples where final_frame.png renders the entire bounce path as a static
    polyline. But freshly generating new samples via this task's own
    DataFactory generator (current version, same day) showed the visual
    style has since changed upstream: final_frame.png now shows only the
    ball resting at its final position, no path drawn at all. Whatever the
    IoU check was validated against no longer reflects what this task's data
    actually looks like going forward, so it's been removed rather than kept
    as a silently-broken metric.

    task_specific is now `final_position` alone (100%): distance between the
    ball's position in the generated final frame and in GT's final frame,
    reusing the original _track_ball_positions detector unchanged. No VLM
    judge for this task (removed from llm_judge.py's RUBRICS/judge_harness.py
    too) -- final_position is a plain, deterministic, already-validated
    check; a judge would add non-determinism without covering anything
    final_position doesn't already cover once the path is gone. bounce_count/
    physics/trajectory/smoothness (the original 4 video-mode sub-metrics)
    remain unrecoverable from a single frame under the current visual style
    and are not replaced.
    """

    def _evaluate_task_specific(self, video_frames, gt_frames, gt_first_frame, gt_final_frame, eval_info):
        if not video_frames or gt_final_frame is None:
            return 0.0
        final_frame = video_frames[-1]

        gen_pos = self._track_ball_positions([final_frame])
        gt_pos = self._track_ball_positions([gt_final_frame])
        if not gen_pos or not gt_pos:
            score = 0.2  # detection failed; matches the original's own fallback convention
        else:
            dist = np.sqrt((gen_pos[0][0] - gt_pos[0][0]) ** 2 + (gen_pos[0][1] - gt_pos[0][1]) ** 2)
            score = max(0.0, 1.0 - dist / 100.0)

        self._last_task_details = {'final_position': score}
        return score


class KeyDoorMatchingImageEvaluator(ImageOnlyEvalMixin, KeyDoorMatchingEvaluator):
    """
    G-45_key_door_matching, image-only mode.

    key_collected / door_reached / sequence (70% of the original weight)
    require the agent to be spatially co-located with the key/door in a
    *sampled* frame -- with only [first, final], an agent that legitimately
    collected the key mid-path is almost always scored as if it never did.
    Redefined:
      - agent_movement (kept, reweighted): distance(start, end) -- already
        only needs 2 points, reuses _evaluate_agent_movement unchanged.
      - key_collected (redefined): visual presence/absence diff -- a key seen
        in first_frame that is gone from final_frame counts as collected,
        regardless of path. Trades "did it walk there" for "did it visibly
        happen", the only thing 2 frames can actually tell us; a model could
        in principle game this by deleting the key sprite without navigating
        to it -- a known, documented limitation of image-only scoring here.
      - door_reached (redefined): is the agent's final position near a door
        detected in first_frame, with matching key/door color.
      - sequence (dropped): key-then-door ordering is unrecoverable from 2
        static frames. Renormalized across the other 3.
    """

    IMAGE_TASK_WEIGHTS = {
        'agent_movement': 0.30 / 0.90,
        'key_collected': 0.35 / 0.90,
        'door_reached': 0.25 / 0.90,
    }

    def _evaluate_task_specific(self, video_frames, gt_frames, gt_first_frame, gt_final_frame, eval_info):
        if len(video_frames) < 2:
            return 0.0
        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        first_pos = self._detect_agent(first_frame)
        last_pos = self._detect_agent(last_frame)
        positions = [p for p in (first_pos, last_pos) if p is not None]
        if len(positions) < 2:
            self._last_task_details = {'error': 'agent_not_detected'}
            return 0.0

        movement_score, _ = self._evaluate_agent_movement(positions)
        scores = {'agent_movement': movement_score}

        if movement_score < 0.3:
            scores['key_collected'] = 0.0
            scores['door_reached'] = 0.0
            self._last_task_details = scores
            return sum(scores.get(k, 0.0) * w for k, w in self.IMAGE_TASK_WEIGHTS.items())

        first_keys = self._detect_keys(first_frame)
        last_keys = self._detect_keys(last_frame)
        first_doors = self._detect_doors(first_frame)

        collected_color = None
        for key in first_keys:
            still_there = any(
                lk['color'] == key['color'] and safe_distance(lk['center'], key['center']) < 50
                for lk in last_keys
            )
            if not still_there:
                collected_color = key['color']
                break
        scores['key_collected'] = 1.0 if collected_color else 0.0

        door_score = 0.0
        if collected_color and first_doors:
            best_door = min(first_doors, key=lambda d: safe_distance(last_pos, d['center']))
            best_dist = safe_distance(last_pos, best_door['center'])
            if best_dist < 50:
                door_score = 1.0 if best_door['color'] == collected_color else 0.0
        scores['door_reached'] = door_score

        self._last_task_details = scores
        return sum(scores[k] * self.IMAGE_TASK_WEIGHTS[k] for k in self.IMAGE_TASK_WEIGHTS)


class GlassRefractionImageEvaluator(ImageOnlyEvalMixin, GlassRefractionEvaluator):
    """
    O-18_glass_refraction. Replaces O-62_gravity_physics in the locked task
    list (2026-08-25): gravity_physics's official generator doesn't clamp the
    ball at ground level, so for any parameter draw where the ball hits the
    ground before the stated duration elapses, the simulated height goes
    negative and the ball is rendered off-canvas -- confirmed by inspection
    (4 of 5 downloaded GT samples have an empty final_frame) and by computing
    h(3) = h0 + v0*3 - 0.5*g*9 for each, which is negative in exactly those 4
    cases. Every scoring method (rule-based or VLM judge) inherits this
    upstream data defect. glass_refraction has no such failure mode -- Snell's
    law refraction geometry always stays on-canvas by construction -- and its
    task_specific logic only reads video_frames[0]/[-1] (Tier 1, no
    degradation needed), a strict upgrade over gravity_physics's Tier 3
    status. It ships under VBVR-Bench's own In-Domain_50 folder, but our
    project picks its own train/test split rather than inheriting VBVR's --
    we simply exclude glass_refraction's data from InternVL-U training and
    treat it as our OOD task. (Knowledge's ID slot in the current
    LOCKED_TASKS set went to ball_bounces_given_time instead of the
    similarly-clean mirror_reflection -- a deliberate choice, see
    LOCKED_TASKS' comment below for the reasoning; don't infer a specific
    per-category ratio from this docstring, it drifts -- LOCKED_TASKS is the
    source of truth.)
    """
    pass


# =============================================================================
# Registry: full generator task_name -> image-only evaluator class
# =============================================================================

TASK_IMAGE_EVALUATOR_MAP = {
    'O-12_shape_color_then_scale_data-generator': ShapeColorThenScaleImageEvaluator,
    'O-13_shape_outline_then_move_data-generator': ShapeOutlineThenMoveImageEvaluator,
    'O-19_mirror_reflection_data-generator': MirrorReflectionImageEvaluator,
    'O-44_rotation_puzzle_data-generator': RotationPuzzleImageEvaluator,
    'G-3_stable_sort_data-generator': StableSortImageEvaluator,
    'G-18_grid_shortest_path_data-generator': GridShortestPathImageEvaluator,
    'O-11_shape_color_then_move_data-generator': ShapeColorThenMoveImageEvaluator,
    'O-39_maze_data-generator': MazePathfindingImageEvaluator,
    'O-65_animal_size_sorting_data-generator': AnimalSizeSortingImageEvaluator,
    'O-36_grid_shift_data-generator': GridShiftImageEvaluator,
    'G-5_multi_object_placement_data-generator': MultiObjectPlacementImageEvaluator,
    'O-6_2d_geometric_transformation_data-generator': GeometricTransformationImageEvaluator,
    'O-15_ball_bounces_given_time_data-generator': BallBounceImageEvaluator,
    'G-45_key_door_matching_data-generator': KeyDoorMatchingImageEvaluator,
    'O-18_glass_refraction_data-generator': GlassRefractionImageEvaluator,
}


def get_image_evaluator(task_name: str, device: str = 'cpu'):
    """Get the image-only evaluator for a given task name."""
    cls = TASK_IMAGE_EVALUATOR_MAP[task_name]
    return cls(device=device, task_name=task_name)


# =============================================================================
# Locked task list (2026-08-25): the 10 tasks actually used for InternVL-U
# training/eval, out of the 15 implemented above. Ratio isn't uniform per
# category -- user-selected per category based on (a) whether the ID task's
# image-only score is reliable and (b) whether multiple ID tasks in a
# category are redundant with each other:
#   Abstraction:     0 ID (both shape_color_then_scale and
#                    shape_outline_then_move dropped as redundant with each
#                    other) + 1 OOD
#   Knowledge:       1 ID (ball_bounces_given_time -- kept deliberately
#                    despite its documented Known Issue, over the cleaner
#                    Tier-1 mirror_reflection) + 1 OOD
#   Perception:      2 ID (both kept) + 1 OOD
#   Spatiality:      0 ID (both grid_shortest_path and key_door_matching
#                    dropped -- the two most eval-problematic tasks in the
#                    set) + 1 OOD
#   Transformation:  2 ID (both kept) + 1 OOD
# All 15 evaluator classes above stay implemented (cheap to re-include a
# task later); this set is just which ones test_harness.py / the artifacts
# currently render.
# =============================================================================

LOCKED_TASKS = {
    # ID (5)
    'O-15_ball_bounces_given_time_data-generator',   # Knowledge
    'G-3_stable_sort_data-generator',                # Perception
    'G-5_multi_object_placement_data-generator',     # Perception
    'O-36_grid_shift_data-generator',                # Transformation
    'O-44_rotation_puzzle_data-generator',           # Transformation
    # OOD (5, all)
    'O-11_shape_color_then_move_data-generator',     # Abstraction
    'O-18_glass_refraction_data-generator',          # Knowledge
    'O-65_animal_size_sorting_data-generator',       # Perception
    'O-39_maze_data-generator',                      # Spatiality
    'O-6_2d_geometric_transformation_data-generator',  # Transformation
}
