"""Rule-based scorer for O-6_2d_geometric_transformation (VBVR, OOD).

Scene: a light grey field (BGR 240,240,240) holding one coloured polygon at its
start pose, a thin outlined *target silhouette* showing where it must end up, and
a small pivot marker. The task is to rotate the polygon about the pivot so it
lands on the target silhouette.

Measured across the eval split:

  * The GT final frame contains **exactly one** foreground component (30/30
    sampled): the polygon at its target pose. The target outline and the pivot
    marker are gone.
  * Required displacement from start to target: p5 21 px, median 49 px, p95 74 px,
    minimum 17 px -- against a shape whose equivalent side is ~41 px, so the
    move is always substantial.
  * Required rotation spans the full circle (measured best-fit values from -168
    to +178 deg), and the shape is rigid: the best-fit IoU at the correct angle
    is p50 0.924, p5 0.770.

Why this was rewritten
----------------------
`GeometricTransformationEvaluator` (Out_of_Domain_50_part4.py:282) and its
image-only override (evaluators/image_evaluator.py:659) never looked at the
ground-truth final frame at all -- every sub-criterion compared the generated
final frame against the *input*:

1. `_evaluate_rotation_angle` compared `_get_contour_angle` of the input's target
   outline against the final shape. A single contour angle (minAreaRect /
   principal axis) is ambiguous under symmetry and unstable for near-square
   shapes, and it returns a flat `0.5` whenever either detection fails.
2. `_evaluate_position` measured the distance from the final shape to the
   *input's target-outline centre*, with a fixed ladder (`<30 → 1.0`, `<60 → 0.7`,
   `<100 → 0.4`, else 0.2). The floor of 0.2 means a shape left completely
   untouched still scores 0.2, and `0.5` is returned if either centre is missing.
3. `_evaluate_shape_fidelity` compared areas only, with the same `0.5` fallback.
4. `_detect_main_shape` selects contours by `hsv[:,:,1] > 50`. This palette is
   deliberately muted -- (135,160,132), (135,135,171) -- and **returns zero
   objects** for many samples, so those constant fallbacks fired routinely.

The rewrite scores against GT, which is exact and available.

`rotation_center` stays dropped (it needs >= 3 sampled frames to fit an arc, and
a single generated frame draws no trail), and the remaining three weights stay
renormalised exactly as the existing image-only override does.
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from . import cvlib as C
except ImportError:
    import cvlib as C

WEIGHTS = {
    'rotation_angle': 0.35 / 0.70,
    'position_alignment': 0.25 / 0.70,
    'shape_fidelity': 0.10 / 0.70,
}

FG_TOL = 20.0        # L2 BGR from the grey background; flat-region jitter is < 8
MIN_SHAPE_AREA = 300.0   # the polygon is 1400-3000 px^2; outline fragments are < 100
MIN_PART_AREA = 30.0     # anything above this counts as a drawn element

# Rotation bands. GT's own best-fit residual is <= 2 deg, and the rendering
# quantisation of a ~41 px shape makes angles below ~8 deg visually
# indistinguishable; 45 deg is a quarter turn, unambiguously the wrong pose.
ROT_GOOD_DEG = 8.0
ROT_BAD_DEG = 45.0
# Rigid-body IoU: the same polygon at its best-fit angle scores p50 0.924,
# p5 0.770 in GT. Below 0.55 the silhouette is no longer the same shape.
FID_GOOD_IOU = 0.80
FID_BAD_IOU = 0.55


def _parts(img: np.ndarray, min_area: float = MIN_PART_AREA) -> List[Dict]:
    """All drawn elements, found relative to the detected background.

    Saturation-based detection is unusable here: `chroma_mask` returns **zero**
    objects for this task's muted palette (e.g. BGR 135,160,132), which is what
    drove the original evaluator into its constant fallbacks.
    """
    return C.objects(img, C.clean(C.foreground_mask(img, tol=FG_TOL), open_k=3, close_k=3),
                     min_area=min_area)


def _grey_ink(img: np.ndarray, tol: float = 12.0, neutral: float = 22.0) -> int:
    """Count of neutral-grey foreground pixels darker than the background.

    Isolates the dashed target outline and the pivot marker from the coloured
    polygon, which the outline may be touching.
    """
    bg = C.background_bgr(img)
    f = img.astype(np.float32)
    d = np.linalg.norm(f - bg.astype(np.float32), axis=2)
    spread = f.max(axis=2) - f.min(axis=2)
    darker = f.mean(axis=2) < bg.mean() - 6.0
    return int(((d > tol) & (spread < neutral) & darker).sum())


def _main_shape(img: np.ndarray, ref_lab=None) -> Optional[Dict]:
    """The coloured polygon: the largest element, preferring the reference colour."""
    objs = [o for o in _parts(img, MIN_SHAPE_AREA)]
    if not objs:
        return None
    if ref_lab is not None:
        same = [o for o in objs if C.dE(o['lab'], ref_lab) <= C.SAME_COLOR_DE]
        if same:
            objs = same
    return max(objs, key=lambda o: o['area'])


def _score(input_img, cand_img, gt_img) -> Tuple[Dict[str, float], Dict]:
    dbg: Dict = {}
    gt_shape = _main_shape(gt_img)
    in_shape = _main_shape(input_img, gt_shape['lab'] if gt_shape else None)
    cd_shape = _main_shape(cand_img, gt_shape['lab'] if gt_shape else None)
    dbg['found'] = {'gt': gt_shape is not None, 'input': in_shape is not None,
                    'cand': cd_shape is not None}

    # -- layer 1: global gates ------------------------------------------------
    gate = C.blankness(cand_img, gt_img)
    bg_de = C.dE(C.to_lab(C.background_bgr(gt_img)), C.to_lab(C.background_bgr(cand_img)))
    if bg_de > 40.0:
        gate['ceiling'] = min(gate['ceiling'], 0.15)
    dbg['blankness'] = gate
    dbg['background_dE'] = bg_de

    if gt_shape is None or cd_shape is None:
        dbg['error'] = 'no shape detected in GT or candidate'
        return {k: 0.0 for k in WEIGHTS}, dbg

    dbg['shapes'] = {'gt': gt_shape, 'cand': cd_shape, 'input': in_shape}

    # -- layer 2: colour ------------------------------------------------------
    col_de = C.dE(gt_shape['lab'], cd_shape['lab'])
    colour_ok = col_de <= C.SAME_COLOR_DE
    dbg['colour_dE'] = round(col_de, 1)

    # -- layer 3: pose --------------------------------------------------------
    side = float(np.sqrt(max(1.0, gt_shape['area'])))
    err = float(np.hypot(cd_shape['centroid'][0] - gt_shape['centroid'][0],
                         cd_shape['centroid'][1] - gt_shape['centroid'][1]))
    required = (float(np.hypot(in_shape['centroid'][0] - gt_shape['centroid'][0],
                               in_shape['centroid'][1] - gt_shape['centroid'][1]))
                if in_shape is not None else 0.0)

    # Position is scored against *the move the task actually asked for*, not a
    # fixed pixel ladder. A candidate that leaves the shape where it started is
    # `required` px away, which by construction lands at or beyond the zero
    # point -- no credit for the right element in the wrong place.
    # `bad` is the *starting* error. A candidate still that far from GT has made
    # no progress and scores 0 by construction. An earlier version used
    # `max(0.60*side, 0.55*required)`, which on samples with a short required
    # move was dominated by the shape size -- 00010/s0 leaves the shape exactly
    # where it started (err 17 px == required 17 px) and still scored 0.34.
    # The 0.30*side floor only guards against a degenerate `required` near 0.
    good = max(0.10 * side, 3.0)
    bad = max(required, 0.30 * side)
    position_alignment = C.band(err, good, bad)
    dbg['position'] = {'err_px': round(err, 1), 'required_px': round(required, 1),
                       'side_px': round(side, 1), 'good': round(good, 1),
                       'bad': round(bad, 1)}

    rot, rot_iou = C.best_rotation(gt_shape, cd_shape)

    # Object-identity gate. Pose is only meaningful for an object that really is
    # the polygon. A candidate that erased the shape and left only the hollow
    # target outline still yields a component whose centroid sits exactly on
    # GT's, so it would otherwise collect near-full position and rotation credit
    # for drawing no shape at all.
    #
    # The separating quantity is the best-fit IoU, not area: measured on the
    # synthetic fixtures, outline-only scores 0.134 while a shape at 0.6x scale
    # scores 0.377, a substituted circle 0.778 and the true shape 1.000 -- their
    # *areas* overlap (0.239 vs 0.374) but their silhouette overlaps do not.
    # Because best_rotation searches all angles, a correctly drawn shape scores
    # high here whatever its orientation, so this gate is independent of the
    # rotation criterion. The 0.45 upper bound sits well below the p5 (0.770) of
    # genuinely-identical objects measured on the split.
    identity = C.band(1.0 - rot_iou, 1.0 - 0.45, 1.0 - 0.20)
    dbg['identity'] = {'best_fit_iou': round(rot_iou, 3), 'gate': round(identity, 3)}
    position_alignment *= identity
    rotation_angle = C.band(abs(rot), ROT_GOOD_DEG, ROT_BAD_DEG) * identity
    dbg['rotation'] = {'best_fit_deg': round(rot, 1), 'iou_at_best': round(rot_iou, 3)}

    # shape_fidelity: the polygon must be rigid -- same silhouette and area as
    # the input object, only re-posed. Measured against the input, not GT, so a
    # candidate that lands on target by *deforming* the shape is caught.
    if in_shape is None:
        shape_fidelity = 0.0
        dbg['fidelity'] = 'input shape not found'
    else:
        _, rig_iou = C.best_rotation(in_shape, cd_shape)
        ratio = min(cd_shape['area'], in_shape['area']) / max(cd_shape['area'], in_shape['area'], 1.0)
        shape_fidelity = 0.5 * C.band(1.0 - rig_iou, 1.0 - FID_GOOD_IOU, 1.0 - FID_BAD_IOU) \
            + 0.5 * C.band(1.0 - ratio, 0.15, 0.55)
        dbg['fidelity'] = {'rigid_iou': round(rig_iou, 3), 'area_ratio': round(ratio, 3)}

    # -- residual target outline ---------------------------------------------
    # The task is complete only once the dashed target silhouette is gone; GT
    # removes it (leaving the polygon plus the small pivot marker). Counting
    # connected components does not detect this -- when the candidate lands the
    # shape on the outline the two touch and merge into one component
    # (00030/s2) -- and neither does total foreground area, because the outline
    # is a thin stroke worth only ~15% of it.
    #
    # Measured instead as *neutral grey ink*: foreground pixels that are
    # unsaturated and darker than the background. Input holds outline + pivot
    # (p50 695 px), GT holds pivot and the polygon's own border (p50 274 px).
    # The quantity below is the fraction of the outline ink that should have
    # been erased and was not, so it is 0 for a clean solve regardless of how
    # much grey the shape itself carries.
    ink_c, ink_g, ink_i = (_grey_ink(cand_img), _grey_ink(gt_img), _grey_ink(input_img))
    denom = max(1.0, ink_i - ink_g)
    retained = float(np.clip((ink_c - ink_g) / denom, 0.0, 1.0))
    # Deliberately a mild multiplier: leftover outline is a real completion
    # failure but is not one of the three sub-criteria the rubric names.
    residual_mult = float(1.0 - 0.25 * retained)
    dbg['residual'] = {'ink_cand': ink_c, 'ink_gt': ink_g, 'ink_input': ink_i,
                       'outline_retained': round(retained, 3),
                       'multiplier': round(residual_mult, 3)}

    scores = {'rotation_angle': rotation_angle,
              'position_alignment': position_alignment,
              'shape_fidelity': shape_fidelity}
    mult = gate['ceiling'] * residual_mult * (1.0 if colour_ok else 0.5)
    scores = {k: float(np.clip(v, 0.0, 1.0) * mult) for k, v in scores.items()}
    dbg['ceiling'] = gate['ceiling']
    dbg['colour_ok'] = colour_ok
    return scores, dbg


# ------------------------------------------------------------------- API ----

def _coerce(x, size: int = C.WORK):
    if x is None:
        return None
    return C.load(x, size) if isinstance(x, str) else C.resize_to(x, size)


def score(input_img, candidate_img, ground_truth_img) -> Dict[str, float]:
    i, c, g = _coerce(input_img), _coerce(candidate_img), _coerce(ground_truth_img)
    if c is None or g is None:
        return {k: 0.0 for k in WEIGHTS}
    return _score(i if i is not None else g, c, g)[0]


def overall(sub: Dict[str, float]) -> float:
    return float(sum(sub[k] * WEIGHTS[k] for k in WEIGHTS))


def debug(input_img, candidate_img, ground_truth_img, out_dir='.', tag='sample'):
    import os
    os.makedirs(out_dir, exist_ok=True)
    i, c, g = _coerce(input_img), _coerce(candidate_img), _coerce(ground_truth_img)
    if i is None:
        i = g
    scores, dbg = _score(i, c, g)
    vis = c.copy()
    if 'shapes' in dbg:
        gt_s, cd_s = dbg['shapes']['gt'], dbg['shapes']['cand']
        cv2.drawContours(vis, [gt_s['contour']], -1, (0, 160, 0), 2)     # GT pose
        cv2.drawContours(vis, [cd_s['contour']], -1, (0, 0, 230), 2)     # candidate
        p1 = tuple(int(v) for v in gt_s['centroid'])
        p2 = tuple(int(v) for v in cd_s['centroid'])
        cv2.line(vis, p1, p2, (200, 0, 200), 1, cv2.LINE_AA)
        cv2.drawMarker(vis, p1, (0, 160, 0), cv2.MARKER_CROSS, 13, 2)
        cv2.drawMarker(vis, p2, (0, 0, 230), cv2.MARKER_CROSS, 13, 2)
        cv2.putText(vis, f"d={dbg['position']['err_px']:.0f}/{dbg['position']['required_px']:.0f}px"
                         f" rot={dbg['rotation']['best_fit_deg']:.0f}deg",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, f"outline_left={dbg['residual']['outline_retained']:.2f}", (6, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    paths = {'overlay': f'{out_dir}/{tag}_overlay.png',
             'cand_mask': f'{out_dir}/{tag}_cand_mask.png',
             'gt_mask': f'{out_dir}/{tag}_gt_mask.png'}
    cv2.imwrite(paths['overlay'], vis)
    cv2.imwrite(paths['cand_mask'], C.mask_vis(C.clean(C.foreground_mask(c, tol=FG_TOL), 3, 3)))
    cv2.imwrite(paths['gt_mask'], C.mask_vis(C.clean(C.foreground_mask(g, tol=FG_TOL), 3, 3)))
    dbg['debug_images'] = paths
    dbg['vis'] = {'overlay': vis}
    dbg['summary'] = {k: dbg.get(k) for k in ('position', 'rotation', 'fidelity', 'residual')}
    return scores, dbg


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('input'); ap.add_argument('candidate'); ap.add_argument('ground_truth')
    ap.add_argument('--debug-dir')
    a = ap.parse_args()
    if a.debug_dir:
        s, d = debug(a.input, a.candidate, a.ground_truth, a.debug_dir)
        print(json.dumps({'scores': s, 'overall': overall(s), **d.get('summary', {})},
                         indent=2, default=str))
    else:
        s = score(a.input, a.candidate, a.ground_truth)
        print(json.dumps({'scores': s, 'overall': overall(s)}, indent=2))
