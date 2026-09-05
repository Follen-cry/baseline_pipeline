"""Rule-based scorer for G-5_multi_object_placement (VBVR).

Scene: a white field holding N coloured shapes (N in 2..4; circle / square /
triangle, 1591-3245 px^2 in the 512-px working frame) and N small star markers
(a constant ~137 px^2) whose colours match the shapes one-for-one. The task is
to move each shape onto the star of its own colour.

Measured over all 100 GT frames of the eval split: GT contains **zero** visible
star markers -- each star ends up completely covered by its shape -- and every
GT shape centroid sits within 10 px (median 0.6 px) of the matching input star.
So the correct final state is exactly "every shape on its own star", and GT is a
direct, per-object positional reference.

Why this was rewritten
----------------------
The original scorer (In_Domain_50_part1.py:191, image-only override in
evaluators/image_evaluator.py:502) had four independent problems:

1. `_detect_colored_objects` enumerated four hardcoded HSV boxes -- red, blue,
   green, yellow. The split also uses magenta, cyan and orange, which fall in no
   box, so those objects were invisible to the scorer and every sub-criterion
   silently fell through to its `0.2` / `0.3` "detection failed" constant.
2. It ran `findContours` on a per-colour mask with no connected-component
   separation and no shape typing, so it could not tell a circle that turned
   into a square from a correct placement.
3. `color_matching` counted a GT object as matched if *any* candidate object of
   that colour was within 30 px -- a fixed pixel radius applied to objects whose
   width ranges 40-57 px -- and allowed many candidate objects to match the same
   GT object (`break` only exits the inner loop).
4. `star_invariance` compared candidate stars against *input* stars and returned
   0.3 whenever either set was empty. Since a correct solve covers every star,
   the correct answer scores the "detection failed" constant.

Sub-criterion names and weights are unchanged; `path` stays dropped (it measured
motion variance across video frames, undefined for a single generated frame) and
the remaining four are renormalised exactly as the existing image-only override
does.
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from . import cvlib as C
except ImportError:
    import cvlib as C

WEIGHTS = {
    'color_matching': 0.30 / 0.80,
    'alignment': 0.25 / 0.80,
    'fidelity': 0.15 / 0.80,
    'star_invariance': 0.10 / 0.80,
}

# Shapes and star markers are separated by area, not by vertex count. Measured
# across the split: shapes span 1591-3245 px^2, stars a constant 137-138 px^2.
# 600 sits in an empty gap an order of magnitude wide on both sides.
SHAPE_MIN_AREA = 600.0
MARKER_MIN_AREA = 30.0     # below this is anti-aliasing debris, not a marker
MARKER_MAX_AREA = 600.0

# Placement tolerances are expressed as a fraction of the GT object's own
# equivalent side length (sqrt(area), 40-57 px here) rather than as a fixed
# pixel radius, so the same rule applies to a small triangle and a large square.
ON_TARGET_FRAC = 0.60      # centroid within 0.60*side => "on its star"
ALIGN_GOOD_FRAC = 0.20     # <= 0.20*side scores 1.0 on alignment
ALIGN_BAD_FRAC = 0.80      # >= 0.80*side scores 0.0 (objects no longer overlap)
STAR_TOL_PX = 14.0         # a marker is "unmoved" within ~one marker width


def _split(img: np.ndarray) -> Tuple[List[Dict], List[Dict], np.ndarray]:
    mask = C.clean(C.chroma_mask(img), open_k=3, close_k=3)
    objs = C.objects(img, mask, min_area=MARKER_MIN_AREA)
    shapes = [o for o in objs if o['area'] >= SHAPE_MIN_AREA]
    markers = [o for o in objs if MARKER_MIN_AREA <= o['area'] < MARKER_MAX_AREA]
    return shapes, markers, mask


def _score(input_img, cand_img, gt_img) -> Tuple[Dict[str, float], Dict]:
    dbg: Dict = {}
    gt_shapes, gt_markers, gt_mask = _split(gt_img)
    cd_shapes, cd_markers, cd_mask = _split(cand_img)
    in_shapes, in_markers, in_mask = _split(input_img)
    dbg['counts'] = {'gt_shapes': len(gt_shapes), 'cand_shapes': len(cd_shapes),
                     'gt_markers': len(gt_markers), 'cand_markers': len(cd_markers),
                     'input_shapes': len(in_shapes), 'input_markers': len(in_markers)}
    dbg['masks'] = {'gt': gt_mask, 'cand': cd_mask}
    dbg['objs'] = {'gt': gt_shapes, 'cand': cd_shapes}

    # -- layer 1: cheap global gates ------------------------------------------
    gate = C.blankness(cand_img, gt_img)
    gt_bg, cd_bg = C.background_bgr(gt_img), C.background_bgr(cand_img)
    bg_de = C.dE(C.to_lab(gt_bg), C.to_lab(cd_bg))
    if bg_de > 40.0:
        gate['ceiling'] = min(gate['ceiling'], 0.15)
    dbg['blankness'] = gate
    dbg['background_dE'] = bg_de

    if not gt_shapes:
        return {k: 0.0 for k in WEIGHTS}, dbg

    # -- layer 2: colour multiset -------------------------------------------
    # A missing or extra colour is informative on its own and gates the rest.
    def palette(objs):
        pal = []
        for o in objs:
            if not any(C.dE(o['lab'], p) <= C.SAME_COLOR_DE for p in pal):
                pal.append(o['lab'])
        return pal

    gt_pal, cd_pal = palette(gt_shapes), palette(cd_shapes)
    missing = [p for p in gt_pal if not any(C.dE(p, q) <= C.SAME_COLOR_DE for q in cd_pal)]
    extra = [q for q in cd_pal if not any(C.dE(p, q) <= C.SAME_COLOR_DE for p in gt_pal)]
    colour_cover = 1.0 - len(missing) / max(1, len(gt_pal))
    dbg['palette'] = {'gt': len(gt_pal), 'cand': len(cd_pal),
                      'missing': len(missing), 'extra': len(extra)}

    # -- layer 3: per-object structural matching -----------------------------
    pairs = C.match(gt_shapes, cd_shapes, max_de=C.SAME_COLOR_DE, pos_weight=1.0)
    dbg['pairs'] = pairs

    on_target, align_terms, detail = [], [], []
    for gi, ci, d in pairs:
        g = gt_shapes[gi]
        side = float(np.sqrt(max(1.0, g['area'])))
        if ci is None:
            on_target.append(0.0)
            align_terms.append(0.0)
            detail.append({'gt': gi, 'cand': None, 'dist': None, 'side': round(side, 1),
                           'verdict': 'no colour-matched candidate object'})
            continue
        hit = 1.0 if d <= ON_TARGET_FRAC * side else 0.0
        al = C.band(d, ALIGN_GOOD_FRAC * side, ALIGN_BAD_FRAC * side)
        on_target.append(hit)
        align_terms.append(al)
        detail.append({'gt': gi, 'cand': ci, 'dist': round(d, 1), 'side': round(side, 1),
                       'on_target': bool(hit), 'align': round(al, 3)})
    dbg['per_object'] = detail

    # `colour_cover` is diagnostic only. It must NOT multiply `on_target`:
    # cvlib.match already hard-gates on colour, so a GT object whose colour is
    # missing from the candidate is *already* scored 0 as an unmatched object.
    # Multiplying by palette coverage on top of that penalised the same miss
    # twice -- 00060/s0 (one of two objects correct) scored 0.25 where
    # inspection says 0.50, and 00080/s2 (two of three) scored 0.44 where it
    # should be 0.67.
    #
    # An *extra* object in a colour GT does not use is a separate error that
    # per-object matching cannot see (nothing in GT is left unmatched by it),
    # so it keeps its own penalty.
    # Counting *unmatched candidate* shapes here was tried and reverted: when a
    # candidate substitutes one object for another (00060/s0 draws a second blue
    # square where the orange triangle belongs; 00070/s1 draws a second green
    # object where the blue triangle belongs) that is a single error, already
    # scored 0 as an unmatched GT object. Charging the surplus as well
    # reintroduced the same double-count as the colour_cover multiplication.
    # The penalty therefore fires only on genuine surplus -- more objects in the
    # candidate than GT has.
    extra_penalty = min(0.5, 0.25 * max(0, len(cd_shapes) - len(gt_shapes)))
    color_matching = max(0.0, float(np.mean(on_target)) - extra_penalty)
    alignment = max(0.0, float(np.mean(align_terms)) - 0.5 * extra_penalty)

    # fidelity: shape class and area preserved vs the *input* object of that colour
    fid = []
    for gi, ci, d in pairs:
        g = gt_shapes[gi]
        src = min((o for o in in_shapes), key=lambda o: C.dE(o['lab'], g['lab']), default=None)
        if ci is None or src is None or C.dE(src['lab'], g['lab']) > C.SAME_COLOR_DE:
            fid.append(0.0)
            continue
        c = cd_shapes[ci]
        shape_ok = 1.0 if c['shape'] == src['shape'] else 0.0
        ratio = min(c['area'], src['area']) / max(c['area'], src['area'], 1.0)
        area_ok = C.band(1.0 - ratio, 0.15, 0.60)
        fid.append(0.5 * shape_ok + 0.5 * area_ok)
    fidelity = float(np.mean(fid)) if fid else 0.0
    dbg['fidelity_terms'] = [round(x, 3) for x in fid]

    # star_invariance: GT shows no markers (they are covered). A marker the
    # candidate still draws is only acceptable where the input had one -- a
    # marker anywhere else is a hallucinated element.
    if not cd_markers:
        star_invariance = 1.0
        dbg['markers'] = 'none drawn (matches GT)'
    else:
        ok = 0
        for m in cd_markers:
            for s in in_markers:
                if C.dE(m['lab'], s['lab']) <= C.SAME_COLOR_DE and \
                        np.hypot(m['centroid'][0] - s['centroid'][0],
                                 m['centroid'][1] - s['centroid'][1]) <= STAR_TOL_PX:
                    ok += 1
                    break
        star_invariance = ok / len(cd_markers)
        dbg['markers'] = {'drawn': len(cd_markers), 'at_input_positions': ok}

    scores = {'color_matching': color_matching, 'alignment': alignment,
              'fidelity': fidelity, 'star_invariance': star_invariance}
    scores = {k: float(np.clip(v, 0.0, 1.0) * gate['ceiling']) for k, v in scores.items()}
    dbg['ceiling'] = gate['ceiling']
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
    gt_v = C.draw_objects(g, dbg['objs']['gt'], 'GT objects')
    cd_v = C.draw_objects(c, dbg['objs']['cand'], 'CAND objects')
    mt = C.draw_matches(gt_v, cd_v, dbg['objs']['gt'], dbg['objs']['cand'], dbg['pairs'])
    paths = {'gt_objs': f'{out_dir}/{tag}_gt_objs.png',
             'cand_objs': f'{out_dir}/{tag}_cand_objs.png',
             'matches': f'{out_dir}/{tag}_matches.png',
             'cand_mask': f'{out_dir}/{tag}_cand_mask.png'}
    cv2.imwrite(paths['gt_objs'], gt_v)
    cv2.imwrite(paths['cand_objs'], cd_v)
    cv2.imwrite(paths['matches'], mt)
    cv2.imwrite(paths['cand_mask'], C.mask_vis(dbg['masks']['cand']))
    dbg['debug_images'] = paths
    dbg['vis'] = {'match': mt}
    dbg['summary'] = {'per_object': dbg['per_object'], 'palette': dbg['palette'],
                      'markers': dbg['markers']}
    return scores, dbg


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('input'); ap.add_argument('candidate'); ap.add_argument('ground_truth')
    ap.add_argument('--debug-dir')
    a = ap.parse_args()
    if a.debug_dir:
        s, d = debug(a.input, a.candidate, a.ground_truth, a.debug_dir)
        print(json.dumps({'scores': s, 'overall': overall(s),
                          'per_object': d['per_object'], 'palette': d['palette']}, indent=2, default=str))
    else:
        s = score(a.input, a.candidate, a.ground_truth)
        print(json.dumps({'scores': s, 'overall': overall(s)}, indent=2))
