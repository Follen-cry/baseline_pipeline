"""Rule-based scorer for O-11_shape_color_then_move (VBVR, OOD).

Scene: a fixed 2x3 grid of glyphs on white. The top row is a worked example
A -> B -> C; the bottom row is D -> ? -> ?, and the model must produce E and F
by applying the same transformation.

The transformation is exactly what the task name says -- **colour, then move**:

    A --recolour--> B --translate--> C

Measured across all 100 GT frames of the eval split:

  * GT always holds exactly 6 objects; the input frame always exactly 4.
  * Column centres are fixed at x = 95.6 / 239.6 / 383.6 (sd 4.2) in the 512-px
    working frame; rows split cleanly at y = 256.
  * The B->C displacement and the E->F displacement are **identical to within
    1.0 px** in every sample. dx is always +144 (the column pitch); dy varies
    per sample over 0, +/-12, +/-20, +/-30, +/-40, +/-50.

Note the third column is *displaced*, not enlarged: B and C have identical area
and bounding-box size (e.g. 79x79 in both), differing only in y. An earlier read
of these frames as "the shape grows" is wrong.

Why this was rewritten
----------------------
`ShapeColorThenMoveEvaluator` (Out_of_Domain_50_part4.py:704) barely looked at
the image:

1. `_evaluate_second_row` -- 35% of the weight -- is
   `if len(final_bottom) >= 3: return 1.0`. It counts connected components in
   the bottom half of the frame and nothing else. Three blobs of any colour, any
   shape, anywhere in the bottom half scored a perfect 1.0.
2. `_evaluate_first_row_preservation` -- 40% of the weight -- compares *sorted
   hue lists*. A top row with the right colours in the wrong cells scores 1.0,
   and so does a top row whose glyphs have been replaced with different shapes.
3. `_evaluate_color_accuracy` reads B's hue from whatever shape happens to lie
   in the middle third of the input's top half, then compares raw HSV hue with a
   +/-20 window -- unstable for the dark, low-saturation colours in this palette.
4. `_detect_shapes_with_info` keeps `1000 < area < 20000` and records only
   centroid, hue and area: no shape identity at all.

The rewrite scores **per grid cell against GT**, which is available and exact.

Sub-criterion names and weights are unchanged.
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from . import cvlib as C
except ImportError:
    import cvlib as C

WEIGHTS = {
    'first_row_preservation': 0.40,
    'second_row_completion': 0.35,
    'color_accuracy': 0.20,
    'shape_count': 0.05,
}

MIN_AREA = 200.0        # p5 of real object area is 1497; 200 only drops debris
FG_TOL = 25.0           # L2 BGR distance from background; flat-region jitter is < 8
ROW_SPLIT = 256         # rows separate cleanly here (top y<=205, bottom y>=252)
SLOTS = ['A', 'B', 'C', 'D', 'E', 'F']

# Per-cell tolerances. Column centres have sd 4.2 px, so 10 px is inside
# rendering jitter; 45 px is half the column pitch (144 px) minus a glyph
# half-width -- beyond that the object is closer to a different cell.
POS_GOOD_PX = 10.0
POS_BAD_PX = 45.0
# Centred mask IoU, measured on the split rather than assumed:
#   * the *identical* rendered glyph (input cell vs GT cell): exactly 1.000,
#     n=300, min 1.000 -- the renderer reuses the object verbatim;
#   * a candidate glyph that is positionally and chromatically correct
#     (n=1254): p5 0.850, p25 0.954, p50 0.980 -- so a correct regeneration
#     lands at or above 0.85 in 95% of cases;
#   * two genuinely *different* glyphs (a GT top-row glyph vs the bottom-row
#     glyph of the same frame, n=200): p50 0.448, p95 0.750.
# The ramp therefore spans 0.85 -> 0.60, which is where those two populations
# actually separate. An earlier 0.45 was too lenient: a circle inscribed in a
# square overlaps it by pi/4 = 0.785 and was scoring 0.84 as a glyph match.
IOU_GOOD = 0.85
IOU_BAD = 0.60


def _detect(img: np.ndarray) -> List[Dict]:
    """Objects are found relative to the *detected* background, not by saturation.

    A saturation mask breaks completely when a candidate repaints the scene:
    00009/base renders on black and `chroma_mask` returns 18 noise components;
    00092/base renders on orange and it returns 1, having flagged the background
    itself as foreground. Distance-from-background returns a sane 5 objects in
    both cases. On well-formed frames the two agree exactly -- both give 6
    objects on all 100 GT frames -- and the small connecting arrow glyphs the
    background-relative mask additionally picks up measure 24-26 px^2, an order
    of magnitude below MIN_AREA.
    """
    return C.objects(img, C.clean(C.foreground_mask(img, tol=FG_TOL), open_k=3, close_k=3),
                     min_area=MIN_AREA)


def _grid(objs: List[Dict]) -> Dict[str, Dict]:
    """Assign objects to the six named cells by row then column order."""
    top = sorted([o for o in objs if o['centroid'][1] < ROW_SPLIT],
                 key=lambda o: o['centroid'][0])
    bot = sorted([o for o in objs if o['centroid'][1] >= ROW_SPLIT],
                 key=lambda o: o['centroid'][0])
    cells: Dict[str, Dict] = {}
    for name, o in zip('ABC', top):
        cells[name] = o
    for name, o in zip('DEF', bot):
        cells[name] = o
    return cells


def _nearest(objs: List[Dict], target: Dict) -> Optional[Dict]:
    """Candidate object closest to a GT cell, if any lies within POS_BAD_PX."""
    if not objs:
        return None
    best = min(objs, key=lambda o: np.hypot(o['centroid'][0] - target['centroid'][0],
                                            o['centroid'][1] - target['centroid'][1]))
    d = float(np.hypot(best['centroid'][0] - target['centroid'][0],
                       best['centroid'][1] - target['centroid'][1]))
    return best if d <= POS_BAD_PX else None


def _cell_report(gt_obj: Dict, cd_obj: Optional[Dict]) -> Dict:
    """Everything measured about one cell, so a score is always traceable."""
    if cd_obj is None:
        return {'present': False, 'pos': 0.0, 'iou': 0.0, 'colour_dE': None,
                'colour_ok': False, 'score': 0.0}
    d = float(np.hypot(cd_obj['centroid'][0] - gt_obj['centroid'][0],
                       cd_obj['centroid'][1] - gt_obj['centroid'][1]))
    pos = C.band(d, POS_GOOD_PX, POS_BAD_PX)
    iou = C.centred_iou(gt_obj, cd_obj)
    shape = C.band(1.0 - iou, 1.0 - IOU_GOOD, 1.0 - IOU_BAD)
    de = C.dE(gt_obj['lab'], cd_obj['lab'])
    colour_ok = de <= C.SAME_COLOR_DE
    # Position, glyph identity and colour must ALL hold for a cell to be right;
    # a product rather than a mean, so one hard failure cannot be averaged away.
    score = pos * shape * (1.0 if colour_ok else 0.0)
    return {'present': True, 'dist_px': round(d, 1), 'pos': round(pos, 3),
            'iou': round(iou, 3), 'shape': round(shape, 3),
            'colour_dE': round(de, 1), 'colour_ok': colour_ok,
            'score': float(score)}


def _score(input_img, cand_img, gt_img) -> Tuple[Dict[str, float], Dict]:
    dbg: Dict = {}
    gt_objs, cd_objs = _detect(gt_img), _detect(cand_img)
    in_objs = _detect(input_img)
    gt_cells = _grid(gt_objs)
    dbg['counts'] = {'gt': len(gt_objs), 'cand': len(cd_objs), 'input': len(in_objs)}
    dbg['objs'] = {'gt': gt_objs, 'cand': cd_objs}

    # -- layer 1: global gates ------------------------------------------------
    gate = C.blankness(cand_img, gt_img)
    bg_de = C.dE(C.to_lab(C.background_bgr(gt_img)), C.to_lab(C.background_bgr(cand_img)))
    if bg_de > 40.0:
        gate['ceiling'] = min(gate['ceiling'], 0.15)
    dbg['blankness'] = gate
    dbg['background_dE'] = bg_de

    if len(gt_cells) < 6:
        dbg['error'] = f'GT yielded {len(gt_cells)} cells, expected 6'
        return {k: 0.0 for k in WEIGHTS}, dbg

    # -- layer 2/3: per-cell match against GT ---------------------------------
    pool = list(cd_objs)
    reports: Dict[str, Dict] = {}
    matched: Dict[str, Optional[Dict]] = {}
    for name in SLOTS:                       # greedy, one candidate per cell
        got = _nearest(pool, gt_cells[name])
        if got is not None:
            pool.remove(got)
        matched[name] = got
        reports[name] = _cell_report(gt_cells[name], got)
    dbg['cells'] = reports
    dbg['matched'] = matched

    first_row = float(np.mean([reports[n]['score'] for n in 'ABC']))
    second_row = float(np.mean([reports[n]['score'] for n in 'DEF']))

    # colour_accuracy: E and F must carry B's colour, which is what GT's E and F
    # already encode -- so it is scored directly against GT's cell colours.
    col = []
    for n in 'EF':
        m = matched[n]
        col.append(0.0 if m is None else
                   (1.0 if C.dE(gt_cells[n]['lab'], m['lab']) <= C.SAME_COLOR_DE else 0.0))
    color_accuracy = float(np.mean(col))

    n_cd = len(cd_objs)
    shape_count = C.band(abs(n_cd - 6), 0, 3)
    dbg['shape_count'] = {'candidate': n_cd, 'expected': 6}

    # The original evaluator zeroed everything downstream when the first row
    # fell below 0.5. That principle is kept -- the bottom row is only
    # meaningful as an analogy of an intact top row -- but it must not fire on
    # *ordinary* top-row loss, or the same error is charged twice.
    #
    # A first pass used a continuous `0.25 + 0.75 * first_row`, which docked
    # 00040/s3's bottom row to 0.73 although inspection says it is correct: the
    # candidate simply dropped glyph C, an error `first_row_preservation`
    # already scores at 0.67. The gate now holds at 1.0 down to first_row = 0.5
    # (the original cliff) and ramps to 0 only below that, where the example
    # really has been destroyed and the analogy is unreadable.
    analogy_gate = min(1.0, first_row / 0.5)
    second_row *= analogy_gate
    color_accuracy *= analogy_gate
    dbg['analogy_gate'] = round(analogy_gate, 3)

    scores = {'first_row_preservation': first_row,
              'second_row_completion': second_row,
              'color_accuracy': color_accuracy,
              'shape_count': shape_count}
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
    gt_v = C.draw_objects(g, dbg['objs']['gt'], 'GT cells')
    cd_v = C.draw_objects(c, dbg['objs']['cand'], 'CAND cells')
    if 'cells' in dbg:
        for name in SLOTS:
            o = dbg['matched'].get(name)
            r = dbg['cells'][name]
            gx, gy = (int(v) for v in dbg['objs']['gt'] and _grid(dbg['objs']['gt'])[name]['centroid'])
            col = (0, 150, 0) if r['score'] > 0.75 else (0, 165, 255) if r['score'] > 0.3 else (0, 0, 255)
            cv2.putText(gt_v, name, (gx - 8, gy - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            if o is not None:
                cv2.putText(cd_v, f"{name}:{r['score']:.2f}", (int(o['centroid'][0]) - 26,
                            int(o['centroid'][1]) - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    side = np.hstack([gt_v, np.full((gt_v.shape[0], 3, 3), 90, np.uint8), cd_v])
    paths = {'cells': f'{out_dir}/{tag}_cells.png',
             'cand_mask': f'{out_dir}/{tag}_cand_mask.png'}
    cv2.imwrite(paths['cells'], side)
    cv2.imwrite(paths['cand_mask'],
                C.mask_vis(C.clean(C.foreground_mask(c, tol=FG_TOL), 3, 3)))
    dbg['debug_images'] = paths
    dbg['vis'] = {'cells': side}
    dbg['summary'] = dbg.get('cells', {})
    return scores, dbg


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('input'); ap.add_argument('candidate'); ap.add_argument('ground_truth')
    ap.add_argument('--debug-dir')
    a = ap.parse_args()
    if a.debug_dir:
        s, d = debug(a.input, a.candidate, a.ground_truth, a.debug_dir)
        print(json.dumps({'scores': s, 'overall': overall(s), 'cells': d.get('cells')},
                         indent=2, default=str))
    else:
        s = score(a.input, a.candidate, a.ground_truth)
        print(json.dumps({'scores': s, 'overall': overall(s)}, indent=2))
