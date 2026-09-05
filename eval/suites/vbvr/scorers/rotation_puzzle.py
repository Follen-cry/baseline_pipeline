"""Rule-based scorer for O-44_rotation_puzzle (VBVR).

Scene: a 2x2 grid of white tiles on a near-white background; each tile carries a
single coloured L-shaped ("elbow") pipe. The task is to rotate each elbow by a
multiple of 90 degrees so the four arms join into one closed ring.

Why this was rewritten
----------------------
The original evaluator keyed every sub-criterion off `_detect_blue_pipes`, an
HSV window of [100,100,100]..[130,255,255]. The pipes in this task are rendered
in a per-sample colour (measured across the eval split: orange, yellow, magenta,
cyan, ...). For every non-blue sample the mask came back empty, `path_connection`
hit its `return 0.5` fallback and `rotation_accuracy` its `return 0.5`, so most
samples collapsed onto the same handful of constants -- which is where the
"suspiciously identical 0.55" scores came from. See rotation_puzzle_notes.md.

The rewrite reads the actual structure: the tile grid is recovered from the GT
frame, each tile's elbow is described by *which of its four edge midpoints an
arm reaches* (an exact, discrete 2-of-{N,E,S,W} state), and the candidate is
compared cell-by-cell against GT.

Sub-criterion names and weights are unchanged from the original rubric.
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from . import cvlib as C
except ImportError:  # standalone use
    import cvlib as C

WEIGHTS = {
    'path_connection': 0.40,
    'rotation_accuracy': 0.30,
    'position_preservation': 0.20,
    'alignment_precision': 0.10,
}

# Fallback grid, in the 512-px working frame. Measured identical across every
# input/GT frame in the split; only used if tile detection on GT fails.
FALLBACK_CELLS = {'TL': (138, 138, 110, 110), 'TR': (263, 138, 110, 110),
                  'BL': (138, 263, 110, 110), 'BR': (263, 263, 110, 110)}

# An arm is called present by how close the pipe actually gets to the tile edge,
# in pixels -- not by a coverage fraction. Coverage conflates a thin stroke with
# a short arm; reach is the physical quantity the connectivity question is
# actually about, and it separates cleanly. Measured over 2400 candidate arms
# (5 variants x 30 samples x 4 cells x 4 directions): 1197 land at 0-6 px, 1183
# at 40-60 px, and only 19 (0.8%) fall anywhere in between. GT arms reach within
# 2-4 px; a GT non-arm sits 51 px away (the elbow corner). Tiles are 110 px with
# a 15 px inter-tile gutter and a ~6 px stroke, so an arm must come within about
# one stroke width of its own edge for the two ends to meet across the gutter.
ARM_GAP_PX = 8.0  # <= this many px from the tile edge counts as an arm
ARM_FRAC = 0.02   # noise guard: the arm band must also hold some pixels
ARM_SPAN = 0.44   # width of the midpoint window as a fraction of the tile
ARM_DEPTH = 0.20  # depth of the window inward from the tile edge
OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
# Internal junctions of the 2x2 grid: (cell_a, dir_a) meets (cell_b, dir_b)
JUNCTIONS = [('TL', 'E', 'TR', 'W'), ('BL', 'E', 'BR', 'W'),
             ('TL', 'S', 'BL', 'N'), ('TR', 'S', 'BR', 'N')]


# ------------------------------------------------------------------ detect ---

def detect_cells(img: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
    """Recover the 2x2 tile grid: bright, unsaturated squares of similar size."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = C.background_bgr(img)
    bright = int(max(gray[gray > 0].mean(), bg.mean()) )
    mask = ((gray >= 253) & (C.chroma_mask(img) == 0)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, n):
        w, h, a = st[i, 2], st[i, 3], st[i, 4]
        if a < 4000 or w < 60 or h < 60:
            continue
        if abs(w - h) > 0.30 * max(w, h):      # tiles are square
            continue
        if a < 0.70 * w * h:                    # and solid
            continue
        boxes.append((st[i, 0], st[i, 1], w, h, cen[i]))
    if len(boxes) != 4:
        return dict(FALLBACK_CELLS)
    cx = np.mean([b[4][0] for b in boxes])
    cy = np.mean([b[4][1] for b in boxes])
    cells = {}
    for x, y, w, h, c in boxes:
        key = ('T' if c[1] < cy else 'B') + ('L' if c[0] < cx else 'R')
        cells[key] = (int(x), int(y), int(w), int(h))
    if set(cells) != {'TL', 'TR', 'BL', 'BR'}:
        return dict(FALLBACK_CELLS)
    return cells


def pipe_mask(img: np.ndarray) -> np.ndarray:
    return C.clean(C.chroma_mask(img), open_k=0, close_k=3)


def arm_windows(cell: Tuple[int, int, int, int]) -> Dict[str, Tuple[int, int, int, int]]:
    x, y, w, h = cell
    sw, sh = int(ARM_SPAN * w), int(ARM_SPAN * h)
    dw, dh = int(ARM_DEPTH * w), int(ARM_DEPTH * h)
    ox, oy = x + (w - sw) // 2, y + (h - sh) // 2
    return {'N': (ox, y, sw, dh), 'S': (ox, y + h - dh, sw, dh),
            'W': (x, oy, dw, sh), 'E': (x + w - dw, oy, dw, sh)}


def arm_coverage(mask: np.ndarray, cell) -> Dict[str, float]:
    out = {}
    for d, (rx, ry, rw, rh) in arm_windows(cell).items():
        sub = mask[max(0, ry):ry + rh, max(0, rx):rx + rw]
        out[d] = float((sub > 0).mean()) if sub.size else 0.0
    return out


def arm_gap(mask: np.ndarray, cell: Tuple[int, int, int, int], d: str) -> float:
    """Distance in px from the tile's `d` edge to the nearest pipe pixel in the
    central band of that edge. Large (~half a tile) when no arm points that way."""
    x, y, w, h = cell
    if d in 'NS':
        band = mask[y:y + h, x + int(0.28 * w):x + int(0.72 * w)]
        nz = np.nonzero((band > 0).sum(axis=1))[0]
        if not len(nz):
            return float(h)
        return float(nz.min()) if d == 'N' else float(h - 1 - nz.max())
    band = mask[y + int(0.28 * h):y + int(0.72 * h), x:x + w]
    nz = np.nonzero((band > 0).sum(axis=0))[0]
    if not len(nz):
        return float(w)
    return float(nz.min()) if d == 'W' else float(w - 1 - nz.max())


def arm_set(cov: Dict[str, float], gaps: Dict[str, float]) -> frozenset:
    return frozenset(d for d in cov
                     if gaps[d] <= ARM_GAP_PX and cov[d] >= ARM_FRAC)


def arm_profile(mask: np.ndarray, cell, d: str) -> Optional[float]:
    """Centre-of-mass of the arm across the boundary it crosses.

    For a horizontal neighbour (E/W) this is a y coordinate, for a vertical one
    (N/S) an x coordinate -- so the two sides of a junction are directly
    comparable and their difference is the physical misalignment in px.
    """
    rx, ry, rw, rh = arm_windows(cell)[d]
    sub = mask[max(0, ry):ry + rh, max(0, rx):rx + rw]
    ys, xs = np.nonzero(sub > 0)
    if len(xs) == 0:
        return None
    return float(ry + ys.mean()) if d in 'EW' else float(rx + xs.mean())


def cell_state(img: np.ndarray, cells) -> Dict[str, Dict]:
    m = pipe_mask(img)
    st = {}
    for k, cell in cells.items():
        cov = arm_coverage(m, cell)
        gaps = {d: arm_gap(m, cell, d) for d in 'NESW'}
        x, y, w, h = cell
        sub = m[y:y + h, x:x + w]
        st[k] = {'coverage': cov, 'gaps': gaps, 'arms': arm_set(cov, gaps),
                 'pipe_px': int((sub > 0).sum()) if sub.size else 0}
    return st


# ------------------------------------------------------------------- score ---

def _score(input_img, cand_img, gt_img) -> Tuple[Dict[str, float], Dict]:
    dbg: Dict = {}
    cells = detect_cells(gt_img)
    dbg['cells'] = cells
    cand_cells = detect_cells(cand_img)
    dbg['cand_cells'] = cand_cells

    gt_m, cand_m = pipe_mask(gt_img), pipe_mask(cand_img)
    dbg['gt_mask'], dbg['cand_mask'] = gt_m, cand_m

    # -- layer 1: cheap global gates ------------------------------------------
    gate = C.blankness(cand_img, gt_img)
    dbg['blankness'] = gate

    # scene gate: a candidate that repaints the background has abandoned the
    # scene entirely (observed: sample 00058/base renders the ring on black).
    # Foreground *ratio* alone cannot see this, so compare backgrounds directly.
    gt_bg, cand_bg = C.background_bgr(gt_img), C.background_bgr(cand_img)
    bg_de = C.dE(C.to_lab(gt_bg), C.to_lab(cand_bg))
    dbg['background'] = {'gt_bgr': gt_bg.tolist(), 'cand_bgr': cand_bg.tolist(), 'dE': bg_de}
    if bg_de > 40.0:
        gate['ceiling'] = min(gate['ceiling'], 0.15)

    gt_pipe_px = int((gt_m > 0).sum())
    cand_pipe_px = int((cand_m > 0).sum())
    pipe_ratio = cand_pipe_px / max(1, gt_pipe_px)
    dbg['pipe_ratio'] = pipe_ratio
    if pipe_ratio < 0.15:
        gate['ceiling'] = min(gate['ceiling'], 0.1)
    elif pipe_ratio < 0.35:
        gate['ceiling'] = min(gate['ceiling'], 0.5)

    # -- layer 2: colour gate --------------------------------------------------
    gt_col = C.dominant_bgr(gt_img, gt_m)
    cand_col = C.dominant_bgr(cand_img, cand_m)
    col_de = C.dE(C.to_lab(gt_col), C.to_lab(cand_col)) if cand_pipe_px else 999.0
    dbg['colour'] = {'gt_bgr': gt_col.tolist(), 'cand_bgr': cand_col.tolist(), 'dE': col_de}
    colour_ok = col_de <= C.SAME_COLOR_DE
    colour_mult = 1.0 if colour_ok else 0.5

    # -- layer 3: per-cell structure ------------------------------------------
    gt_state = cell_state(gt_img, cells)
    cand_state = cell_state(cand_img, cells)
    in_state = cell_state(input_img, cells)
    dbg['gt_state'] = {k: sorted(v['arms']) for k, v in gt_state.items()}
    dbg['cand_state'] = {k: sorted(v['arms']) for k, v in cand_state.items()}
    dbg['cand_coverage'] = {k: {d: round(x, 3) for d, x in v['coverage'].items()}
                            for k, v in cand_state.items()}
    dbg['cand_gaps'] = {k: {d: round(x, 1) for d, x in v['gaps'].items()}
                        for k, v in cand_state.items()}

    # rotation_accuracy: exact arm-set identity per cell, against GT
    hits = []
    for k in cells:
        g, c = gt_state[k]['arms'], cand_state[k]['arms']
        if g == c:
            hits.append(1.0)
        elif g and c and (g & c):
            hits.append(0.5 * len(g & c) / len(g | c))   # partial: shares an arm
        else:
            hits.append(0.0)
    rotation_accuracy = float(np.mean(hits))
    dbg['per_cell_rotation'] = dict(zip(cells, hits))

    # path_connection: every internal junction GT requires must be joined on both
    # sides in the candidate; dangling arms that point at the outer border are
    # subtracted, since a ring cannot have loose ends.
    req = [(a, da, b, db) for a, da, b, db in JUNCTIONS
           if da in gt_state[a]['arms'] and db in gt_state[b]['arms']]
    joined = [1.0 for a, da, b, db in req
              if da in cand_state[a]['arms'] and db in cand_state[b]['arms']]
    junction_score = len(joined) / len(req) if req else 0.0

    outward = {'TL': ('N', 'W'), 'TR': ('N', 'E'), 'BL': ('S', 'W'), 'BR': ('S', 'E')}
    dangling = sum(1 for k in cells for d in outward[k] if d in cand_state[k]['arms'])
    dbg['junctions'] = {'required': len(req), 'joined': len(joined), 'dangling': dangling}
    path_connection = max(0.0, junction_score - 0.25 * dangling)

    # position_preservation: each input cell still holds a tile and a pipe
    occupied = []
    for k in cells:
        has_tile = k in cand_cells
        had_pipe = in_state[k]['pipe_px'] > 40
        has_pipe = cand_state[k]['pipe_px'] > 40
        # The criterion is about *tiles* holding their grid cell, so a tile that
        # survived with its pipe erased still earns partial credit here -- the
        # global ceiling, not this sub-score, is what floors a blank candidate.
        if has_pipe and has_tile:
            occupied.append(1.0)
        elif has_pipe:
            occupied.append(0.6)
        elif has_tile:
            occupied.append(0.5 if had_pipe else 1.0)
        else:
            occupied.append(0.0)
    position_preservation = float(np.mean(occupied))
    dbg['per_cell_position'] = dict(zip(cells, occupied))

    # alignment_precision: physical offset of the two arms meeting at a junction
    offs = []
    for a, da, b, db in req:
        pa = arm_profile(cand_m, cells[a], da)
        pb = arm_profile(cand_m, cells[b], db)
        if pa is None or pb is None:
            offs.append(None)
            continue
        offs.append(abs(pa - pb))
    good = [o for o in offs if o is not None]
    dbg['alignment_offsets'] = offs
    if not good:
        alignment_precision = 0.0
    else:
        per = [C.band(o, 3.0, 14.0) for o in good]
        # junctions where an arm is missing entirely score 0, not "skipped"
        per += [0.0] * (len(offs) - len(good))
        alignment_precision = float(np.mean(per))

    scores = {
        'path_connection': path_connection,
        'rotation_accuracy': rotation_accuracy,
        'position_preservation': position_preservation,
        'alignment_precision': alignment_precision,
    }
    ceiling = gate['ceiling']
    for k in ('path_connection', 'rotation_accuracy', 'alignment_precision'):
        scores[k] *= colour_mult
    scores = {k: float(np.clip(v, 0.0, 1.0) * ceiling) for k, v in scores.items()}
    dbg['colour_mult'] = colour_mult
    dbg['ceiling'] = ceiling
    return scores, dbg


def score(input_img, candidate_img, ground_truth_img) -> Dict[str, float]:
    """Score one (input, candidate, ground_truth) triple.

    Arguments are BGR arrays or paths. Returns {sub_criterion: 0..1}.
    """
    i, c, g = _coerce(input_img), _coerce(candidate_img), _coerce(ground_truth_img)
    if c is None or g is None:
        return {k: 0.0 for k in WEIGHTS}
    if i is None:
        i = g
    return _score(i, c, g)[0]


def overall(sub: Dict[str, float]) -> float:
    return float(sum(sub[k] * WEIGHTS[k] for k in WEIGHTS))


def _coerce(x, size: int = C.WORK):
    if x is None:
        return None
    if isinstance(x, str):
        return C.load(x, size)
    return C.resize_to(x, size)


def debug(input_img, candidate_img, ground_truth_img, out_dir: str = '.',
          tag: str = 'sample') -> Tuple[Dict[str, float], Dict]:
    """Like score(), but also writes intermediate visualisations as PNGs."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    i, c, g = _coerce(input_img), _coerce(candidate_img), _coerce(ground_truth_img)
    if i is None:
        i = g
    scores, dbg = _score(i, c, g)

    def annotate(img, cells, state, title):
        vis = img.copy()
        for k, (x, y, w, h) in cells.items():
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 140, 0), 1)
            for d, (rx, ry, rw, rh) in arm_windows((x, y, w, h)).items():
                on = d in state[k]['arms']
                cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh),
                              (0, 0, 255) if on else (170, 170, 170), 1)
            cv2.putText(vis, k + ':' + ''.join(sorted(state[k]['arms'])), (x + 3, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, title, (4, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return vis

    cells = dbg['cells']
    gt_v = annotate(g, cells, cell_state(g, cells), 'GT arms')
    cd_v = annotate(c, cells, cell_state(c, cells), 'CAND arms')
    paths = {
        'gt_arms': f'{out_dir}/{tag}_gt_arms.png',
        'cand_arms': f'{out_dir}/{tag}_cand_arms.png',
        'gt_mask': f'{out_dir}/{tag}_gt_mask.png',
        'cand_mask': f'{out_dir}/{tag}_cand_mask.png',
    }
    cv2.imwrite(paths['gt_arms'], gt_v)
    cv2.imwrite(paths['cand_arms'], cd_v)
    cv2.imwrite(paths['gt_mask'], C.mask_vis(dbg['gt_mask']))
    cv2.imwrite(paths['cand_mask'], C.mask_vis(dbg['cand_mask']))
    dbg['debug_images'] = paths
    dbg['vis'] = {'gt': gt_v, 'cand': cd_v}
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
                          'gt_state': d['gt_state'], 'cand_state': d['cand_state'],
                          'junctions': d['junctions'], 'colour_dE': d['colour']['dE']}, indent=2))
    else:
        s = score(a.input, a.candidate, a.ground_truth)
        print(json.dumps({'scores': s, 'overall': overall(s)}, indent=2))
