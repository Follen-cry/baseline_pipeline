"""Shared deterministic CV primitives for the VBVR rule-based scorers.

OpenCV + numpy only. No learned models, no network calls. Every routine here is
a pure function of its pixel input so that a score can always be traced back to
a measured quantity (see each task's <task>_notes.md).

Design notes that apply to all four task scorers:

* Candidates are generated at 512x512 while input/GT frames are 1024x1024, so
  everything is normalised to WORK (512) first and all pixel thresholds in the
  task scorers are expressed in that 512-px frame.
* Background is estimated from the image border rather than assumed white --
  the four tasks use four different backgrounds (255/250/240 grey-white) and
  generated frames carry a faint colour cast, which a hardcoded constant would
  misread as foreground.
* "Same colour" is CIE76 dE in CIELAB, not a hand-written HSV box. The old
  scorers keyed off fixed HSV windows (`lower_blue = [100,100,100]` etc.), which
  silently produced an empty mask -- and a constant fallback score -- whenever
  the sample happened to use a hue outside the window.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

WORK = 512  # common working resolution, px


# ---------------------------------------------------------------- loading ---

def load(path: str, size: int = WORK) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return resize_to(img, size)


def resize_to(img: np.ndarray, size: int = WORK) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


# ------------------------------------------------------------ background ---

def background_bgr(img: np.ndarray, band: int = 12) -> np.ndarray:
    """Modal border colour. Robust to a foreground object touching one edge."""
    b = band
    px = np.concatenate([
        img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
        img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3),
    ])
    # median per channel is enough: >90% of the border is background in every task
    return np.median(px, axis=0)


def foreground_mask(img: np.ndarray, tol: float = 18.0,
                    bg: Optional[np.ndarray] = None) -> np.ndarray:
    """Anything more than `tol` (L2 in BGR) away from the background colour.

    tol=18 chosen from measured background jitter in generated frames: the
    per-pixel spread of a flat background region is <8 in every sampled
    candidate, while the faintest real foreground (pale tile squares on the
    rotation_puzzle background) sits at ~25. See rotation_puzzle_notes.md.
    """
    if bg is None:
        bg = background_bgr(img)
    d = np.linalg.norm(img.astype(np.float32) - bg.astype(np.float32), axis=2)
    return (d > tol).astype(np.uint8) * 255


def chroma_mask(img: np.ndarray, sat: int = 60, min_dist: float = 40.0) -> np.ndarray:
    """Saturated (genuinely coloured) pixels -- excludes white/grey/black.

    Uses HSV saturation AND a max-min channel spread so that dark, low-S but
    clearly coloured pixels (e.g. the near-black purple in
    2d_geometric_transformation) are still kept.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    f = img.astype(np.float32)
    spread = f.max(axis=2) - f.min(axis=2)
    m = (hsv[:, :, 1] >= sat) | (spread >= min_dist)
    # a colour must also not be near-white
    m &= (f.min(axis=2) < 235) | (spread >= min_dist)
    return m.astype(np.uint8) * 255


def clean(mask: np.ndarray, open_k: int = 3, close_k: int = 3) -> np.ndarray:
    out = mask
    if close_k:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if open_k:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    return out


# ----------------------------------------------------------------- colour ---

def to_lab(bgr: Sequence[float]) -> np.ndarray:
    a = np.array(bgr, dtype=np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(a, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


L_WEIGHT = 0.5


def dE(lab_a: np.ndarray, lab_b: np.ndarray, l_weight: float = L_WEIGHT) -> float:
    """CIELAB colour difference with lightness down-weighted.

    Plain CIE76 was too brittle here. Generated frames reproduce an object's
    *hue* faithfully but shift its lightness: the unmoved blue star marker in
    00085/s2 comes back at Lab (90,179,48) against the input's (130,172,51) --
    a 40-unit L shift with a and b within 7 -- which is plainly the same marker
    but fails a 25-dE gate on lightness alone.

    Weighting L by 0.5 measured over the whole split:

      * objects known to be identical (same rendered object in input and GT):
        dE 0.0 at p95
      * markers verified unmoved (candidate marker within 6 px of an input
        star, n=72): p50 6.4, p95 19.7  -- was p95 34.3 unweighted
      * genuinely different palette colours in the same frame: **minimum 39.4**
        -- was 48.7 unweighted

    So SAME_COLOR_DE = 25 sits in a gap roughly 20 units wide on both sides.
    """
    d = np.asarray(lab_a, np.float32) - np.asarray(lab_b, np.float32)
    return float(np.sqrt((l_weight * d[0]) ** 2 + d[1] ** 2 + d[2] ** 2))


SAME_COLOR_DE = 25.0


def mask_mean_bgr(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask > 0
    if not m.any():
        return np.zeros(3, np.float32)
    return img[m].astype(np.float32).mean(axis=0)


def dominant_bgr(img: np.ndarray, mask: np.ndarray, bins: int = 16) -> np.ndarray:
    """Modal colour under `mask`, quantised to `bins` per channel.

    Preferred over the plain mean whenever a region may contain two colours
    (anti-aliased edges, or two objects merged by a morphology close): the mean
    of magenta+green is a grey that matches neither.
    """
    m = mask > 0
    if not m.any():
        return np.zeros(3, np.float32)
    px = img[m].astype(np.int32)
    q = px // (256 // bins)
    key = (q[:, 0] * bins + q[:, 1]) * bins + q[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    best = vals[counts.argmax()]
    sel = key == best
    return px[sel].astype(np.float32).mean(axis=0)


def core_bgr(img: np.ndarray, mask: np.ndarray, pct: float = 70.0) -> np.ndarray:
    """Colour of a region's *core*: the mean of its most saturated pixels.

    `dominant_bgr` is unreliable for small objects. A star marker here is ~137
    px^2 (about 13 px across), so most of its pixels are anti-aliased edge that
    blends toward the background, and the modal colour comes back visibly
    desaturated -- measured dE 33.8 against the identical, unmoved marker in the
    input frame, enough to fail a 25-dE colour gate. Selecting the top
    (100-pct)% of pixels by HSV saturation keeps the interior and drops the
    blend, which brought that same pair to dE < 10.
    """
    m = mask > 0
    if not m.any():
        return np.zeros(3, np.float32)
    px = img[m]
    sat = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)[:, 0, 1].astype(np.float32)
    thr = np.percentile(sat, pct)
    sel = sat >= thr
    if sel.sum() < 3:
        sel = np.ones(len(px), bool)
    return px[sel].astype(np.float32).mean(axis=0)


# ---------------------------------------------------------------- objects ---

def shape_type(cnt: np.ndarray) -> str:
    """Coarse shape class from contour geometry alone.

    Polygon vertex count is tested BEFORE circularity, because the two overlap:
    a perfect square has circularity 4*pi*A/P^2 = pi/4 = 0.785, and rasterising
    plus anti-aliasing pushes that to 0.80-0.81. An earlier version tested
    `circ > 0.80` first and so labelled every square a circle -- which made the
    fidelity sub-criterion score a square-for-circle substitution as correct.

    Squares and elongated rectangles are both returned as 'quad'. The
    square/rect split was label noise here (the same rendered object came back
    'rect' in one frame and 'square' in another, creating fidelity mismatches
    out of nothing); callers that care about elongation should compare
    `rect_wh` / aspect ratio numerically instead.
    """
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri <= 0 or area <= 0:
        return 'unknown'
    circ = 4 * np.pi * area / (peri * peri)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    n = len(approx)
    hull = cv2.convexHull(cnt)
    ha = cv2.contourArea(hull)
    solidity = area / ha if ha > 0 else 0.0

    if solidity < 0.75:
        # concave: a star marker or a cross/plus glyph
        return 'star' if n >= 8 else 'cross'
    if n == 3:
        return 'triangle'
    if n == 4:
        return 'quad'
    if circ >= 0.85:
        return 'circle'
    if n <= 6:
        return 'poly'
    return 'circle' if circ >= 0.80 else 'poly'


def objects(img: np.ndarray, mask: np.ndarray, min_area: float = 40.0) -> List[Dict]:
    """Connected components of `mask` described geometrically and by colour."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (lab == i).astype(np.uint8) * 255
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        cnt = max(cs, key=cv2.contourArea)
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        # Small regions are dominated by anti-aliased edge pixels, so their
        # colour is read from the saturated core instead of the modal bin.
        bgr = core_bgr(img, comp) if area < 600 else dominant_bgr(img, comp)
        (_, _), (rw, rh), ang = cv2.minAreaRect(cnt)
        out.append({
            'centroid': (float(cent[i][0]), float(cent[i][1])),
            'area': area,
            'bbox': (int(x), int(y), int(w), int(h)),
            'contour': cnt,
            'mask': comp,
            'bgr': bgr,
            'lab': to_lab(bgr),
            'shape': shape_type(cnt),
            'rect_wh': (float(rw), float(rh)),
            'angle': float(ang),
            'fill': area / max(1.0, float(w * h)),
        })
    out.sort(key=lambda o: -o['area'])
    return out


def principal_angle(cnt: np.ndarray) -> float:
    """Orientation of the contour's major axis, degrees in [0,180)."""
    pts = cnt.reshape(-1, 2).astype(np.float32)
    pts = pts - pts.mean(axis=0)
    if len(pts) < 3:
        return 0.0
    cov = np.cov(pts.T)
    w, v = np.linalg.eigh(cov)
    vec = v[:, int(np.argmax(w))]
    return float(np.degrees(np.arctan2(vec[1], vec[0])) % 180.0)


def angle_diff180(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def centred_iou(a: Dict, b: Dict) -> float:
    """IoU of two object masks after translating both to a common centroid.

    Compares glyph *identity and size* without position, which a coarse shape
    class ('quad', 'star', 'poly') cannot do -- these tasks render crosses,
    T-glyphs, arrows and bars that all collapse into the same few classes.
    """
    ma, mb = a['mask'] > 0, b['mask'] > 0
    ha, wa = ma.shape
    ca = np.array(a['centroid'])
    cb = np.array(b['centroid'])
    shift = np.round(ca - cb).astype(int)
    Mb = np.zeros_like(mb)
    y0, x0 = shift[1], shift[0]
    ys, xs = np.nonzero(mb)
    ys, xs = ys + y0, xs + x0
    keep = (ys >= 0) & (ys < ha) & (xs >= 0) & (xs < wa)
    Mb[ys[keep], xs[keep]] = True
    inter = float((ma & Mb).sum())
    union = float((ma | Mb).sum())
    return inter / union if union > 0 else 0.0


def best_rotation(target: Dict, source: Dict, coarse: int = 4,
                  fine: int = 1) -> Tuple[float, float]:
    """Rotation (deg) that best maps `source`'s mask onto `target`'s, and its IoU.

    Both masks are translated to a common centroid first, so this measures pure
    orientation, independent of position. Searched exhaustively over 360 deg --
    coarse pass then a fine refinement -- rather than compared via a single
    principal-axis or minAreaRect angle, because those are ambiguous for a
    symmetric shape and unstable for a near-square one, and because an
    exhaustive best fit reports a number that can be stated directly: "best
    alignment at 12 deg".
    """
    ma = target['mask'] > 0
    mb = source['mask'] > 0
    if not ma.any() or not mb.any():
        return 0.0, 0.0
    h, w = ma.shape
    ca, cb = target['centroid'], source['centroid']
    src = mb.astype(np.uint8)

    def iou_at(ang: float) -> float:
        M = cv2.getRotationMatrix2D((float(cb[0]), float(cb[1])), float(ang), 1.0)
        M[0, 2] += ca[0] - cb[0]
        M[1, 2] += ca[1] - cb[1]
        r = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_NEAREST) > 0
        u = (ma | r).sum()
        return float((ma & r).sum()) / u if u else 0.0

    best_a, best_i = 0.0, -1.0
    for ang in range(-180, 180, coarse):
        v = iou_at(ang)
        if v > best_i:
            best_a, best_i = float(ang), v
    for ang in np.arange(best_a - coarse, best_a + coarse + 1e-9, fine):
        v = iou_at(ang)
        if v > best_i:
            best_a, best_i = float(ang), v
    # report the signed rotation in (-180, 180]
    a = (best_a + 180.0) % 360.0 - 180.0
    return a, best_i


# --------------------------------------------------------------- matching ---

def match(gt_objs: List[Dict], cand_objs: List[Dict], max_de: float = SAME_COLOR_DE,
          pos_weight: float = 1.0) -> List[Tuple[int, Optional[int], float]]:
    """Greedy one-to-one GT->candidate assignment on colour then proximity.

    Returns (gt_index, cand_index_or_None, centroid_distance_px). Colour is a
    hard gate (dE <= max_de): an object of the wrong colour is never counted as
    a match, so a candidate cannot earn positional credit by putting *some*
    object in the right place.
    """
    pairs = []
    for gi, g in enumerate(gt_objs):
        for ci, c in enumerate(cand_objs):
            de = dE(g['lab'], c['lab'])
            if de > max_de:
                continue
            d = float(np.hypot(g['centroid'][0] - c['centroid'][0],
                               g['centroid'][1] - c['centroid'][1]))
            pairs.append((de + pos_weight * d, gi, ci, d))
    pairs.sort()
    used_g, used_c, res = set(), set(), {}
    for _, gi, ci, d in pairs:
        if gi in used_g or ci in used_c:
            continue
        used_g.add(gi)
        used_c.add(ci)
        res[gi] = (ci, d)
    return [(gi, *res.get(gi, (None, float('inf')))) for gi in range(len(gt_objs))]


# ------------------------------------------------------------- scoring aid ---

def band(value: float, good: float, bad: float) -> float:
    """Linear 1->0 ramp: <=good scores 1.0, >=bad scores 0.0."""
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return float(1.0 - (value - good) / (bad - good))


def blankness(img: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    """Cheap global gate (design principle 1): how much foreground exists at all.

    `ratio` is candidate foreground pixels / GT foreground pixels. A near-blank
    candidate must never score well no matter what the downstream per-object
    matching does, so task scorers multiply their final sub-scores by `ceiling`.
    """
    fg_c = float((foreground_mask(img) > 0).sum())
    fg_g = float((foreground_mask(ref) > 0).sum())
    ratio = fg_c / max(1.0, fg_g)
    # Boundaries are about *near-blankness*, not size. An earlier 0.45 cut was
    # arbitrary and too aggressive: a correctly placed object rendered at 0.6x
    # linear scale has 0.36 of GT's foreground and was being capped at 0.6 as
    # though the frame were nearly empty -- a size error, which the per-object
    # fidelity criteria already measure. At 0.30 the candidate is missing 70% of
    # GT's ink, which is a genuine "there is almost nothing here" signal.
    if ratio < 0.05:
        ceil = 0.0
    elif ratio < 0.15:
        ceil = 0.25
    elif ratio < 0.30:
        ceil = 0.6
    else:
        ceil = 1.0
    return {'cand_fg': fg_c, 'gt_fg': fg_g, 'ratio': ratio, 'ceiling': ceil}


# ------------------------------------------------------------------ debug ---

_PALETTE = [(0, 0, 255), (0, 200, 0), (255, 100, 0), (200, 0, 200),
            (0, 200, 200), (120, 120, 255), (0, 128, 255)]


def draw_objects(img: np.ndarray, objs: List[Dict], label: str = '') -> np.ndarray:
    vis = img.copy()
    for i, o in enumerate(objs):
        col = _PALETTE[i % len(_PALETTE)]
        cv2.drawContours(vis, [o['contour']], -1, col, 2)
        cx, cy = int(o['centroid'][0]), int(o['centroid'][1])
        cv2.drawMarker(vis, (cx, cy), col, cv2.MARKER_CROSS, 11, 2)
        cv2.putText(vis, f"{i}:{o['shape'][:4]}", (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    if label:
        cv2.putText(vis, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def draw_matches(gt: np.ndarray, cand: np.ndarray, gt_objs, cand_objs, pairs) -> np.ndarray:
    """Side-by-side GT|candidate with a line joining each matched centroid pair."""
    h = max(gt.shape[0], cand.shape[0])
    canvas = np.full((h, gt.shape[1] + cand.shape[1], 3), 255, np.uint8)
    canvas[:gt.shape[0], :gt.shape[1]] = gt
    canvas[:cand.shape[0], gt.shape[1]:] = cand
    off = gt.shape[1]
    for gi, ci, d in pairs:
        g = gt_objs[gi]
        p1 = (int(g['centroid'][0]), int(g['centroid'][1]))
        col = (0, 160, 0) if ci is not None else (0, 0, 255)
        cv2.drawMarker(canvas, p1, col, cv2.MARKER_CROSS, 13, 2)
        if ci is None:
            cv2.putText(canvas, 'UNMATCHED', (p1[0] + 6, p1[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, col, 1, cv2.LINE_AA)
            continue
        c = cand_objs[ci]
        p2 = (int(c['centroid'][0]) + off, int(c['centroid'][1]))
        cv2.line(canvas, p1, p2, col, 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, p2, col, cv2.MARKER_CROSS, 13, 2)
        cv2.putText(canvas, f'{d:.0f}px', (p2[0] + 6, p2[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    return canvas


def mask_vis(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((mask > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
