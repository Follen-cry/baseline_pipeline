"""Build a labeled [input | candidate | ground_truth] strip for visual review."""
import cv2
import numpy as np


def _lab(img, text, h=22):
    bar = np.full((h, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, text, (4, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def strip(paths, labels, height=256, extra=None):
    cells = []
    for p, l in zip(paths, labels):
        im = cv2.imread(p) if isinstance(p, str) else p
        if im is None:
            im = np.zeros((height, height, 3), np.uint8)
        s = height / im.shape[0]
        im = cv2.resize(im, (int(im.shape[1] * s), height), interpolation=cv2.INTER_AREA)
        cells.append(_lab(im, l))
    if extra:
        for im, l in extra:
            s = height / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1] * s), height), interpolation=cv2.INTER_NEAREST)
            cells.append(_lab(im, l))
    H = max(c.shape[0] for c in cells)
    cells = [np.vstack([c, np.full((H - c.shape[0], c.shape[1], 3), 40, np.uint8)]) for c in cells]
    sep = np.full((H, 3, 3), 90, np.uint8)
    out = []
    for c in cells:
        out += [c, sep]
    return np.hstack(out[:-1])


def grid(rows):
    W = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], W - r.shape[1], 3), 40, np.uint8)]) for r in rows]
    return np.vstack(rows)
