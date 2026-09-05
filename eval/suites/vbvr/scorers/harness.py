"""Calibration / held-out harness for the rewritten rule-based scorers.

Picks a deterministic batch of (sample_id, model_variant) pairs for a task,
runs a scorer over them, writes debug visualisations, and renders one review
sheet per instance so the images can actually be looked at (the test-and-refine
loop requires visual verification, not just numeric output).
"""
import importlib
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import samples  # noqa: E402
import montage  # noqa: E402
import cvlib as C  # noqa: E402

# Deterministic batches. A contiguous block of ids turned out to be a bad
# sampler -- slots 0-9 of rotation_puzzle are all near-perfect generations, so
# the batch contained no failures to calibrate against. Instead both batches
# stride the whole 100-sample split (calibration at offset 0, held-out at
# offset 5) and cycle the model variant, so each batch spans base + all four
# pretrained variants and the full easy/hard range without being hand-picked.
BATCH_SLOTS = {'calib': list(range(0, 100, 10)),
               'holdout': list(range(5, 100, 10))}


def batch(task, which='calib'):
    ids = samples.ids_for(task)
    out = []
    for n, slot in enumerate(BATCH_SLOTS[which]):
        out.append((ids[slot], samples.VARIANTS[n % len(samples.VARIANTS)]))
    return out


def run(task, module_name, which='calib', out_dir=None, sheet_h=200):
    mod = importlib.import_module(module_name)
    out_dir = out_dir or f'debug_out/{task}_{which}'
    os.makedirs(out_dir, exist_ok=True)
    rows, recs = [], []
    for k, (rid, variant) in enumerate(batch(task, which)):
        inp, cand, gt = samples.triple(rid, variant)
        tag = f'{k:02d}_{variant.split("_")[0]}_{rid[-5:]}'
        sub, dbg = mod.debug(inp, cand, gt, out_dir, tag)
        ov = mod.overall(sub)
        recs.append({'slot': k, 'id': rid, 'variant': variant, 'tag': tag,
                     'sub': {a: round(b, 3) for a, b in sub.items()},
                     'overall': round(ov, 3),
                     'notes': _jsonable(dbg.get('summary', {}))})
        extra = [(v, n) for n, v in (dbg.get('vis') or {}).items()]
        line = ' '.join(f'{a[:9]}={b:.2f}' for a, b in sub.items())
        rows.append(montage.strip(
            [inp, cand, gt],
            [f'{tag} IN', f'CAND ov={ov:.2f}', 'GT'], sheet_h, extra=extra))
        rows.append(_textbar(line, rows[-1].shape[1]))
    cv2.imwrite(f'{out_dir}/_sheet.png', montage.grid(rows))
    json.dump(recs, open(f'{out_dir}/_scores.json', 'w'), indent=1)
    return recs


def _textbar(text, width, h=20):
    bar = np.full((h, width, 3), 235, np.uint8)
    cv2.putText(bar, text, (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return bar


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return round(float(o), 4)
    return o


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('task'); ap.add_argument('module')
    ap.add_argument('--which', default='calib')
    a = ap.parse_args()
    r = run(a.task, a.module, a.which)
    for x in r:
        print(f"{x['tag']:28s} ov={x['overall']:.3f}  " +
              ' '.join(f'{k[:9]}={v:.2f}' for k, v in x['sub'].items()))
