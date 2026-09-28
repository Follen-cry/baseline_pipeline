#!/usr/bin/env python
"""Control check: is the train -> save -> processor-replace -> LoRA-merge path lossless?

Compares checkpoints tensor by tensor against the original base snapshot, per pipeline component
(vlm/, generation_decoder/, vae/): keys missing/extra, dtype changes, and relative change
||w - w0|| / ||w0|| (per component, and the largest per-tensor values).
  python weight_diff.py <ckpt_dir> [<ckpt_dir> ...]   -> results/diagnosis/weight_diff.csv (appends)
(env: internvlu)
"""
import csv
import glob
import os
import sys

import torch
from safetensors import safe_open

BASE = "/scratch/network/ssd2/junlin/models/InternVL-U"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "diagnosis", "weight_diff.csv")


def tensors(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                out[k] = (f, k)
    return out


def load(ref):
    with safe_open(ref[0], "pt") as h:
        return h.get_tensor(ref[1])


def compare(ckpt):
    rows = []
    for comp in ("vlm", "generation_decoder", "vae"):
        a, b = tensors(os.path.join(BASE, comp)), tensors(os.path.join(ckpt, comp))
        common = sorted(set(a) & set(b))
        num = den = 0.0
        dtype_changes, per = 0, []
        for k in common:
            x, y = load(a[k]), load(b[k])
            if x.dtype != y.dtype:
                dtype_changes += 1
            if x.shape != y.shape:
                per.append((k, float("inf")))
                continue
            x, y = x.float(), y.float()
            d = (y - x).norm().item() ** 2
            n = x.norm().item() ** 2
            num += d
            den += n
            per.append((k, (d ** 0.5) / max(n ** 0.5, 1e-12)))
        per.sort(key=lambda t: -t[1])
        rows.append(dict(ckpt=os.path.basename(ckpt.rstrip("/")), component=comp, n_base=len(a), n_ckpt=len(b),
                         missing=len(set(a) - set(b)), extra=len(set(b) - set(a)), dtype_changes=dtype_changes,
                         rel_change=(num ** 0.5) / max(den ** 0.5, 1e-12),
                         n_tensors_changed=sum(1 for _, v in per if v > 0),
                         top_changed="; ".join(f"{k}={v:.2e}" for k, v in per[:3]),
                         example_missing="; ".join(sorted(set(a) - set(b))[:3]),
                         example_extra="; ".join(sorted(set(b) - set(a))[:3])))
    return rows


def main():
    rows = [r for c in sys.argv[1:] for r in compare(c)]
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in r.items() if k not in ("example_missing", "example_extra") or v})


if __name__ == "__main__":
    main()
