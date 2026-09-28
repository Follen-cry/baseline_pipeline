#!/usr/bin/env python
"""Rule-based (no-VLM) metrics, per item and per model.

  --part pica        PICABench Con: official PicaEval_consistency.evaluate_single_item (masked PSNR on
                     non-edited regions, --size 512 as in the README)          (env: internvlu)
  --part magicbrush  L1 / CLIP-I (ViT-B/32) / DINO (vits16) vs GT, primitives from
                     suites/magicbrush/evaluators/eval_magicbrush.py (upstream image_eval.py port)
                                                                               (env: geneval-eval-env)
Writes results/rule/<part>.csv (bench, uid, model, metric, value).
"""
import argparse
import csv
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.abspath(os.path.join(HERE, ".."))
SRC = "/scratch/network/ssd/junlin/ssl_eval/src"
MODELS = ["base", "T0", "T2", "T3", "copy"]  # copy = input-copy reference


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["pica", "magicbrush"])
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(HERE, "manifest.jsonl"))]
    out = []
    if a.part == "pica":
        C = load(f"{SRC}/PICABench/PicaEval_consistency.py", "PicaEval_consistency")
        for r in rows:
            if r["bench"] != "picabench":
                continue
            for m in MODELS:
                pred = os.path.join(EVAL, "outputs", m, r["out_rel"])
                if not os.path.exists(pred):
                    continue
                item = dict(input_path=r["inputs"][0], output_path=pred, edit_area=r["judge"]["edit_area"])
                psnr = C.evaluate_single_item(item, "/", 512)  # abs paths: os.path.join("/", abs) == abs
                out.append(("picabench", r["uid"], m, "psnr_nonedit", psnr))
    else:
        import torch
        from torchvision import transforms
        from scipy import spatial
        import clip
        E = load(os.path.join(EVAL, "suites/magicbrush/evaluators/eval_magicbrush.py"), "eval_magicbrush")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        cm, ct = clip.load("ViT-B/32", dev)
        cm.eval()
        dm = torch.hub.load("facebookresearch/dino:main", "dino_vits16").eval().to(dev)
        dt = transforms.Compose([transforms.Resize(256, interpolation=3), transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        for r in rows:
            if r["bench"] != "magicbrush":
                continue
            gt = r["judge"]["gt"]
            for m in MODELS:
                pred = os.path.join(EVAL, "outputs", m, r["out_rel"])
                if not os.path.exists(pred):
                    continue
                pair = [(pred, gt)]
                out.append(("magicbrush", r["uid"], m, "l1", E.eval_distance(pair, "l1")))
                out.append(("magicbrush", r["uid"], m, "clip_i", E.eval_clip_i(pair, cm, ct, dev, "clip_i")))
                out.append(("magicbrush", r["uid"], m, "dino", E.eval_clip_i(pair, dm, dt, dev, "dino")))
    d = os.path.join(EVAL, "results", "rule")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{a.part}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bench", "uid", "model", "metric", "value"])
        w.writerows(out)
    print(a.part, len(out), "rows")


if __name__ == "__main__":
    main()
