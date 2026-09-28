#!/usr/bin/env python
"""Cause 3 (training-induced error): what kind of failures do T0/T2/T3 make where they lose to base?

Loss set: for each trained model, its LOSS_K items with the most negative normalised primary-metric
delta vs base (results/examples.csv); union over T0/T2/T3. Every model's output (base too) on the loss
set is tagged, so the modes are comparable across models.

Automatic tags (all 1,210 items x 4 models, CPU):
  no_edit      mean |out - in| < 0.03 (0-1 RGB; same threshold as cause1_rescore.py)
  global_drift |mean colour(out) - mean colour(in)| > 0.06 (per-channel mean, averaged)
  blur         Laplacian variance(out) / Laplacian variance(in) < 0.5 (both at the generation size)
VLM tags (loss set only): the suite's judge model with a DIAGNOSTIC prompt (not a benchmark rubric;
the benchmark judging is untouched), JSON-schema output, temperature 0. One primary label from
LABELS + secondary labels.
Writes results/diagnosis/cause3_auto.csv, cause3_vlm.jsonl, cause3_modes.csv   (env: internvlu)
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EVAL, "pipeline"))
from gen import area_size  # noqa: E402
from judge import Endpoint, with_retries  # noqa: E402

OUT = os.path.join(EVAL, "results", "diagnosis")
MODELS = ["base", "T0", "T2", "T3"]
LOSS_K = 40
TAU_NOEDIT, TAU_DRIFT, TAU_BLUR = 0.03, 0.06, 0.5
LABELS = ["correct", "no_edit", "partial_edit", "instruction_ignored", "wrong_temporal_direction",
          "global_drift", "degraded"]
PROMPT = """You are diagnosing the output of an instruction-based image-editing model.
You are given the editing instruction, the INPUT image (first) and the model's OUTPUT image (second).
Classify the output with exactly one primary label:
- correct: the requested change is clearly made and looks plausible
- no_edit: the output is essentially the input; the requested change is not made
- partial_edit: the requested change is started but clearly incomplete or much too weak
- instruction_ignored: a visible change was made, but it is not the requested one
- wrong_temporal_direction: the output shows an earlier state than the input, or reverses the requested process or direction of time
- global_drift: overall colours, style, framing/zoom or scene identity changed beyond what was requested
- degraded: blur, artifacts or distortion dominate the output
Also list any other labels that apply as secondary labels.

Instruction: {instruction}"""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["primary", "secondary", "reason"],
          "properties": {"primary": {"type": "string", "enum": LABELS},
                         "secondary": {"type": "array", "items": {"type": "string", "enum": LABELS}},
                         "reason": {"type": "string"}}}
cfg = json.load(open(os.path.join(EVAL, "pipeline", "config.json")))["inference"]


def load(p, size):
    return np.asarray(Image.open(p).convert("RGB").resize(size, Image.LANCZOS), np.float32) / 255


def lapvar(a):
    return float(cv2.Laplacian(cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def auto_tags(rows):
    out = []
    for r in rows:
        src = Image.open(r["inputs"][0]).convert("RGB")
        size = area_size(*src.size, cfg["gen_area_side"], cfg["round_to"])
        a = load(r["inputs"][0], size)
        la = lapvar(a)
        for mdl in MODELS:
            o = load(os.path.join(EVAL, "outputs", mdl, r["out_rel"]), size)
            mad = float(np.abs(o - a).mean())
            drift = float(np.abs(o.reshape(-1, 3).mean(0) - a.reshape(-1, 3).mean(0)).mean())
            sharp = lapvar(o) / max(la, 1e-6)
            out.append(dict(uid=r["uid"], model=mdl, bench=r["bench"], group=r["group"], edit_mad=mad,
                            color_shift=drift, sharpness_ratio=sharp, no_edit=mad < TAU_NOEDIT,
                            global_drift=drift > TAU_DRIFT, blur=sharp < TAU_BLUR))
    return pd.DataFrame(out)


def uri(p, max_side=768):
    im = Image.open(p).convert("RGB")
    im.thumbnail((max_side, max_side))
    b = io.BytesIO()
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", required=True)
    ap.add_argument("--served", default="qwen3-vl-30b-fp8")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    man = {json.loads(l)["uid"]: json.loads(l) for l in open(os.path.join(EVAL, "pipeline", "manifest.jsonl"))}
    ap_path = os.path.join(OUT, "cause3_auto.csv")
    if not os.path.exists(ap_path):
        auto_tags(list(man.values())).to_csv(ap_path, index=False)
    auto = pd.read_csv(ap_path)

    ex = pd.read_csv(os.path.join(EVAL, "results", "examples.csv"))
    norm = ex.set_index("uid")[[f"norm_{m}" for m in MODELS]]
    loss = {}
    for t in ("T0", "T2", "T3"):
        d = (norm[f"norm_{t}"] - norm["norm_base"]).sort_values()
        for uid in d.index[:LOSS_K]:
            loss.setdefault(uid, []).append(t)
    ep = Endpoint(a.endpoints.split(","), a.served)
    vpath = os.path.join(OUT, "cause3_vlm.jsonl")
    done = {(json.loads(l)["uid"], json.loads(l)["model"]) for l in open(vpath)} if os.path.exists(vpath) else set()
    lock = threading.Lock()

    def tag(uid, mdl):
        r = man[uid]
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(instruction=r["instruction"])},
            {"type": "image_url", "image_url": {"url": uri(r["inputs"][0])}},
            {"type": "image_url", "image_url": {"url": uri(os.path.join(EVAL, "outputs", mdl, r["out_rel"]))}}]}]

        def fn():
            resp = ep.create(messages=msgs, max_tokens=400, response_format={
                "type": "json_schema", "json_schema": {"name": "failure_mode", "schema": SCHEMA, "strict": True}})
            return json.loads(resp.choices[0].message.content)
        res, tries = with_retries(ep, fn, lambda x: x is not None and x.get("primary") in LABELS)
        rec = dict(uid=uid, model=mdl, bench=r["bench"], group=r["group"], loss_for=loss[uid],
                   primary=res["primary"] if res else None, secondary=res.get("secondary", []) if res else [],
                   reason=res.get("reason") if res else None, parse_ok=res is not None, tries=len(tries))
        with lock, open(vpath, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    todo = [(u, mdl) for u in loss for mdl in MODELS if (u, mdl) not in done]
    print(f"loss set {len(loss)} items; {len(todo)} VLM tags to do", flush=True)
    with ThreadPoolExecutor(48) as exr:
        list(exr.map(lambda x: tag(*x), todo))

    v = pd.DataFrame([json.loads(l) for l in open(vpath)]).drop_duplicates(["uid", "model"], keep="last")
    rows = []
    for scope, uids in [("loss_set", list(loss))] + [(f"loss_set_{t}", [u for u in loss if t in loss[u]]) for t in ("T0", "T2", "T3")]:
        vv = v[v.uid.isin(uids)]
        aa = auto[auto.uid.isin(uids)]
        for mdl in MODELS:
            s, sa = vv[vv.model == mdl], aa[aa.model == mdl]
            r = dict(scope=scope, model=mdl, n=len(s))
            for lab in LABELS:
                r[f"vlm_{lab}"] = float((s.primary == lab).mean()) if len(s) else np.nan
                r[f"vlm_any_{lab}"] = float(s.apply(lambda x: x.primary == lab or lab in (x.secondary or []), axis=1).mean()) if len(s) else np.nan
            for k in ("no_edit", "global_drift", "blur"):
                r[f"auto_{k}"] = float(sa[k].mean())
            r["mean_edit_mad"] = float(sa.edit_mad.mean())
            rows.append(r)
    for mdl in MODELS:  # all items: automatic modes only
        sa = auto[auto.model == mdl]
        rows.append(dict(scope="all_items", model=mdl, n=len(sa), **{f"auto_{k}": float(sa[k].mean()) for k in ("no_edit", "global_drift", "blur")},
                         mean_edit_mad=float(sa.edit_mad.mean())))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cause3_modes.csv"), index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows)[["scope", "model", "n"] + [f"vlm_{l}" for l in LABELS] + ["auto_no_edit", "auto_global_drift", "auto_blur"]].round(2).to_string())


if __name__ == "__main__":
    main()
