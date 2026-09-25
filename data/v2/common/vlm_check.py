#!/usr/bin/env python3
"""VLM content check for v2 video clips (Qwen3-VL-8B-Instruct via vLLM).

CV metrics (clip_qc.py) catch camera motion / cuts / static clips but not content problems:
label noise (a talking head labelled "breaking"), animation / motion graphics / time-lapse,
heavy text overlays. For each manifest row (optionally only those passing clip_qc thresholds)
this shows 4 evenly spaced frames plus the row's label and asks for a JSON verdict:

  shows_label   : the clip visibly shows <label> happening to an object/material/liquid
  real_footage  : real camera footage (not animation, CGI, graphics, slideshow, time-lapse)
  heavy_text    : large overlaid text/subtitles/logos covering a notable part of the frame
  caption       : one factual sentence describing the scene and the physical process
                  (candidate only -- the caption policy per source is decided separately)

Output: one json line per clip (`id`, the fields above, `raw`) in <out>; resumable.
Run in the `vllm` conda env (vllm 0.11.2, transformers 4.57.6).

Usage:
    CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/vllm/bin/python vlm_check.py \
        --manifest raw/<src>/selection_pool.jsonl --qc raw/<src>/qc.jsonl --label-key class \
        --max-cam 0.05 --min-obj 0.02 --out raw/<src>/vlm.jsonl
"""
import argparse, base64, json, os, re
from concurrent.futures import ThreadPoolExecutor

import cv2

MODEL = "/scratch/network/ssd2/junlin/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
FRAME_W = 448
BATCH = 256

PROMPT = """These are 4 frames sampled in order from a short video clip. Its dataset label is "{label}".
Answer with a JSON object only, no other text:
{{"shows_label": true/false,   // the clip visibly shows "{label}" happening to an object, material or liquid (not a metaphor, not just people talking)
 "real_footage": true/false,  // real camera footage, NOT animation, CGI, motion graphics, slideshow or time-lapse
 "heavy_text": true/false,    // large overlaid text, subtitles or logos covering a notable part of the frame
 "caption": "..."}}           // one factual sentence describing the scene and the physical process"""


def frames_b64(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for t in [int(n * k / 4) + n // 8 for k in range(4)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(t, max(0, n - 1)))
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        f = cv2.resize(f, (FRAME_W, int(round(h * FRAME_W / w / 2)) * 2))
        out.append("data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode())
    cap.release()
    return out


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    s = re.sub(r"//[^\n]*", "", m.group(0))
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        d = {}
        for k in ("shows_label", "real_footage", "heavy_text"):
            mm = re.search(rf'"{k}"\s*:\s*(true|false)', s)
            if mm:
                d[k] = mm.group(1) == "true"
        mm = re.search(r'"caption"\s*:\s*"(.*?)"', s, re.S)
        if mm:
            d["caption"] = mm.group(1)
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--qc")
    ap.add_argument("--label-key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-duration", type=float, default=0.0)
    ap.add_argument("--max-cuts", type=int, default=0)
    ap.add_argument("--max-cam", type=float, default=None)
    ap.add_argument("--min-obj", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.manifest)]
    if a.qc:
        qc = {json.loads(l)["id"]: json.loads(l) for l in open(a.qc)}
        def ok(r):
            q = qc.get(r["id"])
            return (q is not None and q.get("decode_ok") and "cam_motion" in q
                    and q["duration"] >= a.min_duration and q["n_cuts"] <= a.max_cuts
                    and (a.max_cam is None or q["cam_motion"] <= a.max_cam)
                    and (a.min_obj is None or q["obj_motion_max"] >= a.min_obj))
        rows = [r for r in rows if ok(r)]
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    rows = [r for r in rows if r["id"] not in done]
    if a.limit:
        rows = rows[:a.limit]
    print(f"[vlm] {len(rows)} clips to check ({len(done)} done)", flush=True)
    if not rows:
        return

    from vllm import LLM, SamplingParams
    snap = os.path.join(MODEL, sorted(os.listdir(MODEL))[-1])
    llm = LLM(model=snap, max_model_len=8192, limit_mm_per_prompt={"image": 4},
              gpu_memory_utilization=0.85, max_num_seqs=64)
    sp = SamplingParams(temperature=0.0, max_tokens=160)
    pool = ThreadPoolExecutor(16)
    with open(a.out, "a") as f:
        for b in range(0, len(rows), BATCH):
            chunk = rows[b:b + BATCH]
            imgs = list(pool.map(lambda r: frames_b64(r["video"]), chunk))
            msgs, keep = [], []
            for r, im in zip(chunk, imgs):
                if len(im) < 2:
                    f.write(json.dumps({"id": r["id"], "error": "decode"}) + "\n")
                    continue
                label = str(r[a.label_key]).replace("_", " ")
                content = [{"type": "image_url", "image_url": {"url": u}} for u in im]
                content.append({"type": "text", "text": PROMPT.format(label=label)})
                msgs.append([{"role": "user", "content": content}])
                keep.append(r)
            outs = llm.chat(msgs, sp, use_tqdm=False)
            for r, o in zip(keep, outs):
                txt = o.outputs[0].text
                f.write(json.dumps({"id": r["id"], **parse(txt), "raw": txt}) + "\n")
            f.flush()
            print(f"[vlm] {min(b + BATCH, len(rows))}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
