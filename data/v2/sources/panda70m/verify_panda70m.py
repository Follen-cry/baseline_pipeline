#!/usr/bin/env python3
"""Visual check of downloaded Panda-70M clips (annotation only; nothing is deleted).

The caption judge (filter_panda70m.py) can't see the video. A manual spot check of 24
downloaded clips found ~40% whose frames don't match the goal: people or swimmers dominating
the frame, CGI animation, near-static scenes, title cards, and cuts missed by Panda's shot
detection. This stage shows the same Qwen3-VL-30B-A3B-Instruct-FP8 model a 2x2 grid of 4 frames
(10/37/63/90% of the clip, in order) plus the caption, and records a visual verdict per clip.

Output: <OUT>/_filter/visual_verify.jsonl, one row per clip_key with `visual` =
  {physical_process_visible, real_footage, single_scene, visible_motion, person_dominant,
   score (1-5)}.
visual_pass = physical_process_visible AND real_footage AND single_scene AND visible_motion
              AND NOT person_dominant AND score >= 3.

Without --keys, the verdicts are also merged into <OUT>/selection_5k.jsonl as a `visual` field.
Spot check (24 clips, 2026-09-24): agrees with a manual label on 17/24. Of its 15 passes, 11
were clean (73%, vs. 58% for the caption filter alone). It misses static scenes and missed cuts,
and over-flags `person_dominant` when firefighters or bystanders are in frame. So use it as a
soft score, not a hard filter.

Usage:
    python verify_panda70m.py                 # every successful clip in download_log.jsonl
    python verify_panda70m.py --keys a,b,c    # specific clips (spot checks)
"""
import argparse, base64, json, os, re, time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests

OUT = "/scratch/network/ssd/junlin/raw/panda70m"
VIDEOS = os.path.join(OUT, "videos")
POOL = os.path.join(OUT, "pool_passed.jsonl")
LOG = os.path.join(OUT, "download_log.jsonl")
VERIFY = os.path.join(OUT, "_filter", "visual_verify.jsonl")
SELECTION = os.path.join(OUT, "selection_5k.jsonl")
VLLM_BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8010/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen3-vl-30b-fp8")
FRACS = (0.10, 0.37, 0.63, 0.90)

PROMPT = """The image is a 2x2 grid of 4 frames from one short video clip, in time order: \
top-left, top-right, bottom-left, bottom-right. Its auto-generated caption is:
"{caption}"

We want real footage of a PHYSICAL PROCESS: objects or materials moving or changing under \
physics (falling, collisions, crashes, fluids flowing or splashing, waves, fire, smoke, \
explosions, sparks, breaking, melting, boiling, spinning, wind-blown motion).

Answer from what you SEE in the frames:
- "physical_process_visible": the physical process is clearly visible and is the main content.
- "real_footage": real camera footage, not CGI / animation / video game / slideshow / title card.
- "single_scene": all 4 frames show the same continuous scene (no cut to a different shot or \
place, no title/text card).
- "visible_motion": objects or materials visibly move or change between frames (camera motion \
alone does not count).
- "person_dominant": a person, face, athlete or swimmer dominates the frames rather than the \
physical process.
- "score": 1-5, overall fit for the goal.

Return ONLY a JSON object with these six keys."""

_JSON_RE = re.compile(r"\{.*\}", re.S)


def grid_jpeg_b64(path: str) -> str:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for f in FRACS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * f))
        ok, im = cap.read()
        if not ok:
            im = frames[-1] if frames else np.zeros((360, 640, 3), np.uint8)
        frames.append(cv2.resize(im, (448, 252)))
    cap.release()
    grid = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:])])
    return base64.b64encode(cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).decode()


def visual_pass(v: dict) -> bool:
    return (v["physical_process_visible"] and v["real_footage"] and v["single_scene"]
            and v["visible_motion"] and not v["person_dominant"] and v["score"] >= 3)


def check(key: str, caption: str) -> dict:
    img = grid_jpeg_b64(os.path.join(VIDEOS, f"{key}.mp4"))
    payload = {"model": VLLM_MODEL, "temperature": 0, "max_tokens": 120, "messages": [{
        "role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
            {"type": "text", "text": PROMPT.format(caption=caption)}]}]}
    for attempt in range(4):
        try:
            r = requests.post(f"{VLLM_BASE}/chat/completions", json=payload, timeout=180)
            r.raise_for_status()
            o = json.loads(_JSON_RE.search(r.json()["choices"][0]["message"]["content"]).group(0))
            v = {k: bool(o[k]) for k in ("physical_process_visible", "real_footage",
                                          "single_scene", "visible_motion", "person_dominant")}
            v["score"] = int(o.get("score", 0))
            v["pass"] = visual_pass(v)
            return {"clip_key": key, "visual": v}
        except Exception as e:  # noqa: BLE001 -- retry transient server / parse errors
            err = e
            time.sleep(2 * (attempt + 1))
    return {"clip_key": key, "error": repr(err)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=VERIFY)
    a = ap.parse_args()
    caps = {}
    for l in open(POOL):
        r = json.loads(l)
        caps[r["clip_key"]] = r["caption"]
    if a.keys:
        keys = a.keys.split(",")
    else:
        keys = [r["clip_key"] for r in map(json.loads, open(LOG)) if r["status"] == "success"]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["clip_key"] for l in open(a.out) if "visual" in l}
    keys = [k for k in keys if k not in done and k in caps]
    print(f"[verify] {len(keys)} clips to check", flush=True)
    n = n_pass = 0
    with open(a.out, "a") as out, ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(lambda k: check(k, caps[k]), keys):
            out.write(json.dumps(r) + "\n")
            n += 1
            n_pass += r.get("visual", {}).get("pass", False)
            if n % 200 == 0:
                out.flush()
                print(f"[verify] {n}/{len(keys)} pass {n_pass}", flush=True)
    print(f"[verify] done: {n} checked, {n_pass} pass -> {a.out}")
    if not a.keys and os.path.exists(SELECTION):
        vis = {}
        for l in open(a.out):
            r = json.loads(l)
            if "visual" in r:
                vis[r["clip_key"]] = r["visual"]
        rows = [json.loads(l) for l in open(SELECTION)]
        with open(SELECTION, "w") as f:
            for r in rows:
                r["visual"] = vis.get(r["id"])
                f.write(json.dumps(r) + "\n")
        n_vis = sum(r["visual"] is not None for r in rows)
        n_ok = sum(bool(r["visual"] and r["visual"]["pass"]) for r in rows)
        print(f"[verify] merged into {SELECTION}: {n_vis}/{len(rows)} annotated, {n_ok} visual pass")


if __name__ == "__main__":
    main()
