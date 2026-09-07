"""Calls a vLLM OpenAI-compatible server to label S4 rows (progression
description + next-frame text prediction) from S4_captioning_manifest.jsonl.

Parses the VLM's "Progression: ...\nNext frame: ..." response (format
required by s4_progression_prompt.build_vlm_label_prompt) into the two GT
text segments, then assembles the final GPT turn:

    {progression} Next frame prediction: {next_frame}. The next frame should look like this: <img>

Usage (test mode -- print instead of writing a training file):
    python label_with_vlm.py --manifest .../S4_captioning_manifest.jsonl \
        --base-url http://localhost:8123/v1 --model qwen3-vl-30b-fp8 \
        --sample-per-source 2 --print-only

Usage (full run):
    python label_with_vlm.py --manifest .../S4_captioning_manifest.jsonl \
        --base-url http://localhost:8123/v1 --model qwen3-vl-30b-fp8 \
        --out-jsonl .../S4_labels.jsonl --workers 16
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from pathlib import Path
from queue import Queue

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from s4_progression_prompt import GEN_TRIGGER, NEXT_FRAME_PREDICTION_MARKER

PROGRESSION_RE = re.compile(r"Progression:\s*(.*?)(?:\n+Next frame:|\Z)", re.S)
NEXT_FRAME_RE = re.compile(r"Next frame:\s*(.*)", re.S)


def parse_vlm_response(text: str) -> tuple[str, str] | None:
    prog_m = PROGRESSION_RE.search(text)
    next_m = NEXT_FRAME_RE.search(text)
    if not prog_m or not next_m:
        return None
    progression = " ".join(prog_m.group(1).split())
    next_frame = " ".join(next_m.group(1).split())
    if not progression or not next_frame:
        return None
    return progression, next_frame


def assemble_gpt_turn(progression: str, next_frame: str) -> str:
    progression = progression.rstrip()
    if not progression.endswith((".", "!", "?")):
        progression += "."
    next_frame = next_frame.rstrip()
    if not next_frame.endswith((".", "!", "?")):
        next_frame += "."
    return f"{progression} {NEXT_FRAME_PREDICTION_MARKER} {next_frame} {GEN_TRIGGER}"


def call_vlm(base_url: str, model: str, prompt: str, image_paths: list[str],
             timeout: int = 120) -> str:
    content = [{"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in image_paths]
    content.append({"type": "text", "text": prompt})
    resp = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 300,
            "temperature": 0.2,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def load_manifest_rows(manifest_path: Path, sample_per_source: int | None) -> list[dict]:
    rows = []
    if sample_per_source is None:
        with open(manifest_path) as f:
            return [json.loads(line) for line in f]
    seen_task = {}
    with open(manifest_path) as f:
        for line in f:
            d = json.loads(line)
            bucket = d["source"] if d["source"] != "vbvr" else f"vbvr:{d.get('task_name', '?')}"
            n = seen_task.get(bucket, 0)
            if n >= sample_per_source:
                continue
            seen_task[bucket] = n + 1
            rows.append(d)
    return rows


def worker(q: Queue, args, out_lock: threading.Lock, out_f, results: list):
    while True:
        row = q.get()
        if row is None:
            q.task_done()
            break
        try:
            raw = call_vlm(args.base_url, args.model, row["vlm_prompt"], row["vlm_images"])
            parsed = parse_vlm_response(raw)
        except Exception as e:
            raw, parsed = f"[ERROR] {e}", None
        rec = {"id": row["id"], "source": row["source"], "raw": raw,
               "parsed_ok": parsed is not None}
        if parsed:
            progression, next_frame = parsed
            rec["progression"] = progression
            rec["next_frame_text"] = next_frame
            rec["gpt_turn"] = assemble_gpt_turn(progression, next_frame)
        if args.print_only:
            results.append((row, rec))
        else:
            with out_lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-url", default="http://localhost:8123/v1")
    ap.add_argument("--model", default="qwen3-vl-30b-fp8")
    ap.add_argument("--sample-per-source", type=int, default=None,
                     help="if set, sample this many rows per source (vbvr broken out per task) "
                          "instead of running the full manifest")
    ap.add_argument("--out-jsonl", default=None)
    ap.add_argument("--print-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not args.print_only and not args.out_jsonl:
        ap.error("--out-jsonl required unless --print-only")

    rows = load_manifest_rows(Path(args.manifest), args.sample_per_source)
    print(f"labeling {len(rows)} rows with {args.workers} workers...", file=sys.stderr)

    q: Queue = Queue()
    for r in rows:
        q.put(r)
    for _ in range(args.workers):
        q.put(None)

    out_f = open(args.out_jsonl, "w") if args.out_jsonl else None
    out_lock = threading.Lock()
    results: list = []
    threads = [threading.Thread(target=worker, args=(q, args, out_lock, out_f, results))
               for _ in range(args.workers)]
    for t in threads:
        t.start()
    q.join()
    for t in threads:
        t.join()
    if out_f:
        out_f.close()

    if args.print_only:
        for row, rec in results:
            print(f"\n=== id={row['id']} source={row['source']} task={row.get('task_name')} ===")
            print("PROMPT CAPTION:", row["caption_prefix"][:150])
            print("RAW:", rec["raw"])
            if rec["parsed_ok"]:
                print("GPT TURN:", rec["gpt_turn"])
            else:
                print("!! PARSE FAILED !!")
    else:
        n_ok = sum(1 for _ in open(args.out_jsonl)) if args.out_jsonl else 0
        print(f"wrote {n_ok} label rows -> {args.out_jsonl}", file=sys.stderr)


if __name__ == "__main__":
    main()
