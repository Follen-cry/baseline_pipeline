#!/usr/bin/env python3
"""Download the full PhyCo-Sim (Kubric) dataset -- v2 source.

Repo: https://huggingface.co/datasets/nnsriram97/phyco_kubric (CC-BY-ND-4.0, GATED: the HF
account behind the token must first click "request access" on the repo page; auto-approved).
29.0 GB over 9 scene folders, each holding per-date *.tar.gz shards plus small metadata
(common_caption_cosmos.txt, props_of_interest.json, data_stats_json.tar.gz). Shards are
compressed tars, so partial download isn't possible: fetch everything, then the 10K
selection (stratified by scene x physical-parameter bins) happens after extraction in a
separate step, once the per-sample layout is known.

Usage:
    python download_phyco_sim.py              # download (resumable)
    python download_phyco_sim.py --extract    # also extract every *.tar.gz in place
"""
import argparse, glob, os, subprocess, sys

from huggingface_hub import snapshot_download

REPO = "nnsriram97/phyco_kubric"
RAW_ROOT = "/scratch/network/ssd/junlin/raw/phyco_sim"
TOKEN_FILE = "/scratch/network/ssd2/junlin/huggingface/token"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(RAW_ROOT, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or open(TOKEN_FILE).read().strip()
    snapshot_download(REPO, repo_type="dataset", local_dir=RAW_ROOT, token=token,
                      max_workers=args.workers)
    print("[download] done", flush=True)
    if args.extract:
        for tgz in sorted(glob.glob(os.path.join(RAW_ROOT, "*", "*.tar.gz"))):
            out = tgz[: -len(".tar.gz")]
            if os.path.isdir(out):
                continue
            os.makedirs(out)
            print(f"[extract] {tgz}", flush=True)
            subprocess.run(["tar", "-xzf", tgz, "-C", out], check=True)
        print("[extract] done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
