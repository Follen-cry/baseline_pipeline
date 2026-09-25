#!/usr/bin/env bash
# End-to-end driver used for the 2026-09-24 build (run on torrnode11, next to the vLLM server).
# Downloads overlap with judging: while a judge stage runs, each round re-exports the pool and
# attempts another chunk of downloads (download_panda70m.py skips clips already attempted).
# Usage: bash run_pipeline.sh [pid of an already-running tier-A judge]
set -u
cd "$(dirname "$0")"
R=/scratch/network/ssd/junlin/raw/panda70m
LOGD=$R/_filter
dl_rounds_while() {  # $1 = pid to wait on
  while kill -0 "$1" 2>/dev/null; do
    python3 filter_panda70m.py --stage export > /dev/null
    python3 download_panda70m.py --target 5000 --workers 4 --limit 300 >> $R/download.log 2>&1
    kill -0 "$1" 2>/dev/null && sleep 60
  done
}
if [ -n "${1:-}" ]; then A=$1; else
  python3 filter_panda70m.py --stage judge --tier A --target-pass 1000000 >> $LOGD/judge.log 2>&1 & A=$!
fi
dl_rounds_while $A
python3 filter_panda70m.py --stage judge --tier B --target-pass 1000000 >> $LOGD/judge_tierB.log 2>&1 & B=$!
dl_rounds_while $B
python3 filter_panda70m.py --stage export | tee $LOGD/export.log
python3 download_panda70m.py --target 5000 --workers 4 >> $R/download.log 2>&1
echo "[pipeline] finished $(date)" >> $R/download.log
