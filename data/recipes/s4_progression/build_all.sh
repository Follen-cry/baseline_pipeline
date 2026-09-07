#!/bin/bash
# Runs the full S4 derivation pipeline: each source's caption-prefix
# recovery, then derive_s4_rows.py. Produces S4_skeleton.jsonl +
# S4_captioning_manifest.jsonl (NOT a training-ready S4_train.jsonl --
# GT captioning still needs an external VLM pass, see README.md).
#
# Paths to the per-source manifest/anchors files point at the OLD
# (not-yet-migrated) ssl_mllm/data tree, same convention already used by
# ../s0s3_baseline/merge_final_s0s3.py -- these intermediate jsonls haven't
# been copied into baseline_pipeline yet (see ../../README.md).
set -euo pipefail
cd "$(dirname "$0")"

BASELINE_ROOT="$(cd ../.. && pwd)"          # .../baseline_pipeline/data
OLD_ROOT="/scratch/network/ssd2/junlin/ssl_mllm/data"  # not-yet-migrated intermediates

S2_JSONL="$BASELINE_ROOT/datasets/final_s0s3/S2_train.jsonl"
CP_DIR="$BASELINE_ROOT/datasets/s4_progression/caption_prefixes"
OUT_DIR="$BASELINE_ROOT/datasets/s4_progression"
mkdir -p "$CP_DIR"

python3 "$BASELINE_ROOT/sources/vbvr/processing/build_s4_caption_prefix.py" \
  --s2-jsonl "$S2_JSONL" \
  --out-jsonl "$CP_DIR/vbvr_s4_caption_prefix.jsonl"

python3 "$BASELINE_ROOT/sources/epic_kitchens/processing/build_s4_caption_prefix.py" \
  --s2-jsonl "$S2_JSONL" \
  --manifest "$OLD_ROOT/datasets/epic_ssl/epic_gen_action/manifest_train.jsonl" \
  --out-jsonl "$CP_DIR/epic_s4_caption_prefix.jsonl"

python3 "$BASELINE_ROOT/sources/nwm/processing/build_s4_caption_prefix.py" \
  --s2-jsonl "$S2_JSONL" \
  --manifest "$OLD_ROOT/datasets/nwm_ssl/nwm_gen_action/manifest_train.jsonl" \
  --out-jsonl "$CP_DIR/nwm_s4_caption_prefix.jsonl"

python3 "$BASELINE_ROOT/sources/panda70m_epic/processing/build_s4_caption_prefix.py" \
  --s2-jsonl "$S2_JSONL" \
  --anchors "$OLD_ROOT/datasets/panda70m_epic_ssl/anchors/anchors_train.jsonl" \
  --out-jsonl "$CP_DIR/panda70m_epic_s4_caption_prefix.jsonl"

python3 "$BASELINE_ROOT/sources/panda70m_v1/processing/build_s4_caption_prefix.py" \
  --s2-jsonl "$S2_JSONL" \
  --manifest "$OLD_ROOT/datasets/panda70m_ssl_v1/panda70m_v1_gen_S2/manifest_train.jsonl" \
  --out-jsonl "$CP_DIR/panda70m_v1_s4_caption_prefix.jsonl"

python3 derive_s4_rows.py \
  --s2-jsonl "$S2_JSONL" \
  --caption-prefix-dir "$CP_DIR" \
  --out-dir "$OUT_DIR"
