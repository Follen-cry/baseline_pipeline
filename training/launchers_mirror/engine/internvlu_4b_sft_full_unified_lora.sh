#!/bin/bash
# =============================================================================
# InternVL-U full unified SFT — LoRA on BOTH the LLM side (rank 16) and the
# generation_decoder side (rank 16). Heavy memory savings vs. the full-finetune
# variant.
#
# Required environment variables:
#   INTERNVLU_CKPT        Path or HF id of the InternVL-U checkpoint. Either
#                         the snapshot root (with model_index.json + vlm/,
#                         generation_decoder/, ...) or the inner vlm/ subdir.
#   META_PATH             Path to the dataset meta JSON.
#   OUTPUT_DIR            Where to write checkpoints / logs.
#
# Optional (with defaults):
#   GPUS                  default 8
#   PER_DEVICE_BATCH_SIZE default 2
#   BATCH_SIZE            default 64
#   USE_LLM_LORA          default 16
#   USE_GEN_LORA          default 16
#   IMGEN_RATIO           default 0.3
#   GEN_LOSS_WEIGHT       default 0.1
#   CFG_DROPOUT           default 0.1
#   GEN_DECODER_LR        default 1e-4   (LoRA rank lets us push the LR up)
#   GEN_LOSS_WARMUP       default 1000
# =============================================================================

set -x

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton-cache-$USER}
mkdir -p $TRITON_CACHE_DIR

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

: "${INTERNVLU_CKPT:=/homes/55/junlin/.cache/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01/}"
: "${META_PATH:=/scratch/network/ssd2/junlin/ssl_mllm/data/meta/spatial_ssrl_co_sft_meta.json}"
: "${OUTPUT_DIR:=/scratch/network/ssd2/junlin/models/internvl-u-ee}"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export INTERNVLU_PKG_PATH=${INTERNVLU_PKG_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/InternVL-U}
export MASTER_PORT=${MASTER_PORT:-34231}
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

mkdir -p "$OUTPUT_DIR"

SNAPSHOT_ARG=""
if [[ -n "${SNAPSHOT_DIR}" ]]; then
  SNAPSHOT_ARG="--snapshot_dir ${SNAPSHOT_DIR}"
fi

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune_u_full.py \
  --model_name_or_path "${INTERNVLU_CKPT}" \
  ${SNAPSHOT_ARG} \
  --conv_style "qwen2_5-chat-v3" \
  --use_fast_tokenizer False \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --use_llm_lora ${USE_LLM_LORA:-16} \
  --vision_select_layer -1 \
  --freeze_gen_decoder False \
  --use_gen_decoder_lora ${USE_GEN_LORA:-16} \
  --freeze_vae True \
  --gen_decoder_lr ${GEN_DECODER_LR:-1e-4} \
  --gen_loss_weight ${GEN_LOSS_WEIGHT:-0.1} \
  --gen_loss_warmup_steps ${GEN_LOSS_WARMUP:-1000} \
  --imgen_ratio ${IMGEN_RATIO:-0.3} \
  --cfg_dropout ${CFG_DROPOUT:-0.1} \
  --cfg_dropout_all_frac ${CFG_DROPOUT_ALL_FRAC:-0.5} \
  --gen_image_size ${GEN_IMAGE_SIZE:-1024} \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 4 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --eval_strategy "no" \
  --save_strategy "no" \
  --save_total_limit 2 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
