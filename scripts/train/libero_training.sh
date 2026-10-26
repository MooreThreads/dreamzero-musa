#!/bin/bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
export DS_ACCELERATOR=${DS_ACCELERATOR:-musa}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="${DREAMZERO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$DREAMZERO_ROOT"
export PYTHONPATH="$DREAMZERO_ROOT:${PYTHONPATH:-}"

if [ -n "${MUSA_VISIBLE_DEVICES:-}" ]; then
  export MUSA_VISIBLE_DEVICES
fi
NUM_GPUS="${NUM_GPUS:-1}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-}"
IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-}"
TOKENIZER_DIR="${TOKENIZER_DIR:-}"
MAX_STEPS="${MAX_STEPS:-1}"
ACTION_HORIZON="${ACTION_HORIZON:-24}"
NUM_ACTION_PER_BLOCK="${NUM_ACTION_PER_BLOCK:-24}"
IMAGE_RESOLUTION_WIDTH="${IMAGE_RESOLUTION_WIDTH:-320}"
IMAGE_RESOLUTION_HEIGHT="${IMAGE_RESOLUTION_HEIGHT:-160}"
FRAME_SEQLEN="${FRAME_SEQLEN:-50}"
REPORT_TO="${REPORT_TO:-none}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
if [ -z "${TF32:-}" ]; then
  if [ "${DS_ACCELERATOR,,}" = "musa" ]; then
    TF32=false
  else
    TF32=true
  fi
fi
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG-groot/vla/configs/deepspeed/zero2_offload.json}"
DEEPSPEED_ARG=()
if [ -n "$DEEPSPEED_CONFIG" ]; then
  DEEPSPEED_ARG=(training_args.deepspeed="$DEEPSPEED_CONFIG")
fi

for required_var in \
  LIBERO_DATA_ROOT OUTPUT_DIR WAN22_CKPT_DIR IMAGE_ENCODER_DIR TOKENIZER_DIR; do
  if [ -z "${!required_var}" ]; then
    echo "ERROR: $required_var must be set" >&2
    exit 2
  fi
done

if [ ! -d "$LIBERO_DATA_ROOT" ]; then
  echo "ERROR: LIBERO dataset not found at $LIBERO_DATA_ROOT" >&2
  exit 1
fi

python3 -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone groot/vla/experiment/experiment.py \
  report_to="$REPORT_TO" \
  data=dreamzero/libero_relative \
  wandb_project=dreamzero \
  train_architecture=full \
  num_frames=33 \
  action_horizon="$ACTION_HORIZON" \
  num_views=2 \
  model=dreamzero/vla \
  model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
  model/dreamzero/transform=dreamzero_cotrain \
  num_frame_per_block=2 \
  num_action_per_block="$NUM_ACTION_PER_BLOCK" \
  num_state_per_block=1 \
  seed=42 \
  training_args.learning_rate=1e-5 \
  "${DEEPSPEED_ARG[@]}" \
  save_steps=999999 \
  training_args.warmup_ratio=0.05 \
  output_dir="$OUTPUT_DIR" \
  per_device_train_batch_size=1 \
  max_steps="$MAX_STEPS" \
  weight_decay=1e-5 \
  save_total_limit=5 \
  upload_checkpoints=false \
  bf16=true \
  tf32="$TF32" \
  eval_bf16=true \
  dataloader_pin_memory=false \
  dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
  image_resolution_width="$IMAGE_RESOLUTION_WIDTH" \
  image_resolution_height="$IMAGE_RESOLUTION_HEIGHT" \
  save_lora_only=false \
  max_chunk_size=4 \
  frame_seqlen="$FRAME_SEQLEN" \
  save_strategy=no \
  libero_data_root="$LIBERO_DATA_ROOT" \
  dit_version="$WAN22_CKPT_DIR" \
  text_encoder_pretrained_path="$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  image_encoder_pretrained_path="$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  vae_pretrained_path="$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
  tokenizer_path="$TOKENIZER_DIR" \
  "$@"
