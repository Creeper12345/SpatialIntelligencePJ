#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-./data}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
MASTER_PORT="${MASTER_PORT:-29510}"
TORCH_HOME_DEFAULT="/inspire/hdd/project/socialsimulation/public/szdai/torch_cache"
export TORCH_HOME="${TORCH_HOME:-$TORCH_HOME_DEFAULT}"
export WANDB_MODE="${WANDB_MODE:-offline}"

RUN_NAME="act_single_A_b${BATCH_SIZE}_pretrained"

echo "=== A-only ACT with ImageNet ResNet18 init ==="
echo "data_dir=${DATA_DIR}"
echo "run_name=${RUN_NAME}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "batch/card=${BATCH_SIZE} epochs=${EPOCHS}"

torchrun   --nproc_per_node=2   --master_port="${MASTER_PORT}"   train_act.py   --data_dir "${DATA_DIR}"   --envs A   --run_name "${RUN_NAME}"   --epochs "${EPOCHS}"   --batch_size "${BATCH_SIZE}"   --num_workers "${NUM_WORKERS}"   --prefetch_factor "${PREFETCH_FACTOR}"   --wandb_mode offline   --pretrained_backbone
