#!/usr/bin/env bash
set -euo pipefail

CHUNK_SIZE="${1:?Usage: bash scripts/train_chunk_ablation.sh <chunk_size> [single|multi] [data_dir]}"
MODE="${2:-single}"
DATA_DIR="${3:-./data}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
MASTER_PORT="${MASTER_PORT:-29520}"
export WANDB_MODE="${WANDB_MODE:-offline}"

case "${MODE}" in
  single)
    ENVS=(A)
    RUN_NAME="act_single_A_b${BATCH_SIZE}_chunk${CHUNK_SIZE}"
    ;;
  multi)
    ENVS=(A B C)
    RUN_NAME="act_multi_ABC_b${BATCH_SIZE}_chunk${CHUNK_SIZE}"
    ;;
  *)
    echo "MODE must be single or multi, got: ${MODE}" >&2
    exit 2
    ;;
esac

echo "=== ACT chunk-size ablation ==="
echo "data_dir=${DATA_DIR}"
echo "mode=${MODE} envs=${ENVS[*]}"
echo "run_name=${RUN_NAME}"
echo "chunk_size=${CHUNK_SIZE} batch/card=${BATCH_SIZE} epochs=${EPOCHS}"

torchrun   --nproc_per_node=2   --master_port="${MASTER_PORT}"   train_act.py   --data_dir "${DATA_DIR}"   --envs "${ENVS[@]}"   --run_name "${RUN_NAME}"   --epochs "${EPOCHS}"   --batch_size "${BATCH_SIZE}"   --num_workers "${NUM_WORKERS}"   --prefetch_factor "${PREFETCH_FACTOR}"   --chunk_size "${CHUNK_SIZE}"   --wandb_mode offline
