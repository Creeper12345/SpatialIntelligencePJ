#!/bin/bash
# train_multi.sh — 多环境 A+B+C 联合训练（2×V100）
# 超参数与 train_single.sh 完全一致，仅 --envs 不同
# 用法：bash scripts/train_multi.sh [DATA_DIR]

set -e

cd "$(dirname "$0")/.."

DATA_DIR=${1:-"./data"}
RUN_NAME="act_multi_ABC"

echo "=== 多环境 ACT 联合训练（Env A+B+C）==="
echo "数据目录: $DATA_DIR"
echo ""

torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    ./train_act.py \
    --data_dir        "$DATA_DIR" \
    --envs            A B C \
    --run_name        "$RUN_NAME" \
    --epochs          100 \
    --batch_size      32 \
    --lr              1e-4 \
    --weight_decay    1e-4 \
    --chunk_size      10 \
    --hidden_dim      512 \
    --dim_feedforward 3200 \
    --nheads          8 \
    --num_encoder_layers 4 \
    --num_decoder_layers 1 \
    --latent_dim      32 \
    --kl_weight       10.0 \
    --num_workers     4 \
    --save_every      10 \
    --wandb_project   hw3_lerobot_calvin

echo "完成！权重: ./checkpoints/$RUN_NAME/"
