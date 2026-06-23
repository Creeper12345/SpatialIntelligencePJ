#!/bin/bash
# eval.sh — Zero-shot 跨环境评估（环境 D）
#
# 用法：bash scripts/eval.sh [DATA_DIR]
# 示例：bash scripts/eval.sh /data/calvin_lerobot

set -e

cd "$(dirname "$0")/.."

DATA_DIR=${1:-"./data"}
SINGLE_CKPT="./checkpoints/act_single_A/best_model.pt"
MULTI_CKPT="./checkpoints/act_multi_ABC/best_model.pt"

echo "=== Zero-shot 评估（环境 D）==="
echo "单环境模型: $SINGLE_CKPT"
echo "多环境模型: $MULTI_CKPT"
echo ""

python ./eval_zeroshot.py \
    --data_dir     "$DATA_DIR" \
    --single_ckpt  "$SINGLE_CKPT" \
    --multi_ckpt   "$MULTI_CKPT" \
    --eval_env     D \
    --batch_size   64 \
    --num_workers  4 \
    --img_h        256 \
    --img_w        256
    # 如需仿真 Success Rate，添加：
    # --run_simulation --num_sim_episodes 100

echo "评估完成！结果保存于 ./logs/"
