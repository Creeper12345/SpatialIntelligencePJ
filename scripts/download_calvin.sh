#!/bin/bash
# download_calvin.sh — 从 HuggingFace 下载 xiaoma26/calvin-lerobot
#
# 用法：
#   bash scripts/download_calvin.sh ./data single   # Env A + D
#   bash scripts/download_calvin.sh ./data bc       # 补 Env B + C
#   bash scripts/download_calvin.sh ./data all      # 全量

set -e

SAVE_DIR=${1:-"./data"}
MODE=${2:-"all"}

echo "Downloading CALVIN LeRobot data"
echo "Output: $SAVE_DIR"
echo "Mode: $MODE"
if [ -n "$HF_ENDPOINT" ]; then
    echo "HF_ENDPOINT: $HF_ENDPOINT"
fi

mkdir -p "$SAVE_DIR"

python3 - <<PYEOF
import os
import sys
from huggingface_hub import snapshot_download

REPO_ID  = "xiaoma26/calvin-lerobot"
SAVE_DIR = "$SAVE_DIR"
MODE     = "$MODE"

MODE_SUBSETS = {
    "single": ["splitA", "splitD"],
    "bc":     ["splitB", "splitC"],
    "cd":     ["splitC", "splitD"],
    "d":      ["splitD"],
    "all":    ["splitA", "splitB", "splitC", "splitD"],
}

subsets = MODE_SUBSETS.get(MODE, MODE_SUBSETS["all"])
print(f"将下载: {subsets}\n")

for subset in subsets:
    dest = os.path.join(SAVE_DIR, subset)
    print(f"Downloading {subset} -> {dest}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=SAVE_DIR,
        allow_patterns=[f"{subset}/*"],
    )
    if not os.path.isdir(dest):
        raise RuntimeError(f"下载后未找到期望目录: {dest}")
    print(f"{subset} done\n")

print("Download complete.")
PYEOF

echo ""
echo "Total size:"
du -sh "$SAVE_DIR"
echo "Next: bash scripts/train_single.sh $SAVE_DIR"
