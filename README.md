# 深度学习与空间智能 HW3 项目提交

本仓库按照课程提交要求组织代码，根目录只保留项目说明与两个任务目录：

```text
.
├── README.md
├── task1/                 # Task 1：3D 视觉资产重建、生成与场景融合代码目录
└── task2/                 # Task 2：CALVIN 上的 ACT 策略训练与跨环境评估代码
```

当前仓库不提交数据集、训练日志、实验结果图片和模型权重。模型权重单独上传，见“模型权重”一节。

## Task 1 目录

`task1/` 预留为 3D 视觉任务代码目录，结构如下：

```text
task1/
├── object_A/              # 多视角真实物体重建
├── object_B/              # 文本到 3D 资产生成
├── object_C/              # 单图到 3D 资产生成
├── background/            # 背景场景重建
└── fusion/                # 物体与背景融合、渲染
```

## Task 2 目录

Task 2 使用 LeRobot 框架中的 ACT 策略，在 CALVIN 数据集上比较单环境训练与多环境训练的跨环境泛化能力。核心代码位于 `task2/`：

```text
task2/
├── train_act.py                     # ACT 训练入口
├── eval_zeroshot.py                 # Env D zero-shot Action L1 评估入口
├── environment.yml                  # Conda 环境配置
├── requirements-local.txt           # 本地绘图所需的最小依赖
├── src/calvin_lerobot/              # CALVIN LeRobot 数据集读取封装
├── scripts/
│   ├── download_calvin.sh           # 数据集下载脚本
│   ├── train_single.sh              # 单环境 A 训练脚本
│   ├── train_multi.sh               # A+B+C 多环境联合训练脚本
│   ├── train_pretrained_A.sh        # ImageNet 初始化的 A-only 消融训练脚本
│   ├── eval.sh                      # zero-shot 评估封装脚本
│   ├── analyze_eval_chunks.py       # Action chunk horizon 分析脚本
│   └── plot_results.py              # 实验结果绘图脚本
└── checkpoints/
    └── MANIFEST.md                  # 期望的模型权重文件说明
```

## Requirements

推荐使用 Conda 创建环境：

```bash
cd task2
conda env create -f environment.yml
conda activate lerobot_calvin
```

如果服务器已有 CUDA/PyTorch 环境，也可以在激活已有环境后安装必要依赖：

```bash
cd task2
pip install lerobot wandb huggingface_hub datasets einops timm transformers accelerate
```

如果只需要在本地重绘图表，可安装最小绘图依赖：

```bash
cd task2
pip install -r requirements-local.txt
```

## 数据准备

Task 2 使用 LeRobot 格式的 CALVIN 数据集：

```text
xiaoma26/calvin-lerobot
```

数据划分与用途如下：

| CALVIN split | 环境 | 用途 |
|---|---|---|
| `splitA` | Env A | A-only 训练与 ABC 训练 |
| `splitB` | Env B | ABC 训练 |
| `splitC` | Env C | ABC 训练 |
| `splitD` | Env D | zero-shot 测试 |

下载数据到 `task2/data`：

```bash
cd task2

# 下载 Env A 与 Env D。
bash scripts/download_calvin.sh ./data single

# 下载 Env B 与 Env C，用于多环境训练。
bash scripts/download_calvin.sh ./data bc

# 或一次性下载全部 split。
bash scripts/download_calvin.sh ./data all
```

校验数据完整性：

```bash
cd task2
python scripts/check_calvin_integrity.py ./data
python scripts/check_calvin_integrity.py ./data --check-parquet --max-parquet 50
```

## Train

主实验使用相同 ACT 网络结构和超参数，主要控制变量为训练环境：

- `act_single_A_b256`：仅使用 Env A 训练。
- `act_multi_ABC_b256`：混合 Env A、Env B、Env C 训练。

训练 A-only baseline：

```bash
cd task2
WANDB_MODE=offline torchrun --nproc_per_node=2 --master_port=29500 train_act.py \
  --data_dir ./data \
  --envs A \
  --run_name act_single_A_b256 \
  --epochs 100 \
  --batch_size 256 \
  --num_workers 16 \
  --prefetch_factor 4 \
  --wandb_mode offline
```

训练 ABC 多环境模型：

```bash
cd task2
WANDB_MODE=offline torchrun --nproc_per_node=2 --master_port=29501 train_act.py \
  --data_dir ./data \
  --envs A B C \
  --run_name act_multi_ABC_b256 \
  --epochs 100 \
  --batch_size 256 \
  --num_workers 16 \
  --prefetch_factor 4 \
  --wandb_mode offline
```

可选：训练 ImageNet 初始化的 A-only 消融模型。

```bash
cd task2
python scripts/prepare_resnet18.py --cache-dir /path/to/torch_cache
export TORCH_HOME=/path/to/torch_cache

WANDB_MODE=offline torchrun --nproc_per_node=2 --master_port=29510 train_act.py \
  --data_dir ./data \
  --envs A \
  --run_name act_single_A_b256_pretrained \
  --epochs 100 \
  --batch_size 256 \
  --num_workers 16 \
  --prefetch_factor 4 \
  --wandb_mode offline \
  --pretrained_backbone
```

## Test

将模型权重放入 `task2/checkpoints/` 后，运行 Env D zero-shot Action L1 评估：

```bash
cd task2
python eval_zeroshot.py \
  --data_dir ./data \
  --single_ckpt ./checkpoints/act_single_A_b256/best_model.pt \
  --multi_ckpt ./checkpoints/act_multi_ABC_b256/best_model.pt \
  --eval_env D \
  --batch_size 128 \
  --num_workers 8 \
  --out_path ./logs/eval_envD_results.json
```

运行 ACT action chunk horizon 分析：

```bash
cd task2
python scripts/analyze_eval_chunks.py \
  --data_dir ./data \
  --single_ckpt ./checkpoints/act_single_A_b256/best_model.pt \
  --multi_ckpt ./checkpoints/act_multi_ABC_b256/best_model.pt \
  --eval_env D \
  --batch_size 128 \
  --num_workers 8
```

如需使用 CALVIN 仿真器计算 success rate，可在安装 `calvin_env` 与 `calvin_agent` 后添加 `--run_simulation`。当前提交报告采用数据集级 Action L1 Error，这是作业允许的跨环境评估指标。

## 模型权重

模型权重不放入 Git 仓库。请从单独提交的权重文件夹中获取最佳权重，并放置为：

```text
task2/checkpoints/
├── act_single_A_b256/
│   └── best_model.pt
└── act_multi_ABC_b256/
    └── best_model.pt
```

本次整理挑出的最佳模型文件为：

```text
single_A_best_model.pt
multi_ABC_best_model.pt
```

这两个文件分别对应 A-only baseline 与 ABC 多环境模型的最佳 checkpoint。
