# 空间智能 HW3 - LeRobot CALVIN ACT

本仓库为课程项目 Task 2 的提交代码，主要研究 ACT 策略在 CALVIN LeRobot 数据集上的跨环境泛化能力。

仓库内容包括：

- 单环境与多环境模仿学习的 ACT 训练代码。
- 未见环境 D 上的 zero-shot 评估代码。
- ACT 动作分块（Action Chunking）分析脚本。
- 与 W&B 兼容的训练配置和实验结果日志。
- 可复现实验图表的绘图脚本和导出的结果表格。

以下大文件不纳入 Git 版本管理：

- `data/` 下的 CALVIN 数据集文件。
- `checkpoints/*.pt` 模型权重文件。
- W&B 二进制 run 目录与 TensorBoard 日志。

模型权重需要从提交材料中的云盘链接单独下载，并按下文说明放入 `checkpoints/` 目录。

## 仓库结构

```text
.
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
├── logs/                            # 导出的评估 CSV/JSON 结果
├── plots/                           # 生成的实验图表
└── checkpoints/
    └── MANIFEST.md                  # 期望的模型权重文件说明
```

## 环境配置

### 方式一：创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate lerobot_calvin
pip install -e ./src
```

`environment.yml` 包含 PyTorch、LeRobot、W&B、绘图依赖，以及可选的 CALVIN 仿真依赖。如果服务器上较难安装 CALVIN simulation 相关组件，数据集级 Action L1 评估仍然可以独立运行。

### 方式二：使用已有 CUDA/PyTorch 环境

本项目训练时使用服务器上已有的 CUDA 环境。激活已有环境后，只需要补充缺失的 Python 包：

```bash
pip install lerobot wandb huggingface_hub datasets einops timm transformers accelerate
pip install -e ./src
```

如果只在本地重绘结果图，安装最小绘图依赖即可：

```bash
pip install -r requirements-local.txt
```

## 数据准备

本项目使用 LeRobot 格式的 CALVIN 数据集：

```text
xiaoma26/calvin-lerobot
```

环境与数据划分对应关系如下：

| CALVIN split | 环境 | 用途 |
|---|---|---|
| `splitA` | Env A | A-only 训练与 ABC 训练 |
| `splitB` | Env B | ABC 训练 |
| `splitC` | Env C | ABC 训练 |
| `splitD` | Env D | 仅用于 zero-shot 评估 |

将数据下载到 `./data`：

```bash
# 先下载 Env A 与 Env D。
bash scripts/download_calvin.sh ./data single

# 再下载 Env B 与 Env C，用于多环境训练。
bash scripts/download_calvin.sh ./data bc

# 或一次性下载全部 split。
bash scripts/download_calvin.sh ./data all
```

下载后建议先校验数据完整性：

```bash
python scripts/check_calvin_integrity.py ./data
python scripts/check_calvin_integrity.py ./data --check-parquet --max-parquet 50
```

## 模型权重

模型权重不存放在 Git 仓库中。下载提交材料中的权重文件后，请按以下目录结构放置：

```text
checkpoints/
├── act_single_A_b256/
│   └── best_model.pt
└── act_multi_ABC_b256/
    └── best_model.pt
```

单独上传的权重文件夹中还提供了两个便于识别的文件名：

```text
single_A_best_model.pt
multi_ABC_best_model.pt
```

如果直接使用这两个文件名，可以将它们复制到上面的目录结构中，也可以在评估命令中手动修改 checkpoint 路径。

## 训练

主实验使用相同的 ACT 网络结构和超参数，主要控制变量是训练数据来源：

- A-only baseline：仅使用 Env A 训练。
- ABC model：混合使用 Env A、Env B、Env C 训练。

### 训练 A-only baseline

```bash
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

### 训练 ABC 多环境模型

```bash
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

### 可选实验：使用 ImageNet 初始化的 ResNet18 训练 A-only

先在可联网机器上准备 ResNet18 权重：

```bash
python scripts/prepare_resnet18.py --cache-dir /path/to/torch_cache
```

然后训练：

```bash
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

## 测试与评估

### Env D zero-shot Action L1

```bash
python eval_zeroshot.py \
  --data_dir ./data \
  --single_ckpt ./checkpoints/act_single_A_b256/best_model.pt \
  --multi_ckpt ./checkpoints/act_multi_ABC_b256/best_model.pt \
  --eval_env D \
  --batch_size 128 \
  --num_workers 8 \
  --out_path ./logs/eval_envD_results.json
```

该命令在完全未见过的 Env D split 上计算数据集级 Action L1 Error。最终报告采用该指标，是因为服务器环境中没有可用的 `calvin_env` 与 `calvin_agent`，无法稳定运行仿真 success rate；作业要求允许使用动作误差作为跨环境评估指标。

### Pretrained A-only 与 ABC 的 zero-shot 对比

```bash
python eval_zeroshot.py \
  --data_dir ./data \
  --single_ckpt ./checkpoints/act_single_A_b256_pretrained/best_model.pt \
  --multi_ckpt ./checkpoints/act_multi_ABC_b256/best_model.pt \
  --eval_env D \
  --batch_size 128 \
  --num_workers 8 \
  --out_path ./logs/eval_envD_pretrainedA_vs_ABC_results.json
```

### Action chunk horizon 分析

```bash
python scripts/analyze_eval_chunks.py \
  --data_dir ./data \
  --single_ckpt ./checkpoints/act_single_A_b256/best_model.pt \
  --multi_ckpt ./checkpoints/act_multi_ABC_b256/best_model.pt \
  --eval_env D \
  --batch_size 128 \
  --num_workers 8
```

### 绘制结果图

```bash
python scripts/plot_results.py --log_dir logs --out_dir plots
```

预期输出：

```text
plots/envD_per_dim_l1.png
plots/envD_per_dim_l1.pdf
plots/envD_chunk_horizon_l1.png
plots/envD_chunk_horizon_l1.pdf
plots/envD_chunk_improvement.png
plots/envD_chunk_improvement.pdf
plots/envD_pretrainedA_vs_ABC_per_dim_l1.png
plots/envD_pretrainedA_vs_ABC_per_dim_l1.pdf
```

## 主要结果

Env D zero-shot Action L1：

| 模型 | 训练环境 | Mean Action L1 |
|---|---|---:|
| ACT A-only | A | 0.1701 |
| ACT ABC | A+B+C | 0.1572 |

与 A-only baseline 相比，ABC 多环境模型在 Env D 上将平均 Action L1 降低了 7.6%。

辅助的 pretrained backbone 实验结果如下：

| 模型 | 训练环境 | Backbone 初始化 | Env D Mean Action L1 |
|---|---|---|---:|
| ACT A-only | A | 随机初始化 | 0.1701 |
| ACT A-only + ImageNet | A | ImageNet 预训练 | 0.1678 |
| ACT ABC | A+B+C | 随机初始化 | 0.1572 |

ImageNet 初始化的 A-only 模型相较随机初始化 A-only baseline 有小幅提升，但仍弱于 ABC 多环境模型。该结果支持报告中的结论：相比单纯依赖 backbone 初始化，覆盖更多视觉环境的数据更有助于提升跨环境鲁棒性。

## 关于 Success Rate

`eval_zeroshot.py` 保留了可选的 `--run_simulation` 参数，用于 CALVIN 仿真 success rate 评估。该功能需要安装 `calvin_env` 与 `calvin_agent`。最终服务器环境中缺少这些依赖，因此提交报告使用 Action L1 Error 作为替代评价指标。
