# Spatial Intelligence HW3 - LeRobot CALVIN ACT

This repository contains the Task 2 code for the course project: cross-environment generalization of ACT policies on the CALVIN LeRobot dataset.

The repository includes:

- ACT training code for single-environment and multi-environment imitation learning.
- Zero-shot evaluation code on unseen CALVIN Env D.
- Action chunking analysis scripts.
- W&B-compatible training configuration and result logs.
- Reproducible plotting scripts and exported result tables.

Large artifacts are intentionally not tracked:

- CALVIN dataset files under `data/`.
- Model weights under `checkpoints/*.pt`.
- W&B binary run folders and TensorBoard logs.

Model weights should be downloaded separately from the submitted cloud-drive link and placed under `checkpoints/` as described below.

## Repository Structure

```text
.
├── train_act.py                     # ACT training entry point
├── eval_zeroshot.py                 # Env D zero-shot Action L1 evaluation
├── environment.yml                  # Conda environment specification
├── requirements-local.txt           # Minimal local plotting dependencies
├── src/calvin_lerobot/              # CALVIN LeRobot dataset wrapper
├── scripts/
│   ├── download_calvin.sh           # Dataset download helper
│   ├── train_single.sh              # A-only training
│   ├── train_multi.sh               # A+B+C joint training
│   ├── train_pretrained_A.sh        # A-only ImageNet-init ablation
│   ├── eval.sh                      # Zero-shot evaluation wrapper
│   ├── analyze_eval_chunks.py       # Action chunk horizon analysis
│   └── plot_results.py              # Plot result figures
├── logs/                            # Exported evaluation CSV/JSON results
├── plots/                           # Generated result figures
├── reports/                         # Runbook and report notes
└── checkpoints/
    └── MANIFEST.md                  # Expected model-weight files
```

## Environment Setup

### Option A: Create a Conda environment

```bash
conda env create -f environment.yml
conda activate lerobot_calvin
pip install -e ./src
```

`environment.yml` includes PyTorch, LeRobot, W&B, plotting dependencies, and optional CALVIN simulation dependencies. If the CALVIN simulation stack is difficult to install on your server, the dataset-level Action L1 evaluation still works without running the simulator.

### Option B: Use an existing CUDA/PyTorch environment

On the training server used for this project, an existing CUDA environment was used. After activating that environment, install only the missing Python packages:

```bash
pip install lerobot wandb huggingface_hub datasets einops timm transformers accelerate
pip install -e ./src
```

For local result plotting only:

```bash
pip install -r requirements-local.txt
```

## Data Preparation

The project uses the LeRobot-format CALVIN dataset from:

```text
xiaoma26/calvin-lerobot
```

Environment mapping:

| CALVIN split | Environment | Usage |
|---|---|---|
| `splitA` | Env A | A-only training and ABC training |
| `splitB` | Env B | ABC training |
| `splitC` | Env C | ABC training |
| `splitD` | Env D | Zero-shot evaluation only |

Download data into `./data`:

```bash
# Download Env A and Env D first.
bash scripts/download_calvin.sh ./data single

# Download Env B and Env C for multi-environment training.
bash scripts/download_calvin.sh ./data bc

# Or download all splits at once.
bash scripts/download_calvin.sh ./data all
```

Validate the downloaded dataset:

```bash
python scripts/check_calvin_integrity.py ./data
python scripts/check_calvin_integrity.py ./data --check-parquet --max-parquet 50
```

## Model Weights

Model weights are not stored in this Git repository. After downloading the submitted weight files, place them as:

```text
checkpoints/
├── act_single_A_b256/
│   └── best_model.pt
└── act_multi_ABC_b256/
    └── best_model.pt
```

The submitted upload package also contains two convenience files:

```text
single_A_best_model.pt
multi_ABC_best_model.pt
```

If using those names directly, either copy them into the directory layout above or update the checkpoint paths in the evaluation command.

## Training

All main experiments use the same ACT architecture and hyperparameters. The key controlled variable is the training data:

- A-only baseline: Env A only.
- ABC model: mixed Env A+B+C.

### Train A-only baseline

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

### Train ABC joint model

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

### Optional: Train A-only with ImageNet-initialized ResNet18

Prepare ResNet18 weights on a machine with internet access:

```bash
python scripts/prepare_resnet18.py --cache-dir /path/to/torch_cache
```

Then train:

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

## Testing and Evaluation

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

This computes dataset-level Action L1 Error on the unseen Env D split. The report uses this metric because the server environment did not include `calvin_env` and `calvin_agent` for simulation success-rate evaluation.

### Pretrained A-only vs ABC zero-shot comparison

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

### Action chunk horizon analysis

```bash
python scripts/analyze_eval_chunks.py \
  --data_dir ./data \
  --single_ckpt ./checkpoints/act_single_A_b256/best_model.pt \
  --multi_ckpt ./checkpoints/act_multi_ABC_b256/best_model.pt \
  --eval_env D \
  --batch_size 128 \
  --num_workers 8
```

### Plot result figures

```bash
python scripts/plot_results.py --log_dir logs --out_dir plots
```

Expected outputs:

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

## Main Results

Env D zero-shot Action L1:

| Model | Training environments | Mean Action L1 |
|---|---|---:|
| ACT A-only | A | 0.1701 |
| ACT ABC | A+B+C | 0.1572 |

The ABC model reduces Env D mean Action L1 by 7.6% relative to the A-only baseline.

Auxiliary pretrained-backbone result:

| Model | Training environments | Backbone initialization | Mean Action L1 on Env D |
|---|---|---|---:|
| ACT A-only | A | random | 0.1701 |
| ACT A-only + ImageNet | A | ImageNet pretrained | 0.1678 |
| ACT ABC | A+B+C | random | 0.1572 |

The pretrained A-only model improves slightly over the random-initialized A-only baseline, but still underperforms the multi-environment ABC model. This supports the conclusion that broader visual-environment coverage contributes more to cross-environment robustness than backbone initialization alone.

## Notes on Success Rate

`eval_zeroshot.py` includes an optional `--run_simulation` flag for CALVIN simulation success-rate evaluation. It requires `calvin_env` and `calvin_agent`. These dependencies were not available in the final server environment, so the submitted report uses Action L1 Error, which is allowed by the assignment as an alternative metric to success rate.
