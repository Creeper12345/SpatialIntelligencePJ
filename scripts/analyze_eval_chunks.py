#!/usr/bin/env python3
"""分析 CALVIN Env D 上的 zero-shot ACT chunk 误差。

该脚本只做推理，复用已训练权重，生成动作维度和 chunk 步长上的表格与图。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from calvin_lerobot.dataset import build_calvin_dataset
from eval_zeroshot import load_model


ACTION_DIMS = ["tx", "ty", "tz", "rx", "ry", "rz", "gripper"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--single_ckpt", required=True)
    p.add_argument("--multi_ckpt", required=True)
    p.add_argument("--eval_env", default="D")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--out_dir", default="./logs")
    p.add_argument("--plot_dir", default="./plots")
    return p.parse_args()


def move_batch_to_input(batch: dict, device: torch.device) -> tuple[dict, torch.Tensor]:
    model_input = {
        "observation.images.image": batch["observation.images.image"].squeeze(1).to(device),
        "observation.state": batch["observation.state"].squeeze(1).to(device),
    }
    if "observation.images.wrist_image" in batch:
        model_input["observation.images.wrist_image"] = (
            batch["observation.images.wrist_image"].squeeze(1).to(device)
        )
    return model_input, batch["action"].to(device)


@torch.no_grad()
def collect_errors(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> np.ndarray:
    """返回形状为 (N, T, 7) 的绝对动作误差。"""
    chunks: list[np.ndarray] = []
    model.eval()

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches and batch_idx > max_batches:
            break

        model_input, action_gt = move_batch_to_input(batch, device)
        if hasattr(model, "predict_action_chunk"):
            pred = model.predict_action_chunk(model_input)
        else:
            pred = model.select_action(model_input)
            if pred.ndim == 2:
                pred = pred.unsqueeze(1)

        horizon = min(pred.shape[1], action_gt.shape[1])
        err = (pred[:, :horizon, :] - action_gt[:, :horizon, :]).abs()
        chunks.append(err.cpu().numpy())

        if batch_idx % 50 == 0:
            print(f"  processed {batch_idx} batches", flush=True)

    return np.concatenate(chunks, axis=0)


def summarize(errors: np.ndarray) -> dict:
    per_horizon = errors.mean(axis=(0, 2))
    per_dim = errors.mean(axis=(0, 1))
    return {
        "mean": float(errors.mean()),
        "per_horizon": per_horizon.tolist(),
        "per_dim": {name: float(value) for name, value in zip(ACTION_DIMS, per_dim)},
        "num_samples": int(errors.shape[0]),
        "chunk_size": int(errors.shape[1]),
    }


def write_csvs(out_dir: Path, single: dict, multi: dict) -> None:
    with (out_dir / "eval_envD_per_horizon.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["horizon_step", "single_A_l1", "multi_ABC_l1", "relative_improvement_pct"]
        )
        writer.writeheader()
        for idx, (s, m) in enumerate(zip(single["per_horizon"], multi["per_horizon"]), start=1):
            writer.writerow(
                {
                    "horizon_step": idx,
                    "single_A_l1": s,
                    "multi_ABC_l1": m,
                    "relative_improvement_pct": (s - m) / s * 100 if s else 0.0,
                }
            )

    with (out_dir / "eval_envD_per_dim.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["action_dim", "single_A_l1", "multi_ABC_l1", "relative_improvement_pct"]
        )
        writer.writeheader()
        for name in ACTION_DIMS:
            s = single["per_dim"][name]
            m = multi["per_dim"][name]
            writer.writerow(
                {
                    "action_dim": name,
                    "single_A_l1": s,
                    "multi_ABC_l1": m,
                    "relative_improvement_pct": (s - m) / s * 100 if s else 0.0,
                }
            )


def maybe_plot(plot_dir: Path, single: dict, multi: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped PNG plots")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    xs = np.arange(1, len(single["per_horizon"]) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, single["per_horizon"], marker="o", label="A-only")
    plt.plot(xs, multi["per_horizon"], marker="o", label="ABC")
    plt.xlabel("Action chunk step")
    plt.ylabel("Env D Action L1")
    plt.title("Zero-shot error by ACT chunk horizon")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "envD_chunk_horizon_l1.png", dpi=200)
    plt.close()

    x = np.arange(len(ACTION_DIMS))
    width = 0.38
    single_values = [single["per_dim"][name] for name in ACTION_DIMS]
    multi_values = [multi["per_dim"][name] for name in ACTION_DIMS]
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, single_values, width, label="A-only")
    plt.bar(x + width / 2, multi_values, width, label="ABC")
    plt.xticks(x, ACTION_DIMS)
    plt.ylabel("Env D Action L1")
    plt.title("Zero-shot error by action dimension")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "envD_per_dim_l1.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print("Loading models ...")
    single_model = load_model(args.single_ckpt, device)
    multi_model = load_model(args.multi_ckpt, device)

    print(f"Loading CALVIN Env {args.eval_env} ...")
    dataset = build_calvin_dataset(
        args.data_dir,
        envs=[args.eval_env],
        action_horizon=single_model.config.chunk_size,
        use_wrist_cam=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    print(f"Samples: {len(dataset)}")

    print("Evaluating A-only model ...")
    single_errors = collect_errors(single_model, loader, device, args.max_batches)
    print("Evaluating ABC model ...")
    multi_errors = collect_errors(multi_model, loader, device, args.max_batches)

    single_summary = summarize(single_errors)
    multi_summary = summarize(multi_errors)
    result = {
        "eval_env": args.eval_env,
        "single_A": single_summary,
        "multi_ABC": multi_summary,
        "relative_improvement_pct": (
            (single_summary["mean"] - multi_summary["mean"]) / single_summary["mean"] * 100
        ),
    }

    json_path = out_dir / "eval_envD_chunk_horizon.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csvs(out_dir, single_summary, multi_summary)
    maybe_plot(Path(args.plot_dir), single_summary, multi_summary)

    print("\n===== Summary =====")
    print(f"A-only mean L1: {single_summary['mean']:.4f}")
    print(f"ABC mean L1:    {multi_summary['mean']:.4f}")
    print(f"Improvement:    {result['relative_improvement_pct']:+.1f}%")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {out_dir / 'eval_envD_per_horizon.csv'}")
    print(f"Wrote: {out_dir / 'eval_envD_per_dim.csv'}")


if __name__ == "__main__":
    main()
