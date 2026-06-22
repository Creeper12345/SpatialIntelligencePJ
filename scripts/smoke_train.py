#!/usr/bin/env python3
"""Task 2 数据集和模型的最小端到端冒烟测试。"""

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

TASK2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK2_ROOT))
sys.path.insert(0, str(TASK2_ROOT / "src"))

from calvin_lerobot.dataset import build_calvin_dataset
from train_act import ACTPolicy, build_act_config, run_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--envs", nargs="+", default=["A"])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--kl_weight", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--use_wrist_cam", action="store_true")
    parser.add_argument("--pretrained_backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = build_calvin_dataset(
        args.data_dir,
        envs=args.envs,
        action_horizon=args.chunk_size,
        fps=args.fps,
        use_wrist_cam=args.use_wrist_cam,
    )
    n = min(args.samples, len(ds))
    loader = DataLoader(Subset(ds, list(range(n))), batch_size=args.batch_size, shuffle=False, num_workers=0)
    first = next(iter(loader))

    print("Device:", device)
    print("Dataset length:", len(ds), "smoke samples:", n)
    for key, value in first.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")

    model = ACTPolicy(build_act_config(args)).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.95))
    loss, l1 = run_epoch(model, loader, optimizer, device, train=True, max_steps=args.steps)
    print(f"Smoke train OK: loss={loss:.6f} l1={l1:.6f}")


if __name__ == "__main__":
    main()
