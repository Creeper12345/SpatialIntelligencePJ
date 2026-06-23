#!/usr/bin/env python3
"""预下载并校验 ACT backbone 使用的 torchvision ResNet18 权重。"""

import argparse
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights, resnet18


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default=None, help="Optional TORCH_HOME directory")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if args.cache_dir:
        torch.hub.set_dir(str(Path(args.cache_dir).expanduser().resolve() / "hub"))
    print("torch hub dir:", torch.hub.get_dir())
    print("downloading/verifying ResNet18_Weights.IMAGENET1K_V1 ...", flush=True)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(args.device)
    model.eval()
    print("OK: resnet18 weights loaded")


if __name__ == "__main__":
    main()
