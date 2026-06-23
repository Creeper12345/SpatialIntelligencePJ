"""Task 2 使用的本地 CALVIN v2.1 Parquet 数据集。

服务器上的 xiaoma26/calvin-lerobot 数据为本地离线文件。这里直接读取
v2.1 parquet episode，并返回 ACTPolicy 需要的字段，避免 LeRobotDataset
访问 HuggingFace 元数据或强制使用 v3.0 数据格式。
"""

from __future__ import annotations

import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

ENV_TO_SUBSET = {
    "A": "splitA",
    "B": "splitB",
    "C": "splitC",
    "D": "splitD",
}

HF_REPO_ID = "xiaoma26/calvin-lerobot"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class CalvinParquetDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        action_horizon: int = 10,
        use_wrist_cam: bool = False,
        image_size: int = 256,
        cache_size: int = 8,
    ):
        self.root = Path(root)
        self.action_horizon = action_horizon
        self.use_wrist_cam = use_wrist_cam
        self.image_size = image_size
        self.cache_size = cache_size
        self._cache: OrderedDict[Path, dict] = OrderedDict()

        meta_path = self.root / "meta" / "episodes.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}")

        self.samples: list[tuple[Path, int]] = []
        for ep in _read_jsonl(meta_path):
            ep_idx = int(ep["episode_index"])
            length = int(ep["length"])
            path = self.root / "data" / f"chunk-{ep_idx // 1000:03d}" / f"episode_{ep_idx:06d}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing parquet for episode {ep_idx}: {path}")
            usable = max(0, length - action_horizon + 1)
            for frame_offset in range(usable):
                self.samples.append((path, frame_offset))

        if not self.samples:
            raise RuntimeError(f"No usable samples in {self.root}; action_horizon={action_horizon}")

    def __len__(self) -> int:
        return len(self.samples)

    def _episode(self, path: Path) -> dict:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached

        table = pq.read_table(path).to_pydict()
        self._cache[path] = table
        self._cache.move_to_end(path)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return table

    def _image_tensor(self, image_entry: dict) -> torch.Tensor:
        data = image_entry.get("bytes")
        if data is None:
            img_path = image_entry.get("path")
            if img_path is None:
                raise ValueError("Image entry has neither bytes nor path")
            image = Image.open(self.root / img_path).convert("RGB")
        else:
            image = Image.open(io.BytesIO(data)).convert("RGB")

        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, offset = self.samples[idx]
        ep = self._episode(path)

        image = self._image_tensor(ep["image"][offset]).unsqueeze(0)
        state = torch.tensor(ep["state"][offset], dtype=torch.float32).unsqueeze(0)
        actions = torch.tensor(
            ep["actions"][offset : offset + self.action_horizon],
            dtype=torch.float32,
        )

        batch = {
            "observation.images.image": image,
            "observation.state": state,
            "action": actions,
        }
        if self.use_wrist_cam:
            batch["observation.images.wrist_image"] = self._image_tensor(ep["wrist_image"][offset]).unsqueeze(0)
        return batch


def build_calvin_dataset(
    data_dir: str,
    envs: List[str],
    split: str = "train",
    obs_horizon: int = 1,
    action_horizon: int = 10,
    fps: int = 30,
    use_wrist_cam: bool = False,
    delta_timestamps: dict | None = None,
) -> Dataset | ConcatDataset:
    del split, obs_horizon, fps, delta_timestamps
    data_dir = Path(data_dir)
    datasets = []
    for env in envs:
        if env not in ENV_TO_SUBSET:
            raise ValueError(f"Unknown env {env}; choices={list(ENV_TO_SUBSET)}")
        root = data_dir / ENV_TO_SUBSET[env]
        if not root.exists():
            raise FileNotFoundError(f"Missing split directory: {root}")
        ds = CalvinParquetDataset(root, action_horizon=action_horizon, use_wrist_cam=use_wrist_cam)
        print(f"[CalvinParquetDataset] env={env} samples={len(ds)} root={root}", flush=True)
        datasets.append(ds)
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    ds = build_calvin_dataset(data_dir, envs=["A"])
    sample = ds[0]
    print("len", len(ds))
    for k, v in sample.items():
        print(k, tuple(v.shape), v.dtype)
