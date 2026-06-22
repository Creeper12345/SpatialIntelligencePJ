#!/usr/bin/env python3
"""校验本地 xiaoma26/calvin-lerobot 数据集。

该脚本只检查本地文件，不访问 HuggingFace。检查内容包括目录结构、
元数据解析、Parquet 可读性、episode 文件覆盖、空文件、Git-LFS 指针
以及可选的 SHA256 manifest。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

SPLITS = ["splitA", "splitB", "splitC", "splitD"]
REQUIRED_META = [
    "info.json",
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "tasks.jsonl",
    "modality.json",
]
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


@dataclass
class SplitReport:
    name: str
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files: int = 0
    bytes: int | None = None
    parquet_files: int = 0
    parquet_rows: int = 0
    episodes_meta: int = 0
    expected_episodes: int | None = None

    def error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSONL: {exc}") from exc
    return rows


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(len(LFS_POINTER_PREFIX))
        return head == LFS_POINTER_PREFIX
    except OSError:
        return False


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> dict[str, dict[str, int | str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "files" not in data:
        raise ValueError("manifest must be a JSON object with key 'files'")
    return data["files"]


def write_manifest(data_dir: Path, manifest_path: Path) -> None:
    files: dict[str, dict[str, int | str]] = {}
    for path in sorted(iter_files(data_dir)):
        if "/.cache/" in path.as_posix() or path.relative_to(data_dir).parts[0] == ".cache":
            continue
        rel = path.relative_to(data_dir).as_posix()
        stat = path.stat()
        files[rel] = {"size": stat.st_size, "sha256": sha256_file(path)}
    payload = {"root": str(data_dir), "files": files}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Wrote manifest: {manifest_path} ({len(files)} files)")


def verify_manifest(data_dir: Path, manifest_path: Path) -> bool:
    expected = load_manifest(manifest_path)
    ok = True
    seen: set[str] = set()
    for rel, meta in expected.items():
        seen.add(rel)
        path = data_dir / rel
        if not path.exists():
            print(f"ERROR manifest missing file: {rel}")
            ok = False
            continue
        size = path.stat().st_size
        if size != meta.get("size"):
            print(f"ERROR manifest size mismatch: {rel}: local={size} expected={meta.get('size')}")
            ok = False
            continue
        digest = sha256_file(path)
        if digest != meta.get("sha256"):
            print(f"ERROR manifest sha256 mismatch: {rel}")
            ok = False
    current = {
        p.relative_to(data_dir).as_posix()
        for p in iter_files(data_dir)
        if p.relative_to(data_dir).parts[0] != ".cache"
    }
    for rel in sorted(current - seen):
        print(f"WARNING manifest extra local file: {rel}")
    return ok


def validate_parquet(path: Path, deep: bool) -> tuple[int, list[str]]:
    problems: list[str] = []
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("pyarrow is required for parquet validation: pip install pyarrow") from exc

    pf = pq.ParquetFile(path)
    rows = pf.metadata.num_rows
    if pf.metadata.num_row_groups <= 0:
        problems.append("no row groups")
    if rows <= 0:
        problems.append("no rows")
    if deep:
        # 读取完整表可发现部分截断或损坏的 Parquet 文件。
        pf.read()
    return rows, problems


def validate_split(split_dir: Path, check_parquet: bool, deep_parquet: bool, max_parquet: int | None) -> SplitReport:
    report = SplitReport(name=split_dir.name)
    if not split_dir.exists():
        report.error(f"missing split directory: {split_dir}")
        return report

    meta_dir = split_dir / "meta"
    data_dir = split_dir / "data"
    if not meta_dir.is_dir():
        report.error("missing meta/ directory")
    if not data_dir.is_dir():
        report.error("missing data/ directory")

    for name in REQUIRED_META:
        path = meta_dir / name
        if not path.exists():
            report.error(f"missing meta/{name}")

    info = None
    if (meta_dir / "info.json").exists():
        try:
            info = read_json(meta_dir / "info.json")
            for key in ["total_episodes", "total_frames", "features"]:
                if key not in info:
                    report.warn(f"info.json lacks key: {key}")
            if isinstance(info, dict) and isinstance(info.get("total_episodes"), int):
                report.expected_episodes = info["total_episodes"]
        except Exception as exc:
            report.error(f"cannot parse meta/info.json: {exc}")

    episodes = []
    if (meta_dir / "episodes.jsonl").exists():
        try:
            episodes = read_jsonl(meta_dir / "episodes.jsonl")
            report.episodes_meta = len(episodes)
        except Exception as exc:
            report.error(f"cannot parse meta/episodes.jsonl: {exc}")

    for name in ["episodes_stats.jsonl", "tasks.jsonl"]:
        path = meta_dir / name
        if path.exists():
            try:
                read_jsonl(path)
            except Exception as exc:
                report.error(f"cannot parse meta/{name}: {exc}")

    for name in ["modality.json", "conversion.json"]:
        path = meta_dir / name
        if path.exists():
            try:
                read_json(path)
            except Exception as exc:
                report.error(f"cannot parse meta/{name}: {exc}")

    meta_files = list(meta_dir.glob("*")) if meta_dir.exists() else []
    parquet_files: list[Path] = []
    missing_parquet = 0
    if data_dir.exists() and episodes:
        for ep in episodes:
            idx = ep.get("episode_index")
            if not isinstance(idx, int):
                report.error(f"episodes.jsonl row lacks integer episode_index: {ep}")
                continue
            path = data_dir / f"chunk-{idx // 1000:03d}" / f"episode_{idx:06d}.parquet"
            if path.exists():
                parquet_files.append(path)
            else:
                missing_parquet += 1
                if missing_parquet <= 10:
                    report.error(f"missing parquet for episode_index={idx}: {path.relative_to(split_dir)}")
        if missing_parquet > 10:
            report.error(f"missing parquet files not fully listed: total_missing={missing_parquet}")
    elif data_dir.exists():
        # 仅在 episodes.jsonl 缺失或损坏时回退扫描，速度会更慢。
        parquet_files = sorted(data_dir.glob("chunk-*/*.parquet"))

    report.files = len(parquet_files) + sum(1 for p in meta_files if p.is_file())
    report.parquet_files = len(parquet_files)

    for path in meta_files:
        if path.is_file() and path.stat().st_size == 0:
            report.error(f"empty file: {path.relative_to(split_dir)}")
        if path.is_file() and is_lfs_pointer(path):
            report.error(f"Git-LFS pointer instead of real file: {path.relative_to(split_dir)}")

    for path in parquet_files[: max_parquet or len(parquet_files)]:
        if path.stat().st_size == 0:
            report.error(f"empty file: {path.relative_to(split_dir)}")
        if is_lfs_pointer(path):
            report.error(f"Git-LFS pointer instead of real file: {path.relative_to(split_dir)}")
    if not parquet_files:
        report.error("no parquet files under data/")

    if report.expected_episodes is not None and parquet_files:
        if len(parquet_files) != report.expected_episodes:
            report.error(
                f"parquet episode count mismatch: files={len(parquet_files)} "
                f"info.total_episodes={report.expected_episodes}"
            )
    if episodes and parquet_files and len(episodes) != len(parquet_files):
        report.warn(f"episodes.jsonl rows={len(episodes)} but parquet files={len(parquet_files)}")

    if check_parquet:
        to_check = parquet_files if max_parquet is None else parquet_files[:max_parquet]
        for path in to_check:
            try:
                rows, problems = validate_parquet(path, deep=deep_parquet)
                report.parquet_rows += rows
                for problem in problems:
                    report.error(f"bad parquet {path.relative_to(split_dir)}: {problem}")
            except Exception as exc:
                report.error(f"cannot read parquet {path.relative_to(split_dir)}: {exc}")

        if max_parquet is not None and len(parquet_files) > max_parquet:
            report.warn(f"parquet check sampled {max_parquet}/{len(parquet_files)} files")
    else:
        report.warn("parquet internal check skipped; add --check-parquet to read parquet footers")

    return report


def print_report(report: SplitReport) -> None:
    status = "OK" if report.ok else "FAIL"
    size_part = "" if report.bytes is None else f" size={report.bytes / (1024 * 1024):.1f}MiB"
    print(
        f"[{status}] {report.name}: files={report.files}{size_part} "
        f"parquet={report.parquet_files} parquet_rows_checked={report.parquet_rows} "
        f"episodes_meta={report.episodes_meta} expected_episodes={report.expected_episodes}"
    )
    for msg in report.errors:
        print(f"  ERROR: {msg}")
    for msg in report.warnings:
        print(f"  WARNING: {msg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local CALVIN LeRobot dataset files.")
    parser.add_argument("data_dir", nargs="?", default="./data", type=Path)
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--check-parquet", action="store_true", help="read parquet footers to detect corrupt/truncated parquet files")
    parser.add_argument("--deep-parquet", action="store_true", help="read full parquet tables, slower but stronger")
    parser.add_argument("--max-parquet", type=int, default=None, help="sample only first N parquet files per split")
    parser.add_argument("--write-manifest", type=Path, default=None, help="write SHA256 manifest for all dataset files")
    parser.add_argument("--verify-manifest", type=Path, default=None, help="verify files against a prior SHA256 manifest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        print(f"ERROR data_dir does not exist: {data_dir}")
        return 2

    if args.verify_manifest is not None:
        ok = verify_manifest(data_dir, args.verify_manifest)
        if not ok:
            return 1

    print(f"Dataset root: {data_dir}", flush=True)
    reports = []
    for split in args.splits:
        print(f"Checking {split} ...", flush=True)
        report = validate_split(data_dir / split, args.check_parquet, args.deep_parquet, args.max_parquet)
        reports.append(report)
        print_report(report)

    ok = all(report.ok for report in reports)
    if args.write_manifest is not None:
        write_manifest(data_dir, args.write_manifest)

    if ok:
        print("RESULT: OK")
        return 0
    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
