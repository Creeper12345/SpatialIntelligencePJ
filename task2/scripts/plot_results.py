#!/usr/bin/env python3
"""根据 Task 2 的 CSV/JSON 结果生成报告图。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def plot_per_dim(log_dir: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = read_csv(log_dir / 'eval_envD_per_dim.csv')
    dims = [r['action_dim'] for r in rows]
    single = [float(r['single_A_l1']) for r in rows]
    multi = [float(r['multi_ABC_l1']) for r in rows]

    x = np.arange(len(dims))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - width / 2, single, width, label='A-only', color='#4C78A8')
    ax.bar(x + width / 2, multi, width, label='ABC', color='#F58518')
    ax.set_xticks(x, dims)
    ax.set_ylabel('Action L1 on Env D')
    ax.set_title('Zero-shot Action Error by Dimension')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / 'envD_per_dim_l1.png', dpi=220)
    fig.savefig(out_dir / 'envD_per_dim_l1.pdf')
    plt.close(fig)


def plot_horizon(log_dir: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = read_csv(log_dir / 'eval_envD_per_horizon.csv')
    steps = [int(r['horizon_step']) for r in rows]
    single = [float(r['single_A_l1']) for r in rows]
    multi = [float(r['multi_ABC_l1']) for r in rows]
    impr = [float(r['relative_improvement_pct']) for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(steps, single, marker='o', label='A-only', color='#4C78A8')
    ax.plot(steps, multi, marker='o', label='ABC', color='#F58518')
    ax.set_xlabel('Action chunk step')
    ax.set_ylabel('Action L1 on Env D')
    ax.set_title('Zero-shot Error Across ACT Chunk Horizon')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / 'envD_chunk_horizon_l1.png', dpi=220)
    fig.savefig(out_dir / 'envD_chunk_horizon_l1.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(steps, impr, marker='o', color='#54A24B')
    ax.set_xlabel('Action chunk step')
    ax.set_ylabel('ABC relative improvement (%)')
    ax.set_title('ABC Improvement Grows Along Chunk Horizon')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / 'envD_chunk_improvement.png', dpi=220)
    fig.savefig(out_dir / 'envD_chunk_improvement.pdf')
    plt.close(fig)



def plot_pretrained_vs_abc(log_dir: Path, out_dir: Path) -> None:
    csv_path = log_dir / 'eval_envD_pretrainedA_vs_ABC_per_dim.csv'
    if not csv_path.exists():
        return

    import matplotlib.pyplot as plt
    import numpy as np

    rows = read_csv(csv_path)
    dims = [r['action_dim'] for r in rows]
    pretrained = [float(r['pretrained_A_l1']) for r in rows]
    multi = [float(r['multi_ABC_l1']) for r in rows]

    x = np.arange(len(dims))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - width / 2, pretrained, width, label='A-only + ImageNet init', color='#72B7B2')
    ax.bar(x + width / 2, multi, width, label='ABC', color='#F58518')
    ax.set_xticks(x, dims)
    ax.set_ylabel('Action L1 on Env D')
    ax.set_title('Pretrained A-only vs ABC Zero-shot Error')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / 'envD_pretrainedA_vs_ABC_per_dim_l1.png', dpi=220)
    fig.savefig(out_dir / 'envD_pretrainedA_vs_ABC_per_dim_l1.pdf')
    plt.close(fig)

def write_summary(log_dir: Path, out_dir: Path) -> None:
    result_path = log_dir / 'eval_envD_chunk_horizon.json'
    if not result_path.exists():
        return
    result = json.loads(result_path.read_text(encoding='utf-8'))
    lines = [
        '# Task 2 Figure Summary',
        '',
        f"A-only mean L1: {result['single_A']['mean']:.4f}",
        f"ABC mean L1: {result['multi_ABC']['mean']:.4f}",
        f"Relative improvement: {result['relative_improvement_pct']:+.1f}%",
        '',
        'Generated figures:',
        '- envD_per_dim_l1.png/pdf',
        '- envD_chunk_horizon_l1.png/pdf',
        '- envD_chunk_improvement.png/pdf',
        '- envD_pretrainedA_vs_ABC_per_dim_l1.png/pdf, if pretrained eval exists',
    ]
    (out_dir / 'figure_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--log_dir', default='logs')
    p.add_argument('--out_dir', default='plots')
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_per_dim(log_dir, out_dir)
    plot_horizon(log_dir, out_dir)
    plot_pretrained_vs_abc(log_dir, out_dir)
    write_summary(log_dir, out_dir)
    print(f'Wrote figures to {out_dir}')


if __name__ == '__main__':
    main()
