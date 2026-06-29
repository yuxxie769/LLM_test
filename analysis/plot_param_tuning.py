from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.phase3_common import load_csv_rows, resolve_repo_path, safe_float  # noqa: E402
from bench.config import load_settings  # noqa: E402


def ensure_matplotlib():
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit('matplotlib is not installed. Install it with `python -m pip install matplotlib`.') from exc
    return plt


def plot_target(rows: list[dict[str, str]], metric: str, title: str, ylabel: str, output_path: Path) -> None:
    plt = ensure_matplotlib()
    ordered = sorted(rows, key=lambda row: safe_float(row['tuning_value']))
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot([safe_float(row['tuning_value']) for row in ordered], [safe_float(row[metric]) for row in ordered], marker='o')
    ax.set_title(title)
    ax.set_xlabel('Tuning Value')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot parameter tuning metrics.')
    parser.add_argument('--csv', default='results/param_tuning/param_tuning.csv')
    parser.add_argument('--output-dir', default='results/param_tuning/plots')
    args = parser.parse_args()

    settings = load_settings()
    rows = load_csv_rows(resolve_repo_path(settings.repo_root, args.csv))
    if not rows:
        raise SystemExit('No tuning rows found. Run aggregation first.')

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row['tuning_target']].append(row)

    output_dir = resolve_repo_path(settings.repo_root, args.output_dir)
    for target, bucket in grouped.items():
        plot_target(bucket, 'request_throughput_qps', f'{target}: QPS', 'QPS', output_dir / f'{target}_qps.png')
        plot_target(bucket, 'p95_e2el_ms', f'{target}: P95 Latency', 'P95 Latency (ms)', output_dir / f'{target}_p95_latency.png')
        plot_target(bucket, 'gpu_memory_used_mb_after', f'{target}: GPU Memory After', 'GPU Memory After (MB)', output_dir / f'{target}_gpu_memory.png')


if __name__ == '__main__':
    main()
