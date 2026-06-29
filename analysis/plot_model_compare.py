from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.phase3_common import load_csv_rows, resolve_repo_path, safe_float, safe_int  # noqa: E402
from bench.config import load_settings  # noqa: E402


def ensure_matplotlib():
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit('matplotlib is not installed. Install it with `python -m pip install matplotlib`.') from exc
    return plt


def pick_focus_combo(rows: list[dict[str, str]]) -> tuple[int, int]:
    preferred = (512, 128)
    combos: dict[tuple[int, int], set[int]] = defaultdict(set)
    for row in rows:
        key = (safe_int(row['input_tokens']), safe_int(row['output_tokens']))
        combos[key].add(safe_int(row['concurrency']))
    if preferred in combos:
        return preferred
    return sorted(combos, key=lambda key: (-len(combos[key]), key[0], key[1]))[0]


def plot_metric(rows: list[dict[str, str]], metric: str, title: str, ylabel: str, output_path: Path) -> None:
    plt = ensure_matplotlib()
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        grouped[row['model_label']].append((safe_int(row['concurrency']), safe_float(row[metric])))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, points in grouped.items():
        points.sort(key=lambda item: item[0])
        ax.plot([point[0] for point in points], [point[1] for point in points], marker='o', label=label)
    ax.set_title(title)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot model compare metrics for the shared Phase 2 workload slice.')
    parser.add_argument('--long-csv', default='results/model_compare/model_compare_long.csv')
    parser.add_argument('--output-dir', default='results/model_compare/plots')
    args = parser.parse_args()

    settings = load_settings()
    rows = load_csv_rows(resolve_repo_path(settings.repo_root, args.long_csv))
    if not rows:
        raise SystemExit('No compare rows found. Run aggregation first.')

    focus_input, focus_output = pick_focus_combo(rows)
    focus_rows = [
        row for row in rows
        if safe_int(row['input_tokens']) == focus_input and safe_int(row['output_tokens']) == focus_output
    ]
    output_dir = resolve_repo_path(settings.repo_root, args.output_dir)

    plot_metric(focus_rows, 'request_throughput_qps', f'Model Compare QPS (input={focus_input}, output={focus_output})', 'QPS', output_dir / 'model_compare_qps.png')
    plot_metric(focus_rows, 'p95_e2el_ms', f'Model Compare P95 Latency (input={focus_input}, output={focus_output})', 'P95 Latency (ms)', output_dir / 'model_compare_p95_latency.png')
    plot_metric(focus_rows, 'gpu_memory_used_mb_after', f'Model Compare GPU Memory (input={focus_input}, output={focus_output})', 'GPU Memory After (MB)', output_dir / 'model_compare_gpu_memory.png')


if __name__ == '__main__':
    main()
