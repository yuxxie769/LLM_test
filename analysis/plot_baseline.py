from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.config import load_settings


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # noqa: PERF203
        raise SystemExit(
            "matplotlib is not installed. Install it with "
            "`python -m pip install matplotlib seaborn`."
        ) from exc
    return plt


def _has_nonempty_metric(rows: list[dict[str, str]], metric_key: str) -> bool:
    return any(row.get(metric_key) not in (None, "") for row in rows)


def grouped_series(rows: list[dict[str, str]], metric_key: str) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        raw_value = row.get(metric_key)
        if raw_value in (None, ""):
            continue
        label = f"in{row['input_tokens']}_out{row['output_tokens']}"
        series[label].append((int(row["concurrency"]), float(raw_value)))
    for points in series.values():
        points.sort(key=lambda item: item[0])
    return dict(series)


def plot_metric(
    *,
    rows: list[dict[str, str]],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt = ensure_matplotlib()
    series_by_label = grouped_series(rows, metric_key)
    if not series_by_label:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, points in series_by_label.items():
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline benchmark results.")
    parser.add_argument(
        "--benchmark-csv",
        default="results/baseline_metrics.csv",
    )
    parser.add_argument(
        "--service-csv",
        default="results/baseline_service_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated plot files. Defaults to results/plots.",
    )
    args = parser.parse_args()

    settings = load_settings()
    benchmark_rows = load_rows(settings.repo_root / args.benchmark_csv)
    service_rows = load_rows(settings.repo_root / args.service_csv)
    plots_dir = Path(args.output_dir) if args.output_dir else settings.plots_dir
    if not benchmark_rows:
        raise SystemExit("No benchmark rows found. Run aggregation first.")
    if not service_rows:
        raise SystemExit("No service rows found. Run aggregation first.")

    plot_metric(
        rows=benchmark_rows,
        metric_key="request_throughput_qps",
        title="Concurrency vs QPS",
        ylabel="QPS",
        output_path=plots_dir / "baseline_qps.png",
    )
    plot_metric(
        rows=benchmark_rows,
        metric_key="p95_e2el_ms",
        title="Concurrency vs P95 Latency",
        ylabel="P95 Latency (ms)",
        output_path=plots_dir / "baseline_p95_latency.png",
    )

    waiting_metric_key = (
        "num_requests_waiting_during_run_max"
        if _has_nonempty_metric(service_rows, "num_requests_waiting_during_run_max")
        else "num_requests_waiting_after"
    )
    waiting_title = (
        "Concurrency vs Max Sampled Waiting Requests"
        if waiting_metric_key == "num_requests_waiting_during_run_max"
        else "Concurrency vs Waiting Requests"
    )
    waiting_ylabel = (
        "Max Sampled Waiting Requests"
        if waiting_metric_key == "num_requests_waiting_during_run_max"
        else "Waiting Requests"
    )
    plot_metric(
        rows=service_rows,
        metric_key=waiting_metric_key,
        title=waiting_title,
        ylabel=waiting_ylabel,
        output_path=plots_dir / "baseline_waiting_requests.png",
    )

    if _has_nonempty_metric(service_rows, "kv_cache_usage_perc_during_run_max"):
        plot_metric(
            rows=service_rows,
            metric_key="kv_cache_usage_perc_during_run_max",
            title="Concurrency vs Max Sampled KV Cache Usage",
            ylabel="Max Sampled KV Cache Usage (%)",
            output_path=plots_dir / "baseline_kv_cache_usage.png",
        )

    if _has_nonempty_metric(service_rows, "server_load_during_run_max"):
        plot_metric(
            rows=service_rows,
            metric_key="server_load_during_run_max",
            title="Concurrency vs Max Sampled Server Load",
            ylabel="Max Sampled Server Load",
            output_path=plots_dir / "baseline_server_load.png",
        )


if __name__ == "__main__":
    main()
