from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.config import load_settings


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Markdown baseline summary.")
    parser.add_argument("--benchmark-csv", default="results/baseline_metrics.csv")
    parser.add_argument("--service-csv", default="results/baseline_service_metrics.csv")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Write summary markdown to this path. Defaults to results/baseline_summary.md.",
    )
    args = parser.parse_args()

    settings = load_settings()
    benchmark_rows = load_rows(settings.repo_root / args.benchmark_csv)
    service_rows = load_rows(settings.repo_root / args.service_csv)

    if not benchmark_rows:
        raise SystemExit("No benchmark rows found. Run aggregation first.")

    batch_run_ids = sorted({row["batch_run_id"] for row in benchmark_rows})
    peak_qps_row = max(benchmark_rows, key=lambda row: _safe_float(row["request_throughput_qps"]))
    worst_latency_row = max(benchmark_rows, key=lambda row: _safe_float(row["p95_e2el_ms"]))
    worst_waiting_row = max(service_rows, key=lambda row: _safe_float(row["num_requests_waiting_after"])) if service_rows else None

    summary = [
        "# Baseline Summary",
        "",
        "## Environment",
        "",
        f"- Base URL: `{settings.vllm_base_url}`",
        f"- Model dir: `{settings.model_dir}`",
        f"- Served model name: `{settings.served_model_name}`",
        f"- Batch run id(s): `{', '.join(batch_run_ids)}`",
        "",
        "## Findings",
        "",
        (
            f"- Peak request throughput appears at concurrency `{peak_qps_row['concurrency']}` "
            f"with input `{peak_qps_row['input_tokens']}` and output `{peak_qps_row['output_tokens']}`: "
            f"`{peak_qps_row['request_throughput_qps']}` req/s."
        ),
        (
            f"- Highest observed P95 latency appears at concurrency `{worst_latency_row['concurrency']}` "
            f"with input `{worst_latency_row['input_tokens']}` and output `{worst_latency_row['output_tokens']}`: "
            f"`{worst_latency_row['p95_e2el_ms']}` ms."
        ),
    ]

    if worst_waiting_row is not None:
        summary.append(
            (
                f"- Largest waiting queue snapshot appears at concurrency "
                f"`{worst_waiting_row['concurrency']}` with "
                f"`{worst_waiting_row['num_requests_waiting_after']}` waiting requests, "
                "which is a useful service-side explanation point for tail-latency growth."
            )
        )

    output_path = Path(args.output_path) if args.output_path else settings.results_dir / "baseline_summary.md"

    summary.extend(
        [
            "",
            "## Notes",
            "",
            "- This summary is generated from aggregated benchmark outputs and service-side snapshots.",
            f"- Review plots and CSV files in `{output_path.parent}` together with the raw JSON files under `results/raw/`.",
        ]
    )
    output_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
