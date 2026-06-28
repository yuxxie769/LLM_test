from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.benchmark_backends.vllm_bench import normalize_benchmark_result
from bench.config import load_settings


def find_combined_files(raw_benchmark_dir: Path) -> list[Path]:
    return sorted(raw_benchmark_dir.glob("*/*.combined.json"))


def filter_combined_files(
    paths: list[Path],
    batch_run_id: str | None,
) -> list[Path]:
    if batch_run_id is None:
        return paths
    needle = f"/{batch_run_id}/"
    return [path for path in paths if needle in f"/{path.as_posix()}"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_benchmark_row(payload: dict) -> dict:
    case = payload["case"]
    benchmark = payload["benchmark"]
    benchmark_result_path = payload.get("benchmark_result_path")
    if benchmark_result_path:
        raw_payload = json.loads(Path(benchmark_result_path).read_text(encoding="utf-8"))
        benchmark = normalize_benchmark_result(raw_payload)
    return {
        "batch_run_id": payload["batch_run_id"],
        "case_id": payload["case_id"],
        "suite": case["suite"],
        "mode": case["mode"],
        "repeat_index": case["repeat_index"],
        "concurrency": case["concurrency"],
        "input_tokens": case["input_tokens"],
        "output_tokens": case["output_tokens"],
        "duration_s": benchmark["duration_s"],
        "completed": benchmark["completed"],
        "failed": benchmark["failed"],
        "request_throughput_qps": benchmark["request_throughput_qps"],
        "output_throughput_tps": benchmark["output_throughput_tps"],
        "total_token_throughput_tps": benchmark["total_token_throughput_tps"],
        "mean_ttft_ms": benchmark["mean_ttft_ms"],
        "median_ttft_ms": benchmark["median_ttft_ms"],
        "p95_ttft_ms": benchmark["ttft_percentiles_ms"].get("p95", 0.0),
        "mean_tpot_ms": benchmark["mean_tpot_ms"],
        "median_tpot_ms": benchmark["median_tpot_ms"],
        "p95_tpot_ms": benchmark["tpot_percentiles_ms"].get("p95", 0.0),
        "mean_itl_ms": benchmark["mean_itl_ms"],
        "median_itl_ms": benchmark["median_itl_ms"],
        "p95_itl_ms": benchmark["itl_percentiles_ms"].get("p95", 0.0),
        "mean_e2el_ms": benchmark["mean_e2el_ms"],
        "median_e2el_ms": benchmark["median_e2el_ms"],
        "p95_e2el_ms": benchmark["e2el_percentiles_ms"].get("p95", 0.0),
        "error_rate": (
            benchmark["failed"] / max(benchmark["completed"] + benchmark["failed"], 1)
        ),
    }


def flatten_service_row(payload: dict) -> dict:
    case = payload["case"]
    service = payload["service"]
    return {
        "batch_run_id": payload["batch_run_id"],
        "case_id": payload["case_id"],
        "suite": case["suite"],
        "mode": case["mode"],
        "repeat_index": case["repeat_index"],
        "concurrency": case["concurrency"],
        "input_tokens": case["input_tokens"],
        "output_tokens": case["output_tokens"],
        "duration_s": service["duration_s"],
        "prompt_tokens_delta": service["prompt_tokens_delta"],
        "generation_tokens_delta": service["generation_tokens_delta"],
        "request_success_delta": service["request_success_delta"],
        "prompt_throughput_toks_per_s": service["prompt_throughput_toks_per_s"],
        "generation_throughput_toks_per_s": service[
            "generation_throughput_toks_per_s"
        ],
        "kv_cache_usage_perc_before": service["kv_cache_usage_perc_before"],
        "kv_cache_usage_perc_after": service["kv_cache_usage_perc_after"],
        "num_requests_running_before": service["num_requests_running_before"],
        "num_requests_running_after": service["num_requests_running_after"],
        "num_requests_waiting_before": service["num_requests_waiting_before"],
        "num_requests_waiting_after": service["num_requests_waiting_after"],
        "gpu_memory_used_mb_before": service["gpu_memory_used_mb_before"],
        "gpu_memory_used_mb_after": service["gpu_memory_used_mb_after"],
        "gpu_memory_used_mb_delta": service["gpu_memory_used_mb_delta"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phase 2 benchmark results.")
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Override raw benchmark directory. Defaults to results/raw/benchmark.",
    )
    parser.add_argument(
        "--batch-run-id",
        default=None,
        help="Only aggregate results for the specified batch_run_id.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV outputs. Defaults to the repo results directory.",
    )
    args = parser.parse_args()

    settings = load_settings()
    raw_dir = Path(args.raw_dir) if args.raw_dir else settings.raw_benchmark_dir
    output_dir = Path(args.output_dir) if args.output_dir else settings.results_dir
    combined_files = filter_combined_files(
        find_combined_files(raw_dir),
        args.batch_run_id,
    )

    benchmark_rows: list[dict] = []
    service_rows: list[dict] = []
    for path in combined_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        benchmark_rows.append(flatten_benchmark_row(payload))
        service_rows.append(flatten_service_row(payload))

    write_csv(output_dir / "baseline_metrics.csv", benchmark_rows)
    write_csv(output_dir / "baseline_service_metrics.csv", service_rows)
    print(
        json.dumps(
            {
                "batch_run_id": args.batch_run_id,
                "combined_files": len(combined_files),
                "baseline_metrics_csv": str(output_dir / "baseline_metrics.csv"),
                "baseline_service_metrics_csv": str(output_dir / "baseline_service_metrics.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
