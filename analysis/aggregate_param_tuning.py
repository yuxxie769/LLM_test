from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.aggregate_results import flatten_benchmark_row, flatten_service_row  # noqa: E402
from analysis.phase3_common import resolve_repo_path, safe_float, write_csv  # noqa: E402
from bench.config import load_settings  # noqa: E402


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [safe_float(row.get(key)) for row in rows if row.get(key) not in (None, '')]
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_entry(entry: dict[str, Any], settings) -> dict[str, Any]:
    combined_files = sorted((settings.raw_benchmark_dir / entry['batch_run_id']).glob('*.combined.json'))
    if not combined_files:
        raise FileNotFoundError(f"No combined results found for {entry['batch_run_id']}")

    benchmark_rows = []
    service_rows = []
    for path in combined_files:
        payload = json.loads(path.read_text(encoding='utf-8'))
        benchmark_rows.append(flatten_benchmark_row(payload))
        service_rows.append(flatten_service_row(payload))

    workload = entry['workload']
    params = entry['parameters']
    startup = entry.get('startup_diagnostics') or {}
    startup_log = startup.get('vllm_startup_log') or {}
    return {
        'experiment_family': entry.get('experiment_family', 'single_variable'),
        'tuning_target': entry['tuning_target'],
        'tuning_value': entry['tuning_value'],
        'tuning_label': entry.get('tuning_label', ''),
        'batch_run_id': entry['batch_run_id'],
        'warmup_batch_run_id': entry.get('warmup_batch_run_id', ''),
        'warmup_repeat_count': int(entry.get('warmup_repeat_count', workload.get('warmup_repeats', 0))),
        'repeat_count': len(combined_files),
        'measured_repeat_count': int(entry.get('measured_repeat_count', workload.get('repeat', len(combined_files)))),
        'concurrency': workload['concurrency'],
        'input_tokens': workload['input_tokens'],
        'output_tokens': workload['output_tokens'],
        'max_model_len': params['max_model_len'],
        'max_num_batched_tokens': params['max_num_batched_tokens'],
        'max_num_seqs': params['max_num_seqs'],
        'gpu_memory_utilization': params['gpu_memory_utilization'],
        'request_throughput_qps': mean_metric(benchmark_rows, 'request_throughput_qps'),
        'p50_e2el_ms': mean_metric(benchmark_rows, 'median_e2el_ms'),
        'p95_e2el_ms': mean_metric(benchmark_rows, 'p95_e2el_ms'),
        'output_throughput_tps': mean_metric(benchmark_rows, 'output_throughput_tps'),
        'error_rate': mean_metric(benchmark_rows, 'error_rate'),
        'mean_ttft_ms': mean_metric(benchmark_rows, 'mean_ttft_ms'),
        'mean_tpot_ms': mean_metric(benchmark_rows, 'mean_tpot_ms'),
        'num_requests_waiting_after': mean_metric(service_rows, 'num_requests_waiting_after'),
        'num_requests_waiting_peak': mean_metric(service_rows, 'num_requests_waiting_during_run_max') or mean_metric(service_rows, 'num_requests_waiting_after'),
        'kv_cache_usage_perc_after': mean_metric(service_rows, 'kv_cache_usage_perc_after'),
        'kv_cache_usage_perc_peak': mean_metric(service_rows, 'kv_cache_usage_perc_during_run_max') or mean_metric(service_rows, 'kv_cache_usage_perc_after'),
        'gpu_memory_used_mb_before': mean_metric(service_rows, 'gpu_memory_used_mb_before'),
        'gpu_memory_used_mb_after': mean_metric(service_rows, 'gpu_memory_used_mb_after'),
        'gpu_memory_used_mb_peak': mean_metric(service_rows, 'gpu_memory_used_mb_during_run_max') or mean_metric(service_rows, 'gpu_memory_used_mb_after'),
        'gpu_memory_used_mb_before_start': startup.get('gpu_memory_used_mb_before_start'),
        'gpu_memory_used_mb_after_health': startup.get('gpu_memory_used_mb_after_health'),
        'gpu_memory_used_mb_after_stop': entry.get('gpu_memory_used_mb_after_stop'),
        'vllm_model_weights_memory_gb': startup_log.get('model_weights_memory_gb'),
        'vllm_gpu_kv_cache_size_tokens': startup_log.get('gpu_kv_cache_size_tokens'),
        'vllm_num_gpu_blocks': startup_log.get('num_gpu_blocks'),
        'vllm_num_cpu_blocks': startup_log.get('num_cpu_blocks'),
        'vllm_startup_memory_line_count': startup_log.get('memory_line_count'),
        'vllm_startup_cuda_graph_line_count': startup_log.get('cuda_graph_line_count'),
        'vllm_startup_allocator_line_count': startup_log.get('allocator_line_count'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate Phase 3 parameter tuning results.')
    parser.add_argument(
        '--manifest',
        action='append',
        required=True,
        help='Path to results/param_tuning/raw/<sweep_run_id>/manifest.json. Provide multiple times to merge split sweeps.',
    )
    parser.add_argument('--output-dir', default='results/param_tuning')
    args = parser.parse_args()

    settings = load_settings()
    manifest_paths = [resolve_repo_path(settings.repo_root, value) for value in args.manifest]
    output_dir = resolve_repo_path(settings.repo_root, args.output_dir)

    manifests = [json.loads(path.read_text(encoding='utf-8')) for path in manifest_paths]
    rows = []
    for manifest in manifests:
        rows.extend(
            summarize_entry(entry, settings)
            for entry in manifest['entries']
            if entry.get('status') == 'completed'
        )
    rows.sort(key=lambda row: (row.get('experiment_family', ''), row['tuning_target'], safe_float(row['tuning_value']), row.get('tuning_label', '')))
    write_csv(output_dir / 'param_tuning.csv', rows)
    print(
        json.dumps(
            {
                'manifest_paths': [str(path) for path in manifest_paths],
                'param_tuning_csv': str(output_dir / 'param_tuning.csv'),
                'rows': len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
