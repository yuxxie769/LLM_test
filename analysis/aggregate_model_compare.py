from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.phase3_common import (  # noqa: E402
    CASE_FIELDS,
    case_key,
    load_csv_rows,
    mean_metric,
    resolve_repo_path,
    safe_float,
    sort_case_key,
    write_csv,
    write_json,
)
from bench.config import load_settings  # noqa: E402

BENCHMARK_METRICS = (
    'duration_s',
    'completed',
    'failed',
    'request_throughput_qps',
    'output_throughput_tps',
    'total_token_throughput_tps',
    'mean_ttft_ms',
    'median_ttft_ms',
    'p95_ttft_ms',
    'mean_tpot_ms',
    'median_tpot_ms',
    'p95_tpot_ms',
    'mean_itl_ms',
    'median_itl_ms',
    'p95_itl_ms',
    'mean_e2el_ms',
    'median_e2el_ms',
    'p95_e2el_ms',
    'error_rate',
)
SERVICE_METRICS = (
    'prompt_throughput_toks_per_s',
    'generation_throughput_toks_per_s',
    'kv_cache_usage_perc_before',
    'kv_cache_usage_perc_after',
    'kv_cache_usage_perc_during_run_max',
    'num_requests_waiting_after',
    'num_requests_waiting_during_run_max',
    'gpu_memory_used_mb_before',
    'gpu_memory_used_mb_after',
    'gpu_memory_used_mb_during_run_max',
)
PAIRWISE_METRICS = (
    'request_throughput_qps',
    'p95_e2el_ms',
    'output_throughput_tps',
    'error_rate',
    'gpu_memory_used_mb_after',
    'num_requests_waiting_observed',
    'kv_cache_usage_perc_observed',
)


def aggregate_case_rows(rows: list[dict[str, str]], metrics: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(case_key(row), []).append(row)

    aggregated: dict[tuple[str, ...], dict[str, Any]] = {}
    for key, bucket in grouped.items():
        payload: dict[str, Any] = {field: bucket[0][field] for field in CASE_FIELDS}
        payload['repeat_count'] = len(bucket)
        payload['batch_run_ids'] = ','.join(sorted({row['batch_run_id'] for row in bucket}))
        for metric in metrics:
            payload[metric] = mean_metric(bucket, metric)
        aggregated[key] = payload
    return aggregated


def parse_batch_spec(spec: str) -> tuple[str, str]:
    if '=' not in spec:
        raise ValueError(f'Invalid --batch spec: {spec}. Expected label=path_or_batch_id')
    label, value = spec.split('=', 1)
    return label.strip(), value.strip()


def load_batch(label: str, batch_value: str, settings) -> dict[str, Any]:
    batch_dir = resolve_repo_path(settings.repo_root, batch_value)
    if not batch_dir.exists():
        batch_dir = settings.results_dir / 'batches' / batch_value
    if not batch_dir.is_dir():
        raise FileNotFoundError(batch_dir)

    benchmark_rows = load_csv_rows(batch_dir / 'baseline_metrics.csv')
    service_rows = load_csv_rows(batch_dir / 'baseline_service_metrics.csv')
    benchmark_by_key = aggregate_case_rows(benchmark_rows, BENCHMARK_METRICS)
    service_by_key = aggregate_case_rows(service_rows, SERVICE_METRICS)

    rows_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for key, benchmark_payload in benchmark_by_key.items():
        row = dict(benchmark_payload)
        service_payload = service_by_key.get(key, {})
        row.update({metric: service_payload.get(metric, 0.0) for metric in SERVICE_METRICS})
        row['num_requests_waiting_observed'] = (
            row.get('num_requests_waiting_during_run_max')
            or row.get('num_requests_waiting_after')
            or 0.0
        )
        row['kv_cache_usage_perc_observed'] = (
            row.get('kv_cache_usage_perc_during_run_max')
            or row.get('kv_cache_usage_perc_after')
            or 0.0
        )
        row['model_label'] = label
        row['batch_dir'] = str(batch_dir)
        rows_by_key[key] = row

    return {
        'label': label,
        'batch_dir': str(batch_dir),
        'benchmark_rows': len(benchmark_rows),
        'service_rows': len(service_rows),
        'rows_by_key': rows_by_key,
    }


def build_pairwise_rows(reference_rows: dict[tuple[str, ...], dict[str, Any]], candidate_rows: dict[tuple[str, ...], dict[str, Any]], reference_label: str, candidate_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common_keys = sorted(set(reference_rows) & set(candidate_rows), key=sort_case_key)
    for key in common_keys:
        reference = reference_rows[key]
        candidate = candidate_rows[key]
        row: dict[str, Any] = {field: reference[field] for field in CASE_FIELDS}
        row['reference_model'] = reference_label
        row['candidate_model'] = candidate_label
        row['reference_batch_dir'] = reference['batch_dir']
        row['candidate_batch_dir'] = candidate['batch_dir']
        for metric in PAIRWISE_METRICS:
            reference_value = safe_float(reference.get(metric))
            candidate_value = safe_float(candidate.get(metric))
            delta = candidate_value - reference_value
            row[f'{metric}_reference'] = reference_value
            row[f'{metric}_candidate'] = candidate_value
            row[f'{metric}_delta'] = delta
            row[f'{metric}_delta_pct'] = 0.0 if reference_value == 0 else (delta / reference_value) * 100.0
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate two or more Phase 2 batches into a model compare report.')
    parser.add_argument('--batch', action='append', required=True, help='Batch spec in label=path_or_batch_id form. Provide at least two.')
    parser.add_argument('--output-dir', default='results/model_compare', help='Directory for model compare outputs.')
    args = parser.parse_args()

    if len(args.batch) < 2:
        raise SystemExit('Provide at least two --batch arguments.')

    settings = load_settings()
    batches = [load_batch(*parse_batch_spec(spec), settings=settings) for spec in args.batch]
    common_keys = set.intersection(*(set(batch['rows_by_key']) for batch in batches))
    if not common_keys:
        raise SystemExit('No overlapping cases found across the provided batches.')

    long_rows: list[dict[str, Any]] = []
    for key in sorted(common_keys, key=sort_case_key):
        for batch in batches:
            long_rows.append(dict(batch['rows_by_key'][key]))

    output_dir = resolve_repo_path(settings.repo_root, args.output_dir)
    write_csv(output_dir / 'model_compare_long.csv', long_rows)

    pairwise_rows: list[dict[str, Any]] = []
    if len(batches) == 2:
        pairwise_rows = build_pairwise_rows(
            batches[0]['rows_by_key'],
            batches[1]['rows_by_key'],
            batches[0]['label'],
            batches[1]['label'],
        )
        write_csv(output_dir / 'model_compare.csv', pairwise_rows)

    manifest = {
        'batches': [
            {
                'label': batch['label'],
                'batch_dir': batch['batch_dir'],
                'benchmark_rows': batch['benchmark_rows'],
                'service_rows': batch['service_rows'],
                'comparable_cases': len(common_keys),
            }
            for batch in batches
        ],
        'common_cases': len(common_keys),
        'model_compare_long_csv': str(output_dir / 'model_compare_long.csv'),
        'model_compare_csv': str(output_dir / 'model_compare.csv') if pairwise_rows else None,
    }
    write_json(output_dir / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
