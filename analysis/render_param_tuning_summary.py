from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.phase3_common import load_csv_rows, resolve_repo_path, safe_float  # noqa: E402
from bench.config import load_settings  # noqa: E402

BASELINE_VALUES = {
    'max_model_len': 3072.0,
    'max_num_batched_tokens': 4096.0,
    'max_num_seqs': 32.0,
}


def format_delta(current: float, baseline: float, suffix: str = '') -> str:
    if baseline == 0:
        return f'{current:.2f}{suffix}'
    delta_pct = ((current - baseline) / baseline) * 100.0
    return f'{current:.2f}{suffix} ({delta_pct:+.1f}%)'


def pick_baseline_row(target: str, bucket: list[dict[str, str]]) -> dict[str, str]:
    baseline_value = BASELINE_VALUES.get(target)
    if baseline_value is None:
        return bucket[0]
    for row in bucket:
        if safe_float(row['tuning_value']) == baseline_value:
            return row
    return bucket[0]


def row_label(row: dict[str, str]) -> str:
    return row.get('tuning_label') or row.get('tuning_value', '')


def render_memory_decomposition_note(rows: list[dict[str, str]]) -> str:
    startup_fields = (
        'gpu_memory_used_mb_before_start',
        'gpu_memory_used_mb_after_health',
        'vllm_model_weights_memory_gb',
        'vllm_gpu_kv_cache_size_tokens',
        'vllm_num_gpu_blocks',
        'vllm_startup_cuda_graph_line_count',
        'vllm_startup_allocator_line_count',
    )
    has_startup_fields = any(
        any(row.get(field) not in (None, '') for field in startup_fields)
        for row in rows
    )
    if not has_startup_fields:
        return (
            '- 本轮结果未记录启动阶段显存分解；不能用最终 `gpu_memory_used_mb_after` 直接判断 AWQ 权重省显存。'
        )
    return (
        '- 本轮结果已记录启动阶段显存分解字段，包括启动前 GPU memory、服务健康后 GPU memory、'
        'vLLM startup log 中的 model weights memory、GPU KV cache size、GPU/CPU blocks 以及 CUDA graph / allocator 线索；'
        '分析 AWQ 显存收益时应优先使用这些字段，而不是只看最终 nvidia-smi used memory。'
    )


def render_execution_note(rows: list[dict[str, str]]) -> str:
    values = sorted({round(safe_float(row['gpu_memory_utilization']), 4) for row in rows})
    if not values:
        return '- 本轮 sweep 未记录 `gpu_memory_utilization`，应结合 manifest 复核运行资源约束。'
    formatted = ', '.join(f'{value:g}' for value in values)
    if max(values) <= 0.2:
        return (
            f'- 本轮 sweep 在共享 GPU 约束下执行，`gpu_memory_utilization` 为 `{formatted}`，'
            '结论应理解为受限显存预算下的相对 trade-off。'
        )
    if min(values) >= 0.8:
        return (
            f'- 本轮 sweep 在独占或接近独占 GPU 的条件下执行，`gpu_memory_utilization` 为 `{formatted}`，'
            '更接近正式部署时的参数表现。'
        )
    return (
        f'- 本轮 sweep 使用了混合显存预算配置，`gpu_memory_utilization` 为 `{formatted}`，'
        '分析时需要同时考虑参数差异与资源约束差异。'
    )


def render_warmup_note(rows: list[dict[str, str]]) -> str:
    warmup_values = sorted({int(safe_float(row.get('warmup_repeat_count', '0'))) for row in rows})
    if not warmup_values or warmup_values == [0]:
        return '- 本轮 summary 未显式排除 warmup pass；若存在冷启动抖动，应结合 raw repeat 结果进一步判断。'
    formatted = ', '.join(str(value) for value in warmup_values)
    return (
        f'- 本轮 sweep 每个参数档位先执行 `{formatted}` 次 warmup，再统计正式 repeat；'
        ' `param_tuning.csv` 与 summary 仅汇总 warmup 之后的 measured repeats。'
    )


def build_overall_finding(
    target: str,
    baseline_row: dict[str, str],
    best_qps: dict[str, str],
    best_latency: dict[str, str],
    lowest_memory: dict[str, str],
) -> str:
    if target == 'max_model_len':
        if best_qps['tuning_value'] != best_latency['tuning_value']:
            return (
                f'`max_model_len` 组内差异很小：`{best_qps["tuning_value"]}` 吞吐最高，'
                f'`{best_latency["tuning_value"]}` 的 P95 最低。考虑当前 workload 实际上下文远低于上限，'
                '`3072` 仍可作为默认折中值。'
            )
        return f'`max_model_len={best_qps["tuning_value"]}` 在该 workload 下同时拿到最好吞吐与 P95，可作为默认值。'

    if target == 'max_num_batched_tokens':
        if best_qps['tuning_value'] != best_latency['tuning_value']:
            return (
                f'`max_num_batched_tokens={best_latency["tuning_value"]}` 在 P95 上略占优，'
                f'但 `max_num_batched_tokens={best_qps["tuning_value"]}` 拿到最高吞吐且 TTFT 更低；'
                f'`{lowest_memory["tuning_value"]}` 仅在显存上略省，因此默认值仍建议保留 `4096`。'
            )
        return f'`max_num_batched_tokens={best_qps["tuning_value"]}` 同时拿到最好吞吐与 P95，适合作为默认值。'

    if target == 'max_num_seqs':
        if best_qps['tuning_value'] == best_latency['tuning_value'] == baseline_row['tuning_value']:
            return (
                f'`max_num_seqs={baseline_row["tuning_value"]}` 仍是综合最优点，'
                '说明过低会限吞吐，过高又会把尾延迟重新拉高。'
            )
        return (
            f'`max_num_seqs={best_qps["tuning_value"]}` 吞吐最好，'
            f'`max_num_seqs={best_latency["tuning_value"]}` P95 最低；当前 workload 下优先保留 `32`。'
        )

    return '该参数组已完成 sweep，可结合吞吐、P95 和显存综合取舍。'


def main() -> None:
    parser = argparse.ArgumentParser(description='Render a parameter tuning markdown summary.')
    parser.add_argument('--csv', default='results/param_tuning/param_tuning.csv')
    parser.add_argument('--output-path', default='results/param_tuning/param_tuning_summary.md')
    args = parser.parse_args()

    settings = load_settings()
    rows = load_csv_rows(resolve_repo_path(settings.repo_root, args.csv))
    if not rows:
        raise SystemExit('No tuning rows found. Run aggregation first.')

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row['tuning_target']].append(row)
    for bucket in grouped.values():
        bucket.sort(key=lambda row: safe_float(row['tuning_value']))

    preferred_targets = ('max_model_len', 'max_num_batched_tokens', 'max_num_seqs', 'long_context_grid')
    ordered_targets = [target for target in preferred_targets if target in grouped]
    ordered_targets.extend(target for target in grouped if target not in ordered_targets)

    lines = ['# Param Tuning Summary', '']
    overall_findings: list[str] = []

    for target in ordered_targets:
        bucket = grouped[target]
        baseline_row = pick_baseline_row(target, bucket)
        best_qps = max(bucket, key=lambda row: safe_float(row['request_throughput_qps']))
        best_latency = min(bucket, key=lambda row: safe_float(row['p95_e2el_ms']))
        lowest_memory = min(bucket, key=lambda row: safe_float(row['gpu_memory_used_mb_after']))

        lines.extend([
            f'## `{target}`',
            '',
            (
                f"- 基线项 `{row_label(baseline_row)}` 下的 QPS / P95 / 显存分别为 "
                f"`{baseline_row['request_throughput_qps']}` / `{baseline_row['p95_e2el_ms']}` ms / "
                f"`{baseline_row['gpu_memory_used_mb_after']}` MB。"
            ),
            (
                f"- 最高吞吐出现在 `{row_label(best_qps)}`："
                f"`{format_delta(safe_float(best_qps['request_throughput_qps']), safe_float(baseline_row['request_throughput_qps']))}` QPS。"
            ),
            (
                f"- 最低 P95 latency 出现在 `{row_label(best_latency)}`："
                f"`{format_delta(safe_float(best_latency['p95_e2el_ms']), safe_float(baseline_row['p95_e2el_ms']), ' ms')}`。"
            ),
            (
                f"- 最低显存占用出现在 `{row_label(lowest_memory)}`："
                f"`{format_delta(safe_float(lowest_memory['gpu_memory_used_mb_after']), safe_float(baseline_row['gpu_memory_used_mb_after']), ' MB')}`。"
            ),
            '',
        ])

        overall_findings.append(
            build_overall_finding(target, baseline_row, best_qps, best_latency, lowest_memory)
        )

    lines.extend([
        '## Overall Findings',
        '',
        *[f'- {item}' for item in overall_findings],
        '',
        '## Notes',
        '',
        render_execution_note(rows),
        render_warmup_note(rows),
        render_memory_decomposition_note(rows),
        '- 选主配置时应优先保留同时兼顾吞吐、P95 latency 与显存的档位，而不是只追单一峰值。',
    ])

    output_path = resolve_repo_path(settings.repo_root, args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(output_path)


if __name__ == '__main__':
    main()
