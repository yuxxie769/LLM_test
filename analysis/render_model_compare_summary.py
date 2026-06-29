from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.phase3_common import load_csv_rows, resolve_repo_path, safe_float  # noqa: E402
from bench.config import load_settings  # noqa: E402


def describe_case(row: dict[str, str]) -> str:
    return (
        f"concurrency `{row['concurrency']}`, input `{row['input_tokens']}`, "
        f"output `{row['output_tokens']}`"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Render a model compare markdown summary.')
    parser.add_argument('--compare-csv', default='results/model_compare/model_compare.csv')
    parser.add_argument('--output-path', default='results/model_compare/model_compare_summary.md')
    args = parser.parse_args()

    settings = load_settings()
    compare_csv = resolve_repo_path(settings.repo_root, args.compare_csv)
    rows = load_csv_rows(compare_csv)
    if not rows:
        raise SystemExit('No compare rows found. Run aggregation first.')

    reference_model = rows[0]['reference_model']
    candidate_model = rows[0]['candidate_model']
    qps_wins = sum(1 for row in rows if safe_float(row['request_throughput_qps_delta']) > 0)
    latency_wins = sum(1 for row in rows if safe_float(row['p95_e2el_ms_delta']) < 0)
    memory_wins = sum(1 for row in rows if safe_float(row['gpu_memory_used_mb_after_delta']) < 0)
    best_qps = max(rows, key=lambda row: safe_float(row['request_throughput_qps_delta_pct']))
    memory_improvements = [
        row for row in rows if safe_float(row['gpu_memory_used_mb_after_delta']) < 0
    ]
    if memory_improvements:
        memory_extreme = min(
            memory_improvements,
            key=lambda row: safe_float(row['gpu_memory_used_mb_after_delta']),
        )
        memory_extreme_line = (
            f'- 最大显存下降出现在 {describe_case(memory_extreme)}：'
            f'`{memory_extreme["gpu_memory_used_mb_after_delta"]}` MB '
            f'({reference_model}=`{memory_extreme["gpu_memory_used_mb_after_reference"]}`, '
            f'{candidate_model}=`{memory_extreme["gpu_memory_used_mb_after_candidate"]}`).'
        )
    else:
        memory_extreme = min(
            rows,
            key=lambda row: safe_float(row['gpu_memory_used_mb_after_delta']),
        )
        memory_extreme_line = (
            f'- 候选模型没有出现 `gpu_memory_used_mb_after` 更低的 case；最小显存差距出现在 {describe_case(memory_extreme)}：'
            f'`+{safe_float(memory_extreme["gpu_memory_used_mb_after_delta"]):.1f}` MB '
            f'({reference_model}=`{memory_extreme["gpu_memory_used_mb_after_reference"]}`, '
            f'{candidate_model}=`{memory_extreme["gpu_memory_used_mb_after_candidate"]}`).'
        )
    latency_regressions = [
        row for row in rows if safe_float(row['p95_e2el_ms_delta']) > 0
    ]
    if latency_regressions:
        latency_extreme = max(
            latency_regressions,
            key=lambda row: safe_float(row['p95_e2el_ms_delta_pct']),
        )
        latency_extreme_line = (
            f'- 候选模型最明显的尾延迟回退出现在 {describe_case(latency_extreme)}：'
            f'`{latency_extreme["p95_e2el_ms_delta_pct"]}`% '
            f'({reference_model}=`{latency_extreme["p95_e2el_ms_reference"]}` ms, '
            f'{candidate_model}=`{latency_extreme["p95_e2el_ms_candidate"]}` ms)。'
        )
    else:
        latency_extreme = min(
            rows,
            key=lambda row: safe_float(row['p95_e2el_ms_delta_pct']),
        )
        latency_extreme_line = (
            f'- 候选模型没有出现 P95 latency 回退；最大尾延迟改善出现在 {describe_case(latency_extreme)}：'
            f'`{latency_extreme["p95_e2el_ms_delta_pct"]}`% '
            f'({reference_model}=`{latency_extreme["p95_e2el_ms_reference"]}` ms, '
            f'{candidate_model}=`{latency_extreme["p95_e2el_ms_candidate"]}` ms)。'
        )

    lines = [
        '# Model Compare Summary',
        '',
        '## Scope',
        '',
        f'- Reference model: `{reference_model}`',
        f'- Candidate model: `{candidate_model}`',
        f'- Comparable cases: `{len(rows)}`',
        '',
        '## Findings',
        '',
        (
            f'- `{candidate_model}` 在 `{qps_wins}/{len(rows)}` 个重叠 case 上吞吐高于 `{reference_model}`，'
            f'在 `{latency_wins}/{len(rows)}` 个 case 上 P95 latency 更低，'
            f'在 `{memory_wins}/{len(rows)}` 个 case 上显存更低。'
        ),
        (
            f'- 最大吞吐提升出现在 {describe_case(best_qps)}：'
            f'`{best_qps["request_throughput_qps_delta_pct"]}`% '
            f'({reference_model}=`{best_qps["request_throughput_qps_reference"]}`, '
            f'{candidate_model}=`{best_qps["request_throughput_qps_candidate"]}`).'
        ),
        memory_extreme_line,
        latency_extreme_line,
        '',
        '## Notes',
        '',
        '- 模型优劣不要只看单一指标，至少要同时结合吞吐、P95 latency 和显存。',
        '- 进一步决定 Phase 3 sweep 主模型时，应优先保留在目标 workload 下 trade-off 更稳定的一方。',
    ]

    output_path = resolve_repo_path(settings.repo_root, args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(output_path)


if __name__ == '__main__':
    main()
