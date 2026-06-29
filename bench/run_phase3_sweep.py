from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import replace
from itertools import product
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from analysis.phase3_common import write_json  # noqa: E402
from bench.benchmark_backends import BenchmarkCase  # noqa: E402
from bench.collect_metrics import query_gpu_memory_used_mb  # noqa: E402
from bench.config import load_settings  # noqa: E402
from bench.run_single_case import run_case  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def tail_text(path: Path, line_count: int = 40) -> str:
    if not path.exists():
        return ''
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    return '\n'.join(lines[-line_count:])


def wait_for_health(
    base_url: str,
    timeout_s: float,
    process: subprocess.Popen[str],
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"{base_url.rstrip('/')}/health"
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_excerpt = tail_text(log_path)
            raise RuntimeError(
                'vLLM process exited before becoming healthy.\n'
                f'log_path: {log_path}\n'
                f'last_log_lines:\n{log_excerpt}'
            )
        try:
            with urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except URLError as exc:
            last_error = exc
        except OSError as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f'vLLM health check timed out for {url}: {last_error}')


def resolve_gpu_memory_utilization(default_value: float) -> float:
    override = os.environ.get('PHASE3_GPU_MEMORY_UTILIZATION')
    if override in (None, ''):
        return float(default_value)
    return float(override)


MODEL_WEIGHTS_RE = re.compile(
    r'(?:Model loading took|Loading model weights took|model weights.*?)[^0-9]*(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>GiB|GB|MiB|MB)',
    re.IGNORECASE,
)
AVAILABLE_KV_CACHE_MEMORY_RE = re.compile(
    r'Available KV cache memory:\s*(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>GiB|GB|MiB|MB)',
    re.IGNORECASE,
)
GPU_KV_CACHE_SIZE_RE = re.compile(
    r'GPU KV cache size:\s*(?P<value>[0-9,]+)\s*tokens',
    re.IGNORECASE,
)
CUDA_GRAPH_POOL_MEMORY_RE = re.compile(
    r'CUDA graph pool memory:\s*(?P<actual>[0-9]+(?:\.[0-9]+)?)\s*(?P<actual_unit>GiB|GB|MiB|MB)\s*\(actual\),\s*'
    r'(?P<estimated>[0-9]+(?:\.[0-9]+)?)\s*(?P<estimated_unit>GiB|GB|MiB|MB)\s*\(estimated\)',
    re.IGNORECASE,
)
GPU_BLOCKS_RE = re.compile(
    r'(?:#\s*GPU blocks|num gpu blocks)\s*[:=]\s*(?P<gpu>[0-9,]+).*?(?:#\s*CPU blocks|num cpu blocks)\s*[:=]\s*(?P<cpu>[0-9,]+)',
    re.IGNORECASE,
)


def _memory_to_gb(value: str, unit: str) -> float:
    number = float(value.replace(',', ''))
    if unit.lower() in {'mib', 'mb'}:
        return number / 1024.0
    return number


def parse_vllm_startup_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {
            'model_weights_memory_gb': None,
            'available_kv_cache_memory_gb': None,
            'gpu_kv_cache_size_tokens': None,
            'num_gpu_blocks': None,
            'num_cpu_blocks': None,
            'cuda_graph_pool_memory_actual_gb': None,
            'cuda_graph_pool_memory_estimated_gb': None,
            'cuda_graph_line_count': 0,
            'allocator_line_count': 0,
            'memory_line_count': 0,
            'startup_memory_lines': [],
            'startup_cuda_graph_lines': [],
            'startup_allocator_lines': [],
        }

    lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
    memory_lines: list[str] = []
    cuda_graph_lines: list[str] = []
    allocator_lines: list[str] = []
    model_weights_memory_gb = None
    available_kv_cache_memory_gb = None
    gpu_kv_cache_size_tokens = None
    num_gpu_blocks = None
    num_cpu_blocks = None
    cuda_graph_pool_memory_actual_gb = None
    cuda_graph_pool_memory_estimated_gb = None

    for line in lines:
        lower = line.lower()
        if any(token in lower for token in ('memory', 'kv cache', 'gpu block', 'cpu block')):
            memory_lines.append(line)
        if any(token in lower for token in ('cuda graph', 'cudagraph', 'graph capture')):
            cuda_graph_lines.append(line)
        if any(token in lower for token in ('allocator', 'pytorch_cuda_alloc_conf', 'cuda malloc')):
            allocator_lines.append(line)

        weights_match = MODEL_WEIGHTS_RE.search(line)
        if weights_match:
            model_weights_memory_gb = _memory_to_gb(
                weights_match.group('value'),
                weights_match.group('unit'),
            )

        available_kv_match = AVAILABLE_KV_CACHE_MEMORY_RE.search(line)
        if available_kv_match:
            available_kv_cache_memory_gb = _memory_to_gb(
                available_kv_match.group('value'),
                available_kv_match.group('unit'),
            )

        kv_match = GPU_KV_CACHE_SIZE_RE.search(line)
        if kv_match:
            gpu_kv_cache_size_tokens = int(kv_match.group('value').replace(',', ''))

        cuda_graph_pool_match = CUDA_GRAPH_POOL_MEMORY_RE.search(line)
        if cuda_graph_pool_match:
            cuda_graph_pool_memory_actual_gb = _memory_to_gb(
                cuda_graph_pool_match.group('actual'),
                cuda_graph_pool_match.group('actual_unit'),
            )
            cuda_graph_pool_memory_estimated_gb = _memory_to_gb(
                cuda_graph_pool_match.group('estimated'),
                cuda_graph_pool_match.group('estimated_unit'),
            )

        blocks_match = GPU_BLOCKS_RE.search(line)
        if blocks_match:
            num_gpu_blocks = int(blocks_match.group('gpu').replace(',', ''))
            num_cpu_blocks = int(blocks_match.group('cpu').replace(',', ''))

    return {
        'model_weights_memory_gb': model_weights_memory_gb,
        'available_kv_cache_memory_gb': available_kv_cache_memory_gb,
        'gpu_kv_cache_size_tokens': gpu_kv_cache_size_tokens,
        'num_gpu_blocks': num_gpu_blocks,
        'num_cpu_blocks': num_cpu_blocks,
        'cuda_graph_pool_memory_actual_gb': cuda_graph_pool_memory_actual_gb,
        'cuda_graph_pool_memory_estimated_gb': cuda_graph_pool_memory_estimated_gb,
        'cuda_graph_line_count': len(cuda_graph_lines),
        'allocator_line_count': len(allocator_lines),
        'memory_line_count': len(memory_lines),
        'startup_memory_lines': memory_lines[-30:],
        'startup_cuda_graph_lines': cuda_graph_lines[-30:],
        'startup_allocator_lines': allocator_lines[-30:],
    }


def expand_workloads(workload_config: dict[str, Any]) -> list[dict[str, Any]]:
    dimensions = {
        'concurrency': workload_config.get('concurrency'),
        'input_tokens': workload_config.get('input_tokens'),
        'output_tokens': workload_config.get('output_tokens'),
    }
    if not any(isinstance(value, list) for value in dimensions.values()):
        return [dict(workload_config)]

    static = {
        key: value
        for key, value in workload_config.items()
        if key not in dimensions
    }
    keys = list(dimensions)
    value_lists = [
        value if isinstance(value, list) else [value]
        for value in (dimensions[key] for key in keys)
    ]
    workloads: list[dict[str, Any]] = []
    for values in product(*value_lists):
        workload = dict(static)
        workload.update(dict(zip(keys, values)))
        workloads.append(workload)
    return workloads


def build_single_variable_entries(
    tuning_config: dict[str, Any],
    sweep_run_id: str,
    selected_targets: set[str] | None,
) -> list[dict[str, Any]]:
    workload = tuning_config['fixed_workload']
    baseline_params = dict(tuning_config['baseline_params'])
    baseline_params['gpu_memory_utilization'] = resolve_gpu_memory_utilization(
        baseline_params['gpu_memory_utilization']
    )

    entries: list[dict[str, Any]] = []
    for target, target_spec in tuning_config['tuning_targets'].items():
        if selected_targets and target not in selected_targets:
            continue
        for value in target_spec['values']:
            params = dict(baseline_params)
            params[target] = value
            entry_workload = dict(workload)
            entries.append(
                {
                    'experiment_family': 'single_variable',
                    'tuning_target': target,
                    'tuning_value': value,
                    'tuning_label': f'{target}={value}',
                    'batch_run_id': f"{sweep_run_id}-{target}-{value}",
                    'warmup_batch_run_id': f"{sweep_run_id}-{target}-{value}-warmup",
                    'parameters': params,
                    'workload': entry_workload,
                    'warmup_repeat_count': int(entry_workload.get('warmup_repeats', 0)),
                    'measured_repeat_count': int(entry_workload.get('repeat', 1)),
                    'status': 'planned',
                    'result_count': 0,
                    'warmup_result_count': 0,
                }
            )
    return entries


def build_long_context_entries(
    config: dict[str, Any],
    sweep_run_id: str,
    selected_targets: set[str] | None,
) -> list[dict[str, Any]]:
    long_config = config.get('long_context_tuning') or {}
    if not long_config.get('enabled', False):
        return []

    target = long_config.get('tuning_target', 'long_context_grid')
    if selected_targets and target not in selected_targets:
        return []

    fixed_params = dict(long_config.get('fixed_params', {}))
    fixed_params['gpu_memory_utilization'] = resolve_gpu_memory_utilization(
        fixed_params.get('gpu_memory_utilization', 0.9)
    )
    parameter_grid = long_config['parameter_grid']
    max_model_len_values = parameter_grid['max_model_len']
    max_num_batched_tokens_values = parameter_grid['max_num_batched_tokens']
    workloads = expand_workloads(long_config['workload_matrix'])

    entries: list[dict[str, Any]] = []
    index = 0
    for workload in workloads:
        for max_model_len, max_num_batched_tokens in product(
            max_model_len_values,
            max_num_batched_tokens_values,
        ):
            index += 1
            params = dict(fixed_params)
            params['max_model_len'] = max_model_len
            params['max_num_batched_tokens'] = max_num_batched_tokens
            tuning_label = (
                f"c{workload['concurrency']}_in{workload['input_tokens']}_out{workload['output_tokens']}_"
                f"mml{max_model_len}_mbt{max_num_batched_tokens}"
            )
            batch_run_id = f'{sweep_run_id}-{target}-{tuning_label}'
            entries.append(
                {
                    'experiment_family': 'long_context_grid',
                    'tuning_target': target,
                    'tuning_value': index,
                    'tuning_label': tuning_label,
                    'batch_run_id': batch_run_id,
                    'warmup_batch_run_id': f'{batch_run_id}-warmup',
                    'parameters': params,
                    'workload': dict(workload),
                    'warmup_repeat_count': int(workload.get('warmup_repeats', 0)),
                    'measured_repeat_count': int(workload.get('repeat', 1)),
                    'status': 'planned',
                    'result_count': 0,
                    'warmup_result_count': 0,
                }
            )
    return entries


def build_plan_entries(
    config: dict[str, Any],
    sweep_run_id: str,
    selected_targets: set[str] | None,
) -> list[dict[str, Any]]:
    entries = build_single_variable_entries(
        config['param_tuning'],
        sweep_run_id,
        selected_targets,
    )
    entries.extend(build_long_context_entries(config, sweep_run_id, selected_targets))
    return entries


def start_service(
    entry: dict[str, Any],
    settings,
    model_dir: str,
    served_model_name: str,
    timeout_s: float,
) -> tuple[subprocess.Popen[str], Path, dict[str, Any]]:
    log_path = settings.repo_root / 'logs' / f"{entry['batch_run_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['MODEL_DIR'] = model_dir
    env['SERVED_MODEL_NAME'] = served_model_name
    env['VLLM_HOST'] = settings.vllm_host
    env['VLLM_PORT'] = str(settings.vllm_port)
    env['MAX_MODEL_LEN'] = str(entry['parameters']['max_model_len'])
    env['SERVICE_MAX_MODEL_LEN'] = str(entry['parameters']['max_model_len'])
    env['GPU_MEMORY_UTILIZATION'] = str(entry['parameters']['gpu_memory_utilization'])
    env['VLLM_MAX_NUM_BATCHED_TOKENS'] = str(entry['parameters']['max_num_batched_tokens'])
    env['VLLM_MAX_NUM_SEQS'] = str(entry['parameters']['max_num_seqs'])
    startup_diagnostics = {
        'gpu_memory_used_mb_before_start': query_gpu_memory_used_mb(),
        'gpu_memory_used_mb_after_health': None,
        'vllm_startup_log': {},
    }
    handle = log_path.open('w', encoding='utf-8')
    process = subprocess.Popen(
        ['bash', str(settings.repo_root / 'scripts' / 'run_vllm_local.sh')],
        cwd=settings.repo_root,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    try:
        wait_for_health(settings.vllm_base_url, timeout_s, process, log_path)
        startup_diagnostics['gpu_memory_used_mb_after_health'] = query_gpu_memory_used_mb()
        startup_diagnostics['vllm_startup_log'] = parse_vllm_startup_log(log_path)
    except Exception:
        startup_diagnostics['gpu_memory_used_mb_after_failure'] = query_gpu_memory_used_mb()
        startup_diagnostics['vllm_startup_log'] = parse_vllm_startup_log(log_path)
        entry['startup_diagnostics'] = startup_diagnostics
        stop_service(process)
        raise
    return process, log_path, startup_diagnostics


def stop_service(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def build_case(
    entry: dict[str, Any],
    workload: dict[str, Any],
    repeat_index: int,
    num_prompts: int | None,
    *,
    is_warmup: bool,
) -> BenchmarkCase:
    case_suffix = (
        f"{entry['tuning_target']}_v{entry['tuning_value']}_"
        f"c{workload['concurrency']}_in{workload['input_tokens']}_"
        f"out{workload['output_tokens']}"
    )
    if is_warmup:
        case_id = f'{case_suffix}_warmup_r{repeat_index}'
    else:
        case_id = f'{case_suffix}_r{repeat_index}'
    return BenchmarkCase(
        suite=workload['suite'],
        mode=workload.get('mode', 'non_stream'),
        concurrency=int(workload['concurrency']),
        input_tokens=int(workload['input_tokens']),
        output_tokens=int(workload['output_tokens']),
        repeat_index=repeat_index,
        num_prompts=int(num_prompts) if num_prompts is not None else None,
        case_id=case_id,
    )


def run_repeats(
    *,
    batch_run_id: str,
    entry: dict[str, Any],
    runtime_settings,
    repeat_count: int,
    is_warmup: bool,
) -> list[str]:
    if repeat_count <= 0:
        return []
    workload = entry['workload']
    num_prompts = workload.get('num_prompts')
    combined_paths: list[str] = []
    for repeat_index in range(1, repeat_count + 1):
        case = build_case(
            entry,
            workload,
            repeat_index,
            num_prompts,
            is_warmup=is_warmup,
        )
        run_case(batch_run_id=batch_run_id, case=case, settings=runtime_settings)
        combined_paths.append(
            str(
                runtime_settings.raw_benchmark_dir
                / batch_run_id
                / f'{case.resolved_case_id()}.combined.json'
            )
        )
    return combined_paths


def execute_entry(entry: dict[str, Any], runtime_settings) -> tuple[list[str], list[str]]:
    workload = entry['workload']
    measured_repeat_count = int(workload.get('repeat', 1))
    warmup_repeat_count = int(workload.get('warmup_repeats', 0))
    previous_budget = os.environ.get('SERVICE_MAX_MODEL_LEN')
    os.environ['SERVICE_MAX_MODEL_LEN'] = str(entry['parameters']['max_model_len'])
    try:
        warmup_paths = run_repeats(
            batch_run_id=entry['warmup_batch_run_id'],
            entry=entry,
            runtime_settings=runtime_settings,
            repeat_count=warmup_repeat_count,
            is_warmup=True,
        )
        measured_paths = run_repeats(
            batch_run_id=entry['batch_run_id'],
            entry=entry,
            runtime_settings=runtime_settings,
            repeat_count=measured_repeat_count,
            is_warmup=False,
        )
    finally:
        if previous_budget is None:
            os.environ.pop('SERVICE_MAX_MODEL_LEN', None)
        else:
            os.environ['SERVICE_MAX_MODEL_LEN'] = previous_budget
    return measured_paths, warmup_paths


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a Phase 3 single-variable parameter sweep.')
    parser.add_argument('--config', default='bench/phase3_sweep.yaml')
    parser.add_argument('--target', action='append', default=None, help='Specific tuning target(s) to run.')
    parser.add_argument('--output-dir', default='results/param_tuning')
    parser.add_argument('--sweep-run-id', default=None)
    parser.add_argument('--model-dir', default=None)
    parser.add_argument('--served-model-name', default=None)
    parser.add_argument('--health-timeout-s', type=float, default=240.0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    settings = load_settings()
    config_path = settings.repo_root / args.config
    config = load_config(config_path)
    sweep_run_id = args.sweep_run_id or datetime.now(timezone.utc).strftime('phase3-sweep-%Y%m%dT%H%M%SZ')
    selected_targets = set(args.target or []) or None
    entries = build_plan_entries(config, sweep_run_id, selected_targets)
    if not entries:
        raise SystemExit('No sweep entries selected.')

    output_dir = settings.repo_root / args.output_dir
    manifest_path = output_dir / 'raw' / sweep_run_id / 'manifest.json'
    model_dir = args.model_dir or str(settings.model_dir)
    served_model_name = args.served_model_name or settings.served_model_name
    manifest: dict[str, Any] = {
        'sweep_run_id': sweep_run_id,
        'config_path': str(config_path),
        'output_dir': str(output_dir),
        'model_dir': model_dir,
        'served_model_name': served_model_name,
        'base_url': settings.vllm_base_url,
        'entries': entries,
    }
    write_json(manifest_path, manifest)

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    base_settings = load_settings()
    for entry in entries:
        process = None
        try:
            process, log_path, startup_diagnostics = start_service(
                entry,
                base_settings,
                model_dir,
                served_model_name,
                args.health_timeout_s,
            )
            entry['startup_diagnostics'] = startup_diagnostics
            runtime_settings = replace(
                base_settings,
                model_dir=Path(model_dir),
                served_model_name=served_model_name,
            )
            combined_paths, warmup_paths = execute_entry(entry, runtime_settings)
            entry['status'] = 'completed'
            entry['result_count'] = len(combined_paths)
            entry['warmup_result_count'] = len(warmup_paths)
            entry['combined_paths'] = combined_paths
            entry['warmup_combined_paths'] = warmup_paths
            entry['service_log_path'] = str(log_path)
        except Exception as exc:  # noqa: BLE001
            entry['status'] = 'failed'
            entry['error'] = str(exc)
            write_json(manifest_path, manifest)
            raise
        finally:
            stop_service(process)
            if process is not None:
                entry['gpu_memory_used_mb_after_stop'] = query_gpu_memory_used_mb()
            write_json(manifest_path, manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
