from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.config import Settings


@dataclass(frozen=True)
class BenchmarkCase:
    suite: str
    mode: str
    concurrency: int
    input_tokens: int
    output_tokens: int
    repeat_index: int = 1
    num_prompts: int | None = None
    request_rate: float | None = None
    temperature: float = 0.0
    case_id: str | None = None

    def resolved_case_id(self) -> str:
        if self.case_id:
            return self.case_id
        return (
            f"{self.suite}_c{self.concurrency}_in{self.input_tokens}"
            f"_out{self.output_tokens}_r{self.repeat_index}"
        )


def build_vllm_bench_command(
    *,
    settings: Settings,
    case: BenchmarkCase,
    result_dir: Path,
    result_filename: str,
    metadata: dict[str, str] | None = None,
) -> list[str]:
    num_prompts = case.num_prompts
    if num_prompts is None:
        num_prompts = (
            settings.benchmark_stream_num_prompts
            if case.mode == "stream"
            else settings.benchmark_num_prompts
        )

    request_rate = case.request_rate if case.request_rate is not None else float("inf")
    metadata = metadata or {}

    command = [
        str(settings.venv_python),
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        settings.benchmark_backend,
        "--host",
        settings.vllm_host,
        "--port",
        str(settings.vllm_port),
        "--endpoint",
        settings.vllm_endpoint,
        "--model",
        str(settings.model_dir),
        "--served-model-name",
        settings.served_model_name,
        "--tokenizer",
        str(settings.model_dir),
        "--tokenizer-mode",
        settings.tokenizer_mode,
        "--dataset-name",
        "random",
        "--input-len",
        str(case.input_tokens),
        "--output-len",
        str(case.output_tokens),
        "--num-prompts",
        str(num_prompts),
        "--num-warmups",
        str(settings.benchmark_warmups),
        "--seed",
        str(settings.benchmark_seed),
        "--max-concurrency",
        str(case.concurrency),
        "--temperature",
        str(case.temperature),
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,95,99",
    ]

    if request_rate != float("inf"):
        command.extend(["--request-rate", str(request_rate)])

    if metadata:
        command.append("--metadata")
        command.extend(f"{key}={value}" for key, value in metadata.items())

    return command


def _normalize_percentiles(items: list[list[float]] | list[tuple[float, float]]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for percentile, value in items:
        normalized[f"p{int(percentile)}"] = float(value)
    return normalized


def normalize_benchmark_result(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "duration_s": float(raw.get("duration", 0.0)),
        "completed": int(raw.get("completed", 0)),
        "failed": int(raw.get("failed", 0)),
        "total_input_tokens": int(raw.get("total_input", 0)),
        "total_output_tokens": int(raw.get("total_output", 0)),
        "request_throughput_qps": float(raw.get("request_throughput", 0.0)),
        "output_throughput_tps": float(raw.get("output_throughput", 0.0)),
        "total_token_throughput_tps": float(raw.get("total_token_throughput", 0.0)),
        "mean_ttft_ms": float(raw.get("mean_ttft_ms", 0.0)),
        "median_ttft_ms": float(raw.get("median_ttft_ms", 0.0)),
        "mean_tpot_ms": float(raw.get("mean_tpot_ms", 0.0)),
        "median_tpot_ms": float(raw.get("median_tpot_ms", 0.0)),
        "mean_itl_ms": float(raw.get("mean_itl_ms", 0.0)),
        "median_itl_ms": float(raw.get("median_itl_ms", 0.0)),
        "mean_e2el_ms": float(raw.get("mean_e2el_ms", 0.0)),
        "median_e2el_ms": float(raw.get("median_e2el_ms", 0.0)),
        "ttft_percentiles_ms": _normalize_percentiles(
            raw.get("percentiles_ttft_ms", [])
        ),
        "tpot_percentiles_ms": _normalize_percentiles(
            raw.get("percentiles_tpot_ms", [])
        ),
        "itl_percentiles_ms": _normalize_percentiles(
            raw.get("percentiles_itl_ms", [])
        ),
        "e2el_percentiles_ms": _normalize_percentiles(
            raw.get("percentiles_e2el_ms", [])
        ),
    }


def run_vllm_benchmark(
    *,
    settings: Settings,
    case: BenchmarkCase,
    result_dir: Path,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    result_filename = f"{case.resolved_case_id()}.benchmark.json"
    command = build_vllm_bench_command(
        settings=settings,
        case=case,
        result_dir=result_dir,
        result_filename=result_filename,
        metadata=metadata,
    )

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    completed = subprocess.run(
        command,
        cwd=settings.repo_root,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    result_path = result_dir / result_filename
    if completed.returncode != 0:
        raise RuntimeError(
            "vLLM benchmark failed.\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    if not result_path.exists():
        raise FileNotFoundError(f"Expected benchmark result file: {result_path}")

    raw_result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "result_path": str(result_path),
        "raw_result": raw_result,
        "normalized_result": normalize_benchmark_result(raw_result),
        "python_executable": sys.executable,
    }
