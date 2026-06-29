from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_model_dir() -> Path:
    autodl = Path("/root/autodl-tmp/qwen2.5-0.5b")
    if autodl.is_dir():
        return autodl
    return Path("/root/models/qwen2.5-0.5b")


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    venv_python: Path
    model_dir: Path
    served_model_name: str
    vllm_base_url: str
    vllm_host: str
    vllm_port: int
    vllm_endpoint: str
    benchmark_backend: str
    benchmark_num_prompts: int
    benchmark_stream_num_prompts: int
    benchmark_warmups: int
    benchmark_seed: int
    service_metrics_poll_interval_s: float
    results_dir: Path
    raw_benchmark_dir: Path
    raw_prometheus_dir: Path
    plots_dir: Path
    tokenizer_mode: str


def _parse_base_url(base_url: str) -> tuple[str, int]:
    stripped = base_url.removeprefix("http://").removeprefix("https://")
    host, _, port = stripped.partition(":")
    return host or "127.0.0.1", int(port or "80")


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parent.parent
    vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:19100")
    default_host, default_port = _parse_base_url(vllm_base_url)
    results_dir = repo_root / "results"
    return Settings(
        repo_root=repo_root,
        venv_python=repo_root / ".venv" / "bin" / "python3",
        model_dir=Path(
            os.environ.get("MODEL_DIR", str(_default_model_dir()))
        ),
        served_model_name=os.environ.get(
            "SERVED_MODEL_NAME", "qwen-05b-local"
        ),
        vllm_base_url=vllm_base_url,
        vllm_host=os.environ.get("VLLM_HOST", default_host),
        vllm_port=int(os.environ.get("VLLM_PORT", str(default_port))),
        vllm_endpoint=os.environ.get("VLLM_ENDPOINT", "/v1/chat/completions"),
        benchmark_backend=os.environ.get("BENCHMARK_BACKEND", "openai-chat"),
        benchmark_num_prompts=int(os.environ.get("BENCHMARK_NUM_PROMPTS", "40")),
        benchmark_stream_num_prompts=int(
            os.environ.get("BENCHMARK_STREAM_NUM_PROMPTS", "20")
        ),
        benchmark_warmups=int(os.environ.get("BENCHMARK_NUM_WARMUPS", "0")),
        benchmark_seed=int(os.environ.get("BENCHMARK_SEED", "20260627")),
        service_metrics_poll_interval_s=float(
            os.environ.get("SERVICE_METRICS_POLL_INTERVAL_S", "0.5")
        ),
        results_dir=results_dir,
        raw_benchmark_dir=results_dir / "raw" / "benchmark",
        raw_prometheus_dir=results_dir / "raw" / "prometheus",
        plots_dir=results_dir / "plots",
        tokenizer_mode=os.environ.get("TOKENIZER_MODE", "auto"),
    )
