from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{.*\})?\s+(?P<value>[-+eE0-9\.]+)$"
)
LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


@dataclass(frozen=True)
class MetricSample:
    name: str
    labels: dict[str, str]
    value: float


def fetch_metrics_text(base_url: str, timeout: float = 5.0) -> str:
    url = f"{base_url.rstrip('/')}/metrics"
    completed = subprocess.run(
        [
            "curl",
            "--noproxy",
            "*",
            "--max-time",
            str(int(timeout)),
            "-fsS",
            url,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"failed to fetch metrics from {url}: {completed.stderr.strip()}"
        )
    return completed.stdout


def parse_metrics_text(text: str) -> list[MetricSample]:
    samples: list[MetricSample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE_RE.match(line)
        if not match:
            continue
        labels_text = match.group("labels") or ""
        labels = {key: value for key, value in LABEL_RE.findall(labels_text)}
        samples.append(
            MetricSample(
                name=match.group("name"),
                labels=labels,
                value=float(match.group("value")),
            )
        )
    return samples


def _sum_metric(
    samples: list[MetricSample],
    names: tuple[str, ...],
    predicate: Callable[[MetricSample], bool] | None = None,
) -> float:
    total = 0.0
    for sample in samples:
        if sample.name not in names:
            continue
        if predicate is not None and not predicate(sample):
            continue
        total += sample.value
    return total


def query_gpu_memory_used_mb() -> float | None:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    if completed.returncode != 0:
        return None

    first_line = completed.stdout.strip().splitlines()[0]
    try:
        return float(first_line)
    except (IndexError, ValueError):
        return None


def collect_service_metrics(base_url: str, served_model_name: str | None = None) -> dict:
    text = fetch_metrics_text(base_url)
    samples = parse_metrics_text(text)

    def predicate(sample: MetricSample) -> bool:
        if not served_model_name:
            return True
        model_name = sample.labels.get("model_name")
        return model_name in (None, "", served_model_name)

    snapshot = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "served_model_name": served_model_name,
        "num_requests_running": _sum_metric(
            samples, ("vllm:num_requests_running",), predicate
        ),
        "num_requests_waiting": _sum_metric(
            samples, ("vllm:num_requests_waiting",), predicate
        ),
        "kv_cache_usage_perc": _sum_metric(
            samples, ("vllm:kv_cache_usage_perc",), predicate
        ),
        "prompt_tokens_total": _sum_metric(
            samples, ("vllm:prompt_tokens_total", "vllm:prompt_tokens"), predicate
        ),
        "generation_tokens_total": _sum_metric(
            samples,
            ("vllm:generation_tokens_total", "vllm:generation_tokens"),
            predicate,
        ),
        "request_success_total": _sum_metric(
            samples, ("vllm:request_success_total", "vllm:request_success"), predicate
        ),
        "gpu_memory_used_mb": query_gpu_memory_used_mb(),
        "raw_sample_count": len(samples),
    }
    return snapshot


def derive_service_delta(before: dict, after: dict, duration_s: float) -> dict:
    duration_s = max(duration_s, 1e-9)
    prompt_delta = max(
        0.0, float(after["prompt_tokens_total"]) - float(before["prompt_tokens_total"])
    )
    generation_delta = max(
        0.0,
        float(after["generation_tokens_total"])
        - float(before["generation_tokens_total"]),
    )
    success_delta = max(
        0.0,
        float(after["request_success_total"]) - float(before["request_success_total"]),
    )
    gpu_before = before.get("gpu_memory_used_mb")
    gpu_after = after.get("gpu_memory_used_mb")
    gpu_delta = (
        None
        if gpu_before is None or gpu_after is None
        else float(gpu_after) - float(gpu_before)
    )

    return {
        "duration_s": duration_s,
        "prompt_tokens_delta": prompt_delta,
        "generation_tokens_delta": generation_delta,
        "request_success_delta": success_delta,
        "prompt_throughput_toks_per_s": prompt_delta / duration_s,
        "generation_throughput_toks_per_s": generation_delta / duration_s,
        "kv_cache_usage_perc_before": before["kv_cache_usage_perc"],
        "kv_cache_usage_perc_after": after["kv_cache_usage_perc"],
        "num_requests_running_before": before["num_requests_running"],
        "num_requests_running_after": after["num_requests_running"],
        "num_requests_waiting_before": before["num_requests_waiting"],
        "num_requests_waiting_after": after["num_requests_waiting"],
        "gpu_memory_used_mb_before": gpu_before,
        "gpu_memory_used_mb_after": gpu_after,
        "gpu_memory_used_mb_delta": gpu_delta,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
