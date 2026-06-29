from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
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


@dataclass
class ServiceMetricsSampler:
    base_url: str
    served_model_name: str | None = None
    interval_s: float = 0.5
    snapshots: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    _stop_event: Event = field(default_factory=Event, init=False, repr=False)
    _thread: Thread | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        if self.interval_s <= 0:
            return
        self._thread = Thread(target=self._run, name="service-metrics-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> tuple[list[dict], list[str]]:
        if self._thread is None:
            return self.snapshots, self.errors
        self._stop_event.set()
        self._thread.join(timeout=max(self.interval_s * 4, 5.0))
        return self.snapshots, self.errors

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                snapshot = collect_service_metrics(self.base_url, self.served_model_name)
                snapshot["sample_phase"] = "during_run"
                self.snapshots.append(snapshot)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(
                    f"{datetime.now(timezone.utc).isoformat()} failed to sample service metrics: {exc}"
                )
            if self._stop_event.wait(self.interval_s):
                break


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


def _has_metric(
    samples: list[MetricSample],
    names: tuple[str, ...],
    predicate: Callable[[MetricSample], bool] | None = None,
) -> bool:
    for sample in samples:
        if sample.name not in names:
            continue
        if predicate is not None and not predicate(sample):
            continue
        return True
    return False


def _sum_http_success_metric(samples: list[MetricSample]) -> float:
    total = 0.0
    success_statuses = {"2xx", "200", "200 OK"}
    success_handlers = {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/messages",
        "/generate",
        "/invocations",
    }
    for sample in samples:
        if sample.name != "http_requests_total":
            continue
        if sample.labels.get("method") != "POST":
            continue
        if sample.labels.get("status") not in success_statuses:
            continue
        if sample.labels.get("handler") not in success_handlers:
            continue
        total += sample.value
    return total


def fetch_server_load(base_url: str, timeout: float = 5.0) -> float | None:
    url = f"{base_url.rstrip('/')}/load"
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
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    value = payload.get("server_load")
    return None if value is None else float(value)


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

    running_metric_names = ("vllm:num_requests_running",)
    waiting_metric_names = ("vllm:num_requests_waiting",)
    kv_metric_names = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
    prompt_metric_names = ("vllm:prompt_tokens_total", "vllm:prompt_tokens")
    generation_metric_names = ("vllm:generation_tokens_total", "vllm:generation_tokens")
    request_success_metric_names = ("vllm:request_success_total", "vllm:request_success")

    has_running_metric = _has_metric(samples, running_metric_names, predicate)
    has_waiting_metric = _has_metric(samples, waiting_metric_names, predicate)
    has_kv_metric = _has_metric(samples, kv_metric_names, predicate)
    has_prompt_metric = _has_metric(samples, prompt_metric_names, predicate)
    has_generation_metric = _has_metric(samples, generation_metric_names, predicate)
    has_request_success_metric = _has_metric(
        samples, request_success_metric_names, predicate
    )

    vllm_metric_names = sorted({
        sample.name for sample in samples if sample.name.startswith("vllm:")
    })
    http_request_success_total = _sum_http_success_metric(samples)

    snapshot = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "served_model_name": served_model_name,
        "vllm_metric_names": vllm_metric_names,
        "has_vllm_running_metric": has_running_metric,
        "has_vllm_waiting_metric": has_waiting_metric,
        "has_vllm_kv_metric": has_kv_metric,
        "has_vllm_prompt_metric": has_prompt_metric,
        "has_vllm_generation_metric": has_generation_metric,
        "has_vllm_request_success_metric": has_request_success_metric,
        "num_requests_running": _sum_metric(samples, running_metric_names, predicate)
        if has_running_metric
        else None,
        "num_requests_waiting": _sum_metric(samples, waiting_metric_names, predicate)
        if has_waiting_metric
        else None,
        "kv_cache_usage_perc": _sum_metric(samples, kv_metric_names, predicate)
        if has_kv_metric
        else None,
        "prompt_tokens_total": _sum_metric(samples, prompt_metric_names, predicate)
        if has_prompt_metric
        else None,
        "generation_tokens_total": _sum_metric(
            samples, generation_metric_names, predicate
        )
        if has_generation_metric
        else None,
        "http_request_success_total": http_request_success_total,
        "request_success_total": _sum_metric(
            samples, request_success_metric_names, predicate
        )
        if has_request_success_metric
        else http_request_success_total,
        "request_success_source": "vllm" if has_request_success_metric else "http_requests_total",
        "server_load": fetch_server_load(base_url),
        "gpu_memory_used_mb": query_gpu_memory_used_mb(),
        "raw_sample_count": len(samples),
    }
    return snapshot


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    weight = rank - lower_index
    return lower_value + (upper_value - lower_value) * weight


def _summarize_gauge_series(
    snapshots: list[dict],
    *,
    source_key: str,
    output_prefix: str,
) -> dict[str, float | int | None]:
    values = [
        float(snapshot[source_key])
        for snapshot in snapshots
        if snapshot.get(source_key) is not None
    ]
    if not values:
        return {
            f"{output_prefix}_sample_count": 0,
            f"{output_prefix}_avg": None,
            f"{output_prefix}_max": None,
            f"{output_prefix}_p95": None,
        }
    return {
        f"{output_prefix}_sample_count": len(values),
        f"{output_prefix}_avg": sum(values) / len(values),
        f"{output_prefix}_max": max(values),
        f"{output_prefix}_p95": _percentile(values, 95.0),
    }


def derive_service_delta(
    before: dict,
    after: dict,
    duration_s: float,
    benchmark_result: dict | None = None,
    samples_during_run: list[dict] | None = None,
    sample_interval_s: float | None = None,
    sampling_errors: list[str] | None = None,
) -> dict:
    duration_s = max(duration_s, 1e-9)
    samples_during_run = samples_during_run or []
    sampling_errors = sampling_errors or []

    def delta_or_none(before_value, after_value) -> float | None:
        if before_value is None or after_value is None:
            return None
        return max(0.0, float(after_value) - float(before_value))

    vllm_metrics_absent = not before.get("vllm_metric_names") and not after.get(
        "vllm_metric_names"
    )

    prompt_delta = delta_or_none(
        before.get("prompt_tokens_total"), after.get("prompt_tokens_total")
    )
    if vllm_metrics_absent:
        prompt_delta = None
    prompt_source = "vllm"
    if prompt_delta is None and benchmark_result is not None:
        prompt_delta = float(benchmark_result.get("total_input_tokens", 0.0))
        prompt_source = "benchmark_result"
    elif prompt_delta is None:
        prompt_delta = 0.0
        prompt_source = "unavailable"

    generation_delta = delta_or_none(
        before.get("generation_tokens_total"), after.get("generation_tokens_total")
    )
    if vllm_metrics_absent:
        generation_delta = None
    generation_source = "vllm"
    if generation_delta is None and benchmark_result is not None:
        generation_delta = float(benchmark_result.get("total_output_tokens", 0.0))
        generation_source = "benchmark_result"
    elif generation_delta is None:
        generation_delta = 0.0
        generation_source = "unavailable"

    success_delta = delta_or_none(
        before.get("request_success_total"), after.get("request_success_total")
    )
    if vllm_metrics_absent and (after.get("request_success_source") != "http_requests_total"):
        success_delta = None
    success_source = after.get("request_success_source") or before.get(
        "request_success_source", "vllm"
    )
    if success_delta is None and benchmark_result is not None:
        success_delta = float(benchmark_result.get("completed", 0.0))
        success_source = "benchmark_result"
    elif success_delta is None:
        success_delta = 0.0
        success_source = "unavailable"

    gpu_before = before.get("gpu_memory_used_mb")
    gpu_after = after.get("gpu_memory_used_mb")
    gpu_delta = (
        None
        if gpu_before is None or gpu_after is None
        else float(gpu_after) - float(gpu_before)
    )

    payload = {
        "duration_s": duration_s,
        "prompt_tokens_delta": prompt_delta,
        "generation_tokens_delta": generation_delta,
        "request_success_delta": success_delta,
        "prompt_throughput_toks_per_s": prompt_delta / duration_s,
        "generation_throughput_toks_per_s": generation_delta / duration_s,
        "prompt_tokens_source": prompt_source,
        "generation_tokens_source": generation_source,
        "request_success_source": success_source,
        "service_metrics_during_run_sample_count": len(samples_during_run),
        "service_metrics_sampling_interval_s": sample_interval_s,
        "service_metrics_sampling_error_count": len(sampling_errors),
        "service_metrics_sampling_errors": sampling_errors,
        "kv_cache_usage_perc_before": None if vllm_metrics_absent else before.get("kv_cache_usage_perc"),
        "kv_cache_usage_perc_after": None if vllm_metrics_absent else after.get("kv_cache_usage_perc"),
        "num_requests_running_before": None if vllm_metrics_absent else before.get("num_requests_running"),
        "num_requests_running_after": None if vllm_metrics_absent else after.get("num_requests_running"),
        "num_requests_waiting_before": None if vllm_metrics_absent else before.get("num_requests_waiting"),
        "num_requests_waiting_after": None if vllm_metrics_absent else after.get("num_requests_waiting"),
        "server_load_before": before.get("server_load"),
        "server_load_after": after.get("server_load"),
        "gpu_memory_used_mb_before": gpu_before,
        "gpu_memory_used_mb_after": gpu_after,
        "gpu_memory_used_mb_delta": gpu_delta,
        "vllm_metric_names_before": before.get("vllm_metric_names", []),
        "vllm_metric_names_after": after.get("vllm_metric_names", []),
    }

    for source_key, output_prefix in (
        ("num_requests_running", "num_requests_running_during_run"),
        ("num_requests_waiting", "num_requests_waiting_during_run"),
        ("kv_cache_usage_perc", "kv_cache_usage_perc_during_run"),
        ("server_load", "server_load_during_run"),
        ("gpu_memory_used_mb", "gpu_memory_used_mb_during_run"),
    ):
        payload.update(
            _summarize_gauge_series(
                samples_during_run,
                source_key=source_key,
                output_prefix=output_prefix,
            )
        )

    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
