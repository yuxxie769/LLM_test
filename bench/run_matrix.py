from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from bench.benchmark_backends import BenchmarkCase
from bench.config import load_settings
from bench.run_single_case import run_case


def load_matrix(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def expand_suite(name: str, spec: dict[str, Any]) -> list[BenchmarkCase]:
    dims = spec["dimensions"]
    repeat = int(spec.get("repeat", 1))
    mode = spec.get("mode", "non_stream")
    num_prompts = spec.get("num_prompts")
    request_rate = spec.get("request_rate")

    combinations = itertools.product(
        dims["concurrency"],
        dims["input_tokens"],
        dims["output_tokens"],
    )

    cases: list[BenchmarkCase] = []
    for concurrency, input_tokens, output_tokens in combinations:
        for repeat_index in range(1, repeat + 1):
            cases.append(
                BenchmarkCase(
                    suite=name,
                    mode=mode,
                    concurrency=int(concurrency),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    repeat_index=repeat_index,
                    num_prompts=num_prompts,
                    request_rate=request_rate,
                )
            )
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 2 benchmark matrix.")
    parser.add_argument(
        "--matrix",
        default="bench/matrix.yaml",
        help="Path to the matrix YAML file.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        help="Specific suite(s) to run. Defaults to all suites in matrix.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N expanded cases.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded cases without executing.",
    )
    parser.add_argument(
        "--batch-run-id",
        default=None,
        help="Explicit batch run id. Defaults to the current UTC timestamp.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()
    matrix_path = settings.repo_root / args.matrix
    matrix = load_matrix(matrix_path)

    suites = args.suite or list(matrix.keys())
    cases: list[BenchmarkCase] = []
    for suite_name in suites:
        if suite_name not in matrix:
            raise KeyError(f"Unknown suite: {suite_name}")
        cases.extend(expand_suite(suite_name, matrix[suite_name]))

    if args.limit is not None:
        cases = cases[: args.limit]

    if args.dry_run:
        print(
            json.dumps(
                [case.__dict__ for case in cases],
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    batch_run_id = args.batch_run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for case in cases:
        try:
            result = run_case(batch_run_id=batch_run_id, case=case, settings=settings)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            failures.append({"case": case.__dict__, "error": str(exc)})
            break

    manifest = {
        "batch_run_id": batch_run_id,
        "matrix": str(matrix_path),
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "requested_suites": suites,
        "run_mode": "matrix",
        "planned_cases": len(cases),
        "completed_cases": len(results),
        "failed_cases": len(failures),
        "stopped_early": bool(failures),
        "results": results,
        "failures": failures,
    }
    manifest_path = settings.raw_benchmark_dir / batch_run_id / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
