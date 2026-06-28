from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.config import load_settings
from bench.run_matrix import expand_suite, load_matrix


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def expected_cases_from_matrix(
    *,
    repo_root: Path,
    matrix_path: str,
    suites: list[str],
) -> int:
    matrix = load_matrix(repo_root / matrix_path)
    return sum(len(expand_suite(suite, matrix[suite])) for suite in suites)


def resolve_expected_cases(
    *,
    settings,
    manifest: dict[str, Any] | None,
    explicit_expected_cases: int | None,
) -> int | None:
    if explicit_expected_cases is not None:
        return explicit_expected_cases

    if not manifest:
        return None

    planned_cases = manifest.get("planned_cases")
    if planned_cases is not None:
        return int(planned_cases)

    matrix_value = manifest.get("matrix")
    suites = manifest.get("requested_suites", [])
    if not matrix_value or not suites:
        return None

    matrix_path = Path(matrix_value)
    if matrix_path.is_absolute():
        matrix_path = matrix_path.relative_to(settings.repo_root)

    return expected_cases_from_matrix(
        repo_root=settings.repo_root,
        matrix_path=matrix_path.as_posix(),
        suites=suites,
    )


def validate_batch(
    *,
    settings,
    batch_run_id: str,
    output_dir: Path,
    expected_cases: int | None = None,
    allow_missing_manifest: bool = False,
) -> dict[str, Any]:
    manifest_path = settings.raw_benchmark_dir / batch_run_id / "manifest.json"
    manifest = load_manifest(manifest_path) if manifest_path.exists() else None

    if manifest is None and not allow_missing_manifest:
        raise FileNotFoundError(manifest_path)

    suites = manifest.get("requested_suites", []) if manifest else []
    resolved_expected_cases = resolve_expected_cases(
        settings=settings,
        manifest=manifest,
        explicit_expected_cases=expected_cases,
    )
    combined_files = sorted(
        (settings.raw_benchmark_dir / batch_run_id).glob("*.combined.json")
    )
    benchmark_csv = output_dir / "baseline_metrics.csv"
    service_csv = output_dir / "baseline_service_metrics.csv"
    summary_md = output_dir / "baseline_summary.md"
    plot_files = sorted((output_dir / "plots").glob("*.png"))

    benchmark_rows = load_csv_rows(benchmark_csv)
    service_rows = load_csv_rows(service_csv)

    errors: list[str] = []

    if manifest is None:
        if not allow_missing_manifest:
            errors.append("manifest.json is missing")
    else:
        if int(manifest.get("failed_cases", 0)) != 0:
            errors.append(
                f"manifest reports failed_cases={int(manifest.get('failed_cases', 0))}"
            )
        if int(manifest.get("completed_cases", 0)) != len(combined_files):
            errors.append(
                "manifest completed_cases does not match number of combined json files"
            )
    if len(benchmark_rows) != len(combined_files):
        errors.append("benchmark csv row count does not match combined json count")
    if len(service_rows) != len(combined_files):
        errors.append("service csv row count does not match combined json count")
    incomplete_rows = [
        row["case_id"]
        for row in benchmark_rows
        if float(row.get("completed", "0") or 0) < 40 and float(row.get("failed", "0") or 0) == 0
    ]
    if incomplete_rows:
        errors.append(
            "benchmark rows with completed < 40 despite zero failures: " + ", ".join(incomplete_rows)
        )
    if not summary_md.exists():
        errors.append("baseline_summary.md is missing")
    if resolved_expected_cases is not None and resolved_expected_cases != len(combined_files):
        errors.append(
            f"expected_cases={resolved_expected_cases} does not match combined json count={len(combined_files)}"
        )

    summary = {
        "batch_run_id": batch_run_id,
        "manifest_path": str(manifest_path),
        "manifest_present": manifest is not None,
        "requested_suites": suites,
        "expected_cases": resolved_expected_cases,
        "completed_cases": int(manifest.get("completed_cases", 0)) if manifest else None,
        "combined_files": len(combined_files),
        "benchmark_rows": len(benchmark_rows),
        "service_rows": len(service_rows),
        "summary_path": str(summary_md),
        "plot_files": [str(path) for path in plot_files],
        "errors": errors,
    }

    if errors:
        raise SystemExit(json.dumps(summary, ensure_ascii=False, indent=2))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Phase 2 batch output.")
    parser.add_argument("--batch-run-id", required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Batch output directory, e.g. results/batches/<batch_run_id>.",
    )
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=None,
        help="Optional explicit expected case count, used for ad-hoc or partial runs.",
    )
    parser.add_argument(
        "--allow-missing-manifest",
        action="store_true",
        help="Allow validation to proceed when manifest.json is absent.",
    )
    args = parser.parse_args()

    settings = load_settings()
    summary = validate_batch(
        settings=settings,
        batch_run_id=args.batch_run_id,
        output_dir=Path(args.output_dir),
        expected_cases=args.expected_cases,
        allow_missing_manifest=args.allow_missing_manifest,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
