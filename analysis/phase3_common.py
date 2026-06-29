from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

CASE_FIELDS = (
    'suite',
    'mode',
    'concurrency',
    'input_tokens',
    'output_tokens',
)


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def safe_float(value: Any) -> float:
    if value in (None, ''):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: Any) -> int:
    if value in (None, ''):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def mean_metric(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [safe_float(row.get(key)) for row in rows if row.get(key) not in (None, '')]
    if not values:
        return 0.0
    return sum(values) / len(values)


def case_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, '')) for field in CASE_FIELDS)


def sort_case_key(key: tuple[str, ...]) -> tuple[Any, ...]:
    suite, mode, concurrency, input_tokens, output_tokens = key
    return (
        suite,
        mode,
        safe_int(concurrency),
        safe_int(input_tokens),
        safe_int(output_tokens),
    )
