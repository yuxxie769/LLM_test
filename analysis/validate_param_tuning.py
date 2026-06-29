from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.phase3_common import load_csv_rows, resolve_repo_path  # noqa: E402
from bench.config import load_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate Phase 3 parameter tuning outputs.')
    parser.add_argument(
        '--manifest',
        action='append',
        required=True,
        help='Path to results/param_tuning/raw/<sweep_run_id>/manifest.json. Provide multiple times to validate merged split sweeps.',
    )
    parser.add_argument('--output-dir', default='results/param_tuning')
    args = parser.parse_args()

    settings = load_settings()
    manifest_paths = [resolve_repo_path(settings.repo_root, value) for value in args.manifest]
    output_dir = resolve_repo_path(settings.repo_root, args.output_dir)
    manifests = [json.loads(path.read_text(encoding='utf-8')) for path in manifest_paths]

    csv_path = output_dir / 'param_tuning.csv'
    summary_md = output_dir / 'param_tuning_summary.md'
    plot_files = sorted((output_dir / 'plots').glob('*.png'))
    errors: list[str] = []

    if not csv_path.exists():
        errors.append('param_tuning.csv is missing')
        rows = []
    else:
        rows = load_csv_rows(csv_path)
        if not rows:
            errors.append('param_tuning.csv has no rows')

    completed_entries = []
    for manifest in manifests:
        completed_entries.extend(
            entry for entry in manifest.get('entries', []) if entry.get('status') == 'completed'
        )
    if completed_entries and len(rows) != len(completed_entries):
        errors.append('row count does not match number of completed sweep entries across manifests')
    if not summary_md.exists():
        errors.append('param_tuning_summary.md is missing')
    if len(plot_files) < 3:
        errors.append('fewer than 3 tuning plots were generated')

    summary = {
        'manifest_paths': [str(path) for path in manifest_paths],
        'completed_entries': len(completed_entries),
        'rows': len(rows),
        'plot_files': [str(path) for path in plot_files],
        'errors': errors,
    }
    if errors:
        raise SystemExit(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
