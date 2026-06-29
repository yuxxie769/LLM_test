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
    parser = argparse.ArgumentParser(description='Validate model compare outputs.')
    parser.add_argument('--manifest', default='results/model_compare/manifest.json')
    parser.add_argument('--output-dir', default='results/model_compare')
    args = parser.parse_args()

    settings = load_settings()
    manifest_path = resolve_repo_path(settings.repo_root, args.manifest)
    output_dir = resolve_repo_path(settings.repo_root, args.output_dir)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    compare_csv = output_dir / 'model_compare.csv'
    long_csv = output_dir / 'model_compare_long.csv'
    summary_md = output_dir / 'model_compare_summary.md'
    plot_files = sorted((output_dir / 'plots').glob('*.png'))
    errors: list[str] = []

    if len(manifest.get('batches', [])) < 2:
        errors.append('manifest contains fewer than two batches')
    if int(manifest.get('common_cases', 0)) <= 0:
        errors.append('manifest reports no overlapping comparable cases')
    if not compare_csv.exists():
        errors.append('model_compare.csv is missing')
    if not long_csv.exists():
        errors.append('model_compare_long.csv is missing')
    if not summary_md.exists():
        errors.append('model_compare_summary.md is missing')
    if len(plot_files) < 3:
        errors.append('fewer than 3 compare plots were generated')
    if compare_csv.exists() and not load_csv_rows(compare_csv):
        errors.append('model_compare.csv has no rows')
    if long_csv.exists() and not load_csv_rows(long_csv):
        errors.append('model_compare_long.csv has no rows')

    summary = {
        'manifest_path': str(manifest_path),
        'output_dir': str(output_dir),
        'common_cases': manifest.get('common_cases', 0),
        'plot_files': [str(path) for path in plot_files],
        'errors': errors,
    }
    if errors:
        raise SystemExit(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
