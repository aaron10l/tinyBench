"""Split a graded results JSON into per-template JSON files.

Usage:
    python scripts/split_results_by_template.py <input_file> <output_dir>

Creates <output_dir>/<template_id>.json for each unique template_id.
Does NOT modify the original file.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split graded results by template_id")
    parser.add_argument("input_file", type=Path, help="Path to graded JSON file")
    parser.add_argument("output_dir", type=Path, help="Directory to write per-template files")
    args = parser.parse_args()

    records = json.loads(args.input_file.read_text())
    by_template: dict[str, list] = defaultdict(list)
    for r in records:
        by_template[r["template_id"]].append(r)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for tid, recs in sorted(by_template.items()):
        out_path = args.output_dir / f"{tid}.json"
        out_path.write_text(json.dumps(recs, indent=2))
        print(f"{tid}: {len(recs)} records -> {out_path}")

    print(f"\nTotal: {len(records)} records split into {len(by_template)} files")


if __name__ == "__main__":
    main()
