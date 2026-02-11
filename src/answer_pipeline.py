"""Answer pipeline — compute answers for generated instances.

Walks data/instances/**/manifest.json, runs the answer computers, and
updates questions_and_answers.json and manifest.json in place.

Usage:
    python src/answer_pipeline.py                          # all instances
    python src/answer_pipeline.py --instances-dir <path>   # custom dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from answer_computers import COMPUTE_FN
from io_utils import load_csv, load_json, save_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    """Print a simple aligned table with box-drawing borders."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        padded = [c.ljust(col_widths[i]) for i, c in enumerate(cells)]
        return "  " + "  ".join(padded)

    divider = "  " + "  ".join("-" * w for w in col_widths)

    print(fmt_row(headers))
    print(divider)
    for row in rows:
        print(fmt_row(row))


def _compute_answer_safe(
    template_id: str,
    df,
    slot_assignments: Dict[str, str],
    effects: Dict[str, Any],
) -> tuple:
    """Try to compute an answer; return (answer, status).

    status is one of "OK", "SKIP", or "ERROR".
    """
    compute_fn = COMPUTE_FN.get(template_id)
    if compute_fn is None:
        return None, "SKIP"
    try:
        answer = compute_fn(df, slot_assignments, effects)
        return answer, "OK"
    except NotImplementedError:
        return None, "SKIP"
    except Exception as exc:
        print(f"  ERROR computing {template_id}: {exc}")
        return None, "ERROR"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def process_instance(manifest_path: Path) -> List[List[str]]:
    """Process a single instance directory. Returns table rows."""
    manifest = load_json(manifest_path)
    instance_dir = manifest_path.parent

    dataset_name = manifest["dataset_name"]
    injector_type = manifest["phenomenon"]["injector_type"]

    df = load_csv(instance_dir / "table.csv")

    effects = {injector_type: manifest["phenomenon"]["effects"]}

    rows: List[List[str]] = []
    updated = False

    for qa in manifest["qa_pairs"]:
        template_id = qa["template_id"]
        answer, status = _compute_answer_safe(
            template_id, df, qa["slot_assignments"], effects,
        )
        if status == "OK":
            qa["answer"] = answer
            updated = True
        rows.append([dataset_name, injector_type, template_id, repr(answer), status])

    if updated:
        save_json(manifest, manifest_path)
        save_json(manifest["qa_pairs"], instance_dir / "questions_and_answers.json")

    return rows


def run(instances_dir: Path) -> None:
    """Discover all instances and compute answers."""
    manifests = sorted(instances_dir.glob("**/manifest.json"))
    if not manifests:
        print(f"No manifest.json files found under {instances_dir}")
        sys.exit(1)

    print(f"Found {len(manifests)} instance(s) under {instances_dir}\n")

    all_rows: List[List[str]] = []
    for mp in manifests:
        all_rows.extend(process_instance(mp))

    _print_table(
        ["Dataset", "Injector", "Template", "Answer", "Status"],
        all_rows,
    )

    counts = {}
    for row in all_rows:
        status = row[-1]
        counts[status] = counts.get(status, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"\nTotal: {len(all_rows)} QA pairs ({summary})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute answers for generated instances")
    parser.add_argument(
        "--instances-dir",
        type=str,
        default="data/instances",
        help="Root directory containing instance folders (default: data/instances)",
    )
    args = parser.parse_args()
    run(Path(args.instances_dir))


if __name__ == "__main__":
    main()
