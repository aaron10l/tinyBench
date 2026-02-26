"""Phenomena injection pipeline.

Matches templates to a dataset, resolves injector parameters from slot
assignments, applies phenomena injections, and generates QA pairs.

Usage:
    python src/phenomena_pipeline.py --summary data/standardized/summaries/bike_sharing_10000.json --seed 42

TODO: 
- Fix cause of FutureWarning: Setting an item of incompatible dtype is deprecated. Currently suppressing with warnings.filterwarnings.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from find_applicable_templates import get_template_matches, TemplateMatch
from injectors import INJECT_FN
from answer_computers import COMPUTE_FN
from io_utils import load_csv, save_csv, load_json, save_json


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PhenomenaResult:
    injector_type: str
    params: Dict[str, Any]
    effects: Dict[str, Any]


# ---------------------------------------------------------------------------
# Parameter resolution
# ---------------------------------------------------------------------------

_SLOT_RE = re.compile(r"^\{(\w+)\}$")


def resolve_injector_params(
    param_mapping: Dict[str, str],
    slot_assignments: Dict[str, str],
) -> Dict[str, str]:
    """Substitute ``{SLOT_NAME}`` references with concrete slot values."""
    resolved: Dict[str, str] = {}
    for param_name, value in param_mapping.items():
        m = _SLOT_RE.match(value)
        if m:
            slot_name = m.group(1)
            resolved[param_name] = slot_assignments.get(slot_name, value)
        else:
            resolved[param_name] = value
    return resolved


# ---------------------------------------------------------------------------
# Phenomena collection (with deduplication)
# ---------------------------------------------------------------------------

def _freeze(params: Dict[str, Any]) -> tuple:
    return tuple(sorted(params.items()))


def collect_required_phenomena(
    matches: List[TemplateMatch],
) -> List[Dict[str, Any]]:
    """Gather unique injector specs from compatible template matches."""
    seen: set[tuple] = set()
    specs: List[Dict[str, Any]] = []

    for m in matches:
        if not m.is_compatible:
            continue
        for phenom in m.template.get("phenomena", []):
            injector = phenom["injector"]
            resolved = resolve_injector_params(
                phenom.get("param_mapping", {}),
                m.slot_assignments,
            )
            key = (injector, _freeze(resolved))
            if key not in seen:
                seen.add(key)
                specs.append({"type": injector, "params": resolved})

    # import json
    # print(json.dumps(specs, indent=2))
    return specs


# ---------------------------------------------------------------------------
# Injection runner
# ---------------------------------------------------------------------------

def run_injection_pipeline(
    df: pd.DataFrame,
    phenomena_specs: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, List[PhenomenaResult]]:
    """Apply injectors sequentially and collect results."""
    results: List[PhenomenaResult] = []
    for spec in phenomena_specs:
        injector_type = spec["type"]
        params = spec["params"]
        inject_fn = INJECT_FN.get(injector_type)
        if inject_fn is None:
            print(f"Warning: unknown injector '{injector_type}', skipping")
            continue
        df, phenom_dict = inject_fn(df, params, rng)
        results.append(PhenomenaResult(
            injector_type=phenom_dict["type"],
            params=phenom_dict["params"],
            effects=phenom_dict["effects"],
        ))
    return df, results


# ---------------------------------------------------------------------------
# QA generation
# ---------------------------------------------------------------------------

def _compute_answer_safe(
    template_id: str,
    df: pd.DataFrame,
    slot_assignments: Dict[str, str],
    effects: Dict[str, Any],
) -> Any:
    compute_fn = COMPUTE_FN.get(template_id)
    if compute_fn is None:
        return None
    try:
        return compute_fn(df, slot_assignments, effects)
    except Exception as exc:
        print(f"  Warning: answer computer {template_id} failed: {str(exc)[:80]}")
        return None


def generate_qa_pairs(
    matches: List[TemplateMatch],
    df: pd.DataFrame,
    phenomena_results: List[PhenomenaResult],
) -> List[Dict[str, Any]]:
    """Build a list of QA dicts from compatible template matches."""
    # Index effects by injector type for lookup
    effects_by_type: Dict[str, Dict[str, Any]] = {}
    for r in phenomena_results:
        effects_by_type[r.injector_type] = r.effects

    qa_pairs: List[Dict[str, Any]] = []
    for m in matches:
        if not m.is_compatible:
            continue
        template_id = m.template["template_id"]

        # Collect effects relevant to this template's phenomena
        template_effects: Dict[str, Any] = {}
        for phenom in m.template.get("phenomena", []):
            injector = phenom["injector"]
            if injector in effects_by_type:
                template_effects[injector] = effects_by_type[injector]

        answer = _compute_answer_safe(
            template_id, df, m.slot_assignments, template_effects,
        )

        qa_pairs.append({
            "template_id": template_id,
            "category": m.template.get("category", ""),
            "answer_format": m.template.get("answer_format", ""),
            "question": m.rendered_question,
            "slot_assignments": m.slot_assignments,
            "answer": answer,
        })

    return qa_pairs


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def _get_templates_for_phenomenon(
    matches: List[TemplateMatch],
    injector_type: str,
) -> List[TemplateMatch]:
    """Return templates that use the given injector type."""
    result = []
    for m in matches:
        if not m.is_compatible:
            continue
        for phenom in m.template.get("phenomena", []):
            if phenom["injector"] == injector_type:
                result.append(m)
                break
    return result


def _fmt_params(params: dict, max_len: int = 50) -> str:
    """Format params dict as compact key=val string, truncated if needed."""
    parts = [f"{k}={v}" for k, v in params.items()]
    s = ", ".join(parts)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


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


def build_instance(
    summary_path: str | Path,
    seed: int,
    templates_dir: str | Path = "templates",
) -> List[Path]:
    """Full pipeline: load -> match -> inject one phenomenon per instance -> QA -> save.

    Creates one output directory per phenomenon, each with its own table.csv
    containing only that single phenomenon injection.
    """
    summary_path = Path(summary_path)
    templates_dir = Path(templates_dir)
    summary = load_json(summary_path)

    # Derive paths from the summary
    dataset_filename = summary["dataset"]
    dataset_dir = summary_path.parent.parent  # up from summaries/ to standardized/
    base_dataset = dataset_dir / dataset_filename
    target = summary.get("target", "")
    dataset_name = Path(dataset_filename).stem

    # 1. Load dataset
    df_original = load_csv(base_dataset)

    # 2. Get template matches
    matches = get_template_matches(summary_path, base_dataset, templates_dir, seed=seed)

    compatible = [m for m in matches if m.is_compatible]

    # 3. Collect required phenomena
    phenomena_specs = collect_required_phenomena(matches)

    # 4. Create one instance per phenomenon
    output_dirs: List[Path] = []
    table_rows: List[List[str]] = []

    id_no_cols = summary.get("by_kind", {}).get("id_no", [])

    for spec in phenomena_specs:
        injector_type = spec["type"]
        params = spec["params"]

        if injector_type == "fi_leakage_topk" and id_no_cols:
            params = {**params, "id_no_cols": id_no_cols}

        # Start from fresh copy of original data
        df = df_original.copy()

        # Create a fresh RNG deterministically from seed + injector identity
        rng_key = f"{seed}:{injector_type}:{sorted(params.items())}"
        rng_seed = int(hashlib.md5(rng_key.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(rng_seed)

        # Run single injection
        inject_fn = INJECT_FN.get(injector_type)
        if inject_fn is None:
            table_rows.append([injector_type, "-", _fmt_params(params), "SKIP"])
            continue

        try:
            df_injected, phenom_dict = inject_fn(df, params, rng)
        except Exception as exc:
            reason = str(exc)[:30] + "..." if len(str(exc)) > 60 else str(exc)
            table_rows.append([injector_type, "-", _fmt_params(params), f"SKIP ({reason})"])
            continue

        phenom_result = PhenomenaResult(
            injector_type=phenom_dict["type"],
            params=phenom_dict["params"],
            effects=phenom_dict["effects"],
        )

        # Get templates that use this specific phenomenon
        relevant_matches = _get_templates_for_phenomenon(matches, injector_type)

        # Generate QA pairs only for relevant templates
        qa_pairs = generate_qa_pairs(relevant_matches, df_injected, [phenom_result])

        # Save outputs to phenomenon-specific directory
        out_dir = dataset_dir.parent / "instances" / dataset_name / f"seed_{seed}" / injector_type
        out_dir.mkdir(parents=True, exist_ok=True)

        save_csv(df_injected, out_dir / "table.csv")

        manifest = {
            "dataset_name": dataset_name,
            "seed": seed,
            "base_dataset": str(base_dataset),
            "target": target,
            "phenomenon": {
                "injector_type": phenom_result.injector_type,
                "params": phenom_result.params,
                "effects": phenom_result.effects,
            },
            "templates_applied": [
                m.template["template_id"]
                for m in relevant_matches
            ],
            "qa_pairs": qa_pairs,
        }
        save_json(manifest, out_dir / "manifest.json")
        save_json(qa_pairs, out_dir / "questions_and_answers.json")

        template_ids = ", ".join(m.template["template_id"] for m in relevant_matches)
        table_rows.append([injector_type, template_ids, _fmt_params(params), "OK"])
        output_dirs.append(out_dir)

    # Print compact summary
    print(f"\n{dataset_name}  ({len(compatible)}/{len(matches)} templates, seed {seed})")
    _print_table(
        ["Injector", "Template", "Params", "Status"],
        table_rows,
    )

    return output_dirs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Setting an item of incompatible dtype is deprecated.*",
        category=FutureWarning,
    )
    parser = argparse.ArgumentParser(description="Phenomena injection pipeline")
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Path to dataset summary JSON. If omitted, runs on all summaries in data/standardized/summaries/.",
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default="templates",
        help="Path to templates directory (default: templates)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    if args.summary:
        build_instance(args.summary, args.seed, args.templates_dir)
    else:
        summaries_dir = Path("data/standardized/summaries")
        summary_files = sorted(summaries_dir.glob("*.json"))
        if not summary_files:
            print(f"No summary files found in {summaries_dir}")
            return
        for summary_path in summary_files:
            build_instance(str(summary_path), args.seed, args.templates_dir)


if __name__ == "__main__":
    main()
