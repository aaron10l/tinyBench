from __future__ import annotations

import csv
import dataclasses
import json
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Any

def load_dataset_summary(summary_path: Path) -> Dict[str, Any]:
    """Load a dataset summary JSON."""
    return json.loads(summary_path.read_text())

def load_templates(templates_dir: Path) -> List[tuple[Path, Dict[str, Any]]]:
    """Load all template JSONs recursively, excluding template_schema.json."""
    templates = []
    for template_file in templates_dir.rglob("*.json"):
        # Skip the schema file
        if template_file.name == "template_schema.json":
            continue
        try:
            template = json.loads(template_file.read_text())
            templates.append((template_file, template))
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping invalid JSON in {template_file.relative_to(templates_dir)}: {e}")
        except Exception as e:
            print(f"Warning: Error loading {template_file.relative_to(templates_dir)}: {e}")
    return templates

def _get_feature_pool(columns: Dict[str, Any], target: str, feature_pool: Dict[str, Any]) -> List[str]:
    exclude_target = feature_pool.get("exclude_target", False)
    exclude_kinds = set(feature_pool.get("exclude_kinds", []))
    exclude_dtypes = set(feature_pool.get("exclude_dtypes", []))

    pool = []
    for col_name, col_info in columns.items():
        if exclude_target and col_name == target:
            continue
        if col_info.get("kind") in exclude_kinds:
            continue
        if col_info.get("dtype") in exclude_dtypes:
            continue
        pool.append(col_name)
    return pool

def _column_matches_constraints(col_info: Dict[str, Any], slot_spec: Dict[str, Any]) -> bool:
    if "kind_any_of" in slot_spec:
        if col_info.get("kind") not in slot_spec["kind_any_of"]:
            return False
    if "dtype_any_of" in slot_spec:
        if col_info.get("dtype") not in slot_spec["dtype_any_of"]:
            return False
    unique_count = col_info.get("unique_count", 0)
    if "unique_count_min" in slot_spec and unique_count < slot_spec["unique_count_min"]:
        return False
    if "unique_count_max" in slot_spec and unique_count > slot_spec["unique_count_max"]:
        return False
    return True

def _load_column_values(dataset_path: Path, columns: List[str]) -> Dict[str, List[str]]:
    values_by_column: Dict[str, set] = {col: set() for col in columns}
    if not dataset_path.exists():
        return {col: [] for col in columns}

    with dataset_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            for col in columns:
                val = row.get(col)
                if val is None:
                    continue
                values_by_column[col].add(val)

    return {col: sorted(values) for col, values in values_by_column.items()}

def _render_question(question: str, assignments: Dict[str, str]) -> str:
    class _Missing(dict):
        def __missing__(self, key: str) -> str:
            return f"{{{key}}}"

    return question.format_map(_Missing(assignments))

def assign_slot_values(
    template: Dict[str, Any],
    summary: Dict[str, Any],
    dataset_path: Path,
) -> tuple[bool, List[str], Dict[str, str]]:
    is_compatible, reasons, slot_matches = check_requirements(template, summary)
    if not is_compatible:
        return is_compatible, reasons, {}

    slots = template.get("slots", {})
    assigned: Dict[str, str] = {}
    used_columns = set()

    # Assign target-sourced columns first
    for slot_name, slot_spec in slots.items():
        if slot_spec.get("type") != "column":
            continue
        if slot_spec.get("source") != "dataset.target":
            continue
        matches = slot_matches.get(slot_name, [])
        if matches:
            assigned[slot_name] = matches[0]
            used_columns.add(matches[0])

    # Assign remaining column slots without reusing columns
    for slot_name, slot_spec in slots.items():
        if slot_spec.get("type") != "column":
            continue
        if slot_spec.get("source") == "dataset.target":
            continue
        matches = slot_matches.get(slot_name, [])
        chosen = next((col for col in matches if col not in used_columns), None)
        if chosen:
            assigned[slot_name] = chosen
            used_columns.add(chosen)

    # Prepare values for column_values slots
    value_slots = {
        slot_name: slot_spec
        for slot_name, slot_spec in slots.items()
        if slot_spec.get("source") == "column_values"
    }
    needed_columns = []
    for slot_spec in value_slots.values():
        source_column_slot = slot_spec.get("source_column")
        if source_column_slot and source_column_slot in assigned:
            needed_columns.append(assigned[source_column_slot])

    values_by_column = _load_column_values(dataset_path, needed_columns)
    used_values: Dict[str, set] = {col: set() for col in needed_columns}

    for slot_name, slot_spec in value_slots.items():
        source_column_slot = slot_spec.get("source_column")
        if not source_column_slot:
            continue
        source_column = assigned.get(source_column_slot)
        if not source_column:
            continue
        candidates = values_by_column.get(source_column, [])
        available = [val for val in candidates if val not in used_values[source_column]]
        if not available:
            continue

        # Determine which position to pick from
        position = slot_spec.get("position", "random")
        if position == "middle":
            chosen = available[len(available) // 2]
        else:  # "random" or default
            chosen = random.choice(available)

        assigned[slot_name] = chosen
        used_values[source_column].add(chosen)

    # Fill scalar slots (int/float/string) with defaults
    for slot_name, slot_spec in slots.items():
        if slot_name in assigned:
            continue
        slot_type = slot_spec.get("type")
        if slot_type in {"int", "float", "string"}:
            if "default" in slot_spec and slot_spec["default"] is not None:
                assigned[slot_name] = str(slot_spec["default"])
            elif "min" in slot_spec and slot_spec["min"] is not None:
                assigned[slot_name] = str(slot_spec["min"])

    return is_compatible, reasons, assigned

def check_requirements(template: Dict[str, Any], summary: Dict[str, Any]) -> tuple[bool, List[str], Dict[str, List[str]]]:
    """
    Validate if a template's requirements and slot constraints are met by the dataset.
    Returns (is_compatible, reasons) where reasons is a list of why it's incompatible.
    """
    requirements = template.get("requirements", [])
    slots = template.get("slots", {})
    feature_pool = template.get("feature_pool", {})
    columns = summary.get("columns", {})
    by_kind = summary.get("by_kind", {})
    target = summary.get("target", "")
    
    reasons = []
    slot_matches: Dict[str, List[str]] = {}
    
    # Check slot constraints for column-type slots
    for slot_name, slot_spec in slots.items():
        if slot_spec.get("type") != "column":
            continue
        
        # checking if the slot sources from the target column
        slot_source = slot_spec.get("source")
        if slot_source == "dataset.target":
            if not target or target not in columns:
                reasons.append(f"Slot '{slot_name}' requires a target column, but none is defined")
                slot_matches[slot_name] = []
                continue
            col_info = columns[target]
            if not _column_matches_constraints(col_info, slot_spec):
                reasons.append(
                    f"Slot '{slot_name}' requires target column to match constraints, but '{target}' does not"
                )
                slot_matches[slot_name] = []
            else:
                slot_matches[slot_name] = [target]
            continue

        # Otherwise, check that at least one column can satisfy the slot
        if not columns:
            reasons.append(f"Slot '{slot_name}' requires a column, but dataset has none")
            slot_matches[slot_name] = []
            continue

        matching_cols = [
            col_name
            for col_name, col_info in columns.items()
            if _column_matches_constraints(col_info, slot_spec)
        ]
        slot_matches[slot_name] = matching_cols
        if not matching_cols:
            reasons.append(f"Slot '{slot_name}' has no compatible columns in the dataset")

    # Check slots that derive values from another column slot
    for slot_name, slot_spec in slots.items():
        if slot_spec.get("source") != "column_values":
            continue
        source_column_slot = slot_spec.get("source_column")
        if not source_column_slot:
            reasons.append(f"Slot '{slot_name}' requires 'source_column' to be set")
            slot_matches[slot_name] = []
            continue
        source_slot_spec = slots.get(source_column_slot)
        if not source_slot_spec or source_slot_spec.get("type") != "column":
            reasons.append(
                f"Slot '{slot_name}' requires source_column '{source_column_slot}' to be a column slot"
            )
            slot_matches[slot_name] = []
            continue
        source_matches = slot_matches.get(source_column_slot, [])
        if not source_matches:
            reasons.append(
                f"Slot '{slot_name}' requires source_column '{source_column_slot}' to have matches"
            )
            slot_matches[slot_name] = []
            continue
        slot_matches[slot_name] = [f"<values of {source_column_slot}>"]
    
    # Check requirements
    for req in requirements:
        req_type = req.get("type")
        
        if req_type == "min_feature_count":
            min_count = req.get("min", 0)
            feature_count = len(_get_feature_pool(columns, target, feature_pool))
            if feature_count < min_count:
                reasons.append(f"Requires at least {min_count} features, but dataset has {feature_count}")
        
        elif req_type == "min_count_by_kind":
            kind = req.get("kind")
            min_count = req.get("min", 0)
            actual_count = len(by_kind.get(kind, []))
            if actual_count < min_count:
                reasons.append(f"Requires at least {min_count} columns of kind '{kind}', but dataset has {actual_count}")
        
        elif req_type == "min_count_by_dtype":
            dtype = req.get("dtype")
            min_count = req.get("min", 0)
            actual_count = sum(1 for c in columns.values() if c.get("dtype") == dtype)
            if actual_count < min_count:
                reasons.append(f"Requires at least {min_count} columns of dtype '{dtype}', but dataset has {actual_count}")
        
        elif req_type == "min_count_by_dtype_any_of":
            dtypes = req.get("dtypes", [])
            min_count = req.get("min", 0)
            actual_count = sum(1 for c in columns.values() if c.get("dtype") in dtypes)
            if actual_count < min_count:
                reasons.append(f"Requires at least {min_count} columns of dtype {dtypes}, but dataset has {actual_count}")
    
    return (len(reasons) == 0, reasons, slot_matches)

@dataclasses.dataclass
class TemplateMatch:
    template: Dict[str, Any]
    template_path: Path
    is_compatible: bool
    reasons: List[str]
    slot_assignments: Dict[str, str]
    feature_pool: List[str]
    rendered_question: str


def get_template_matches(
    summary_path: Path,
    dataset_path: Path,
    templates_dir: Path,
    seed: int = 42,
) -> List[TemplateMatch]:
    """Return a list of TemplateMatch objects for all templates."""
    # Seed Python's random module for deterministic slot assignment
    path_hash = int(hashlib.md5(str(summary_path).encode()).hexdigest(), 16) % (2**31)
    random.seed(seed + path_hash)
    summary = load_dataset_summary(summary_path)
    templates = load_templates(templates_dir)
    columns = summary.get("columns", {})
    target = summary.get("target", "")

    matches: List[TemplateMatch] = []
    for template_path, template in templates:
        is_compatible, reasons, assignments = assign_slot_values(
            template, summary, dataset_path,
        )
        fp = _get_feature_pool(
            columns, target, template.get("feature_pool", {}),
        )
        rendered = _render_question(template.get("question", ""), assignments)
        matches.append(TemplateMatch(
            template=template,
            template_path=template_path,
            is_compatible=is_compatible,
            reasons=reasons,
            slot_assignments=assignments,
            feature_pool=fp,
            rendered_question=rendered,
        ))
    return matches


def find_applicable_templates(summary_path: Path, templates_dir: Path) -> List[tuple[Path, Dict]]:
    """
    Find all templates applicable to a dataset.
    Returns list of (template_path, template_dict) tuples.
    """
    summary = load_dataset_summary(summary_path)
    templates = load_templates(templates_dir)
    
    applicable = []
    for template_path, template in templates:
        is_compatible, reasons, _slot_matches = check_requirements(template, summary)
        if is_compatible:
            applicable.append((template_path, template))
    
    return applicable

if __name__ == "__main__":
    summary_path = Path("data/standardized/summaries/bike_sharing_10k.json")
    templates_dir = Path("templates")
    
    summary = load_dataset_summary(summary_path)
    dataset_path = Path("data/standardized") / summary.get("dataset", "")
    templates = load_templates(templates_dir)
    
    applicable = []
    incompatible = []
    
    for template_path, template in templates:
        is_compatible, reasons, assignments = assign_slot_values(
            template,
            summary,
            dataset_path,
        )
        if is_compatible:
            applicable.append((template_path, template))
            print(f"  {template['template_id']}")
            print(f"    Category: {template['category']}")
            print(f"    File: {template_path.relative_to(templates_dir)}")
            if assignments:
                print(f"    Slot assignments:")
                for slot_name, value in assignments.items():
                    print(f"      - {slot_name}: {value}")
            rendered_question = _render_question(template.get("question", ""), assignments)
            if rendered_question:
                print(f"    Rendered question:")
                print(f"      {rendered_question}")
            print()
        else:
            incompatible.append((template_path, template, reasons))
    
    print(f"Found {len(applicable)} applicable templates:\n")
    
    if incompatible:
        print(f"\nFound {len(incompatible)} incompatible templates:\n")
        for path, tmpl, reasons in incompatible:
            print(f"  {tmpl['template_id']}")
            print(f"    Category: {tmpl['category']}")
            print(f"    File: {path.relative_to(templates_dir)}")
            print(f"    Reasons:")
            for reason in reasons:
                print(f"      - {reason}")
            print()