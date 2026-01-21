# src/injectors/duplicate_feature.py
from __future__ import annotations

from typing import Any, Dict, Tuple
import hashlib
import numpy as np
import pandas as pd


def _feature_name_from_hash(source: str, salt: int) -> str:
    """
    Deterministically generate a 4-char hex hash and format as feature_{hash}.
    'salt' lets us retry if there's a name collision.
    """
    h = hashlib.sha256(f"{source}:{salt}".encode("utf-8")).hexdigest()[:4]
    return f"feature_{h}"


def inject_duplicate_feature(
    df: pd.DataFrame,
    params: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Copies source_col into a new column.

    params:
      - source_col: optional str (random non-row_id if omitted)
      - new_col: optional str (if provided, uses this exact name)
                otherwise generates 'feature_{4charhash}' and resolves collisions.
    """
    if "row_id" not in df.columns:
        raise ValueError("Expected 'row_id' column in standardized dataset.")

    out = df.copy()

    # Pick source column
    source = params.get("source_col")
    if source is None:
        candidates = [c for c in out.columns if c != "row_id"]
        source = str(rng.choice(candidates))
    if source not in out.columns:
        raise ValueError(f"source_col not in df: {source}")

    # Determine new column name
    requested_new = params.get("new_col")
    if requested_new is not None:
        new_col = str(requested_new)
        if new_col in out.columns:
            raise ValueError(f"new_col already exists: {new_col}")
    else:
        # Generate feature_{4charhash} and retry with a salt if it collides
        salt = 0
        new_col = _feature_name_from_hash(source, salt)
        while new_col in out.columns:
            salt += 1
            new_col = _feature_name_from_hash(source, salt)

    # Copy data
    out[new_col] = out[source]
    print(f"Created duplicate column '{new_col}' from '{source}'.")

    phenomenon = {
        "type": "duplicate_feature",
        "params": {"source_col": source, "new_col": new_col},
        "effects": {"created_column": new_col},
        "questions": [
            {
                "question": f"Which column is a duplicate of '{source}'?",
                "answer": new_col,
                "answer_format": "categorical",
                "template_id": "dup_find_copy_v0",
            },
            {
                "question": "Which pair of columns are identical for all rows (excluding row_id)? "
                            "Return the two column names sorted alphabetically.",
                "answer": sorted([source, new_col]),
                "answer_format": "categorical_list",
                "template_id": "dup_identical_pair_v0",
            },
            {
                "question": f"How many rows have {source} != {new_col}? Return an integer.",
                "answer": 0,
                "answer_format": "integer",
                "template_id": "dup_mismatch_count_v0",
            },
        ],
    }


    return out, phenomenon