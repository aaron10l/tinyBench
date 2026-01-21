# src/injectors/mcar_missingness.py
from __future__ import annotations

from typing import Any, Dict, Tuple, List
import numpy as np
import pandas as pd


def inject_mcar_missingness(
    df: pd.DataFrame,
    params: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Randomly nulls a fraction of cells in selected columns (MCAR).
    params:
      - mcar_rate: float in (0,1)
      - columns: optional list[str] (defaults to all except row_id)
    """
    if "row_id" not in df.columns:
        raise ValueError("Expected 'row_id' column in standardized dataset.")

    rate = float(params.get("mcar_rate", -1))
    if not (0.0 < rate < 1.0):
        raise ValueError("mcar_rate must be in (0, 1).")

    cols: List[str] = params.get("columns") or [c for c in df.columns if c != "row_id"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Unknown columns for MCAR: {missing}")

    out = df.copy()
    n_rows = len(out)
    n_cols = len(cols)

    total_cells = n_rows * n_cols
    k = int(round(rate * total_cells))

    # If nothing will be nulled, still return phenomenon + questions (answers will reflect no injection)
    if k <= 0:
        nulled_cells = 0
    else:
        flat = rng.choice(total_cells, size=k, replace=False)
        r_idx = flat // n_cols
        c_idx = flat % n_cols

        cols_arr = np.array(cols, dtype=object)
        for r, c in zip(r_idx, c_idx):
            out.at[out.index[int(r)], str(cols_arr[int(c)])] = np.nan

        nulled_cells = int(k)

    # ----- Build a few high-signal questions + ground truth answers -----
    # Missing fraction per eligible column
    miss_frac = {c: float(out[c].isna().mean()) for c in cols}

    # Column with highest missing fraction (tie-break by name for determinism)
    max_frac = max(miss_frac.values()) if miss_frac else 0.0
    max_cols = sorted([c for c, v in miss_frac.items() if v == max_frac])
    max_missing_col = max_cols[0] if max_cols else None

    # Rows with >=1 missing among eligible cols
    rows_with_any_missing = int(out[cols].isna().any(axis=1).sum()) if cols else 0

    # Row with most missing among eligible cols (tie-break by smallest row_id)
    if cols:
        miss_count = out[cols].isna().sum(axis=1)
        max_miss = int(miss_count.max()) if len(miss_count) else 0
        candidate_rows = out.loc[miss_count == max_miss, "row_id"]
        row_with_most_missing = int(candidate_rows.min()) if len(candidate_rows) else None
    else:
        max_miss = 0
        row_with_most_missing = None

    # Choose one "focus column" to ask about; deterministic choice = first in cols
    focus_col = cols[0] if cols else None
    focus_missing_frac = float(out[focus_col].isna().mean()) if focus_col else None
    focus_non_missing_count = int(out[focus_col].notna().sum()) if focus_col else None

    phenomenon = {
        "type": "mcar_missingness",
        "params": {"mcar_rate": rate, "columns": cols},
        "effects": {"nulled_cells": nulled_cells},
        "questions": [
            {
                "template_id": "mcar_max_missing_col_v0",
                "answer_format": "categorical",
                "question": "Which column (excluding row_id) has the highest fraction of missing values? Return the single column name.",
                "answer": max_missing_col,
            },
            {
                "template_id": "mcar_rows_with_any_missing_v0",
                "answer_format": "integer",
                "question": f"Considering only columns {cols}, how many rows have at least one missing value? Return the count.",
                "answer": rows_with_any_missing,
            },
            {
                "template_id": "mcar_row_with_most_missing_v0",
                "answer_format": "row_id",
                "question": f"Considering only columns {cols}, which row_id has the largest number of missing values? If tied, return the smallest row_id.",
                "answer": row_with_most_missing,
                "evidence": {
                    "max_missing_count": max_miss
                },
            },
        ],
    }

    return out, phenomenon