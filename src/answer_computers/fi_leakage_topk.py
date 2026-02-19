from __future__ import annotations
from typing import Any
import warnings
import numpy as np
import pandas as pd

"""
Ranks features by absolute Pearson correlation with the outcome column.
Matches the question template which explicitly asks for Pearson correlation.
"""

def compute_answer(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    outcome_col = slot_assignments["OUTCOME_COL"]
    k = int(slot_assignments.get("K", 5))
    target = pd.to_numeric(df[outcome_col], errors="coerce")

    feature_cols = [c for c in df.columns if c != outcome_col]
    correlations = {}
    for col in feature_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().all():
            continue
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            corr = target.corr(numeric)
        if pd.notna(corr):
            correlations[col] = abs(corr)

    ranked = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:k]]
