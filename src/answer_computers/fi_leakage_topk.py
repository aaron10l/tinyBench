from __future__ import annotations
from typing import Any
import warnings
import numpy as np
import pandas as pd

"""
This answer computer is used to compute the answer to the question "Which features are most predictive of the outcome?"
It uses the Pearson correlation coefficient to rank the features by their predictive power.

TODO:
- Is this question too ambiguous? There are multiple ways to solve.. (pearsons correlation, EBMs, etc)
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
