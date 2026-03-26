from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _spearman_bin_rho(
    feature_vals: pd.Series,
    outcome_vals: pd.Series,
    n_bins: int = 5,
) -> float:
    """Measure monotonicity via Spearman rho of binned means."""
    try:
        bins = pd.qcut(feature_vals, q=n_bins, duplicates="drop")
    except ValueError:
        return 1.0
    bin_means = outcome_vals.groupby(bins, observed=True).mean().sort_index()
    if len(bin_means) < 3:
        return 1.0
    rho, _ = spearmanr(range(len(bin_means)), bin_means.values)
    if np.isnan(rho):
        return 1.0
    return float(rho)


def compute_answer(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Return the feature with the most non-monotone relationship.

    Answer format: "feature_name"
    """
    inj = effects["fi_monotone_classify"]
    injected_feature = inj["injected_feature"]

    outcome_col = slot_assignments["OUTCOME_COL"]
    outcome_vals = pd.to_numeric(df[outcome_col], errors="coerce")

    # Identify numeric features (same filtering as the injector)
    numeric_cols = []
    for c in df.select_dtypes(include=[np.number]).columns:
        if c == outcome_col:
            continue
        if df[c].nunique() < 10:
            continue
        # Skip ID-like columns (all unique integers in monotone sequence)
        if df[c].nunique() == len(df) and df[c].is_monotonic_increasing:
            continue
        numeric_cols.append(c)

    # Find the feature with the lowest |rho| (most non-monotone)
    min_rho = float("inf")
    most_nonmonotone = None
    for col in numeric_cols:
        feature_vals = pd.to_numeric(df[col], errors="coerce")
        rho = _spearman_bin_rho(feature_vals, outcome_vals)
        if abs(rho) < min_rho:
            min_rho = abs(rho)
            most_nonmonotone = col

    # Verify the injected feature is the most non-monotone
    if most_nonmonotone != injected_feature:
        raise ValueError(
            f"Injected feature '{injected_feature}' is not the most "
            f"non-monotone; '{most_nonmonotone}' has lower |rho|"
        )

    return injected_feature
