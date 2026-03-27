from __future__ import annotations
from typing import Any

import pandas as pd


def compute_answer(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Return the feature with the inverted-U peak and its peak value.

    Answer format: "feature_name, peak_value"
    """
    inj = effects["sf_nonmonotone_peak"]
    feature = inj["peak_feature"]
    peak = inj["peak_value"]

    # Verify the inverted-U is detectable in the data
    outcome_col = slot_assignments["OUTCOME_COL"]
    feature_vals = pd.to_numeric(df[feature], errors="coerce")
    outcome_vals = pd.to_numeric(df[outcome_col], errors="coerce")

    bins = pd.qcut(feature_vals, q=5, duplicates="drop")
    bin_means = outcome_vals.groupby(bins, observed=True).mean().sort_index()
    bin_labels = bin_means.index.tolist()
    peak_bin = bin_means.idxmax()

    if peak_bin == bin_labels[0] or peak_bin == bin_labels[-1]:
        raise ValueError(
            f"Inverted-U not confirmed for feature '{feature}' — "
            f"peak bin is at an extreme"
        )

    return f"{feature}, {peak}"
