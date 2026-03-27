"""Non-monotone peak injector.

Injects an inverted-U (quadratic) relationship between one randomly
chosen numeric feature and the outcome column.  The injection is
additive — the original outcome values are shifted so that rows where
the chosen feature is near the peak value get a boost and rows far
from the peak get a penalty.  Other features' relationships are
preserved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _decimal_places(series: pd.Series) -> int:
    """Return the median number of decimal places in a numeric series."""
    dp_counts = []
    for v in series.dropna():
        s = str(float(v))
        if "." in s:
            dp_counts.append(len(s.split(".")[1]))
        else:
            dp_counts.append(0)
    return int(np.median(dp_counts)) if dp_counts else 2


def _nonmonotone_score(feature_vals: pd.Series, outcome_vals: pd.Series) -> float:
    """Measure how non-monotone the feature→outcome relationship is.

    Bins the feature into 5 quantile bins, computes mean outcome per
    bin, and returns: max(bin_means) - min(first_bin_mean, last_bin_mean).
    A high score means the peak is in the interior — inverted-U shaped.
    """
    try:
        bins = pd.qcut(feature_vals, q=5, duplicates="drop")
    except ValueError:
        return 0.0
    bin_means = outcome_vals.groupby(bins, observed=True).mean().sort_index()
    if len(bin_means) < 3:
        return 0.0
    interior_max = bin_means.iloc[1:-1].max()
    edge_min = min(bin_means.iloc[0], bin_means.iloc[-1])
    return float(interior_max - edge_min)


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Inject an inverted-U relationship between one feature and the outcome.

    Parameters
    ----------
    df : DataFrame to modify (will be copied).
    params : Must contain ``outcome_col`` (str).
             Optional: ``effect_strength`` (float, default 2.0).
    rng : NumPy random Generator for reproducibility.

    Returns
    -------
    (modified_df, phenomenon_dict)
    """
    df = df.copy()
    outcome_col = params["outcome_col"]
    effect_strength = float(params.get("effect_strength", "2.0"))

    # --- Find eligible numeric features (not outcome, enough unique values) ---
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != outcome_col and df[c].nunique() >= 10
    ]

    if len(numeric_cols) < 2:
        raise ValueError(
            f"Need at least 2 eligible numeric features, found {len(numeric_cols)}"
        )

    # --- Pick the feature to inject into ---
    chosen_idx = rng.integers(0, len(numeric_cols))
    chosen_feature = numeric_cols[chosen_idx]

    feature_vals = df[chosen_feature].astype(float)
    outcome_vals = df[outcome_col].astype(float)

    # --- Determine the peak value (interior of the distribution) ---
    pct = rng.uniform(0.35, 0.65)
    peak_raw = float(np.percentile(feature_vals.dropna(), pct * 100))

    # Round peak to the feature's natural decimal precision
    feature_dp = _decimal_places(df[chosen_feature])
    peak = round(peak_raw, feature_dp)

    # --- Compute injection parameters ---
    feat_min = float(feature_vals.min())
    feat_max = float(feature_vals.max())
    half_range = (feat_max - feat_min) / 2.0
    if half_range == 0:
        raise ValueError(f"Feature '{chosen_feature}' has zero range")

    outcome_std = float(outcome_vals.std())
    if outcome_std == 0 or np.isnan(outcome_std):
        outcome_std = 1.0
    amplitude = effect_strength * outcome_std

    # --- Inject the inverted-U additively ---
    # quadratic term: 1 at peak, decreasing toward edges
    normalised_dist = (feature_vals - peak) / half_range
    quadratic = 1.0 - normalised_dist ** 2
    df[outcome_col] = outcome_vals + amplitude * quadratic

    # Round outcome to its original decimal precision
    outcome_dp = _decimal_places(outcome_vals)
    df[outcome_col] = df[outcome_col].round(outcome_dp)

    # --- Validate: inverted-U shape is detectable ---
    new_outcome = df[outcome_col].astype(float)

    try:
        bins = pd.qcut(feature_vals, q=5, duplicates="drop")
    except ValueError:
        raise ValueError(
            f"Cannot bin feature '{chosen_feature}' into quantiles"
        )
    bin_means = new_outcome.groupby(bins, observed=True).mean().sort_index()
    bin_labels = bin_means.index.tolist()

    if len(bin_labels) < 3:
        raise ValueError(
            f"Feature '{chosen_feature}' produced fewer than 3 bins"
        )

    # Peak bin must not be at an extreme
    peak_bin = bin_means.idxmax()
    if peak_bin == bin_labels[0] or peak_bin == bin_labels[-1]:
        raise ValueError(
            f"Peak bin is at an extreme for feature '{chosen_feature}' — "
            f"inverted-U shape not achieved"
        )

    # Amplitude must be meaningful
    injected_score = _nonmonotone_score(feature_vals, new_outcome)
    if injected_score < 0.5 * outcome_std:
        raise ValueError(
            f"Injected non-monotone amplitude too weak: "
            f"{injected_score:.4f} < {0.5 * outcome_std:.4f}"
        )

    # No other feature should have a stronger non-monotone signal
    for other_col in numeric_cols:
        if other_col == chosen_feature:
            continue
        other_vals = df[other_col].astype(float)
        other_score = _nonmonotone_score(other_vals, new_outcome)
        if other_score >= injected_score:
            raise ValueError(
                f"Feature '{other_col}' has a non-monotone score "
                f"({other_score:.4f}) >= injected feature '{chosen_feature}' "
                f"({injected_score:.4f})"
            )

    effects = {
        "peak_feature": str(chosen_feature),
        "peak_value": float(peak),
        "amplitude": float(amplitude),
        "half_range": float(half_range),
    }

    return df, {
        "type": "sf_nonmonotone_peak",
        "params": params,
        "effects": effects,
    }
