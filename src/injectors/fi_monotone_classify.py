"""Monotone-vs-non-monotone classification injector.

Injects a V-shaped (valley) relationship between one randomly chosen
numeric feature and the outcome column.  The injection is additive —
outcome values are shifted DOWN near the midpoint of the chosen feature
and left unchanged at the edges, creating a clear direction reversal.

The chosen feature is selected from features that currently have an
approximately monotone relationship with the outcome.  After injection
it must become the feature with the lowest |rho| (most non-monotone),
ensuring it stands out clearly in the classification task.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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


def _spearman_bin_rho(
    feature_vals: pd.Series,
    outcome_vals: pd.Series,
    n_bins: int = 5,
) -> float:
    """Measure monotonicity via Spearman rho of binned means.

    Bins *feature_vals* into *n_bins* quantiles, computes the mean of
    *outcome_vals* within each bin, and returns the Spearman rank
    correlation between bin index and bin mean.  Returns 1.0 when
    binning fails (conservative: treat as monotone).
    """
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


def _has_direction_reversal(
    feature_vals: pd.Series,
    outcome_vals: pd.Series,
    n_bins: int = 5,
) -> bool:
    """Return True if bin-means change direction at least once."""
    try:
        bins = pd.qcut(feature_vals, q=n_bins, duplicates="drop")
    except ValueError:
        return False
    bin_means = outcome_vals.groupby(bins, observed=True).mean().sort_index()
    if len(bin_means) < 3:
        return False
    diffs = bin_means.diff().dropna().values
    signs = np.sign(diffs)
    for i in range(len(signs) - 1):
        if signs[i] != 0 and signs[i + 1] != 0 and signs[i] != signs[i + 1]:
            return True
    return False


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Inject a V-shaped valley into one feature's relationship with outcome.

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
    effect_strength = float(params.get("effect_strength", "3.0"))

    # --- Find eligible numeric features (not outcome, not ID-like, enough unique) ---
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

    if len(numeric_cols) < 3:
        raise ValueError(
            f"Need at least 3 eligible numeric features, found {len(numeric_cols)}"
        )

    outcome_vals = df[outcome_col].astype(float)

    # --- Pre-injection: compute monotonicity for all features ---
    pre_rhos = {}
    for col in numeric_cols:
        pre_rhos[col] = _spearman_bin_rho(df[col].astype(float), outcome_vals)

    # --- Pick a feature that is currently approximately monotone ---
    # Only inject into features with |rho| >= 0.6 so the reversal is striking
    monotone_candidates = [
        col for col in numeric_cols if abs(pre_rhos[col]) >= 0.6
    ]
    if len(monotone_candidates) < 1:
        raise ValueError(
            "No features with |rho| >= 0.6 to inject into; "
            "dataset has no clearly monotone features"
        )

    # Shuffle candidates and try each until one passes validation
    order = rng.permutation(len(monotone_candidates))
    last_error = None

    for idx in order:
        chosen_feature = monotone_candidates[idx]
        feature_vals = df[chosen_feature].astype(float)

        # --- Determine the midpoint (valley location) ---
        midpoint = float(np.percentile(feature_vals.dropna(), 50))
        feature_dp = _decimal_places(df[chosen_feature])
        midpoint = round(midpoint, feature_dp)

        # --- Compute injection parameters ---
        feat_min = float(feature_vals.min())
        feat_max = float(feature_vals.max())
        half_range = (feat_max - feat_min) / 2.0
        if half_range == 0:
            last_error = f"Feature '{chosen_feature}' has zero range"
            continue

        outcome_std = float(outcome_vals.std())
        if outcome_std == 0 or np.isnan(outcome_std):
            outcome_std = 1.0
        amplitude = effect_strength * outcome_std

        # --- Inject the V-shape (valley) additively ---
        # valley = 1 at midpoint, ~0 at edges  →  subtracting it creates a dip
        norm_dist = (feature_vals - midpoint) / half_range
        valley = 1.0 - np.abs(norm_dist)
        trial_outcome = outcome_vals - amplitude * valley

        # Round outcome to its original decimal precision
        outcome_dp = _decimal_places(outcome_vals)
        trial_outcome = trial_outcome.round(outcome_dp)

        # --- Post-validation ---
        # 1. Injected feature must now be non-monotone (|rho| < 0.5)
        injected_rho = _spearman_bin_rho(feature_vals, trial_outcome)
        if abs(injected_rho) >= 0.5:
            last_error = (
                f"Injected feature '{chosen_feature}' is still monotone: "
                f"|rho| = {abs(injected_rho):.4f} >= 0.5"
            )
            continue

        # 2. Must have at least one direction reversal in bin means
        if not _has_direction_reversal(feature_vals, trial_outcome):
            last_error = (
                f"No direction reversal detected in bin means for '{chosen_feature}'"
            )
            continue

        # 3. Injected feature must have the lowest |rho| of all features
        dominance_ok = True
        for col in numeric_cols:
            if col == chosen_feature:
                continue
            other_rho = _spearman_bin_rho(df[col].astype(float), trial_outcome)
            if abs(other_rho) <= abs(injected_rho):
                last_error = (
                    f"Feature '{col}' has |rho| = {abs(other_rho):.4f} <= "
                    f"injected feature '{chosen_feature}' "
                    f"|rho| = {abs(injected_rho):.4f}; "
                    f"injected feature is not the most non-monotone"
                )
                dominance_ok = False
                break
        if not dominance_ok:
            continue

        # --- All checks passed — commit the injection ---
        df[outcome_col] = trial_outcome

        effects = {
            "injected_feature": str(chosen_feature),
            "midpoint": float(midpoint),
            "amplitude": float(amplitude),
            "half_range": float(half_range),
        }

        return df, {
            "type": "fi_monotone_classify",
            "params": params,
            "effects": effects,
        }

    # All candidates exhausted
    raise ValueError(
        f"No monotone candidate survived validation. "
        f"Last error: {last_error}"
    )
