"""Heteroskedastic noise injector.

Makes group means identical (centered on global mean) for a metric column while giving one group
much higher variance / heavier tails than the other.

TODO:
- Should we center each group on its own initial mean? or just the global mean?
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """
    Parameters
    ----------
    df : DataFrame to modify (will be copied).
    params : Must contain ``group_col`` (str), ``metric_col`` (str),
             ``group_a`` (str), and ``group_b`` (str).
    rng : NumPy random Generator for reproducibility.

    Returns
    -------
    (modified_df, phenomenon_dict)
    """
    df = df.copy()
    group_col = params["group_col"]
    metric_col = params["metric_col"]
    group_a = params["group_a"]
    group_b = params["group_b"]

    # Convert group values to match the column dtype
    col_dtype = df[group_col].dtype
    try:
        group_a = col_dtype.type(group_a)
        group_b = col_dtype.type(group_b)
    except (ValueError, TypeError):
        pass  # Keep as string if conversion fails

    # Randomly pick one of the two specified groups to be risky
    risky_group = group_a if rng.random() < 0.5 else group_b
    safe_group = group_b if risky_group == group_a else group_a
    variance_multiplier = 5.0

    global_mean = df[metric_col].astype(float).mean()
    global_std = df[metric_col].astype(float).std()
    if global_std == 0:
        global_std = 1.0

    # Only modify the two specified groups
    for g, is_risky in [(risky_group, True), (safe_group, False)]:
        mask = df[group_col] == g
        n = mask.sum()
        if n == 0:
            continue
        if is_risky:
            # High variance, centered on global mean
            df.loc[mask, metric_col] = global_mean + rng.normal(
                0, global_std * variance_multiplier, n
            )
        else:
            # Low variance, centered on global mean
            df.loc[mask, metric_col] = global_mean + rng.normal(
                0, global_std * 0.5, n
            )


    # Verify that the group means stayed close enough after random generation
    vals_risky = df.loc[df[group_col] == risky_group, metric_col].astype(float)
    vals_safe = df.loc[df[group_col] == safe_group, metric_col].astype(float)
    pooled_std = pd.concat([vals_risky, vals_safe]).std()
    if pooled_std > 0:
        mean_diff = abs(vals_risky.mean() - vals_safe.mean()) / pooled_std
        if mean_diff > 0.5:
            raise ValueError(
                f"Injection skipped: group means drifted too far apart "
                f"(normalised diff={mean_diff:.4f}, threshold=0.5)"
            )

    effects = {
        "risky_group": str(risky_group),
        "safe_group": str(safe_group),
        "variance_multiplier": variance_multiplier,
        "global_mean": float(global_mean),
    }

    return df, {
        "type": "anomaly_riskier_group",
        "params": params,
        "effects": effects,
    }
