"""Simpson's paradox / composition shift injector.

Changes group sampling rates across time periods so that within-group
trends oppose the aggregate trend.

TODO:
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
    params : Must contain ``outcome_col`` (str), ``time_col`` (str),
             ``group_col`` (str), and ``specific_date`` (str).
    rng : NumPy random Generator for reproducibility.

    Returns
    -------
    (modified_df, phenomenon_dict)
    """
    df = df.copy()
    outcome_col = params["outcome_col"]
    time_col = params["time_col"]
    group_col = params["group_col"]
    specific_date = params["specific_date"]

    # Convert specific_date to match the column dtype
    col_dtype = df[time_col].dtype
    try:
        specific_date_typed = col_dtype.type(specific_date)
    except (ValueError, TypeError):
        specific_date_typed = specific_date

    # Split into two periods: before and after specific_date
    mask_a = df[time_col] < specific_date_typed  # Period A: before
    mask_b = df[time_col] >= specific_date_typed  # Period B: after

    groups = df[group_col].unique()
    if len(groups) < 2:
        return df, {
            "type": "simpsons_paradox_injection",
            "params": params,
            "effects": {"split_point": specific_date, "group_col": group_col},
        }

    # Strategy: Create composition shift by subsampling (removing rows).
    # - Period A: Remove some low-outcome group rows (high-outcome overrepresented)
    # - Period B: Remove some high-outcome group rows (low-outcome overrepresented)
    # This makes aggregate trend oppose within-group trends (Simpson's paradox).

    group_means = df.groupby(group_col)[outcome_col].mean()
    sorted_groups = group_means.sort_values()
    low_group = sorted_groups.index[0]
    high_group = sorted_groups.index[-1]
    middle_groups = [g for g in sorted_groups.index if g not in (low_group, high_group)]

    # Improve outcomes within each group from A to B (within-group improvement)
    outcome_std = df[outcome_col].astype(float).std()
    if outcome_std == 0:
        outcome_std = 1.0
    boost = outcome_std * 0.3

    # Boost period B outcomes within each group
    df.loc[mask_b, outcome_col] = df.loc[mask_b, outcome_col].astype(float) + boost

    # Composition shift via subsampling:
    # - Remove ~40% of low-outcome group from Period A
    # - Remove ~40% of high-outcome group from Period B
    drop_fraction = 0.4

    # Rows to potentially drop
    drop_candidates_a = df[mask_a & (df[group_col] == low_group)].index
    drop_candidates_b = df[mask_b & (df[group_col] == high_group)].index

    n_drop_a = int(len(drop_candidates_a) * drop_fraction)
    n_drop_b = int(len(drop_candidates_b) * drop_fraction)

    rows_to_drop = []
    if n_drop_a > 0:
        drop_idx_a = rng.choice(drop_candidates_a, size=n_drop_a, replace=False)
        rows_to_drop.extend(drop_idx_a)
    if n_drop_b > 0:
        drop_idx_b = rng.choice(drop_candidates_b, size=n_drop_b, replace=False)
        rows_to_drop.extend(drop_idx_b)

    df = df.drop(index=rows_to_drop).reset_index(drop=True)

    # Recompute masks after dropping rows
    try:
        specific_date_typed = col_dtype.type(specific_date)
    except (ValueError, TypeError):
        specific_date_typed = specific_date
    mask_a_after = df[time_col] < specific_date_typed
    mask_b_after = df[time_col] >= specific_date_typed

    effects = {
        "split_point": str(specific_date),
        "group_col": group_col,
        "low_group": str(low_group),
        "high_group": str(high_group),
        "boost": float(boost),
        "n_rows_dropped": len(rows_to_drop),
        "drop_fraction": drop_fraction,
    }

    return df, {
        "type": "simpsons_paradox_injection",
        "params": params,
        "effects": effects,
    }
