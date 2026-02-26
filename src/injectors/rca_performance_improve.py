"""Simpson's paradox / composition shift injector.

Changes group sampling rates across time periods so that within-group
trends oppose the aggregate trend.

Strategy:
  1. Neutralize natural within-group temporal trends (B mean → A mean per group).
  2. Measure the composition gap: after neutralization any aggregate A-B
     difference is purely due to group proportions differing across periods.
  3. Shift all period-B values up by epsilon = composition_gap * 0.5.
     This guarantees every group improves (B > A by epsilon) while the
     aggregate still declines (gap shrinks by epsilon, not eliminated).
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
    params : Must contain ``outcome_col``, ``time_col``, ``group_col``,
             and ``specific_date``.
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

    mask_a = df[time_col] < specific_date_typed
    mask_b = df[time_col] >= specific_date_typed

    groups = df[group_col].unique()
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups to inject Simpson's paradox")
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        raise ValueError("One or both time periods are empty")

    # Step 1 — Neutralize within-group temporal trends (set B mean == A mean per group)
    for g in groups:
        mask_a_g = mask_a & (df[group_col] == g)
        mask_b_g = mask_b & (df[group_col] == g)
        if mask_a_g.sum() == 0 or mask_b_g.sum() == 0:
            continue
        delta = (
            df.loc[mask_a_g, outcome_col].astype(float).mean()
            - df.loc[mask_b_g, outcome_col].astype(float).mean()
        )
        df.loc[mask_b_g, outcome_col] = df.loc[mask_b_g, outcome_col].astype(float) + delta

    # Step 2 — Measure composition gap (purely compositional after neutralization)
    agg_a = df.loc[mask_a, outcome_col].astype(float).mean()
    agg_b = df.loc[mask_b, outcome_col].astype(float).mean()
    composition_gap = agg_a - agg_b

    if composition_gap <= 0:
        raise ValueError(
            "Natural composition does not support Simpson's paradox: "
            "high-mean groups are not sufficiently over-represented in period A"
        )

    # Step 3 — Shift all period-B values up by epsilon
    # epsilon < composition_gap ensures aggregate B stays below aggregate A
    epsilon = composition_gap * 0.5
    df.loc[mask_b, outcome_col] = df.loc[mask_b, outcome_col].astype(float) + epsilon

    # Final verification
    agg_a_final = df.loc[mask_a, outcome_col].astype(float).mean()
    agg_b_final = df.loc[mask_b, outcome_col].astype(float).mean()

    within_group_ok = all(
        df.loc[mask_b & (df[group_col] == g), outcome_col].astype(float).mean()
        > df.loc[mask_a & (df[group_col] == g), outcome_col].astype(float).mean()
        for g in groups
        if (mask_a & (df[group_col] == g)).sum() > 0
        and (mask_b & (df[group_col] == g)).sum() > 0
    )
    paradox_achieved = within_group_ok and (agg_b_final < agg_a_final)

    # Print proof of Simpson's paradox
    # print(f"\n  Within-group trends ({group_col}):")
    # for g in sorted(groups, key=str):
    #     vals_a = df.loc[mask_a & (df[group_col] == g), outcome_col].astype(float)
    #     vals_b = df.loc[mask_b & (df[group_col] == g), outcome_col].astype(float)
    #     if len(vals_a) == 0 or len(vals_b) == 0:
    #         continue
    #     diff = vals_b.mean() - vals_a.mean()
    #     arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
    #     print(f"    '{g}':  A={vals_a.mean():.3f}  →  B={vals_b.mean():.3f}  ({arrow}{abs(diff):.3f})")

    # agg_diff = agg_b_final - agg_a_final
    # agg_arrow = "↑" if agg_diff > 0 else ("↓" if agg_diff < 0 else "→")
    # print(f"\n  Aggregate trend:")
    # print(f"    A={agg_a_final:.3f}  →  B={agg_b_final:.3f}  ({agg_arrow}{abs(agg_diff):.3f})")
    # print(f"\n  Simpson's paradox injected: {'YES' if paradox_achieved else 'NO'}")

    if not paradox_achieved:
        raise ValueError(
            "Simpson's paradox not achieved after adjustment "
            "(within-group or aggregate condition failed)"
        )

    group_means_sorted = df.groupby(group_col)[outcome_col].mean().sort_values()
    mid = len(group_means_sorted) // 2
    effects = {
        "split_point": str(specific_date),
        "group_col": group_col,
        "low_groups": [str(g) for g in group_means_sorted.index[:mid]],
        "high_groups": [str(g) for g in group_means_sorted.index[mid:]],
        "epsilon": float(epsilon),
        "composition_gap": float(composition_gap),
        "paradox_verified": True,
    }

    return df, {
        "type": "rca_performance_improve",
        "params": params,
        "effects": effects,
    }
