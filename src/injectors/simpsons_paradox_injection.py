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

    # Step 1 — Compute group structure on original data
    group_means = df.groupby(group_col)[outcome_col].mean().sort_values()
    groups_sorted = group_means.index.tolist()
    mid = len(groups_sorted) // 2
    low_groups = set(groups_sorted[:mid])
    high_groups = set(groups_sorted[mid:])

    # Step 2 — Neutralize natural temporal trends so that within each group,
    # period B mean equals period A mean.  This ensures the composition shift
    # (step 3) is the sole driver of the aggregate difference.
    for g in df[group_col].unique():
        vals_a = df.loc[mask_a & (df[group_col] == g), outcome_col].astype(float)
        vals_b = df.loc[mask_b & (df[group_col] == g), outcome_col].astype(float)
        if len(vals_a) == 0 or len(vals_b) == 0:
            continue
        adjustment = vals_a.mean() - vals_b.mean()  # shift B to match A
        mask_gb = mask_b & (df[group_col] == g)
        df.loc[mask_gb, outcome_col] = (
            df.loc[mask_gb, outcome_col].astype(float) + adjustment
        )

    # Step 3 — Composition shift + per-group calibration on the final data
    paradox_achieved = False
    final_drop_fraction = None
    n_rows_dropped = 0
    epsilon = 0.0

    for drop_fraction in [0.5, 0.6, 0.7, 0.8]:
        df_candidate = df.copy()
        rows_to_drop = []

        # Period A: drop from low-mean groups (makes A's aggregate higher)
        for g in low_groups:
            candidates = df_candidate[mask_a & (df_candidate[group_col] == g)].index
            n_drop = int(len(candidates) * drop_fraction)
            if n_drop > 0:
                drop_idx = rng.choice(candidates, size=n_drop, replace=False)
                rows_to_drop.extend(drop_idx)

        # Period B: drop from high-mean groups (makes B's aggregate lower)
        for g in high_groups:
            candidates = df_candidate[mask_b & (df_candidate[group_col] == g)].index
            n_drop = int(len(candidates) * drop_fraction)
            if n_drop > 0:
                drop_idx = rng.choice(candidates, size=n_drop, replace=False)
                rows_to_drop.extend(drop_idx)

        df_candidate = df_candidate.drop(index=rows_to_drop).reset_index(drop=True)

        # Measure the composition gap on the neutralized, shifted data
        mask_a_new = df_candidate[time_col] < specific_date_typed
        mask_b_new = df_candidate[time_col] >= specific_date_typed
        agg_a = df_candidate.loc[mask_a_new, outcome_col].astype(float).mean()
        agg_b = df_candidate.loc[mask_b_new, outcome_col].astype(float).mean()
        gap = agg_a - agg_b  # positive because trends are neutralized

        if gap <= 0:
            continue  # shouldn't happen after neutralization, but be safe

        # Per-group adjustment: set each group's B mean to A mean + epsilon.
        # Use 30% of gap so the paradox is preserved (aggregate B rises by
        # ~epsilon but stays below aggregate A).
        epsilon = gap * 0.3
        for g in df_candidate[group_col].unique():
            vals_a = df_candidate.loc[
                mask_a_new & (df_candidate[group_col] == g), outcome_col
            ].astype(float)
            vals_b = df_candidate.loc[
                mask_b_new & (df_candidate[group_col] == g), outcome_col
            ].astype(float)
            if len(vals_a) == 0 or len(vals_b) == 0:
                continue
            current_trend = vals_b.mean() - vals_a.mean()
            adjustment = epsilon - current_trend
            mask_gb = mask_b_new & (df_candidate[group_col] == g)
            df_candidate.loc[mask_gb, outcome_col] = (
                df_candidate.loc[mask_gb, outcome_col].astype(float) + adjustment
            )

        # Final verification
        agg_a_final = df_candidate.loc[mask_a_new, outcome_col].astype(float).mean()
        agg_b_final = df_candidate.loc[mask_b_new, outcome_col].astype(float).mean()

        if agg_b_final < agg_a_final:
            df = df_candidate
            paradox_achieved = True
            final_drop_fraction = drop_fraction
            n_rows_dropped = len(rows_to_drop)
            break

    if not paradox_achieved:
        raise ValueError(
            "Cannot achieve Simpson's paradox on this dataset "
            "(groups may be too homogeneous)"
        )

    # Step 3 — Record effects
    effects = {
        "split_point": str(specific_date),
        "group_col": group_col,
        "low_groups": [str(g) for g in groups_sorted[:mid]],
        "high_groups": [str(g) for g in groups_sorted[mid:]],
        "epsilon": float(epsilon),
        "n_rows_dropped": n_rows_dropped,
        "drop_fraction": final_drop_fraction,
        "paradox_verified": True,
    }

    return df, {
        "type": "simpsons_paradox_injection",
        "params": params,
        "effects": effects,
    }
