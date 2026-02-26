from __future__ import annotations
from typing import Any
import pandas as pd


def compute_answer(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    effects["rca_performance_improve"]  # assert key exists

    outcome_col = slot_assignments["OUTCOME_COL"]
    time_col = slot_assignments["TIME_COL"]
    group_col = slot_assignments["GROUP_COL"]
    specific_date = slot_assignments["SPECIFIC_DATE"]

    # Match dtype for the split point
    col_dtype = df[time_col].dtype
    try:
        split = col_dtype.type(specific_date)
    except (ValueError, TypeError):
        split = specific_date

    mask_a = df[time_col] < split
    mask_b = df[time_col] >= split

    if mask_a.sum() == 0 or mask_b.sum() == 0:
        raise ValueError(
            f"One period is empty: period_A={mask_a.sum()} rows, "
            f"period_B={mask_b.sum()} rows (split='{specific_date}')"
        )

    # Compute within-group trends
    groups = df[group_col].unique()
    improved_count = 0
    worsened_count = 0
    comparable_groups = 0

    for g in sorted(groups, key=str):
        vals_a = df.loc[mask_a & (df[group_col] == g), outcome_col].astype(float)
        vals_b = df.loc[mask_b & (df[group_col] == g), outcome_col].astype(float)
        if len(vals_a) == 0 or len(vals_b) == 0:
            continue
        comparable_groups += 1
        diff = vals_b.mean() - vals_a.mean()
        if diff > 0:
            improved_count += 1
        elif diff < 0:
            worsened_count += 1

    if comparable_groups == 0:
        raise ValueError("No groups had rows in both periods — cannot determine trend.")

    # Aggregate trend
    agg_a = df.loc[mask_a, outcome_col].astype(float).mean()
    agg_b = df.loc[mask_b, outcome_col].astype(float).mean()
    agg_direction = "IMPROVED" if agg_b > agg_a else "WORSENED" if agg_b < agg_a else "UNCHANGED"
    within_group_answer = "IMPROVED" if improved_count == comparable_groups else (
        "WORSENED" if worsened_count == comparable_groups else "MIXED"
    )

    # Verify Simpson's paradox for this injector:
    # all groups improve while aggregate worsens.
    if within_group_answer != "IMPROVED" or agg_direction != "WORSENED":
        raise ValueError(
            f"No Simpson's paradox detected: within-group trend ({within_group_answer}) "
            f"and aggregate trend ({agg_direction}) do not match the injected pattern. "
            f"Groups improved={improved_count}, worsened={worsened_count}, "
            f"comparable={comparable_groups}. "
            f"Aggregate mean: A={agg_a:.4f}, B={agg_b:.4f}"
        )

    return within_group_answer
