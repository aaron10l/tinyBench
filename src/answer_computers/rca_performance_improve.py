from __future__ import annotations
from typing import Any
import pandas as pd


def compute_answer(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    inj = effects["simpsons_paradox_injection"]

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

    # Compute within-group mean change from period A to period B
    groups = df[group_col].unique()
    improved_count = 0
    worsened_count = 0
    unchanged_count = 0

    for g in groups:
        vals_a = df.loc[mask_a & (df[group_col] == g), outcome_col].astype(float)
        vals_b = df.loc[mask_b & (df[group_col] == g), outcome_col].astype(float)
        if len(vals_a) == 0 or len(vals_b) == 0:
            continue
        diff = vals_b.mean() - vals_a.mean()
        if diff > 0:
            improved_count += 1
        elif diff < 0:
            worsened_count += 1
        else:
            unchanged_count += 1

    if improved_count == 0 and worsened_count == 0:
        raise ValueError("No groups had rows in both periods — cannot determine trend.")

    # Also compute aggregate trend for a sanity reference
    agg_a = df.loc[mask_a, outcome_col].astype(float).mean()
    agg_b = df.loc[mask_b, outcome_col].astype(float).mean()
    agg_direction = "IMPROVED" if agg_b > agg_a else "WORSENED" if agg_b < agg_a else "UNCHANGED"

    # The within-group answer: majority of groups should agree
    if improved_count > worsened_count:
        within_group_answer = "IMPROVED"
    elif worsened_count > improved_count:
        within_group_answer = "WORSENED"
    else:
        within_group_answer = "UNCHANGED"

    # For a proper Simpson's paradox, within-group should oppose aggregate
    if within_group_answer == agg_direction:
        raise ValueError(
            f"No Simpson's paradox detected: within-group trend ({within_group_answer}) "
            f"matches aggregate trend ({agg_direction}). "
            f"Groups improved={improved_count}, worsened={worsened_count}, unchanged={unchanged_count}. "
            f"Aggregate mean: A={agg_a:.4f}, B={agg_b:.4f}"
        )

    return within_group_answer
