"""Change-point injector.

After row t, makes a prediction column depend on an interaction
(XOR-like, piecewise, multiplicative) of two features while keeping
the marginal mean roughly constant.

post change rows outcome col is overwrriten with:
  new_value = pre_mean + (xor_signal - mean(xor_signal)) * 1.5 * pre_std + noise

TODO:
- Move selection of features for XOR signal to be done upstream of this injector.
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
    params : Must contain ``pred_col`` (str), ``order_col`` (str),
             and ``specific_row_id`` (the value in order_col where change begins).
    rng : NumPy random Generator for reproducibility.

    Returns
    -------
    (modified_df, phenomenon_dict)
    """
    df = df.copy()
    pred_col = params["pred_col"]
    order_col = params["order_col"]
    specific_row_id = params["specific_row_id"]

    # Convert specific_row_id to match the column dtype
    col_dtype = df[order_col].dtype
    try:
        specific_row_id_typed = col_dtype.type(specific_row_id)
    except (ValueError, TypeError):
        specific_row_id_typed = specific_row_id

    # Change point: rows where order_col >= specific_row_id
    post_change_mask = df[order_col] >= specific_row_id_typed
    pre_change_mask = ~post_change_mask

    # Pick two numeric feature columns for the interaction
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in (pred_col, order_col)
    ]

    if len(numeric_cols) < 2:
        return df, {
            "type": "rca_retrain_point",
            "params": params,
            "effects": {"change_row_id": None},
        }

    chosen = rng.choice(len(numeric_cols), size=2, replace=False)
    feat_a = numeric_cols[chosen[0]]
    feat_b = numeric_cols[chosen[1]]

    # Compute the interaction: XOR-like on median-split binarization
    med_a = df[feat_a].astype(float).median()
    med_b = df[feat_b].astype(float).median()
    bin_a = (df[feat_a].astype(float) > med_a).astype(float)
    bin_b = (df[feat_b].astype(float) > med_b).astype(float)
    interaction = (bin_a + bin_b) % 2  # XOR

    # Compute stats from pre-change rows
    pre_mean = df.loc[pre_change_mask, pred_col].astype(float).mean()
    pre_std = df.loc[pre_change_mask, pred_col].astype(float).std()
    if pre_std == 0 or np.isnan(pre_std):
        pre_std = 1.0
    if np.isnan(pre_mean):
        pre_mean = df[pred_col].astype(float).mean()

    # Construct post-change values: base = pre_mean + interaction effect
    # Shift so that the mean remains ~ pre_mean
    post_n = post_change_mask.sum()
    interaction_post = interaction.loc[post_change_mask].values
    effect_size = pre_std * 1.5
    raw = pre_mean + (interaction_post - interaction_post.mean()) * effect_size
    # Add small noise
    raw = raw + rng.normal(0, pre_std * 0.2, post_n)

    df.loc[post_change_mask, pred_col] = raw

    effects = {
        "change_row_id": str(specific_row_id),
        "n_rows_changed": int(post_n),
        "feat_a": feat_a,
        "feat_b": feat_b,
        "effect_size": float(effect_size),
    }

    return df, {
        "type": "rca_retrain_point",
        "params": params,
        "effects": effects,
    }
