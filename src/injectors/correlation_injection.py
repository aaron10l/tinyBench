# src/injectors/correlation_injection.py
from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


def inject_correlation_injection(
    df: pd.DataFrame,
    params: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Makes y_col "too correlated" with x_col by blending standardized values.
    params:
      - x_col: optional str (random numeric if omitted)
      - y_col: optional str (random numeric if omitted)
      - alpha: float in (0,1] (higher => more correlation, default 0.95)
      - noise_std: float >= 0 (default 0.05) in standardized units
    """
    if "row_id" not in df.columns:
        raise ValueError("Expected 'row_id' column in standardized dataset.")

    out = df.copy()
    numeric_cols = [c for c in out.columns if c != "row_id" and pd.api.types.is_numeric_dtype(out[c])]
    if len(numeric_cols) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation injection.")

    x = params.get("x_col")
    y = params.get("y_col")
    if x is None or y is None:
        x, y = rng.choice(numeric_cols, size=2, replace=False).tolist()

    if x not in out.columns or y not in out.columns:
        raise ValueError(f"Bad x_col/y_col: {x}, {y}")

    alpha = float(params.get("alpha", 0.95))
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")

    noise_std = float(params.get("noise_std", 0.05))
    if noise_std < 0.0:
        raise ValueError("noise_std must be >= 0.")

    mask = out[x].notna() & out[y].notna()
    n_mod = int(mask.sum())
    if n_mod == 0:
        phenomenon = {
            "type": "correlation_injection",
            "params": {"x_col": x, "y_col": y, "alpha": alpha, "noise_std": noise_std},
            "effects": {"rows_modified": 0, "achieved_corr": None},
            "questions": [
                {
                    "template_id": "corr_value_pair_v0",
                    "answer_format": "float_3dp",
                    "question": f"What is the Pearson correlation between '{x}' and '{y}'? Return a number rounded to 3 decimals.",
                    "answer": None,
                },
                {
                    "template_id": "corr_most_corr_with_x_v0",
                    "answer_format": "categorical",
                    "question": f"Among numeric columns excluding '{x}', which column has the highest absolute Pearson correlation with '{x}'? Return the single column name.",
                    "answer": None,
                },
                {
                    "template_id": "corr_most_corr_pair_v0",
                    "answer_format": "categorical_list",
                    "question": "Among all numeric columns (excluding row_id), which pair has the highest absolute Pearson correlation? Return the two column names sorted alphabetically.",
                    "answer": None,
                },
            ],
        }
        return out, phenomenon

    x_vals = out.loc[mask, x].to_numpy(dtype=float)
    y_vals = out.loc[mask, y].to_numpy(dtype=float)

    x_mu, x_sd = float(x_vals.mean()), float(x_vals.std()) or 1.0
    y_mu, y_sd = float(y_vals.mean()), float(y_vals.std()) or 1.0

    xs = (x_vals - x_mu) / x_sd
    ys = (y_vals - y_mu) / y_sd

    eps = rng.normal(0.0, noise_std, size=len(xs))
    y_new_s = alpha * xs + (1.0 - alpha) * ys + eps
    y_new = y_new_s * y_sd + y_mu

    out.loc[mask, y] = y_new

    achieved = float(np.corrcoef(xs, (y_new - y_mu) / y_sd)[0, 1])

    # ---- Build 3 questions + ground-truth answers from the modified table ----
    # Q1: correlation value between injected pair
    corr_xy = float(out[[x, y]].corr(method="pearson").iloc[0, 1])
    corr_xy_rounded = round(corr_xy, 3)

    # Q2: which column is most correlated with x (excluding x)
    # Use absolute correlation, ignore columns with all-NaN or constant variance (corr becomes NaN).
    corrs_with_x: Dict[str, float] = {}
    for c in numeric_cols:
        if c == x:
            continue
        val = float(out[[x, c]].corr(method="pearson").iloc[0, 1])
        if not np.isnan(val):
            corrs_with_x[c] = abs(val)
    if corrs_with_x:
        max_abs = max(corrs_with_x.values())
        best_cols = sorted([c for c, v in corrs_with_x.items() if v == max_abs])
        most_corr_with_x = best_cols[0]
    else:
        most_corr_with_x = None

    # Q3: which numeric pair has the highest absolute correlation (excluding self-pairs)
    best_pair = None
    best_abs = -1.0
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            a, b = numeric_cols[i], numeric_cols[j]
            val = float(out[[a, b]].corr(method="pearson").iloc[0, 1])
            if np.isnan(val):
                continue
            aval = abs(val)
            if aval > best_abs:
                best_abs = aval
                best_pair = tuple(sorted([a, b]))
            elif aval == best_abs and best_pair is not None:
                candidate = tuple(sorted([a, b]))
                if candidate < best_pair:
                    best_pair = candidate

    phenomenon = {
        "type": "correlation_injection",
        "params": {"x_col": x, "y_col": y, "alpha": alpha, "noise_std": noise_std},
        "effects": {"rows_modified": n_mod, "achieved_corr": achieved},
        "questions": [
            {
                "template_id": "corr_value_pair_v0",
                "answer_format": "float_3dp",
                "question": f"What is the Pearson correlation between '{x}' and '{y}'? Return a number rounded to 3 decimals.",
                "answer": corr_xy_rounded,
            },
            {
                "template_id": "corr_most_corr_with_x_v0",
                "answer_format": "categorical",
                "question": f"Among numeric columns excluding '{x}', which column has the highest absolute Pearson correlation with '{x}'? Return the single column name.",
                "answer": most_corr_with_x,
            },
            {
                "template_id": "corr_most_corr_pair_v0",
                "answer_format": "categorical_list",
                "question": "Among all numeric columns (excluding row_id), which pair has the highest absolute Pearson correlation? Return the two column names sorted alphabetically.",
                "answer": list(best_pair) if best_pair is not None else None,
            },
        ],
    }
    return out, phenomenon