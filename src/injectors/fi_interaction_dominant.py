"""Interaction-dominant feature pair injector.

Injects a pure XOR-like interaction between two randomly chosen numeric
features.  The outcome is additively shifted so that rows where exactly
one of the two features is above its median get a boost, while rows
where both are above or both are below get a penalty.  Neither feature
alone has a net main effect on the outcome — only their interaction does.
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


def _xor_signal(col_a: pd.Series, col_b: pd.Series) -> pd.Series:
    """Compute a centred XOR interaction signal from two numeric columns."""
    med_a = col_a.astype(float).median()
    med_b = col_b.astype(float).median()
    bin_a = (col_a.astype(float) > med_a).astype(float)
    bin_b = (col_b.astype(float) > med_b).astype(float)
    interaction = (bin_a + bin_b) % 2  # XOR: 1 when exactly one is above median
    return interaction - interaction.mean()


def _cohens_d(group1: pd.Series, group0: pd.Series) -> float:
    """Compute Cohen's d between two groups."""
    n1, n0 = len(group1), len(group0)
    if n1 < 2 or n0 < 2:
        return 0.0
    m1, m0 = group1.mean(), group0.mean()
    pooled_std = np.sqrt(
        ((n1 - 1) * group1.var() + (n0 - 1) * group0.var()) / (n1 + n0 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float(abs(m1 - m0) / pooled_std)


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Inject a pure XOR interaction between two features and the outcome.

    Parameters
    ----------
    df : DataFrame to modify (will be copied).
    params : Must contain ``outcome_col`` (str).
             Optional: ``effect_strength`` (float, default 1.5).
    rng : NumPy random Generator for reproducibility.

    Returns
    -------
    (modified_df, phenomenon_dict)
    """
    df = df.copy()
    outcome_col = params["outcome_col"]
    effect_strength = float(params.get("effect_strength", "1.5"))

    # --- Find eligible numeric features ---
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != outcome_col and df[c].nunique() >= 5
    ]

    if len(numeric_cols) < 3:
        raise ValueError(
            f"Need at least 3 eligible numeric features, found {len(numeric_cols)}"
        )

    # --- Pick two features for the interaction ---
    chosen = rng.choice(len(numeric_cols), size=2, replace=False)
    feat_a = numeric_cols[chosen[0]]
    feat_b = numeric_cols[chosen[1]]

    outcome_vals = df[outcome_col].astype(float)
    outcome_std = float(outcome_vals.std())
    if outcome_std == 0 or np.isnan(outcome_std):
        outcome_std = 1.0

    # --- Compute the XOR interaction and inject additively ---
    interaction = _xor_signal(df[feat_a], df[feat_b])
    amplitude = effect_strength * outcome_std
    df[outcome_col] = outcome_vals + amplitude * interaction

    # Round to original decimal precision
    outcome_dp = _decimal_places(outcome_vals)
    df[outcome_col] = df[outcome_col].round(outcome_dp)

    # --- Validate the injection ---
    new_outcome = df[outcome_col].astype(float)

    # 1. XOR pattern is clear: mean(XOR=1) > mean(XOR=0)
    med_a = df[feat_a].astype(float).median()
    med_b = df[feat_b].astype(float).median()
    bin_a = (df[feat_a].astype(float) > med_a).astype(float)
    bin_b = (df[feat_b].astype(float) > med_b).astype(float)
    xor_binary = ((bin_a + bin_b) % 2).astype(int)
    group1 = new_outcome[xor_binary == 1]
    group0 = new_outcome[xor_binary == 0]
    interaction_d = _cohens_d(group1, group0)

    if interaction_d < 0.5:
        raise ValueError(
            f"XOR interaction too weak: Cohen's d = {interaction_d:.4f} < 0.5"
        )

    # 2. Neither feature alone has a strong main effect
    for feat in [feat_a, feat_b]:
        corr = abs(float(df[feat].astype(float).corr(new_outcome)))
        if corr > 0.7:
            raise ValueError(
                f"Feature '{feat}' has too strong a main effect: "
                f"|r| = {corr:.4f} > 0.7"
            )

    # 3. Spot-check that no other pair has a stronger interaction
    # Sample up to 10 random pairs from the remaining features
    other_cols = [c for c in numeric_cols if c not in (feat_a, feat_b)]
    n_checks = min(10, len(other_cols) * (len(other_cols) - 1) // 2)
    if len(other_cols) >= 2 and n_checks > 0:
        checked = 0
        all_pairs = [(other_cols[i], other_cols[j])
                     for i in range(len(other_cols))
                     for j in range(i + 1, len(other_cols))]
        if len(all_pairs) > n_checks:
            pair_indices = rng.choice(len(all_pairs), size=n_checks, replace=False)
            check_pairs = [all_pairs[idx] for idx in pair_indices]
        else:
            check_pairs = all_pairs

        for ca, cb in check_pairs:
            other_xor = _xor_signal(df[ca], df[cb]) + 0.5
            other_binary = (other_xor > 0).astype(int)
            og1 = new_outcome[other_binary == 1]
            og0 = new_outcome[other_binary == 0]
            other_d = _cohens_d(og1, og0)
            if other_d >= interaction_d:
                raise ValueError(
                    f"Pair ({ca}, {cb}) has interaction d={other_d:.4f} "
                    f">= injected pair ({feat_a}, {feat_b}) d={interaction_d:.4f}"
                )

    effects = {
        "feat_a": str(feat_a),
        "feat_b": str(feat_b),
        "amplitude": float(amplitude),
    }

    return df, {
        "type": "fi_interaction_dominant",
        "params": params,
        "effects": effects,
    }
