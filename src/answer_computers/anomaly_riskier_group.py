from __future__ import annotations
from typing import Any
import pandas as pd
from scipy import stats as scipystats

"""
This answer computer is used to compute the answer to the question "Which group is riskier?"
First, it checks that means ar similar, and variances are very different.
Then, it returns the group with the higher variance, assuming it matches the expected risky group.
"""

def compute_answer(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    inj = effects["heteroskedastic_injection"]

    group_col = slot_assignments["GROUP_COL"]
    metric_col = slot_assignments["RISK_METRIC_COL"]
    group_a = slot_assignments["GROUP_A"]
    group_b = slot_assignments["GROUP_B"]

    # Subset to the two groups of interest
    col_dtype = df[group_col].dtype
    try:
        group_a_typed = col_dtype.type(group_a)
        group_b_typed = col_dtype.type(group_b)
    except (ValueError, TypeError):
        group_a_typed = group_a
        group_b_typed = group_b

    vals_a = df.loc[df[group_col] == group_a_typed, metric_col].astype(float)
    vals_b = df.loc[df[group_col] == group_b_typed, metric_col].astype(float)

    mean_a, mean_b = vals_a.mean(), vals_b.mean()
    var_a, var_b = vals_a.var(), vals_b.var()

    # 1. Check that the means are similar (relative to pooled std)
    pooled_std = pd.concat([vals_a, vals_b]).std()
    if pooled_std == 0:
        raise ValueError(
            f"Pooled std is zero — metric '{metric_col}' is constant across both groups."
        )
    mean_diff = abs(mean_a - mean_b) / pooled_std
    if mean_diff > 0.5:
        raise ValueError(
            f"Group means are too different: "
            f"{group_a}={mean_a:.4f}, {group_b}={mean_b:.4f} "
            f"(normalised diff={mean_diff:.4f}, threshold=0.5)"
        )

    # 2. Check that the variances are statistically different (Levene's test)
    _, p_value = scipystats.levene(vals_a, vals_b)
    if p_value > 0.05:
        raise ValueError(
            f"Variances are not statistically different: "
            f"{group_a}_var={var_a:.4f}, {group_b}_var={var_b:.4f} "
            f"(Levene p={p_value:.4f}, threshold=0.05)"
        )

    # 3. The group with higher variance is the risky group
    computed_risky = group_a if var_a > var_b else group_b
    expected_risky = inj["risky_group"]
    if str(computed_risky) != str(expected_risky):
        raise ValueError(
            f"Computed risky group '{computed_risky}' (var={max(var_a, var_b):.4f}) "
            f"does not match effects risky group '{expected_risky}'"
        )

    return expected_risky
