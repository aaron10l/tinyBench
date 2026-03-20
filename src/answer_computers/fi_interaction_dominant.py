from __future__ import annotations
from typing import Any

import pandas as pd


def compute_answer(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Return the feature pair with the strongest interaction effect.

    Answer format: "column_a, column_b" (sorted alphabetically).
    """
    inj = effects["fi_interaction_dominant"]
    feat_a = inj["feat_a"]
    feat_b = inj["feat_b"]

    # Verify the XOR interaction is detectable
    outcome_col = slot_assignments["OUTCOME_COL"]
    outcome = pd.to_numeric(df[outcome_col], errors="coerce")
    a_vals = pd.to_numeric(df[feat_a], errors="coerce")
    b_vals = pd.to_numeric(df[feat_b], errors="coerce")

    bin_a = (a_vals > a_vals.median()).astype(int)
    bin_b = (b_vals > b_vals.median()).astype(int)
    xor = (bin_a + bin_b) % 2

    mean_xor1 = outcome[xor == 1].mean()
    mean_xor0 = outcome[xor == 0].mean()
    if mean_xor1 <= mean_xor0:
        raise ValueError(
            f"XOR interaction not confirmed: mean(XOR=1)={mean_xor1:.4f} "
            f"<= mean(XOR=0)={mean_xor0:.4f}"
        )

    # Return sorted pair for consistent ordering
    pair = sorted([feat_a, feat_b])
    return f"{pair[0]}, {pair[1]}"
