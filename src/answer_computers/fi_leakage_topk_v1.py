from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

"""
Ranks features by per-feature AUROC with the outcome column.
Matches the question template which asks for AUROC-maximizing features.
AUROC is computed via Mann-Whitney U / (n_pos * n_neg), then reflected
so anti-correlated features still rank highly (consistent with v0's abs(corr)).
"""


def compute_answer(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    outcome_col = slot_assignments["OUTCOME_COL"]
    k = int(slot_assignments.get("K", 5))
    target = pd.to_numeric(df[outcome_col], errors="coerce")

    # Binarize target at median if not already binary
    if target.nunique() > 2:
        target_binary = (target >= target.median()).astype(int)
    else:
        target_binary = target.astype(int)

    pos_mask = target_binary == 1
    neg_mask = target_binary == 0
    n_pos, n_neg = pos_mask.sum(), neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        return []

    id_no_cols = set(effects.get("fi_leakage_topk", {}).get("id_no_cols", []))
    feature_cols = [c for c in df.columns if c != outcome_col and c not in id_no_cols]

    auroc_scores: dict[str, float] = {}
    for col in feature_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().all():
            continue
        pos_vals = numeric[pos_mask].dropna()
        neg_vals = numeric[neg_mask].dropna()
        if len(pos_vals) == 0 or len(neg_vals) == 0:
            continue
        try:
            stat, _ = mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
            auc = stat / (len(pos_vals) * len(neg_vals))
            auroc_scores[col] = max(auc, 1 - auc)
        except Exception:
            continue

    ranked = sorted(auroc_scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:k]]
