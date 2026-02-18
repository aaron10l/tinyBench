"""Name-swap injector.

Permutes column names among non-target columns (derangement) so that every
column receives a different column's name.  This disables name-based
heuristics while keeping plausible-looking headers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _derangement(rng: np.random.Generator, n: int, max_iter: int = 1000) -> np.ndarray:
    """Return a derangement (no fixed points) of range(n).

    Uses rejection sampling.  Falls back to rotate-by-1 if sampling
    doesn't converge within *max_iter* attempts.
    """
    if n < 2:
        raise ValueError("Cannot derange fewer than 2 elements")

    for _ in range(max_iter):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm

    # Fallback: simple rotation guarantees no fixed points
    return np.roll(np.arange(n), 1)


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """
    Parameters
    ----------
    df : DataFrame to modify (will be copied).
    params : Must contain ``target_col`` (str).
    rng : NumPy random Generator for reproducibility.

    Returns
    -------
    (modified_df, phenomenon_dict)
    """
    df = df.copy()
    target_col = params["target_col"]
    exclude_cols = set(params.get("exclude_cols", []))
    exclude_cols.update(params.get("id_no_cols", []))
    exclude_cols.add(target_col)

    columns_to_swap = [c for c in df.columns if c not in exclude_cols]

    if len(columns_to_swap) < 2:
        # Nothing meaningful to swap — return unmodified
        return df, {
            "type": "name_swap_injection",
            "params": params,
            "effects": {"original_to_new": {}, "new_to_original": {}},
        }

    perm = _derangement(rng, len(columns_to_swap))
    rename_map = {
        columns_to_swap[i]: columns_to_swap[int(perm[i])]
        for i in range(len(columns_to_swap))
    }

    df = df.rename(columns=rename_map)

    reverse_map = {v: k for k, v in rename_map.items()}

    effects = {
        "original_to_new": rename_map,
        "new_to_original": reverse_map,
    }

    return df, {
        "type": "name_swap_injection",
        "params": params,
        "effects": effects,
    }
