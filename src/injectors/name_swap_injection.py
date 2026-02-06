"""Name-swap injector.

Renames all columns except the target column to random strings.
Goal is to disable name-based heuristics for feature importance.

TODO:
- discuss with team if this injection is useful.
"""
from __future__ import annotations

import string

import numpy as np
import pandas as pd


# Pool of neutral, non-descriptive column name prefixes
_PREFIXES = [
    "var", "col", "attr", "field", "val", "feat", "x", "v", "f", "c", "a", "m",
]


def _generate_random_name(rng: np.random.Generator, length: int = 4) -> str:
    """Generate a random column name like 'var_a3x9' or 'col_m2k7'."""
    prefix = rng.choice(_PREFIXES)
    suffix = "".join(rng.choice(list(string.ascii_lowercase + string.digits), size=length))
    return f"{prefix}_{suffix}"


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

    # Generate unique random names for all non-target columns
    columns_to_rename = [c for c in df.columns if c != target_col]

    used_names = {target_col}
    rename_map = {}

    for old_name in columns_to_rename:
        # Generate unique new name
        new_name = _generate_random_name(rng)
        while new_name in used_names:
            new_name = _generate_random_name(rng)

        rename_map[old_name] = new_name
        used_names.add(new_name)

    # Apply the renaming
    df = df.rename(columns=rename_map)

    # Build reverse mapping for answer computation (new_name -> old_name)
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
