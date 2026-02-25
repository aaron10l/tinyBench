"""Bad-rows indicator injector v1.

Same as bad_rows_injection, but renames every column except the outcome column
to a random 5-letter string after injection. Column names give no semantic
signal — the model must reason purely from the data values.
"""
from __future__ import annotations

import string

import numpy as np
import pandas as pd


def _unique_random_name(rng: np.random.Generator, taken: set[str], length: int = 5) -> str:
    letters = list(string.ascii_lowercase)
    while True:
        name = "".join(rng.choice(letters, size=length))
        if name not in taken:
            taken.add(name)
            return name


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """
    Parameters
    ----------
    df : DataFrame to modify (will be copied).
    params : Must contain ``outcome_col`` (str).
    rng : NumPy random Generator for reproducibility.

    Returns
    -------
    (modified_df, phenomenon_dict)
    """
    df = df.copy()
    outcome_col = params["outcome_col"]

    # Mark ~10% of rows as "bad"
    bad_fraction = 0.1
    n = len(df)
    n_bad = max(1, int(n * bad_fraction))
    bad_indices = rng.choice(n, size=n_bad, replace=False)

    # Pick from neutral names so the column doesn't stand out before renaming
    _CANDIDATE_NAMES = [
        "region", "channel", "segment", "cohort", "tier",
        "variant", "cluster", "period", "quarter", "sample",
        "trial", "run", "split", "bucket", "partition",
        "group", "class", "phase", "batch", "slot",
    ]
    existing = set(df.columns)
    candidates = [n for n in _CANDIDATE_NAMES if n not in existing]
    indicator_col = candidates[rng.integers(0, len(candidates))] + "_flag"

    insert_pos = rng.integers(0, max(1, len(df.columns)))
    if insert_pos == len(df.columns):
        insert_pos = max(0, len(df.columns) - 1)
    df.insert(insert_pos, indicator_col, 0)
    df.iloc[bad_indices, df.columns.get_loc(indicator_col)] = 1

    # Corrupt the outcome for bad rows: add large noise
    outcome_values = df[outcome_col].astype(float)
    outcome_std = outcome_values.std()
    if outcome_std == 0:
        outcome_std = 1.0

    corruption = np.abs(rng.normal(0, outcome_std * 0.5, n_bad))
    corruption = np.minimum(corruption, outcome_std * 2.0)
    new_values = outcome_values.iloc[bad_indices].values + corruption
    non_null = outcome_values.dropna()
    if len(non_null) > 0 and np.all(np.isclose(non_null.values, np.rint(non_null.values))):
        new_values = np.rint(new_values).astype(float)
    df.iloc[bad_indices, df.columns.get_loc(outcome_col)] = new_values

    # Rename every column except outcome_col to a random 5-letter string.
    taken = {outcome_col}
    rename_map = {}
    for col in df.columns:
        if col == outcome_col:
            continue
        rename_map[col] = _unique_random_name(rng, taken)

    df = df.rename(columns=rename_map)
    renamed_indicator_col = rename_map[indicator_col]

    effects = {
        "indicator_col": renamed_indicator_col,
        "n_bad_rows": int(n_bad),
        "bad_fraction": bad_fraction,
        "bad_indices": bad_indices.tolist(),
    }

    return df, {
        "type": "bad_rows_injection_v1",
        "params": params,
        "effects": effects,
    }
