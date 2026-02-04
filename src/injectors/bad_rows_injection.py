"""Bad-rows indicator injector.

Injects a column that identifies rows with corrupted / nonsensical outcomes.
Removing rows flagged by this column dramatically improves outcome consistency.

TODO:
- Is the name of the column too obvious/too big of a hint?
- i am currently injecting 10% of rows as bad. All of these rows are labeled as bad via the check_flag column.
  should i have the check_flag column only label 90% of the bad rows as bad, and mislabel the other 10% as good?
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
    params : Must contain ``outcome_col`` (str).
    rng : NumPy random Generator for reproducibility.

    Returns
    -------
    (modified_df, phenomenon_dict)
    """
    df = df.copy()
    outcome_col = params["outcome_col"]

    # Mark ~15% of rows as "bad"
    bad_fraction = 0.1
    n = len(df)
    n_bad = max(1, int(n * bad_fraction))
    bad_indices = rng.choice(n, size=n_bad, replace=False)

    # Create the indicator column (looks like a sensor/ETL flag)
    indicator_col = "_check_flag"
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
    # Match the column's numeric style: if all values are integer-like, keep whole numbers.
    non_null = outcome_values.dropna()
    if len(non_null) > 0 and np.all(np.isclose(non_null.values, np.rint(non_null.values))):
        new_values = np.rint(new_values).astype(float)
    df.iloc[bad_indices, df.columns.get_loc(outcome_col)] = new_values

    print(f"[bad_rows_injection] Modified rows: {n_bad}")

    effects = {
        "indicator_col": indicator_col,
        "n_bad_rows": int(n_bad),
        "bad_fraction": bad_fraction,
        "bad_indices": bad_indices.tolist(),
    }

    return df, {
        "type": "bad_rows_injection",
        "params": params,
        "effects": effects,
    }
