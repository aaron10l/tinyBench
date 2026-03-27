from __future__ import annotations
from typing import Any
import pandas as pd


def compute_answer(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    inj = effects.get("dq_bad_row_indicator") or effects.get("dq_bad_row_indicator_v1")
    return inj["indicator_col"]
