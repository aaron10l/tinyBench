from __future__ import annotations
from typing import Any
import pandas as pd


def compute_answer(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    inj = effects["anomaly_data_quality_filter"]
    return inj["indicator_col"]
