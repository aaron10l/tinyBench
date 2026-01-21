# src/io_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))