import csv
import random
from pathlib import Path
from typing import List, Dict, Optional


def _parse_int(value: str) -> Optional[int]:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            f = float(s)
        except ValueError:
            return None
        return int(f) if f.is_integer() else None


def _find_row_id_column(rows: List[Dict[str, str]], columns: List[str]) -> Optional[str]:
    rng = random.Random(0)
    sample_cols = rng.sample(columns, k=min(100, len(columns)))
    for col in sample_cols:
        for base in (0, 1):
            ok = True
            for row_idx, row in enumerate(rows, start=1):
                val = _parse_int(row.get(col, ""))
                if val is None or val != row_idx - base:
                    ok = False
                    break
            if ok:
                return col
    return None


def _read_rows(path: Path, limit: int = 10_000) -> tuple[List[Dict[str, str]], List[str]]:
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            return [], []
        rows = []
        for _, row in zip(range(limit), reader):
            rows.append(row)
        return rows, list(reader.fieldnames)


def _write_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base_dir = Path(__file__).parent
    standardized_dir = base_dir.parent / "standardized"
    standardized_dir.mkdir(parents=True, exist_ok=True)
    limit = 10_000

    for csv_path in base_dir.glob("*.csv"):
        rows, fieldnames = _read_rows(csv_path, limit=limit)
        if not fieldnames:
            continue

        id_col = _find_row_id_column(rows, fieldnames)
        add_row_id = id_col is None

        if add_row_id:
            out_fields = ["row_id"] + fieldnames
        elif id_col != "row_id":
            out_fields = ["row_id" if name == id_col else name for name in fieldnames]
        else:
            out_fields = fieldnames

        out_rows = []
        for row_idx, row in enumerate(rows, start=1):
            row = dict(row)
            if id_col and id_col != "row_id":
                row["row_id"] = row.pop(id_col, "")
            if add_row_id:
                row["row_id"] = str(row_idx)
            out_rows.append(row)

        out_name = f"{csv_path.stem}_{len(out_rows)}{csv_path.suffix}"
        _write_rows(standardized_dir / out_name, out_rows, out_fields)


if __name__ == "__main__":
    main()