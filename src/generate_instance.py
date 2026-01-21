# src/generate_instance.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

# Make src/ importable when running: python src/generate_instance.py
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from io_utils import load_csv, save_csv, load_json, save_json

from injectors.mcar_missingness import inject_mcar_missingness
from injectors.duplicate_feature import inject_duplicate_feature
from injectors.correlation_injection import inject_correlation_injection


INJECT_FN = {
    "mcar_missingness": inject_mcar_missingness,
    "duplicate_feature": inject_duplicate_feature,
    "correlation_injection": inject_correlation_injection,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out_root", default="data/instances", type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)

    base_csv = Path(cfg["base_dataset"])
    # print(f"base_csv: {base_csv}")
    summary_path = Path(cfg.get("summary", ""))
    # print(f"summary_path: {summary_path}")

    df = load_csv(base_csv)

    summary = load_json(summary_path) if summary_path.exists() else {}

    dataset_name = cfg.get("dataset_name", base_csv.stem)
    instance_id = f"{dataset_name}_seed_{args.seed}"

    rng = np.random.default_rng(args.seed)

    phenomena = []
    for spec in cfg["injectors"]:
        print(f"Applying injector: {spec}")
        injection_type = spec["type"]
        params = spec.get("params", {})

        if injection_type not in INJECT_FN:
            known = ", ".join(sorted(INJECT_FN.keys()))
            raise KeyError(f"Unknown injector type '{injection_type}'. Known: {known}")

        df, ph = INJECT_FN[injection_type](df, params, rng)
        phenomena.append(ph)

    out_dir = Path(args.out_root) / dataset_name / f"seed_{args.seed}"
    save_csv(df, out_dir / "table.csv")

    save_json(
        {
            "dataset_instance_id": instance_id,
            "seed": args.seed,
            "base_dataset": str(base_csv),
            "phenomena": phenomena,
        },
        out_dir / "manifest.json",
    )

    save_json(
        {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "columns": list(df.columns),
            "null_fraction_by_col": {c: float(df[c].isna().mean()) for c in df.columns},
        },
        out_dir / "instance_summary.json",
    )

    print(f"Wrote instance to: {out_dir}")


if __name__ == "__main__":
    main()