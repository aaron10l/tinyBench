"""Generate semantic layer components for benchmark instances (batch mode).

Produces four files per instance directory:
  - graphs.json.gz        EBM shape functions for all features
  - umap_embedding.json   2-D UMAP projection of the feature space
  - feature_metadata.json Per-feature stats + LLM-generated descriptions
  - semantic_metadata.json Dataset-level metadata + timing

Usage:
    python src/generate_semantic_components.py                   # walk all instances
    python src/generate_semantic_components.py --force            # regenerate all
    python src/generate_semantic_components.py --instances-dir data/instances
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Vendor path — must happen before any intelligible_ai imports
# ---------------------------------------------------------------------------

_VENDOR_PATH = Path(__file__).parent.parent / "vendor" / "intelligible-ai" / "src"
if str(_VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(_VENDOR_PATH))

from intelligible_ai.models.ebm import build_ebm
from intelligible_ai.surprise_finder.grapher import EBMGraph, extract_graph, graph_to_text

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEMANTIC_FILES = [
    "graphs.json.gz",
    "umap_embedding.json",
    "feature_metadata.json",
    "semantic_metadata.json",
]


def semantic_components_exist(instance_dir: Path) -> bool:
    """Return True only when all four component files are present."""
    return all((instance_dir / f).exists() for f in SEMANTIC_FILES)


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------


def generate_semantic_components(
    csv_path: Path,
    target_col: str,
    output_dir: Path,
) -> dict:
    """Train an EBM and write the four semantic component files.

    Returns a dict mapping component name → file path string.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1 — Load data & train EBM
    # ------------------------------------------------------------------
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"  [semantic] Training EBM on {len(df)} rows, {len(X.columns)} features...")
    ebm = build_ebm(list(X.columns), y)
    ebm.fit(X, y)
    print("  [semantic] EBM trained.")

    importances = ebm.term_importances()
    n_features = len(ebm.feature_names_in_)

    # ------------------------------------------------------------------
    # Step 2 — graphs.json.gz
    # ------------------------------------------------------------------
    main_effects: dict = {}
    for i in range(n_features):
        feat_name = str(ebm.feature_names_in_[i])
        importance = float(importances[i])
        try:
            g = extract_graph(ebm, i)
            if g.feature_type == "continuous":
                x_vals_json = [[float(a), float(b)] for (a, b) in g.x_vals]
            else:
                x_vals_json = [str(v) for v in g.x_vals]
            scores = [float(s) for s in g.scores]
            stds = [float(s) for s in g.stds]
            z = 1.96
            lower = [float(g.scores[j]) - z * float(g.stds[j]) for j in range(len(g.scores))]
            upper = [float(g.scores[j]) + z * float(g.stds[j]) for j in range(len(g.scores))]
            main_effects[feat_name] = {
                "importance": importance,
                "feature_type": g.feature_type,
                "x_vals": x_vals_json,
                "scores": scores,
                "stds": stds,
                "lower": lower,
                "upper": upper,
            }
        except Exception as exc:
            print(f"  [semantic] WARNING: graph extraction failed for {feat_name!r}: {exc}")
            main_effects[feat_name] = {"importance": importance, "error": str(exc)}

    interactions = []
    for i in range(n_features, len(ebm.term_names_)):
        term = ebm.term_names_[i]
        interactions.append({
            "features": list(term) if not isinstance(term, str) else [term],
            "importance": float(importances[i]),
        })

    graphs_data = {"main_effects": main_effects, "interactions": interactions}
    graphs_path = output_dir / "graphs.json.gz"
    with gzip.open(graphs_path, "wt", encoding="utf-8") as f:
        json.dump(graphs_data, f)
    print(f"  [semantic] Saved {graphs_path}")

    # ------------------------------------------------------------------
    # Step 3 — umap_embedding.json
    # ------------------------------------------------------------------
    has_umap = False
    umap_path = output_dir / "umap_embedding.json"
    try:
        import umap as umap_module

        reducer = umap_module.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(X.fillna(-1000))
        umap_data = {
            "x": [float(v) for v in embedding[:, 0]],
            "y": [float(v) for v in embedding[:, 1]],
            "target": y.tolist(),
        }
        with open(umap_path, "w") as f:
            json.dump(umap_data, f)
        has_umap = True
        print(f"  [semantic] Saved {umap_path}")
    except ImportError:
        print("  [semantic] WARNING: umap-learn not installed; skipping UMAP embedding")
        with open(umap_path, "w") as f:
            json.dump({}, f)

    # ------------------------------------------------------------------
    # Step 4 — feature_metadata.json (stats + LLM descriptions)
    # ------------------------------------------------------------------
    feat_names_list = list(ebm.feature_names_in_)
    feature_stats: dict[str, dict] = {}
    stats_lines: list[str] = []

    for col in X.columns:
        s = X[col]
        stats: dict = {
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique()),
            "n_missing": int(s.isna().sum()),
        }
        if pd.api.types.is_numeric_dtype(s):
            stats["min"] = float(s.min())
            stats["max"] = float(s.max())
            stats["mean"] = round(float(s.mean()), 4)
            stats["std"] = round(float(s.std()), 4)
            stats_lines.append(
                f"{col}: dtype={stats['dtype']}, n_unique={stats['n_unique']}, "
                f"min={stats['min']}, max={stats['max']}, mean={stats['mean']}"
            )
        else:
            top5 = s.value_counts().head(5).to_dict()
            stats["top_values"] = {str(k): int(v) for k, v in top5.items()}
            stats_lines.append(
                f"{col}: dtype={stats['dtype']}, n_unique={stats['n_unique']}, "
                f"top values={list(top5.keys())[:3]}"
            )
        feature_stats[col] = stats

    feature_descriptions: dict[str, str] = {}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            import openai

            oai_client = openai.OpenAI(api_key=api_key)
            stats_text = "\n".join(stats_lines)
            resp = oai_client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1024,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a data analyst. Return a JSON object mapping "
                            "feature names to 1-2 sentence plain English descriptions."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Dataset has {len(df)} rows. Target: {target_col}. "
                            f"Features:\n{stats_text}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            feature_descriptions = json.loads(raw)
            print(f"  [semantic] Got LLM descriptions for {len(feature_descriptions)} features")
        except Exception as exc:
            print(f"  [semantic] WARNING: LLM feature descriptions failed: {exc}")
    else:
        print("  [semantic] WARNING: OPENAI_API_KEY not set; skipping LLM descriptions")

    feature_metadata: dict[str, dict] = {}
    for col in X.columns:
        col_str = str(col)
        idx = feat_names_list.index(col_str) if col_str in feat_names_list else -1
        entry: dict = {
            "description": feature_descriptions.get(col_str, ""),
            "importance": float(importances[idx]) if idx >= 0 else 0.0,
        }
        entry.update(feature_stats[col])
        feature_metadata[col_str] = entry

    meta_path = output_dir / "feature_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(feature_metadata, f, indent=2)
    print(f"  [semantic] Saved {meta_path}")

    # ------------------------------------------------------------------
    # Step 5 — semantic_metadata.json
    # ------------------------------------------------------------------
    duration = time.time() - t_start
    semantic_metadata = {
        "target": target_col,
        "n_rows": len(df),
        "n_features": len(X.columns),
        "feature_names": [str(c) for c in X.columns],
        "compute_duration_seconds": round(duration, 2),
        "has_umap": has_umap,
    }
    sem_path = output_dir / "semantic_metadata.json"
    with open(sem_path, "w") as f:
        json.dump(semantic_metadata, f, indent=2)
    print(f"  [semantic] Saved {sem_path}")

    return {
        "graphs": str(graphs_path),
        "umap_embedding": str(umap_path),
        "feature_metadata": str(meta_path),
        "semantic_metadata": str(sem_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-generate semantic layer components for all benchmark instances."
    )
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=Path("data/instances"),
        help="Root directory to walk for manifest.json files (default: data/instances)",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=None,
        help="Filter instances by dataset name substrings (e.g. '_100 _500 _1000')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Regenerate components even if they already exist",
    )
    args = parser.parse_args()

    manifests = sorted(args.instances_dir.glob("**/manifest.json"))
    if args.dataset:
        manifests = [m for m in manifests if any(m.parent.parts[-3].endswith(d) for d in args.dataset)]
    if not manifests:
        print(f"No manifest.json files found under {args.instances_dir}")
        return

    total = len(manifests)
    print(f"Found {total} instance(s) under {args.instances_dir}\n")

    skipped = 0
    succeeded = 0
    failed = 0
    t_total = time.time()

    for idx, manifest_path in enumerate(manifests, 1):
        instance_dir = manifest_path.parent
        # Build a short label from the path: dataset/seed/injector
        parts = instance_dir.relative_to(args.instances_dir).parts
        label = "/".join(parts)

        if not args.force and semantic_components_exist(instance_dir):
            print(f"[{idx}/{total}] {label} — skipped (already exists)")
            skipped += 1
            continue

        # Read target column from manifest
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            target_col = manifest["target"]
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[{idx}/{total}] {label} — FAILED (bad manifest: {exc})")
            failed += 1
            continue

        csv_path = instance_dir / "table.csv"
        if not csv_path.exists():
            print(f"[{idx}/{total}] {label} — FAILED (table.csv not found)")
            failed += 1
            continue

        print(f"[{idx}/{total}] {label} — generating...")
        try:
            generate_semantic_components(csv_path, target_col, instance_dir)
            succeeded += 1
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failed += 1

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed:.1f}s — {succeeded} generated, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
