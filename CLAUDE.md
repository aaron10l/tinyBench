# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TinyBench is an automated QA generation system for benchmarking LLMs on tabular data analysis tasks. It injects known statistical phenomena into datasets, generates questions about them, computes ground-truth answers, then evaluates how well LLMs can detect and reason about these phenomena.

## Running the Pipelines

All scripts run from the repo root using the `.venv` virtual environment (Python 3.9). There is no requirements.txt; dependencies are managed directly in the venv. All pipeline scripts use `src/` as the working directory for imports (run from repo root with `python src/...`).

```bash
# 1. Generate benchmark instances (phenomena injection + QA generation)
python src/phenomena_pipeline.py --seed 42
python src/phenomena_pipeline.py --summary data/standardized/summaries/bike_sharing_100.json --seed 42

# 2. Compute ground-truth answers for generated instances
python src/answer_pipeline.py
python src/answer_pipeline.py --instances-dir data/instances

# 3. Evaluate models — provider auto-detected from model name
python src/eval_pipeline.py --models deepseek-r1:8b qwen2.5:7b          # Ollama only
python src/eval_pipeline.py --models claude-opus-4-6 gpt-4o              # API only (needs .env)
python src/eval_pipeline.py --models deepseek-r1:8b claude-opus-4-6      # mixed
```

There is no formal test suite.

## Architecture

The system follows a four-stage pipeline:

1. **Data Preparation** — Base CSVs in `data/base/` are standardized into `data/standardized/` at multiple row counts (100, 500, 1000). Summaries in `data/standardized/summaries/` describe column metadata (dtype, kind, cardinality).

2. **Template Matching & Injection** (`phenomena_pipeline.py` + `find_applicable_templates.py`) — JSON templates in `templates/` define parameterized questions with slot constraints. The matcher validates dataset compatibility and assigns concrete column/value slots. Each matched template triggers a phenomena injector that modifies a copy of the data.

3. **Answer Computation** (`answer_pipeline.py`) — Walks `data/instances/{dataset}/seed_{N}/{injector}/manifest.json` files and runs the corresponding answer computer to produce ground-truth answers.

4. **Evaluation** (`eval_pipeline.py`) — Sends table + question to models, collects responses, saves results to `data/results/`. Provider is auto-detected from the model name; supports Ollama, Anthropic, and OpenAI in a single run.

## Dispatch Registries

Injectors and answer computers are dispatched via dicts in their respective `__init__.py` files:

- `src/injectors/__init__.py` — `INJECT_FN`: maps injector name (str) → `inject(df, params, rng)` function
- `src/answer_computers/__init__.py` — `COMPUTE_FN`: maps template_id (str, versioned like `"anomaly_riskier_group_v0"`) → `compute_answer(df, slot_assignments, effects)` function

## Injector ↔ Answer Computer Pairs

| Phenomenon | Injector | Answer Computer (template_id) | Category |
|---|---|---|---|
| Heteroskedastic groups | `heteroskedastic_injection` | `anomaly_riskier_group_v0` | surprise_anomaly |
| Bad row indicators | `bad_rows_injection` | `anomaly_data_quality_filter_v0` | surprise_anomaly |
| Misleading column names | `name_swap_injection` | `fi_leakage_topk_v0` | feature_importance |
| Simpson's paradox | `simpsons_paradox_injection` | `rca_performance_improve_v0` | root_cause |
| Change-point shift | `changepoint_injection` | `rca_retrain_point_v0` | root_cause |

### Injector interface

```python
def inject(df: pd.DataFrame, params: dict, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    # Returns (modified_df, {"type": str, "params": dict, "effects": dict})
```

Injectors must copy the dataframe, validate that the injection succeeded (e.g., Levene's test for variance), and raise `ValueError` on failure (the pipeline catches and skips).

### Answer computer interface

```python
def compute_answer(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    # effects is keyed by injector type: {"heteroskedastic_injection": {...}}
```

Answer computers verify effects before computing. Raise on verification failure.

## Adding a New Phenomenon

1. Create `src/injectors/{name}_injection.py` exporting `inject(df, params, rng)`
2. Create `src/answer_computers/{template_id}.py` exporting `compute_answer(df, slot_assignments, effects)`
3. Register both in their `__init__.py`: add to `INJECT_FN` and `COMPUTE_FN` dicts
4. Create template JSON in `templates/{category}/` (schema: `templates/template_schema.json`)
5. Template `phenomena[].param_mapping` uses `{SLOT_NAME}` references resolved from slot assignments

## Template System

Templates in `templates/{category}/` define:
- **Slots** with type (`column`/`string`/`int`/`float`), source (`dataset.target`/`column_values`/default), and constraints (`kind_any_of`, `dtype_any_of`, `unique_count_min/max`)
- **Requirements** (`min_feature_count`, `min_count_by_kind`, `min_count_by_dtype`, `min_count_by_dtype_any_of`)
- **Feature pool** rules for excluding columns (`exclude_target`, `exclude_kinds`, `exclude_dtypes`)
- **Phenomena** with `injector` name and `param_mapping` from slots to injector arguments

Schema: `templates/template_schema.json`

## Key Design Decisions

- **Deterministic generation**: RNG seeded via `MD5(f"{seed}:{injector_type}:{sorted(params.items())}") % 2^31`
- **One phenomenon per instance**: Each `data/instances/.../` directory has exactly one injected phenomenon with its own copy of the table
- **Manifest-driven**: `manifest.json` in each instance directory is the source of truth for metadata, QA pairs, and answers
- **Lazy API clients**: `eval_pipeline.py` only initializes Anthropic/OpenAI clients when the requested models need them
- **Deduplication**: Same injector + same resolved params = single instance even if multiple templates reference it

## Environment

- API keys go in `.env` (git-ignored): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
- `data/instances/` is git-ignored (generated output)
- Detailed phenomenon documentation: `pipeline_questions_phenomena.md`
