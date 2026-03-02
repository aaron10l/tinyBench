"""Unified QA evaluation pipeline — supports Anthropic and OpenAI models."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

from io_utils import load_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    # Anthropic
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    # OpenAI
    "gpt-4o",
    "gpt-4o-mini",
    "o4-mini",
]
DEFAULT_INSTANCES_DIR = Path("data/instances")
DEFAULT_OUTPUT = Path("data/results/eval_results.json")

THINKING_MODELS = {"claude-opus-4-6", "claude-sonnet-4-6"}

SYSTEM_PROMPT = (
    "You are a data analyst. You are given a CSV dataset and a question about it.\n"
    "Answer the question concisely. Do not explain your reasoning."
)

SYSTEM_PROMPT_SEMANTIC = (
    "You are an expert data analyst specializing in identifying genuine statistical anomalies "
    "in Explainable Boosting Machine (EBM) model outputs.\n\n"
    "EBMs are Generalized Additive Models where predictions are computed additively: "
    "prediction = intercept + f₁(x₁) + f₂(x₂) + ... Each graph shows ONE term's learned "
    "contribution. The X-axis shows feature values; the Y-axis shows that feature's additive "
    "contribution to predictions.\n\n"
    "For CLASSIFICATION: Y-axis is log-odds. Score 0 = neutral; ±0.3 is meaningful; ±0.7 is strong.\n"
    "For REGRESSION: Y-axis is in target units. Significance depends on target scale.\n\n"
    "Normal EBM behavior (NOT anomalies): piecewise constant (step function) appearance, "
    "sharp jumps between bins, confidence bands widening at sparse tails.\n\n"
    "You are a data analyst. Answer the question concisely. Do not explain your reasoning."
)

# Shared JSON schema used by all providers for structured output.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

MAX_TOOL_ITER = 10

SYSTEM_PROMPT_TOOLS = (
    "You are a data analyst. You are given a CSV dataset and a question about it.\n"
    "You have access to a Python execution tool. The variable df is already loaded "
    "as a pandas DataFrame. Available libraries: pandas (as pd), numpy (as np), scipy.\n"
    "Use run_python to compute values. When you have the final answer, respond with text only."
)

SYSTEM_PROMPT_SEMANTIC_TOOLS = (
    SYSTEM_PROMPT_SEMANTIC + "\n\n"
    "You also have access to a Python execution tool. The variable df is already loaded "
    "as a pandas DataFrame. Available libraries: pandas (as pd), numpy (as np), scipy.\n"
    "Use run_python to compute values. When you have the final answer, respond with text only."
)

TOOL_RUN_PYTHON = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Execute Python code. df is pre-loaded as a pandas DataFrame. Use print() to produce output.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}

ANTHROPIC_TOOL_RUN_PYTHON = {
    "name": "run_python",
    "description": "Execute Python code. df is pre-loaded as a pandas DataFrame. Use print() to produce output.",
    "input_schema": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
        "additionalProperties": False,
    },
}

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def detect_provider(model: str) -> str:
    """Infer the provider from the model name."""
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    raise ValueError(
        f"Cannot infer provider for model '{model}'. "
        "Anthropic models start with 'claude'; "
        "OpenAI models start with 'gpt-', 'o1', 'o3', or 'o4'."
    )


# ---------------------------------------------------------------------------
# Query functions  (client, model, csv_text, question) -> (answer, thinking)
# ---------------------------------------------------------------------------

def _build_user_content(csv_text: str, question: str, semantic_context: str = "") -> str:
    """Assemble user message, optionally injecting semantic context between data and question."""
    parts = []
    if csv_text:
        parts.append(f"## Dataset\n{csv_text}")
    if semantic_context:
        parts.append(semantic_context)
    parts.append(f"## Question\n{question}")
    return "\n\n".join(parts)


def query_anthropic(
    client,  # anthropic.Anthropic
    model: str,
    csv_text: str,
    question: str,
    semantic_context: str = "",
) -> tuple[str, str | None]:
    """Query an Anthropic model using tool-use for structured output."""
    system = SYSTEM_PROMPT_SEMANTIC if semantic_context else SYSTEM_PROMPT
    user_content = _build_user_content(csv_text, question, semantic_context)
    use_thinking = model in THINKING_MODELS
    kwargs: dict = dict(
        model=model,
        max_tokens=16_000,
        system=system,
        messages=[{"role": "user", "content": user_content}],
        tools=[
            {
                "name": "submit_answer",
                "description": "Submit the answer to the question.",
                "input_schema": ANSWER_SCHEMA,
            }
        ],
        # Forced tool_choice is incompatible with extended thinking; use "auto" instead.
        tool_choice={"type": "auto"} if use_thinking else {"type": "tool", "name": "submit_answer"},
    )
    if use_thinking:
        kwargs["thinking"] = {"type": "adaptive"}

    msg = client.messages.create(**kwargs)

    thinking: str | None = None
    answer = ""
    for block in msg.content:
        if block.type == "thinking":
            thinking = block.thinking
        elif block.type == "tool_use" and block.name == "submit_answer":
            answer = block.input.get("answer", "")

    return answer, thinking


def query_openai(
    client,  # openai.OpenAI
    model: str,
    csv_text: str,
    question: str,
    semantic_context: str = "",
) -> tuple[str, str | None]:
    """Query an OpenAI model with JSON schema structured output."""
    system = SYSTEM_PROMPT_SEMANTIC if semantic_context else SYSTEM_PROMPT
    user_content = _build_user_content(csv_text, question, semantic_context)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": ANSWER_SCHEMA,
            },
        },
    )
    raw = resp.choices[0].message.content or ""
    try:
        answer = json.loads(raw)["answer"]
    except (json.JSONDecodeError, KeyError):
        answer = raw.strip()
    return answer, None


def query_openai_style_tools(
    client,        # openai.OpenAI
    model: str,
    csv_text: str,
    question: str,
    sandbox: "PythonSandbox",
    csv_path: Path,
    semantic_context: str = "",
) -> tuple[str, str | None, list[str]]:
    """Tool-calling query via OpenAI API."""
    system = SYSTEM_PROMPT_SEMANTIC_TOOLS if semantic_context else SYSTEM_PROMPT_TOOLS
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": _build_user_content(csv_text, question, semantic_context)},
    ]
    collected_code: list[str] = []
    for _ in range(MAX_TOOL_ITER):
        resp = client.chat.completions.create(
            model=model, max_tokens=2048, messages=messages,
            tools=[TOOL_RUN_PYTHON], tool_choice="auto",
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_unset=True))

        if not msg.tool_calls:
            return msg.content or "", None, collected_code  # final answer

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            code = args.get("code", "")
            collected_code.append(code)
            print(f"\n[sandbox] running code:\n{code}")
            result_str = sandbox.run(code, csv_path)
            print(f"[sandbox] result: {result_str!r}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

    return "", None, collected_code  # max iterations reached


def query_anthropic_tools(
    client,        # anthropic.Anthropic
    model: str,
    csv_text: str,
    question: str,
    sandbox: "PythonSandbox",
    csv_path: Path,
    semantic_context: str = "",
) -> tuple[str, str | None, list[str]]:
    """Tool-calling query via Anthropic API."""
    system = SYSTEM_PROMPT_SEMANTIC_TOOLS if semantic_context else SYSTEM_PROMPT_TOOLS
    use_thinking = model in THINKING_MODELS
    kwargs: dict = dict(
        model=model, max_tokens=16_000, system=system,
        tools=[ANTHROPIC_TOOL_RUN_PYTHON],
        tool_choice={"type": "auto"},
    )
    if use_thinking:
        kwargs["thinking"] = {"type": "adaptive"}

    messages: list[dict] = [
        {"role": "user", "content": _build_user_content(csv_text, question, semantic_context)},
    ]
    accumulated_thinking: str | None = None
    collected_code: list[str] = []

    for _ in range(MAX_TOOL_ITER):
        kwargs["messages"] = messages
        msg = client.messages.create(**kwargs)

        tool_use_blocks, text_blocks = [], []
        for block in msg.content:
            if block.type == "thinking":
                accumulated_thinking = block.thinking
            elif block.type == "tool_use":
                tool_use_blocks.append(block)
            elif block.type == "text":
                text_blocks.append(block.text)

        messages.append({"role": "assistant", "content": msg.content})

        if not tool_use_blocks:
            return " ".join(text_blocks), accumulated_thinking, collected_code  # final answer

        tool_results = []
        for block in tool_use_blocks:
            code = block.input.get("code", "")
            collected_code.append(code)
            result_str = sandbox.run(code, csv_path)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_str})

        messages.append({"role": "user", "content": tool_results})

    return "", accumulated_thinking, collected_code  # max iterations reached


# ---------------------------------------------------------------------------
# Provider registry  (mirrors INJECT_FN / COMPUTE_FN pattern)
# ---------------------------------------------------------------------------

QUERY_FN: dict[str, Callable] = {
    "anthropic": query_anthropic,
    "openai": query_openai,
}

QUERY_FN_TOOLS: dict[str, Callable] = {
    "anthropic": query_anthropic_tools,
    "openai": query_openai_style_tools,
}


# ---------------------------------------------------------------------------
# Semantic context builder
# ---------------------------------------------------------------------------

_TOP_FEATURES_FOR_SHAPES = 5  # how many shape functions to include in the prompt


def _build_semantic_context(instance_dir: Path) -> str:
    """Build a compact semantic context block from pre-generated component files.

    Loads semantic_metadata.json, feature_metadata.json, and graphs.json.gz,
    then formats them into a text block suitable for injection into the prompt.
    """
    import gzip
    import json

    # Vendor imports (sys.path already patched by run_eval when semantic=True)
    from intelligible_ai.surprise_finder.grapher import EBMGraph, graph_to_text
    import numpy as np

    sem_meta = json.loads((instance_dir / "semantic_metadata.json").read_text())
    feat_meta = json.loads((instance_dir / "feature_metadata.json").read_text())

    target = sem_meta["target"]
    n_rows = sem_meta["n_rows"]
    n_features = sem_meta["n_features"]

    # Sort features by importance descending
    sorted_feats = sorted(
        feat_meta.items(),
        key=lambda kv: kv[1].get("importance", 0.0),
        reverse=True,
    )

    lines: list[str] = [
        "## Semantic Context",
        f"Target: {target} | Rows: {n_rows} | Features: {n_features}",
        "",
        "Feature Importances (EBM, ranked):",
    ]
    for rank, (fname, finfo) in enumerate(sorted_feats, 1):
        imp = finfo.get("importance", 0.0)
        desc = finfo.get("description", "")
        stats_parts = []
        if "min" in finfo and "max" in finfo:
            stats_parts.append(f"Range {finfo['min']}–{finfo['max']}")
        if "mean" in finfo:
            stats_parts.append(f"mean {finfo['mean']}")
        if "n_unique" in finfo:
            stats_parts.append(f"{finfo['n_unique']} unique values")
        stats_str = ", ".join(stats_parts)
        desc_str = f" — {desc}" if desc else ""
        suffix = f". {stats_str}." if stats_str else ""
        lines.append(f"{rank}. {fname} ({imp:.3f}){desc_str}{suffix}")

    # EBM shape functions for top N features
    try:
        with gzip.open(instance_dir / "graphs.json.gz", "rt", encoding="utf-8") as f:
            graphs_data = json.load(f)

        main_effects = graphs_data.get("main_effects", {})
        # Take top N features that have valid graph data
        top_feats_with_graphs = [
            (fname, finfo)
            for fname, finfo in sorted_feats
            if fname in main_effects and "error" not in main_effects[fname]
        ][:_TOP_FEATURES_FOR_SHAPES]

        if top_feats_with_graphs:
            lines.append("")
            lines.append("## EBM Shape Functions")
            for fname, _ in top_feats_with_graphs:
                entry = main_effects[fname]
                feat_type = entry["feature_type"]
                scores = np.array(entry["scores"])
                stds = np.array(entry["stds"])
                if feat_type == "continuous":
                    x_vals = [tuple(pair) for pair in entry["x_vals"]]
                else:
                    x_vals = entry["x_vals"]
                graph = EBMGraph(
                    feature_name=fname,
                    feature_type=feat_type,
                    x_vals=x_vals,
                    scores=scores,
                    stds=stds,
                )
                try:
                    shape_text = graph_to_text(graph)
                    lines.append(shape_text)
                except Exception as exc:
                    lines.append(f"[{fname}: shape function unavailable — {exc}]")
    except Exception as exc:
        lines.append(f"\n[EBM graphs unavailable: {exc}]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Instance discovery
# ---------------------------------------------------------------------------

def discover_instances(
    instances_dir: Path,
    dataset: str | None = None,
    injector: str | None = None,
) -> list[Path]:
    """Return manifests sorted by CSV size (smallest datasets first).

    Args:
        dataset:  Substring match against the dataset folder name (e.g. 'bike_sharing_100').
        injector: Substring match against the injector folder name (e.g. 'name_swap').
    """
    manifests = list(instances_dir.glob("**/manifest.json"))
    if dataset:
        manifests = [m for m in manifests if m.parts[-4] == dataset]
    if injector:
        manifests = [m for m in manifests if injector in m.parts[-2]]
    return sorted(manifests, key=lambda m: (m.parent / "table.csv").stat().st_size)


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def run_eval(
    models: list[str],
    instances_dir: Path,
    output_path: Path,
    dataset: str | None = None,
    injector: str | None = None,
    tools: bool = False,
    columns_only: bool = False,
    semantic: bool = False,
    csv_incontext: bool = False,
) -> list[dict]:
    manifests = discover_instances(instances_dir, dataset=dataset, injector=injector)
    if columns_only and dataset is None:
        manifests = [m for m in manifests if m.parts[-4].endswith("_100")]
    if not manifests:
        print(f"No manifests found under {instances_dir}")
        sys.exit(1)

    # Detect providers for each model up front
    model_providers: dict[str, str] = {}
    for model in models:
        try:
            model_providers[model] = detect_provider(model)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

    # Sandbox (lazy, only when tools=True)
    from sandbox import PythonSandbox
    sandbox = None
    if tools:
        sandbox = PythonSandbox()
        if sandbox.ping():
            print(f"[sandbox] ready ({sandbox._python})")
        else:
            print(f"[sandbox] WARNING: could not launch Python at {sandbox._python}")

    # Semantic layer setup (lazy — only when --semantic is passed)
    if semantic:
        _vendor = Path(__file__).parent.parent / "vendor" / "intelligible-ai" / "src"
        if str(_vendor) not in sys.path:
            sys.path.insert(0, str(_vendor))
        from generate_semantic_components import semantic_components_exist

    # Lazy client initialization — only for providers actually needed
    from dotenv import load_dotenv
    load_dotenv()

    clients: dict[str, object] = {}

    if any(p == "anthropic" for p in model_providers.values()):
        import anthropic as _anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set in .env")
            sys.exit(1)
        clients["anthropic"] = _anthropic.Anthropic(api_key=api_key)

    if any(p == "openai" for p in model_providers.values()):
        import openai as _openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set in .env")
            sys.exit(1)
        clients["openai"] = _openai.OpenAI(api_key=api_key)

    results: list[dict] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        instance_dir = manifest_path.parent
        if columns_only:
            import pandas as pd
            cols = pd.read_csv(instance_dir / "table.csv", nrows=0).columns.tolist()
            csv_text = "Columns: " + ", ".join(cols)
        elif csv_incontext:
            csv_text = (instance_dir / "table.csv").read_text()
        else:
            csv_text = ""

        dataset = manifest["dataset_name"]
        injector = manifest["phenomenon"]["injector_type"]

        # Load pre-computed semantic components (generate via generate_semantic_components.py)
        semantic_context = ""
        semantic_ok = False
        if semantic:
            if semantic_components_exist(instance_dir):
                try:
                    semantic_context = _build_semantic_context(instance_dir)
                    semantic_ok = True
                except Exception as exc:
                    print(f"  [semantic] WARNING: context build failed for {dataset}/{injector}: {exc} — skipping")
                    continue
            else:
                print(f"  [semantic] WARNING: components not found for {dataset}/{injector} — skipping (run generate_semantic_components.py first)")
                continue

        for qa in manifest["qa_pairs"]:
            if qa["answer"] is None:
                continue

            question = qa["question"]
            expected = qa["answer"]
            template_id = qa["template_id"]
            answer_format = qa["answer_format"]

            for model in models:
                provider = model_providers[model]
                use_tools = tools and provider in QUERY_FN_TOOLS

                print(f"[{model}] {dataset}/{injector} → ", end="", flush=True)

                query_fn = QUERY_FN_TOOLS[provider] if use_tools else QUERY_FN[provider]
                client = clients[provider]

                thinking: str | None = None
                code_list: list[str] = []
                try:
                    if use_tools:
                        model_answer, thinking, code_list = query_fn(
                            client, model, csv_text, question,
                            sandbox, instance_dir / "table.csv",
                            semantic_context=semantic_context,
                        )
                    else:
                        model_answer, thinking = query_fn(
                            client, model, csv_text, question,
                            semantic_context=semantic_context,
                        )
                except Exception as exc:
                    model_answer = f"ERROR: {exc}"

                print(f"expected: {expected} | got: {model_answer}")

                result = {
                    "dataset": dataset,
                    "injector": injector,
                    "template_id": template_id,
                    "model": model,
                    "question": question,
                    "expected_answer": expected,
                    "model_answer": model_answer,
                    "answer_format": answer_format,
                }
                if tools:
                    result["tools_enabled"] = use_tools
                    result["code"] = code_list
                if thinking:
                    result["thinking"] = thinking
                if semantic:
                    result["semantic"] = semantic_ok
                results.append(result)
                output_path.write_text(json.dumps(results, indent=2))

    print(f"\nResults saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    if not results:
        return

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        counts[r["model"]][r["template_id"]] += 1

    templates = sorted({r["template_id"] for r in results})
    models = sorted(counts.keys())

    col_w = max(len(t) for t in templates) + 2
    model_w = max(len(m) for m in models) + 2
    header = "model".ljust(model_w) + "".join(t.ljust(col_w) for t in templates)
    print(f"\n{'=' * len(header)}")
    print("Summary (model × template counts)")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))
    for m in models:
        row = m.ljust(model_w) + "".join(
            str(counts[m].get(t, 0)).ljust(col_w) for t in templates
        )
        print(row)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run QA eval against any combination of Anthropic and OpenAI models.\n"
            "Provider is auto-detected from the model name:\n"
            "  - Anthropic: name starts with 'claude'\n"
            "  - OpenAI:    name starts with 'gpt-', 'o1', 'o3', or 'o4'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to evaluate (provider auto-detected)",
    )
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=DEFAULT_INSTANCES_DIR,
        help="Root directory containing instance manifests",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the JSON results file (auto-named per model when omitted)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Filter instances by dataset name substring (e.g. 'bike_sharing_100')",
    )
    parser.add_argument(
        "--injector",
        default=None,
        help="Filter instances by injector name substring (e.g. 'name_swap')",
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        default=False,
        help="Enable Python code execution tool.",
    )
    parser.add_argument(
        "--columns-only",
        action="store_true",
        default=False,
        help="Replace the full CSV with just column names in the prompt (blind test).",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        default=False,
        help="Generate and include semantic layer components (EBM shapes, feature metadata) in the prompt.",
    )
    parser.add_argument(
        "--csv-incontext",
        action="store_true",
        default=False,
        help="Include the full CSV data in the prompt context.",
    )
    args = parser.parse_args()

    if args.output is None:
        if len(args.models) == 1:
            safe = args.models[0].replace(":", "_").replace("/", "_")
            base = Path(f"data/results/eval_results_{safe}.json")
        else:
            base = DEFAULT_OUTPUT
        if args.tools:
            base = base.with_stem(base.stem + "_tools")
        if args.columns_only:
            base = base.with_stem(base.stem + "_columns_only")
        if args.semantic:
            base = base.with_stem(base.stem + "_semantic")
        if args.csv_incontext:
            base = base.with_stem(base.stem + "_csv")
        args.output = base

    results = run_eval(
        args.models, args.instances_dir, args.output,
        dataset=args.dataset, injector=args.injector, tools=args.tools,
        columns_only=args.columns_only, semantic=args.semantic,
        csv_incontext=args.csv_incontext,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
