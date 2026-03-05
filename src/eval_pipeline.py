"""Unified QA evaluation pipeline — supports Anthropic and OpenAI models."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

SYSTEM_PROMPT = "You are a data analyst. Answer the question about the dataset concisely."

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

_TOOL_SPECS = {
    "run_python": {
        "description": "Execute Python code. df is pre-loaded as a pandas DataFrame. Use print() to produce output.",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    },
    "load_data": {
        "description": "Load the full dataset as a CSV string. Call this to inspect column names, data types, and raw values.",
        "properties": {},
        "required": [],
    },
    "get_semantic_context": {
        "description": "Retrieve a specific EBM semantic component for this dataset.",
        "properties": {
            "component": {
                "type": "string",
                "enum": ["overview", "feature_importances", "shape_function"],
                "description": (
                    "'overview': dataset metadata (target, row/feature counts). "
                    "'feature_importances': all features ranked by EBM importance with descriptions and stats. "
                    "'shape_function': EBM shape function for one feature — also pass 'feature' to specify which one."
                ),
            },
            "feature": {
                "type": "string",
                "description": "Feature name. Required when component='shape_function'.",
            },
        },
        "required": ["component"],
    },
}

def _openai_tool(name: str) -> dict:
    spec = _TOOL_SPECS[name]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": spec["description"],
            "parameters": {"type": "object", "properties": spec["properties"],
                           "required": spec["required"], "additionalProperties": False},
        },
    }

def _anthropic_tool(name: str) -> dict:
    spec = _TOOL_SPECS[name]
    return {
        "name": name,
        "description": spec["description"],
        "input_schema": {"type": "object", "properties": spec["properties"],
                         "required": spec["required"], "additionalProperties": False},
    }

def _print_request(system: str, user_content: str, tools: list[dict]) -> None:
    """Print the full initial request payload for debugging."""
    sep = "─" * 72
    print(f"\n{sep}")
    print("DEBUG: INITIAL REQUEST")
    print(sep)
    print("SYSTEM PROMPT:")
    print(system)
    print(sep)
    print("USER MESSAGE:")
    print(user_content)
    if tools:
        print(sep)
        print("TOOLS:")
        print(json.dumps(tools, indent=2))
    print(sep + "\n")


def _build_tools_system_prompt(enabled_tools: set) -> str:
    base = SYSTEM_PROMPT_SEMANTIC if "get_semantic_context" in enabled_tools else SYSTEM_PROMPT
    tool_lines = []
    if "load_data" in enabled_tools:
        tool_lines.append("- load_data(): retrieves the full dataset as a CSV string")
    if "get_semantic_context" in enabled_tools:
        tool_lines.append("- get_semantic_context(component, feature?): retrieves EBM components — overview, feature_importances, or shape_function")
    if "run_python" in enabled_tools:
        tool_lines.append("- run_python(code): executes Python; df is pre-loaded as a DataFrame (pandas, numpy, scipy available)")
    tools_str = "\nTools:\n" + "\n".join(tool_lines)
    return base + tools_str + "\nRespond with text only when you have the final answer."


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

def _build_user_content(csv_text: str, question: str) -> str:
    """Assemble user message from dataset and question."""
    parts = []
    if csv_text:
        parts.append(f"## Dataset\n{csv_text}")
    parts.append(f"## Question\n{question}")
    return "\n\n".join(parts)


def query_anthropic(
    client,  # anthropic.Anthropic
    model: str,
    csv_text: str,
    question: str,
) -> tuple[str, str | None, dict]:
    """Query an Anthropic model using tool-use for structured output."""
    user_content = _build_user_content(csv_text, question)
    use_thinking = model in THINKING_MODELS
    kwargs: dict = dict(
        model=model,
        max_tokens=16_000,
        system=SYSTEM_PROMPT,
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

    start = time.time()
    ttft: float | None = None
    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            if ttft is None and event.type == "content_block_delta":
                ttft = time.time() - start
        msg = stream.get_final_message()
    total_latency = time.time() - start

    metrics = {
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
        "ttft_s": ttft,
        "total_latency_s": total_latency,
    }

    thinking: str | None = None
    answer = ""
    for block in msg.content:
        if block.type == "thinking":
            thinking = block.thinking
        elif block.type == "tool_use" and block.name == "submit_answer":
            answer = block.input.get("answer", "")

    return answer, thinking, metrics


def query_openai(
    client,  # openai.OpenAI
    model: str,
    csv_text: str,
    question: str,
) -> tuple[str, str | None, dict]:
    """Query an OpenAI model with JSON schema structured output."""
    user_content = _build_user_content(csv_text, question)

    start = time.time()
    ttft: float | None = None
    chunks: list[str] = []
    usage = None
    stream = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if ttft is None and chunk.choices and chunk.choices[0].delta.content:
            ttft = time.time() - start
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
        if chunk.usage:
            usage = chunk.usage
    raw = "".join(chunks)
    total_latency = time.time() - start

    metrics = {
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
        "ttft_s": ttft,
        "total_latency_s": total_latency,
    }

    try:
        answer = json.loads(raw)["answer"]
    except (json.JSONDecodeError, KeyError):
        answer = raw.strip()
    return answer, None, metrics


def query_openai_style_tools(
    client,        # openai.OpenAI
    model: str,
    csv_text: str,
    question: str,
    sandbox: "PythonSandbox",
    csv_path: Path,
    enabled_tools: set = frozenset({"run_python"}),
) -> tuple[str, str | None, list[dict], dict]:
    """Tool-calling query via OpenAI API."""
    system = _build_tools_system_prompt(enabled_tools)
    tools_list = [_openai_tool(t) for t in ("load_data", "run_python", "get_semantic_context") if t in enabled_tools]
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": _build_user_content(csv_text, question)},
    ]
    tool_use_log: list[dict] = []
    t0 = time.time()
    total_input = total_output = 0

    for _ in range(MAX_TOOL_ITER):
        resp = client.chat.completions.create(
            model=model, max_tokens=2048, messages=messages,
            tools=tools_list, tool_choice="auto",
        )
        if resp.usage:
            total_input += resp.usage.prompt_tokens
            total_output += resp.usage.completion_tokens
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_unset=True))

        if not msg.tool_calls:
            metrics = {"input_tokens": total_input, "output_tokens": total_output,
                       "ttft_s": None, "total_latency_s": time.time() - t0}
            return msg.content or "", None, tool_use_log, metrics  # final answer

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            if tool_name == "load_data":
                tool_use_log.append({"tool": "load_data"})
                result_str = csv_path.read_text()
            elif tool_name == "get_semantic_context":
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                component = args.get("component", "overview")
                feature = args.get("feature", "")
                tool_use_log.append({"tool": "get_semantic_context", "input": {"component": component, **({"feature": feature} if feature else {})}})
                try:
                    if component == "overview":
                        result_str = _semantic_overview(csv_path.parent)
                    elif component == "feature_importances":
                        result_str = _semantic_feature_importances(csv_path.parent)
                    elif component == "shape_function":
                        result_str = _semantic_shape_function(csv_path.parent, feature)
                    else:
                        result_str = f"Unknown component: {component}"
                except Exception as exc:
                    result_str = f"[get_semantic_context failed: {exc}]"
            else:  # run_python
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                code = args.get("code", "")
                tool_use_log.append({"tool": "run_python", "input": code})
                print(f"\n[sandbox] running code:\n{code}")
                result_str = sandbox.run(code, csv_path)
                print(f"[sandbox] result: {result_str!r}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

    metrics = {"input_tokens": total_input, "output_tokens": total_output,
               "ttft_s": None, "total_latency_s": time.time() - t0}
    return "", None, tool_use_log, metrics  # max iterations reached


def query_anthropic_tools(
    client,        # anthropic.Anthropic
    model: str,
    csv_text: str,
    question: str,
    sandbox: "PythonSandbox",
    csv_path: Path,
    enabled_tools: set = frozenset({"run_python"}),
) -> tuple[str, str | None, list[dict], dict]:
    """Tool-calling query via Anthropic API."""
    system = _build_tools_system_prompt(enabled_tools)
    tools_list = [_anthropic_tool(t) for t in ("load_data", "run_python", "get_semantic_context") if t in enabled_tools]
    use_thinking = model in THINKING_MODELS
    kwargs: dict = dict(
        model=model, max_tokens=16_000, system=system,
        tools=tools_list,
        tool_choice={"type": "auto"},
    )
    if use_thinking:
        kwargs["thinking"] = {"type": "adaptive"}

    messages: list[dict] = [
        {"role": "user", "content": _build_user_content(csv_text, question)},
    ]
    accumulated_thinking: str | None = None
    tool_use_log: list[dict] = []
    t0 = time.time()
    total_input = total_output = 0

    for _ in range(MAX_TOOL_ITER):
        kwargs["messages"] = messages
        msg = client.messages.create(**kwargs)
        total_input += msg.usage.input_tokens
        total_output += msg.usage.output_tokens

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
            metrics = {"input_tokens": total_input, "output_tokens": total_output,
                       "ttft_s": None, "total_latency_s": time.time() - t0}
            return " ".join(text_blocks), accumulated_thinking, tool_use_log, metrics  # final answer

        tool_results = []
        for block in tool_use_blocks:
            if block.name == "load_data":
                tool_use_log.append({"tool": "load_data"})
                result_str = csv_path.read_text()
            elif block.name == "get_semantic_context":
                component = block.input.get("component", "overview")
                feature = block.input.get("feature", "")
                tool_use_log.append({"tool": "get_semantic_context", "input": {"component": component, **({"feature": feature} if feature else {})}})
                try:
                    if component == "overview":
                        result_str = _semantic_overview(csv_path.parent)
                    elif component == "feature_importances":
                        result_str = _semantic_feature_importances(csv_path.parent)
                    elif component == "shape_function":
                        result_str = _semantic_shape_function(csv_path.parent, feature)
                    else:
                        result_str = f"Unknown component: {component}"
                except Exception as exc:
                    result_str = f"[get_semantic_context failed: {exc}]"
            else:  # run_python
                code = block.input.get("code", "")
                tool_use_log.append({"tool": "run_python", "input": code})
                result_str = sandbox.run(code, csv_path)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_str})

        messages.append({"role": "user", "content": tool_results})

    metrics = {"input_tokens": total_input, "output_tokens": total_output,
               "ttft_s": None, "total_latency_s": time.time() - t0}
    return "", accumulated_thinking, tool_use_log, metrics  # max iterations reached


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
# Semantic context helpers (called at tool-dispatch time)
# ---------------------------------------------------------------------------

def _semantic_overview(instance_dir: Path) -> str:
    """Return dataset metadata: target, row/feature counts, feature list."""
    sem_meta = json.loads((instance_dir / "semantic_metadata.json").read_text())
    feat_meta = json.loads((instance_dir / "feature_metadata.json").read_text())
    target = sem_meta["target"]
    n_rows = sem_meta["n_rows"]
    n_features = sem_meta["n_features"]
    features = list(feat_meta.keys())
    lines = [
        f"Target: {target}",
        f"Rows: {n_rows}",
        f"Features ({n_features}): {', '.join(features)}",
    ]
    return "\n".join(lines)


def _semantic_feature_importances(instance_dir: Path) -> str:
    """Return all features ranked by EBM importance with descriptions and stats."""
    feat_meta = json.loads((instance_dir / "feature_metadata.json").read_text())
    sorted_feats = sorted(
        feat_meta.items(),
        key=lambda kv: kv[1].get("importance", 0.0),
        reverse=True,
    )
    lines = ["Feature Importances (EBM, ranked):"]
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
    return "\n".join(lines)


def _semantic_shape_function(instance_dir: Path, feature: str) -> str:
    """Return the EBM shape function text for the named feature."""
    import gzip

    # Lazy vendor sys.path setup
    _vendor = Path(__file__).parent.parent / "vendor" / "intelligible-ai" / "src"
    if str(_vendor) not in sys.path:
        sys.path.insert(0, str(_vendor))
    from intelligible_ai.surprise_finder.grapher import EBMGraph, graph_to_text
    import numpy as np

    if not feature:
        return "Error: 'feature' parameter is required for component='shape_function'"

    with gzip.open(instance_dir / "graphs.json.gz", "rt", encoding="utf-8") as f:
        graphs_data = json.load(f)

    main_effects = graphs_data.get("main_effects", {})
    if feature not in main_effects:
        available = list(main_effects.keys())
        return f"Feature '{feature}' not found. Available features: {', '.join(available)}"

    entry = main_effects[feature]
    if "error" in entry:
        return f"Shape function for '{feature}' has an error: {entry['error']}"

    feat_type = entry["feature_type"]
    scores = np.array(entry["scores"])
    stds = np.array(entry["stds"])
    if feat_type == "continuous":
        x_vals = [tuple(pair) for pair in entry["x_vals"]]
    else:
        x_vals = entry["x_vals"]
    graph = EBMGraph(
        feature_name=feature,
        feature_type=feat_type,
        x_vals=x_vals,
        scores=scores,
        stds=stds,
    )
    return graph_to_text(graph)


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
    enabled_tools: set = frozenset(),
    columns_only: bool = False,
    csv_incontext: bool = False,
    debug: bool = False,
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

    # Sandbox (lazy, only when run_python is enabled)
    from sandbox import PythonSandbox
    sandbox = None
    if "run_python" in enabled_tools:
        sandbox = PythonSandbox()
        if sandbox.ping():
            print(f"[sandbox] ready ({sandbox._python})")
        else:
            print(f"[sandbox] WARNING: could not launch Python at {sandbox._python}")

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
    _debug_printed = False

    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        instance_dir = manifest_path.parent
        if "load_data" in enabled_tools:
            csv_text = ""
        elif columns_only:
            import pandas as pd
            cols = pd.read_csv(instance_dir / "table.csv", nrows=0).columns.tolist()
            csv_text = "Columns: " + ", ".join(cols)
        elif csv_incontext:
            csv_text = (instance_dir / "table.csv").read_text()
        else:
            csv_text = ""

        dataset = manifest["dataset_name"]
        injector = manifest["phenomenon"]["injector_type"]

        for qa in manifest["qa_pairs"]:
            if qa["answer"] is None:
                continue

            question = qa["question"]
            expected = qa["answer"]
            template_id = qa["template_id"]
            answer_format = qa["answer_format"]

            for model in models:
                provider = model_providers[model]
                use_tools = bool(enabled_tools) and provider in QUERY_FN_TOOLS

                print(f"[{model}] {dataset}/{injector} → ", end="", flush=True)

                query_fn = QUERY_FN_TOOLS[provider] if use_tools else QUERY_FN[provider]
                client = clients[provider]

                thinking: str | None = None
                tool_use_log: list[dict] = []
                metrics: dict = {}
                if debug and not _debug_printed:
                    system = _build_tools_system_prompt(enabled_tools) if use_tools else SYSTEM_PROMPT
                    user_content = _build_user_content(csv_text, question)
                    tools_list = (
                        [_anthropic_tool(t) for t in ("load_data", "run_python", "get_semantic_context") if t in enabled_tools]
                        if provider == "anthropic" else
                        [_openai_tool(t) for t in ("load_data", "run_python", "get_semantic_context") if t in enabled_tools]
                    ) if use_tools else []
                    _print_request(system, user_content, tools_list)
                    _debug_printed = True
                try:
                    if use_tools:
                        model_answer, thinking, tool_use_log, metrics = query_fn(
                            client, model, csv_text, question,
                            sandbox, instance_dir / "table.csv",
                            enabled_tools=enabled_tools,
                        )
                    else:
                        model_answer, thinking, metrics = query_fn(
                            client, model, csv_text, question,
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
                result["metrics"] = metrics
                if enabled_tools:
                    result["tools_enabled"] = sorted(enabled_tools)
                    result["tool_use"] = tool_use_log
                if thinking:
                    result["thinking"] = thinking
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
        nargs="+",
        choices=["load_data", "run_python", "get_semantic_context"],
        default=None,
        metavar="TOOL",
        help="Tools to expose to the model: load_data, run_python, get_semantic_context, or any combination.",
    )
    parser.add_argument(
        "--columns-only",
        action="store_true",
        default=False,
        help="Replace the full CSV with just column names in the prompt (blind test).",
    )
    parser.add_argument(
        "--csv-incontext",
        action="store_true",
        default=False,
        help="Include the full CSV data in the prompt context.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print the full initial request (system prompt, user message, tools) before the first API call.",
    )
    args = parser.parse_args()

    if args.output is None:
        if len(args.models) == 1:
            safe = args.models[0].replace(":", "_").replace("/", "_")
            base = Path(f"data/results/eval_results_{safe}.json")
        else:
            base = DEFAULT_OUTPUT
        if args.tools:
            suffix = "_tools_" + "_".join(sorted(args.tools))
            base = base.with_stem(base.stem + suffix)
        if args.columns_only:
            base = base.with_stem(base.stem + "_columns_only")
        if args.csv_incontext:
            base = base.with_stem(base.stem + "_csv")
        args.output = base

    enabled_tools = set(args.tools) if args.tools else set()
    results = run_eval(
        args.models, args.instances_dir, args.output,
        dataset=args.dataset, injector=args.injector,
        enabled_tools=enabled_tools,
        columns_only=args.columns_only,
        csv_incontext=args.csv_incontext,
        debug=args.debug,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
