"""Unified QA evaluation pipeline — supports Ollama, Anthropic, and OpenAI models."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import requests

from io_utils import load_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    # Ollama
    "deepseek-r1:8b",
    "qwen2.5:7b",
    "llama3.2:3b",
    "phi4:latest",
    # Anthropic
    "claude-opus-4-6",
    "claude-sonnet-4-5-20250929",
    # OpenAI
    "gpt-4o",
    "gpt-4o-mini",
]
DEFAULT_INSTANCES_DIR = Path("data/instances")
DEFAULT_OUTPUT = Path("data/results/eval_results.json")

OLLAMA_URL = "http://localhost:11434/api/chat"

THINKING_MODELS = {"claude-opus-4-6"}

SYSTEM_PROMPT = (
    "You are a data analyst. You are given a CSV dataset and a question about it.\n"
    "Answer the question concisely. Do not explain your reasoning."
)

# Shared JSON schema used by all providers for structured output.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

OLLAMA_V1_BASE_URL = "http://localhost:11434/v1"
MAX_TOOL_ITER = 10

SYSTEM_PROMPT_TOOLS = (
    "You are a data analyst. You are given a CSV dataset and a question about it.\n"
    "You have access to a Python execution tool. The variable df is already loaded "
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
    if ":" in model:
        return "ollama"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    raise ValueError(
        f"Cannot infer provider for model '{model}'. "
        "Ollama models contain ':', Anthropic models start with 'claude', "
        "OpenAI models start with 'gpt-', 'o1', 'o3', or 'o4'."
    )


# ---------------------------------------------------------------------------
# Tool-support detection
# ---------------------------------------------------------------------------

_ollama_tool_cache: dict[str, bool] = {}


def check_ollama_tool_support(model: str) -> bool:
    """Return True if the model's Ollama template supports tool calling."""
    if model in _ollama_tool_cache:
        return _ollama_tool_cache[model]
    try:
        r = subprocess.run(
            ["ollama", "show", model, "--modelfile"],
            capture_output=True, text=True, timeout=15,
        )
        supported = r.returncode == 0 and ".ToolCalls" in r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        supported = False
    _ollama_tool_cache[model] = supported
    return supported


# ---------------------------------------------------------------------------
# Query functions  (client, model, csv_text, question) -> (answer, thinking)
# ---------------------------------------------------------------------------

def query_ollama(
    _client: None,
    model: str,
    csv_text: str,
    question: str,
) -> tuple[str, str | None]:
    """Query an Ollama model via /api/chat with structured JSON output."""
    user_content = f"## Dataset\n{csv_text}\n\n## Question\n{question}"
    payload = {
        "model": model,
        "stream": False,
        "format": ANSWER_SCHEMA,
        "options": {"num_predict": 512},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    message = data.get("message", {})

    # Ollama 0.7+ separates thinking into message.thinking
    thinking: str | None = message.get("thinking") or None

    raw_content = message.get("content", "")
    try:
        answer = json.loads(raw_content)["answer"]
    except (json.JSONDecodeError, KeyError):
        answer = raw_content.strip()

    return answer, thinking


def query_anthropic(
    client,  # anthropic.Anthropic
    model: str,
    csv_text: str,
    question: str,
) -> tuple[str, str | None]:
    """Query an Anthropic model using tool-use for structured output."""
    user_content = f"## Dataset\n{csv_text}\n\n## Question\n{question}"
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
) -> tuple[str, str | None]:
    """Query an OpenAI model with JSON schema structured output."""
    user_content = f"## Dataset\n{csv_text}\n\n## Question\n{question}"
    resp = client.chat.completions.create(
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
    )
    raw = resp.choices[0].message.content or ""
    try:
        answer = json.loads(raw)["answer"]
    except (json.JSONDecodeError, KeyError):
        answer = raw.strip()
    return answer, None


def query_ollama_tools(
    _client: None,
    model: str,
    csv_text: str,
    question: str,
    sandbox: "PythonSandbox",
    csv_path: Path,
) -> tuple[str, str | None, list[str]]:
    """Tool-calling query via Ollama native /api/chat endpoint."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT_TOOLS},
        {"role": "user", "content": f"## Dataset\n{csv_text}\n\n## Question\n{question}"},
    ]
    thinking: str | None = None
    collected_code: list[str] = []

    for i in range(MAX_TOOL_ITER):
        payload = {
            "model": model,
            "stream": False,
            "messages": messages,
            "tools": [TOOL_RUN_PYTHON],
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
        resp.raise_for_status()
        message = resp.json().get("message", {})

        if message.get("thinking"):
            thinking = message["thinking"]

        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            return message.get("content", ""), thinking, collected_code

        messages.append({
            "role": "assistant",
            "content": message.get("content", ""),
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            args = tc.get("function", {}).get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            code = args.get("code", "")
            collected_code.append(code)
            print(f"\n[sandbox] running code:\n{code}")
            result_str = sandbox.run(code, csv_path)
            print(f"[sandbox] result: {result_str!r}")
            messages.append({"role": "tool", "content": result_str})

    return "", thinking, collected_code


def query_openai_style_tools(
    client,        # openai.OpenAI (real or pointed at localhost:11434/v1)
    model: str,
    csv_text: str,
    question: str,
    sandbox: "PythonSandbox",
    csv_path: Path,
) -> tuple[str, str | None, list[str]]:
    """Tool-calling query via OpenAI-compatible API (works for both OpenAI and Ollama)."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT_TOOLS},
        {"role": "user", "content": f"## Dataset\n{csv_text}\n\n## Question\n{question}"},
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
) -> tuple[str, str | None, list[str]]:
    """Tool-calling query via Anthropic API."""
    use_thinking = model in THINKING_MODELS
    kwargs: dict = dict(
        model=model, max_tokens=16_000, system=SYSTEM_PROMPT_TOOLS,
        tools=[ANTHROPIC_TOOL_RUN_PYTHON],
        tool_choice={"type": "auto"},
    )
    if use_thinking:
        kwargs["thinking"] = {"type": "adaptive"}

    messages: list[dict] = [
        {"role": "user", "content": f"## Dataset\n{csv_text}\n\n## Question\n{question}"},
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
    "ollama": query_ollama,
    "anthropic": query_anthropic,
    "openai": query_openai,
}

QUERY_FN_TOOLS: dict[str, Callable] = {
    "anthropic": query_anthropic_tools,
    "openai": query_openai_style_tools,
    "ollama_tools": query_ollama_tools,
}


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

    # Determine effective provider per model (handles Ollama tool-support check)
    effective_providers: dict[str, str | None] = {}
    for model in models:
        provider = model_providers[model]
        if tools and provider == "ollama":
            if check_ollama_tool_support(model):
                effective_providers[model] = "ollama_tools"
                print(f"[tools] {model}: tool support confirmed")
            else:
                effective_providers[model] = None  # will be skipped
                print(f"[tools] {model}: no tool support — skipping")
        else:
            effective_providers[model] = provider

    # Sandbox (lazy, only when tools=True)
    from sandbox import PythonSandbox
    sandbox = None
    if tools:
        sandbox = PythonSandbox()
        if sandbox.ping():
            print(f"[sandbox] ready ({sandbox._python})")
        else:
            print(f"[sandbox] WARNING: could not launch Python at {sandbox._python}")

    # Lazy client initialization — only for providers actually needed
    from dotenv import load_dotenv
    load_dotenv()

    clients: dict[str, object] = {"ollama": None}

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

    if tools and any(v == "ollama_tools" for v in effective_providers.values()):
        clients["ollama_tools"] = None  # uses requests directly via native /api/chat

    results: list[dict] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        instance_dir = manifest_path.parent
        if columns_only:
            import pandas as pd
            cols = pd.read_csv(instance_dir / "table.csv", nrows=0).columns.tolist()
            csv_text = "Columns: " + ", ".join(cols)
        else:
            csv_text = (instance_dir / "table.csv").read_text()


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
                eff = effective_providers[model]
                if tools and eff is None:
                    continue  # model doesn't support tools, skip

                print(f"[{model}] {dataset}/{injector} → ", end="", flush=True)

                query_fn = QUERY_FN_TOOLS[eff] if tools else QUERY_FN[model_providers[model]]
                client = clients[eff if tools else model_providers[model]]

                thinking: str | None = None
                code_list: list[str] = []
                try:
                    if tools:
                        model_answer, thinking, code_list = query_fn(client, model, csv_text, question, sandbox, instance_dir / "table.csv")
                    else:
                        model_answer, thinking = query_fn(client, model, csv_text, question)
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
                    result["tools_enabled"] = True
                    result["code"] = code_list
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
            "Run QA eval against any combination of Ollama, Anthropic, and OpenAI models.\n"
            "Provider is auto-detected from the model name:\n"
            "  - Ollama:    name contains ':'  (e.g. deepseek-r1:8b)\n"
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
        help="Enable Python code execution tool. Ollama models without tool support are skipped.",
    )
    parser.add_argument(
        "--columns-only",
        action="store_true",
        default=False,
        help="Replace the full CSV with just column names in the prompt (blind test).",
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
        args.output = base

    results = run_eval(
        args.models, args.instances_dir, args.output,
        dataset=args.dataset, injector=args.injector, tools=args.tools,
        columns_only=args.columns_only,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
