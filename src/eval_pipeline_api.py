"""Evaluate QA instances against frontier models (Anthropic & OpenAI APIs)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import anthropic
import openai
from dotenv import load_dotenv

from io_utils import load_json

load_dotenv()

ANTHROPIC_MODELS = ["claude-sonnet-4-5-20250929", "claude-opus-4-6"]
THINKING_MODELS = {"claude-opus-4-6"}
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini"]
DEFAULT_MODELS = ANTHROPIC_MODELS + OPENAI_MODELS
DEFAULT_INSTANCES_DIR = Path("data/instances")
DEFAULT_OUTPUT = Path("data/results/eval_results.json")

SYSTEM_PROMPT = (
    "You are a data analyst. You are given a CSV dataset and a question about it.\n"
    "Answer the question concisely. Do not explain your reasoning."
)


def build_user_message(csv_text: str, question: str) -> str:
    return f"## Dataset\n{csv_text}\n\n## Question\n{question}"


def query_anthropic(client: anthropic.Anthropic, model: str, csv_text: str, question: str) -> tuple[str, str | None]:
    kwargs: dict = dict(
        model=model,
        max_tokens=16_000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(csv_text, question)}],
    )
    if model in THINKING_MODELS:
        kwargs["thinking"] = {"type": "adaptive"}

    msg = client.messages.create(**kwargs)

    thinking = None
    answer = ""
    for block in msg.content:
        if block.type == "thinking":
            thinking = block.thinking
        elif block.type == "text":
            answer = block.text.strip()

    return answer, thinking


def query_openai(client: openai.OpenAI, model: str, csv_text: str, question: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(csv_text, question)},
        ],
    )
    return resp.choices[0].message.content.strip()


def discover_instances(instances_dir: Path) -> list[Path]:
    """Return manifests sorted by CSV size (smallest datasets first)."""
    manifests = list(instances_dir.glob("**/manifest.json"))
    return sorted(manifests, key=lambda m: (m.parent / "table.csv").stat().st_size)


def run_eval(
    models: list[str],
    instances_dir: Path,
    output_path: Path,
) -> list[dict]:
    manifests = discover_instances(instances_dir)
    if not manifests:
        print(f"No manifests found under {instances_dir}")
        sys.exit(1)

    # Init clients lazily based on which models are requested
    anthropic_client = None
    openai_client = None

    need_anthropic = any(m in ANTHROPIC_MODELS for m in models)
    need_openai = any(m in OPENAI_MODELS for m in models)

    if need_anthropic:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set in .env")
            sys.exit(1)
        anthropic_client = anthropic.Anthropic(api_key=api_key)

    if need_openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set in .env")
            sys.exit(1)
        openai_client = openai.OpenAI(api_key=api_key)

    results: list[dict] = []

    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        instance_dir = manifest_path.parent
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
                print(
                    f"[{model}] {dataset}/{injector} → ",
                    end="",
                    flush=True,
                )
                thinking = None
                try:
                    if model in ANTHROPIC_MODELS:
                        model_answer, thinking = query_anthropic(
                            anthropic_client, model, csv_text, question
                        )
                    elif model in OPENAI_MODELS:
                        model_answer = query_openai(
                            openai_client, model, csv_text, question
                        )
                    else:
                        model_answer = f"ERROR: unknown model {model}"
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
                if thinking:
                    result["thinking"] = thinking
                results.append(result)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")

    return results


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run QA eval against frontier models (Anthropic & OpenAI)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to evaluate",
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
    args = parser.parse_args()

    if args.output is None:
        if len(args.models) == 1:
            safe = args.models[0].replace(":", "_").replace("/", "_")
            args.output = Path(f"data/results/eval_results_{safe}.json")
        else:
            args.output = DEFAULT_OUTPUT

    results = run_eval(args.models, args.instances_dir, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()
