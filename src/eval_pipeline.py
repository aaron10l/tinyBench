"""Evaluate QA instances against local Ollama models."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import requests

from io_utils import load_json

DEFAULT_MODELS = ["deepseek-r1:8b", "qwen2.5:7b", "llama3.2:3b", "phi4:latest"]
DEFAULT_INSTANCES_DIR = Path("data/instances")
DEFAULT_OUTPUT = Path("data/results/eval_results.json")
OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = (
    "You are a data analyst. You are given a CSV dataset and a question about it.\n"
    "Answer the question concisely. Do not explain your reasoning."
)


def build_prompt(csv_text: str, question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n## Dataset\n{csv_text}\n\n## Question\n{question}"


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def query_ollama(model: str, prompt: str) -> tuple[str, str | None]:
    """Return (answer, thinking) — thinking is None when the model doesn't emit <think> tags."""
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=1800,
    )
    resp.raise_for_status()
    raw = resp.json()["response"]

    thinking = None
    think_match = _THINK_RE.search(raw)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = _THINK_RE.sub("", raw).strip()
    else:
        answer = raw.strip()

    return answer, thinking


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
                    model_answer, thinking = query_ollama(
                        model, build_prompt(csv_text, question)
                    )
                except requests.RequestException as exc:
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

    # model × template_id counts
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        counts[r["model"]][r["template_id"]] += 1

    templates = sorted({r["template_id"] for r in results})
    models = sorted(counts.keys())

    # Header
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
    parser = argparse.ArgumentParser(description="Run QA eval against Ollama models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Ollama model tags to evaluate",
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
