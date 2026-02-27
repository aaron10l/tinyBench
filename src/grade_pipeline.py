"""LLM grader pipeline — grades eval results using OpenAI API.

Reads an eval results JSON, grades each entry as CORRECT / PARTIAL / INCORRECT,
and writes the same entries with added "grade" and "reasoning" fields.

Usage:
    python src/grade_pipeline.py --input data/results/eval_results_cluster.json
    python src/grade_pipeline.py --input data/results/eval_results_cluster.json --output data/results/graded.json --model gpt-4o
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


GRADE_SYSTEM_PROMPT = """\
You are grading an LLM's answer to a data analysis question about a CSV dataset.

You will be given:
- The question asked
- The expected correct answer
- The model's answer

Grade the response as exactly one of:
- CORRECT: The answer matches the expected answer (allow minor formatting differences, \
extra explanation is fine as long as the core answer is right)
- PARTIAL: The answer is partially correct (e.g. some items in a list are right, \
or the direction is right but the specific value is wrong)
- INCORRECT: The answer is wrong, irrelevant, or doesn't address the question

Return JSON with "grade" and "reasoning" fields.
"""


def grade_entry(client, model: str, entry: dict) -> tuple[str, str]:
    """Return (grade, reasoning) for a single result entry."""
    model_answer = entry.get("model_answer", "")

    # Auto-grade timeouts and errors without calling the API
    if isinstance(model_answer, str) and model_answer.startswith("ERROR:"):
        return "INCORRECT", "Request errored or timed out — no answer produced."

    expected = entry["expected_answer"]
    user_content = (
        f"Question: {entry['question']}\n\n"
        f"Expected answer: {json.dumps(expected)}\n\n"
        f"Model answer: {model_answer}"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GRADE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "grade",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "grade": {
                            "type": "string",
                            "enum": ["CORRECT", "PARTIAL", "INCORRECT"],
                        },
                        "reasoning": {"type": "string"},
                    },
                    "required": ["grade", "reasoning"],
                    "additionalProperties": False,
                },
            },
        },
    )
    result = json.loads(resp.choices[0].message.content)
    return result["grade"], result["reasoning"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade eval results with an LLM")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to eval results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write graded results (default: input file with _graded suffix)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for grading (default: gpt-4o-mini)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_graded{input_path.suffix}"

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    import openai
    client = openai.OpenAI(api_key=api_key)

    entries = json.loads(input_path.read_text())
    print(f"Grading {len(entries)} entries with {args.model}...")

    graded = []
    for i, entry in enumerate(entries):
        grade, reasoning = grade_entry(client, args.model, entry)
        graded_entry = {**entry, "grade": grade, "reasoning": reasoning}
        graded.append(graded_entry)
        print(f"  [{i + 1}/{len(entries)}] {entry['model']} / {entry['dataset']} / {entry['injector']} → {grade}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graded, indent=2) + "\n")
    print(f"\nWrote {len(graded)} graded entries to {output_path}")

    # Print summary
    from collections import Counter
    by_model: dict[str, Counter] = {}
    for e in graded:
        m = e["model"]
        if m not in by_model:
            by_model[m] = Counter()
        by_model[m][e["grade"]] += 1

    print("\nSummary:")
    for m, counts in sorted(by_model.items()):
        total = sum(counts.values())
        print(f"  {m}: {counts['CORRECT']}C / {counts['PARTIAL']}P / {counts['INCORRECT']}I  (n={total})")


if __name__ == "__main__":
    main()
