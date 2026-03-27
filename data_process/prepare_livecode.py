"""
Prepare LiveCodeBench dataset for OrchestratorR1 evaluation.

Uses cassanof/livecodebench_lite_filtered (202 samples, curated subset).
Competition-level coding problems from LeetCode, AtCoder, Codeforces.

Usage:
    python data_process/prepare_livecode.py --output data/test_livecode.jsonl
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def format_test_cases(input_output) -> str:
    """Convert input_output field to readable test case format."""
    if not input_output:
        return ""
    if isinstance(input_output, str):
        try:
            input_output = json.loads(input_output)
        except json.JSONDecodeError:
            return input_output

    if isinstance(input_output, dict):
        inputs = input_output.get("inputs", [])
        outputs = input_output.get("outputs", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
        parts = []
        for inp, out in zip(inputs[:5], outputs[:5]):  # Limit to 5 cases
            inp_str = inp.strip() if isinstance(inp, str) else str(inp)
            out_str = out.strip() if isinstance(out, str) else str(out)
            parts.append(f"Input:\n{inp_str}\nExpected Output:\n{out_str}")
        return "\n---\n".join(parts)

    return str(input_output)


def main():
    parser = argparse.ArgumentParser(description="Prepare LiveCodeBench test set")
    parser.add_argument("--output", type=str, default="data/test_livecode.jsonl")
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    print("Loading cassanof/livecodebench_lite_filtered (test split)...")
    ds = load_dataset("cassanof/livecodebench_lite_filtered", split="test",
                      streaming=True)

    records = []
    skipped_no_tests = 0

    for ex in ds:
        title = (ex.get("title") or "").strip()
        content = (ex.get("question") or "").strip()
        difficulty = (ex.get("difficulty") or "unknown").strip().lower()
        source = ex.get("source", "unknown")
        question_id = ex.get("id", "")
        starter_code = ex.get("starter_code") or ""
        input_output = ex.get("input_output", "")

        if not content:
            continue

        test_str = format_test_cases(input_output)
        if not test_str:
            skipped_no_tests += 1
            continue

        input_text = f"# {title}\n\n{content}"
        if starter_code.strip():
            input_text += f"\n\n## Starter Code\n```python\n{starter_code.strip()}\n```"

        records.append({
            "input": input_text,
            "answer": "",
            "source": "livecodebench",
            "difficulty": f"code_{difficulty}",
            "task_id": str(question_id),
            "test_cases": test_str,
            "platform": source,
        })

        if len(records) >= args.max_samples:
            break

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(records)} samples → {args.output}")
    if skipped_no_tests:
        print(f"Skipped (no tests): {skipped_no_tests}")

    diff_counts = {}
    plat_counts = {}
    for r in records:
        diff_counts[r["difficulty"]] = diff_counts.get(r["difficulty"], 0) + 1
        plat_counts[r["platform"]] = plat_counts.get(r["platform"], 0) + 1
    print("Difficulty distribution:")
    for d, c in sorted(diff_counts.items()):
        print(f"  {d}: {c}")
    print("Platform distribution:")
    for p, c in sorted(plat_counts.items()):
        print(f"  {p}: {c}")


if __name__ == "__main__":
    main()
