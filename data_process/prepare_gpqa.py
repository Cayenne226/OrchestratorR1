"""
Prepare GPQA Diamond dataset for OrchestratorR1 evaluation.

Uses hendrydong/gpqa_diamond_mc (public mirror, 198 samples with MCQ format).
Each problem contains the question text with choices (A)-(D) embedded.

Usage:
    python data_process/prepare_gpqa.py --output data/test_gpqa.jsonl
"""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


def extract_choices(problem_text: str) -> list[str]:
    """Extract choices (A)-(D) from the problem text."""
    # Match patterns like (A) ..., (B) ..., etc.
    pattern = r'\(([A-D])\)\s*(.+?)(?=\([A-D]\)|$)'
    matches = re.findall(pattern, problem_text, re.DOTALL)
    if matches:
        return [f"({letter}) {text.strip()}" for letter, text in matches]
    return []


def extract_answer_letter(solution: str) -> str:
    """Extract answer letter from solution like '\\boxed{D}' or just 'D'."""
    # Match \boxed{X}
    m = re.search(r'\\boxed\{([A-D])\}', solution)
    if m:
        return m.group(1)
    # Match standalone letter
    m = re.search(r'^([A-D])$', solution.strip())
    if m:
        return m.group(1)
    return solution.strip()


def extract_domain(domain: str) -> str:
    """Normalize domain name."""
    domain = domain.strip().lower()
    if "physic" in domain:
        return "physics"
    if "chem" in domain:
        return "chemistry"
    if "bio" in domain:
        return "biology"
    return domain


def main():
    parser = argparse.ArgumentParser(description="Prepare GPQA Diamond test set")
    parser.add_argument("--output", type=str, default="data/test_gpqa.jsonl")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Max samples (default 500, Diamond has 198)")
    args = parser.parse_args()

    print("Loading hendrydong/gpqa_diamond_mc (test split)...")
    ds = load_dataset("hendrydong/gpqa_diamond_mc", split="test", streaming=True)

    records = []
    for ex in ds:
        problem = ex.get("problem", "").strip()
        solution = ex.get("solution", "").strip()
        domain = ex.get("domain", "").strip()

        if not problem or not solution:
            continue

        answer_letter = extract_answer_letter(solution)
        choices = extract_choices(problem)

        records.append({
            "input": problem,
            "answer": answer_letter,
            "source": "gpqa_diamond",
            "difficulty": "expert_reasoning",
            "domain": extract_domain(domain),
            "choices": choices,
        })

        if len(records) >= args.max_samples:
            break

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(records)} samples → {args.output}")
    domain_counts = {}
    for r in records:
        domain_counts[r["domain"]] = domain_counts.get(r["domain"], 0) + 1
    for d, c in sorted(domain_counts.items()):
        print(f"  {d}: {c}")


if __name__ == "__main__":
    main()
