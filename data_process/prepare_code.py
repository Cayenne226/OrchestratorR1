"""
Prepare code task data from HumanEval and MBPP.

Converts code benchmark tasks into the unified format:
    {"input": "task description", "answer": "canonical solution", "source": "humaneval",
     "task_id": "HumanEval/0", "test_cases": "...", "difficulty": "code"}

Usage:
    python data_process/prepare_code.py --source humaneval,mbpp --split test --output data/test_code.jsonl
    python data_process/prepare_code.py --source mbpp --split train --output data/train_code.jsonl
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def load_humaneval(max_samples: int, split: str = "test") -> list:
    """Load HumanEval dataset. Only has 'test' split (164 problems)."""
    if split == "train":
        print("  [INFO] HumanEval has no train split, skipping.")
        return []
    try:
        ds = load_dataset("openai_humaneval", split="test")
    except Exception as e:
        print(f"[WARN] Failed to load HumanEval: {e}")
        print("  Trying alternative: openai/humaneval")
        try:
            ds = load_dataset("openai/humaneval", split="test")
        except Exception as e2:
            print(f"[WARN] Also failed: {e2}")
            return []

    records = []
    for ex in ds:
        prompt = ex.get("prompt", "").strip()
        solution = ex.get("canonical_solution", "").strip()
        test = ex.get("test", "").strip()
        task_id = ex.get("task_id", "")

        if not prompt or not solution:
            continue

        records.append({
            "input": prompt,
            "answer": solution,
            "source": "humaneval",
            "task_id": task_id,
            "test_cases": test,
            "difficulty": "code",
        })

        if len(records) >= max_samples:
            break

    return records


def load_mbpp(max_samples: int, split: str = "test") -> list:
    """Load MBPP sanitized dataset.

    Splits: train (374), test (427), validation (90).
    """
    hf_split = split if split in ("train", "test", "validation") else "test"
    try:
        ds = load_dataset("mbpp", "sanitized", split=hf_split)
    except Exception as e:
        print(f"[WARN] Failed to load MBPP sanitized split={hf_split}: {e}")
        try:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized",
                              split=hf_split)
        except Exception as e2:
            print(f"[WARN] Also failed: {e2}")
            return []

    records = []
    for ex in ds:
        prompt = ex.get("prompt", "").strip()
        code = ex.get("code", "").strip()
        tests = ex.get("test_list", [])
        task_id = ex.get("task_id", "")

        if not prompt or not code:
            continue

        records.append({
            "input": prompt,
            "answer": code,
            "source": "mbpp",
            "task_id": str(task_id),
            "test_cases": "\n".join(tests) if tests else "",
            "difficulty": "code",
        })

        if len(records) >= max_samples:
            break

    return records


LOADERS = {
    "humaneval": load_humaneval,
    "mbpp": load_mbpp,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="humaneval",
                        help="Comma-separated: humaneval,mbpp")
    parser.add_argument("--max_per_source", type=int, default=500)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    sources = [s.strip() for s in args.source.split(",")]
    all_records = []

    for source in sources:
        if source not in LOADERS:
            print(f"[WARN] Unknown source: {source}, skipping")
            continue
        print(f"Loading {source}...")
        records = LOADERS[source](args.max_per_source)
        print(f"  Loaded {len(records)} samples from {source}")
        all_records.extend(records)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_records)} samples → {args.output}")
    source_counts = {}
    for r in all_records:
        source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()
