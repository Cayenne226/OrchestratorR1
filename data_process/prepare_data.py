"""
Prepare training data for Orchestrator-R1.

Merges QA datasets (same sources as Router-R1) into a unified JSONL format:
    {"input": "question text", "answer": "gold answer or list"}

Usage:
    python data_process/prepare_data.py \
        --sources nq,hotpotqa,triviaqa \
        --split train \
        --max_per_source 1000 \
        --output data/train.jsonl

    python data_process/prepare_data.py \
        --sources nq,hotpotqa,triviaqa \
        --split test \
        --max_per_source 200 \
        --output data/test.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


DATASET_CONFIGS = {
    "nq": {
        "hf_name": "natural_questions",
        "hf_config": "default",
        "question_key": "question",
        "answer_fn": lambda ex: [a["text"] for a in ex["annotations"][0]["short_answers"]]
                                if ex["annotations"][0]["short_answers"] else None,
    },
    "triviaqa": {
        "hf_name": "trivia_qa",
        "hf_config": "rc.nocontext",
        "question_key": "question",
        "answer_fn": lambda ex: ex["answer"]["aliases"],
    },
    "hotpotqa": {
        "hf_name": "hotpot_qa",
        "hf_config": "fullwiki",
        "question_key": "question",
        "answer_fn": lambda ex: ex["answer"],
    },
    "popqa": {
        "hf_name": "akariasai/PopQA",
        "hf_config": None,
        "question_key": "question",
        "answer_fn": lambda ex: ex["possible_answers"],
    },
    "2wikimultihop": {
        "hf_name": "xanhho/2WikiMultihopQA",
        "hf_config": None,
        "question_key": "question",
        "answer_fn": lambda ex: ex["answer"],
    },
    "musique": {
        "hf_name": "musique",
        "hf_config": None,
        "question_key": "question",
        "answer_fn": lambda ex: ex["answer"],
    },
    "bamboogle": {
        "hf_name": "ChaiML/bamboogle",
        "hf_config": None,
        "question_key": "Question",
        "answer_fn": lambda ex: ex["Answer"],
    },
}


def load_source(name: str, split: str, max_samples: int) -> list[dict]:
    cfg = DATASET_CONFIGS[name]
    hf_split = "train" if split == "train" else "validation"

    try:
        ds = load_dataset(
            cfg["hf_name"],
            cfg["hf_config"],
            split=hf_split,
            streaming=True,
        )
    except Exception as e:
        print(f"[WARN] Failed to load {name}: {e}")
        return []

    records = []
    for ex in ds:
        question = ex.get(cfg["question_key"], "").strip()
        if not question:
            continue
        try:
            answer = cfg["answer_fn"](ex)
        except Exception:
            continue
        if not answer:
            continue
        records.append({"input": question, "answer": answer, "source": name})
        if len(records) >= max_samples:
            break

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources",        type=str, default="nq,hotpotqa,triviaqa",
                        help="Comma-separated dataset names")
    parser.add_argument("--split",          type=str, default="train",
                        choices=["train", "test"])
    parser.add_argument("--max_per_source", type=int, default=1000)
    parser.add_argument("--output",         type=str, required=True)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    sources = [s.strip() for s in args.sources.split(",")]

    all_records = []
    for source in sources:
        if source not in DATASET_CONFIGS:
            print(f"[WARN] Unknown source: {source}, skipping")
            continue
        print(f"Loading {source}...")
        records = load_source(source, args.split, args.max_per_source)
        print(f"  Loaded {len(records)} samples from {source}")
        all_records.extend(records)

    random.shuffle(all_records)

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
