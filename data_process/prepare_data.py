"""
Prepare training data for Orchestrator-R1.

Merges QA datasets into a unified JSONL format:
    {"input": "question text", "answer": "gold answer or list", "source": "nq"}

Router-R1 aligned setup (NQ 7k + HotpotQA 7k):
    python data_process/prepare_data.py --preset router_r1_train --output data/train.jsonl
    python data_process/prepare_data.py --preset router_r1_test  --output data/test.jsonl

Custom:
    python data_process/prepare_data.py --sources nq,hotpotqa --split train --max_per_source 7000 --output data/train.jsonl
    python data_process/prepare_data.py --sources nq,hotpotqa --split test  --max_per_source 500  --output data/test.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


DIFFICULTY_MAP = {
    "nq": "simple",
    "triviaqa": "simple",
    "popqa": "simple",
    "hotpotqa": "multi_hop",
    "2wikimultihop": "multi_hop",
    "musique": "multi_hop",
    "bamboogle": "multi_hop",
}

DATASET_CONFIGS = {
    # ── Track 1: Simple QA ─────────────────────────────────────────────────────
    "nq": {
        # nq_open: clean Q/A pairs from Natural Questions (no HTML documents)
        # train: 87,925 rows, validation: 3,610 rows
        "hf_name": "google-research-datasets/nq_open",
        "hf_config": None,
        "question_key": "question",
        "answer_fn": lambda ex: ex["answer"],  # list[str]
        "train_split": "train",
        "test_split": "validation",
    },
    "triviaqa": {
        "hf_name": "trivia_qa",
        "hf_config": "rc.nocontext",
        "question_key": "question",
        "answer_fn": lambda ex: ex["answer"]["aliases"],
        "train_split": "train",
        "test_split": "validation",
    },
    "popqa": {
        "hf_name": "akariasai/PopQA",
        "hf_config": None,
        "question_key": "question",
        "answer_fn": lambda ex: [a for a in ex["possible_answers"].split("; ") if a.strip()] if isinstance(ex["possible_answers"], str) else ex["possible_answers"],
        "train_split": "train",
        "test_split": "test",
    },
    # ── Track 2: Multi-hop Reasoning ───────────────────────────────────────────
    "hotpotqa": {
        # HotpotQA distractor setting — multi-hop QA
        # train: 90,447 rows, validation: 7,405 rows
        "hf_name": "hotpot_qa",
        "hf_config": "distractor",
        "question_key": "question",
        "answer_fn": lambda ex: ex["answer"],  # str
        "train_split": "train",
        "test_split": "validation",
    },
    "2wikimultihop": {
        "hf_name": "xanhho/2WikiMultihopQA",
        "hf_config": None,
        "question_key": "question",
        "answer_fn": lambda ex: ex["answer"],
        "train_split": "train",
        "test_split": "validation",
    },
    "musique": {
        "hf_name": "drt/musique",
        "hf_config": None,
        "question_key": "question",
        "answer_fn": lambda ex: ex["answer"],
        "train_split": "train",
        "test_split": "validation",
    },
    "bamboogle": {
        "hf_name": "ChaiML/bamboogle",
        "hf_config": None,
        "question_key": "Question",
        "answer_fn": lambda ex: ex["Answer"],
        "train_split": "train",
        "test_split": "train",       # bamboogle has only one split
    },
}

# ── Presets for reproducible experiment setups ────────────────────────────────
PRESETS = {
    "router_r1_train": {
        "sources": ["nq", "hotpotqa"],
        "split": "train",
        "max_per_source": 7000,       # 7k NQ + 7k HotpotQA = 14k total
    },
    "router_r1_test": {
        "sources": ["nq", "hotpotqa"],
        "split": "test",
        "max_per_source": 500,        # 500 NQ + 500 HotpotQA = 1k test
    },
    # ── OrchestratorR1: 6 datasets × 3 tracks ──────────────────────────────
    "orch_r1_train": {
        "sources": ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihop", "musique"],
        "split": "train",
        "max_per_source": 1000,       # 6 × 1k = 6k total
    },
    "orch_r1_test": {
        "sources": ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihop", "musique"],
        "split": "test",
        "max_per_source": 500,        # 6 × 500 = 3k total
    },
}


def load_source(name: str, split: str, max_samples: int) -> list[dict]:
    cfg = DATASET_CONFIGS[name]
    # Use dataset-specific split name (some datasets use "test" instead of "validation")
    hf_split = cfg["train_split"] if split == "train" else cfg["test_split"]

    print(f"  Loading {cfg['hf_name']} (config={cfg['hf_config']}, split={hf_split})...")
    try:
        ds = load_dataset(
            cfg["hf_name"],
            cfg["hf_config"],
            split=hf_split,
            streaming=True,
        )
    except Exception as e:
        print(f"  [WARN] Failed to load {name}: {e}")
        return []

    records = []
    for ex in ds:
        question = ex.get(cfg["question_key"], "")
        if isinstance(question, dict):
            question = question.get("text", "")
        question = str(question).strip()
        if not question:
            continue
        try:
            answer = cfg["answer_fn"](ex)
        except Exception:
            continue
        if not answer:
            continue
        # Normalize: answer can be str or list[str]
        if isinstance(answer, str):
            answer = answer.strip()
        elif isinstance(answer, list):
            answer = [a.strip() for a in answer if isinstance(a, str) and a.strip()]
            if not answer:
                continue
            # Single-element list → flatten to str
            if len(answer) == 1:
                answer = answer[0]
        difficulty = DIFFICULTY_MAP.get(name, "simple")
        records.append({"input": question, "answer": answer, "source": name, "difficulty": difficulty})
        if len(records) >= max_samples:
            break

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Prepare QA training/test data. Use --preset for quick setup."
    )
    parser.add_argument("--preset",         type=str, default=None,
                        choices=list(PRESETS.keys()),
                        help="Use a predefined config (overrides --sources/--split/--max_per_source)")
    parser.add_argument("--sources",        type=str, default="nq,hotpotqa",
                        help="Comma-separated dataset names")
    parser.add_argument("--split",          type=str, default="train",
                        choices=["train", "test"])
    parser.add_argument("--max_per_source", type=int, default=7000)
    parser.add_argument("--output",         type=str, required=True)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    # Apply preset if specified
    if args.preset:
        preset = PRESETS[args.preset]
        sources = preset["sources"]
        split = preset["split"]
        max_per_source = preset["max_per_source"]
        print(f"Using preset: {args.preset}")
        print(f"  sources={sources}, split={split}, max_per_source={max_per_source}")
    else:
        sources = [s.strip() for s in args.sources.split(",")]
        split = args.split
        max_per_source = args.max_per_source

    random.seed(args.seed)

    all_records = []
    for source in sources:
        if source not in DATASET_CONFIGS:
            print(f"[WARN] Unknown source: {source}, skipping")
            continue
        print(f"Loading {source}...")
        records = load_source(source, split, max_per_source)
        print(f"  → {len(records)} samples from {source}")
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
