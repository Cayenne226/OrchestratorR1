"""
Strip <think>...</think> blocks from SFT data for the w/o-think ablation.

Usage:
    python data_process/strip_think.py \
        --input data/sft_warmup.jsonl \
        --output data/sft_warmup_nothink.jsonl
"""

import argparse
import json
import re

THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove all <think>...</think> blocks from text."""
    return THINK_PATTERN.sub("", text).lstrip()


def main():
    parser = argparse.ArgumentParser(description="Remove <think> blocks from SFT JSONL data")
    parser.add_argument("--input",  type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    args = parser.parse_args()

    kept = skipped = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            original = record["output"]
            stripped = strip_think(original)

            # Validate: must still contain <call> or <answer>
            if "<call" not in stripped and "<answer>" not in stripped:
                skipped += 1
                continue

            record["output"] = stripped
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Done: {kept} kept, {skipped} skipped (no valid tags after stripping)")


if __name__ == "__main__":
    main()
