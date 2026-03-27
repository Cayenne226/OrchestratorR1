"""
Direct-GPT-4o Baseline: Send each test question directly to GPT-4o, no orchestration.

This establishes the "strong model upper bound" — what you get by just asking the best
model directly, without any multi-agent orchestration.

Usage:
    python eval/run_direct_gpt4o.py \
        --data_paths data/test_qa.jsonl data/test_code.jsonl data/test_gpqa.jsonl data/test_livecode.jsonl \
        --api_base "http://35.220.164.252:3888/v1/" \
        --api_key "YOUR_KEY" \
        --output eval/results/direct_gpt4o.json \
        --model gpt-4o \
        --max_workers 8

    # Run on a single test set:
    python eval/run_direct_gpt4o.py \
        --data_paths data/test_qa.jsonl \
        --output eval/results/direct_gpt4o_qa.json
"""

import argparse
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI


# ── Task-specific prompts ──────────────────────────────────────────────────────

QA_SYSTEM = (
    "You are a helpful assistant. Answer the question with ONLY the answer itself — "
    "a short phrase or entity name. No explanation, no full sentences, no hedging. "
    "Examples: 'Paris', 'Albert Einstein', '1969', 'hydrogen bonding'."
)

GPQA_SYSTEM = (
    "You are an expert scientist. For the following multiple-choice question, "
    "reason step by step, then give your final answer as \\boxed{X} where X is A, B, C, or D."
)

CODE_HUMANEVAL_SYSTEM = (
    "You are an expert Python programmer. Complete the given function. "
    "Output ONLY the function body (the code that goes inside the function), nothing else. "
    "Do not include the function signature or docstring. Do not use markdown formatting."
)

CODE_MBPP_SYSTEM = (
    "You are an expert Python programmer. Write the complete Python function. "
    "Output ONLY the function code starting with 'def ...'. "
    "You MUST use the exact function name specified in the prompt. "
    "Do not use markdown formatting or add any explanation."
)

LIVECODE_SYSTEM = (
    "You are an expert competitive programmer. Write a complete Python solution "
    "that reads from stdin and writes to stdout. Output ONLY the Python code, "
    "no explanation or markdown formatting."
)


def get_prompt_for_record(record: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) based on record type."""
    source = record.get("source", "")
    difficulty = record.get("difficulty", "")

    if source == "gpqa_diamond" or difficulty == "expert_reasoning":
        return GPQA_SYSTEM, record["input"]

    if source in ("humaneval", "mbpp") or difficulty == "code":
        if source == "mbpp" or (source != "humaneval" and "def " not in record["input"]):
            # For MBPP: extract expected function name and signature from test cases
            test_cases = record.get("test_cases", "")
            func_name_match = re.search(r'assert\s+(\w+)\s*\(', test_cases)
            func_hint = ""
            if func_name_match:
                func_name = func_name_match.group(1)
                # Extract full call signature for better hint
                sig_match = re.search(r'assert\s+' + func_name + r'\s*\(([^)]*)\)', test_cases)
                if sig_match:
                    n_args = len([a.strip() for a in sig_match.group(1).split(',') if a.strip()])
                    func_hint = (f"\n\nIMPORTANT: The function MUST be named '{func_name}' "
                                 f"and accept {n_args} argument(s). "
                                 f"Example call: {func_name}({sig_match.group(1).strip()})")
                else:
                    func_hint = f"\n\nIMPORTANT: The function MUST be named '{func_name}'."
            return CODE_MBPP_SYSTEM, record["input"] + func_hint
        return CODE_HUMANEVAL_SYSTEM, record["input"]

    if source == "livecodebench" or difficulty.startswith("code_"):
        return LIVECODE_SYSTEM, record["input"]

    # Default: QA
    return QA_SYSTEM, record["input"]


# ── API call with retry ────────────────────────────────────────────────────────

def call_gpt4o(client: OpenAI, model: str, system: str, user: str,
               max_tokens: int = 2048, temperature: float = 0.0,
               max_retries: int = 3) -> tuple[str, float]:
    """Call GPT-4o API. Returns (response_text, cost_usd)."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,
            )
            text = resp.choices[0].message.content or ""
            total_tokens = resp.usage.total_tokens if resp.usage else 0
            # GPT-4o pricing: $2.50 / 1M input, $10.00 / 1M output
            # Approximate with blended rate
            cost = total_tokens * 5.0 / 1_000_000
            return text, cost
        except Exception as e:
            if attempt == max_retries - 1:
                return f"[API error: {str(e)}]", 0.0
            time.sleep(2 ** attempt)
    return "[API error: max retries exceeded]", 0.0


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Direct-GPT-4o baseline evaluation")
    parser.add_argument("--data_paths", nargs="+", required=True,
                        help="Paths to test JSONL files")
    parser.add_argument("--api_base", type=str,
                        default="http://35.220.164.252:3888/v1/")
    parser.add_argument("--api_key", type=str,
                        default="sk-YlG8W7NPhqBSb3WIgsDJl7xekcBoUuAI8YE1kNtF3UY48ITM")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per file (for quick testing)")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Parallel API call threads")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    # Load all test data
    all_records = []
    for path in args.data_paths:
        with open(path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        if args.max_samples:
            records = records[:args.max_samples]
        print(f"Loaded {len(records)} from {path}")
        all_records.extend(records)

    print(f"\nTotal: {len(all_records)} samples")
    print(f"Model: {args.model}")
    print(f"Workers: {args.max_workers}")

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    # Import metrics (lazy to avoid import issues when used standalone)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from eval.metrics import compute_metric

    # Run evaluation in parallel
    results = [None] * len(all_records)
    completed = 0
    lock = Lock()
    start_time = time.time()

    def eval_one(idx: int, record: dict) -> dict:
        system, user = get_prompt_for_record(record)
        pred, cost = call_gpt4o(client, args.model, system, user,
                                temperature=args.temperature)

        metrics = compute_metric(pred, record)

        return {
            "idx": idx,
            "input": record["input"][:200],  # truncate for readability
            "gold": record.get("answer", ""),
            "pred": pred,
            "source": record.get("source", ""),
            "difficulty": record.get("difficulty", ""),
            "cost": cost,
            "metrics": metrics,
        }

    print(f"\nStarting evaluation...")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(eval_one, i, rec): i
            for i, rec in enumerate(all_records)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                results[idx] = {
                    "idx": idx,
                    "input": all_records[idx]["input"][:200],
                    "gold": all_records[idx].get("answer", ""),
                    "pred": f"[Error: {e}]",
                    "source": all_records[idx].get("source", ""),
                    "difficulty": all_records[idx].get("difficulty", ""),
                    "cost": 0.0,
                    "metrics": {"em": 0.0, "f1": 0.0},
                }
            with lock:
                completed += 1
                if completed % 50 == 0 or completed == len(all_records):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"  [{completed}/{len(all_records)}] "
                          f"{rate:.1f} samples/sec, "
                          f"{elapsed:.0f}s elapsed")

    # Compute summary statistics
    elapsed = time.time() - start_time

    # Per-source breakdown
    source_stats = {}
    for r in results:
        src = r["source"]
        if src not in source_stats:
            source_stats[src] = {"em_sum": 0, "f1_sum": 0, "cost_sum": 0, "n": 0,
                                 "extra": {}}
        s = source_stats[src]
        s["n"] += 1
        s["em_sum"] += r["metrics"].get("em", 0)
        s["f1_sum"] += r["metrics"].get("f1", 0)
        s["cost_sum"] += r["cost"]
        # Track extra metrics
        for k, v in r["metrics"].items():
            if k not in ("em", "f1"):
                s["extra"].setdefault(k, 0)
                s["extra"][k] += v

    # Print results
    print(f"\n{'=' * 70}")
    print(f"Direct-GPT-4o Baseline Results ({args.model})")
    print(f"{'=' * 70}")
    print(f"{'Source':<20} {'N':>5} {'EM':>8} {'F1':>8} {'Cost':>10} {'Extra':>15}")
    print(f"{'-' * 70}")

    total_em = total_f1 = total_cost = total_n = 0
    per_source_summary = {}

    for src in sorted(source_stats.keys()):
        s = source_stats[src]
        n = s["n"]
        em = s["em_sum"] / n if n > 0 else 0
        f1 = s["f1_sum"] / n if n > 0 else 0
        cost = s["cost_sum"] / n if n > 0 else 0

        extra_str = ""
        extra_dict = {}
        for k, v in s["extra"].items():
            val = v / n if n > 0 else 0
            extra_str += f" {k}={val:.3f}"
            extra_dict[k] = round(val, 4)

        print(f"  {src:<18} {n:>5} {em:>8.4f} {f1:>8.4f} ${cost:>9.6f}{extra_str}")

        per_source_summary[src] = {
            "n": n, "em": round(em, 4), "f1": round(f1, 4),
            "avg_cost": round(cost, 6), **extra_dict,
        }

        total_em += s["em_sum"]
        total_f1 += s["f1_sum"]
        total_cost += s["cost_sum"]
        total_n += n

    print(f"{'-' * 70}")
    overall_em = total_em / total_n if total_n > 0 else 0
    overall_f1 = total_f1 / total_n if total_n > 0 else 0
    overall_cost = total_cost / total_n if total_n > 0 else 0
    print(f"  {'OVERALL':<18} {total_n:>5} {overall_em:>8.4f} {overall_f1:>8.4f} ${overall_cost:>9.6f}")
    print(f"\nTotal API cost: ${total_cost:.4f}")
    print(f"Time: {elapsed:.1f}s ({total_n / elapsed:.1f} samples/sec)")

    # Save results
    output_data = {
        "config": {
            "model": args.model,
            "temperature": args.temperature,
            "n_samples": total_n,
            "data_paths": args.data_paths,
        },
        "summary": {
            "n_samples": total_n,
            "em": round(overall_em, 4),
            "f1": round(overall_f1, 4),
            "avg_cost_usd": round(overall_cost, 6),
            "total_cost_usd": round(total_cost, 4),
            "avg_turns": 1.0,
            "elapsed_seconds": round(elapsed, 1),
        },
        "per_source": per_source_summary,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
