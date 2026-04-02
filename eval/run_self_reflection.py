"""
Self-Reflection Baseline: Same LLM does 5 rounds of self-reflection before final answer.

This baseline tests whether iterative self-refinement with a single model can match
multi-agent orchestration. The model generates an initial answer, then critiques and
refines it for N rounds.

Usage:
    python eval/run_self_reflection.py \
        --data_paths data/test_qa.jsonl data/test_code.jsonl \
            data/test_gpqa.jsonl data/test_livecode.jsonl \
        --output eval/results/self_reflection.json \
        --model gpt-4o \
        --n_rounds 5

    # Single track:
    python eval/run_self_reflection.py \
        --data_paths data/test_qa.jsonl \
        --output eval/results/self_reflection_qa.json
"""

import argparse
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI


# ── Task-specific initial prompts ─────────────────────────────────────────────

QA_INITIAL = (
    "Answer the question with ONLY the answer itself — "
    "a short phrase or entity name. No explanation, no full sentences."
)

GPQA_INITIAL = (
    "For the following multiple-choice question, reason step by step, "
    "then give your final answer as \\boxed{X} where X is A, B, C, or D."
)

CODE_HUMANEVAL_INITIAL = (
    "Complete the given function. Output ONLY the function body "
    "(the code that goes inside the function), nothing else. "
    "Do not include the function signature or docstring. No markdown."
)

CODE_MBPP_INITIAL = (
    "Write the complete Python function. Output ONLY the function code "
    "starting with 'def ...'. No markdown or explanation."
)

LIVECODE_INITIAL = (
    "Write a complete Python solution that reads from stdin and writes to stdout. "
    "Output ONLY the Python code, no explanation or markdown."
)

REFLECTION_PROMPT = (
    "Review your previous answer carefully. Consider:\n"
    "1. Is the answer correct? Are there any logical errors?\n"
    "2. Did you miss any important details from the question?\n"
    "3. Can you improve the answer?\n\n"
    "If your answer was correct, restate it. If not, provide the corrected answer.\n"
    "Output ONLY the final answer in the same format as before."
)


def get_initial_prompt(record: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) based on record type."""
    source = record.get("source", "")
    difficulty = record.get("difficulty", "")

    if source == "gpqa_diamond" or difficulty == "expert_reasoning":
        return GPQA_INITIAL, record["input"]

    if source in ("humaneval", "mbpp") or difficulty == "code":
        if source == "mbpp" or (source != "humaneval" and "def " not in record["input"]):
            test_cases = record.get("test_cases", "")
            func_name_match = re.search(r'assert\s+(\w+)\s*\(', test_cases)
            func_hint = ""
            if func_name_match:
                func_hint = f"\n\nThe function MUST be named '{func_name_match.group(1)}'."
            return CODE_MBPP_INITIAL, record["input"] + func_hint
        return CODE_HUMANEVAL_INITIAL, record["input"]

    if source == "livecodebench" or difficulty.startswith("code_"):
        return LIVECODE_INITIAL, record["input"]

    return QA_INITIAL, record["input"]


# ── API call with retry ──────────────────────────────────────────────────────

def call_llm(client: OpenAI, model: str, messages: list[dict],
             max_tokens: int = 2048, temperature: float = 0.0,
             max_retries: int = 3) -> tuple[str, float]:
    """Call LLM API. Returns (response_text, cost_usd)."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,
            )
            text = resp.choices[0].message.content or ""
            total_tokens = resp.usage.total_tokens if resp.usage else 0
            cost = total_tokens * 5.0 / 1_000_000
            return text, cost
        except Exception as e:
            if attempt == max_retries - 1:
                return f"[API error: {str(e)}]", 0.0
            time.sleep(2 ** attempt)
    return "[API error: max retries exceeded]", 0.0


# ── Self-reflection loop ─────────────────────────────────────────────────────

def run_self_reflection(client: OpenAI, model: str, record: dict,
                        n_rounds: int = 5,
                        temperature: float = 0.0) -> dict:
    """Run N rounds of self-reflection on a single record."""
    system_prompt, user_prompt = get_initial_prompt(record)
    total_cost = 0.0

    # Round 1: initial answer
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    answer, cost = call_llm(client, model, messages, temperature=temperature)
    total_cost += cost
    messages.append({"role": "assistant", "content": answer})

    # Rounds 2..N: self-reflection
    for round_idx in range(1, n_rounds):
        messages.append({"role": "user", "content": REFLECTION_PROMPT})
        refined, cost = call_llm(client, model, messages, temperature=temperature)
        total_cost += cost
        messages.append({"role": "assistant", "content": refined})
        answer = refined

    return {
        "pred": answer,
        "total_cost": total_cost,
        "n_turns": n_rounds,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Self-Reflection baseline evaluation")
    parser.add_argument("--data_paths", nargs="+", required=True,
                        help="Paths to test JSONL files")
    parser.add_argument("--api_base", type=str,
                        default="YOUR_API_BASE")
    parser.add_argument("--api_key", type=str,
                        default="YOUR_API_KEY")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_rounds", type=int, default=5,
                        help="Number of self-reflection rounds")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per file (for quick testing)")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Parallel evaluation threads")
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
    print(f"Rounds: {args.n_rounds}")
    print(f"Workers: {args.max_workers}")

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    # Import metrics
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from eval.metrics import compute_metric

    # Run evaluation in parallel
    results = [None] * len(all_records)
    completed = 0
    lock = Lock()
    start_time = time.time()

    def eval_one(idx: int, record: dict) -> dict:
        rollout = run_self_reflection(client, args.model, record,
                                      n_rounds=args.n_rounds,
                                      temperature=args.temperature)
        metrics = compute_metric(rollout["pred"], record)
        return {
            "idx": idx,
            "input": record["input"][:200],
            "gold": record.get("answer", ""),
            "pred": rollout["pred"],
            "source": record.get("source", ""),
            "difficulty": record.get("difficulty", ""),
            "cost": rollout["total_cost"],
            "n_turns": rollout["n_turns"],
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
                    "n_turns": 0,
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
        for k, v in r["metrics"].items():
            if k not in ("em", "f1"):
                s["extra"].setdefault(k, 0)
                s["extra"][k] += v

    # Print results
    print(f"\n{'=' * 70}")
    print(f"Self-Reflection Baseline ({args.model}, {args.n_rounds} rounds)")
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
            "method": "self_reflection",
            "n_rounds": args.n_rounds,
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
            "avg_turns": float(args.n_rounds),
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
