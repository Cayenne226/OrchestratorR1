"""
Evaluate baseline methods on the same test data for fair comparison.

Baselines:
  1. Direct-Strong: Send query directly to GPT-4o (no orchestration)
  2. Fixed-Pipeline: Always run full pipeline (refiner→decomposer→executor→critic→synthesizer)

Router-R1 and ReAct baselines should be evaluated using their own codebases.

Usage:
    python eval/baselines.py \
        --method direct_strong \
        --data_path data/test.jsonl \
        --api_base YOUR_API_BASE \
        --api_key YOUR_API_KEY \
        --output eval/results/direct_strong.json

    python eval/baselines.py \
        --method fixed_pipeline \
        --data_path data/test.jsonl \
        --api_base YOUR_API_BASE \
        --api_key YOUR_API_KEY \
        --output eval/results/fixed_pipeline.json
"""

import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm

from orchestrator_r1.agent_pool.agent_registry import AgentRegistry
from orchestrator_r1.orchestrator.reward import compute_em, compute_f1
from eval.metrics import compute_metric

# Suffix appended to every baseline query to elicit short answers,
# matching the concise style that OrchestratorR1 produces via <answer> tags.
SHORT_ANSWER_SUFFIX = "\n\nAnswer with only the short factual answer, nothing else."


def _is_code_task(record: dict) -> bool:
    """Check if this record is a code task (should not append short-answer suffix)."""
    source = record.get("source", "")
    difficulty = record.get("difficulty", "")
    return source in ("humaneval", "mbpp", "livecodebench") or difficulty.startswith("code")


def eval_direct_strong(record: dict, registry: AgentRegistry) -> dict:
    """Baseline: Send query directly to strong model, no orchestration."""
    question = record["input"]
    suffix = "" if _is_code_task(record) else SHORT_ANSWER_SUFFIX
    response, cost = registry.dispatch("executor_strong", question + suffix)

    return {
        "pred": response.strip(),
        "n_turns": 1,
        "total_cost": cost,
        "agent_calls": [{"agent_type": "executor_strong", "cost": cost}],
    }


def eval_direct_cheap(record: dict, registry: AgentRegistry) -> dict:
    """Baseline: Send query directly to cheap model, no orchestration."""
    question = record["input"]
    suffix = "" if _is_code_task(record) else SHORT_ANSWER_SUFFIX
    response, cost = registry.dispatch("executor_cheap", question + suffix)

    return {
        "pred": response.strip(),
        "n_turns": 1,
        "total_cost": cost,
        "agent_calls": [{"agent_type": "executor_cheap", "cost": cost}],
    }


def eval_fixed_pipeline(record: dict, registry: AgentRegistry) -> dict:
    """Baseline: Always run the full 6-step pipeline regardless of task complexity."""
    question = record["input"]
    is_code = _is_code_task(record)
    suffix = "" if is_code else SHORT_ANSWER_SUFFIX
    agent_calls = []
    total_cost = 0.0

    # Step 1: Refiner
    refined, cost = registry.dispatch("refiner", question)
    agent_calls.append({"agent_type": "refiner", "cost": cost})
    total_cost += cost

    # Step 2: Decomposer
    plan, cost = registry.dispatch("decomposer", refined)
    agent_calls.append({"agent_type": "decomposer", "cost": cost})
    total_cost += cost

    # Step 3: Executor (strong, on the refined+decomposed task)
    exec_query = f"Based on this plan: {plan}\n\nAnswer the original question: {question}{suffix}"
    result, cost = registry.dispatch("executor_strong", exec_query)
    agent_calls.append({"agent_type": "executor_strong", "cost": cost})
    total_cost += cost

    # Step 4: Critic
    critic_query = f"Question: {question}\nAnswer: {result}\n\nIs this answer correct and complete?"
    feedback, cost = registry.dispatch("critic", critic_query)
    agent_calls.append({"agent_type": "critic", "cost": cost})
    total_cost += cost

    # Step 5: Executor again if critic found issues
    if any(w in feedback.lower() for w in ["incorrect", "incomplete", "missing", "wrong"]):
        retry_query = f"Original question: {question}\nPrevious answer: {result}\nFeedback: {feedback}\nPlease provide a corrected answer.{suffix}"
        result, cost = registry.dispatch("executor_strong", retry_query)
        agent_calls.append({"agent_type": "executor_strong", "cost": cost})
        total_cost += cost

    # Step 6: Synthesizer
    if is_code:
        synth_query = f"Question: {question}\nCode: {result}\n\nCombine into a clean, final solution. Output only the code, nothing else."
    else:
        synth_query = (
            f"Question: {question}\n"
            f"Context: {result}\n\n"
            f"Based on the context above, what is the answer to the question? "
            f"Reply with ONLY the answer entity (a name, number, date, or short phrase). "
            f"No explanation, no full sentences."
        )
    final, cost = registry.dispatch("synthesizer", synth_query)
    agent_calls.append({"agent_type": "synthesizer", "cost": cost})
    total_cost += cost

    return {
        "pred": final.strip(),
        "n_turns": len(agent_calls),
        "total_cost": total_cost,
        "agent_calls": agent_calls,
    }


METHODS = {
    "direct_strong": eval_direct_strong,
    "direct_cheap": eval_direct_cheap,
    "fixed_pipeline": eval_fixed_pipeline,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",     type=str, required=True,
                        choices=list(METHODS.keys()))
    parser.add_argument("--data_path",  type=str, required=True)
    parser.add_argument("--api_base",   type=str, required=True)
    parser.add_argument("--api_key",    type=str, required=True)
    parser.add_argument("--output",     type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    registry = AgentRegistry(api_base=args.api_base, api_key=args.api_key)
    eval_fn = METHODS[args.method]

    # Load test data
    records = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.max_samples:
        records = records[:args.max_samples]

    # Evaluate
    results = []
    total_em = total_f1 = total_cost = total_turns = total_latency = 0.0

    for record in tqdm(records, desc=f"Eval {args.method}"):
        t0 = time.time()
        output = eval_fn(record, registry)
        latency_sec = time.time() - t0
        total_latency += latency_sec
        gold = record["answer"]
        pred = output["pred"]

        metrics = compute_metric(pred, record)
        em = metrics.get("em", 0.0)
        f1 = metrics.get("f1", 0.0)
        total_em += em
        total_f1 += f1
        total_cost += output["total_cost"]
        total_turns += output["n_turns"]

        results.append({
            "input": record["input"],
            "gold": gold,
            "pred": pred,
            "em": em,
            "f1": f1,
            "metrics": metrics,
            "n_turns": output["n_turns"],
            "total_cost": output["total_cost"],
            "agent_calls": output["agent_calls"],
            "latency_sec": round(latency_sec, 3),
            "source": record.get("source", ""),
        })

    n = len(results)
    summary = {
        "method": args.method,
        "n_samples": n,
        "em": total_em / n if n > 0 else 0,
        "f1": total_f1 / n if n > 0 else 0,
        "avg_cost_usd": total_cost / n if n > 0 else 0,
        "avg_turns": total_turns / n if n > 0 else 0,
        "avg_latency_sec": total_latency / n if n > 0 else 0,
    }

    print("\n=== Baseline Results ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
