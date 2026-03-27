"""
Evaluate the w/o reactive (open-loop) ablation.

Uses OpenLoopGenerationManager: model generates all <call> tags at once,
then receives all agent responses in bulk, then generates <answer>.

This ablation tests: does reactive (step-by-step) orchestration matter?
If open-loop performs similarly, then reactive is not a meaningful contribution.

Usage:
    python eval/run_ablation_openloop.py \
        --model_path checkpoints/orchestrator_grpo/final \
        --data_path data/test_qa.jsonl \
        --api_base YOUR_API_BASE \
        --api_key YOUR_API_KEY \
        --output eval/results/ablation_openloop.json
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from orchestrator_r1.agent_pool.agent_registry import AgentRegistry
from orchestrator_r1.orchestrator.generation_openloop import OpenLoopGenerationManager
from orchestrator_r1.orchestrator.generation import GenerationConfig
from eval.metrics import compute_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--api_base", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval/results/ablation_openloop.json")
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load model
    model_path = args.model_path
    if not model_path.startswith("/") and not model_path.startswith(".") and "\\" not in model_path and ":" not in model_path:
        local_only = False
    else:
        model_path = str(Path(model_path).resolve())
        local_only = True

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", local_files_only=local_only,
    ).to(args.device)
    model.eval()

    registry = AgentRegistry(api_base=args.api_base, api_key=args.api_key)
    gen_config = GenerationConfig(max_turns=args.max_turns)

    # Use open-loop manager instead of reactive
    manager = OpenLoopGenerationManager(model, tokenizer, registry, gen_config)

    # Load test data
    records = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.max_samples:
        records = records[:args.max_samples]

    # Run evaluation
    results = []
    totals = {"em": 0, "f1": 0, "cost": 0, "turns": 0}

    for record in tqdm(records, desc="Eval (open-loop)"):
        rollout = manager.rollout(record["input"])
        pred = rollout.answer or ""

        metrics = compute_metric(pred, record)
        totals["em"] += metrics.get("em", 0)
        totals["f1"] += metrics.get("f1", 0)
        totals["cost"] += rollout.total_cost
        totals["turns"] += rollout.n_turns

        results.append({
            "input": record["input"][:200],
            "gold": record.get("answer", ""),
            "pred": pred,
            "source": record.get("source", ""),
            "difficulty": record.get("difficulty", ""),
            "metrics": metrics,
            "n_turns": rollout.n_turns,
            "total_cost": rollout.total_cost,
            "agent_calls": rollout.agent_calls,
        })

    n = len(results)
    summary = {
        "method": "ablation_openloop",
        "n_samples": n,
        "em": totals["em"] / n if n else 0,
        "f1": totals["f1"] / n if n else 0,
        "avg_cost_usd": totals["cost"] / n if n else 0,
        "avg_turns": totals["turns"] / n if n else 0,
    }

    print("\n=== Open-Loop Ablation Results ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Per-source breakdown
    source_stats = {}
    for r in results:
        src = r["source"]
        if src not in source_stats:
            source_stats[src] = {"em": 0, "f1": 0, "n": 0}
        source_stats[src]["em"] += r["metrics"].get("em", 0)
        source_stats[src]["f1"] += r["metrics"].get("f1", 0)
        source_stats[src]["n"] += 1

    print("\nPer-source:")
    for src in sorted(source_stats):
        s = source_stats[src]
        print(f"  {src}: EM={s['em']/s['n']:.4f} F1={s['f1']/s['n']:.4f} (n={s['n']})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
