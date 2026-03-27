"""
Evaluate Orchestrator-R1 on QA benchmarks.

Usage:
    python eval/eval_orchestrator.py \
        --model_path checkpoints/orchestrator_r1/final \
        --data_path data/test.jsonl \
        --api_base YOUR_API_BASE \
        --api_key YOUR_API_KEY \
        --output eval/results/orchestrator_r1.json
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from orchestrator_r1.agent_pool.agent_registry import AgentRegistry
from orchestrator_r1.orchestrator.generation import OrchestratorGenerationManager, GenerationConfig
from orchestrator_r1.orchestrator.reward import compute_em, compute_f1
from eval.metrics import compute_metric


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  type=str, required=True)
    parser.add_argument("--data_path",   type=str, required=True)
    parser.add_argument("--api_base",    type=str, required=True)
    parser.add_argument("--api_key",     type=str, required=True)
    parser.add_argument("--output",      type=str, default="eval/results/orchestrator_r1.json")
    parser.add_argument("--max_turns",   type=int, default=6)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--worker_pool", type=str, default="cheap",
                        choices=["cheap", "matched"],
                        help="Worker pool configuration: 'cheap' or 'matched'")
    parser.add_argument("--device",      type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model (supports both local path and HuggingFace repo ID)
    model_path = args.model_path
    if not model_path.startswith("/") and not model_path.startswith(".") and "\\" not in model_path and ":" not in model_path:
        # Looks like a HF repo ID (e.g. "username/model-name")
        local_only = False
    else:
        model_path = str(Path(model_path).resolve())
        local_only = True
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", local_files_only=local_only,
    ).to(args.device)
    model.eval()

    registry = AgentRegistry(api_base=args.api_base, api_key=args.api_key,
                              worker_pool=args.worker_pool)
    gen_config = GenerationConfig(max_turns=args.max_turns)
    manager = OrchestratorGenerationManager(model, tokenizer, registry, gen_config)

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
    total_em = total_f1 = total_cost = total_turns = 0.0

    for record in tqdm(records, desc="Evaluating"):
        rollout = manager.rollout(record["input"])
        gold = record["answer"]
        pred = rollout.answer or ""

        em = compute_em(pred, gold)
        f1 = compute_f1(pred, gold)
        metrics = compute_metric(pred, record)
        total_em   += em
        total_f1   += f1
        total_cost += rollout.total_cost
        total_turns += rollout.n_turns

        results.append({
            "input":       record["input"],
            "gold":        gold,
            "pred":        pred,
            "em":          em,
            "f1":          f1,
            "metrics":     metrics,
            "n_turns":     rollout.n_turns,
            "total_cost":  rollout.total_cost,
            "agent_calls": rollout.agent_calls,
            "source":      record.get("source", ""),
            "difficulty":  record.get("difficulty", ""),
        })

    n = len(results)
    summary = {
        "n_samples":    n,
        "worker_pool":  args.worker_pool,
        "em":           total_em / n,
        "f1":           total_f1 / n,
        "avg_cost_usd": total_cost / n,
        "avg_turns":    total_turns / n,
    }
    print("\n=== Evaluation Results ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
