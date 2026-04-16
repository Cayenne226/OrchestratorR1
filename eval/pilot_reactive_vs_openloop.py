"""
Pilot experiment: reactive vs. open-loop under normal and adversarial conditions.

Purpose:
  Validate the core hypothesis before investing 5 weeks of GPU time.
  If reactive >> open-loop under adversarial noise → paper angle is "robustness".
  If gap is small in both conditions → reconsider direction.

Usage:
    python eval/pilot_reactive_vs_openloop.py \
        --model_path Qwen/Qwen2.5-3B-Instruct \
        --data_path data/test_qa.jsonl \
        --api_base YOUR_API_BASE \
        --api_key YOUR_API_KEY \
        --n_samples 100 \
        --output eval/results/pilot_reactive_vs_openloop.json

    # Use --device cpu if no GPU available (slow but works)
"""

import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from orchestrator_r1.agent_pool.agent_registry import AgentRegistry
from orchestrator_r1.orchestrator.generation import OrchestratorGenerationManager, GenerationConfig
from orchestrator_r1.orchestrator.generation_openloop import OpenLoopGenerationManager
from eval.metrics import compute_em, compute_f1


def run_condition(name, manager, records, registry_with_noise=None):
    """Run a single experimental condition, return per-sample results."""
    results = []
    for record in tqdm(records, desc=name):
        t0 = time.time()
        rollout = manager.rollout(record["input"])
        latency = time.time() - t0
        pred = rollout.answer or ""
        gold = record["answer"]
        results.append({
            "input":   record["input"],
            "gold":    gold,
            "pred":    pred,
            "em":      compute_em(pred, gold),
            "f1":      compute_f1(pred, gold),
            "cost":    rollout.total_cost,
            "n_turns": rollout.n_turns,
            "latency": round(latency, 3),
            "calls":   rollout.agent_calls,
        })
    return results


def summarize(results):
    n = len(results)
    if n == 0:
        return {}
    return {
        "n":          n,
        "em":         round(sum(r["em"] for r in results) / n, 4),
        "f1":         round(sum(r["f1"] for r in results) / n, 4),
        "avg_cost":   round(sum(r["cost"] for r in results) / n, 6),
        "avg_turns":  round(sum(r["n_turns"] for r in results) / n, 2),
        "avg_latency": round(sum(r["latency"] for r in results) / n, 2),
    }


class NoisyRegistry(AgentRegistry):
    """Wraps AgentRegistry to inject adversarial noise via dispatch_with_noise."""

    def __init__(self, base_registry, corrupt_prob=0.3, timeout_prob=0.0):
        # Steal internals from the base registry
        self.agents = base_registry.agents
        self.pool_name = base_registry.pool_name
        self._corrupt_prob = corrupt_prob
        self._timeout_prob = timeout_prob

    def dispatch(self, agent_type, query):
        """Override dispatch to inject noise transparently."""
        resp, cost, _meta = self.dispatch_with_noise(
            agent_type, query,
            corrupt_prob=self._corrupt_prob,
            timeout_prob=self._timeout_prob,
        )
        return resp, cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  type=str, required=True)
    parser.add_argument("--data_path",   type=str, required=True)
    parser.add_argument("--api_base",    type=str, required=True)
    parser.add_argument("--api_key",     type=str, required=True)
    parser.add_argument("--n_samples",   type=int, default=100)
    parser.add_argument("--max_turns",   type=int, default=6)
    parser.add_argument("--noise_prob",  type=float, default=0.3,
                        help="Probability of corrupted agent response in adversarial condition")
    parser.add_argument("--output",      type=str,
                        default="eval/results/pilot_reactive_vs_openloop.json")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--only_multihop", action="store_true",
                        help="Filter to multi_hop questions only")
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model_path}")
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

    # Load data — filter to multi-hop if requested
    records = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if args.only_multihop and r.get("difficulty") != "multi_hop":
                continue
            records.append(r)
    records = records[:args.n_samples]
    print(f"Loaded {len(records)} samples")

    # Setup registries
    clean_registry = AgentRegistry(api_base=args.api_base, api_key=args.api_key)
    noisy_registry = NoisyRegistry(clean_registry, corrupt_prob=args.noise_prob)

    gen_config = GenerationConfig(max_turns=args.max_turns)

    # === 4 conditions ===
    conditions = {}

    # 1. Reactive + clean
    print("\n=== Condition 1/4: Reactive (clean) ===")
    mgr = OrchestratorGenerationManager(model, tokenizer, clean_registry, gen_config)
    conditions["reactive_clean"] = run_condition("reactive_clean", mgr, records)

    # 2. Open-loop + clean
    print("\n=== Condition 2/4: Open-loop (clean) ===")
    mgr = OpenLoopGenerationManager(model, tokenizer, clean_registry, gen_config)
    conditions["openloop_clean"] = run_condition("openloop_clean", mgr, records)

    # 3. Reactive + adversarial
    print("\n=== Condition 3/4: Reactive (adversarial, {:.0%} noise) ===".format(args.noise_prob))
    mgr = OrchestratorGenerationManager(model, tokenizer, noisy_registry, gen_config)
    conditions["reactive_adversarial"] = run_condition("reactive_adversarial", mgr, records)

    # 4. Open-loop + adversarial
    print("\n=== Condition 4/4: Open-loop (adversarial, {:.0%} noise) ===".format(args.noise_prob))
    mgr = OpenLoopGenerationManager(model, tokenizer, noisy_registry, gen_config)
    conditions["openloop_adversarial"] = run_condition("openloop_adversarial", mgr, records)

    # === Summary ===
    summaries = {k: summarize(v) for k, v in conditions.items()}

    print("\n" + "=" * 70)
    print("PILOT RESULTS: reactive vs. open-loop")
    print("=" * 70)
    print(f"{'Condition':<30} {'EM':>8} {'F1':>8} {'Cost':>10} {'Turns':>6}")
    print("-" * 70)
    for name, s in summaries.items():
        print(f"{name:<30} {s['em']:>8.4f} {s['f1']:>8.4f} {s['avg_cost']:>10.6f} {s['avg_turns']:>6.2f}")

    # Key comparisons
    rc = summaries["reactive_clean"]
    oc = summaries["openloop_clean"]
    ra = summaries["reactive_adversarial"]
    oa = summaries["openloop_adversarial"]

    print("\n--- Key Deltas ---")
    print(f"Clean gap (reactive - openloop):       EM={rc['em']-oc['em']:+.4f}  F1={rc['f1']-oc['f1']:+.4f}")
    print(f"Adversarial gap (reactive - openloop):  EM={ra['em']-oa['em']:+.4f}  F1={ra['f1']-oa['f1']:+.4f}")
    print(f"Reactive degradation (clean→adv):       EM={ra['em']-rc['em']:+.4f}  F1={ra['f1']-rc['f1']:+.4f}")
    print(f"Open-loop degradation (clean→adv):      EM={oa['em']-oc['em']:+.4f}  F1={oa['f1']-oc['f1']:+.4f}")

    # Decision guidance
    adv_gap_em = ra["em"] - oa["em"]
    clean_gap_em = rc["em"] - oc["em"]
    print("\n--- Interpretation ---")
    if adv_gap_em > 0.05 and clean_gap_em < 0.03:
        print(">> STRONG SIGNAL: Small clean gap + large adversarial gap")
        print(">> Paper angle: 'robustness to intermediate failures'")
    elif adv_gap_em > 0.05 and clean_gap_em > 0.05:
        print(">> Reactive wins in both conditions — straightforward superiority story")
    elif adv_gap_em < 0.03 and clean_gap_em < 0.03:
        print(">> WARNING: Gap is small in both conditions. Consider:")
        print("   - Is the SFT model too weak to show the effect?")
        print("   - Try with a stronger base model or after GRPO training")
        print("   - The reactive advantage may emerge only with RL training")
    else:
        print(">> Mixed results — investigate per-sample patterns")

    # Save
    output = {
        "config": {
            "model_path": args.model_path,
            "n_samples": len(records),
            "noise_prob": args.noise_prob,
            "max_turns": args.max_turns,
            "only_multihop": args.only_multihop,
        },
        "summaries": summaries,
        "conditions": conditions,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
