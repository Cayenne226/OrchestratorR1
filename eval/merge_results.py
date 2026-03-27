"""
Merge individual per-test-set baseline results into a single combined result file.

Usage:
    python eval/merge_results.py \
        --inputs eval/results/direct_gpt4o_qa.json eval/results/direct_gpt4o_gpqa.json \
                 eval/results/direct_gpt4o_code.json eval/results/direct_gpt4o_livecode.json \
        --output eval/results/direct_gpt4o_all.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    all_results = []
    total_cost = 0
    configs = []

    for path in args.inputs:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_results.extend(data.get("results", []))
        total_cost += data.get("summary", {}).get("total_cost_usd", 0)
        configs.append({"file": path, "n": data["summary"]["n_samples"]})

    # Compute aggregate stats
    source_stats = {}
    for r in all_results:
        src = r.get("source", "unknown")
        if src not in source_stats:
            source_stats[src] = {"em": 0, "f1": 0, "cost": 0, "n": 0}
        s = source_stats[src]
        s["n"] += 1
        s["em"] += r.get("metrics", {}).get("em", 0)
        s["f1"] += r.get("metrics", {}).get("f1", 0)
        s["cost"] += r.get("cost", 0)

    n = len(all_results)
    total_em = sum(s["em"] for s in source_stats.values())
    total_f1 = sum(s["f1"] for s in source_stats.values())

    per_source = {}
    for src, s in source_stats.items():
        nn = s["n"]
        per_source[src] = {
            "n": nn,
            "em": round(s["em"] / nn, 4),
            "f1": round(s["f1"] / nn, 4),
            "avg_cost": round(s["cost"] / nn, 6),
        }

    output_data = {
        "config": {"merged_from": configs, "n_samples": n},
        "summary": {
            "n_samples": n,
            "em": round(total_em / n, 4) if n else 0,
            "f1": round(total_f1 / n, 4) if n else 0,
            "total_cost_usd": round(total_cost, 4),
            "avg_cost_usd": round(total_cost / n, 6) if n else 0,
        },
        "per_source": per_source,
    }

    # Print
    print(f"{'Source':<20} {'N':>5} {'EM':>8} {'F1':>8} {'Cost':>10}")
    print("-" * 55)
    for src in sorted(per_source):
        v = per_source[src]
        print(f"  {src:<18} {v['n']:>5} {v['em']:>8.4f} {v['f1']:>8.4f} ${v['avg_cost']:>9.6f}")
    print("-" * 55)
    print(f"  {'OVERALL':<18} {n:>5} {output_data['summary']['em']:>8.4f} "
          f"{output_data['summary']['f1']:>8.4f} ${output_data['summary']['avg_cost_usd']:>9.6f}")
    print(f"\nTotal cost: ${total_cost:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nMerged {n} results → {args.output}")


if __name__ == "__main__":
    main()
