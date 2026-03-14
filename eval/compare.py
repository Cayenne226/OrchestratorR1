"""
Compare multiple evaluation results and generate summary tables.

Reads result JSON files from eval/results/ and outputs:
  - Console comparison table
  - Per-source breakdown
  - Comparison JSON

Usage:
    python eval/compare.py \
        --results eval/results/orch_grpo.json eval/results/router_r1.json \
        --labels "Orch-GRPO" "Router-R1" \
        --output eval/results/comparison.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional


def load_result(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_per_source(results: List[dict]) -> Dict[str, dict]:
    """Group results by source and compute per-source metrics."""
    source_groups = {}
    for r in results:
        src = r.get("source", "unknown")
        if src not in source_groups:
            source_groups[src] = {"em_sum": 0, "f1_sum": 0, "cost_sum": 0, "turns_sum": 0, "n": 0}
        g = source_groups[src]
        g["em_sum"] += r.get("em", 0)
        g["f1_sum"] += r.get("f1", 0)
        g["cost_sum"] += r.get("total_cost", 0)
        g["turns_sum"] += r.get("n_turns", 0)
        g["n"] += 1

    per_source = {}
    for src, g in source_groups.items():
        n = g["n"]
        per_source[src] = {
            "n": n,
            "em": g["em_sum"] / n if n > 0 else 0,
            "f1": g["f1_sum"] / n if n > 0 else 0,
            "avg_cost": g["cost_sum"] / n if n > 0 else 0,
            "avg_turns": g["turns_sum"] / n if n > 0 else 0,
        }
    return per_source


def compute_agent_distribution(results: List[dict]) -> Dict[str, int]:
    """Count agent type usage across all results."""
    dist = {}
    for r in results:
        for call in r.get("agent_calls", []):
            agent_type = call.get("agent_type", "unknown")
            dist[agent_type] = dist.get(agent_type, 0) + 1
    return dist


def print_comparison_table(data_list: List[dict], labels: List[str]):
    """Print a formatted comparison table to console."""

    # Header
    header = f"{'Metric':<20}"
    for label in labels:
        header += f"{label:>16}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    # Summary metrics
    metrics = ["em", "f1", "avg_cost_usd", "avg_turns", "n_samples"]
    metric_labels = {
        "em": "EM",
        "f1": "F1",
        "avg_cost_usd": "Avg Cost ($)",
        "avg_turns": "Avg Turns",
        "n_samples": "N Samples",
    }

    for metric in metrics:
        row = f"{metric_labels.get(metric, metric):<20}"
        for data in data_list:
            val = data.get("summary", {}).get(metric, "N/A")
            if isinstance(val, float):
                row += f"{val:>16.4f}"
            else:
                row += f"{val:>16}"
        print(row)

    print("=" * len(header))

    # Per-source breakdown if available
    all_sources = set()
    per_source_list = []
    for data in data_list:
        ps = compute_per_source(data.get("results", []))
        per_source_list.append(ps)
        all_sources.update(ps.keys())

    if all_sources:
        print(f"\n{'Per-source F1':<20}", end="")
        for label in labels:
            print(f"{label:>16}", end="")
        print()
        print("-" * (20 + 16 * len(labels)))

        for src in sorted(all_sources):
            row = f"  {src:<18}"
            for ps in per_source_list:
                val = ps.get(src, {}).get("f1", "N/A")
                if isinstance(val, float):
                    row += f"{val:>16.4f}"
                else:
                    row += f"{val:>16}"
            print(row)

    # Agent distribution
    print(f"\n{'Agent Distribution':<20}", end="")
    for label in labels:
        print(f"{label:>16}", end="")
    print()
    print("-" * (20 + 16 * len(labels)))

    all_agents = set()
    agent_dists = []
    for data in data_list:
        ad = compute_agent_distribution(data.get("results", []))
        agent_dists.append(ad)
        all_agents.update(ad.keys())

    for agent in sorted(all_agents):
        row = f"  {agent:<18}"
        for ad in agent_dists:
            count = ad.get(agent, 0)
            row += f"{count:>16}"
        print(row)

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True,
                        help="Paths to result JSON files")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each result (default: filenames)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output comparison JSON path")
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [Path(p).stem for p in args.results]

    if len(args.labels) != len(args.results):
        print(f"Error: {len(args.labels)} labels but {len(args.results)} result files")
        return

    # Load all results
    data_list = []
    for path in args.results:
        data = load_result(path)
        data_list.append(data)

    # Print comparison
    print_comparison_table(data_list, args.labels)

    # Save comparison JSON
    if args.output:
        comparison = {}
        for label, data in zip(args.labels, data_list):
            comparison[label] = {
                "summary": data.get("summary", {}),
                "per_source": compute_per_source(data.get("results", [])),
                "agent_distribution": compute_agent_distribution(data.get("results", [])),
            }

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"Comparison saved to {args.output}")


if __name__ == "__main__":
    main()
