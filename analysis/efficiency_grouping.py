"""
Efficiency Grouping Analysis (T4.6)

Groups results by task difficulty (Simple/Multi-hop/Code) and computes
per-group statistics: avg turns, avg cost, F1. Proves the orchestrator
takes shortcuts on easy questions and invests more on hard ones.

Usage:
    python analysis/efficiency_grouping.py \
        --eval_json eval/results/orchestrator_r1.json \
        --baseline_json eval/results/fixed_pipeline.json \
        --output figures/efficiency_grouping.pdf

    # Compare cheap vs matched:
    python analysis/efficiency_grouping.py \
        --eval_json eval/results/orch_cheap.json eval/results/orch_matched.json \
        --labels "Cheap Pool" "Matched Pool" \
        --output figures/efficiency_grouping_comparison.pdf
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)


# ── Grouping configuration ───────────────────────────────────────────────────

SOURCE_TO_GROUP = {
    "nq": "Simple",
    "triviaqa": "Simple",
    "popqa": "Simple",
    "hotpotqa": "Multi-hop",
    "2wikimultihop": "Multi-hop",
    "2wiki": "Multi-hop",
    "musique": "Multi-hop",
    "humaneval": "Code",
    "mbpp": "Code",
    "gpqa_diamond": "Expert",
    "livecodebench": "LiveCode",
}

GROUP_ORDER = ["Simple", "Multi-hop", "Code", "Expert", "LiveCode"]
GROUP_COLORS = {
    "Simple": "#3498DB",
    "Multi-hop": "#E74C3C",
    "Code": "#2ECC71",
    "Expert": "#9B59B6",
    "LiveCode": "#E67E22",
}


def compute_group_stats(eval_data: dict) -> dict:
    """Compute per-group statistics from eval results."""
    groups = {}

    for r in eval_data["results"]:
        source = r.get("source", "")
        group = SOURCE_TO_GROUP.get(source)
        if group is None:
            continue

        if group not in groups:
            groups[group] = {"turns": [], "costs": [], "f1s": [], "ems": []}

        groups[group]["turns"].append(r.get("n_turns", 1))
        groups[group]["costs"].append(r.get("total_cost", r.get("cost", 0)))

        metrics = r.get("metrics", {})
        groups[group]["f1s"].append(metrics.get("f1", r.get("f1", 0)))
        groups[group]["ems"].append(metrics.get("em", r.get("em", 0)))

    stats = {}
    for g, data in groups.items():
        n = len(data["turns"])
        stats[g] = {
            "n": n,
            "avg_turns": mean(data["turns"]) if data["turns"] else 0,
            "avg_cost": mean(data["costs"]) if data["costs"] else 0,
            "avg_f1": mean(data["f1s"]) if data["f1s"] else 0,
            "avg_em": mean(data["ems"]) if data["ems"] else 0,
        }

    return stats


def print_stats_table(stats: dict, label: str = ""):
    """Print a formatted table of group statistics."""
    if label:
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")

    print(f"\n{'Group':<12} {'N':>5} {'Avg Turns':>10} {'Avg Cost':>12} "
          f"{'F1':>8} {'EM':>8}")
    print("-" * 60)

    for g in GROUP_ORDER:
        if g in stats:
            s = stats[g]
            print(f"  {g:<10} {s['n']:>5} {s['avg_turns']:>10.2f} "
                  f"${s['avg_cost']:>10.6f} {s['avg_f1']:>8.4f} {s['avg_em']:>8.4f}")

    print()

    # Efficiency ratio
    if "Simple" in stats and "Multi-hop" in stats:
        simple = stats["Simple"]
        multihop = stats["Multi-hop"]
        if simple["avg_turns"] > 0 and simple["avg_cost"] > 0:
            turn_ratio = multihop["avg_turns"] / simple["avg_turns"]
            cost_ratio = multihop["avg_cost"] / simple["avg_cost"] if simple["avg_cost"] > 0 else float("inf")
            print(f"  Multi-hop/Simple turn ratio:  {turn_ratio:.2f}x")
            print(f"  Multi-hop/Simple cost ratio:  {cost_ratio:.2f}x")


def plot_efficiency(stats_list: list[dict], labels: list[str],
                    output_path: str, baseline_stats: dict = None):
    """Plot grouped bar chart with dual Y-axis (turns + cost)."""
    groups_present = [g for g in GROUP_ORDER if any(g in s for s in stats_list)]
    if not groups_present:
        print("No data to plot.")
        return

    n_groups = len(groups_present)
    n_methods = len(stats_list) + (1 if baseline_stats else 0)
    width = 0.8 / n_methods
    x = np.arange(n_groups)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    method_colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]

    for m_idx, (stats, label) in enumerate(zip(stats_list, labels)):
        turns = [stats.get(g, {}).get("avg_turns", 0) for g in groups_present]
        costs = [stats.get(g, {}).get("avg_cost", 0) for g in groups_present]
        color = method_colors[m_idx % len(method_colors)]

        offset = (m_idx - n_methods / 2 + 0.5) * width
        bars = ax1.bar(x + offset, turns, width * 0.9, label=f"{label} (turns)",
                       color=color, alpha=0.7)
        ax2.plot(x + offset, costs, "D-", color=color, markersize=6,
                 label=f"{label} (cost)", linewidth=1.5)

    if baseline_stats:
        m_idx = len(stats_list)
        turns = [baseline_stats.get(g, {}).get("avg_turns", 0) for g in groups_present]
        costs = [baseline_stats.get(g, {}).get("avg_cost", 0) for g in groups_present]
        offset = (m_idx - n_methods / 2 + 0.5) * width
        ax1.bar(x + offset, turns, width * 0.9, label="Fixed-Pipeline (turns)",
                color="#95A5A6", alpha=0.5)
        ax2.plot(x + offset, costs, "s--", color="#95A5A6", markersize=6,
                 label="Fixed-Pipeline (cost)", linewidth=1.5)

    ax1.set_xlabel("Task Category", fontsize=12)
    ax1.set_ylabel("Average Turns", fontsize=12, color="#E74C3C")
    ax2.set_ylabel("Average Cost ($)", fontsize=12, color="#3498DB")
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups_present, fontsize=11)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    ax1.set_title("Efficiency by Task Difficulty", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved efficiency plot to {output_path}")


def plot_f1_comparison(stats_list: list[dict], labels: list[str],
                       output_path: str):
    """Plot F1 comparison across groups as a grouped bar chart."""
    groups_present = [g for g in GROUP_ORDER if any(g in s for s in stats_list)]
    if not groups_present:
        return

    n_groups = len(groups_present)
    n_methods = len(stats_list)
    width = 0.8 / n_methods
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(8, 5))
    method_colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]

    for m_idx, (stats, label) in enumerate(zip(stats_list, labels)):
        f1s = [stats.get(g, {}).get("avg_f1", 0) for g in groups_present]
        offset = (m_idx - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, f1s, width * 0.9, label=label,
               color=method_colors[m_idx % len(method_colors)], alpha=0.8)

    ax.set_xlabel("Task Category", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(groups_present, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_title("F1 by Task Difficulty", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    f1_path = output_path.replace(".pdf", "_f1.pdf")
    Path(f1_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved F1 comparison to {f1_path}")


def main():
    parser = argparse.ArgumentParser(description="Efficiency grouping analysis")
    parser.add_argument("--eval_json", nargs="+", required=True,
                        help="Path(s) to eval result JSON files")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each eval JSON")
    parser.add_argument("--baseline_json", type=str, default=None,
                        help="Path to Fixed-Pipeline baseline results")
    parser.add_argument("--output", type=str,
                        default="figures/efficiency_grouping.pdf")
    args = parser.parse_args()

    labels = args.labels or [Path(p).stem for p in args.eval_json]

    stats_list = []
    for path, label in zip(args.eval_json, labels):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        stats = compute_group_stats(data)
        stats_list.append(stats)
        print_stats_table(stats, label)

    baseline_stats = None
    if args.baseline_json:
        with open(args.baseline_json, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        baseline_stats = compute_group_stats(baseline_data)
        print_stats_table(baseline_stats, "Fixed-Pipeline Baseline")

    plot_efficiency(stats_list, labels, args.output, baseline_stats)
    plot_f1_comparison(stats_list, labels, args.output)


if __name__ == "__main__":
    main()
