"""
Pareto Frontier: Cost vs Quality (T4.3)

Plots the cost-quality Pareto frontier for OrchestratorR1 across different
cost penalty weights (alpha), overlaid with baseline fixed-point results.

Usage:
    python analysis/pareto_curve.py \
        --alpha_dir eval/results/ \
        --baselines eval/results/direct_gpt4o.json eval/results/react_baseline.json \
        --baseline_names "Direct-GPT-4o" "ReAct" \
        --output figures/pareto_curve.pdf

    # Cheap vs matched comparison:
    python analysis/pareto_curve.py \
        --alpha_dir eval/results/ \
        --pool cheap \
        --output figures/pareto_curve_cheap.pdf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)


# ── Style constants ───────────────────────────────────────────────────────────

OURS_COLOR = "#E74C3C"
BASELINE_STYLES = {
    "Direct-GPT-4o":    {"color": "#3498DB", "marker": "*",  "size": 200},
    "Router-R1":        {"color": "#2ECC71", "marker": "*",  "size": 200},
    "ReAct":            {"color": "#9B59B6", "marker": "^",  "size": 150},
    "Self-Reflection":  {"color": "#E67E22", "marker": "D",  "size": 120},
    "Fixed-Pipeline":   {"color": "#95A5A6", "marker": "v",  "size": 120},
    "w/o Reactive":     {"color": "#F39C12", "marker": "s",  "size": 120},
}

DEFAULT_ALPHAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]


def load_alpha_results(alpha_dir: str, pool: str = "cheap",
                       alphas: list[float] = None) -> list[dict]:
    """Load eval results for different alpha values.

    Expected file pattern: alpha_{alpha}_{pool}.json or alpha_{alpha}.json
    """
    alphas = alphas or DEFAULT_ALPHAS
    results = []
    alpha_dir = Path(alpha_dir)

    for a in alphas:
        # Try multiple naming patterns
        candidates = [
            alpha_dir / f"alpha_{a}_{pool}.json",
            alpha_dir / f"alpha_{a}.json",
            alpha_dir / f"orch_grpo_7b_alpha{a}_{pool}.json",
        ]
        for path in candidates:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append({
                    "alpha": a,
                    "f1": data["summary"]["f1"],
                    "cost": data["summary"]["avg_cost_usd"],
                    "em": data["summary"].get("em", 0),
                })
                break
        else:
            print(f"  Warning: No results found for alpha={a} pool={pool}")

    return results


def load_baseline(path: str) -> dict:
    """Load a single baseline result file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "f1": data["summary"]["f1"],
        "cost": data["summary"]["avg_cost_usd"],
        "em": data["summary"].get("em", 0),
    }


def plot_pareto(alpha_results: list[dict], baselines: dict[str, dict],
                output_path: str, title: str = ""):
    """Plot the Pareto frontier with baselines."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot OrchestratorR1 Pareto frontier
    if alpha_results:
        costs = [r["cost"] for r in alpha_results]
        f1s = [r["f1"] for r in alpha_results]
        ax.plot(costs, f1s, "o-", color=OURS_COLOR, markersize=8, linewidth=2,
                label="OrchestratorR1", zorder=10)

        for r in alpha_results:
            ax.annotate(
                f'α={r["alpha"]}',
                (r["cost"], r["f1"]),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=8,
                color=OURS_COLOR,
            )

    # Plot baselines
    for name, data in baselines.items():
        style = BASELINE_STYLES.get(name, {"color": "#7F8C8D", "marker": "o", "size": 120})
        ax.scatter(
            data["cost"],
            data["f1"],
            marker=style["marker"],
            s=style["size"],
            color=style["color"],
            label=name,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xlabel("Average Cost per Query ($)", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(title or "Cost vs Quality Pareto Frontier", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    # Format x-axis as currency
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.4f}"))

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Pareto curve to {output_path}")


def print_pareto_table(alpha_results: list[dict], baselines: dict[str, dict]):
    """Print a text table of all data points."""
    print(f"\n{'Method':<25} {'Alpha':>6} {'F1':>8} {'EM':>8} {'Cost':>12}")
    print("-" * 65)

    for r in alpha_results:
        print(f"  {'OrchestratorR1':<23} {r['alpha']:>6.1f} {r['f1']:>8.4f} "
              f"{r['em']:>8.4f} ${r['cost']:>10.6f}")

    print("-" * 65)
    for name, data in baselines.items():
        print(f"  {name:<23} {'—':>6} {data['f1']:>8.4f} "
              f"{data['em']:>8.4f} ${data['cost']:>10.6f}")

    # Check domination
    if alpha_results and baselines:
        print("\nDomination analysis:")
        for bname, bdata in baselines.items():
            for r in alpha_results:
                if r["f1"] >= bdata["f1"] and r["cost"] <= bdata["cost"]:
                    print(f"  α={r['alpha']:.1f} dominates {bname} "
                          f"(F1 {r['f1']:.4f} >= {bdata['f1']:.4f}, "
                          f"Cost ${r['cost']:.6f} <= ${bdata['cost']:.6f})")


def main():
    parser = argparse.ArgumentParser(description="Pareto frontier analysis")
    parser.add_argument("--alpha_dir", type=str, default="eval/results/",
                        help="Directory containing alpha sweep results")
    parser.add_argument("--pool", type=str, default="cheap",
                        choices=["cheap", "matched"])
    parser.add_argument("--baselines", nargs="*", default=[],
                        help="Paths to baseline result JSON files")
    parser.add_argument("--baseline_names", nargs="*", default=[],
                        help="Display names for baselines")
    parser.add_argument("--output", type=str,
                        default="figures/pareto_curve.pdf")
    args = parser.parse_args()

    # Load alpha sweep results
    alpha_results = load_alpha_results(args.alpha_dir, pool=args.pool)

    # Load baselines
    baselines = {}
    for i, path in enumerate(args.baselines):
        name = args.baseline_names[i] if i < len(args.baseline_names) else Path(path).stem
        try:
            baselines[name] = load_baseline(path)
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not load baseline {path}: {e}")

    if not alpha_results and not baselines:
        print("No data found. Ensure eval results exist in the specified directory.")
        print(f"Expected pattern: {args.alpha_dir}/alpha_{{alpha}}_{args.pool}.json")
        sys.exit(1)

    print_pareto_table(alpha_results, baselines)
    title = f"Cost vs Quality Pareto Frontier ({args.pool.title()} Pool)"
    plot_pareto(alpha_results, baselines, args.output, title=title)


if __name__ == "__main__":
    main()
