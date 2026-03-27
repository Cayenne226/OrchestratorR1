"""
Agent Call Distribution Heatmap (T4.1)

Generates a heatmap showing which agent types are invoked for each dataset,
proving the orchestrator learns task-dependent routing strategies.

Usage:
    python analysis/agent_distribution.py \
        --eval_json eval/results/orchestrator_r1.json \
        --output figures/heatmap_agent_distribution.pdf

    # Side-by-side comparison (cheap vs matched):
    python analysis/agent_distribution.py \
        --eval_json eval/results/orch_grpo_7b_cheap.json eval/results/orch_grpo_7b_matched.json \
        --labels "Cheap Pool" "Matched Pool" \
        --output figures/heatmap_agent_distribution_comparison.pdf
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
    import seaborn as sns
except ImportError:
    print("ERROR: matplotlib and seaborn required. Install with: pip install matplotlib seaborn")
    sys.exit(1)


# ── Data sources and agent types ──────────────────────────────────────────────

SOURCES = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihop", "musique",
           "humaneval", "mbpp"]
SOURCE_LABELS = ["NQ", "TriviaQA", "PopQA", "HotpotQA", "2Wiki", "MuSiQue",
                 "HumanEval", "MBPP"]
AGENTS = ["refiner", "decomposer", "executor_cheap", "executor_strong",
          "critic", "synthesizer"]
AGENT_LABELS = ["Refiner", "Decomposer", "Exec-Cheap", "Exec-Strong",
                "Critic", "Synthesizer"]


def build_distribution_matrix(eval_data: dict) -> np.ndarray:
    """Build a (n_sources x n_agents) frequency matrix from eval results."""
    matrix = np.zeros((len(SOURCES), len(AGENTS)))

    for i, src in enumerate(SOURCES):
        src_results = [r for r in eval_data["results"] if r.get("source") == src]
        if not src_results:
            continue

        total_calls = 0
        agent_counts = {a: 0 for a in AGENTS}

        for r in src_results:
            calls = r.get("agent_calls", [])
            total_calls += len(calls)
            for c in calls:
                agent_type = c.get("agent_type", "")
                if agent_type in agent_counts:
                    agent_counts[agent_type] += 1

        if total_calls > 0:
            for j, agent in enumerate(AGENTS):
                matrix[i][j] = agent_counts[agent] / total_calls

    return matrix


def plot_single_heatmap(matrix: np.ndarray, output_path: str, title: str = ""):
    """Plot a single heatmap."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=AGENT_LABELS,
        yticklabels=SOURCE_LABELS,
        ax=ax,
        vmin=0,
        vmax=0.5,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Agent Type", fontsize=12)
    ax.set_ylabel("Dataset", fontsize=12)
    ax.set_title(title or "Agent Call Distribution (Normalized per Dataset)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {output_path}")


def plot_comparison_heatmap(matrices: list[np.ndarray], labels: list[str],
                            output_path: str):
    """Plot side-by-side heatmaps for cheap vs matched pool comparison."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, matrix, label in zip(axes, matrices, labels):
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            xticklabels=AGENT_LABELS,
            yticklabels=SOURCE_LABELS,
            ax=ax,
            vmin=0,
            vmax=0.5,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Agent Type", fontsize=11)
        ax.set_ylabel("Dataset", fontsize=11)

    plt.suptitle("Agent Call Distribution: Cheap vs Matched Pool",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison heatmap to {output_path}")


def print_summary(matrix: np.ndarray, label: str = ""):
    """Print a text summary of the distribution patterns."""
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

    print(f"\n{'Source':<15}", end="")
    for a in AGENT_LABELS:
        print(f"{a:>12}", end="")
    print()
    print("-" * (15 + 12 * len(AGENTS)))

    for i, src in enumerate(SOURCE_LABELS):
        print(f"{src:<15}", end="")
        for j in range(len(AGENTS)):
            print(f"{matrix[i][j]:>12.3f}", end="")
        print()

    # Highlight emergent patterns
    print("\nEmergent patterns:")
    simple_idx = [0, 1, 2]  # NQ, TriviaQA, PopQA
    multihop_idx = [3, 4, 5]  # HotpotQA, 2Wiki, MuSiQue
    code_idx = [6, 7]  # HumanEval, MBPP

    exec_cheap_col = AGENTS.index("executor_cheap")
    decomposer_col = AGENTS.index("decomposer")
    synthesizer_col = AGENTS.index("synthesizer")
    exec_strong_col = AGENTS.index("executor_strong")
    critic_col = AGENTS.index("critic")

    simple_exec = np.mean(matrix[simple_idx, exec_cheap_col])
    multihop_decomp = np.mean(matrix[multihop_idx, decomposer_col])
    multihop_synth = np.mean(matrix[multihop_idx, synthesizer_col])
    code_strong = np.mean(matrix[code_idx, exec_strong_col])
    code_critic = np.mean(matrix[code_idx, critic_col])

    print(f"  Simple QA -> exec_cheap:  {simple_exec:.3f}")
    print(f"  Multi-hop -> decomposer:  {multihop_decomp:.3f}")
    print(f"  Multi-hop -> synthesizer: {multihop_synth:.3f}")
    print(f"  Code -> exec_strong:      {code_strong:.3f}")
    print(f"  Code -> critic:           {code_critic:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Agent call distribution heatmap")
    parser.add_argument("--eval_json", nargs="+", required=True,
                        help="Path(s) to eval result JSON files")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each eval JSON (for comparison)")
    parser.add_argument("--output", type=str, default="figures/heatmap_agent_distribution.pdf")
    args = parser.parse_args()

    labels = args.labels or [Path(p).stem for p in args.eval_json]

    matrices = []
    for path in args.eval_json:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        matrix = build_distribution_matrix(data)
        matrices.append(matrix)

    if len(matrices) == 1:
        print_summary(matrices[0], labels[0])
        plot_single_heatmap(matrices[0], args.output, title=labels[0])
    else:
        for m, l in zip(matrices, labels):
            print_summary(m, l)
        plot_comparison_heatmap(matrices, labels, args.output)


if __name__ == "__main__":
    main()
