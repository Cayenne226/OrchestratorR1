"""
Agent Call Distribution Heatmap (T4.1)

Generates a heatmap showing which agent types are invoked for each dataset,
proving the orchestrator learns task-dependent routing strategies.

The 4-role pool (executor / decomposer / critic / synthesizer) is visualized
with the executor split into its weak/strong tiers to surface tier-selection
behavior alongside role-selection behavior (5 columns total).

Usage:
    python analysis/agent_distribution.py \
        --eval_json eval/results/orchestrator_r1.json \
        --output figures/heatmap_agent_distribution.pdf

    # Side-by-side comparison (e.g. base vs trained):
    python analysis/agent_distribution.py \
        --eval_json eval/results/orch_base.json eval/results/orch_grpo.json \
        --labels "Base" "GRPO-trained" \
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


# ── Data sources and agent columns ────────────────────────────────────────────

SOURCES = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihop", "musique",
           "humaneval", "mbpp"]
SOURCE_LABELS = ["NQ", "TriviaQA", "PopQA", "HotpotQA", "2Wiki", "MuSiQue",
                 "HumanEval", "MBPP"]

# Each column is (agent_type, tier_or_None) — tier filter only applies to executor.
AGENT_COLUMNS: list[tuple[str, str | None]] = [
    ("executor",    "weak"),
    ("executor",    "strong"),
    ("decomposer",  None),
    ("critic",      None),
    ("synthesizer", None),
]
AGENT_LABELS = ["Exec-Weak", "Exec-Strong", "Decomposer", "Critic", "Synthesizer"]


def _matches_column(call: dict, agent_type: str, tier: str | None) -> bool:
    if call.get("agent_type") != agent_type:
        return False
    if tier is None:
        return True
    return call.get("tier") == tier


def build_distribution_matrix(eval_data: dict) -> np.ndarray:
    """Build a (n_sources x n_columns) frequency matrix from eval results."""
    matrix = np.zeros((len(SOURCES), len(AGENT_COLUMNS)))

    for i, src in enumerate(SOURCES):
        src_results = [r for r in eval_data["results"] if r.get("source") == src]
        if not src_results:
            continue

        total_calls = 0
        col_counts = [0] * len(AGENT_COLUMNS)

        for r in src_results:
            calls = r.get("agent_calls", [])
            total_calls += len(calls)
            for c in calls:
                for j, (agent_type, tier) in enumerate(AGENT_COLUMNS):
                    if _matches_column(c, agent_type, tier):
                        col_counts[j] += 1
                        break

        if total_calls > 0:
            for j in range(len(AGENT_COLUMNS)):
                matrix[i][j] = col_counts[j] / total_calls

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
    ax.set_xlabel("Agent (executor split by tier)", fontsize=12)
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
    """Plot side-by-side heatmaps for multi-condition comparison."""
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
        ax.set_xlabel("Agent (executor split by tier)", fontsize=11)
        ax.set_ylabel("Dataset", fontsize=11)

    plt.suptitle("Agent Call Distribution Comparison",
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
        print(f"{a:>14}", end="")
    print()
    print("-" * (15 + 14 * len(AGENT_COLUMNS)))

    for i, src in enumerate(SOURCE_LABELS):
        print(f"{src:<15}", end="")
        for j in range(len(AGENT_COLUMNS)):
            print(f"{matrix[i][j]:>14.3f}", end="")
        print()

    # Highlight emergent patterns
    print("\nEmergent patterns:")
    simple_idx = [0, 1, 2]
    multihop_idx = [3, 4, 5]
    code_idx = [6, 7]

    exec_weak_col   = AGENT_LABELS.index("Exec-Weak")
    exec_strong_col = AGENT_LABELS.index("Exec-Strong")
    decomposer_col  = AGENT_LABELS.index("Decomposer")
    synthesizer_col = AGENT_LABELS.index("Synthesizer")
    critic_col      = AGENT_LABELS.index("Critic")

    simple_weak     = np.mean(matrix[simple_idx, exec_weak_col])
    multihop_decomp = np.mean(matrix[multihop_idx, decomposer_col])
    multihop_synth  = np.mean(matrix[multihop_idx, synthesizer_col])
    code_strong     = np.mean(matrix[code_idx, exec_strong_col])
    code_critic     = np.mean(matrix[code_idx, critic_col])

    print(f"  Simple QA -> exec(weak):    {simple_weak:.3f}")
    print(f"  Multi-hop -> decomposer:    {multihop_decomp:.3f}")
    print(f"  Multi-hop -> synthesizer:   {multihop_synth:.3f}")
    print(f"  Code -> exec(strong):       {code_strong:.3f}")
    print(f"  Code -> critic:             {code_critic:.3f}")


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
