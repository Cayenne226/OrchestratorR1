"""
Recompute metrics from previously saved evaluation results.

Useful when metrics.py is updated (e.g., fixing code execution) without
re-running the expensive API calls.

Usage:
    python eval/recompute_metrics.py eval/results/direct_gpt4o_code.json
    python eval/recompute_metrics.py eval/results/direct_gpt4o_livecode.json
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metrics import compute_metric


def recompute(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    # Need to load original test data to get full records
    data_paths = data.get("config", {}).get("data_paths", [])

    # Build lookup from input prefix to full record
    full_records = {}
    for dp in data_paths:
        try:
            with open(dp, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        key = rec["input"][:200]
                        full_records[key] = rec
        except FileNotFoundError:
            print(f"[WARN] Cannot find {dp}, using saved record fields")

    # Recompute
    totals = {"em": 0, "f1": 0, "n": 0}
    source_stats = {}

    for r in results:
        pred = r["pred"]
        # Try to find full record from test data
        key = r.get("input", "")[:200]
        full_rec = full_records.get(key, r)

        metrics = compute_metric(pred, full_rec)
        r["metrics"] = metrics

        src = r.get("source", "unknown")
        if src not in source_stats:
            source_stats[src] = {"em": 0, "f1": 0, "n": 0, "extra": {}}
        s = source_stats[src]
        s["n"] += 1
        s["em"] += metrics.get("em", 0)
        s["f1"] += metrics.get("f1", 0)
        for k, v in metrics.items():
            if k not in ("em", "f1"):
                s["extra"].setdefault(k, 0)
                s["extra"][k] += v

        totals["em"] += metrics.get("em", 0)
        totals["f1"] += metrics.get("f1", 0)
        totals["n"] += 1

    n = totals["n"]
    data["summary"]["em"] = round(totals["em"] / n, 4) if n else 0
    data["summary"]["f1"] = round(totals["f1"] / n, 4) if n else 0

    # Update per_source
    per_source = {}
    for src, s in source_stats.items():
        nn = s["n"]
        per_source[src] = {
            "n": nn,
            "em": round(s["em"] / nn, 4) if nn else 0,
            "f1": round(s["f1"] / nn, 4) if nn else 0,
            "avg_cost": data.get("per_source", {}).get(src, {}).get("avg_cost", 0),
        }
        for k, v in s["extra"].items():
            per_source[src][k] = round(v / nn, 4) if nn else 0
    data["per_source"] = per_source

    # Print
    print(f"{'Source':<20} {'N':>5} {'EM':>8} {'F1':>8}")
    print("-" * 45)
    for src in sorted(source_stats):
        s = source_stats[src]
        nn = s["n"]
        print(f"  {src:<18} {nn:>5} {s['em']/nn:>8.4f} {s['f1']/nn:>8.4f}")
    print("-" * 45)
    print(f"  {'OVERALL':<18} {n:>5} {totals['em']/n:>8.4f} {totals['f1']/n:>8.4f}")

    # Save back
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nUpdated {path}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"\n=== {p} ===")
        recompute(p)
