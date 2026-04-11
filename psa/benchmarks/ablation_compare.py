"""
ablation_compare.py — Compare two benchmark result files side-by-side.

Usage:
    python -m psa.benchmarks.ablation_compare results_a.jsonl results_b.jsonl

Reports:
  - Per-config summary (anchor count distribution, avg F1)
  - Shortlist overlap (Jaccard similarity of selected_anchor_ids per query)
  - Questions where A hits gold but B misses (and vice versa)
"""

import json
import sys
from collections import Counter
from typing import Dict, List


def _load_results(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _anchor_count_dist(records: List[Dict]) -> Dict[int, int]:
    return dict(Counter(len(r.get("selected_anchor_ids", [])) for r in records))


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def _quick_f1(gold: str, pred: str) -> float:
    g = set(gold.lower().split())
    p = set(pred.lower().split())
    if not g or not p:
        return 0.0
    common = g & p
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def compare(path_a: str, path_b: str) -> None:
    records_a = _load_results(path_a)
    records_b = _load_results(path_b)

    if len(records_a) != len(records_b):
        print(f"WARNING: different record counts: {len(records_a)} vs {len(records_b)}")

    n = min(len(records_a), len(records_b))

    print(f"\n{'Metric':<30} {'A':>12} {'B':>12}")
    print("-" * 56)

    # Anchor count distributions
    dist_a = _anchor_count_dist(records_a)
    dist_b = _anchor_count_dist(records_b)
    print(f"{'Anchor distribution A':<30} {dict(sorted(dist_a.items()))}")
    print(f"{'Anchor distribution B':<30} {dict(sorted(dist_b.items()))}")

    # Average anchor count
    avg_a = sum(len(r.get("selected_anchor_ids", [])) for r in records_a) / max(len(records_a), 1)
    avg_b = sum(len(r.get("selected_anchor_ids", [])) for r in records_b) / max(len(records_b), 1)
    print(f"{'Avg anchors':<30} {avg_a:>12.2f} {avg_b:>12.2f}")

    # Shortlist overlap (Jaccard per query)
    jaccards = []
    for ra, rb in zip(records_a[:n], records_b[:n]):
        sa = set(ra.get("selected_anchor_ids", []))
        sb = set(rb.get("selected_anchor_ids", []))
        jaccards.append(_jaccard(sa, sb))

    avg_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0
    identical = sum(1 for j in jaccards if j == 1.0)
    print(f"{'Avg Jaccard overlap':<30} {avg_jaccard:>12.3f}")
    print(f"{'Identical selections':<30} {identical:>12}/{n}")

    # Per-question divergence summary
    a_only_better = 0
    b_only_better = 0
    for ra, rb in zip(records_a[:n], records_b[:n]):
        sa = set(ra.get("selected_anchor_ids", []))
        sb = set(rb.get("selected_anchor_ids", []))
        if sa != sb:
            gold = str(ra.get("answer_gold", ""))
            f1_a = _quick_f1(gold, str(ra.get("answer_generated", "")))
            f1_b = _quick_f1(gold, str(rb.get("answer_generated", "")))
            if f1_a > f1_b + 0.05:
                a_only_better += 1
            elif f1_b > f1_a + 0.05:
                b_only_better += 1

    print(f"{'A clearly better (F1+0.05)':<30} {a_only_better:>12}")
    print(f"{'B clearly better (F1+0.05)':<30} {b_only_better:>12}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python -m psa.benchmarks.ablation_compare <results_a.jsonl> <results_b.jsonl>"
        )
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
