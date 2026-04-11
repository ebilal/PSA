"""
data_split.py — Query-grouped train/val splitter for selector training.

Groups all examples sharing a source_query_id into the same split to prevent
data leakage. Used by both CLI training and lifecycle retraining.
"""

import hashlib
import json
import logging
import random
from collections import defaultdict
from typing import Dict

logger = logging.getLogger("psa.training.data_split")


def _group_key(example: dict) -> str:
    """Stable group key: source_query_id if present, else hash of query."""
    qid = example.get("source_query_id")
    if qid:
        return str(qid)
    query = example.get("query", "")
    return hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()[:12]


def split_train_val(
    examples_path: str,
    train_path: str,
    val_path: str,
    val_fraction: float = 0.15,
    seed: int = 42,
    min_val_queries: int = 10,
) -> Dict[str, float]:
    """
    Split training examples into train and val by query group.

    All examples sharing a source_query_id (or hash of query if absent) land
    in the same split. Deterministic via fixed seed.

    Parameters
    ----------
    examples_path: Input JSONL with all generated training examples.
    train_path: Output JSONL for training set.
    val_path: Output JSONL for validation set.
    val_fraction: Target fraction of examples in val (by query group count).
    seed: Random seed for reproducibility.
    min_val_queries: Safety floor — at least this many query groups in val.

    Returns
    -------
    Stats dict with n_train_queries, n_val_queries, n_train_examples,
    n_val_examples, train_positive_rate, val_positive_rate.
    """
    # Load all examples
    examples = []
    with open(examples_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    # Group by query
    groups: Dict[str, list] = defaultdict(list)
    for ex in examples:
        groups[_group_key(ex)].append(ex)

    # Shuffle groups deterministically
    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    # Assign groups to val until we hit the target fraction
    total_examples = len(examples)
    target_val = max(
        int(total_examples * val_fraction),
        1,
    )
    target_val_queries = max(
        int(len(group_keys) * val_fraction),
        min_val_queries,
    )

    val_keys = set()
    val_count = 0
    for key in group_keys:
        if len(val_keys) >= target_val_queries and val_count >= target_val:
            break
        val_keys.add(key)
        val_count += len(groups[key])

    # Safety: ensure minimum val queries
    if len(val_keys) < min_val_queries:
        for key in group_keys:
            if key not in val_keys:
                val_keys.add(key)
                if len(val_keys) >= min_val_queries:
                    break

    # Write splits
    train_examples = []
    val_examples = []
    for key in group_keys:
        if key in val_keys:
            val_examples.extend(groups[key])
        else:
            train_examples.extend(groups[key])

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    # Compute stats
    def _pos_rate(exs):
        if not exs:
            return 0.0
        positives = sum(1 for ex in exs if ex.get("label") == 1)
        return positives / len(exs)

    stats = {
        "n_train_queries": len(group_keys) - len(val_keys),
        "n_val_queries": len(val_keys),
        "n_train_examples": len(train_examples),
        "n_val_examples": len(val_examples),
        "train_positive_rate": round(_pos_rate(train_examples), 4),
        "val_positive_rate": round(_pos_rate(val_examples), 4),
    }

    logger.info(
        "Train/val split: %d/%d queries, %d/%d examples, positive rate: train=%.1f%% val=%.1f%%",
        stats["n_train_queries"],
        stats["n_val_queries"],
        stats["n_train_examples"],
        stats["n_val_examples"],
        stats["train_positive_rate"] * 100,
        stats["val_positive_rate"] * 100,
    )

    return stats
