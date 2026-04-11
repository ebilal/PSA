# Selector Training Fixes + Ablation Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the trained selector's single-anchor collapse by correcting train/val leakage, adding cardinality control (`min_k`, `rerank_only`), plumbing those knobs end-to-end, adding diagnostic metrics to the benchmark scorer, and enabling four ablation configurations.

**Architecture:** New `psa/training/data_split.py` provides a query-grouped splitter used by both CLI and lifecycle. `AnchorSelector` gains `min_k` and `rerank_only` parameters with clear precedence rules. Pipeline and benchmark CLI plumb `max_k`, `min_k`, `rerank_only` end-to-end. The benchmark `score()` function reports anchor-count distribution and gold-hit rate. A post-hoc comparison script computes cross-run shortlist overlap.

**Tech Stack:** Python 3.13, sentence-transformers CrossEncoder, existing `call_llm()` infrastructure.

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `psa/training/data_split.py` | Create | `split_train_val()` utility — query-grouped, deterministic, with safety fallbacks |
| `psa/selector.py` | Modify | Add `min_k`, `rerank_only` to `AnchorSelector`; update `_trained_select()` logic |
| `psa/pipeline.py` | Modify | Add `selector_max_k`, `selector_min_k`, `selector_rerank_only` to `from_tenant()` |
| `psa/benchmarks/longmemeval.py` | Modify | Add `max_k`, `min_k`, `rerank_only` params to `run()`; config label for filenames; anchor-count + gold-hit metrics in `score()` |
| `psa/cli.py` | Modify | Fix train/val split in `cmd_train()`; add `--max-k`, `--min-k`, `--rerank-only` to benchmark CLI |
| `psa/lifecycle.py` | Modify | Use `split_train_val()` in `_retrain_selector()` |
| `psa/benchmarks/ablation_compare.py` | Create | Post-hoc cross-run comparison script (shortlist overlap, side-by-side metrics) |
| `tests/test_data_split.py` | Create | Tests for `split_train_val()` |
| `tests/test_selector.py` | Modify | Add tests for `min_k` backfill, `rerank_only`, precedence |
| `tests/test_benchmarks.py` | Modify | Test new diagnostic metrics in `score()` |

---

### Task 1: Query-grouped train/val split utility

**Files:**
- Create: `psa/training/data_split.py`
- Create: `tests/test_data_split.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_data_split.py`:

```python
"""Tests for psa.training.data_split — query-grouped train/val splitter."""

import json
import hashlib

import pytest


def _make_example(query_id: str, query: str, label: int) -> dict:
    return {
        "query": query,
        "anchor_card": "some card text",
        "label": label,
        "anchor_id": 1,
        "query_family": "bridge",
        "example_type": "positive" if label == 1 else "hard_negative",
        "source_query_id": query_id,
    }


def _write_examples(path, examples):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def _read_examples(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_split_creates_two_files(tmp_path):
    """split_train_val writes train and val files."""
    from psa.training.data_split import split_train_val

    examples_path = str(tmp_path / "examples.jsonl")
    # 4 queries, 3 examples each = 12 examples
    examples = []
    for i in range(4):
        for j in range(3):
            examples.append(_make_example(f"q{i}", f"query {i}", j % 2))
    _write_examples(examples_path, examples)

    train_path = str(tmp_path / "train.jsonl")
    val_path = str(tmp_path / "val.jsonl")
    stats = split_train_val(examples_path, train_path, val_path, val_fraction=0.25, seed=42)

    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    assert stats["n_train_examples"] + stats["n_val_examples"] == 12
    assert stats["n_train_queries"] + stats["n_val_queries"] == 4


def test_split_is_query_grouped(tmp_path):
    """All examples for a given source_query_id land in the same split."""
    from psa.training.data_split import split_train_val

    examples_path = str(tmp_path / "examples.jsonl")
    examples = []
    for i in range(10):
        for j in range(5):
            examples.append(_make_example(f"q{i}", f"query {i}", j % 2))
    _write_examples(examples_path, examples)

    train_path = str(tmp_path / "train.jsonl")
    val_path = str(tmp_path / "val.jsonl")
    split_train_val(examples_path, train_path, val_path, val_fraction=0.15, seed=42)

    train = _read_examples(train_path)
    val = _read_examples(val_path)
    train_qids = {ex["source_query_id"] for ex in train}
    val_qids = {ex["source_query_id"] for ex in val}
    assert train_qids & val_qids == set(), "Query IDs must not overlap between train and val"


def test_split_deterministic(tmp_path):
    """Same seed produces same split."""
    from psa.training.data_split import split_train_val

    examples_path = str(tmp_path / "examples.jsonl")
    examples = [_make_example(f"q{i}", f"query {i}", i % 2) for i in range(20)]
    _write_examples(examples_path, examples)

    stats1 = split_train_val(
        examples_path, str(tmp_path / "t1.jsonl"), str(tmp_path / "v1.jsonl"), seed=42
    )
    stats2 = split_train_val(
        examples_path, str(tmp_path / "t2.jsonl"), str(tmp_path / "v2.jsonl"), seed=42
    )
    assert _read_examples(str(tmp_path / "v1.jsonl")) == _read_examples(str(tmp_path / "v2.jsonl"))


def test_split_fallback_when_no_query_id(tmp_path):
    """Examples without source_query_id use hash of query as group key."""
    from psa.training.data_split import split_train_val

    examples_path = str(tmp_path / "examples.jsonl")
    examples = []
    for i in range(6):
        ex = _make_example(None, f"query {i}", i % 2)
        del ex["source_query_id"]  # missing key
        examples.append(ex)
    _write_examples(examples_path, examples)

    train_path = str(tmp_path / "train.jsonl")
    val_path = str(tmp_path / "val.jsonl")
    stats = split_train_val(examples_path, train_path, val_path, val_fraction=0.3, seed=42)

    assert stats["n_train_examples"] + stats["n_val_examples"] == 6


def test_split_min_val_queries_safety(tmp_path):
    """When val_fraction yields too few queries, fall back to min_val_queries."""
    from psa.training.data_split import split_train_val

    examples_path = str(tmp_path / "examples.jsonl")
    # 3 queries — 15% of 3 is < 1, so safety fallback should kick in
    examples = [_make_example(f"q{i}", f"query {i}", i % 2) for i in range(3)]
    _write_examples(examples_path, examples)

    train_path = str(tmp_path / "train.jsonl")
    val_path = str(tmp_path / "val.jsonl")
    stats = split_train_val(
        examples_path, train_path, val_path, val_fraction=0.15, seed=42, min_val_queries=1
    )

    assert stats["n_val_queries"] >= 1


def test_split_reports_positive_rates(tmp_path):
    """Stats dict includes positive rate for both splits."""
    from psa.training.data_split import split_train_val

    examples_path = str(tmp_path / "examples.jsonl")
    examples = []
    for i in range(10):
        examples.append(_make_example(f"q{i}", f"query {i}", 1))  # all positive
        examples.append(_make_example(f"q{i}", f"query {i}", 0))  # all negative
    _write_examples(examples_path, examples)

    train_path = str(tmp_path / "train.jsonl")
    val_path = str(tmp_path / "val.jsonl")
    stats = split_train_val(examples_path, train_path, val_path, val_fraction=0.2, seed=42)

    assert "train_positive_rate" in stats
    assert "val_positive_rate" in stats
    assert 0.0 <= stats["train_positive_rate"] <= 1.0
    assert 0.0 <= stats["val_positive_rate"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data_split.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'psa.training.data_split'`

- [ ] **Step 3: Implement `split_train_val()`**

Create `psa/training/data_split.py`:

```python
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
from typing import Dict, Optional

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
        1,  # at least 1 example
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
        "Train/val split: %d/%d queries, %d/%d examples, "
        "positive rate: train=%.1f%% val=%.1f%%",
        stats["n_train_queries"],
        stats["n_val_queries"],
        stats["n_train_examples"],
        stats["n_val_examples"],
        stats["train_positive_rate"] * 100,
        stats["val_positive_rate"] * 100,
    )

    return stats
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_data_split.py -v`
Expected: all 6 PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check psa/training/data_split.py tests/test_data_split.py
uv run ruff format psa/training/data_split.py tests/test_data_split.py
git add psa/training/data_split.py tests/test_data_split.py
git commit -m "feat: add query-grouped train/val splitter (psa/training/data_split.py)"
```

---

### Task 2: Fix CLI and lifecycle to use query-grouped split

**Files:**
- Modify: `psa/cli.py:540-570`
- Modify: `psa/lifecycle.py:493-512`

- [ ] **Step 1: Fix `cmd_train()` in `psa/cli.py`**

Find the current split logic (around line 546-567):

```python
    # Generate training data
    training_path = os.path.join(tenant.root_dir, "training", "training_data.jsonl")
    anchor_cards = {c.anchor_id: c.to_stable_card_text() for c in atlas.cards}
    gen = DataGenerator(oracle_labels_path=labels_path, anchor_cards=anchor_cards)
    n_written = gen.generate(output_path=training_path, n_examples=max(1000, label_count * 20))
    print(f"  Generated {n_written} training examples.")

    # Split off 15% as val set for threshold calibration
    import random as _random

    val_path = os.path.join(tenant.root_dir, "training", "val_data.jsonl")
    with open(training_path) as fh:
        all_lines = [ln for ln in fh if ln.strip()]
    _random.shuffle(all_lines)
    val_size = max(50, int(0.15 * len(all_lines)))
    with open(val_path, "w") as fh:
        fh.writelines(all_lines[:val_size])

    # Train
    output_dir = os.path.join(tenant.root_dir, "models", "selector_latest")
    trainer = SelectorTrainer(output_dir=output_dir, atlas_version=atlas.version)
    try:
        sv = trainer.train(train_data_path=training_path, val_data_path=val_path)
```

Replace with:

```python
    # Generate training data
    examples_path = os.path.join(tenant.root_dir, "training", "training_data.jsonl")
    anchor_cards = {c.anchor_id: c.to_stable_card_text() for c in atlas.cards}
    gen = DataGenerator(oracle_labels_path=labels_path, anchor_cards=anchor_cards)
    n_written = gen.generate(output_path=examples_path, n_examples=max(1000, label_count * 20))
    print(f"  Generated {n_written} training examples.")

    # Query-grouped train/val split (no leakage)
    from .training.data_split import split_train_val

    train_path = os.path.join(tenant.root_dir, "training", "train_data.jsonl")
    val_path = os.path.join(tenant.root_dir, "training", "val_data.jsonl")
    split_stats = split_train_val(examples_path, train_path, val_path)
    print(
        f"  Split: {split_stats['n_train_queries']}/{split_stats['n_val_queries']} queries, "
        f"positive rate: train={split_stats['train_positive_rate']:.1%} "
        f"val={split_stats['val_positive_rate']:.1%}"
    )

    # Train
    output_dir = os.path.join(tenant.root_dir, "models", "selector_latest")
    trainer = SelectorTrainer(output_dir=output_dir, atlas_version=atlas.version)
    try:
        sv = trainer.train(train_data_path=train_path, val_data_path=val_path)
```

Also remove the `import random as _random` line that is no longer needed.

- [ ] **Step 2: Fix `_retrain_selector()` in `psa/lifecycle.py`**

Find the training section (around line 493-512):

```python
        # Generate training data from existing labels
        training_path = os.path.join(tenant.root_dir, "training", "training_data.jsonl")
        anchor_cards = {c.anchor_id: c.to_stable_card_text() for c in atlas.cards}
        gen = DataGenerator(oracle_labels_path=labels_path, anchor_cards=anchor_cards)
        n_written = gen.generate(output_path=training_path, n_examples=max(1000, label_count * 20))

        if n_written < 100:
            logger.info("Only %d training examples generated. Staying in cosine mode.", n_written)
            return False

        # Train
        selector_version = state.get("selector_version", 0) + 1
        output_dir = os.path.join(tenant.root_dir, "models", f"selector_v{selector_version}")
        trainer = SelectorTrainer(
            output_dir=output_dir,
            atlas_version=atlas.version,
        )

        try:
            sv = trainer.train(train_data_path=training_path, version=selector_version)
```

Replace with:

```python
        # Generate training data from existing labels
        examples_path = os.path.join(tenant.root_dir, "training", "training_data.jsonl")
        anchor_cards = {c.anchor_id: c.to_stable_card_text() for c in atlas.cards}
        gen = DataGenerator(oracle_labels_path=labels_path, anchor_cards=anchor_cards)
        n_written = gen.generate(output_path=examples_path, n_examples=max(1000, label_count * 20))

        if n_written < 100:
            logger.info("Only %d training examples generated. Staying in cosine mode.", n_written)
            return False

        # Query-grouped train/val split (no leakage)
        from .training.data_split import split_train_val

        train_path = os.path.join(tenant.root_dir, "training", "train_data.jsonl")
        val_path = os.path.join(tenant.root_dir, "training", "val_data.jsonl")
        split_stats = split_train_val(examples_path, train_path, val_path)
        logger.info(
            "Train/val split: %d/%d queries, pos rate: train=%.1f%% val=%.1f%%",
            split_stats["n_train_queries"],
            split_stats["n_val_queries"],
            split_stats["train_positive_rate"] * 100,
            split_stats["val_positive_rate"] * 100,
        )

        # Train
        selector_version = state.get("selector_version", 0) + 1
        output_dir = os.path.join(tenant.root_dir, "models", f"selector_v{selector_version}")
        trainer = SelectorTrainer(
            output_dir=output_dir,
            atlas_version=atlas.version,
        )

        try:
            sv = trainer.train(train_data_path=train_path, val_data_path=val_path, version=selector_version)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/ -v -q`
Expected: all PASS (existing tests should not break).

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff check psa/cli.py psa/lifecycle.py
uv run ruff format psa/cli.py psa/lifecycle.py
git add psa/cli.py psa/lifecycle.py
git commit -m "fix: use query-grouped train/val split in CLI and lifecycle (fixes val leakage)"
```

---

### Task 3: Add `min_k` and `rerank_only` to AnchorSelector

**Files:**
- Modify: `psa/selector.py`
- Modify: `tests/test_selector.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_selector.py`:

```python
def test_trained_select_min_k_backfill():
    """When threshold filters below min_k, backfill from top-scored items."""
    candidates = [_make_candidate(i, dense_score=0.9 - i * 0.1) for i in range(6)]

    mock_ce = MagicMock()
    # Scores: only anchor 0 passes threshold=0.5, but min_k=3 forces top-3
    mock_ce.predict.return_value = [0.8, 0.3, 0.2, 0.1, 0.05, 0.01]

    result = _trained_select(
        query="test query",
        candidates=candidates,
        cross_encoder=mock_ce,
        max_k=6,
        threshold=0.5,
        min_k=3,
    )
    assert len(result) == 3
    assert result[0].anchor_id == 0
    assert result[0].selector_score == 0.8


def test_trained_select_rerank_only_returns_max_k():
    """rerank_only=True ignores threshold, returns exactly max_k."""
    candidates = [_make_candidate(i, dense_score=0.9 - i * 0.1) for i in range(8)]

    mock_ce = MagicMock()
    # All scores below any reasonable threshold
    mock_ce.predict.return_value = [0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007]

    result = _trained_select(
        query="test query",
        candidates=candidates,
        cross_encoder=mock_ce,
        max_k=6,
        threshold=0.5,
        rerank_only=True,
    )
    assert len(result) == 6


def test_trained_select_rerank_only_beats_min_k_precedence():
    """rerank_only takes precedence — min_k is ignored."""
    candidates = [_make_candidate(i, dense_score=0.9 - i * 0.1) for i in range(8)]

    mock_ce = MagicMock()
    mock_ce.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

    result = _trained_select(
        query="test query",
        candidates=candidates,
        cross_encoder=mock_ce,
        max_k=6,
        threshold=0.5,
        min_k=3,
        rerank_only=True,
    )
    # rerank_only → returns max_k regardless of min_k or threshold
    assert len(result) == 6


def test_trained_select_min_k_capped_at_candidate_count():
    """min_k never returns more than available candidates."""
    candidates = [_make_candidate(i, dense_score=0.9 - i * 0.1) for i in range(2)]

    mock_ce = MagicMock()
    mock_ce.predict.return_value = [0.1, 0.05]

    result = _trained_select(
        query="test query",
        candidates=candidates,
        cross_encoder=mock_ce,
        max_k=6,
        threshold=0.5,
        min_k=4,  # more than candidates available
    )
    assert len(result) == 2


def test_selector_init_min_k_and_rerank_only():
    """AnchorSelector accepts min_k and rerank_only."""
    sel = AnchorSelector(mode="cosine", min_k=3, rerank_only=False)
    assert sel.min_k == 3
    assert sel.rerank_only is False

    sel2 = AnchorSelector(mode="cosine", rerank_only=True)
    assert sel2.rerank_only is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_selector.py -v -k "min_k or rerank"`
Expected: FAIL — `_trained_select()` and `AnchorSelector` don't accept `min_k`/`rerank_only` yet.

- [ ] **Step 3: Update `_trained_select()` in `psa/selector.py`**

Change the signature from:

```python
def _trained_select(
    query: str,
    candidates: List[AnchorCandidate],
    cross_encoder,
    max_k: int,
    threshold: float,
) -> List[SelectedAnchor]:
```

To:

```python
def _trained_select(
    query: str,
    candidates: List[AnchorCandidate],
    cross_encoder,
    max_k: int,
    threshold: float,
    min_k: Optional[int] = None,
    rerank_only: bool = False,
) -> List[SelectedAnchor]:
```

Replace the selection logic (lines 138-171) with:

```python
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    if rerank_only:
        # Pure reranking: ignore threshold and min_k, return top max_k
        selected = [
            SelectedAnchor(
                anchor_id=cand.anchor_id,
                selector_score=float(score),
                mode="trained",
                candidate=cand,
            )
            for cand, score in scored[:max_k]
        ]
    else:
        # Threshold filtering
        selected = []
        for cand, score in scored[:max_k]:
            if float(score) < threshold:
                break
            selected.append(
                SelectedAnchor(
                    anchor_id=cand.anchor_id,
                    selector_score=float(score),
                    mode="trained",
                    candidate=cand,
                )
            )

        # min_k backfill: if threshold filtered too aggressively, take top min_k
        if min_k is not None and len(selected) < min_k:
            backfill_count = min(min_k, len(scored))
            selected = [
                SelectedAnchor(
                    anchor_id=cand.anchor_id,
                    selector_score=float(score),
                    mode="trained",
                    candidate=cand,
                )
                for cand, score in scored[:backfill_count]
            ]

        # Zero-anchor fallback (existing behavior): always return at least top-1
        if not selected and scored:
            top_cand, top_score = scored[0]
            logger.debug(
                "Trained selector threshold rejected all candidates; "
                "returning top-1 fallback (score=%.4f)",
                float(top_score),
            )
            selected = [
                SelectedAnchor(
                    anchor_id=top_cand.anchor_id,
                    selector_score=float(top_score),
                    mode="trained",
                    candidate=top_cand,
                )
            ]

    return selected
```

- [ ] **Step 4: Update `AnchorSelector` class**

Change `__init__` to accept the new parameters:

```python
    def __init__(
        self,
        mode: str = "cosine",
        model_path: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD,
        max_k: int = DEFAULT_MAX_K,
        min_k: Optional[int] = None,
        rerank_only: bool = False,
    ):
```

Store them:

```python
        self.min_k = min_k
        self.rerank_only = rerank_only
```

Update `select()` to pass them to `_trained_select()`:

```python
        if self.mode == "trained" and self._cross_encoder is not None:
            return _trained_select(
                query=query,
                candidates=candidates,
                cross_encoder=self._cross_encoder,
                max_k=self.max_k,
                threshold=self.threshold,
                min_k=self.min_k,
                rerank_only=self.rerank_only,
            )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_selector.py -v`
Expected: all PASS.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check psa/selector.py tests/test_selector.py
uv run ruff format psa/selector.py tests/test_selector.py
git add psa/selector.py tests/test_selector.py
git commit -m "feat: add min_k and rerank_only to AnchorSelector for cardinality control"
```

---

### Task 4: Pipeline plumbing (`from_tenant`, `run()`, CLI flags)

**Files:**
- Modify: `psa/pipeline.py:383-447`
- Modify: `psa/benchmarks/longmemeval.py:130-183`
- Modify: `psa/cli.py:821-845` and `psa/cli.py:1362-1375`

- [ ] **Step 1: Add parameters to `PSAPipeline.from_tenant()`**

In `psa/pipeline.py`, change the `from_tenant()` signature from:

```python
    @classmethod
    def from_tenant(
        cls,
        tenant_id: str = "default",
        token_budget: int = 6000,
        selector_mode: str = "cosine",
        selector_model_path: Optional[str] = None,
        psa_mode: str = "side-by-side",
        base_dir: Optional[str] = None,
    ) -> "PSAPipeline":
```

To:

```python
    @classmethod
    def from_tenant(
        cls,
        tenant_id: str = "default",
        token_budget: int = 6000,
        selector_mode: str = "cosine",
        selector_model_path: Optional[str] = None,
        selector_max_k: int = 6,
        selector_min_k: Optional[int] = None,
        selector_rerank_only: bool = False,
        psa_mode: str = "side-by-side",
        base_dir: Optional[str] = None,
    ) -> "PSAPipeline":
```

Update the `selector_kwargs` construction (around line 434):

```python
        selector_kwargs = dict(
            mode=selector_mode,
            model_path=selector_model_path,
            max_k=selector_max_k,
            min_k=selector_min_k,
            rerank_only=selector_rerank_only,
        )
        if selector_threshold is not None:
            selector_kwargs["threshold"] = selector_threshold
        selector = AnchorSelector(**selector_kwargs)
```

- [ ] **Step 2: Add parameters to `longmemeval.run()` with config label**

In `psa/benchmarks/longmemeval.py`, change the `run()` signature to:

```python
def run(
    split: str = "val",
    limit: Optional[int] = None,
    tenant_id: str = BENCH_TENANT,
    results_dir: str = RESULTS_DIR_DEFAULT,
    token_budget: int = 6000,
    selector_mode: str = "cosine",
    selector_model_path: Optional[str] = None,
    max_k: int = 6,
    min_k: Optional[int] = None,
    rerank_only: bool = False,
) -> str:
```

Update the `PSAPipeline.from_tenant()` call:

```python
        pipeline = PSAPipeline.from_tenant(
            tenant_id=tenant_id,
            token_budget=token_budget,
            selector_mode=selector_mode,
            selector_model_path=selector_model_path,
            selector_max_k=max_k,
            selector_min_k=min_k,
            selector_rerank_only=rerank_only,
        )
```

Build a config label and use it for the filename:

```python
    # Build deterministic config label for filename
    label_parts = [selector_mode]
    if rerank_only:
        label_parts.append("rerank")
    elif min_k is not None:
        label_parts.append(f"min{min_k}")
    label_parts.append(f"k{max_k}")
    config_label = "_".join(label_parts)

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(results_dir, f"results_{split}_{config_label}_{ts}.jsonl")
```

Also update the existing `out_path` line that uses `selector_mode` only — replace it with the new `config_label` version above.

- [ ] **Step 3: Add CLI flags**

In `psa/cli.py`, add to the `p_lme_run` argparse section (after `--selector-model`):

```python
    p_lme_run.add_argument(
        "--max-k",
        type=int,
        default=6,
        help="Maximum anchors to select (default: 6)",
    )
    p_lme_run.add_argument(
        "--min-k",
        type=int,
        default=None,
        help="Minimum anchors to select (backfill from top-scored if threshold filters too many)",
    )
    p_lme_run.add_argument(
        "--rerank-only",
        action="store_true",
        default=False,
        help="Ignore threshold, return top max-k reranked by cross-encoder",
    )
```

Update the `cmd_benchmark()` handler for `lme_action == "run"` to read and pass these:

```python
        max_k = getattr(args, "max_k", 6)
        min_k = getattr(args, "min_k", None)
        rerank_only = getattr(args, "rerank_only", False)
        print(
            f"Running LongMemEval ({split} split, {'all' if not limit else limit} questions, "
            f"selector={selector_mode}, max_k={max_k}"
            f"{f', min_k={min_k}' if min_k else ''}"
            f"{', rerank_only' if rerank_only else ''})..."
        )
        out_path = run(
            split=split,
            limit=limit,
            tenant_id=tenant_id,
            selector_mode=selector_mode,
            selector_model_path=selector_model_path,
            max_k=max_k,
            min_k=min_k,
            rerank_only=rerank_only,
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_pipeline.py tests/test_benchmarks.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check psa/pipeline.py psa/benchmarks/longmemeval.py psa/cli.py
uv run ruff format psa/pipeline.py psa/benchmarks/longmemeval.py psa/cli.py
git add psa/pipeline.py psa/benchmarks/longmemeval.py psa/cli.py
git commit -m "feat: plumb max_k, min_k, rerank_only end-to-end (pipeline, benchmark, CLI)"
```

---

### Task 5: Diagnostic metrics in `score()` — anchor-count distribution and gold-hit rate

**Files:**
- Modify: `psa/benchmarks/longmemeval.py` (the `score()` function)
- Modify: `tests/test_benchmarks.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_benchmarks.py`:

```python
def test_score_reports_anchor_count_distribution():
    """score() returns anchor_count_distribution in results."""
    from psa.benchmarks.longmemeval import score, _load_results
    from unittest.mock import patch, MagicMock

    results_path = str(tmp_path / "results.jsonl")
    with open(results_path, "w") as f:
        for i in range(5):
            r = _make_result_record(question_id=f"q{i}")
            r["selected_anchor_ids"] = list(range(i + 1))  # 1, 2, 3, 4, 5 anchors
            f.write(json.dumps(r) + "\n")

    mock_pipeline = MagicMock()
    mock_pipeline.store = MagicMock()
    mock_pipeline.atlas = MagicMock()

    with patch("psa.benchmarks.longmemeval.PSAPipeline.from_tenant", return_value=mock_pipeline), \
         patch("psa.benchmarks.longmemeval.backtrack_gold_anchors", return_value=[]), \
         patch("psa.benchmarks.longmemeval._oracle_labels_path", return_value=str(tmp_path / "labels.jsonl")):
        result = score(results_path, method="exact", tenant_id="test_tenant")

    assert "anchor_count_distribution" in result
    dist = result["anchor_count_distribution"]
    assert dist[1] == 1  # 1 question with 1 anchor
    assert dist[5] == 1  # 1 question with 5 anchors
```

Note: this test uses `tmp_path` — make it a parametrized test function that takes `tmp_path` as a fixture.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmarks.py::test_score_reports_anchor_count_distribution -v`
Expected: FAIL — `anchor_count_distribution` key not in result.

- [ ] **Step 3: Add metrics to `score()`**

In the `score()` function in `psa/benchmarks/longmemeval.py`, add after the existing `recall_scores` loop:

```python
    # Anchor count distribution
    anchor_count_dist: Dict[int, int] = {}
    for r in records:
        n = len(r.get("selected_anchor_ids", []))
        anchor_count_dist[n] = anchor_count_dist.get(n, 0) + 1

    # Gold-hit rate (fraction of questions where selected anchors include a gold anchor)
    gold_hit_count = sum(1 for s in recall_scores if s == 1.0)
    gold_hit_rate = gold_hit_count / len(recall_scores) if recall_scores else None
```

Add to the `result` dict before the return:

```python
    result["anchor_count_distribution"] = anchor_count_dist
    if gold_hit_rate is not None:
        result["gold_hit_rate"] = round(gold_hit_rate, 4)
```

Update the print output in `psa/cli.py`'s `cmd_benchmark()` for the score action — add after the existing prints:

```python
        if "anchor_count_distribution" in scores:
            dist = scores["anchor_count_distribution"]
            print(f"  Anchor distribution: {dict(sorted(dist.items()))}")
        if "gold_hit_rate" in scores:
            print(f"  Gold-hit rate: {scores['gold_hit_rate']:.3f}")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_benchmarks.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check psa/benchmarks/longmemeval.py psa/cli.py tests/test_benchmarks.py
uv run ruff format psa/benchmarks/longmemeval.py psa/cli.py tests/test_benchmarks.py
git add psa/benchmarks/longmemeval.py psa/cli.py tests/test_benchmarks.py
git commit -m "feat: add anchor-count distribution and gold-hit rate to benchmark score()"
```

---

### Task 6: Post-hoc ablation comparison script

**Files:**
- Create: `psa/benchmarks/ablation_compare.py`

- [ ] **Step 1: Write the script**

Create `psa/benchmarks/ablation_compare.py`:

```python
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
            # Quick F1 comparison
            gold = str(ra.get("answer_gold", ""))
            f1_a = _quick_f1(gold, str(ra.get("answer_generated", "")))
            f1_b = _quick_f1(gold, str(rb.get("answer_generated", "")))
            if f1_a > f1_b + 0.05:
                a_only_better += 1
            elif f1_b > f1_a + 0.05:
                b_only_better += 1

    print(f"{'A clearly better (F1+0.05)':<30} {a_only_better:>12}")
    print(f"{'B clearly better (F1+0.05)':<30} {b_only_better:>12}")


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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m psa.benchmarks.ablation_compare <results_a.jsonl> <results_b.jsonl>")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
```

- [ ] **Step 2: Lint and commit**

```bash
uv run ruff check psa/benchmarks/ablation_compare.py
uv run ruff format psa/benchmarks/ablation_compare.py
git add psa/benchmarks/ablation_compare.py
git commit -m "feat: add ablation comparison script for cross-run shortlist overlap analysis"
```

---

### Task 7: Full test suite and final lint

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: all PASS (except the pre-existing `test_training_dependencies_importable` if training extras are not installed).

- [ ] **Step 2: Run full lint**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: clean.

- [ ] **Step 3: Final commit if any formatting changes**

```bash
git add -u
git commit -m "style: final lint pass for selector training fixes"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Query-grouped train/val split utility | Task 1 |
| Fix CLI `cmd_train()` split leakage | Task 2 |
| Fix lifecycle `_retrain_selector()` split | Task 2 |
| `min_k` on AnchorSelector | Task 3 |
| `rerank_only` on AnchorSelector | Task 3 |
| Precedence: `rerank_only` > `threshold + min_k` | Task 3 |
| `selector_max_k` in `from_tenant()` | Task 4 |
| `selector_min_k` in `from_tenant()` | Task 4 |
| `selector_rerank_only` in `from_tenant()` | Task 4 |
| Config label for benchmark filenames (includes max_k) | Task 4 |
| `--max-k`, `--min-k`, `--rerank-only` CLI flags | Task 4 |
| Anchor count distribution metric | Task 5 |
| Gold-hit rate metric | Task 5 |
| Shortlist overlap cross-run comparison | Task 6 |
| Ablation run commands (no code — uses flags from Task 4) | Documented in spec, not a task |

**Placeholder scan:** No TBD/TODO found. All code blocks are complete.

**Type consistency:** `min_k: Optional[int] = None` and `rerank_only: bool = False` are consistent across `_trained_select()`, `AnchorSelector.__init__()`, `PSAPipeline.from_tenant()`, `longmemeval.run()`, and CLI args.
