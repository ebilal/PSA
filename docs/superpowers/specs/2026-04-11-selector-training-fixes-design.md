# Selector Training Fixes + Ablation Harness — Design Spec

## Problem

The trained cross-encoder selector collapses 62% of queries to a single anchor (309/500), destroying retrieval coverage. Cosine selector (always k=6) achieves R@5=0.856 while trained achieves R@5=0.754. The trained selector's ranking signal may be useful, but cardinality control is broken and training foundations have correctness bugs.

### Root causes (confirmed)

1. **Val/train split leakage.** `cli.py` writes val as a random 15% of `training_data.jsonl`, then passes the full file (including val rows) to `trainer.train()`. Threshold τ is calibrated on data the model already saw. The split is line-level, not query-grouped, so the same `source_query_id` appears in both sets via oversampled/adversarial examples.

2. **Pair classifier → set construction mismatch.** The cross-encoder is trained as a binary pair scorer `(query, anchor) → good/bad`, but inference applies a hard threshold and `break`s at the first score drop. This naturally favors one confident anchor over a useful set of 3–6.

3. **Pair-level threshold calibration.** `_compute_threshold()` maximizes F-beta over pooled (score, label) pairs. A steep per-query score distribution means one anchor dominates, producing high pairwise recall that masks per-query set collapse to k=1.

4. **No validation split in lifecycle retraining.** `lifecycle.py` trains on the full dataset with no holdout. The automated retraining path has no threshold calibration at all.

5. **Query pattern overfitting risk.** Cards now include `generated_query_patterns` in `to_stable_card_text()`. The cross-encoder may be overfitting to one strong lexical match between query text and a pattern, scoring everything else low.

### Evidence

| Metric | Cosine (k=6) | Trained (τ=0.05) |
|--------|-------------|-----------------|
| R@5 | 0.856 | 0.754 |
| Exact F1 | 0.195 | 0.169 |
| LLM judge | 0.312 | 0.286 |
| Anchors=1 | 0/500 | 309/500 |

When trained selects 1 anchor and misses gold, cosine had the gold anchor 49% of the time (56/114 cases). F1 scales linearly with anchor count: 1 anchor → F1=0.098, 6 anchors → F1=0.222. In 78% of queries, trained's selection is a strict subset of cosine's.

## Goal

Fix training data leakage, add cardinality control to the selector, and run four diagnostic ablation configurations to determine whether the cross-encoder ranking signal is worth keeping. The ablation results inform whether to invest in training objective rework (listwise loss) or retire the CE path.

## Design

### 1. Query-grouped train/val split

**New file: `psa/training/data_split.py`**

A pure, reusable utility:

```python
def split_train_val(
    examples_path: str,
    train_path: str,
    val_path: str,
    val_fraction: float = 0.15,
    seed: int = 42,
    min_val_queries: int = 10,
) -> dict:
```

**Behavior:**

- Reads examples from `examples_path` (JSONL, one dict per line).
- Groups by `source_query_id`. Fallback when absent: stable hash of the `query` field (e.g., `hashlib.md5(query.encode()).hexdigest()[:12]`). This avoids large in-memory keys and keeps logs clean.
- Shuffles query groups deterministically (seed=42).
- Assigns entire groups to val until `val_fraction` of total examples is reached.
- Safety: if val ends up with fewer than `min_val_queries` unique queries or zero positive examples, falls back to a capped minimum holdout by query count and logs a clear warning.
- Writes `train_path` and `val_path` as separate JSONL files. The original `examples_path` is left untouched.
- Returns stats dict: `{n_train_queries, n_val_queries, n_train_examples, n_val_examples, train_positive_rate, val_positive_rate}`.
- Logs group-level stats by `source_query_id` so leakage detection is obvious.

**Callers:**

- `cli.py:cmd_train()`: after `DataGenerator.generate(output_path=examples_path)`, calls `split_train_val(examples_path, train_path, val_path)`, then passes `train_data_path=train_path` and `val_data_path=val_path` to `trainer.train()`. No more rewriting files in place.
- `lifecycle.py`: same pattern. Currently has no validation split at all — this is not optional.

Both rely on the existing `trainer.train(train_data_path, val_data_path)` API. Callers do not call `_evaluate()` or `_compute_threshold()` directly.

### 2. `min_k` and `rerank_only` on AnchorSelector

**Modified file: `psa/selector.py`**

Two new parameters on `AnchorSelector.__init__()`:

- `min_k: Optional[int] = None` — after threshold filtering, if fewer than `min_k` anchors pass, backfill from top-scored items regardless of threshold.
- `rerank_only: bool = False` — ignore threshold and `min_k` entirely, return the top `max_k` candidates reranked by cross-encoder score.

**Precedence:** `rerank_only` > `threshold + min_k`. When `rerank_only=True`, the other two parameters have no effect.

**Selection logic in `_trained_select()`:**

```
scored = sorted(zip(candidates, scores), key=score, reverse=True)
if rerank_only:
    selected = scored[:max_k]
else:
    selected = [c for c in scored[:max_k] if score >= threshold]
    if min_k is not None and len(selected) < min_k:
        selected = scored[:min(min_k, len(candidates))]
```

**Tests:**

- Threshold selects fewer than `min_k` → backfill occurs, returns `min_k` items.
- `rerank_only=True` → returns exactly `max_k` items (or all candidates if fewer).
- `min_k` never exceeds candidate count.
- `rerank_only` takes precedence over threshold and `min_k`.

### 3. Pipeline plumbing

**Modified files: `psa/pipeline.py`, `psa/benchmarks/longmemeval.py`, `psa/cli.py`**

These are generic selector configuration knobs, not ablation-specific wiring.

**`PSAPipeline.from_tenant()`** gains:

- `selector_min_k: Optional[int] = None`
- `selector_rerank_only: bool = False`

Passed through to `AnchorSelector(...)` construction.

**`longmemeval.run()`** gains matching parameters, flows to `from_tenant()`.

**CLI flags** on `psa benchmark longmemeval run`:

- `--min-k INT`
- `--rerank-only` (boolean flag)
- `--max-k INT` (default 6, for explicit control in ablations)

**Config label:** computed once in `run()` from the active selector settings. Encodes `selector_mode`, `max_k`, `min_k`, and `rerank_only` into a deterministic label string. Used for both the output filename and the printed summary. Example labels:

- `cosine_k6`
- `trained_rerank_k6`
- `trained_min3_k6`
- `trained_k6`

This keeps filenames unambiguous if `max_k` is varied later.

### 4. Ablation runs

No code changes — just benchmark commands using the new flags.

| Label | Command flags | What it tests |
|-------|--------------|---------------|
| `cosine_k6` | `--selector cosine --max-k 6` | Baseline: top-6 by cosine, no CE |
| `trained_rerank_k6` | `--selector trained --rerank-only --max-k 6` | Is CE ranking useful? (fixed k=6, no filtering) |
| `trained_min3_k6` | `--selector trained --min-k 3 --max-k 6` | Does a floor help? |
| `trained_k6` | `--selector trained --max-k 6` | Current threshold behavior |

All four hold `max_k=6` constant for fair comparison.

**Metrics per run:**

- R@5 (anchor-level recall via `backtrack_gold_anchors`)
- Exact F1
- LLM-as-judge
- Anchor count distribution (the main symptom — should show whether single-anchor collapse is eliminated)
- Shortlist overlap with cosine top-6 (diagnostic for `trained-rerank`)
- Gold-hit rate in selected set

**Success criteria:**

- If `trained_rerank_k6` matches or exceeds `cosine_k6` on R@5 and F1: ranking is valuable, filtering was the problem. Invest in better threshold calibration or min_k as the product fix.
- If `trained_rerank_k6` matches cosine on F1 but improves gold-hit rate or overlap quality: CE is still useful once synthesis/cardinality is fixed. Don't kill it on one metric.
- If `trained_rerank_k6` trails `cosine_k6` across all metrics with k held constant: ranking itself is harmful. Consider retiring the CE path (Approach C).
- If `trained_min3_k6` > `trained_k6`: min_k is a viable product fix for the single-anchor collapse.

## Out of scope

These are informed by ablation results and deferred to a follow-up cycle:

- **Listwise/groupwise training loss** — only if ablation shows ranking signal is valuable but pair-level training is limiting it.
- **Memory-level reranker** — only if ablation shows anchor-level CE ranking is not worth the complexity.
- **Query pattern ablation** — comparing cards with/without `generated_query_patterns` to test overfitting hypothesis. Follow-up experiment once foundations are fixed.
- **Multi-positive training** — for queries with multiple winning anchors, train toward all positives jointly. Deferred until current ablation results are analyzed.
