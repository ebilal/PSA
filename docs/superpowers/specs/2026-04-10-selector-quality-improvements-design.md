# PSA Selector & Pipeline Quality Improvements

**Date:** 2026-04-10
**Goal:** Maximize answer quality from the trained selector pipeline. Improvements must be permanent code changes that benefit all users, not benchmark-specific fixes.
**Benchmark target:** Match or exceed MemPalace's 96.6% R@5 on LongMemEval (500 questions, zero API calls for retrieval).

---

## Current State

### Benchmark Results (2026-04-10 baseline)

| Metric | Cosine Selector | Trained Selector |
|--------|:-:|:-:|
| Exact F1 | 0.175 | 0.124 |
| LLM-as-judge | 0.322 | 0.159 |
| Avg tokens used | 4,371 | 1,211 |
| Avg anchors selected | 4.0 | 1.1 |
| Queries with 0 anchors | 0 | **327 (65%)** |

### Competitive Landscape (LongMemEval R@5)

| System | R@5 | LLM Required |
|--------|:-:|---|
| MemPalace (hybrid + Haiku rerank) | 100% | Optional |
| MemPalace (raw verbatim, no LLM) | 96.6% | None |
| Mastra | 94.87% | Yes (GPT) |
| PSA (current) | **unmeasured** | Yes (for answer gen) |

### Root Causes

1. **Oracle labeler produces failure-only labels** — no positive training signal
2. **Training data 98% negative** — model learns to reject everything
3. **Threshold (tau=0.30) calibrated on skewed data** — 65% of queries get zero anchors
4. **Packer ignores selector signal** — doesn't use cross-encoder scores for ranking
5. **No source enrichment** — packer uses short summaries (~250 chars), losing conversational detail
6. **No R@5 metric** — can't compare against MemPalace

---

## Design

### Section 1: Oracle Labeler — Generate Balanced Training Signal

**Problem:** The oracle labeling flow only writes labels for queries where the pipeline performed poorly. The `DataGenerator` then produces training data with near-zero positive examples, so the model can't learn what a good anchor looks like.

**Changes:**

- **`OracleLabeler.label_batch()`** — label all queries (successes and failures), not just poor-performing ones. Write both success and failure labels with properly identified winning anchor sets via the existing two-stage evaluation (cheap proxy scoring + expensive task-success judging).
- **`lifecycle.py`** — when collecting queries for oracle labeling, include a representative sample of successful queries alongside poor-performing ones.
- **`benchmarks/longmemeval.py`** — add an `oracle-label` subcommand (`psa benchmark longmemeval oracle-label`) that invokes the real `OracleLabeler` on benchmark results. This is separate from `score()` (which remains fast) because oracle labeling requires ~30-60 min of LLM calls. The workflow becomes: `ingest → run → score (fast metrics) → oracle-label (LLM evaluation) → train`.

### Section 2: Training Data Generation — Balanced Mix

**Problem:** `DataGenerator` relies on `winning_oracle_set` being populated to create positives. When empty, it produces only negatives. Even with the oracle labeler fix, the mix needs enforcement.

**Changes:**

- **`data_generator.py`** — add a minimum positive ratio floor. If positive:negative falls below 1:3, oversample positives or undersample negatives to reach at least 25% positive. The current 60/20/20 split (synthetic/hard-neg/adversarial) assumes balance within each category but doesn't enforce it.
- **Validation split** — hold out 15% of oracle labels for threshold calibration. Currently training and threshold selection can use the same data, which is a source of overfitting.
- **Skew warning** — log a warning if `DataGenerator.generate()` produces <10% positives, so users know their labels are imbalanced.

### Section 3: Selector Threshold Calibration — Less Aggressive Default

**Problem:** Threshold tau=0.30 was calibrated via Youden's J on skewed data. 65% of queries got zero anchors. Even with better data, Youden's J optimizes classification accuracy, not downstream answer quality. The failure mode is asymmetric: selecting a mediocre anchor is far better than selecting none.

**Changes:**

- **`train_selector.py`** — change threshold selection from Youden's J to F-beta with beta=2 (recall-weighted). Penalizes false negatives (rejecting good anchors) more than false positives (including mediocre ones). Matches the real cost asymmetry.
- **Maximum threshold cap** — tau can never exceed 0.25 regardless of calibration. Safety rail against aggressive filtering.
- **`selector.py`** — fallback floor: if the trained selector returns 0 anchors, fall back to the top-1 candidate by raw cross-encoder score. Zero anchors should never happen.

### Section 4: Retriever Tuning — Wider Shortlist

**Problem:** Retriever surfaces 24 candidates from 256 anchors (9.4% coverage). If the right anchor isn't in those 24, the selector can never find it. Multi-session and compositional questions need wider coverage.

**Changes:**

- **`anchor.py` / `retriever.py`** — increase default shortlist from 24 to 32. Minimal latency cost (retriever is <5ms).
- **`selector.py`** — increase `DEFAULT_MAX_K` from 4 to 6. The packer enforces token budget, so more anchors just means more clusters to draw from. Packer truncates if budget exhausted.

### Section 5: Packer Scoring — Benchmarkable Weights

**Problem:** Packer ranks memories by `0.7 * relevance_cosine + 0.3 * quality_score`, ignoring the trained selector's cross-encoder scores entirely. Memories from high-confidence anchors should outrank memories from marginal anchors.

**Changes:**

- **`packer.py`** — when a trained selector is active, use a configurable three-weight tuple `(selector_weight, cosine_weight, quality_weight)` for memory ranking. Default: `(0.4, 0.3, 0.3)` (the "balanced" option — final default determined by weight sweep).
- **`pipeline.py`** — pass selector scores through to the packer. Currently discarded after selection. Add scores to the memory fetch step so the packer can use them.
- **Benchmark `run` subcommand** — add `--packer-weights` flag (e.g., `--packer-weights 0.5,0.3,0.2`).
- **Cosine mode fallback** — when no trained selector is active, use the current `(0.0, 0.7, 0.3)` formula unchanged.

**Weight sweep to benchmark:**

| Name | selector | cosine | quality |
|------|:--:|:--:|:--:|
| selector-only | 0.7 | 0.0 | 0.3 |
| selector-heavy | 0.5 | 0.35 | 0.15 |
| balanced | 0.4 | 0.3 | 0.3 |

Winner gets baked in as the default.

### Section 6: R@5 Metric for Competitive Comparison

**Problem:** Our benchmark measures Exact F1 and LLM-as-judge (answer quality). MemPalace reports R@5 (retrieval recall). Can't compare without the same metric.

**What R@5 means:** For each question, LongMemEval provides `answer_session_ids` — sessions containing the answer. R@5 = fraction of questions where at least one answer session is covered by the top-5 retrieved results.

**PSA adaptation:** PSA retrieves anchors, not sessions. To compute R@5, check whether the memories fetched from selected anchors include any memory whose `source_path` maps back to an `answer_session_id`.

**Changes:**

- **`benchmarks/longmemeval.py`** — add `recall_at_k()` to the `score()` function. Map retrieved memories back to source sessions via `source_path`, compare against `answer_session_ids`.
- **Results JSONL** — store `answer_session_ids` per question (currently not captured) so R@5 can be computed post-hoc.
- **Score output** — report R@5 alongside Exact F1 and LLM-as-judge.

### Section 7: Source-Backed Context Enrichment

**Problem:** PSA's consolidation creates small typed memory objects (~250 chars avg) for efficient indexing and training. But when assembling the final context, these compressed summaries lose conversational detail. MemPalace's 96.6% comes from raw verbatim storage. The original text exists in `raw_sources` — we just don't surface it.

**What already exists:**
- `RawSource.full_text` — original conversation, stored immutably
- `MemoryObject.source_ids` — links each memory to its source
- `EvidenceSpan` — `source_id` + `start_offset` + `end_offset` for specific ranges
- `MemoryStore.get_source()` — retrieves raw source by ID

**Design principle:** Memory objects remain small for indexing and training. They serve as pointers. The packer uses them to surface the right slice of the original conversation — structured retrieval with verbatim fidelity.

**Changes:**

- **`packer.py`** — for top-N ranked memories, retrieve evidence spans from `raw_sources`. Extract relevant portions using offsets and include as supporting context below the memory summary.
- **Fallback** — if evidence spans aren't populated (older data), find where the memory's key terms appear in `full_text` and extract a ~500 char window around keyword matches.
- **Budget cap** — cap source enrichment per memory to prevent long conversations from blowing token budget. The packer's existing budget enforcement handles this.

### Section 8: Assistant-Turn Awareness

**Problem:** Questions about what the AI said ("what did you suggest about X?") fail when memory objects only capture user-side content. MemPalace showed a targeted two-pass fix for this failure mode.

**Changes:**

- **`pipeline.py`** — lightweight query classifier: detect assistant-reference triggers ("you suggested", "you told me", "remind me what you", etc.). When triggered, pass a flag to the packer.
- **`packer.py`** — when assistant-reference flag is set and source enrichment is active, include assistant turns from raw source text (not just user turns). The evidence span extraction from Section 7 already works on `full_text` which contains both sides — this just tells it to include assistant content when the query asks about it.

---

## Bug Fixes (Already in Working Tree)

These were hit during the benchmark run and are already fixed:

1. **`benchmarks/longmemeval.py`** — `load_dataset` → `hf_hub_download` (HuggingFace files have no extension)
2. **`benchmarks/longmemeval.py`** — `_normalize_text` crashes on int `answer_gold` → `str()` cast
3. **`train_selector.py`** — `PairDataset` crashes on None labels (sentence-transformers v5.x) → None filter
4. **`benchmarks/longmemeval.py`** — `run()` now accepts `selector_mode` / `selector_model_path`
5. **`cli.py`** — benchmark `run` exposes `--selector` / `--selector-model` flags
6. **`benchmarks/longmemeval.py`** — `score()` generates positive oracle labels for successful answers (F1 >= 0.5)

---

## What Stays Unchanged

- Embedding model (BAAI/bge-base-en-v1.5, 768-dim)
- Atlas architecture (224 learned + 32 novelty anchors, spherical k-means)
- Cross-encoder base model (cross-encoder/ms-marco-MiniLM-L-6-v2)
- Consolidation pipeline (Qwen extraction of typed MemoryObjects)
- Memory store schema (SQLite WAL)
- RRF fusion constant (k=60)

---

## Validation Plan

1. Run full oracle labeling on LongMemEval 500 questions (~30-60 min with Azure GPT-5.4-mini)
2. Retrain selector on balanced oracle labels
3. Benchmark cosine vs trained selector on all metrics (R@5, Exact F1, LLM-as-judge)
4. Sweep packer weight combinations
5. Compare R@5 against MemPalace baseline (96.6%)
6. Per-question-type breakdown to identify remaining failure modes
