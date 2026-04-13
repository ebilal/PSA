# Memory Activation System Design

## Goal

Replace PSA's retriever→selector lookup chain with a single-pass memory activation system that scores all 256 atlas anchors directly, learns inter-anchor co-activation patterns, and adaptively selects how many anchors each query needs. Target: R@5 ≥ 0.95 on LongMemEval (up from 0.894).

## Motivation

The current pipeline is a filter chain that loses information at every stage:

```
query → embed → BM25+dense retriever(256→24) → cross-encoder(24→6) → pack
```

Bottleneck analysis on LongMemEval (500 queries):
- **Retriever recall@24 = 0.739**: 26% of gold anchors never reach the selector
- **Selector recall@6 = 0.580**: loses another 16% from what it receives
- **Fixed max_k=6**: structurally cannot cover queries needing 4-10 gold anchors

The retriever is the larger bottleneck, but both stages contribute. No amount of selector tuning gets past 0.739.

## Architecture

New pipeline:

```
query → embed → FullAtlasScorer(256 scores) → CoActivationSelector(adaptive-k) → fetch → pack
```

### Component 1: FullAtlasScorer

Replaces both `AnchorRetriever` and `AnchorSelector._trained_select()`.

**What it does:** Scores all 256 anchors in the atlas using the existing cross-encoder (ms-marco-MiniLM-L-6-v2) in one batched forward pass on MPS.

**Interface:**
```python
class FullAtlasScorer:
    def __init__(self, model_path: str, atlas: Atlas, device: str = "mps")
    def score_all(self, query: str) -> List[AnchorScore]
    # AnchorScore: (anchor_id, ce_score, centroid: ndarray)
```

**Details:**
- Builds a batch of 256 `(query, card.to_card_text())` string pairs
- Single batched forward pass through cross-encoder (~50-100ms on MPS)
- Returns all 256 scores unfiltered — filtering is the co-activation layer's job
- Training: uses existing `SelectorTrainer` three-phase curriculum unchanged
- Fallback: if no trained model, cosine scoring over all 256 centroids

### Component 2: CoActivationSelector

Learns inter-anchor co-activation patterns and adaptive query-dependent selection.

**What it does:** Takes 256 individual cross-encoder scores and refines them using attention over anchor representations. When anchors A and B both score high, the model can boost anchor C because it has learned they form a complementary evidence cluster. Produces a per-query adaptive threshold that determines how many anchors to select.

**Architecture:**
- Input per anchor: `[ce_score (1-dim), centroid (768-dim)]` → projected to 128-dim
- Query conditioning: query embedding (768-dim) → projected to 128-dim, concatenated → 256-dim tokens
- 2-layer transformer encoder (256 tokens × 256-dim, 4 heads, feedforward 512)
- Score head: linear → 1 scalar per anchor (refined score)
- Threshold head: attention-pooled summary of all 256 tokens + query embedding → scalar τ_q
- Parameter count: ~2M

**Interface:**
```python
class CoActivationSelector:
    def __init__(self, model_path: str, device: str = "mps")
    def select(
        self, query_vec: ndarray, anchor_scores: List[AnchorScore]
    ) -> List[SelectedAnchor]
```

**Adaptive threshold:** The threshold head produces per-query τ_q. Anchors with refined score > τ_q are selected. Floor of 1 (always return at least one). No ceiling — if a query needs 8 anchors, it gets 8.

**Key property:** Permutation-equivariant. Uses atlas centroids (not learned per-anchor embeddings), so it survives atlas rebuilds without retraining the architecture — just retrain on new labels after rebuild.

### Component 3: Pipeline Integration

**Changes to `PSAPipeline`:**
- `__init__` accepts optional `full_atlas_scorer` and `coactivation_selector`
- `query()`: if both present, skip `self._retriever` and use new path; otherwise fall back to existing chain
- `PSAResult` gains `selection_mode: str` — `"legacy"`, `"full_atlas"`, or `"coactivation"`

**Graceful degradation (three levels):**
1. Trained co-activation + trained cross-encoder → full new path
2. Trained cross-encoder only → FullAtlasScorer + simple top-k (rerank_only behavior)
3. No trained models → existing cosine retriever→selector (current behavior)

**`from_tenant()` factory:** Loads co-activation model from `~/.psa/models/coactivation_v{N}/` if present. Falls back transparently.

**Lifecycle:** `lifecycle.py` slow path: atlas rebuild → cross-encoder train → co-activation train (sequential).

## Training Pipeline

### Oracle Labels (expanded)

Existing 24-candidate oracle labels remain valid for cross-encoder training (pairwise examples). For co-activation training, we generate expanded labels over all 256 anchors by running FullAtlasScorer on each oracle query. Each expanded label becomes:

```json
{
  "query": "...",
  "query_vec": [768-dim],
  "anchor_scores": [256 ce_scores],
  "anchor_centroids": [256 × 768-dim],
  "gold_anchor_ids": [list of gold anchors],
  "gold_k": 3
}
```

### Cross-Encoder Training

No change. Existing `SelectorTrainer` three-phase curriculum (warm start → hard negatives → adversarial). The cross-encoder learns pairwise relevance; it just gets applied to all 256 anchors at inference.

### Co-Activation Training

1. Freeze cross-encoder
2. Train CoActivationSelector on expanded oracle labels:
   - Score loss: BCE — refined scores should be 1.0 for gold anchors, 0.0 for rest
   - Threshold loss: MSE — predicted τ_q should yield gold_k
   - Combined: `L = BCE_score + 0.3 * MSE_threshold`
3. Single-phase, 5-10 epochs, lr=1e-4, batch=16 queries
4. Validation: held-out queries, measure recall@predicted_k

**Training gates:** Same as existing selector (300+ oracle labels, recall@24 ≥ 0.95). Co-activation trains only after cross-encoder.

**Model saved to:** `~/.psa/models/coactivation_v{N}/` with `coactivation_version.json`.

**Estimated time:** ~2M params, 500 queries × 10 epochs → ~3000 gradient steps. 2-5 minutes on MPS.

## Files

### New files

| File | Responsibility |
|------|---------------|
| `psa/full_atlas_scorer.py` | `FullAtlasScorer` — batched cross-encoder over all 256 anchors |
| `psa/coactivation.py` | `CoActivationSelector` — transformer + adaptive threshold |
| `psa/training/train_coactivation.py` | Training loop for co-activation model |
| `tests/test_full_atlas_scorer.py` | Unit tests for FullAtlasScorer |
| `tests/test_coactivation.py` | Unit tests for CoActivationSelector |

### Modified files

| File | Change |
|------|--------|
| `psa/pipeline.py` | New query path with graceful fallback |
| `psa/training/oracle_labeler.py` | Expand labels to 256-anchor scores + centroids |
| `psa/training/data_generator.py` | New example type for co-activation training |
| `psa/lifecycle.py` | Co-activation training step after selector training |
| `psa/cli.py` | `psa train --coactivation` flag; benchmark supports new modes |
| `psa/benchmarks/longmemeval.py` | New configs for `full_atlas` and `coactivation` runs |

### Unchanged

- `psa/retriever.py` — kept as fallback, not deleted
- `psa/selector.py` — kept for legacy/cosine mode
- `psa/packer.py`, `psa/synthesizer.py` — downstream, no changes
- `psa/atlas.py` — co-activation consumes centroids, doesn't modify them

## Success Criteria

**Primary:** R@5 ≥ 0.95 on LongMemEval val (500 queries)

**Secondary:**
- F1 ≥ 0.20 (must not regress below cosine baseline of 0.202)
- LLM-judge ≥ 0.35 (must not regress below current 0.352)
- End-to-end latency ≤ 400ms on MPS (M-series Mac)
- Co-activation training ≤ 10 minutes on MPS

**Testing:**

Unit tests (no model loading):
- FullAtlasScorer: mock cross-encoder, verify 256 scores, batching, cosine fallback
- CoActivationSelector: mock scores + centroids, verify output shape, variable k, min-1 floor
- Training: verify 256-anchor data generation, loss computation

Integration tests:
- Full pipeline with `selection_mode="coactivation"` on small test atlas
- Graceful degradation through all three fallback levels

Benchmark:
- `psa benchmark longmemeval run --selector coactivation`
- Compare against cosine and trained_rerank baselines
- Report adaptive-k distribution

**Done when:** R@5 ≥ 0.95, no F1/LLM regression, all tests pass, ≤ 400ms latency, `psa train --coactivation` works end-to-end on laptop.

## Constraints

- Must run entirely on a laptop (Apple Silicon MPS, no cloud GPU)
- Must integrate as a plugin for Claude Code, Codex, OpenClaw via hooks/MCP
- Must handle atlas evolution: rebuilds, new memories, forgetting, consolidation
- No learned per-anchor parameters — centroids from the atlas provide anchor identity
- Backward compatible: existing tenants work unchanged until they train the new models
