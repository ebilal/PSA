# Two-Level Memory Activation System Design

## Goal

Extend the memory activation system with two improvements: (1) enrich Level 1 anchor scoring with per-anchor memory type distributions and metadata features, and (2) add a Level 2 memory-level scorer that ranks individual memories within selected anchors, replacing the packer's heuristic. Target: R@5 >= 0.93, F1 >= 0.22, LLM-judge >= 0.40.

## Motivation

The current system selects anchors (cluster-level) but has no learned model at the memory level. After Level 1 picks ~5-10 anchors, the pipeline fetches up to 50 memories per anchor and hands them to the packer, which ranks by `0.7 * cosine + 0.3 * quality_score`. This heuristic doesn't consider memory type, recency, or textual relevance to the query.

Two gaps:
- **R@5 = 0.912**: Level 1 co-activation model sees only `(ce_score, centroid)` per anchor. Adding type distribution and metadata features gives it richer signal for anchor selection.
- **F1 = 0.172, LLM-judge = 0.362**: The packer receives all memories from selected anchors and uses a simple heuristic. A learned memory scorer can prioritize the memories that actually answer the query.

## Architecture

```
query -> embed
  -> Level 1: FullAtlasScorer (256 scores)
     -> CoActivationSelector (enriched: +type_dist, +count, +quality per anchor)
     -> adaptive-k anchors
  -> fetch memories (up to 50/anchor)
  -> Level 2: MemoryScorer
     -> Stage A: cross-encoder scores (query, memory_body) pairs
     -> Stage B: re-ranker MLP on (ce_score + metadata) -> final ranking
  -> packer fills token budget in Level 2's order (no heuristic re-ranking)
```

## Level 1 Enhancement: Enriched Co-Activation Features

The co-activation model currently takes `(ce_score, centroid)` per anchor. We add a per-anchor feature vector:

- `type_distribution`: 6-dim (fraction of each MemoryType: EPISODIC, SEMANTIC, PROCEDURAL, FAILURE, TOOL_USE, WORKING_DERIVATIVE)
- `memory_count`: 1-dim (normalized by max count across anchors)
- `avg_quality`: 1-dim (mean quality_score of memories in this anchor)

Total: 8 additional dims per anchor. The `anchor_proj` layer changes from `Linear(1 + 768, d_model//2)` to `Linear(1 + 768 + 8, d_model//2)`.

**Where features come from:** Precomputed at atlas build time. Stored as new fields on `AnchorCard`. Updated during lifecycle when memories are added/pruned. At query time, read from the atlas -- no extra DB queries.

**Production generalization:** All features are derivable from the data itself (memory type, count, quality_score are on every MemoryObject). No gold labels or benchmark data required. Works identically for Claude session logs mined in production.

**Training:** Same oracle labels, same loss. The co-activation model just sees richer input. Retrain with `psa train --coactivation`.

## Level 2: MemoryScorer

After Level 1 selects anchors, Level 2 scores individual memories within those anchors. Takes ~200-500 fetched memories, produces a relevance-ranked list. The packer fills the token budget in Level 2's order -- no heuristic re-ranking.

### Stage A: Cross-Encoder Scoring

Reuse the existing MiniLM cross-encoder to score `(query, memory_body)` pairs. Same model as Level 1 (no additional model loading). Batched in chunks of 64. Returns a textual relevance score per memory.

### Stage B: Memory Re-Ranker

A small MLP that takes per-memory features and outputs a final relevance score:

| Feature | Dim | Source |
|---------|-----|--------|
| `ce_score` | 1 | Stage A (textual relevance) |
| `memory_type` | 6 | One-hot encoding of MemoryType |
| `quality_score` | 1 | From MemoryObject |
| `body_token_count` | 1 | Normalized by token budget |
| `recency` | 1 | Days since created, log-scaled |
| `cosine_to_query` | 1 | Embedding similarity |

Input: 11-dim per memory. Architecture: `Linear(11, 32) -> ReLU -> Linear(32, 1) -> sigmoid`. ~400 parameters. Trains in seconds.

### Interface

```python
class MemoryScorer:
    def __init__(self, cross_encoder, reranker_model, device="cpu")
    def score(
        self, query: str, query_vec: ndarray, memories: List[MemoryObject]
    ) -> List[ScoredMemory]
    # ScoredMemory: (memory_object_id, final_score, memory)
    # Sorted by final_score descending

    @classmethod
    def from_model_path(cls, model_path, cross_encoder, device="cpu")
```

## Pipeline Integration

### Changes to PSAPipeline

- New optional `memory_scorer: Optional[MemoryScorer]` attribute
- In `query()`, after `_fetch_memories()`: if memory_scorer present, call it to re-rank; pass result to packer with `pre_ranked=True`
- If no memory_scorer, fall back to current behavior

### Changes to EvidencePacker

- `pack_memories_direct()` gains `pre_ranked: bool = False` parameter
- When `pre_ranked=True`, skip internal `0.7*cosine + 0.3*quality` ranking; pack in received order

### Graceful Degradation (4 levels)

1. CoActivation + MemoryScorer -> full two-level system
2. CoActivation only -> anchor activation, packer heuristic for memories
3. FullAtlas top-k -> cross-encoder scoring, packer heuristic
4. Legacy retriever+selector -> current behavior

### Loading

`from_tenant()` loads MemoryScorer from `~/.psa/models/memory_scorer_v{N}/` if present. Same try/except + logger.debug pattern as co-activation loading.

## Training Data for Level 2

### Benchmark (LongMemEval)

Each query has a gold answer. For each fetched memory, compute token-level F1 overlap between `memory.body` and `gold_answer`. Memories with F1 above 0.3 are positive, below are negative. Produces `(query, memory_features, label)` triples.

### Production

During oracle labeling (`local`/`api` mode), extend with leave-one-out attribution: for each memory in the packed set, measure TaskSuccess with and without it. If removing a memory drops the score, it was useful (positive). Runs during lifecycle slow path (batched overnight). Expensive (~N extra LLM calls per query) but automatable.

V1 trains on benchmark data (text overlap). Production attribution comes as a later enhancement.

### Training Data Format

```json
{
  "query": "...",
  "ce_score": 0.61,
  "memory_type": "PROCEDURAL",
  "quality_score": 0.85,
  "body_token_count": 120,
  "recency_days": 14.0,
  "cosine_to_query": 0.72,
  "label": 1
}
```

Saved as JSONL. MLP trains via BCE loss. Same `psa train` command with `--memory-scorer` flag.

**Training gates:** Minimum 200 labeled query-memory pairs with at least 25% positive rate.

## Files

### New files

| File | Responsibility |
|------|---------------|
| `psa/memory_scorer.py` | `MemoryReRanker` (MLP nn.Module) + `MemoryScorer` (inference wrapper) |
| `psa/training/memory_scorer_data.py` | Generate memory-level labels from benchmark or attribution |
| `psa/training/train_memory_scorer.py` | Training loop for re-ranker MLP |
| `tests/test_memory_scorer.py` | Unit tests |

### Modified files

| File | Change |
|------|--------|
| `psa/coactivation.py` | Extend `anchor_proj` input dim from 769 to 777 (+8 features) |
| `psa/anchor.py` | New fields: `type_distribution`, `avg_quality`, `memory_count_norm` |
| `psa/atlas.py` | Compute per-anchor features at build time |
| `psa/pipeline.py` | Add MemoryScorer to query flow, pass `pre_ranked` to packer |
| `psa/packer.py` | Add `pre_ranked` parameter to `pack_memories_direct()` |
| `psa/training/coactivation_data.py` | Include anchor features in npz |
| `psa/cli.py` | `psa train --memory-scorer` flag |
| `psa/lifecycle.py` | Add memory scorer training to slow path |

### Unchanged

- `psa/full_atlas_scorer.py` -- fix device to MPS (remove CPU workaround); Level 2 reuses same cross-encoder
- `psa/retriever.py`, `psa/selector.py` -- legacy fallback, untouched
- `psa/synthesizer.py` -- downstream of memory ranking, no changes

## Success Criteria

| Metric | Target | Current |
|--------|--------|---------|
| R@5 (gold-hit) | >= 0.93 | 0.912 |
| Exact F1 | >= 0.22 | 0.172 |
| LLM-as-judge | >= 0.40 | 0.362 |
| Latency (MPS) | <= 400ms | ~340ms (measured) |
| Memory scorer training | <= 1 min | N/A |

## Production Viability

### Latency (resolved)

The SIGSEGV that forced CPU was caused by loading models on mixed devices (embedding on MPS, cross-encoder on CPU). Loading both on MPS is stable and fast:
- Level 1 (256 anchors): ~150ms on MPS
- Level 2 (300 memories): ~190ms on MPS
- Full two-level pipeline: ~340ms total

Fix: `FullAtlasScorer.from_model_path()` must load on MPS (not CPU). Remove the `torch.set_default_device('cpu')` workaround. Use `device="mps"` consistently for all sentence-transformers models.

### Scaling (resolved by forgetting)

`forgetting.py` enforces per-anchor budget (soft archive after threshold, hard delete after 90 days) and global cap. This bounds Level 2's input: with ~30-50 memories per anchor and ~5-10 selected anchors, Level 2 sees at most ~300-500 memories. The forgetting mechanism IS the scaling solution.

### Training labels for Level 2 in production

Use `call_llm()` (cloud API with local Ollama fallback) to generate memory-level relevance labels. For each `(query, memory)` pair, one LLM call:

> "Given this query and this memory, is this memory relevant to answering the query? Respond with 1 (relevant) or 0 (not relevant)."

Same `call_llm()` function used by the oracle labeler. Works from day one with any LLM endpoint. Runs during lifecycle slow path alongside anchor-level oracle labeling.

## Constraints

- Laptop-only (Apple Silicon MPS, no cloud GPU)
- Must work with Claude/Codex/OpenClaw session logs (no benchmark-only assumptions)
- All models load on MPS (never mix MPS + CPU devices in same process)
- Atlas evolution: features update on rebuild, models retrain via lifecycle
- Backward compatible: tenants without Level 2 model fall back to packer heuristic
