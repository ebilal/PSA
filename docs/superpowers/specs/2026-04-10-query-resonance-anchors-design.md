# Query-Resonance Anchors + Synthesis

**Date:** 2026-04-10
**Goal:** Replace the retrieval-then-pack paradigm with one where anchors encode what they know how to answer, and synthesis replaces ranked-list context assembly.

---

## Problem Statement

The current PSA pipeline has three compounding weaknesses:

1. **Anchor cards describe contents, not answer-patterns.** The cross-encoder scores `(query, "this cluster covers memories about dependency management")` — abstract description vs concrete query. A card that says "what did we decide about package managers?" gives far stronger signal.

2. **Oracle labels are circular.** The OracleLabeler evaluates combinations of anchors the retriever already surfaced. Gold anchor IDs are unknown, so `SupportCoverage` always scores 0. The model learns from its own mistakes, not from ground truth.

3. **Packing is the wrong output format.** Selecting 3 anchors and dumping 15 ranked memory bullets forces the downstream LLM to reconstruct the story from fragments. A hand-crafted scoring formula (`W_sel × selector + W_cos × cosine + W_qual × quality`) approximates relevance but doesn't produce coherent context. The formula is obsolete once synthesis is available.

---

## Design

### Section 1: Anchor Cards — Query Pattern Fingerprints

`AnchorCard` gets two new fields:

```python
generated_query_patterns: List[str]  # seeded at build time, stable
query_fingerprint: List[str]         # accumulated from real usage, capped at 50
```

**At atlas build time**, after existing card generation (name, meaning, include/exclude terms), one additional LLM call generates 10–15 example queries the anchor can answer. Prompt: "Given these memory objects, write 10–15 specific questions a user might ask that this cluster can answer." These are concrete, varied, phrased like real queries — not descriptions of the cluster.

**At inference time**, every query that activates an anchor **above the selection threshold** gets appended to `query_fingerprint`. Below-threshold activations are not accumulated — they would pollute the fingerprint with noise. Fingerprint is capped at 50 entries; oldest evicted first (FIFO).

**`to_stable_card_text()`** — used by cross-encoder at training and inference — includes `generated_query_patterns` but not `query_fingerprint`. Accumulated queries change too fast to be stable training targets. The cross-encoder learns to match queries against seed patterns. Accumulated queries feed BM25 retrieval, making anchors discoverable via the actual language users have used.

**Fingerprint persistence**: stored in `~/.psa/tenants/{tenant_id}/atlas_v{N}/fingerprints.jsonl` — separate from the atlas JSON, which is rebuilt on atlas rebuild. The existing `_match_anchors()` logic in `atlas.py` inherits fingerprints from matched predecessor anchors on rebuild, so accumulated signal survives atlas rebuilds.

Cross-encoder input becomes:

```
Query: "what was our decision about the database migration approach?"

Anchor: Schema Decisions | Covers choices about data model design...
Includes: migration, schema, database
Example questions this anchor answers:
  - What did we decide about the primary key strategy?
  - When did we switch from Alembic to manual migrations?
  - Why did we choose Postgres over MySQL?
```

### Section 2: Oracle Labeling — Session Backtracking

**Root cause of circular labels**: `OracleLabeler.label()` evaluates combinations of anchors the retriever surfaced. Gold anchor IDs are always unknown, so `score_support_coverage()` returns 0.0 for every set. The winning set is chosen by `ProceduralUtility` and `TaskSuccess` alone, with no ground-truth signal about which anchors actually contained the answer.

**Fix**: session backtracking — tracing from known ground truth to anchor IDs, independent of the retriever.

**New function** `backtrack_gold_anchors(answer_session_ids, store, atlas) → List[int]`:

```python
# For each answer session, find memory objects sourced from it,
# then find which anchors those memory objects belong to.
for session_id in answer_session_ids:
    memories = store.get_by_source_session(session_id)  # new method
    for m in memories:
        if m.anchor_id is not None:
            gold_anchor_ids.add(m.anchor_id)
```

Deterministic, no LLM calls. Works for any benchmark or dataset that provides ground-truth source references (LongMemEval provides `answer_session_ids`).

**OracleLabeler change**: `label()` accepts `gold_anchor_ids: Optional[List[int]]`. When provided, `score_support_coverage()` receives real gold anchor IDs instead of an empty list. An anchor set that covers gold anchors now scores higher at stage 1, so it reaches stage 2's expensive TaskSuccess evaluation. Labels describe which anchor combinations actually contained the answer.

**`psa benchmark longmemeval oracle-label`** subcommand calls `backtrack_gold_anchors()` for each question and passes results to `OracleLabeler.label()`. This replaces the heuristic labels written by `score()`.

**For production** (no external ground truth): the session-stop hook captures which anchor IDs were activated per query. After task completion, one LLM call — "did the retrieved context help complete this task?" — produces an outcome signal. This is weak but real: positive outcome anchors were good selections, negative outcome anchors were not. These become production oracle labels over time, displacing benchmark-derived labels for the patterns that matter in real usage.

**Training target change**: oracle labels now train the cross-encoder on `(query, anchor_card_with_query_patterns)` pairs. The model learns to match query intent against the anchor's compiled answer-patterns — not surface text overlap.

### Section 3: Synthesis at Activation

The packer's scoring formula (`W_sel × selector_score + W_cos × cosine + W_qual × quality`) is removed. It was compensating for a missing step — approximating relevance through weighted proxy signals because the pipeline had no way to judge what's relevant to *this* query. Synthesis makes it obsolete.

**New `psa/synthesizer.py`** — `AnchorSynthesizer` class, instantiated once in `PSAPipeline.__init__()`:

```python
class AnchorSynthesizer:
    def synthesize(
        self,
        query: str,
        memories: List[MemoryObject],  # all memories from all selected anchors
        token_budget: int = 700,
    ) -> str:
        """
        Synthesize a single query-conditioned narrative from all selected anchor memories.
        Returns coherent prose, not a list of facts.
        """
```

**Combined synthesis**: all memories from all selected anchors are passed together in one LLM call. The synthesizer produces one unified narrative — not one synthesis per anchor. This yields coherent context that can weave related facts across anchors rather than producing separate per-anchor paragraphs.

**Prompt structure**:

```
You are synthesizing memory context for an AI assistant.

Query: {query}

Relevant memories from personal history:
{memory_titles_and_summaries}

Write a focused, coherent paragraph (5–8 sentences) presenting what's most
relevant to help answer the query. Weave related facts into a narrative where
possible. Be specific and factual. Do not add information not present in the memories.
```

**Token budget**:
- Memory input to synthesizer is capped to avoid exceeding context limits (trim by cosine similarity to query — already computed during retrieval — lowest scores dropped first; no new scoring formula)
- Synthesis output is bounded at ~700 tokens
- Remaining budget goes to system prompt and user message downstream

**LLM loading**: `AnchorSynthesizer` uses the existing `call_llm()` infrastructure (cloud first, Qwen/Ollama fallback). Ollama is always available locally. The synthesizer object is instantiated once in `PSAPipeline.__init__()` — no per-query initialization. Connection pooling is handled by the existing `call_llm()` HTTP client.

**Fallback**: if synthesis raises an exception (LLM unavailable, timeout), `PSAPipeline.query()` catches it and falls back to current packing behavior. The pipeline stays functional.

**Pipeline flow after this change**:

```
embed query →
AnchorRetriever (BM25 + dense, RRF, top-32) →
AnchorSelector (cross-encoder on query-pattern cards) →
fetch MemoryObjects for selected anchors →
AnchorSynthesizer (one LLM call, combined narrative) →
PackedContext (synthesis text, token count, timing)
```

The `EvidencePacker` is simplified to: collect memories from selected anchors respecting token budget, pass to synthesizer. The scoring/ranking formula is removed entirely.

---

## Files Changed

| File | Change |
|------|--------|
| `psa/anchor.py` | Add `generated_query_patterns`, `query_fingerprint` to `AnchorCard`; update `to_stable_card_text()` |
| `psa/atlas.py` | `_generate_anchor_card()`: add LLM call for query patterns; `FingerprintStore` for persistence and rebuild inheritance |
| `psa/synthesizer.py` | New: `AnchorSynthesizer` class |
| `psa/pipeline.py` | Instantiate `AnchorSynthesizer`; fingerprint accumulation (above-threshold only); call synthesis instead of packer scoring |
| `psa/packer.py` | Remove scoring formula; simplify to memory collection + token budget cap |
| `psa/memory_store.py` | Add `get_by_source_session(session_id)` method |
| `psa/training/oracle_labeler.py` | Add `backtrack_gold_anchors()`; add `gold_anchor_ids` param to `label()` |
| `psa/benchmarks/longmemeval.py` | `oracle_label()` uses `backtrack_gold_anchors()` |
| `README.md` | Update Quick Start and architecture sections to reflect synthesis-based context assembly |

---

## What Stays Unchanged

- Atlas clustering (spherical k-means, 224 learned + 32 novelty anchors)
- Embedding model (BAAI/bge-base-en-v1.5, 768-dim)
- Cross-encoder base model (cross-encoder/ms-marco-MiniLM-L-6-v2)
- MemoryStore schema (SQLite WAL)
- Retriever (BM25 + dense, RRF k=60)
- `call_llm()` infrastructure and cloud/Ollama routing

---

## Validation Plan

1. Atlas rebuild with query-pattern generation — verify card text includes patterns
2. Fingerprint accumulation — verify only above-threshold activations write, FIFO eviction holds
3. Session backtracking — verify gold anchor IDs match expected sessions for LongMemEval examples
4. Oracle labeling — verify SupportCoverage is non-zero with backtracked gold anchors
5. Synthesis — verify output is coherent prose, not bullet list; fallback triggers correctly
6. LongMemEval benchmark — compare R@5, Exact F1, LLM-as-judge against current cosine baseline (0.684 R@5)
7. Qualitative check — run Quick Start from README, verify synthesized context is meaningfully better than raw memory list
