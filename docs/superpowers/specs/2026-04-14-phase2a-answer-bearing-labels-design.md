# Phase 2A: Answer-Bearing Training Labels + Learned Constraint Scoring

## Goal

Replace lexical-overlap training labels with LLM-graded utility labels and fold constraint features into the learned MLP, replacing rule-backed scoring on the primary path. Target: F1 >= 0.22 (up from 0.179), LLM-judge >= 0.40 (up from 0.372).

## Motivation

Phase 1 added query frames, memory facets, and a rule-backed constraint scorer. LLM-judge improved (0.330 -> 0.372) but F1 plateaued at 0.179. Two root causes:

1. **Wrong training signal.** The memory scorer MLP trains on token-level F1 overlap between `memory.body` and `gold_answer`. This rewards word overlap, not task utility. A memory that shares vocabulary with the gold answer but doesn't actually help answer the question gets labeled positive. A memory that contains the right evidence in different words gets labeled negative.

2. **Rule-backed constraint weights are fixed.** The ConstraintScorer uses hand-tuned weights (0.17 entity, 0.12 speaker, etc.) that can't adapt to query type or learn from data. The MLP should learn these weights end-to-end.

## Architecture

```
Training:
  query + memories -> batched LLM grading (0-3 scale per memory)
    -> mine hard negatives (same-anchor wrong-episode, right-topic wrong-time)
    -> 16-dim feature vectors (11 existing + 5 tri-state constraint features)
    -> train MLP with weighted MSE on graded labels

Inference (primary path, trained 16-dim model):
  query -> query_frame -> Level 1 -> fetch
    -> Level 2 MLP (16-dim input, constraints learned)
    -> packer

Fallback (old 11-dim model or no model):
  query -> query_frame -> Level 1 -> fetch
    -> [optional old 11-dim MLP]
    -> rule-backed ConstraintScorer
    -> packer
```

## Component 1: Graded Utility Labels

### LLM grading prompt (batched, one call per query)

For each query, collect ALL memories from the selected anchors (up to 50/anchor, ~200 unique after dedup). This is broader than the current Level 2 top-30 -- deliberately so, because the goal is to discover answer-bearing memories that the current scorer may rank low. Send them in a single batched call:

```
Query: {query}

Rate each memory's utility for answering this query:
0 = Irrelevant -- no useful information for this query
1 = Background -- provides general context but doesn't answer the query
2 = Supporting -- contains evidence that helps answer the query
3 = Direct answer -- contains the specific information needed to answer

Memory 1: {body[:200]}
Memory 2: {body[:200]}
...

Respond as JSON: {"1": score, "2": score, ...}
```

### Label normalization

- Grade 0 -> 0.0 (true negative)
- Grade 1 -> 0.33 (weak positive / background)
- Grade 2 -> 0.67 (supporting evidence)
- Grade 3 -> 1.0 (direct answer)

### Cost control

- Candidate pool: all memories from selected anchors (~200 unique per query after dedup)
- Pre-filter: memories with cosine < 0.1 auto-labeled 0 (not sent to LLM) -- only very distant memories skipped
- Batched: 500 queries = 500 LLM calls (~40K tokens each for ~200 memories)
- Cost: ~$2 on gpt-4o-mini, free on local Ollama (~30 min)
- Memory bodies truncated to 200 chars in prompt
- Broader pool is intentional: discovers answer-bearing memories the current scorer misses
- Pool management for crowded tenants:
  - Gather all fetched memories from selected anchors, dedupe by memory_object_id
  - Sort by max(ce_score, cosine_to_query) descending
  - Cap at 200 memories per query
  - If pool exceeds prompt token limit (~40K), batch into 2 calls of 100 each
  - Memories beyond the cap are auto-labeled 0

### Production mode

Same prompt via `call_llm()` with cloud API + Ollama fallback. Runs during lifecycle slow path alongside oracle labeling.

## Component 2: Hard Negative Mining

### Mix ratio

- 50% easy negatives (random grade-0 from other anchors)
- 25% same-anchor wrong-episode
- 25% right-topic wrong-time

### Same-anchor wrong-episode (tightened definition)

A candidate qualifies when ALL of:
- Same anchor as a positive (grade 2-3) memory
- No `source_ids` overlap with the positive: `len(set(positive.source_ids) & set(candidate.source_ids)) == 0` (not from the same episode)
- CE score or cosine within 0.3 of the positive (truly confusable)
- LLM-graded 0 (irrelevant)
- Cap: up to 3 per query

### Right-topic wrong-time (tightened definition)

A candidate qualifies when ALL of:
- Query has a `time_constraint` in the QueryFrame
- Candidate has populated `mentioned_at` (not None/empty)
- Candidate's temporal marker is explicitly outside the query's time window
- Candidate is in a selected anchor (topically relevant)
- CE score or cosine within 0.3 of the nearest positive (confusable)
- LLM-graded 0 only (grade-1 items are weak positives, never mined as hard negatives)
- Cap: up to 3 per query

Candidates with missing `mentioned_at` are NEVER mined as temporal negatives. They go to easy negatives or are skipped.

### Skip rule

Queries with no grade 2-3 memories after LLM grading are skipped entirely for hard-negative mining. No positive reference point means no meaningful hard negatives.

### Per-query caps

- Up to 5 positives (grade 2-3)
- Up to 3 same-anchor wrong-episode negatives
- Up to 3 right-topic wrong-time negatives
- Fill to 15 total with easy negatives (random grade-0 from other anchors)

### Future expansion (Phase 2B+)

- Right entity wrong speaker: after speaker_role/actor_entity separation is robust
- Right concept stale version: after `supersedes`/`contradicts` relation table exists

## Component 3: Expanded MLP with Tri-State Constraint Features

### New input features (5 dims added to existing 11)

Each feature uses explicit tri-state encoding:
- `1.0` = constrained + match
- `0.0` = constrained + mismatch
- `0.5` = unconstrained OR facet missing (unknown)

| Feature | Dim | Tri-state logic |
|---------|-----|-----------------|
| `entity_overlap` | 1 | Graded Jaccard [0,1] when both sets present. 0.5 when either set empty/unknown. NOT collapsed to binary. |
| `speaker_match` | 1 | frame.speaker_role_constraint == memory.speaker_role. Either None -> 0.5 |
| `actor_match` | 1 | frame.entity_constraint in memory.actor_entities (lowercased). Either empty/None -> 0.5. Alias resolution deferred to Phase 2B. |
| `temporal_match` | 1 | memory.mentioned_at within frame.time_constraint. Either None -> 0.5 |
| `type_match` | 1 | frame.answer_target maps to memory.memory_type. Returns 0.5 for targets with no expected type (comparison, temporal_change, prior_statement). |

### MLP architecture

`Linear(16, 32) -> ReLU -> Linear(32, 1) -> Sigmoid`

Sigmoid in both training and inference. ~600 params.

### Loss

Weighted MSE on sigmoid-bounded outputs [0, 1]:
- Grade 3 examples: weight 2.0
- Grade 2 examples: weight 1.5
- All others: weight 1.0

This is per-example weighting, not pair weighting. Pairwise ranking loss is deferred to Phase 2B+.

MSE is documented as a pragmatic starting point. The ideal long-term objective is pairwise ranking loss layered on top of regression.

## Component 4: Pipeline Integration + Fallback

### Primary path (trained 16-dim model, `supports_constraints: true` in metadata)

```
query -> query_frame -> Level 1 -> fetch
  -> MemoryScorer.score(query, query_vec, memories, query_frame)
     (builds 16-dim features internally, constraints learned by MLP)
  -> packer (pre_ranked=True)
```

ConstraintScorer.adjust_scores() is NOT called.

### Fallback path 1 (old 11-dim model, `supports_constraints: false` or absent)

```
query -> query_frame -> Level 1 -> fetch
  -> MemoryScorer.score(query, query_vec, memories)  # 11-dim, no frame
  -> ConstraintScorer.adjust_scores(scored, query_frame)  # rule-backed
  -> packer (pre_ranked=True)
```

Detected by `supports_constraints` field in `memory_scorer_version.json`.

### Fallback path 2 (no MLP at all)

```
query -> query_frame -> Level 1 -> fetch
  -> ConstraintScorer.adjust_scores(scored, query_frame)
  -> packer (pre_ranked=True)
```

### Null frame behavior

When `extract_query_frame()` fails or returns very low confidence, a null frame is used (all fields None/empty). Both learned and rule-backed paths treat this as all-0.5 constraint features. No special branching -- just neutral inputs.

### Sorting guarantee

All fallback paths sort by score descending before setting `_pre_ranked = True`.

## Files

### New files

| File | Responsibility |
|------|---------------|
| `psa/training/graded_labeler.py` | Batched LLM grading (0-3) + hard negative mining + training data assembly |
| `tests/test_graded_labeler.py` | Unit tests for grading, mining, and data assembly |

### Modified files

| File | Change |
|------|--------|
| `psa/memory_scorer.py` | Expand MLP 11->16 dims; accept QueryFrame; build tri-state constraint features; `supports_constraints` metadata |
| `psa/training/train_memory_scorer.py` | Weighted MSE loss; handle 16-dim input; save `supports_constraints: true` |
| `psa/pipeline.py` | Check `supports_constraints`; remove ConstraintScorer from primary path; null-frame handling |
| `psa/cli.py` | `psa train --memory-scorer` uses graded labeler |
| `psa/lifecycle.py` | Slow path uses graded labeler for memory scorer retraining |

### Unchanged (kept as fallback)

| File | Status |
|------|--------|
| `psa/constraint_scorer.py` | Kept for fallback + ablation, not deleted |
| `psa/query_frame.py` | No changes |
| `psa/coactivation.py` | No changes |
| `psa/training/memory_scorer_data.py` | Old labeler kept for backward compat |

## Success Criteria

| Metric | Target | Current |
|--------|--------|---------|
| Exact F1 | >= 0.22 | 0.179 |
| LLM-as-judge | >= 0.40 | 0.372 |
| R@5 | >= 0.920 (hold) | 0.920 |
| Graded label quality | >= 80% agreement with gold on spot-check |
| Training cost | <= $1 per full label run on gpt-4o-mini |

## Constraints

- Laptop-only (Apple Silicon MPS, no cloud GPU)
- LLM calls via call_llm() with cloud API + Ollama fallback
- All models on MPS (no mixed devices)
- Old 11-dim models fall back to rule-backed scorer (never mixed-dim)
- Missing facets = 0.5 (unknown/neutral), never 0.0 (mismatch)
- Null query frame = all-neutral constraint features, no special branches
