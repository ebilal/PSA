# Phase 1: Query Frames, Retrieval Facets, and Constraint Scoring

## Goal

Shift PSA from pure semantic matching to task-utility matching. Add structured query analysis (query frames), richer metadata on memories (retrieval facets), and constraint-aware scoring after Level 2. Target: F1 >= 0.22 (up from 0.177), R@5 >= 0.93 (up from 0.920).

## Motivation

PSA currently matches queries to memories by meaning. Benchmarks and real usage reward matching by task utility -- finding the evidence that actually answers the question under constraints like time, speaker, entity, and change-over-time.

Three gaps:
1. **No query analysis.** The pipeline doesn't know if a query asks for a fact, a temporal change, or a specific person's statement. Level 1 and Level 2 must infer this implicitly from embeddings.
2. **Memories lack structured facets.** Entities, timestamps, speaker identity, and stance (prefers/stopped/switched) are present in the raw text but not extracted. The consolidation LLM prompt doesn't ask for them. The normalizer discards speaker and timestamp info.
3. **Level 2 scores by semantic relevance only.** The MemoryReRanker MLP sees (ce_score, type, quality, recency, cosine) but has no signal for constraint satisfaction. It can't distinguish "right topic, wrong time" from "right topic, right time."

## Architecture

```
query -> extract_query_frame(query)
            |
            |  QueryFrame: answer_target, entities, time_constraint,
            |              speaker_role, speaker_entity, retrieval_mode
            v
  Level 1: CoActivationSelector
            (frame features: answer_target + retrieval_mode one-hots
             injected into query_proj alongside query embedding)
            * Version-gated: only injected if model was trained with frame dims.
              Old models get zero-padded input via query_frame_dim metadata check.
            |
            v
  Level 2: MemoryScorer (existing MLP, unchanged in Phase 1)
            |
            v
  Constraint Scorer (rule-backed, post-Level-2)
            - entity_overlap, speaker_match, temporal_consistency,
              stance_relevance, answer_target_type_match
            - Applies boost/penalty multipliers to Level 2 scores
            - Tri-state: match=1.0, mismatch=0.0, unknown/unconstrained=0.5
            |
            v
  Packer (fills token budget in constraint-adjusted order)
```

**Key design decisions from review:**
- **Constraint scoring is rule-backed in Phase 1, not learned.** The MLP stays at 11 dims. Learned constraint weighting is deferred to Phase 2 when better training labels (LLM-judged constraint satisfaction) exist. Lexical-overlap labels would teach the wrong thing.
- **Level 1 frame injection is version-gated.** `coactivation_version.json` stores `query_frame_dim`. Old models (query_frame_dim=0 or absent) get zero-padded input. New models trained after Phase 1 get real frame features.
- **Speaker is split into role + entity.** Memory facet: `speaker_role` ("user"/"assistant") and `actor_entities` (["Alice"]). Query frame: `speaker_role_constraint` and `entity_constraint`. These are matched separately.

## Component 1: Richer Facets on Memories

### Ingestion-time extraction (new memories)

Extend the consolidation LLM prompt to return 4 additional facet groups:

| Facet | Type | Example |
|-------|------|---------|
| `entities` | `List[str]` | `["Alice", "GraphQL", "auth-service"]` |
| `temporal_markers` | `Dict` | `{"mentioned_at": "2026-03", "relative": "last week", "valid_from": null, "valid_to": null}` |
| `speaker_role` | `str` | `"user"`, `"assistant"`, `"system"`, `"external"` |
| `actor_entities` | `List[str]` | `["Alice", "Bob"]` -- named people/actors mentioned |
| `stance` | `str` or null | `"prefers"`, `"stopped"`, `"switched"`, `"failed"`, `"fixed"`, `"deprecated"`, null |

Zero additional LLM calls -- the consolidation pipeline already makes one call per chunk. Just a bigger JSON response.

### Structured conversation parser (new)

`normalize.py` stays unchanged (it returns flat transcript strings and callers depend on that). Instead, create a new `psa/conversation_parser.py` that returns structured turns:

```python
@dataclass
class ConversationTurn:
    role: str           # "user", "assistant", "system"
    text: str
    timestamp: Optional[str]  # ISO datetime if available
    metadata: Dict      # format-specific extras

def parse_conversation(path: str) -> List[ConversationTurn]
```

This parser handles the same formats as `normalize.py` (Claude JSON, ChatGPT, Claude Code JSONL, Codex JSONL, Slack JSON, plain text) but preserves structured metadata. `convo_miner.py` uses it instead of `normalize()` for the PSA path. `normalize()` remains for legacy callers.

The consolidation LLM prompt receives `speaker_role` and `timestamp` from the parsed turn, giving it the context to populate facets accurately.

### Heuristic backfill for existing memories

`psa repair --backfill-facets` extracts facets from **raw source records** (via `source_ids` -> `memory_sources` table), not from the rewritten `memory.body`. Raw sources have the original conversation turns with speaker markers and timestamps.

Extraction heuristics applied to raw source text:
- **Entities:** CamelCase words, quoted strings, file paths, URLs, `@mentions`, known names from entity registry
- **Temporal:** date patterns (2026-03, March 2026), relative phrases (last week, yesterday, Q3)
- **Speaker role:** turn structure ("> " prefix = user, unmarked = assistant)
- **Actor entities:** named person references in turn text
- **Stance:** keyword scan for "switched to", "stopped using", "deprecated", "fixed", "prefers"

Falls back to `memory.body` only when raw source is unavailable. Also serves as the always-available extractor when `use_llm=False`.

### Storage

New columns on `memory_objects` table (added via ALTER TABLE migration):
- `entities TEXT` (JSON-encoded list of strings)
- `actor_entities TEXT` (JSON-encoded list of strings)
- `speaker_role TEXT`
- `stance TEXT`
- `mentioned_at TEXT`

Existing `validity_interval` column gets populated from `temporal_markers.valid_from`/`valid_to`.

New fields on MemoryObject dataclass:
- `entities: List[str] = field(default_factory=list)`
- `actor_entities: List[str] = field(default_factory=list)`
- `speaker_role: Optional[str] = None`
- `stance: Optional[str] = None`
- `mentioned_at: Optional[str] = None`

## Component 2: Query Frame Extraction

### What it produces

```python
@dataclass
class QueryFrame:
    answer_target: str                  # fact, preference, procedure, failure,
                                        # temporal_change, prior_statement, comparison
    entities: List[str]                 # extracted entity mentions
    time_constraint: Optional[str]      # "last week", "before March", "currently", None
    speaker_role_constraint: Optional[str]   # "user", "assistant", None
    entity_constraint: Optional[str]    # "Alice", None (named actor)
    retrieval_mode: str                 # single_hop, compare_over_time, multi_hop, abstention_risk
    confidence: float                   # pattern matcher confidence, 0.0-1.0
```

### Hybrid extraction

**Pattern matcher (fast path, < 5ms):** Regex + keyword rules handle common signals:
- Temporal: "used to", "currently", "before/after X", "changed", "when did" -> temporal_change / compare_over_time
- Speaker role: "you told me", "I said" -> speaker_role_constraint
- Entity constraint: "Alice said", "what did Bob" -> entity_constraint
- Entities: quoted strings, CamelCase, known entity names from entity registry
- Failure: "went wrong", "broke", "failed", "bug" -> answer_target=failure
- Procedure: "how do I", "steps to", "how to" -> answer_target=procedure

If the pattern matcher produces a frame with confidence >= 0.6 (answer_target + retrieval_mode both identified), skip the LLM call.

**LLM fallback (slow path, ~200-500ms):** One `call_llm()` call with a short prompt asking for the frame as JSON. Uses cloud API with local Ollama fallback. Only triggered when pattern matcher confidence < 0.6.

### Where it's used

- **Level 1:** Frame features (answer_target one-hot 7-dim + retrieval_mode one-hot 4-dim = 11 dims) concatenated with query embedding before `query_proj`. Changes `query_proj` from `Linear(768, d_model//2)` to `Linear(779, d_model//2)`. **Version-gated:** `coactivation_version.json` stores `query_frame_dim`. On load, if saved model has `query_frame_dim=0` or absent, the frame features are zero-padded. Models only use real frame features after retraining.
- **Constraint scorer:** Frame fields used to compute constraint match scores against memory facets.

### New file

`psa/query_frame.py` -- QueryFrame dataclass + `extract_query_frame()` function.

## Component 3: Constraint Scorer (Rule-Backed)

A lightweight post-Level-2 scoring layer that adjusts memory scores based on constraint satisfaction. **Not learned** -- uses fixed weights in Phase 1. Learned weighting deferred to Phase 2 when better training labels exist.

### Constraint features

Each feature uses tri-state scoring:
- **Constrained + match:** 1.0
- **Constrained + mismatch:** 0.0
- **Constrained + unknown (memory facet missing):** 0.5
- **Unconstrained (no constraint in query frame):** 0.5

| Feature | Computation |
|---------|-------------|
| `entity_overlap` | Jaccard: `len(frame.entities & memory.entities) / len(frame.entities \| memory.entities)`. If memory.entities empty: 0.5. If frame.entities empty: 0.5 |
| `speaker_role_match` | 1.0 if `frame.speaker_role_constraint == memory.speaker_role`, 0.0 if mismatch, 0.5 if either is None/empty |
| `actor_entity_match` | 1.0 if `frame.entity_constraint in memory.actor_entities`, 0.0 if not found, 0.5 if memory.actor_entities empty or no constraint |
| `temporal_consistency` | 1.0 if memory's `mentioned_at`/`validity_interval` falls within `frame.time_constraint`, 0.0 if outside, 0.5 if either is None |
| `stance_relevance` | 1.0 if frame.answer_target aligns with memory.stance (temporal_change + switched/stopped), 0.0 if contradicts, 0.5 if neutral/missing |
| `answer_target_type_match` | 1.0 if frame.answer_target maps to memory.memory_type (failure->FAILURE, procedure->PROCEDURAL), 0.5 partial, 0.0 mismatch |

### Score adjustment

```python
constraint_boost = (
    0.20 * entity_overlap
  + 0.15 * speaker_role_match
  + 0.15 * actor_entity_match
  + 0.20 * temporal_consistency
  + 0.15 * stance_relevance
  + 0.15 * answer_target_type_match
)
# Weighted blend: 70% Level 2 score + 30% constraint boost
final_score = 0.70 * level2_score + 0.30 * constraint_boost
```

Weights are hand-tuned for Phase 1. Phase 2 replaces this with learned weighting inside the MLP.

### Interface

```python
class ConstraintScorer:
    def adjust_scores(
        self, scored_memories: List[ScoredMemory], query_frame: QueryFrame
    ) -> List[ScoredMemory]
```

### New file

`psa/constraint_scorer.py`

## Pipeline Integration

### Changes to pipeline.query()

1. Call `extract_query_frame(query)` at start (after embed, before Level 1)
2. Pass frame features to CoActivationSelector (version-gated)
3. Level 2 MemoryScorer runs as before (11-dim MLP, unchanged)
4. ConstraintScorer.adjust_scores() runs after Level 2, before packer
5. Packer receives constraint-adjusted scores

### Graceful degradation

- No query frame extractor: skip frame, constraint scorer uses all-0.5 defaults (no effect)
- No facets on memories: constraint scorer uses 0.5 for missing facets (no penalty)
- Old coactivation model: zero-padded frame features (no effect until retrained)
- No constraint scorer: skip step, Level 2 scores pass through to packer

## Files

### New files

| File | Responsibility |
|------|---------------|
| `psa/query_frame.py` | QueryFrame dataclass + extract_query_frame() (pattern matcher + LLM fallback) |
| `psa/conversation_parser.py` | Structured conversation parser preserving role + timestamp |
| `psa/facet_extractor.py` | Heuristic facet extraction from raw source text |
| `psa/constraint_scorer.py` | Rule-backed constraint scoring (post-Level-2) |
| `tests/test_query_frame.py` | Unit tests |
| `tests/test_conversation_parser.py` | Unit tests |
| `tests/test_facet_extractor.py` | Unit tests |
| `tests/test_constraint_scorer.py` | Unit tests |

### Modified files

| File | Change |
|------|--------|
| `psa/consolidation.py` | Extend LLM prompt to return facets; accept structured turns |
| `psa/convo_miner.py` | Use conversation_parser instead of normalize() for PSA path |
| `psa/memory_object.py` | Add entities, actor_entities, speaker_role, stance, mentioned_at fields + DB columns |
| `psa/coactivation.py` | Version-gated query_proj expansion (768 -> 779 when query_frame_dim > 0) |
| `psa/pipeline.py` | Add query frame extraction + constraint scorer to query flow |
| `psa/cli.py` | Add `psa repair --backfill-facets` command |

### Unchanged in Phase 1

- `psa/memory_scorer.py` -- MLP stays at 11 dims (constraint features are rule-backed, not learned)
- `psa/training/memory_scorer_data.py` -- training labels unchanged (learned constraints deferred to Phase 2)
- `psa/normalize.py` -- stays as-is (new parser is separate)

## Success Criteria

| Metric | Target | Current |
|--------|--------|---------|
| R@5 | >= 0.93 | 0.920 |
| Exact F1 | >= 0.22 | 0.177 |
| LLM-judge | >= 0.40 | 0.358 |
| Query frame extraction latency | < 5ms pattern matcher, < 500ms with LLM fallback |
| Facet backfill throughput | 1000 memories/min with heuristic extractor |

## Constraints

- Laptop-only (Apple Silicon MPS, no cloud GPU required)
- LLM calls use call_llm() with cloud API + local Ollama fallback
- All models on MPS (no mixed devices)
- Backward compatible: old coactivation models get zero-padded frame input; memories without facets get 0.5 (neutral) constraint scores
- normalize.py is not modified -- new conversation_parser.py is a separate module
- MemoryReRanker MLP is not modified in Phase 1 -- constraint scoring is rule-backed

## Phase 2 Preview (not in scope)

- Fold constraint features into MemoryReRanker MLP (11 -> 16+ dims)
- Train with LLM-judged constraint satisfaction labels (not lexical F1)
- Hard negatives: right topic wrong time, right entity wrong speaker
- Evidence bundle retrieval (primary + linked support memories)
- Abstention calibration
