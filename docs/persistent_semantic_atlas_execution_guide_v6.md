# Persistent Semantic Atlas
## End-to-end execution guide for agent memory

## 1. Purpose

This document is the consolidated and corrected version of the **Persistent Semantic Atlas (PSA)** design.

The goal is not to build a mystical "general memory" system. The goal is concrete:

> given a new task, retrieve the most useful prior experience under a strict token budget, more reliably than standard RAG.

PSA is therefore a **memory retrieval substrate for agents**. It improves on naive RAG by storing and retrieving **memory objects** derived from prior experience rather than relying only on similarity search over raw chunks.


---

## 2. What PSA is and is not

### PSA is
- a tenant-scoped memory system for agents
- a consolidation pipeline that converts raw experience into reusable memory objects
- a versioned atlas of stable anchors over those memory objects
- a runtime retrieval stack that selects memory by **task utility**, not just semantic closeness
- a packed-context builder that feeds the final model only the most useful memory for the task

### PSA is not
- a lossless compression of an unbounded database into one prompt
- an LLM that scans the full database at runtime
- a guarantee that one atlas version lasts forever
- a replacement for raw storage, logs, ACLs, or online working memory
- a full cognitive architecture by itself

---

## 3. Design principles

1. **Store memory objects, not just text chunks.**
2. **Keep raw provenance.** Every memory object must link back to source records and chunks.
3. **Separate routing from evidence.** First choose memory regions, then fetch detailed memory objects and source evidence.
4. **Keep runtime cheap.** The hot path uses a fast retriever and a small reranker, not a large LLM over the whole atlas.
5. **Version the atlas.** Anchors are stable within an atlas version, not forever.
6. **Use one atlas per tenant or security domain.**
7. **Train against downstream utility.** The selector should learn which memory helps the final task model succeed, not just which memory sounds similar.
8. **Make drift explicit.** Rebuilds are part of the design, not an admission of failure.

---

## 4. Memory model

PSA stores multiple memory types because agent memory is not only semantic recall.

## 4.1 Memory types

### A. Episodic memory
A record of a prior run or interaction.

Required fields:
- `episode_id`
- `task_type`
- `initial_request`
- `goal`
- `environment`
- `tools_used`
- `key_decisions`
- `critical_observations`
- `outcome`
- `success_label`
- `quality_score`
- `compressed_lesson`
- `source_links`

### B. Semantic memory
A stable fact, concept, explanation, or mapping abstracted from one or more sources.

Required fields:
- `concept_id`
- `statement`
- `summary`
- `confidence`
- `evidence_links`
- `validity_interval`

### C. Procedural memory
A reusable method, playbook, checklist, or action skeleton.

Required fields:
- `procedure_id`
- `trigger_conditions`
- `preconditions`
- `ordered_steps`
- `expected_outcome`
- `known_failure_points`
- `tool_requirements`
- `evidence_links`

### D. Failure memory
A reusable warning about what goes wrong.

Required fields:
- `failure_id`
- `failure_condition`
- `symptoms`
- `root_cause`
- `mitigation`
- `related_procedures`
- `evidence_links`

### E. Tool-use memory
A stable observation about tool behavior.

Required fields:
- `tool_memory_id`
- `tool_name`
- `scenario`
- `recommended_parameters`
- `known_bad_parameters`
- `failure_conditions`
- `retry_or_backoff_policy`
- `evidence_links`

### F. Working-memory derivative
A compact state artifact from a completed task worth preserving.

Examples:
- final constraint summary
- unresolved issue summary
- stable user preference learned from the episode
- partial plan that repeatedly proves useful

Only persist this type after consolidation. Do not dump live scratchpad content into long-term memory.

---

## 4.2 Canonical memory object schema

All persisted memory should use one canonical wrapper.

```json
{
  "tenant_id": "tenant_123",
  "memory_object_id": "mo_001",
  "memory_type": "episodic|semantic|procedural|failure|tool_use|working_derivative",
  "source_ids": ["src_101", "src_102"],
  "created_at": "...",
  "updated_at": "...",
  "title": "...",
  "body": "...",
  "summary": "...",
  "metadata": {
    "task_type": "...",
    "tool_names": ["..."],
    "success_label": true,
    "quality_score": 0.87,
    "validity_interval": null,
    "acl_scope": "tenant_123"
  },
  "evidence_chunk_ids": ["ch_111", "ch_112"],
  "embedding_ref": "vec://memory/mo_001",
  "primary_anchor_id": "A118",
  "secondary_anchor_ids": ["A144"],
  "assignment_confidence": 0.91
}
```

---

## 5. Security and tenancy

### Decision
**One atlas per tenant or security domain.**

Do not mix tenants in one atlas.

Reason:
- anchor cards summarize memory regions
- mixed-tenant anchor cards can leak information before evidence fetch
- ACL filtering only at evidence time is too late

Therefore:
- source ingestion is tenant-scoped
- consolidation is tenant-scoped
- atlas induction is tenant-scoped
- selector training is tenant-scoped or security-domain-scoped
- runtime retrieval and packing are tenant-scoped

---

## 6. Source ingestion and chunking

Chunks are not the main memory abstraction, but they remain necessary for provenance, evidence retrieval, and auditability.

## 6.1 Raw source schema

```json
{
  "tenant_id": "tenant_123",
  "source_id": "src_101",
  "source_type": "transcript|document|chat|issue|note|html|markdown",
  "created_at": "...",
  "updated_at": "...",
  "title": "...",
  "full_text": "...",
  "metadata": {
    "author": "...",
    "speaker_map": {},
    "timestamp_range": null,
    "acl_scope": "tenant_123"
  }
}
```

## 6.2 Chunking strategy

Chosen strategy: **structure-aware hierarchical chunking**.

### Procedure
1. Parse structure using the best available boundaries:
   - headings
   - sections
   - paragraphs
   - speaker turns
   - lists
   - table blocks
   - code blocks if present
2. Create section chunks.
3. Split sections into coherent mid chunks.
4. Split mid chunks into fine chunks.
5. Preserve parent-child links.
6. Only use small overlap where boundary safety requires it.

### Target sizes
- fine chunk: 80–180 tokens
- mid chunk: 220–450 tokens
- section chunk: 500–1500 tokens

### Fallback rule
If structural parsing fails:
1. sentence segmentation
2. paragraph grouping
3. fixed token windows as last resort only

Mark fallback-parsed sources with `parse_quality = low`. This must reduce consolidation confidence.

## 6.3 Chunk schema

```json
{
  "tenant_id": "tenant_123",
  "source_id": "src_101",
  "chunk_id": "ch_201",
  "level": "fine|mid|section",
  "parent_section_id": "sec_12",
  "parent_mid_chunk_id": "mid_88",
  "start_offset": 1200,
  "end_offset": 1460,
  "text": "...",
  "metadata": {
    "section_title": "Root cause analysis",
    "speaker": null,
    "timestamp_start": null,
    "timestamp_end": null,
    "parse_quality": "high",
    "acl_scope": "tenant_123"
  },
  "embedding_ref": "vec://chunks/ch_201"
}
```

---

## 7. Consolidation layer

This is the key layer that turns retrieval into memory.

## 7.1 Goal
Convert raw traces and chunks into typed memory objects with durable utility.

## 7.2 Chosen offline model
Use **`Qwen2.5-7B-Instruct`** for offline consolidation, synthetic query generation, and adversarial rewrite generation in V1.

Reason:
- capable enough for structured extraction and summarization
- practical to self-host for batch offline work
- cheaper than relying on frontier models for every offline pass
- already consistent with the rest of the V4 design choices

This model is **not** the runtime selector. It is an offline worker.

## 7.3 Exact consolidation workflow

For each completed episode or ingested source:

1. load raw source + chunks + metadata + tool logs if available
2. run a structured extraction prompt with `Qwen2.5-7B-Instruct`
3. ask for zero or more memory objects of the allowed types
4. require chunk-level evidence references for each important claim
5. require a retention score in `[0,1]`
6. require uncertainty fields when evidence is weak
7. reject malformed outputs and re-run once with repair prompt if needed
8. deduplicate against nearest existing memory objects of the same type
9. persist only objects above the retention threshold

## 7.4 Retention rule
Persist a memory object only if:
- `retention_score >= 0.65`
- and at least one of the following is true:
  - reusable successful strategy
  - costly failure worth avoiding
  - stable fact or concept
  - recurring tool behavior
  - durable user or workflow constraint
  - episode likely to be retrieved again

Do not persist:
- transient scratchpad chatter
- repetitive low-value traces
- unsupported speculation
- unresolved internal reasoning with no reusable value

## 7.5 Deduplication rule
Before writing a new object:
1. retrieve nearest existing objects of the same memory type
2. compare title, summary, and body similarity
3. if cosine similarity > 0.92 and semantics match, update existing object rather than creating a duplicate
4. preserve evidence links from both old and new supporting chunks

---

## 8. Embedding and indexing choices

## 8.1 Chosen embedding model
Use **`BAAI/bge-base-en-v1.5`** as the single embedding model for one atlas version.

Reason:
- good quality for English-first semantic retrieval
- inexpensive and easy to self-host
- stable enough for V1
- compatible with FAISS-based dense retrieval

## 8.2 What gets embedded
Embed:
- memory object `title + summary + body`
- anchor cards
- fine chunks
- optional mid chunks for evidence expansion

Do **not** use full raw source documents as the main retrieval unit.

## 8.3 Preprocessing
For each embedded text:
1. normalize whitespace
2. preserve headings as inline markers
3. strip repeated boilerplate where possible
4. prepend memory-type marker for memory objects, e.g. `[TYPE=PROCEDURAL]`

## 8.4 Normalization and metadata
- always L2-normalize vectors before retrieval or clustering
- store embedding model name, version, dimension, normalization rule, timestamp, and atlas version
- if embedding model changes materially, build a **new atlas version**

## 8.5 Retrieval indices
Use:
- **BM25** for sparse retrieval over anchor cards
- **FAISS** for dense retrieval over anchor card embeddings
- FAISS or direct vector store for memory-object lookup if needed

This is the V1 exact retrieval stack.

---

## 9. Atlas induction

## 9.1 Goal
Create a bounded set of anchors that act as stable semantic addresses over memory objects.

## 9.2 Chosen atlas size for V1
- **224 learned anchors**
- **32 novelty anchors**
- total **256 anchors per tenant per atlas version**

This preserves the V4 design decision.

## 9.3 Induction input
Induce anchors from **memory object embeddings**, not raw chunks.

Why:
- memory objects are already semantically consolidated
- raw chunks are too noisy and literal
- the atlas should represent reusable memory, not formatting artifacts

## 9.4 Two-stage induction process

### Stage A: coarse partitioning
Run **mini-batch spherical k-means** on normalized memory object embeddings.

Configuration:
- `k = 224`
- `seeds = 3`
- `max_iterations = 200`
- minibatch size set by corpus and hardware

### Stage B: semantic refinement
For each cluster:
1. sample 10 central memory objects
2. sample 10 diverse internal memory objects
3. sample 10 boundary memory objects
4. sample 10 nearest external negatives
5. use the offline model to draft anchor card text
6. review purity and separability
7. split impure clusters
8. merge near-duplicate clusters
9. rerun assignment for affected objects

The geometric step is only the first pass. The final atlas is established after refinement and review.

## 9.5 Stability requirement
Run induction at least 3 times with different seeds.
Reject unstable proposals if:
- too many anchors are inconsistent across runs
- anchor purity is low
- inter-anchor overlap is too high
- many anchors are obviously artifacts rather than semantic regions


## 10. Anchor cards and assignments

## 10.1 Anchor card format
Each anchor card must be compact, structured, and operational.

```text
[Anchor ID] A118
[Name] Outbound call reschedule handling
[Meaning] Memory about detecting, validating, and executing patient reschedule requests.
[Memory types] episodic; procedural; failure; tool_use
[Include] confirmed callback requests; timezone handling; callback-date validation; successful reschedule workflows
[Exclude] generic refusal handling; appointment cancellation unrelated to callback workflow
[Prototype examples] "call me next Tuesday"; "wrong timezone caused callback error"; "validated UTC plus local time before scheduling"
[Near but different] A121 refusal handling; A144 callback extraction failures
```

Cards must be:
- short enough for fast retrieval and reranking
- discriminative, not decorative
- grounded in real memory objects
- explicit about memory types present
- stable in format across anchors

## 10.2 Assignment policy
Each memory object receives:
- exactly one primary anchor
- optionally one secondary anchor

## 10.3 Why keep secondary assignment
Secondary assignment is only for genuinely dual-use memory.
It helps with:
- bridge cases
- dual-use procedures
- overlapping failure and procedure families
- retrieval fallback when one region is underspecified

Target rate:
- secondary assignment on no more than 20% of memory objects

## 10.4 Assignment procedure
1. compute similarity to all anchor cards
2. shortlist top 5 anchors
3. rerank `(memory_object, anchor_card)` pairs using the same selector architecture class as runtime reranking
4. assign primary to top anchor
5. assign secondary only if the second score is within a narrow margin and the second anchor adds real retrieval value
6. store assignment confidence

Low-confidence assignments feed drift monitoring and human review queues.

---

## 11. Runtime pipeline

## 11.1 Step 1: fast anchor retrieval
At query time, first run a cheap non-LLM retriever over anchor cards.

Chosen retrieval method:
- BM25 over anchor card text
- dense similarity over anchor card embeddings
- fused ranking

Output:
- top **24** candidate anchors

## 11.2 Step 2: selector reranking
The selector sees:
- the user query or task
- the 24 shortlisted anchor cards

The selector outputs:
- **1 to 4** anchor IDs
- score per anchor
- optional short rationale for debugging only

### Chosen selector model
Use **`cross-encoder/ms-marco-MiniLM-L-6-v2`** as the starting checkpoint, then fine-tune it on atlas-specific selector data.

Why this choice:
- small and fast enough for real-time reranking
- already strong on pairwise relevance ranking
- easy to fine-tune on limited hardware
- safer than putting a general-purpose LLM in the hot path

### Exact inference rule
1. score all 24 shortlisted anchor cards independently
2. sort by score descending
3. keep anchors with score >= threshold `τ`
4. return at least 1 anchor and at most 4 anchors
5. if more than 4 pass threshold, keep top 4 only

Tune `τ` on the held-out real-query development set.

## 11.3 Step 3: memory fetch
For each selected anchor:
- fetch linked memory objects
- rank them by task utility and memory type
- fetch supporting fine chunks only as evidence backing
- expand to parent chunks only if needed for context

## 11.4 Step 4: deterministic evidence packer
The packer builds a global context pack for the final model.

Chosen policy:
1. rank candidate memory objects by local relevance and type utility
2. include a balanced mix when needed:
   - episodic examples
   - procedural guidance
   - failure warnings
   - tool-use notes
3. deduplicate by object ID, text hash, and similarity threshold
4. greedily fill a **global 6000-token budget**
5. attach supporting source extracts only where they materially strengthen confidence or execution

This avoids another hot-path LLM call in V1.

## 11.5 Step 5: final model input
The final model receives:
- the current task or query
- the selected anchor cards
- the packed memory context
- optional supporting source evidence

It does **not** receive the full atlas in normal operation.

---

## 12. Packed context design

Recommended layout:

```text
TASK
...

SELECTED MEMORY REGIONS
[A118] ...
[A144] ...

PROCEDURAL GUIDANCE
- ...
- ...

RELEVANT PRIOR EPISODES
- Episode ... outcome ... lesson ...
- Episode ... outcome ... lesson ...

FAILURE WARNINGS
- ...
- ...

TOOL-USE NOTES
- ...
- ...

SUPPORTING EVIDENCE
- source extract ...
- source extract ...
```

The final model should see memory by **role**, not as a flat retrieval dump.

---

## 13. Latency budget

Reasonable target:
- anchor retrieval: 5–15 ms
- selector rerank: 10–40 ms
- memory fetch + pack: 20–80 ms
- final model: dominant cost

Critical requirement:
- selector + packer must never dominate time-to-first-token

If the selector path becomes the main latency cost, PSA loses its advantage.

---

## 14. Oracle labeling for selector training

The selector must be trained against **task utility under a token budget**, not raw semantic closeness.

## 14.1 Candidate generation
For each training query:
1. run hybrid retrieval over anchor cards
2. keep the top 8 anchors for oracle construction
3. score:
   - top 8 singles
   - top 10 pairs
   - top 5 triples
   - top 2 quadruples only for queries tagged as high complexity

Do not exhaustively enumerate all combinations.

## 14.2 Oracle scoring function

\[
OracleScore = 0.45 \cdot SupportCoverage + 0.20 \cdot TaskSuccess + 0.15 \cdot ProceduralUtility - 0.10 \cdot NoisePenalty - 0.10 \cdot TokenCost
\]

Where:
- `SupportCoverage` measures whether selected anchors cover the needed evidence or memory objects
- `TaskSuccess` measures end-task correctness or answer quality using a frozen offline teacher
- `ProceduralUtility` rewards memory that improves action choice, not just topical overlap
- `NoisePenalty` penalizes irrelevant extra anchors
- `TokenCost` penalizes large packed contexts

Keep this formulation simple in V1. Do not add more reward terms yet.

## 14.3 Chosen offline teacher
Use the **same final task model that will be used at runtime** as the expensive scorer for `TaskSuccess` on the top candidate sets.

Reason:
- selector labels should match the real downstream model's utility
- this avoids training the selector against one generator and deploying against another

If runtime-model cost is too high, use a frozen **`Qwen2.5-7B-Instruct`** teacher as the temporary scorer, but then keep the runtime task model fixed during V1.

## 14.4 Cost control rule
Only run the expensive `TaskSuccess` term on the top 3 candidate sets after cheap proxy scoring.

This is mandatory for budget control.

---

## 15. Training data

## 15.1 Main training source
The main selector training data comes from **your own corpus of episodes and derived memory objects**.

This is non-negotiable. The selector is atlas-specific. Public datasets cannot substitute for your internal memory schema.

## 15.2 Query families to generate
Generate at least these five families:
1. direct single-anchor factual or procedural queries
2. contrastive boundary queries
3. two-anchor compositional queries
4. three-anchor bridge queries
5. experience-based queries asking for prior successful or failed patterns

Examples:
- "What worked last time when this tool timed out?"
- "How should I handle a callback reschedule with timezone ambiguity?"
- "What prior episodes resemble this failure?"
- "What is the safest procedure here?"

## 15.3 Public-data policy
Do **not** use a public dataset as the main selector training source.

Chosen public-data use for V1:
- use **NanoBEIR** only as a lightweight external regression test for retrieval-stack sanity
- use it to compare dense vs hybrid retrieval settings
- do not use it to claim end-task memory performance

## 15.4 Adversarial hardening
Use adversarial rewrites only after supervised warm start.

Allowed rewrites:
- lexical compression
- alias substitution
- distractor insertion
- conversational phrasing
- reference flattening
- messy shorthand

Do not allow rewrites that change required evidence or task semantics.

## 15.5 Real-query evaluation set
Collect a held-out human-written query set early.

Minimum target:
- 200–500 real queries
- never used in training
- used as the main trusted evaluation set

This is mandatory.

---

## 16. Selector training approach

## 16.1 Training sequence

### Phase 1: supervised warm start
Train the selector on oracle-labeled query-anchor pairs.

### Phase 2: hard-negative curriculum
Add:
- nearby anchors
- wrong but tempting procedural anchors
- wrong failure families
- semantically adjacent tool-use anchors

### Phase 3: adversarial hardening
Add verified adversarial rewrites that preserve semantics.

### Explicit non-choice for V1
Do **not** start with RL or GAN-style co-training.

Reason:
- too unstable
- too easy to reward-hack
- too expensive relative to likely benefit in V1

## 16.2 Exact training format
Convert each query into pairwise training examples over the top-24 shortlisted anchors.

For each query:
- positives = anchors in the oracle-selected set
- hard negatives = shortlisted anchors not in the oracle set
- easy negatives = randomly sampled anchors outside the shortlist

Each training item is:
- input text: `query [SEP] anchor_card`
- label: `1` if positive, `0` otherwise

This is the exact training format for V1.

## 16.3 Loss function
Use:
- binary cross-entropy over all query-anchor pairs
- plus margin-ranking loss on the hardest negative in each batch

\[
L = L_{BCE} + 0.2 \cdot L_{margin}
\]

## 16.4 Starting hyperparameters
- base model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- max sequence length: 320 tokens
- optimizer: AdamW
- learning rate: `2e-5`
- weight decay: `0.01`
- warmup ratio: `0.1`
- batch size: `32`
- gradient accumulation: `2` if needed
- epochs: `3`
- early stopping metric: real-query development-set task success

Do not over-tune on synthetic validation loss.

## 16.5 Data mix per epoch
- 60% synthetic internal queries
- 20% hard-negative augmented internal queries
- 20% adversarially rewritten internal queries

Do not mix public dataset examples into selector fine-tuning in V1.

## 16.6 Stop rule
Stop training when one of the following happens:
- real-query dev-set task success stops improving for 2 evaluations
- selector hit rate improves but end-task success does not
- selector begins returning broader anchor sets without task gains

---

## 17. Exact V1 model stack

Use this exact stack in V1:
- embeddings: `BAAI/bge-base-en-v1.5`
- sparse retrieval: BM25
- dense retrieval index: FAISS over anchor-card embeddings
- selector: fine-tuned `cross-encoder/ms-marco-MiniLM-L-6-v2`
- offline consolidation and query generation: `Qwen2.5-7B-Instruct`
- final runtime task model: your production answer / planner model

This is the recommended exact stack for the first build.

---

## 18. Evaluation and acceptance gates

## 18.1 Required baselines

### Baseline A
Dense retrieval + strong reranker + top-k chunks into the final model.

### Baseline B
PSA without selector training, using shortlist retrieval only.

### Baseline C
PSA with trained selector and deterministic packer.

## 18.2 Required metrics
Track at least:
- shortlist recall@24
- selector anchor hit rate
- end-task success on held-out real queries
- packed context token count
- latency per stage
- novelty assignment rate
- atlas health metrics

## 18.3 Minimum deployment gates
Suggested starting gates:
- shortlist recall@24 >= 0.95
- selector improves real-query task success over Baseline A
- packed context tokens reduced by >= 30% vs top-k chunk baseline at matched or better task success
- selector + packer latency <= 120 ms median
- novelty assignment rate remains within the tolerance window defined below

Set numeric gates before experiments begin.

---

## 19. Atlas health, drift, and rebuild policy

A frozen-forever atlas is a bad assumption.

## 19.1 Health metrics
Track continuously:
- novelty assignment rate
- rolling selector recall decay
- anchor purity drift
- fraction of queries routed to novelty anchors
- fraction of low-confidence assignments
- distribution shift in memory-type composition per anchor

## 19.2 Trigger thresholds
Example trigger policy:
- novelty assignment rate > 8%
- or selector recall drops by > 5%
- or too many anchors exceed purity drift threshold

Any one of these triggers human review.

## 19.3 Rebuild policy
When triggered:
1. keep current atlas live as version `N`
2. build candidate atlas `N+1` offline
3. evaluate `N+1` against held-out real queries
4. create mapping from old anchors to new anchors where possible
5. migrate gradually

Do not pretend no rebuild will ever be needed.

---

## 20. Why this improves on RAG as agent memory

Standard RAG usually:
- stores chunks independently
- retrieves by semantic similarity
- dumps top-k chunks into context

That is weak for agent memory because it does not naturally preserve:
- procedures
- failures
- tool behavior
- prior outcomes
- reusable lessons

PSA improves on that by:
- introducing a consolidation layer
- organizing memory around reusable memory objects
- routing through stable memory regions
- selecting memory by task utility rather than topical similarity alone
- packing context by memory role rather than by source chunk score alone

This still does not make PSA a full cognitive architecture, but it makes it a much better memory substrate than ordinary RAG.

---

## 21. End-to-end execution plan

## Phase 0: baseline first
Before committing fully, run the strong reranked-RAG baseline.

If it solves most of the problem cheaply, stop.

## Phase 1: corpus and consolidation
- build parsing and chunking
- build memory consolidation pipeline
- define retention policies
- generate first memory object store

## Phase 2: atlas induction
- embed memory objects
- induce anchors
- refine anchors
- write anchor cards
- review manually

## Phase 3: selector data and training
- generate oracle labels
- build synthetic internal training queries
- collect real query eval set
- train selector
- harden with hard negatives and rewrites

## Phase 4: packer and runtime
- implement fast retriever
- implement selector
- implement deterministic packer
- connect final model
- measure latency and quality

---

## 22. Final formulation

> Build a tenant-scoped, versioned semantic atlas over consolidated agent memory objects rather than raw text alone. After each episode, convert raw traces into reusable episodic, semantic, procedural, failure, tool-use, and working-derivative memories. Assign those memory objects to stable anchor regions. At query time, use a fast non-LLM retriever and a small trained selector to choose the most useful memory regions, then pack a compact context containing the right mix of prior episodes, procedures, failure warnings, tool notes, and supporting evidence for the current task.

That is the corrected and internally consistent version of the PSA idea.
