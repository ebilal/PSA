# PSA Observability Tooling + LongMemEval Benchmarking — Design Spec

**Date:** 2026-04-10  
**Status:** Approved  
**Scope:** Two independent features sharing one implementation plan

---

## Overview

Two additions to the PSA system:

1. **Observability tooling** — CLI + Python API to inspect exactly what context gets injected into Claude's context window via the MCP server, with persistent query logging for regression tracking.
2. **LongMemEval benchmarking** — A harness to establish a baseline score on the LongMemEval benchmark, generate oracle labels from benchmark signal, and measure selector improvement after training.

---

## Part 1 — Observability Tooling

### Problem

PSA injects formatted context into Claude via the `psa_atlas_search` MCP tool. Currently there is no way to see what text Claude actually receives, which anchors were chosen, or whether a system change (atlas rebuild, selector retrain) changed the output for a given query.

### Solution

A new `psa/inspect.py` module (importable Python API) with a `psa inspect` CLI entry point, plus persistent query logging.

### Data Model

```python
@dataclass
class CandidateTrace:
    anchor_id: str
    bm25_score: float
    dense_score: float
    rrf_score: float        # combined RRF rank score
    selected: bool
    selector_score: float   # cosine or cross-encoder score

@dataclass
class PackerSectionTrace:
    role: str               # "FAILURE", "EPISODIC", "SEMANTIC", etc.
    memory_count: int
    token_cost: int
    items: List[str]        # actual text items packed into this section

@dataclass
class InspectResult:
    run_id: str             # "{timestamp}_{query_hash[:6]}" — stable ID for diffing
    query: str
    tenant_id: str
    context_text: str       # exact string Claude receives via psa_atlas_search
    tokens_used: int
    token_budget: int
    sections: List[PackerSectionTrace]      # per-role breakdown
    selected_anchors: List[str]             # winning anchor IDs
    candidates: List[CandidateTrace]        # all 24 candidates (verbose only)
    timing: QueryTiming                     # per-stage ms from existing PSAResult
```

### Integration Points

- `psa/inspect.py` wraps `PSAPipeline` directly — no changes to pipeline internals.
- `AnchorCandidate` already carries BM25/dense scores. `SelectedAnchor` already carries selector scores. Both are exposed in `PSAResult.candidates` and `PSAResult.selected_anchors`.
- `PackedContext` currently builds `PackedSection` objects internally but does not expose them. **One change required:** add `sections: List[PackedSection]` as a public field on `PackedContext` (already built, just needs to be kept on the dataclass instead of discarded).

### CLI Interface

```
# Interactive inspection
psa inspect "query"                     # brief: context text + anchor summary + token stats
psa inspect "query" --verbose           # full trace: all candidates + scores, per-section breakdown
psa inspect "query" --tenant myproject  # target a specific tenant

# Query log management
psa log list                            # show recent logged queries (run_id, query, tokens, timestamp)
psa log show <run_id>                   # full InspectResult for a past run
psa log diff <run_id_a> <run_id_b>      # diff two runs: which anchors changed, token delta
psa log diff --query "query"            # auto-find last 2 runs for this query and diff them
```

### Default (brief) Output Format

```
Query: "what auth pattern did we use?"
Tenant: default | Run: 20260410_a3f91c

CONTEXT INJECTED (1 847 tokens / 6 000 budget)
─────────────────────────────────────────────
FAILURE WARNINGS
- Auth token stored in localStorage — this caused XSS exposure in March

PROCEDURAL GUIDANCE
- Always validate JWT expiry server-side before trusting claims

FACTS & CONCEPTS
- We use RS256 signed JWTs with 15-minute expiry and refresh tokens

─────────────────────────────────────────────
Anchors selected (2/24): auth-jwt-pattern, session-security-incidents
Timing: embed 12ms | retrieve 34ms | select 8ms | fetch 5ms | pack 3ms | total 62ms
```

### Verbose Additions

```
ANCHOR CANDIDATES (24 total, 2 selected)
  ✓ auth-jwt-pattern          rrf=0.94  bm25=0.81  dense=0.88  selector=0.91
  ✓ session-security-incidents rrf=0.87  bm25=0.72  dense=0.79  selector=0.84
  ✗ oauth-provider-setup      rrf=0.61  bm25=0.55  dense=0.64  selector=0.43
  ... (21 more)

PACKER SECTIONS
  FAILURE WARNINGS     1 item    87 tokens
  PROCEDURAL GUIDANCE  1 item    62 tokens
  FACTS & CONCEPTS     3 items  201 tokens
  (EPISODIC: empty — no episodic memories under selected anchors)
```

### Persistent Query Log

- Location: `~/.psa/tenants/{tenant_id}/query_log.jsonl`
- Format: one JSON object per line (full `InspectResult.to_dict()`)
- Written automatically on every `psa inspect` call
- No rotation at this stage — file grows indefinitely

### README Updates

Two new sections in the top-level `README.md`:

**MCP Server / Claude CLI** section covering:
- Install command: `claude mcp add psa -- python -m psa.mcp_server`
- What tools Claude has access to (`psa_atlas_search`, `psa_store_memory`, etc.)
- The PALACE_PROTOCOL Claude follows on wake-up
- What `psa_atlas_search` returns (the formatted context text)

**Observability** section covering:
- `psa inspect` with example output
- `psa log diff` for regression tracking

The `psa/README.md` (package-level, currently outdated with old mempalace module names) is replaced with an accurate module table reflecting current code.

---

## Part 2 — LongMemEval Benchmarking

### Problem

PSA has a trained anchor selector but no external benchmark to validate whether it actually improves retrieval quality. The LongMemEval benchmark provides multi-session conversation histories and factual questions — a direct match for PSA's ingest + query pipeline. Using it enables:
- A baseline score (PSA as-is, before training)
- Oracle labels derived from benchmark failures
- A post-training score to measure selector improvement

### Dataset

HuggingFace: `xiaowu0162/longmemeval`  
Two splits: `val` (used for training signal), `test` (final evaluation only).  
Format: session histories + questions + gold answers.

### Architecture

```
longmemeval ingest   →  convo_miner.py (existing)  →  MemoryStore (isolated tenant)
                     →  atlas build (automatic)

longmemeval run      →  PSAPipeline.query(question)
                     →  LLM generates answer from packed context
                     →  results written to ~/.psa/benchmarks/longmemeval/

longmemeval score    →  exact/F1 match + LLM-as-judge
                     →  oracle labels written to oracle_labels.jsonl (existing format)
                     →  psa train picks up labels (no changes needed)
```

### Sub-commands

#### `psa benchmark longmemeval ingest [--tenant longmemeval_bench]`

1. Downloads `xiaowu0162/longmemeval` from HuggingFace (`datasets` library).
2. Converts each session to the conversation transcript format `convo_miner.py` expects.
3. Runs ingestion into an isolated tenant (default: `longmemeval_bench`) to avoid polluting the user's real memory.
4. Automatically runs `psa atlas build` when ingestion completes.

One-time setup. Re-running is idempotent (skips already-ingested sessions by source hash).

#### `psa benchmark longmemeval run [--split val|test] [--limit N] [--tenant longmemeval_bench]`

For each question in the split:
1. Calls `PSAPipeline.query(question)` on the bench tenant.
2. Sends `packed_context + question` to the configured LLM to generate an answer.
3. Writes one record per question to `~/.psa/benchmarks/longmemeval/results_{timestamp}.jsonl`:

```json
{
  "question_id": "q_0042",
  "question": "What did I say about my dog's vet visit?",
  "context_text": "RELEVANT PRIOR EPISODES\n- ...",
  "answer_generated": "You mentioned your dog had a checkup in January...",
  "answer_gold": "The dog had a routine checkup, everything was fine.",
  "tokens_used": 1240,
  "token_budget": 6000,
  "selected_anchors": ["pets-health-events"],
  "timing_ms": {"total": 71.3}
}
```

`--limit N` runs only the first N questions (useful for quick sanity checks).  
Default split is `val` — `test` split is reserved for final evaluation.

#### `psa benchmark longmemeval score [--results <file>] [--method exact|llm|both]`

Reads the results file and scores each question:

- **exact** — string normalization + F1 token overlap (fast, no LLM cost)
- **llm** — sends `(question, gold, generated)` to the configured LLM via the existing `psa/llm.py` caller, asks pass/fail with reason
- **both** — runs both, reports agreement rate

Output:
```
LongMemEval Score (val split, 500 questions)
  Exact F1:       0.41
  LLM-as-judge:   0.53  (agreement with exact: 78%)

Oracle labels written: 234  →  ~/.psa/tenants/longmemeval_bench/oracle_labels.jsonl
Run 'psa train' to train the selector on these labels.
```

Failures are written as oracle labels in the **exact same format** as `OracleLabeler` produces — `psa train` picks them up with zero changes.

### Baseline → Train → Measure Loop

```bash
# One-time setup
psa benchmark longmemeval ingest

# Establish baseline
psa benchmark longmemeval run --split val
psa benchmark longmemeval score           # note baseline score, oracle labels written

# Train selector
psa train                                 # existing command

# Measure improvement
psa benchmark longmemeval run --split val --limit 500
psa benchmark longmemeval score --method both
```

### Files Added

```
psa/benchmarks/__init__.py
psa/benchmarks/longmemeval.py       # all three sub-commands + scoring logic
~/.psa/benchmarks/longmemeval/      # results files (not in repo)
```

The existing `psa benchmark` CLI entry point in `cli.py:cmd_benchmark()` is extended to route `psa benchmark longmemeval <subcmd>`. The existing argument parser gets a `longmemeval` sub-parser.

### Dependencies

- `datasets` (HuggingFace) — added as optional dependency under `psa[benchmark]` extra in `pyproject.toml`
- No other new dependencies — uses existing `PSAPipeline`, `convo_miner`, `llm.py`, oracle label format

---

## Out of Scope

- Web dashboard or real-time monitoring UI
- Automatic log rotation or log size limits
- Support for benchmarks other than LongMemEval
- Uploading results to any external service

---

## Implementation Order

1. Expose `sections` on `PackedContext` (one-line change, unblocks inspect)
2. `psa/inspect.py` — `InspectResult` dataclass + `inspect_query()` function
3. `psa inspect` + `psa log` CLI commands
4. README updates (MCP section + Observability section + replace `psa/README.md`)
5. `psa/benchmarks/longmemeval.py` — `ingest` sub-command
6. `longmemeval run` sub-command
7. `longmemeval score` sub-command + oracle label writer
8. Extend `psa benchmark` CLI routing
