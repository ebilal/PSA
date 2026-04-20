# PSA — Persistent Semantic Atlas

PSA gives an AI agent a memory that survives beyond a single chat session.

It ingests conversation history and project files, extracts structured memory objects, organizes them into a semantic atlas, and retrieves the most relevant context when the agent needs it again. The system is laptop-first: SQLite for storage, local embeddings, and an optional local LLM. Nothing has to leave your machine unless you explicitly configure a cloud model.

PSA works with Claude Code, Codex, and any client that can talk to an MCP server or run session hooks.

---

## What PSA is for

Use PSA when you want an agent to remember things like:

- why a design decision was made
- how a deployment or debugging workflow works
- what failed before and should not be repeated
- tool quirks, API gotchas, and project-specific conventions

Instead of searching a pile of raw transcripts, PSA stores typed memories such as failures, procedures, tool-use notes, episodes, and semantic facts. That makes retrieval more useful at answer time and lets the packer prioritize high-value context first.

## How to think about it

Most memory systems are "embedding search over chunks." PSA is trying to do something closer to recollection:

- it stores structured memory objects instead of raw chunks
- it groups related memories into semantic regions called anchors
- it activates relevant anchors first, then scores memories inside them
- it forgets low-value memories and stale query patterns over time

If you only need one sentence: PSA is a persistent memory layer for AI agents, with typed memories and atlas-based retrieval.

## Installation Guide

This is the full setup path for a PSA install that can ingest data, answer queries, and maintain itself over time through the lifecycle job. If you only want the minimum path to a first query, you can stop after Step 6.

### 1. Install dependencies

Requirements:

- Python 3.9 or newer
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) with `qwen2.5:7b` if you want local extraction

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Ollama and pull the default local model
brew install ollama
ollama pull qwen2.5:7b

# Clone the repo and install dependencies
git clone https://github.com/ebilal/PSA.git memnexus
cd memnexus
uv sync
```

If you plan to train selector models, also install the training extras:

```bash
uv sync --extra training
```

### 2. Create the PSA config

PSA can run with just defaults, but the install guide should leave you with a complete working setup, not an implicit one. Create `~/.psa/config.json` and make the lifecycle behavior explicit:

```json
{
  "tenant_id": "default",
  "psa_mode": "primary",
  "token_budget": 6000,
  "max_memories": 50000,
  "anchor_memory_budget": 100,
  "trace_queries": true,
  "nightly_hour": 0,
  "advertisement_decay": {
    "tracking_enabled": true,
    "removal_enabled": true,
    "tau_days": 45,
    "grace_days": 21,
    "sustained_cycles": 14
  }
}
```

Why this matters:

- `trace_queries` is the source of truth for query activity and is required for advertisement-ledger accumulation.
- the lifecycle job handles both memory forgetting and advertisement decay
- without an explicit config file, the current code falls back to defaults that are harder to understand from the outside

### 3. Choose an LLM backend

By default, PSA expects a local model through Ollama for memory extraction and oracle labeling.

If you want to use a cloud model instead, create or edit `~/.psa/llm.json`:

```json
{
  "provider": "cloud",
  "cloud_model": "gpt-4o-mini",
  "cloud_api_key": "sk-...",
  "cloud_api_base": "https://api.openai.com/v1",
  "local_endpoint": "http://localhost:11434/v1/chat/completions",
  "local_model": "qwen2.5:7b",
  "local_fallback": true
}
```

Set `"provider": "local"` if you want PSA to stay fully local.

### 4. Ingest some data

You need memories before PSA can answer anything useful.

```bash
# Mine conversations
uv run psa mine ~/.claude/projects/ --mode convos

# Or mine project files
uv run psa mine ~/projects/my_app
```

Re-running `mine` is safe. PSA deduplicates against `raw_sources`.

### 5. Build the atlas

Once you have roughly 200 memories, build the atlas:

```bash
uv run psa status
uv run psa atlas build
uv run psa atlas status
uv run psa atlas health
```

If the corpus is too small, `psa atlas build` will fail with `AtlasCorpusTooSmall`. Mine more data, then rerun it.

### 6. Run your first query

```bash
uv run psa search "how does the atlas work"
```

For a deeper debug view of what happened during retrieval:

```bash
uv run psa inspect "how does the atlas work" --verbose
```

### 7. Label queries and train the selector

For full Level 1 behavior, you should generate oracle labels and train the selector models.

```bash
uv run psa label --n-queries 0
uv run psa train --force
uv run psa train --force --coactivation
```

What these commands mean:

- `uv run psa label --n-queries 0` labels all currently available unlabeled queries for the tenant.
- `uv run psa label --n-queries 300` labels a batch of 300 queries instead of all of them.
- `uv run psa label --reset --n-queries 0` deletes existing labels and rebuilds the label set from scratch.
- `uv run psa train --force` trains the selector model only.
- `uv run psa train --force --coactivation` trains the selector model and then trains the co-activation model.

Normal usage is to run one or the other:

- run `uv run psa train --force` if you only want selector training
- run `uv run psa train --force --coactivation` if you want selector training plus co-activation training

You do not usually need to run both commands back to back. The `--coactivation` form already includes selector training first.

The current training defaults were selected from a bounded local sweep on real tenant data and are intended for laptop-scale retraining. The selector default learning rate is `3e-5`; the co-activation defaults remain `1e-4` / `0.01` / `16`, with the epoch default reduced to `8`.

If you are still developing locally and do not have enough labels yet, you can defer this step. PSA will continue to work in the baseline retrieval mode.

### 8. Install the lifecycle job

PSA is designed to keep itself healthy over time. Install the nightly lifecycle job once the initial atlas is working.

```bash
uv run psa lifecycle install
uv run psa lifecycle status
```

You can also run the lifecycle manually:

```bash
uv run psa lifecycle run
```

On a healthy install, lifecycle is where the system keeps itself current:

- memory forgetting archives low-value memories and enforces caps
- advertisement decay prunes stale `generated_query_patterns` on anchor cards
- atlas rebuilds, labeling, and retraining run only when health signals warrant them

Use these commands to inspect the advertisement side of lifecycle:

```bash
uv run psa advertisement status
uv run psa advertisement diff
```

## First Week Checklist

If this is your first time using PSA, this is the practical path:

1. Mine conversations or project files.
2. Confirm memories exist with `uv run psa status`.
3. Build the atlas once you have enough data.
4. Use `uv run psa search "..."` for normal lookups.
5. Use `uv run psa inspect "..." --verbose` when a retrieval looks wrong.
6. Generate labels and train the selector once you have enough real queries.
7. Install the lifecycle job so memory forgetting and advertisement decay can run automatically.
8. Add the MCP server once the CLI flow makes sense and you want low-latency interactive use.

## Using With Claude

If you want PSA available inside Claude Code, the MCP server is the main integration path.

### 1. Add the MCP server

```bash
claude mcp add psa -- uv run --project /path/to/memnexus python -m psa.mcp_server
```

Replace `/path/to/memnexus` with your local checkout path.

### 2. Verify the server is available

In Claude Code, confirm the PSA tools are present. The important ones are:

- `psa_atlas_search` for full memory retrieval
- `psa_store_memory` for writing typed memories
- `psa_atlas_status` and `psa_atlas_health` for checking system state

### 3. Use PSA in normal agent workflows

Typical flow in Claude Code:

1. Start a session in a project where PSA has already ingested data.
2. Ask Claude a question that depends on prior context.
3. Claude can call `psa_atlas_search` to retrieve packed memory context.
4. When useful new durable information appears, Claude can call `psa_store_memory`.

### 4. Optional: use hooks as well

If you want PSA to participate in session-start or harness-driven workflows, the CLI also exposes hook integration:

```bash
uv run psa hook run --hook session-start --harness claude-code
```

The MCP server is still the primary interactive path. Hooks are additive.

## Core design philosophy

**Memory is not just retrieval.** PSA does not treat memory as a flat list of chunks. It first decides which regions of memory space matter, then scores the memories inside those regions.

**Typed, structured memories.** The unit of storage is a `MemoryObject` with an explicit type: failure, procedural, tool-use, episodic, semantic, or working-derivative. Types affect packing priority, so the system can surface failures before it repeats them.

**The atlas is learned.** A one-time spherical k-means clustering identifies 256 semantic regions (up to 224 learned + 32 novelty). Each region gets an LLM-generated anchor card describing what it contains and what kinds of queries it can answer.

**Forgetting is a feature.** PSA prunes memories that stop being useful and decays stale query patterns on anchor cards so the system can stay relevant instead of growing forever.

**Laptop-first operation.** Via the MCP server, query latency is typically ~200–500ms on Apple Silicon after the one-time warmup. The CLI pays the full model-load cost on each invocation, so it is better for scripting and debugging than for interactive use.

---

## How the system works

If you want the mental model before reading the command reference, read this section. If you only want to get running, the installation guide above is enough.

### Ingestion

Raw text is stored immutably. Memory objects are derived indexes over that raw text.

```
Raw text (files or conversations)
  │
  ▼
convo_miner.py or miner.py
  │  (structure-aware chunking)
  ▼
consolidation.py
  │  (LLM extracts typed memory objects — 6 types)
  │  (quality filter: retention_score ≥ 0.65)
  │  (dedup by cosine > 0.92)
  ▼
embeddings.py  — BAAI/bge-base-en-v1.5, 768-dim, L2-normalized
  │
  ▼
memory_object.py + SQLite
     raw_sources     — immutable source records
     memory_objects  — typed objects with embeddings
```

Supported conversation formats: Claude Code JSONL, Claude AI JSON, ChatGPT exports, Codex JSONL, Slack JSON, plain text.

**Three-level chunking.** Each source is chunked at three granularities before LLM extraction: fine (80–180 tokens), mid (220–450 tokens), and section (500–1500 tokens). Chunks nest: each section contains mid chunks, each mid contains fine chunks. The LLM extracts memory objects from each fine chunk independently. This means a single document can yield many memories from different regions without losing the surrounding context that mid and section chunks provide. `quality_score` (aliased from the LLM's `retention_score`) is set once at extraction time and never updated — it reflects the LLM's confidence at the moment of ingestion, not subsequent usage.

### Atlas and anchors

Once you have ~200+ memories, `psa atlas build` runs spherical k-means (k=224 learned + 32 novelty, 3 seeds, stability-checked). The result is an `Atlas` with up to 256 `AnchorCard` objects.

**What an anchor card is.** Each card is an LLM-generated semantic description of one cluster, written at atlas build time by analyzing the memories that fall inside it. It is the anchor's identity — used by the retriever to match incoming queries, by the selector to decide which anchors to open, and by the advertisement-decay system to track which query patterns are still earning attention.

```text
+----------------------------------------------------------------------------------+
| AnchorCard                                                                       |
+----------------------------------------------------------------------------------+
| anchor_id: 42                                                                    |
| name: "schema-decisions"                                                         |
|                                                                                  |
| meaning:                                                                         |
|   Covers schema choices, migration tradeoffs, and why the project settled        |
|   on Postgres.                                                                   |
|                                                                                  |
| memory_types: ["semantic", "procedural"]                                         |
| include_terms: ["migration", "schema", "postgres", "ddl"]                        |
| exclude_terms: ["ui", "css", "frontend"]                                         |
|                                                                                  |
| prototype_examples:                                                              |
|   - JWT setup                                                                    |
|   - Postgres migration plan                                                      |
|   - Backfill strategy for user table                                             |
|                                                                                  |
| near_but_different:                                                              |
|   - API auth middleware                                                          |
|   - Admin UI workflow                                                            |
|                                                                                  |
| generated_query_patterns  <-- advertisements this anchor uses to attract queries |
|   - What did we decide about migrations?                                         |
|   - Why Postgres instead of SQLite?                                              |
|   - How do tokens expire?                                                        |
+----------------------------------------------------------------------------------+

Retriever: uses the full card text for BM25 + dense anchor matching
Selector: scores the stable card text to decide which anchors to open
Advertisement decay: reinforces or prunes stale generated_query_patterns over time
```

A card contains:

| Field | Purpose |
|---|---|
| `name` | Short human-readable label (e.g. "atlas-rebuild-triggers") |
| `meaning` | 1–3 sentences describing what belongs in this cluster |
| `include_terms` | Up to 8 keywords that signal membership |
| `exclude_terms` | Up to 4 keywords that signal non-membership |
| `prototype_examples` | Representative memory titles from this cluster |
| `near_but_different` | Memory titles that are close but belong elsewhere |
| `generated_query_patterns` | 10–15 specific questions this anchor can answer |

The `generated_query_patterns` are the anchor's **advertisements** — the queries it claims to serve. They are the unit that the advertisement-decay system tracks and prunes (see [Advertisement decay](#advertisement-decay) below).

**Anchor identity is durable across rebuilds.** When `psa atlas build` runs again, new clusters are matched to existing anchor IDs by centroid similarity. Only genuinely novel clusters get new IDs. This means trained selector models and ledger history stay valid across rebuilds.

**The refined card surface.** Once the atlas is built, `anchor_cards_refined.json` becomes the live surface that the retriever and advertisement ledger read from. Lifecycle-based advertisement decay mutates this refined surface directly. Separate refinement workflows such as `psa atlas refine`, `psa atlas curate`, and `psa atlas promote-refinement` still exist for manual card improvement, but they are not the normal advertisement-decay path.

### Query pipeline

The default path (Level 1) scores all anchors in one pass; Level 2 is the legacy retrieval-shortlist fallback.

```
Query
  │
  ▼
Embed (BAAI/bge-base-en-v1.5)
  │
  ▼
╭─ LEVEL 1 (default, when full_atlas_scorer is present) ─────────╮
│   full_atlas_scorer.py                                          │
│     Batched cross-encoder scores ALL 256 anchors                │
│   coactivation.py (if trained model exists)                     │
│     Transformer re-ranks using inter-anchor co-occurrence       │
│   Adaptive threshold: selects 1–20 anchors                      │
╰─────────────────────────────────────────────────────────────────╯
            │
            ▼ (else falls back to)
╭─ LEVEL 2 (legacy) ─────────────────────────────────────────────╮
│   retriever.py                                                  │
│     BM25 + dense RRF over anchor cards, top-24 shortlist        │
│   selector.py                                                   │
│     Cosine or trained cross-encoder selects 1–6 anchors         │
╰─────────────────────────────────────────────────────────────────╯
            │
            ▼
memory_scorer.py (optional MLP re-ranker, research-only)
            │
            ▼
synthesizer.py + packer.py
  │  (LLM narrative synthesis + type-prioritized token packing)
  ▼
PackedContext (6000 tokens by default, failure-first ordering)
```

**Coactivation adaptive threshold.** When a trained co-activation model exists, the anchor count is not capped by a fixed `max_k`. Instead, the model outputs a per-query sigmoid threshold learned from oracle-labeled co-occurrence patterns. Queries that are narrow and specific activate few anchors; broad compositional queries activate many. The threshold is data-dependent, not a hyperparameter. `max_k` (default 6) only applies on the cosine-selector path (Level 2 without co-activation).

**Packer section atomicity.** The packer fills the token budget by processing memory types in priority order (`FAILURE → PROCEDURAL → TOOL_USE → EPISODIC → SEMANTIC → WORKING_DERIVATIVE`). Each type's memories are collected into a section. If a full section fits within the remaining budget it is included; if it doesn't fit entirely, individual memories within it are trimmed until they fit. This means the priority order is a guarantee about what gets in first, not a guarantee that lower-priority types are excluded — they appear if budget remains.

**Memory types, in packing priority:**

| Type | What it captures | Priority |
|------|-----------------|----------|
| FAILURE | What went wrong and why | Highest |
| PROCEDURAL | How-to steps, workflows | High |
| TOOL_USE | Tool/API behavior, flags, quirks | High |
| EPISODIC | Specific events, debugging sessions | Medium |
| SEMANTIC | Facts, concepts, domain knowledge | Medium |
| WORKING_DERIVATIVE | Scratch notes, intermediate reasoning | Low |

### Lifecycle

`psa lifecycle run` is the nightly orchestrator. Fast path runs every time; slow path only when health or new-memory volume triggers it.

```
Fast path:
  1. Mine new sessions (dedup against raw_sources)
  2. Clean up archived memories (hard-delete > 90 days old)
  3. Advertisement decay pass (ledger-based pattern pruning)
  4. Prune overloaded anchors to per-anchor budget
  5. Enforce global memory cap (default 50k)

Slow path (triggered by novelty > 8% or skew > 3x or forced):
  6. Rebuild atlas (preserves anchor identity)
  7. Run oracle labeling on recent queries (if LLM available)
  8. Retrain selector if gates met (≥300 labels, ≥200 held-out queries, R@24 ≥ 0.95)
```

### Memory forgetting

`psa/forgetting.py` scores each memory for disposability. **Time alone is not a disposal signal.** A memory that has gone a year without being packed is not more disposable than one packed yesterday — what matters is how it compares to its peers under pressure.

Per-anchor pruning uses:

```
forgetting_score =
  + low_usage_pressure_within_anchor    # bottom of anchor's usage distribution → 1.0; top → 0.0
  + min(overflow / budget, 1.0)          # crowding pressure — anchor over its 100-memory budget
  - min(log(1 + pack_count) / 3.0, 1.0)  # usage protection — 20 packs ≈ full protection
  - quality_score                        # quality protection — set once at extraction
```

Global-cap enforcement uses the same formula but substitutes a hybrid pressure: `0.7 * anchor_local + 0.3 * tenant_wide`, so scores remain comparable across anchors.

Range is roughly −2 to 2. Score > 0 is a candidate for pruning. Newly ingested memories get a 24-hour absolute grace period (score forced to −10). Low-scoring memories are archived (soft delete) and hard-deleted after 90 days in archive. The global cap is 50k memories; the per-anchor budget is 100.

Small-anchor stability rule: anchors with fewer than 5 active memories get `low_usage_pressure = 0`, so pruning there is driven entirely by crowding and quality (percentiles on tiny samples are too noisy to trust).

### Advertisement decay

An anchor card is not the same as an advertisement. The **card** is the anchor's full semantic identity. The **advertisements** are the `generated_query_patterns` field on that card — the specific questions the anchor claims to answer. Each pattern is independently tracked: when a pattern attracts queries that lead to that anchor being selected, it earns attention; when it stops, it decays. The card itself survives; only its stale pattern strings are pruned.

Anchor cards accumulate `generated_query_patterns` over time. Those patterns are the "ads" a card uses to attract queries. Stale or misleading ads should fade.

Operationally, advertisement decay belongs to the lifecycle path, not as an afterthought:

- every `psa search` can append per-pattern signals into the persistent `pattern_ledger` when `advertisement_decay.tracking_enabled=true`
- `psa lifecycle run` decays ledger state, evaluates stale patterns, and can remove them from `anchor_cards_refined.json` when `advertisement_decay.removal_enabled=true`
- `psa advertisement status` and `psa advertisement diff` are the inspection surface for that lifecycle state

The ledger uses exponential decay plus per-pattern attribution. A shadow-policy counterfactual also runs so you can inspect how a stricter policy would behave, but the primary operational model is still the lifecycle-integrated one above.

`tracking_enabled` and `trace_queries` are coupled: the ledger write only fires when trace succeeds (trace → ledger ordering). Setting `tracking_enabled=true` without `trace_queries=true` is a no-op — the config loader enforces this at load time.

There are also manual card-refinement commands elsewhere in the product, but for a normal installation you should think of advertisement decay as part of lifecycle in the same way memory forgetting is.

---

## Day-to-day usage

All commands below use `psa ...` for readability. If you are running directly from the repository without installing the CLI into your shell, prefix them with `uv run`, for example `uv run psa status`.

### Ingesting data

```bash
# Conversations (Claude Code, ChatGPT, Slack, Codex)
psa mine ~/.claude/projects/ --mode convos

# Project files (code, markdown, notes)
psa mine ~/projects/my_app

# Dry-run first to see what it will ingest
psa mine <dir> --dry-run --limit 50
```

The miner dedups against `raw_sources` so re-running on the same directory is safe and cheap.

### Searching and inspecting

```bash
# Full pipeline query with packed context
psa search "why did we switch to GraphQL"

# See everything the pipeline did — all 24 candidates, selector scores, timing
# Also writes to query_log.jsonl for psa log to read
psa inspect "why did we switch to GraphQL" --verbose

# Browse the psa inspect log (populated by psa inspect, not psa search)
psa log list
psa log show <run_id>
psa log diff <baseline_run_id> <newer_run_id>

# Diagnostic rollups over the query_trace.jsonl (populated by psa search)
psa diag activation       # anchor activation + carry rates
psa diag advertisement    # memory-count vs activation gap per anchor
psa diag misses           # below-threshold queries + near-miss anchors
```

### Lifecycle

```bash
# Run manually
psa lifecycle run

# Install nightly cron (macOS launchd plist)
psa lifecycle install

# Inspect state
psa lifecycle status
```

The nightly lifecycle is idempotent. First run is slow (model loads, possibly rebuild). Subsequent runs are fast unless health triggers slow path.

### Training

**Oracle labeling.** Before training, each query needs a ground-truth label: which subset of the 24 retrieved anchor candidates actually helped answer it? PSA generates these labels automatically using a two-stage LLM process (`psa label`):

1. **Cheap stage** — For each candidate anchor set, the LLM scores four terms in a single batched call (~1 API call per query vs ~23 individual calls): `SupportCoverage`, `ProceduralUtility`, `NoisePenalty`, `TokenCost`.
2. **Expensive stage** — For the top-3 surviving sets only, the LLM runs `TaskSuccess`: given this packed context, does an agent actually produce the right answer? Expensive because it requires a full generation, so only the strongest candidates reach it.

Final oracle score:
```
OracleScore = 0.45×SupportCoverage + 0.20×TaskSuccess
            + 0.15×ProceduralUtility − 0.10×NoisePenalty − 0.10×TokenCost
```

Labels are persisted to `~/.psa/tenants/{tenant_id}/training/oracle_labels.jsonl`.

**Selector training.** Once enough labels exist, the cross-encoder selector trains on a three-phase curriculum: warm start (random negatives) → hard negatives (anchors that score well but aren't correct) → adversarial rewrites (paraphrased queries). Co-activation training fits a small transformer over oracle-labeled anchor co-occurrence patterns.

```bash
# 1. Label all available unlabeled queries
psa label --n-queries 0

# 2. Train the selector (cross-encoder)
psa train --force

# 3. Also train the co-activation model (separate step, after selector training)
psa train --force --coactivation
```

Use `psa label --n-queries N` when you want a fixed-size batch instead of all available queries. Use `psa label --reset --n-queries 0` to discard existing labels and rebuild from scratch.

`psa train --force --coactivation` is not an extra step after `psa train --force`; it already trains the selector first and then trains co-activation.

Gates: selector trains once ≥300 oracle labels, ≥200 held-out queries, and R@24 ≥ 0.95 on the retriever. Use `--force` to override the gates during development.

### Benchmarks

LongMemEval is the primary benchmark harness. Runs against a dedicated `longmemeval_bench` tenant (never touches your `default` tenant).

```bash
psa benchmark longmemeval ingest                     # download + ingest dataset
psa benchmark longmemeval run --split val --limit 100  # run queries, generate answers
psa benchmark longmemeval score --results <path>     # score answers
psa benchmark longmemeval oracle-label --results <path> --mode fast
```

LongMemEval currently has a known regression path unrelated to advertisement decay; treat benchmark issues separately from lifecycle behavior.

### Advertisement maintenance

Advertisement decay is maintained through normal query traffic plus lifecycle. The commands you actually need day-to-day are:

```bash
psa advertisement status                     # ledger distribution + grace/risk counts
psa advertisement diff                       # compare primary vs shadow policy
psa advertisement rebuild-ledger --dry-run   # regenerate ledger from trace (dry-run)
psa advertisement purge --older-than-days 90 # hard-delete archived rows past retention
```

---

## Command reference

### Ingest and project setup

| Command | Purpose |
|---|---|
| `psa init <dir>` | Auto-detect people/projects from folder structure |
| `psa mine <dir>` | Mine project files into typed memories (default mode) |
| `psa mine <dir> --mode convos` | Mine conversation exports |
| `psa mine <dir> --dry-run --limit N` | Preview without writing |
| `psa split <dir>` | Split concatenated transcript mega-files before mining |
| `psa migrate` | Non-destructive ChromaDB → PSA MemoryStore migration |

### Search and inspection (read-only)

| Command | Purpose |
|---|---|
| `psa search "query"` | Full pipeline query with packed context |
| `psa search "query" --results N` | Top-N results |
| `psa inspect "query" --verbose` | Show all 24 candidates, selector scores, timing |
| `psa status` | Memory store overview |
| `psa log list` / `show <id>` / `diff <a> <b>` | Per-query trace log |
| `psa wake-up` | Show session wake-up context |

### Atlas

| Command | Purpose |
|---|---|
| `psa atlas build` | Build or rebuild atlas (requires ~200 memories) |
| `psa atlas status` | Version, anchor count, memory count |
| `psa atlas health` | Novelty rate, skew, rebuild recommendation |
| `psa atlas rebuild` | Force rebuild (preserves anchor identity) |
| `psa atlas refine --miss-log PATH` | Stage 1 ngram refinement from miss log |
| `psa atlas curate` | Stage 1 production-signal curation (oracle + fingerprints) |
| `psa atlas promote-refinement` | Promote the current candidate to live refined cards |

All atlas commands take `--tenant <name>` (default: `default`).

### Lifecycle

| Command | Purpose |
|---|---|
| `psa lifecycle run` | Run the nightly pipeline manually |
| `psa lifecycle run --label-batch N` | Cap labeling at N queries |
| `psa lifecycle status` | Last run time, memory count, selector mode |
| `psa lifecycle install` | Install macOS launchd plist |
| `psa lifecycle uninstall` | Remove the launchd plist |

### Labeling and training

| Command | Purpose |
|---|---|
| `psa label --n-queries N` | Oracle-label `N` queries for selector training |
| `psa label --n-queries 0` | Label all currently available unlabeled queries |
| `psa label --sessions-dir PATH --n-queries 0` | Label all available queries from a specific sessions directory |
| `psa label --reset --n-queries 0` | Delete all labels and rebuild them from scratch |
| `psa train` | Train cross-encoder selector when training gates are met |
| `psa train --force` | Train selector even if gates aren't met |
| `psa train --force --coactivation` | Train selector, then also train the co-activation transformer; this already includes selector training |
| `psa train --memory-scorer` | Train Level 2 MLP re-ranker (research-only) |

### Benchmarks

| Command | Purpose |
|---|---|
| `psa benchmark longmemeval ingest` | Download + ingest LongMemEval dataset |
| `psa benchmark longmemeval run --split val --limit N` | Run N queries, generate answers |
| `psa benchmark longmemeval run --selector coactivation` | Run with full activation system |
| `psa benchmark longmemeval score --results <path>` | Score generated answers |
| `psa benchmark longmemeval oracle-label --results <path>` | Generate oracle labels |

Uses the dedicated `longmemeval_bench` tenant.

### Advertisement decay

| Command | Purpose |
|---|---|
| `psa advertisement status` | Ledger distribution + grace/risk counts |
| `psa advertisement diff` | B-vs-A counterfactual report |
| `psa advertisement rebuild-ledger --dry-run` | Regenerate ledger from trace (dry-run) |
| `psa advertisement rebuild-ledger` | Actually materialize the regeneration |
| `psa advertisement purge --older-than-days 90` | Hard-delete archived rows past retention |

### Diagnostics

| Command | Purpose |
|---|---|
| `psa diag activation` | Anchor activation + carry rates |
| `psa diag advertisement` | Memory-count vs activation gap per anchor |
| `psa diag misses` | Below-threshold queries + near-miss anchors |

### Harness integration

| Command | Purpose |
|---|---|
| `psa hook run --hook session-start --harness claude-code` | Session hooks (Claude Code, Codex) |
| `psa instructions <name>` | Emit skill instructions to stdout |
| `psa repair` | Rebuild palace vector index after corruption |
| `psa legacy <subcmd>` | Legacy palace path (set `PSA_MODE=off` for full legacy mode) |

### MCP server (canonical interactive path)

The MCP server is the recommended way to use PSA interactively. It is a persistent process: the embedding model and atlas load once on first query (~10 s), then every subsequent query runs in ~200–500 ms. The CLI (`psa search`) pays the full load cost on every invocation and is better suited for scripting.

```bash
claude mcp add psa -- uv run --project /path/to/memnexus python -m psa.mcp_server
```

Exposed tools: `psa_atlas_search`, `psa_store_memory`, `psa_status`, `psa_atlas_status`, `psa_atlas_health`, `psa_list_anchors`, `psa_rebuild_atlas`, `psa_search`, `psa_check_duplicate`.

`psa_atlas_search` is the primary tool — it runs the full Level 1 pipeline (embed → anchor scoring → selector → packer) and returns packed context. Advertisement tracking fires inline if `tracking_enabled=true`. Atlas reloads triggered by `psa advertisement` removals are picked up automatically between queries without restarting the server.

---

## Operational guidance

Which commands are safe to run, which mutate live state, which are research-only, and which settings you still need to opt into explicitly.

### Safe (read-only, no state mutation)

- `psa search`, `psa inspect`, `psa status`
- `psa atlas status`, `psa atlas health`
- `psa advertisement status`, `psa advertisement diff`
- `psa advertisement rebuild-ledger --dry-run`
- `psa log list`, `psa log show`, `psa log diff`
- `psa diag {activation,advertisement,misses}`
- `psa lifecycle status`

Any of these can run against your live `default` tenant without side effects (beyond appending to the trace log when `trace_queries=true`).

### Mutates live state (runs fine, but changes things)

| Command | What it mutates |
|---|---|
| `psa mine`, `psa split` | Appends to `raw_sources` and `memory_objects` SQLite tables |
| `psa atlas build`, `psa atlas rebuild` | Writes new `atlas_vN/` directory; updates `lifecycle_state.json` |
| `psa atlas promote-refinement` | Overwrites `anchor_cards_refined.json` from the candidate |
| `psa lifecycle run` | Mines, prunes, may rebuild, may retrain |
| `psa label`, `psa train` | Writes `oracle_labels.jsonl` and model artifacts |
| `psa advertisement purge` | Hard-deletes archived ledger rows |
| `psa advertisement rebuild-ledger` (no `--dry-run`) | Overwrites ledger rows from trace |
| `psa migrate` | Writes memories from ChromaDB palace; source palace untouched |
| `psa repair` | Rebuilds palace vector index |

### Explicitly configured

| Setting | Default | Impact |
|---|---|---|
| `advertisement_decay.tracking_enabled` | `true` | Query traffic updates the advertisement ledger during `psa search` |
| `advertisement_decay.removal_enabled` | `true` | `psa lifecycle run` can prune stale advertisement patterns automatically |
| Cloud LLM | off (uses local Ollama) | Set `provider: "cloud"` in `~/.psa/llm.json` to enable |

### Research-only

These paths exist but are not recommended for day-to-day use:

- `psa train --memory-scorer` — Level 2 MLP re-ranker; help text notes "Research-only; produces a benchmark-derived model."
- `psa legacy *` — legacy ChromaDB palace path. Use `PSA_MODE=off` for full legacy mode.
- Co-activation selector at scale — works well in benchmarks but requires ≥300 oracle labels to train well.

---

## Configuration

### `~/.psa/config.json`

| Key | Default | Description |
|---|---|---|
| `psa_mode` | `"primary"` | `"primary"` / `"side-by-side"` / `"off"` |
| `tenant_id` | `"default"` | Active tenant (env: `PSA_TENANT_ID`) |
| `token_budget` | `6000` | Max tokens in packed context |
| `max_memories` | `50000` | Global memory cap |
| `anchor_memory_budget` | `100` | Per-anchor memory limit |
| `trace_queries` | `true` | Emit per-query trace to `query_trace.jsonl` |
| `nightly_hour` | `0` | Hour (0-23) used by the installed lifecycle job |
| `advertisement_decay.tracking_enabled` | `true` | Advertisement ledger writes during `psa search` (requires `trace_queries=true`) |
| `advertisement_decay.removal_enabled` | `true` | Advertisement removals during `psa lifecycle run` |
| `advertisement_decay.tau_days` | `45` | Exponential decay half-life |
| `advertisement_decay.grace_days` | `30` | Fresh-pattern grace period |
| `advertisement_decay.sustained_cycles` | `21` | Primary removal threshold |
| `advertisement_decay.min_patterns_floor` | `5` | Minimum number of patterns retained per anchor |
| `advertisement_decay.shadow.*` | (see spec) | Shadow-policy knobs |

See env var overrides: `PSA_MODE`, `PSA_TENANT_ID`, `PSA_AD_DECAY_TRACKING_ENABLED`, `PSA_AD_DECAY_REMOVAL_ENABLED`, `PSA_AD_DECAY_TAU_DAYS`, etc.

### `~/.psa/llm.json`

| Key | Description |
|---|---|
| `provider` | `"cloud"` (cloud-first, local fallback) or `"local"` (Ollama only) |
| `cloud_model` | Any litellm model string: `"gpt-4o-mini"`, `"azure/gpt-4o"`, ... |
| `cloud_api_key` | API key for cloud provider |
| `cloud_api_base`, `cloud_api_version` | For Azure / custom endpoints |
| `local_model` | Local model (default `"qwen2.5:7b"`) |
| `local_endpoint` | Ollama endpoint |
| `local_fallback` | If `true`, falls back to local when cloud fails |

---

## Project layout

```
psa/
  cli.py                    # Top-level entry point
  pipeline.py               # Query orchestration (Level 1 + Level 2 + trace/ledger hooks)
  full_atlas_scorer.py      # Level 1 — batched cross-encoder over all anchors
  coactivation.py           # Co-activation transformer + adaptive threshold
  retriever.py              # Level 2 — BM25 + dense RRF + RetrievalResult
  selector.py               # Level 2 — cosine / trained cross-encoder selection
  atlas.py                  # AtlasBuilder (k-means) + AtlasManager + AtlasHealth
  anchor.py                 # AnchorCard dataclass + pattern-card helpers
  memory_object.py          # MemoryObject + MemoryStore (SQLite WAL)
  memory_scorer.py          # Level 2 MLP re-ranker (research-only)
  consolidation.py          # Raw text → typed memory objects (LLM)
  forgetting.py             # Memory-level decay + pruning
  lifecycle.py              # Nightly orchestrator
  miner.py / convo_miner.py # Project + conversation ingestion
  normalize.py              # Multi-format conversation parsing
  embeddings.py             # BGE wrapper
  llm.py                    # Unified cloud+local caller
  trace.py                  # Per-query trace (JSONL)
  mcp_server.py             # MCP server for Claude Code
  config.py                 # MempalaceConfig
  health.py                 # AtlasHealth monitor
  advertisement/            # Advertisement decay + ledger + maintenance CLI
    metadata.py, reinforcement.py, decay.py, writer.py
    config.py, attribution.py, ledger.py, reload.py, cli.py
  training/
    oracle_labeler.py, data_generator.py, data_split.py
    train_selector.py, train_coactivation.py, train_memory_scorer.py
    coactivation_data.py, memory_scorer_data.py
  curation/                 # Production-signal curation (psa atlas curate)
    curator.py, extractor_heuristic.py, extractor_llm.py, ngrams.py, pool.py
  diag/                     # Diagnostic rollups (psa diag *)
    activation.py, advertisement.py, misses.py, trace_reader.py
  benchmarks/
    longmemeval.py, ablation_compare.py
  instructions/             # Skill instructions (psa instructions <name>)
```

---

## License

MIT
