# PSA — Persistent Semantic Atlas

A persistent memory system for AI coding agents. PSA ingests your conversation history and project files, extracts typed memory objects, clusters them into a learned semantic atlas, and retrieves them via a two-level learned activation model — not a flat embedding search.

Every conversation you have with an AI disappears when the session ends. PSA keeps that context around. It runs entirely on your laptop: local SQLite, local embeddings, optional local LLM. No data leaves your machine unless you configure a cloud LLM for extraction.

Works with Claude Code, Codex, and any client that speaks MCP or provides session hooks.

---

## Core design philosophy

**Memory is not data retrieval.** You don't look up what you know — you just know it. PSA is built around this. Instead of a flat vector search, PSA learns a two-level activation model that identifies which regions of memory space are relevant to a query, then scores individual memories within those regions.

**Typed, structured memories.** The unit of storage is a `MemoryObject` with an explicit type (failure, procedural, tool-use, episodic, semantic, working-derivative), not a raw text chunk. Types drive packing priority: failures get surfaced first because they prevent repeating mistakes.

**The atlas is learned.** A one-time spherical k-means clustering identifies 256 semantic regions (up to 224 learned + 32 novelty). Each anchor gets an LLM-generated card describing what it holds and what queries it answers. The cards are what the retriever, selector, and advertisement-decay ledger all operate on.

**Forgetting is a feature.** Memories that stop being useful get pruned. Advertisement patterns on anchor cards that stop earning attention get marked stale. The system self-curates so it doesn't grow unboundedly.

**Laptop-first.** Everything runs locally. 525 memories + 125 anchors → query latency ~300–500ms on Apple Silicon with the default local embedding model.

---

## How the system works

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

### Atlas and anchors

Once you have ~500+ memories, `psa atlas build` runs spherical k-means (k=224 learned + 32 novelty, 3 seeds, stability-checked). The result is an `Atlas` with up to 256 `AnchorCard` objects. Each anchor holds:

- A centroid (768-dim embedding, cluster center)
- A card (`name`, `meaning`, `include_terms`, `exclude_terms`, `generated_query_patterns`, `prototype_examples`, `near_but_different`)
- Type distribution (what memory types populate this anchor)
- Quality + count metadata

Anchor identity is durable across rebuilds. When `AtlasBuilder.rebuild` runs, new clusters are matched to existing anchors by centroid similarity. Only genuinely novel clusters get new IDs. Trained models stay valid across rebuilds.

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
│     Cosine or trained cross-encoder selects 1–4 anchors         │
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
  3. Advertisement decay pass (stage 2, if tracking_enabled)
  4. Prune overloaded anchors to per-anchor budget
  5. Enforce global memory cap (default 50k)

Slow path (triggered by novelty > 8% or skew > 3x or forced):
  6. Rebuild atlas (preserves anchor identity)
  7. Run oracle labeling on recent queries (if LLM available)
  8. Retrain selector if gates met (≥300 labels, ≥200 held-out queries, R@24 ≥ 0.95)
```

### Memory forgetting

`psa/forgetting.py` scores memories for disposability:
- **Recency** — days since last used in a packed context
- **Anchor crowding** — too many memories under one anchor
- **Usage** — frequently packed memories are protected
- **Quality** — high `retention_score` is protected

Low-scoring memories are archived (soft delete) and hard-deleted after 90 days. Newly mined memories get a 24-hour grace period.

### Advertisement decay

Anchor cards accumulate `generated_query_patterns` over time. Those patterns are the "ads" a card uses to attract queries. Stale or misleading ads should fade.

**Stage 1 (shipped, operator-driven).** `psa atlas decay` runs an R1 substring reinforcement check against the trace log, produces a candidate file, operator reviews, then promotes via `psa atlas promote-refinement`. Protections: P1 low-activation shield, P3 pinned-pattern exemption. Nothing mutates live state until promotion.

**Stage 2 (shipped, default-off).** A persistent `pattern_ledger` SQLite table with exponential decay, inline per-query attribution via BM25 argmax, and a shadow-policy counterfactual. When `advertisement_decay.tracking_enabled=true`, every `psa search` writes ledger signals trace-first (trace → ledger; ledger skipped if trace fails). When `removal_enabled=true`, `psa lifecycle run` mutates `anchor_cards_refined.json` directly — but **never** without explicit operator sign-off after calibration.

Stage 1 and stage 2 coexist via a refined-hash gate: candidate meta records the refined file's SHA-256 at generation time, and `psa atlas promote-refinement` refuses stale candidates whose recorded hash no longer matches.

Current rollout: tracking is **on** in the default tenant's config, removal is **off**. See [Current rollout status](#current-rollout-status).

---

## Install

### Prerequisites

- Python 3.13 (repository is pinned via `.python-version`; CI also tests 3.9 and 3.11)
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) running locally with `qwen2.5:7b` (unless you're using a cloud LLM)

### Setup

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install Ollama and pull the default local model
brew install ollama
ollama pull qwen2.5:7b

# 3. Clone and install PSA
git clone https://github.com/ebilal/memnexus.git
cd memnexus
uv sync                       # core + dev dependencies
uv sync --extra training      # also installs PyTorch (needed for training)
```

### LLM configuration

PSA uses an LLM for memory extraction (consolidation) and oracle labeling. By default, local Ollama. To use a cloud API (faster, higher quality), edit `~/.psa/llm.json`:

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

Set `"provider": "local"` to disable cloud entirely. Environment overrides: `PSA_LLM_MODEL`, `PSA_LLM_API_KEY`, `PSA_LLM_API_BASE`, `QWEN_ENDPOINT`.

### First atlas build

Minimum viable sequence on a fresh install:

```bash
# 1. Mine some source — conversations or project files.
psa mine ~/.claude/projects/ --mode convos

# 2. Check what you have.
psa status

# 3. Build the atlas (needs ~500 memories; raises AtlasCorpusTooSmall otherwise).
psa atlas build

# 4. Verify the atlas is healthy.
psa atlas status
psa atlas health

# 5. First real query.
psa search "how does the atlas work"
```

`psa atlas build` runs k-means and calls the LLM once per cluster to generate cards. Expect a few minutes depending on memory count and LLM latency.

---

## Day-to-day usage

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
psa inspect "why did we switch to GraphQL" --verbose

# Browse the per-query trace log
psa log list
psa log show <run_id>
psa log diff <baseline_run_id> <newer_run_id>

# Diagnostic rollups over the trace log
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

Selector training is a three-phase curriculum (warm start → hard negatives → adversarial rewrites). Co-activation training fits a small transformer over oracle-labeled anchor co-occurrence patterns.

```bash
# 1. Label queries — LLM judges which anchor sets help answer each query
psa label --n-queries 300

# 2. Train the selector (cross-encoder)
psa train --force

# 3. Also train the co-activation model (once selector is trained)
psa train --force --coactivation
```

Gates: selector trains once ≥300 oracle labels, ≥200 held-out queries, and R@24 ≥ 0.95 on the retriever. Use `--force` to override the gates during development.

### Benchmarks

LongMemEval is the primary benchmark harness. Runs against a dedicated `longmemeval_bench` tenant (never touches your `default` tenant).

```bash
psa benchmark longmemeval ingest                     # download + ingest dataset
psa benchmark longmemeval run --split val --limit 100  # run queries, generate answers
psa benchmark longmemeval score --results <path>     # score answers
psa benchmark longmemeval oracle-label --results <path> --mode fast
```

See [Current rollout status](#current-rollout-status) for the current LongMemEval issue.

### Advertisement calibration flow

Stage 2 advertisement decay is opt-in. To run in tracking-only mode (safe, no live state mutation):

```bash
# 1. Enable tracking in ~/.psa/config.json (see Configuration section)
#    "advertisement_decay": { "tracking_enabled": true, "removal_enabled": false }
#    Requires "trace_queries": true (the default).

# 2. Run normal queries for 1–2 weeks. Every psa search writes ledger signals inline.

# 3. Periodically inspect
psa advertisement status    # ledger distribution, grace, at-risk, shadow-only-at-risk
psa advertisement diff      # B-vs-A counterfactual (primary vs shadow policy)

# 4. Reassess after grace expires (default 21 days) and shadow data accumulates.
#    Reassessment inputs: shadow agreement rate, count of removal-eligible patterns,
#    5–10 manually inspected would-be removals, recommendation.

# 5. Only then — and only with explicit sign-off — flip removal_enabled=true.
#    Nightly lifecycle will then start removing patterns from anchor_cards_refined.json.

# Maintenance
psa advertisement rebuild-ledger --dry-run  # regenerate ledger from trace (dry-run)
psa advertisement purge --older-than-days 90  # hard-delete archived rows past retention
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
| `psa atlas build` | Build or rebuild atlas (requires ~500 memories) |
| `psa atlas status` | Version, anchor count, memory count |
| `psa atlas health` | Novelty rate, skew, rebuild recommendation |
| `psa atlas rebuild` | Force rebuild (preserves anchor identity) |
| `psa atlas refine --miss-log PATH` | Stage 1 ngram refinement from miss log |
| `psa atlas curate` | Stage 1 production-signal curation (oracle + fingerprints) |
| `psa atlas decay` | Stage 1 advertisement forgetting dry-run/candidate |
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
| `psa label --n-queries N` | Oracle-label N queries for selector training |
| `psa label --reset` | Delete all labels and start fresh |
| `psa train` | Train cross-encoder selector |
| `psa train --force` | Train even if gates aren't met |
| `psa train --force --coactivation` | Also train co-activation transformer |
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

### Advertisement decay (stage 2)

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

### MCP server (Claude Code integration)

```bash
claude mcp add psa -- uv run --project /path/to/memnexus python -m psa.mcp_server
```

Exposed tools: `psa_atlas_search`, `psa_store_memory`, `psa_status`, `psa_atlas_status`, `psa_atlas_health`, `psa_list_anchors`, `psa_rebuild_atlas`, `psa_search`, `psa_check_duplicate`.

---

## Operational guidance

Which commands are safe to run, which mutate live state, which are research-only, and what's default-off.

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

### Default-off / opt-in

| Setting | Default | Impact |
|---|---|---|
| `advertisement_decay.tracking_enabled` | `false` | When off, `psa search` skips all ledger writes — zero overhead |
| `advertisement_decay.removal_enabled` | `false` | When off, `psa lifecycle run` decays/evaluates but never mutates `anchor_cards_refined.json` |
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
| `advertisement_decay.tracking_enabled` | `false` | Stage 2 ledger writes (requires `trace_queries=true`) |
| `advertisement_decay.removal_enabled` | `false` | Stage 2 live removals in lifecycle |
| `advertisement_decay.tau_days` | `45` | Exponential decay half-life |
| `advertisement_decay.grace_days` | `21` | Fresh-pattern grace period |
| `advertisement_decay.sustained_cycles` | `14` | Primary removal threshold |
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

## Current rollout status

As of 2026-04-18:

- **Stage 2 advertisement-forgetting is merged on `main`** (merge `5776cfb`, Level 1 integration fix `c82f2a8`).
- **Tracking is ON** in the `default` tenant's `~/.psa/config.json`. Every `psa search` writes ledger signals inline.
- **Removal is OFF** and stays off until calibration completes. Re-enabling requires explicit sign-off plus evidence: shadow agreement rate, count of removal-eligible patterns, and 5–10 manually inspected would-be removals.
- **LongMemEval regression path is blocked** by a pre-existing model-dimensionality mismatch in the selector artifact (`linear(): input and weight.T shapes cannot be multiplied (65x11 and 16x32)`). Same error fires with stage 2 disabled, so it's unrelated to stage 2. Treated as separate follow-up work; not the gate for stage 2 rollout.

**Tripwires to watch** during the calibration window:
- `psa log` showing `Advertisement decay pass failed` or `AdvertisementDecayConfigError`
- `psa advertisement status` showing `n_active=0` after real query activity (means the inline wiring broke)
- Unexpected shadow disagreement patterns in `psa advertisement diff`

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
  advertisement/            # Stage 1 + Stage 2 advertisement forgetting
    metadata.py, reinforcement.py, decay.py, writer.py   # Stage 1
    config.py, attribution.py, ledger.py, reload.py, cli.py  # Stage 2
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
