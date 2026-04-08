<div align="center">

# PSA — Persistent Semantic Atlas

### Typed memory objects, anchor-based retrieval, and a trained selector for AI agents.

<br>

Every conversation you have with an AI — every decision, every debugging session, every architecture debate — disappears when the session ends. Other memory systems extract snippets and throw away the reasoning. PSA takes a different approach: **ingest everything, derive typed memory objects from it, and retrieve by learned anchor regions — not flat embedding search.**

<br>

[![][version-shield]][release-link]
[![][python-shield]][python-link]
[![][license-shield]][license-link]
[![][discord-shield]][discord-link]

<br>

[Quick Start](#quick-start) · [How It Works](#how-it-works) · [Setup](#setup) · [Commands](#commands) · [Selector Training](#selector-training) · [MCP Server](#mcp-server)

<br>

<table>
<tr>
<td align="center"><strong>256</strong><br><sub>Learned anchor regions<br>per tenant (224 + 32 novelty)</sub></td>
<td align="center"><strong>6</strong><br><sub>Typed memory classes<br>episodic · semantic · procedural · failure · tool-use · working</sub></td>
<td align="center"><strong>$0</strong><br><sub>No subscription. No cloud.<br>Runs entirely local.</sub></td>
</tr>
</table>

</div>

---

## How It Works

PSA has two pipelines that share a single SQLite-backed memory store.

### Ingest Pipeline

Raw source files → typed memory objects derived by a local model.

```
psa mine <dir>
    │
    ├── structure-aware chunking (fine / mid / section)
    ├── Qwen2.5-7B-Instruct (local via Ollama) → typed MemoryObjects
    ├── retention filter (score ≥ 0.65)
    ├── deduplication (cosine > 0.92 → merge)
    └── embed (BAAI/bge-base-en-v1.5) + persist to SQLite
```

Six memory types: **EPISODIC** (specific events), **SEMANTIC** (facts and concepts), **PROCEDURAL** (how-to), **FAILURE** (what went wrong), **TOOL_USE** (tool/API behavior), **WORKING_DERIVATIVE** (scratch notes).

Raw source records are **immutable** — the derived memory objects are indexes over them, not replacements.

### Query Pipeline

```
psa search "query"
    │
    ├── embed query (bge-base-en-v1.5)
    ├── AnchorRetriever — BM25 + dense, fused via RRF → top-24 anchor candidates (5–15ms)
    ├── AnchorSelector — cosine baseline or trained cross-encoder → 1–4 anchors (10–40ms)
    ├── fetch linked MemoryObjects for selected anchors
    └── EvidencePacker — role-organized sections, 6000-token budget
```

Packed context sections in priority order: `FAILURE WARNINGS → PROCEDURAL GUIDANCE → TOOL-USE NOTES → RELEVANT EPISODES → FACTS & CONCEPTS → RAW CONTEXT`.

### Atlas

The 256 anchor regions (224 learned + 32 novelty) are induced from your memory corpus via spherical k-means. Build one after mining enough memories:

```bash
psa atlas build           # requires ≥ 500 memories
psa atlas status          # version, anchor count, memory count
psa atlas health          # novelty rate, utilization skew, rebuild recommendation
```

---

## Setup

### Requirements

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) — Python package manager
- [Ollama](https://ollama.com/) running locally (for consolidation — `qwen2.5:7b`)

```bash
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Ollama and pull the model
ollama pull qwen2.5:7b
```

### Install

```bash
# From PyPI
uv tool install psa                         # base (no atlas, no training)
uv tool install "psa[atlas]"               # + scikit-learn + faiss-cpu (required for atlas + search)
uv tool install "psa[atlas,training]"      # full stack including torch
```

Or install from source:

```bash
git clone https://github.com/ebilal/PSA
cd PSA
uv sync                    # installs dev deps (pytest, ruff, chromadb, faiss)
uv sync --extra training   # also installs torch for selector training
```

### First-time setup

```bash
psa init ~/projects/myapp
```

This creates a default tenant at `~/.psa/tenants/default/` and mines the target directory for an initial atlas.

---

## Quick Start

```bash
# 1. Mine your project files
psa mine ~/projects/myapp

# 2. Mine conversation exports (Claude Code, ChatGPT, Slack)
psa mine ~/chats/ --mode convos

# 3. Build the atlas (once you have ≥ 500 memories)
psa atlas build

# 4. Search
psa search "why did we switch to GraphQL"

# 5. Check what's stored
psa status
psa wake-up                          # load atlas summary as session context
```

### With Claude Code (MCP)

```bash
claude mcp add psa -- uv run --project /path/to/PSA python -m psa.mcp_server
```

Claude then has 25 MCP tools available — PSA search, memory storage, atlas management, knowledge graph, and diary tools. Ask it anything:

> *"What did we decide about auth last month?"*

Claude calls `psa_atlas_search` automatically, gets packed context organized by memory type, and answers with the full reasoning behind the decision.

---

## Commands

### Mining

```bash
psa mine <dir>                          # project files (code, docs, notes)
psa mine <dir> --mode convos            # conversation exports
psa mine <dir> --mode convos --wing myapp  # tag a wing
psa mine <dir> --dry-run                # preview without storing
psa mine <dir> --limit 50              # stop after N files

psa split <dir>                         # split concatenated transcripts into per-session files
psa split <dir> --dry-run
psa split <dir> --min-sessions 3
```

### Search

```bash
psa search "query"                      # PSA pipeline (primary mode)
psa search "query" --wing myapp         # filter by wing
psa search "query" --room auth-migration
psa search "query" --results 10         # top-N results
```

### Atlas

```bash
psa atlas build                         # induce atlas from current memories
psa atlas build --tenant staging        # specific tenant
psa atlas status                        # version, counts, built_at
psa atlas health                        # novelty rate, utilization skew, rebuild trigger
psa atlas rebuild                       # force rebuild
```

### Benchmark

```bash
psa benchmark --query "query"           # compare PSA vs raw ChromaDB side-by-side
```

### Migration (from legacy palace)

```bash
psa migrate                             # migrate ChromaDB drawers → PSA MemoryObjects
psa migrate --collection my_collection --tenant myteam
```

### Session context

```bash
psa wake-up                             # atlas summary + top anchors for session loading
psa status                              # palace + memory store overview
```

### Legacy commands

```bash
psa legacy wake-up                      # raw ChromaDB wake-up (pre-PSA format)
psa legacy compress --wing myapp        # AAAK compression (removed in v4)
```

---

## Configuration

Config is loaded from env vars > `~/.psa/config.json` > defaults.

```json
{
  "psa_mode": "primary",
  "tenant_id": "default",
  "token_budget": 6000,
  "embedding_model": "BAAI/bge-base-en-v1.5",
  "palace_path": "~/.psa/palace",
  "atlas_size": 256,
  "selector_threshold": 0.3
}
```

| Key | Default | Env var |
|-----|---------|---------|
| `psa_mode` | `"primary"` | `PSA_MODE` |
| `tenant_id` | `"default"` | `PSA_TENANT_ID` |
| `token_budget` | `6000` | — |
| `embedding_model` | `"BAAI/bge-base-en-v1.5"` | — |

`psa_mode` controls the search path: `"off"` (raw ChromaDB only), `"side-by-side"` (both), `"primary"` (PSA pipeline only, default).

### Consolidation endpoint

PSA calls a local Qwen2.5-7B model for memory extraction. Configure the endpoint:

```bash
export QWEN_ENDPOINT="http://localhost:11434/v1/chat/completions"  # Ollama default
```

Any OpenAI-compatible API endpoint works here.

### Multi-tenant

```bash
PSA_TENANT_ID=staging psa mine ~/staging-logs
PSA_TENANT_ID=staging psa atlas build
PSA_TENANT_ID=staging psa search "query"
```

Each tenant gets an isolated memory store and atlas at `~/.psa/tenants/{tenant_id}/`.

---

## Selector Training

The anchor selector has two modes:

- **Cosine** (default) — sorts candidates by query-anchor similarity. No training required.
- **Trained** — a fine-tuned `cross-encoder/ms-marco-MiniLM-L-6-v2` that learns task utility, not just topical overlap. This is what makes PSA distinct from embedding-only retrieval.

Training requires `uv sync --extra training`. Apple Silicon (M1–M4) is supported via MPS — no discrete GPU needed.

**Memory requirements:** Qwen 7B q4 via Ollama (~4.7 GB) + bge-base-en-v1.5 (~420 MB) + cross-encoder (~90 MB) = ~5.2 GB peak. Each stage runs sequentially, so 16 GB RAM is sufficient; 20 GB is comfortable.

**Time estimates on M4 MacBook Air (no discrete GPU):**
| Step | Time |
|------|------|
| Oracle labeling — 300 queries | ~1–2 h |
| Generate 12,000 training examples | < 5 min |
| Selector training (3 phases, CPU/MPS) | ~2–4 h |
| **Total** | **~3–6 h** |

### Step 1 — Collect oracle labels

Oracle labeling scores every candidate anchor set for a query and picks the best one. All candidate sets for a query are batched into a **single Qwen call** (not one call per set), so labeling 300 queries takes ~1–2 hours instead of 16+.

By default, the labeler derives queries from your stored memory titles — a self-supervised approach. For better training signal, pass `--sessions-dir` to use the **real questions you typed** during Claude Code sessions instead:

```bash
# Recommended: use real queries from your Claude Code sessions
uv run python -m psa.training.oracle_labeler \
  --tenant default \
  --n-queries 300 \
  --sessions-dir ~/.claude/projects \
  --output ~/.psa/tenants/default/training/oracle_labels.jsonl

# Fallback: self-supervised from memory titles (no --sessions-dir needed)
uv run python -m psa.training.oracle_labeler \
  --tenant default \
  --n-queries 300 \
  --output ~/.psa/tenants/default/training/oracle_labels.jsonl
```

Labels are stored as JSONL with fields: `query_id`, `oracle_anchor_ids`, `candidate_sets`, `proxy_scores`, `task_success_scores`.

**Training gates** — the selector will not train until:
- ≥ 300 oracle-labeled queries
- ≥ 200 held-out real queries (never seen during training)
- Shortlist recall@24 ≥ 0.95 (the retriever must work before the selector can help)
- ≥ 50 examples per query family (single-anchor, contrastive, compositional, bridge, experience)

The system logs gate status when you run `psa atlas status`.

### Step 2 — Generate training data

```bash
uv run python -m psa.training.data_generator \
  --tenant default \
  --labels ~/.psa/tenants/default/training/oracle_labels.jsonl \
  --output ~/.psa/tenants/default/training/training_data.jsonl \
  --n-samples 1000
```

This produces `(query, anchor_card, label)` triples with a 60/20/20 mix:
- 60% synthetic internal queries (from oracle labels)
- 20% hard-negative augmented (shortlisted-but-not-oracle anchors)
- 20% adversarial rewrites (lexical compression, alias substitution, distractor insertion)

### Step 3 — Train

```bash
uv run python -m psa.training.train_selector \
  --tenant default \
  --training-data ~/.psa/tenants/default/training/training_data.jsonl \
  --output-dir ~/.psa/models/selector_v1
```

Hyperparameters (defaults): `lr=2e-5`, `batch=32`, `epochs=3`, `max_seq=320`. Training runs three phases: supervised warm start → hard-negative curriculum → adversarial hardening. Stops when dev-set task success plateaus for 2 evaluations.

The trained model is saved with provenance metadata: atlas version, embedding model ID, teacher model ID.

### Step 4 — Activate

```bash
# In ~/.psa/config.json
{
  "selector_mode": "trained",
  "selector_model_path": "~/.psa/models/selector_v1"
}
```

Or per-query:

```python
from psa.pipeline import PSAPipeline

pipeline = PSAPipeline.from_tenant(
    selector_mode="trained",
    selector_model_path="~/.psa/models/selector_v1",
)
result = pipeline.query("what auth approach did we use?")
```

---

## Auto-Save Hooks (Claude Code)

Two hooks that automatically save memories during work sessions:

```json
{
  "hooks": {
    "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "/path/to/hooks/mempal_save_hook.sh"}]}],
    "PreCompact": [{"matcher": "", "hooks": [{"type": "command", "command": "/path/to/hooks/mempal_precompact_hook.sh"}]}]
  }
}
```

- **Save hook** — fires every 15 messages, runs `psa mine` on `MEMPAL_DIR` (if set)
- **PreCompact hook** — emergency save before context compression

Configure in `.claude/settings.local.json`.

---

## MCP Server

```bash
claude mcp add psa -- uv run --project /path/to/PSA python -m psa.mcp_server
```

### Tools

**Palace (read)**
- `psa_status` — overview + memory protocol
- `psa_list_wings`, `psa_list_rooms`, `psa_get_taxonomy` — navigation
- `psa_search` — semantic search with wing/room filters
- `psa_check_duplicate` — dedup check before filing

**Palace (write)**
- `psa_add_drawer` — file verbatim content to ChromaDB palace
- `psa_delete_drawer` — remove by ID

**PSA Atlas** (requires `psa_mode != "off"`)
- `psa_atlas_search` — full pipeline query → packed context
- `psa_store_memory` — add a typed memory object directly
- `psa_atlas_status` — version, anchor count, memory count
- `psa_list_anchors` — anchor cards + utilization
- `psa_atlas_health` — novelty rate, skew, rebuild recommendation
- `psa_rebuild_atlas` — trigger rebuild



<!-- Link Definitions -->
[version-shield]: https://img.shields.io/badge/version-4.0.0-4dc9f6?style=flat-square&labelColor=0a0e14
[release-link]: https://github.com/ebilal/PSA/releases
[python-shield]: https://img.shields.io/badge/python-3.9+-7dd8f8?style=flat-square&labelColor=0a0e14&logo=python&logoColor=7dd8f8
[python-link]: https://www.python.org/
[license-shield]: https://img.shields.io/badge/license-MIT-b0e8ff?style=flat-square&labelColor=0a0e14
[license-link]: https://github.com/ebilal/PSA/blob/main/LICENSE
[discord-shield]: https://img.shields.io/badge/discord-join-5865F2?style=flat-square&labelColor=0a0e14&logo=discord&logoColor=5865F2
[discord-link]: https://discord.com/invite/ycTQQCu6kn
