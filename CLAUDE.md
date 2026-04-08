# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (dev) — uv manages the venv at .venv
uv sync                          # core + pytest + ruff + chromadb + faiss (dev group)
uv sync --extra training         # also install torch for selector training

# Test
uv run pytest tests/ -v
uv run pytest tests/test_memory_object.py -v          # single file
uv run pytest tests/ -k "test_atlas_build" -v         # single test

# Lint / format
uv run ruff check .
uv run ruff format .
uv run ruff format --check .
```

Python is pinned to 3.13 via `.python-version`. CI tests Python 3.9, 3.11, 3.13. Coverage threshold is 30%.

## Architecture

PSA has two primary pipelines that share the same data store (`MemoryStore` / SQLite):

### Ingest Pipeline

`psa mine <dir>` → `miner.py` or `convo_miner.py` → `consolidation.py` → `memory_object.py`

1. Raw source text is persisted immutably to `RawSource` table
2. `ConsolidationPipeline.consolidate()` chunks the text (fine 80–180 tok / mid 220–450 / section 500–1500) and calls Qwen2.5-7B-Instruct (`QWEN_ENDPOINT`, default `http://localhost:11434/v1/chat/completions`) to extract typed `MemoryObject` instances
3. Objects below `retention_score=0.65` are dropped; near-duplicates (cosine > 0.92) are merged
4. Embeddings (`BAAI/bge-base-en-v1.5`, 768-dim, L2-normalized) are stored in `memory_objects` table as binary blobs

`ConsolidationPipeline(use_llm=False)` disables the Qwen call (used in tests).

Both `miner.py` and `convo_miner.py` also maintain a parallel ChromaDB palace (legacy raw drawers) when chromadb is installed. chromadb is an **optional** dependency (`psa[palace]`), so imports happen inside functions, not at module level.

### Query Pipeline

`psa search "<query>"` → `cli.py:cmd_search()` → `pipeline.py:PSAPipeline`

```
embed query → AnchorRetriever (BM25 + dense, RRF k=60, top-24) →
AnchorSelector (cosine or trained cross-encoder, 1–4 anchors) →
fetch MemoryObjects for selected anchors →
EvidencePacker (role sections, 6000-token budget) → PackedContext
```

`PSAPipeline.from_config()` reads `MempalaceConfig` and constructs the pipeline. `from_tenant()` is the lower-level constructor. Both require an atlas to exist (built with `psa atlas build`); raises `FileNotFoundError` otherwise.

### Atlas

`AtlasBuilder.build_atlas()` in `atlas.py`:
- Minimum 500 memories with embeddings (raises `AtlasCorpusTooSmall` if not met)
- Spherical k-means, k=224 learned + 32 novelty, 3 seeds, stability threshold 15%
- Raises `AtlasUnstable` if seeds disagree
- Saved to `~/.psa/tenants/{tenant_id}/atlas_v{N}/`
- `AtlasManager.get_atlas()` loads the latest version; `get_or_build()` builds if none exists

### Data Model

**`MemoryStore`** (SQLite WAL mode at `~/.psa/tenants/{tenant_id}/memory.sqlite3`):
- `memory_objects` — typed objects with embedding blob and anchor assignment
- `memory_sources` — immutable raw source records
- Key methods: `add()`, `get()`, `query_by_type()`, `query_by_anchor()`, `search_by_embedding()`, `get_by_source_id()`

**`MemoryObject`** is created via `MemoryObject.create(...)` (factory classmethod), not `__init__`. Set `mo.embedding = vector` before `store.add(mo)`.

**Six `MemoryType` values:** `EPISODIC`, `SEMANTIC`, `PROCEDURAL`, `FAILURE`, `TOOL_USE`, `WORKING_DERIVATIVE`

### Config

`MempalaceConfig` loads from env vars > `~/.psa/config.json` > defaults. Key keys:

| Key | Default |
|-----|---------|
| `psa_mode` | `"primary"` (`"off"` \| `"side-by-side"` \| `"primary"`) |
| `tenant_id` | `"default"` |
| `token_budget` | `6000` |
| `embedding_model` | `"BAAI/bge-base-en-v1.5"` |
| `palace_path` | `~/.psa/palace` |

Env overrides: `PSA_MODE`, `PSA_TENANT_ID`.

### Selector Training

Training data lives at `~/.psa/tenants/{tenant_id}/training/`:
- `oracle_labels.jsonl` — produced by `OracleLabeler` (two-stage: Qwen cheap-stage + runtime model TaskSuccess)
- `training_data.jsonl` — produced by `DataGenerator` from oracle labels

Selector trains once gates are met (≥300 oracle labels, ≥200 held-out queries, recall@24 ≥ 0.95). Model saved to `~/.psa/models/selector_v{N}/`.

### Tests

`tests/conftest.py` redirects `HOME` to a temp dir before any psa import to isolate state. MCP server tests (test_mcp_server.py) rely on module-level globals (`_client_cache`, `_collection_cache`) being reset between tests via monkeypatching. `test_atlas.py` patches `MIN_MEMORIES_FOR_ATLAS` to a small value for unit-scale tests. Use `MagicMock(spec=ClassName)` for PSA objects to avoid real model loading.
