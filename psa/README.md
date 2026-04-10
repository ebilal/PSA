# psa/ — Core Package

The Python package that powers PSA.

## Modules

| Module | What it does |
|--------|-------------|
| `cli.py` | CLI entry point — routes mine, search, inspect, atlas, train, benchmark |
| `config.py` | Configuration — env vars, `~/.psa/config.json`, defaults |
| `memory_object.py` | `MemoryObject` dataclass and `MemoryStore` SQLite backend |
| `pipeline.py` | `PSAPipeline` — retriever → selector → packer |
| `retriever.py` | `AnchorRetriever` — BM25 + dense hybrid, RRF fusion |
| `selector.py` | `AnchorSelector` — cosine baseline or trained cross-encoder |
| `packer.py` | `EvidencePacker` — role-organized sections under token budget |
| `atlas.py` | `AtlasBuilder` + `AtlasManager` — spherical k-means clustering |
| `anchor.py` | `AnchorCard` + `AnchorIndex` — FAISS index over anchor centroids |
| `embeddings.py` | `EmbeddingModel` — BAAI/bge-base-en-v1.5, 768-dim, L2-normalized |
| `inspect.py` | `inspect_query()` — full pipeline trace + query log for observability |
| `mcp_server.py` | MCP server for Claude and other MCP-compatible agents |
| `miner.py` | Project file ingest — scans dirs, chunks, stores to SQLite |
| `convo_miner.py` | Conversation ingest — chunks by exchange pair, stores to SQLite |
| `consolidation.py` | LLM-driven memory extraction — chunks, calls LLM, filters, deduplicates |
| `normalize.py` | Converts chat formats (Claude Code, Claude.ai, ChatGPT, Slack, plain text) |
| `searcher.py` | ChromaDB semantic search (legacy palace path) |
| `tenant.py` | Tenant directory management |
| `llm.py` | Unified LLM caller — local (Ollama) + cloud (LiteLLM) with fallback |
| `entity_registry.py` | Entity code registry |
| `entity_detector.py` | Auto-detects people and projects from file content |
| `health.py` | Atlas health checks |
| `lifecycle.py` | Nightly lifecycle pipeline — label, train, rebuild |
| `forgetting.py` | Retention scoring and memory expiry |
| `spellcheck.py` | Name-aware spellcheck |
| `split_mega_files.py` | Splits concatenated transcript files |
| `migrate.py` | Migrates ChromaDB palace to PSA MemoryStore |
| `hooks_cli.py` | Claude Code hook handlers |
| `version.py` | Package version |

## Sub-packages

| Sub-package | What it does |
|-------------|-------------|
| `training/` | Selector training: oracle labeler, data generator, train script |
| `benchmarks/` | Benchmark harnesses: LongMemEval ingest / run / score |
