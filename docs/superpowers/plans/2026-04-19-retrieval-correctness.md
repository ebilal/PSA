# Retrieval Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four structural gaps that cause PSA to silently fail: MCP writes that are never retrievable, a quality-cap that hides relevant memories before scoring sees them, secondary anchor IDs that are stored but never queried, and a hard cold-start block with no fallback path.

**Architecture:** Each fix is surgical — two touch only `memory_object.py` + one call site, two touch only `mcp_server.py`, and one is a consolidation provenance fix. No new abstractions: the hot-assignment pattern already exists in `consolidation.py`; we replicate it in MCP. The cosine re-rank lives inside `query_by_anchor` behind an opt-in `query_vec` param so callers don't break. Secondary anchor fetch is a new method + two pipeline lines. Cold-start uses the already-present `EvidencePacker.pack_memories_direct` with a new `MemoryStore.search_by_embedding`.

**Tech Stack:** Python 3.13, SQLite WAL, numpy (already a dependency via embeddings), pytest, uv

---

## File Map

| File | Change |
|---|---|
| `psa/memory_object.py` | Add `query_vec`/`prefetch_limit` to `query_by_anchor`; add `query_by_secondary_anchor`; add `search_by_embedding` |
| `psa/pipeline.py` | Pass `query_vec` in fetch loop; add secondary anchor fetch with dedup |
| `psa/mcp_server.py` | Hot assignment after `store.add` in `tool_psa_store_memory`; add `_psa_embedding_fallback`; call fallback when no atlas |
| `psa/consolidation.py` | Pass `chunk_map` into `_raw_to_memory_object`; derive `EvidenceSpan` offsets from matched chunk |
| `tests/test_memory_object.py` | Tests for relevance-first fetch and secondary anchor query |
| `tests/test_mcp_server.py` | Tests for hot assignment, cold-start fallback |
| `tests/test_consolidation.py` | Test for precise evidence spans |

---

## Task 1: Hot anchor assignment in `tool_psa_store_memory`

**Files:**
- Modify: `psa/mcp_server.py:398-437`
- Test: `tests/test_mcp_server.py`

### Background

`consolidation.py:606-618` already does hot assignment after `store.add`. MCP's `tool_psa_store_memory` does not. Memories stored via MCP have `primary_anchor_id=NULL` and are therefore invisible to all pipeline queries.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mcp_server.py`:

```python
def test_store_memory_assigns_anchor(tmp_path, monkeypatch):
    """Memories stored via MCP should have a primary_anchor_id after store."""
    import numpy as np
    from unittest.mock import MagicMock, patch
    from psa import mcp_server
    from psa.memory_object import MemoryStore, MemoryObject

    # Fake atlas that assigns anchor 7
    fake_atlas = MagicMock()
    fake_atlas.assign_memory.return_value = (7, None, 0.9)

    # Fake embedding
    fake_vec = list(np.random.rand(768).astype(np.float32))

    store = MemoryStore(db_path=str(tmp_path / "mem.db"))

    monkeypatch.setattr(mcp_server, "_get_psa_store", lambda tid: (None, store))
    monkeypatch.setattr(mcp_server, "_get_psa_atlas", lambda tid: (fake_atlas, None))

    with patch("psa.embeddings.EmbeddingModel") as MockEM:
        MockEM.return_value.embed.return_value = fake_vec
        result = mcp_server.tool_psa_store_memory(
            title="Test title",
            body="Test body",
            tenant_id="test",
        )

    assert result.get("stored") is True
    mo = store.get(result["memory_object_id"])
    assert mo is not None
    assert mo.primary_anchor_id == 7
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd /Users/erhanbilal/Work/Projects/memnexus
uv run pytest tests/test_mcp_server.py::test_store_memory_assigns_anchor -v
```

Expected: `FAILED` — `AssertionError: assert None == 7`

- [ ] **Step 3: Implement hot assignment in `tool_psa_store_memory`**

In `psa/mcp_server.py`, replace the block after `store.add(mo)` (currently lines 430-437) with:

```python
    mo.embedding = embedding
    store.add(mo)

    # Hot assignment: assign to nearest anchor immediately (mirrors consolidation.py hot path)
    _anchor_assigned = False
    atlas, _ = _get_psa_atlas(tenant_id)
    if atlas is not None and mo.embedding is not None:
        try:
            primary_id, secondary_id, confidence = atlas.assign_memory(mo)
            if primary_id >= 0:
                store.update_anchor_assignment(
                    memory_object_id=mo.memory_object_id,
                    primary_anchor_id=primary_id,
                    secondary_anchor_ids=[secondary_id] if secondary_id is not None else [],
                    confidence=confidence,
                )
                _anchor_assigned = True
        except Exception as e:
            logger.debug("Hot assignment failed for MCP memory %s: %s", mo.memory_object_id, e)

    return {
        "memory_object_id": mo.memory_object_id,
        "memory_type": mtype.value,
        "title": title,
        "tenant_id": tenant_id,
        "stored": True,
        "anchor_assigned": _anchor_assigned,
    }
```

You'll need to add `from .memory_object import MemoryObject, MemoryType` and `from .embeddings import EmbeddingModel` at the top of the function (they're already there as local imports — no change needed). The `_get_psa_atlas` helper already exists at `mcp_server.py:371`.

- [ ] **Step 4: Run the test to confirm it passes**

```bash
uv run pytest tests/test_mcp_server.py::test_store_memory_assigns_anchor -v
```

Expected: `PASSED`

- [ ] **Step 5: Run full test suite for regressions**

```bash
uv run pytest tests/test_mcp_server.py -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add psa/mcp_server.py tests/test_mcp_server.py
git commit -m "fix: hot-assign anchor for MCP-stored memories

tool_psa_store_memory was persisting memories with primary_anchor_id=NULL,
making them permanently invisible to all pipeline queries. Now mirrors
the existing hot-assignment pattern from consolidation.py."
```

---

## Task 2: Relevance-first fetch in `query_by_anchor`

**Files:**
- Modify: `psa/memory_object.py:581-597`
- Modify: `psa/pipeline.py:544-549`
- Test: `tests/test_memory_object.py`

### Background

`query_by_anchor` orders by `quality_score DESC LIMIT 50`. In a large anchor the 50th-highest-quality memory is returned, but the query-relevant one may be at rank 200. Adding an optional `query_vec` param: fetch up to `prefetch_limit=200` sorted by quality, then cosine-sort in Python, return top `limit`. Embeddings are L2-normalized so dot-product = cosine similarity.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_memory_object.py`:

```python
def test_query_by_anchor_relevance_ordering(tmp_path):
    """With query_vec, memories are re-ordered by cosine sim, not quality_score."""
    import numpy as np
    from psa.memory_object import MemoryStore, MemoryObject, MemoryType

    store = MemoryStore(db_path=str(tmp_path / "mem.db"))
    dim = 4

    def _unit(v):
        a = np.array(v, dtype=np.float32)
        return list(a / np.linalg.norm(a))

    query_vec = _unit([1.0, 0.0, 0.0, 0.0])

    # high quality, low relevance
    lo_rel = MemoryObject.create(
        tenant_id="t", memory_type=MemoryType.SEMANTIC,
        title="low relevance", body="b", summary="s",
        source_ids=[], classification_reason="", quality_score=0.9,
    )
    lo_rel.embedding = _unit([0.0, 1.0, 0.0, 0.0])  # orthogonal to query

    # low quality, high relevance
    hi_rel = MemoryObject.create(
        tenant_id="t", memory_type=MemoryType.SEMANTIC,
        title="high relevance", body="b", summary="s",
        source_ids=[], classification_reason="", quality_score=0.1,
    )
    hi_rel.embedding = _unit([1.0, 0.0, 0.0, 0.0])  # aligned with query

    for mo in [lo_rel, hi_rel]:
        store.add(mo)
        store.update_anchor_assignment(mo.memory_object_id, primary_anchor_id=1)

    results = store.query_by_anchor("t", anchor_id=1, limit=2, query_vec=query_vec)
    assert results[0].title == "high relevance"
    assert results[1].title == "low relevance"
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_memory_object.py::test_query_by_anchor_relevance_ordering -v
```

Expected: `FAILED` — results come back quality-ordered, not relevance-ordered.

- [ ] **Step 3: Implement relevance-first fetch in `query_by_anchor`**

Replace the `query_by_anchor` method at `psa/memory_object.py:581-597` with:

```python
def query_by_anchor(
    self,
    tenant_id: str,
    anchor_id: int,
    limit: int = 50,
    query_vec: Optional[List[float]] = None,
    prefetch_limit: int = 200,
) -> List[MemoryObject]:
    fetch_n = prefetch_limit if query_vec is not None else limit
    with self._connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM memory_objects
            WHERE tenant_id = ? AND primary_anchor_id = ? AND is_duplicate = 0 AND is_archived = 0
            ORDER BY quality_score DESC, created_at DESC
            LIMIT ?
            """,
            (tenant_id, anchor_id, fetch_n),
        ).fetchall()
    memories = [self._row_to_memory_object(r) for r in rows]
    if query_vec is not None and memories:
        import numpy as np
        qv = np.array(query_vec, dtype=np.float32)
        memories.sort(
            key=lambda m: float(np.dot(qv, np.array(m.embedding, dtype=np.float32)))
            if m.embedding else 0.0,
            reverse=True,
        )
        return memories[:limit]
    return memories
```

Note: `Optional` and `List` are already imported at the top of `memory_object.py`.

- [ ] **Step 4: Update the pipeline fetch call to pass `query_vec`**

In `psa/pipeline.py:544-549`, change:

```python
                        anchor_memories = self.store.query_by_anchor(
                            tenant_id=self.tenant_id,
                            anchor_id=sa.anchor_id,
                            limit=50,
                        )
```

to:

```python
                        anchor_memories = self.store.query_by_anchor(
                            tenant_id=self.tenant_id,
                            anchor_id=sa.anchor_id,
                            limit=50,
                            query_vec=query_vec,
                        )
```

`query_vec` is in scope — it's set at `pipeline.py:286`.

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_memory_object.py::test_query_by_anchor_relevance_ordering tests/test_pipeline.py -v
```

Expected: new test PASSED, pipeline tests unchanged.

- [ ] **Step 6: Commit**

```bash
git add psa/memory_object.py psa/pipeline.py tests/test_memory_object.py
git commit -m "fix: re-rank query_by_anchor by cosine similarity when query_vec provided

quality_score DESC was the only sort key, capping the candidate pool to
top-50 by quality before relevance scoring ever saw them. With query_vec,
we prefetch 200 by quality then sort by dot-product (embeddings are
L2-normalized) before applying the limit."
```

---

## Task 3: Secondary anchor read path

**Files:**
- Modify: `psa/memory_object.py` (add method after `query_by_anchor`)
- Modify: `psa/pipeline.py:540-556`
- Test: `tests/test_memory_object.py`

### Background

`atlas.py:88` assigns a secondary anchor for cross-topic memories; that ID is persisted in `secondary_anchor_ids_json`. The pipeline fetch loop at `pipeline.py:544` only queries `primary_anchor_id`. A cross-topic memory assigned to anchors A (primary) and B (secondary) will never surface when anchor B is selected.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_memory_object.py`:

```python
def test_query_by_secondary_anchor(tmp_path):
    """Memories should be retrievable via their secondary anchor."""
    from psa.memory_object import MemoryStore, MemoryObject, MemoryType

    store = MemoryStore(db_path=str(tmp_path / "mem.db"))

    mo = MemoryObject.create(
        tenant_id="t", memory_type=MemoryType.SEMANTIC,
        title="cross-topic", body="b", summary="s",
        source_ids=[], classification_reason="", quality_score=0.8,
    )
    store.add(mo)
    # primary = 1, secondary = 2
    store.update_anchor_assignment(
        mo.memory_object_id,
        primary_anchor_id=1,
        secondary_anchor_ids=[2],
        confidence=0.85,
    )

    # Primary query returns it
    primary_results = store.query_by_anchor("t", anchor_id=1)
    assert any(m.memory_object_id == mo.memory_object_id for m in primary_results)

    # Secondary query also returns it
    secondary_results = store.query_by_secondary_anchor("t", anchor_id=2)
    assert any(m.memory_object_id == mo.memory_object_id for m in secondary_results)

    # Query for anchor 3 returns nothing
    assert store.query_by_secondary_anchor("t", anchor_id=3) == []
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_memory_object.py::test_query_by_secondary_anchor -v
```

Expected: `FAILED` — `AttributeError: 'MemoryStore' object has no attribute 'query_by_secondary_anchor'`

- [ ] **Step 3: Add `query_by_secondary_anchor` to `MemoryStore`**

Add this method immediately after `query_by_anchor` in `psa/memory_object.py` (after line 597). It mirrors `query_by_anchor` exactly — same `query_vec`/`prefetch_limit` re-ranking so secondary memories don't reintroduce the quality-cap problem:

```python
def query_by_secondary_anchor(
    self,
    tenant_id: str,
    anchor_id: int,
    limit: int = 50,
    query_vec: Optional[List[float]] = None,
    prefetch_limit: int = 200,
) -> List[MemoryObject]:
    """Return memories where anchor_id appears in secondary_anchor_ids_json.

    When query_vec is provided, prefetches up to prefetch_limit by quality then
    re-ranks by cosine similarity before applying limit (same as query_by_anchor).
    """
    fetch_n = prefetch_limit if query_vec is not None else limit
    with self._connect() as conn:
        rows = conn.execute(
            """
            SELECT mo.* FROM memory_objects mo, json_each(mo.secondary_anchor_ids_json) je
            WHERE mo.tenant_id = ? AND je.value = ?
              AND mo.is_duplicate = 0 AND mo.is_archived = 0
            ORDER BY mo.quality_score DESC
            LIMIT ?
            """,
            (tenant_id, anchor_id, fetch_n),
        ).fetchall()
    memories = [self._row_to_memory_object(r) for r in rows]
    if query_vec is not None and memories:
        import numpy as np
        qv = np.array(query_vec, dtype=np.float32)
        memories.sort(
            key=lambda m: float(np.dot(qv, np.array(m.embedding, dtype=np.float32)))
            if m.embedding else 0.0,
            reverse=True,
        )
        return memories[:limit]
    return memories
```

- [ ] **Step 4: Run the new unit test**

```bash
uv run pytest tests/test_memory_object.py::test_query_by_secondary_anchor -v
```

Expected: `PASSED`

- [ ] **Step 5: Update the pipeline fetch loop**

In `psa/pipeline.py:540-556`, add a secondary-anchor fetch pass after the primary one. The full block becomes:

```python
                    # Step 4: Fetch memories for selected anchors (inline for source tracking)
                    t0 = time.perf_counter()
                    seen_ids: set = set()
                    memories: List[MemoryObject] = []
                    for sa in selected:
                        anchor_memories = self.store.query_by_anchor(
                            tenant_id=self.tenant_id,
                            anchor_id=sa.anchor_id,
                            limit=50,
                            query_vec=query_vec,
                        )
                        for mo in anchor_memories:
                            if mo.memory_object_id in _memory_to_source_anchor:
                                continue
                            _memory_to_source_anchor[mo.memory_object_id] = sa.anchor_id
                            if mo.memory_object_id not in seen_ids:
                                seen_ids.add(mo.memory_object_id)
                                memories.append(mo)

                    # Also surface cross-topic memories via secondary anchor assignments
                    for sa in selected:
                        for mo in self.store.query_by_secondary_anchor(
                            tenant_id=self.tenant_id,
                            anchor_id=sa.anchor_id,
                            limit=25,
                            query_vec=query_vec,
                        ):
                            if mo.memory_object_id not in seen_ids:
                                seen_ids.add(mo.memory_object_id)
                                _memory_to_source_anchor[mo.memory_object_id] = sa.anchor_id
                                memories.append(mo)
```

- [ ] **Step 6: Run pipeline tests**

```bash
uv run pytest tests/test_pipeline.py tests/test_memory_object.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add psa/memory_object.py psa/pipeline.py tests/test_memory_object.py
git commit -m "fix: add secondary anchor read path

Secondary anchor IDs were stored by the atlas but never queried.
Cross-topic memories only had one retrieval surface. New
query_by_secondary_anchor method + pipeline secondary fetch pass
(limit=25) surfaces them when any selected anchor matches."
```

---

## Task 4: Cold-start embedding fallback

**Files:**
- Modify: `psa/memory_object.py` (add `search_by_embedding` after `get_all_with_embeddings`)
- Modify: `psa/mcp_server.py` (add `_psa_embedding_fallback`; update `tool_psa_atlas_search`)
- Test: `tests/test_mcp_server.py`

### Background

`tool_psa_atlas_search` returns `{"error": "No PSA atlas..."}` when no atlas exists. New tenants or small tenants have no retrieval path at all. `EvidencePacker.pack_memories_direct` already works without an atlas; we just need to feed it memories ranked by cosine similarity to the query.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mcp_server.py`:

```python
def test_atlas_search_cold_start_fallback(tmp_path, monkeypatch):
    """When no atlas exists, tool_psa_atlas_search returns embedding-based results, not an error."""
    import numpy as np
    from unittest.mock import patch
    from psa import mcp_server
    from psa.memory_object import MemoryStore, MemoryObject, MemoryType

    store = MemoryStore(db_path=str(tmp_path / "mem.db"))
    fake_vec = list(np.ones(768, dtype=np.float32) / np.sqrt(768))

    mo = MemoryObject.create(
        tenant_id="default", memory_type=MemoryType.SEMANTIC,
        title="cold fact", body="cold body", summary="cold summary",
        source_ids=[], classification_reason="", quality_score=0.7,
    )
    mo.embedding = fake_vec
    store.add(mo)

    # No atlas
    monkeypatch.setattr(mcp_server, "_get_psa_pipeline", lambda tid: None)
    monkeypatch.setattr(mcp_server, "_get_psa_store", lambda tid: (None, store))

    with patch("psa.embeddings.EmbeddingModel") as MockEM:
        MockEM.return_value.embed.return_value = fake_vec
        result = mcp_server.tool_psa_atlas_search("cold fact query", tenant_id="default")

    assert "error" not in result
    assert result.get("cold_start") is True
    assert "text" in result
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_mcp_server.py::test_atlas_search_cold_start_fallback -v
```

Expected: `FAILED` — result contains `"error"` key.

- [ ] **Step 3: Add `search_by_embedding` to `MemoryStore`**

Add after `get_all_with_embeddings` (after line 610) in `psa/memory_object.py`:

```python
def search_by_embedding(
    self,
    tenant_id: str,
    query_vec: List[float],
    limit: int = 20,
) -> List[MemoryObject]:
    """Return top-k memories by cosine similarity (dot product; embeddings are L2-normalized)."""
    import numpy as np

    with self._connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM memory_objects
            WHERE tenant_id = ? AND embedding_blob IS NOT NULL
              AND is_duplicate = 0 AND is_archived = 0
            """,
            (tenant_id,),
        ).fetchall()

    if not rows:
        return []

    qv = np.array(query_vec, dtype=np.float32)
    scored: List[tuple] = []
    for row in rows:
        mo = self._row_to_memory_object(row)
        if mo.embedding:
            score = float(np.dot(qv, np.array(mo.embedding, dtype=np.float32)))
            scored.append((score, mo))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [mo for _, mo in scored[:limit]]
```

- [ ] **Step 4: Add `_psa_embedding_fallback` to `mcp_server.py`**

Add this function before `tool_psa_atlas_search` (before line 386) in `psa/mcp_server.py`:

```python
def _psa_embedding_fallback(
    query: str,
    tenant_id: str,
    token_budget: int,
) -> dict:
    """Full-text embedding search when no atlas is available (cold-start path)."""
    from .embeddings import EmbeddingModel
    from .packer import EvidencePacker

    _, store = _get_psa_store(tenant_id)
    em = EmbeddingModel()
    query_vec = em.embed(query)
    memories = store.search_by_embedding(tenant_id, query_vec, limit=20)

    packer = EvidencePacker()
    packed = packer.pack_memories_direct(
        query=query,
        memories=memories,
        token_budget=token_budget,
        query_vec=query_vec,
        store=store,
    )
    result = packed.to_dict() if hasattr(packed, "to_dict") else {
        "text": packed.text,
        "memory_ids": packed.memory_ids,
        "token_count": packed.token_count,
    }
    result["cold_start"] = True
    return result
```

- [ ] **Step 5: Update `tool_psa_atlas_search` to call the fallback**

In `psa/mcp_server.py`, replace the error return in `tool_psa_atlas_search` (lines 388-392):

```python
def tool_psa_atlas_search(query: str, tenant_id: str = DEFAULT_TENANT_ID, token_budget: int = 6000):
    """Run the full PSA pipeline query (retriever + selector + packer)."""
    pipeline = _get_psa_pipeline(tenant_id)
    if pipeline is None:
        return _psa_embedding_fallback(query, tenant_id, token_budget)
    pipeline.token_budget = token_budget
    result = pipeline.query(query)
    return result.to_dict()
```

- [ ] **Step 6: Check `PackedContext.to_dict` exists**

```bash
cd /Users/erhanbilal/Work/Projects/memnexus
grep -n "def to_dict" psa/packer.py
```

If `to_dict` is not defined, add to `PackedContext` in `psa/packer.py`:

```python
def to_dict(self) -> dict:
    return {
        "text": self.text,
        "memory_ids": self.memory_ids,
        "token_count": self.token_count,
    }
```

- [ ] **Step 7: Run tests**

```bash
uv run pytest tests/test_mcp_server.py::test_atlas_search_cold_start_fallback tests/test_mcp_server.py -v
```

Expected: new test PASSED, existing tests unchanged.

- [ ] **Step 8: Commit**

```bash
git add psa/memory_object.py psa/mcp_server.py psa/packer.py tests/test_mcp_server.py
git commit -m "fix: embedding fallback for cold-start tenants with no atlas

tool_psa_atlas_search previously returned an error when no atlas existed.
New search_by_embedding (numpy cosine sim over all memories) feeds
EvidencePacker.pack_memories_direct for a real result even before the
atlas threshold is met."
```

---

## Task 5: Precise evidence spans from chunk offsets

**Files:**
- Modify: `psa/consolidation.py:361-415` (`_raw_to_memory_object`) and `psa/consolidation.py:578` (call site)
- Test: `tests/test_consolidation.py`

### Background

`_raw_to_memory_object` creates `EvidenceSpan(source_id=..., start_offset=0, end_offset=len(source.full_text))` — the full source. The `Chunk` dataclass already has `start_offset`/`end_offset` for each chunk. The LLM returns `evidence_chunk_ids` in its output. We should build the span from the union of those chunks' actual offsets.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_consolidation.py`:

```python
def test_evidence_spans_use_chunk_offsets():
    """EvidenceSpan offsets should match actual chunk positions, not full-source bounds."""
    from psa.consolidation import Chunk, _raw_to_memory_object
    from psa.memory_object import RawSource

    source = RawSource(
        source_id="src1",
        tenant_id="t",
        source_type="prose",
        source_path="test.txt",
        title="test source",
        full_text="AAAAAAAAAA BBBBBBBBBB CCCCCCCCCC",  # 31 chars
        created_at="2026-01-01T00:00:00+00:00",
    )

    chunk_map = {
        "src1_fine_0000": Chunk(
            chunk_id="src1_fine_0000",
            source_id="src1",
            level="fine",
            text="BBBBBBBBBB",
            start_offset=11,
            end_offset=21,
        )
    }

    raw = {
        "type": "semantic",
        "title": "B chunk",
        "body": "something about B",
        "summary": "B",
        "retention_score": 0.9,
        "evidence_chunk_ids": ["src1_fine_0000"],
    }

    mo = _raw_to_memory_object(raw, source, tenant_id="t", chunk_map=chunk_map)
    assert mo is not None
    assert len(mo.evidence_spans) == 1
    span = mo.evidence_spans[0]
    assert span.start_offset == 11
    assert span.end_offset == 21
    # Verify span actually points to the right text
    assert source.full_text[span.start_offset:span.end_offset] == "BBBBBBBBBB"
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_consolidation.py::test_evidence_spans_use_chunk_offsets -v
```

Expected: `FAILED` — `TypeError: _raw_to_memory_object() got an unexpected keyword argument 'chunk_map'`. The function signature doesn't accept `chunk_map` yet; that's exactly what Step 3 adds. This TypeError is the correct initial failure — it confirms the test is wired to the right call site.

- [ ] **Step 3: Update `_raw_to_memory_object` signature and span logic**

In `psa/consolidation.py`, change the function signature from:

```python
def _raw_to_memory_object(
    raw: dict,
    source: RawSource,
    tenant_id: str,
) -> Optional[MemoryObject]:
```

to:

```python
def _raw_to_memory_object(
    raw: dict,
    source: RawSource,
    tenant_id: str,
    chunk_map: Optional[Dict[str, "Chunk"]] = None,
) -> Optional[MemoryObject]:
```

Then replace the `evidence_spans` argument in the `MemoryObject.create(...)` call (currently at consolidation.py:407-413):

```python
        evidence_spans=_build_evidence_spans(source, evidence_chunk_ids, chunk_map),
```

Add this helper function above `_raw_to_memory_object`:

```python
def _build_evidence_spans(
    source: RawSource,
    evidence_chunk_ids: List[str],
    chunk_map: Optional[Dict[str, "Chunk"]],
) -> List[EvidenceSpan]:
    """Build EvidenceSpan list from chunk offsets when available, fall back to full source."""
    if chunk_map and evidence_chunk_ids:
        spans = []
        for cid in evidence_chunk_ids:
            chunk = chunk_map.get(cid)
            if chunk is not None:
                spans.append(
                    EvidenceSpan(
                        source_id=source.source_id,
                        start_offset=chunk.start_offset,
                        end_offset=chunk.end_offset,
                    )
                )
        if spans:
            return spans
    # Fallback: full source (legacy behaviour when chunk_map absent)
    return [
        EvidenceSpan(
            source_id=source.source_id,
            start_offset=0,
            end_offset=len(source.full_text),
        )
    ]
```

Make sure `Dict` is imported — it's already in `from typing import List, Optional, Tuple` at the top; add `Dict` there if missing.

- [ ] **Step 4: Update the call site in `consolidate()`**

In `psa/consolidation.py:542-578`, after the chunks are produced, build a chunk_map and pass it:

Find the line (around 578):
```python
            mo = _raw_to_memory_object(raw, source, self.tenant_id)
```

Change it to:
```python
            chunk_map = {c.chunk_id: c for c in chunks}
            mo = _raw_to_memory_object(raw, source, self.tenant_id, chunk_map=chunk_map)
```

Note: `chunks` is already in scope at this point (set at line 544).

- [ ] **Step 5: Run the new test**

```bash
uv run pytest tests/test_consolidation.py::test_evidence_spans_use_chunk_offsets -v
```

Expected: `PASSED`

- [ ] **Step 6: Run full consolidation test suite**

```bash
uv run pytest tests/test_consolidation.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add psa/consolidation.py tests/test_consolidation.py
git commit -m "fix: derive EvidenceSpan offsets from actual chunk positions

Evidence spans previously covered the full source text (start=0,
end=len), so 'source context' was always the first chunk of the
whole file. Now uses the LLM-identified evidence_chunk_ids to
pick exact byte offsets from the Chunk dataclass. Falls back to
full-source when chunk_map is unavailable."
```

---

## Self-Review

**Spec coverage:**
- P1 write-not-retrievable → Task 1 ✓
- P1 quality-cap before relevance → Task 2 ✓
- P1 secondary anchors dead → Task 3 ✓
- P2 cold-start → Task 4 ✓
- P2 imprecise provenance → Task 5 ✓
- Residual risk (LLM availability) → explicitly excluded per user: "LLM is online API first with local ollama fallback so it should always be available" ✓

**Placeholder scan:** No TBDs, no "add appropriate error handling" phrases, no missing code blocks.

**Type consistency:**
- `query_by_secondary_anchor` added to `MemoryStore` in Task 3; called from pipeline.py in Task 3 Step 5 — names match.
- `search_by_embedding` added to `MemoryStore` in Task 4; called from `_psa_embedding_fallback` in Task 4 Step 4 — names match.
- `_build_evidence_spans` added above `_raw_to_memory_object` in Task 5; called from within same function — names match.
- `chunk_map` param on `_raw_to_memory_object` defaults to `None` so all existing callers still work.
