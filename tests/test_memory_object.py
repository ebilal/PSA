"""
test_memory_object.py — Tests for MemoryObject, RawSource, and MemoryStore.

Covers:
- CRUD operations on memory objects and raw sources
- Evidence chunk / span preservation (provenance never lost)
- Raw source immutability (INSERT OR IGNORE)
- Embedding encode/decode round-trip
- Query helpers (by type, by anchor, by embedding availability)
- Deduplication flag behaviour
"""

import os

import pytest

from psa.memory_object import (
    EvidenceSpan,
    MemoryObject,
    MemoryStore,
    MemoryType,
    RawSource,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_dir):
    db_path = os.path.join(tmp_dir, "test_memory.sqlite3")
    return MemoryStore(db_path=db_path)


@pytest.fixture
def sample_source():
    return RawSource.create(
        tenant_id="test_tenant",
        source_type="project_file",
        full_text="def authenticate(token): ...",
        title="auth.py",
        source_path="/project/auth.py",
    )


@pytest.fixture
def sample_memory(sample_source):
    return MemoryObject.create(
        tenant_id="test_tenant",
        memory_type=MemoryType.PROCEDURAL,
        title="JWT Authentication Pattern",
        body="Tokens expire after 24 hours; refresh via HttpOnly cookie.",
        summary="JWT auth pattern with 24h expiry and cookie refresh.",
        source_ids=[sample_source.source_id],
        classification_reason="Describes a reusable authentication procedure.",
        evidence_chunk_ids=["chunk_001"],
        evidence_spans=[
            EvidenceSpan(source_id=sample_source.source_id, start_offset=0, end_offset=30)
        ],
    )


# ── RawSource tests ──────────────────────────────────────────────────────────


def test_raw_source_create_and_retrieve(store, sample_source):
    store.add_source(sample_source)
    retrieved = store.get_source(sample_source.source_id)
    assert retrieved is not None
    assert retrieved.source_id == sample_source.source_id
    assert retrieved.full_text == sample_source.full_text
    assert retrieved.title == "auth.py"
    assert retrieved.source_type == "project_file"


def test_raw_source_immutable(store, sample_source):
    """INSERT OR IGNORE — a second add with the same source_id is silently ignored."""
    store.add_source(sample_source)
    original_text = sample_source.full_text

    # Try to write a modified version with the same ID (should be ignored)
    modified = RawSource(
        source_id=sample_source.source_id,
        tenant_id=sample_source.tenant_id,
        source_type=sample_source.source_type,
        source_path=sample_source.source_path,
        title=sample_source.title,
        full_text="MODIFIED TEXT",
        created_at=sample_source.created_at,
    )
    store.add_source(modified)

    retrieved = store.get_source(sample_source.source_id)
    assert retrieved.full_text == original_text, "Raw source must not be modified after creation"


def test_raw_source_missing(store):
    assert store.get_source("nonexistent-id") is None


def test_raw_source_title_defaults_to_basename(tmp_dir):
    src = RawSource.create(
        tenant_id="t",
        source_type="project_file",
        full_text="content",
        source_path="/some/path/myfile.py",
    )
    assert src.title == "myfile.py"


def test_raw_source_title_untitled_when_no_path():
    src = RawSource.create(tenant_id="t", source_type="conversation", full_text="hello")
    assert src.title == "untitled"


# ── MemoryObject CRUD ────────────────────────────────────────────────────────


def test_add_and_get_memory(store, sample_source, sample_memory):
    store.add_source(sample_source)
    returned_id = store.add(sample_memory)
    assert returned_id == sample_memory.memory_object_id

    retrieved = store.get(returned_id)
    assert retrieved is not None
    assert retrieved.memory_object_id == sample_memory.memory_object_id
    assert retrieved.memory_type == MemoryType.PROCEDURAL
    assert retrieved.title == "JWT Authentication Pattern"
    assert retrieved.tenant_id == "test_tenant"


def test_evidence_chunk_ids_preserved(store, sample_source, sample_memory):
    store.add_source(sample_source)
    store.add(sample_memory)
    retrieved = store.get(sample_memory.memory_object_id)
    assert retrieved.evidence_chunk_ids == ["chunk_001"]


def test_evidence_spans_preserved(store, sample_source, sample_memory):
    store.add_source(sample_source)
    store.add(sample_memory)
    retrieved = store.get(sample_memory.memory_object_id)
    assert len(retrieved.evidence_spans) == 1
    span = retrieved.evidence_spans[0]
    assert span.source_id == sample_source.source_id
    assert span.start_offset == 0
    assert span.end_offset == 30


def test_source_ids_preserved(store, sample_source, sample_memory):
    store.add_source(sample_source)
    store.add(sample_memory)
    retrieved = store.get(sample_memory.memory_object_id)
    assert sample_source.source_id in retrieved.source_ids


def test_get_missing_memory(store):
    assert store.get("no-such-id") is None


def test_delete_memory(store, sample_memory):
    store.add(sample_memory)
    assert store.delete(sample_memory.memory_object_id) is True
    assert store.get(sample_memory.memory_object_id) is None


def test_delete_nonexistent_returns_false(store):
    assert store.delete("ghost") is False


def test_count(store):
    assert store.count("test_tenant") == 0
    mo = MemoryObject.create(
        tenant_id="test_tenant",
        memory_type=MemoryType.EPISODIC,
        title="T",
        body="B",
        summary="S",
        source_ids=[],
        classification_reason="x",
    )
    store.add(mo)
    assert store.count("test_tenant") == 1


def test_count_is_tenant_scoped(store):
    for tid in ("alpha", "alpha", "beta"):
        mo = MemoryObject.create(
            tenant_id=tid,
            memory_type=MemoryType.SEMANTIC,
            title="T",
            body="B",
            summary="S",
            source_ids=[],
            classification_reason="x",
        )
        store.add(mo)
    assert store.count("alpha") == 2
    assert store.count("beta") == 1


# ── batch_add ────────────────────────────────────────────────────────────────


def test_batch_add(store):
    memories = [
        MemoryObject.create(
            tenant_id="t",
            memory_type=MemoryType.FAILURE,
            title=f"Failure {i}",
            body="body",
            summary="summary",
            source_ids=[],
            classification_reason="x",
        )
        for i in range(5)
    ]
    ids = store.batch_add(memories)
    assert len(ids) == 5
    assert store.count("t") == 5


# ── Embedding round-trip ─────────────────────────────────────────────────────


def test_embedding_encode_decode_roundtrip(store):
    embedding = [0.1 * i for i in range(768)]
    mo = MemoryObject.create(
        tenant_id="t",
        memory_type=MemoryType.SEMANTIC,
        title="T",
        body="B",
        summary="S",
        source_ids=[],
        classification_reason="x",
        embedding=embedding,
    )
    store.add(mo)
    retrieved = store.get(mo.memory_object_id)
    assert retrieved.embedding is not None
    assert len(retrieved.embedding) == 768
    # float32 precision loss is expected; check to 5 decimal places
    for orig, stored in zip(embedding, retrieved.embedding):
        assert abs(orig - stored) < 1e-5


def test_memory_without_embedding(store):
    mo = MemoryObject.create(
        tenant_id="t",
        memory_type=MemoryType.SEMANTIC,
        title="T",
        body="B",
        summary="S",
        source_ids=[],
        classification_reason="x",
    )
    store.add(mo)
    retrieved = store.get(mo.memory_object_id)
    assert retrieved.embedding is None


# ── query_by_type ────────────────────────────────────────────────────────────


def test_query_by_type(store):
    for mtype in (MemoryType.PROCEDURAL, MemoryType.PROCEDURAL, MemoryType.EPISODIC):
        store.add(
            MemoryObject.create(
                tenant_id="t",
                memory_type=mtype,
                title="T",
                body="B",
                summary="S",
                source_ids=[],
                classification_reason="x",
            )
        )

    procedural = store.query_by_type("t", MemoryType.PROCEDURAL)
    episodic = store.query_by_type("t", MemoryType.EPISODIC)
    assert len(procedural) == 2
    assert len(episodic) == 1


def test_query_by_type_excludes_duplicates(store):
    mo = MemoryObject.create(
        tenant_id="t",
        memory_type=MemoryType.SEMANTIC,
        title="T",
        body="B",
        summary="S",
        source_ids=[],
        classification_reason="x",
        is_duplicate=True,
        duplicate_of="other-id",
    )
    store.add(mo)
    results = store.query_by_type("t", MemoryType.SEMANTIC, exclude_duplicates=True)
    assert len(results) == 0
    results_all = store.query_by_type("t", MemoryType.SEMANTIC, exclude_duplicates=False)
    assert len(results_all) == 1


# ── query_by_anchor ──────────────────────────────────────────────────────────


def test_query_by_anchor(store):
    for anchor_id in (1, 1, 2):
        mo = MemoryObject.create(
            tenant_id="t",
            memory_type=MemoryType.EPISODIC,
            title="T",
            body="B",
            summary="S",
            source_ids=[],
            classification_reason="x",
            primary_anchor_id=anchor_id,
        )
        store.add(mo)

    anchor1_results = store.query_by_anchor("t", anchor_id=1)
    anchor2_results = store.query_by_anchor("t", anchor_id=2)
    assert len(anchor1_results) == 2
    assert len(anchor2_results) == 1


# ── get_all_with_embeddings ──────────────────────────────────────────────────


def test_get_all_with_embeddings(store):
    embedding = [0.0] * 768
    with_emb = MemoryObject.create(
        tenant_id="t",
        memory_type=MemoryType.SEMANTIC,
        title="Has embedding",
        body="B",
        summary="S",
        source_ids=[],
        classification_reason="x",
        embedding=embedding,
    )
    without_emb = MemoryObject.create(
        tenant_id="t",
        memory_type=MemoryType.SEMANTIC,
        title="No embedding",
        body="B",
        summary="S",
        source_ids=[],
        classification_reason="x",
    )
    store.batch_add([with_emb, without_emb])

    results = store.get_all_with_embeddings("t")
    assert len(results) == 1
    assert results[0].title == "Has embedding"


# ── update_anchor_assignment ─────────────────────────────────────────────────


def test_update_anchor_assignment(store):
    mo = MemoryObject.create(
        tenant_id="t",
        memory_type=MemoryType.SEMANTIC,
        title="T",
        body="B",
        summary="S",
        source_ids=[],
        classification_reason="x",
    )
    store.add(mo)

    store.update_anchor_assignment(
        memory_object_id=mo.memory_object_id,
        primary_anchor_id=42,
        secondary_anchor_ids=[10, 20],
        confidence=0.87,
    )

    updated = store.get(mo.memory_object_id)
    assert updated.primary_anchor_id == 42
    assert updated.secondary_anchor_ids == [10, 20]
    assert abs(updated.assignment_confidence - 0.87) < 1e-6


# ── success_label round-trip ─────────────────────────────────────────────────


@pytest.mark.parametrize("label,expected", [(True, True), (False, False), (None, None)])
def test_success_label_roundtrip(store, label, expected):
    mo = MemoryObject.create(
        tenant_id="t",
        memory_type=MemoryType.FAILURE,
        title="T",
        body="B",
        summary="S",
        source_ids=[],
        classification_reason="x",
        success_label=label,
    )
    store.add(mo)
    retrieved = store.get(mo.memory_object_id)
    assert retrieved.success_label is expected


# ── get_by_source_session ────────────────────────────────────────────────────


def test_get_by_source_session_returns_memories(tmp_path):
    """get_by_source_session finds memories whose source_path contains the session_id."""
    store = MemoryStore(str(tmp_path / "mem.sqlite3"))
    tenant_id = "test_tenant"

    session_id = "session_abc_123"
    source = RawSource.create(
        tenant_id=tenant_id,
        source_type="conversation",
        full_text="some content",
        title="test session",
        source_path=f"/tmp/convos/{session_id}.jsonl",
    )
    store.add_source(source)

    mo = MemoryObject.create(
        tenant_id=tenant_id,
        memory_type=MemoryType.EPISODIC,
        title="A memory",
        body="body text",
        summary="summary",
        source_ids=[source.source_id],
        classification_reason="test",
    )
    mo.embedding = [0.0] * 768
    store.add(mo)

    # Add a memory from a different source (should NOT be returned)
    other_source = RawSource.create(
        tenant_id=tenant_id,
        source_type="conversation",
        full_text="other content",
        title="other",
        source_path="/tmp/convos/other_session.jsonl",
    )
    store.add_source(other_source)
    other_mo = MemoryObject.create(
        tenant_id=tenant_id,
        memory_type=MemoryType.SEMANTIC,
        title="Other memory",
        body="other body",
        summary="other summary",
        source_ids=[other_source.source_id],
        classification_reason="test",
    )
    other_mo.embedding = [0.0] * 768
    store.add(other_mo)

    results = store.get_by_source_session(session_id, tenant_id=tenant_id)
    assert len(results) == 1
    assert results[0].memory_object_id == mo.memory_object_id


def test_get_by_source_session_returns_empty_for_missing(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.sqlite3"))
    results = store.get_by_source_session("nonexistent_session", tenant_id="test")
    assert results == []


# ── Isolation between stores ─────────────────────────────────────────────────


def test_store_isolation(tmp_dir):
    db_a = os.path.join(tmp_dir, "a.sqlite3")
    db_b = os.path.join(tmp_dir, "b.sqlite3")
    store_a = MemoryStore(db_path=db_a)
    store_b = MemoryStore(db_path=db_b)

    mo = MemoryObject.create(
        tenant_id="t",
        memory_type=MemoryType.EPISODIC,
        title="T",
        body="B",
        summary="S",
        source_ids=[],
        classification_reason="x",
    )
    store_a.add(mo)

    assert store_a.count("t") == 1
    assert store_b.count("t") == 0


def test_query_by_anchor_relevance_ordering(tmp_dir):
    """With query_vec, memories are re-ordered by cosine sim, not quality_score."""
    import numpy as np
    from psa.memory_object import MemoryStore, MemoryObject, MemoryType

    store = MemoryStore(db_path=os.path.join(tmp_dir, "mem.db"))

    def _unit(v):
        a = np.array(v, dtype=np.float32)
        return list(a / np.linalg.norm(a))

    query_vec = _unit([1.0, 0.0, 0.0, 0.0])

    # high quality, low relevance
    lo_rel = MemoryObject.create(
        tenant_id="t",
        memory_type=MemoryType.SEMANTIC,
        title="low relevance",
        body="b",
        summary="s",
        source_ids=[],
        classification_reason="",
        quality_score=0.9,
    )
    lo_rel.embedding = _unit([0.0, 1.0, 0.0, 0.0])  # orthogonal to query

    # low quality, high relevance
    hi_rel = MemoryObject.create(
        tenant_id="t",
        memory_type=MemoryType.SEMANTIC,
        title="high relevance",
        body="b",
        summary="s",
        source_ids=[],
        classification_reason="",
        quality_score=0.1,
    )
    hi_rel.embedding = _unit([1.0, 0.0, 0.0, 0.0])  # aligned with query

    for mo in [lo_rel, hi_rel]:
        store.add(mo)
        store.update_anchor_assignment(mo.memory_object_id, primary_anchor_id=1)

    results = store.query_by_anchor("t", anchor_id=1, limit=2, query_vec=query_vec)
    assert results[0].title == "high relevance"
    assert results[1].title == "low relevance"
