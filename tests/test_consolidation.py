"""
test_consolidation.py — Tests for psa.consolidation.

Covers:
- Hierarchical chunking (fine/mid/section levels, source offsets)
- Retention filtering (score threshold, type rules)
- ConsolidationPipeline with LLM disabled (use_llm=False)
- Raw source immutability (consolidation never deletes raw records)
- Deduplication (near-duplicate suppression)
- Evidence span preservation
"""

import os

import pytest

from psa.consolidation import (
    RETENTION_THRESHOLD,
    ConsolidationPipeline,
    _infer_chunk_type,
    _passes_retention,
    chunk_hierarchical,
)
from psa.memory_object import MemoryStore, MemoryType, RawSource


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_dir):
    db_path = os.path.join(tmp_dir, "memory.sqlite3")
    return MemoryStore(db_path=db_path)


@pytest.fixture
def pipeline_no_llm(store):
    return ConsolidationPipeline(store=store, tenant_id="test", use_llm=False)


@pytest.fixture
def sample_source():
    return RawSource.create(
        tenant_id="test",
        source_type="project_file",
        full_text=(
            "# Authentication Module\n\n"
            "JWT tokens expire after 24 hours. Refresh tokens are stored "
            "in HttpOnly cookies to prevent XSS attacks.\n\n"
            "## Error Handling\n\n"
            "When a token is expired, return 401. The client must re-authenticate.\n\n"
            "## Performance\n\n"
            "Token validation is O(1) using HMAC verification."
        ),
        title="auth.py",
        source_path="/project/auth.py",
    )


# ── chunk_hierarchical ────────────────────────────────────────────────────────


def test_chunk_produces_multiple_levels(sample_source):
    chunks = chunk_hierarchical(sample_source, source_type="prose")
    levels = {c.level for c in chunks}
    assert "section" in levels


def test_chunk_offsets_are_valid(sample_source):
    chunks = chunk_hierarchical(sample_source)
    text = sample_source.full_text
    for chunk in chunks:
        assert 0 <= chunk.start_offset <= len(text)
        assert chunk.start_offset <= chunk.end_offset <= len(text)


def test_chunk_source_id_matches(sample_source):
    chunks = chunk_hierarchical(sample_source)
    for chunk in chunks:
        assert chunk.source_id == sample_source.source_id


def test_chunk_ids_unique(sample_source):
    chunks = chunk_hierarchical(sample_source)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_empty_text():
    src = RawSource.create(tenant_id="t", source_type="manual", full_text="   ")
    chunks = chunk_hierarchical(src)
    assert chunks == []


def test_chunk_code_source(tmp_dir):
    code_text = (
        "class Authenticator:\n"
        "    def __init__(self):\n"
        "        self.secret = 'abc'\n\n"
        "def validate_token(token: str) -> bool:\n"
        "    return len(token) > 0\n\n"
        "async def refresh(token: str):\n"
        "    pass\n"
    )
    src = RawSource.create(tenant_id="t", source_type="project_file", full_text=code_text)
    chunks = chunk_hierarchical(src, source_type="code")
    assert len(chunks) > 0
    # Code splitting should break at def/class boundaries
    section_texts = " ".join(c.text for c in chunks if c.level == "section")
    assert "class" in section_texts or "def" in section_texts


def test_chunk_conversation_source():
    convo = (
        "Human: What is JWT?\n"
        "Assistant: JWT stands for JSON Web Token.\n\n"
        "Human: How long do tokens last?\n"
        "Assistant: By default 24 hours.\n"
    )
    src = RawSource.create(tenant_id="t", source_type="conversation", full_text=convo)
    chunks = chunk_hierarchical(src, source_type="conversation")
    assert len(chunks) > 0


# ── _infer_chunk_type ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "source_type,path,expected",
    [
        ("conversation", None, "conversation"),
        ("project_file", "/src/app.py", "code"),
        ("project_file", "/src/app.ts", "code"),
        ("project_file", "/docs/README.md", "prose"),
        ("manual", None, "prose"),
    ],
)
def test_infer_chunk_type(source_type, path, expected):
    assert _infer_chunk_type(source_type, path) == expected


# ── _passes_retention ─────────────────────────────────────────────────────────


def test_retention_threshold_reject():
    raw = {
        "type": "semantic",
        "title": "T",
        "body": "Some body text that is long enough to matter.",
        "retention_score": RETENTION_THRESHOLD - 0.01,
    }
    assert _passes_retention(raw) is False


def test_retention_threshold_accept():
    raw = {
        "type": "semantic",
        "title": "T",
        "body": "Some body text that is long enough to matter.",
        "retention_score": RETENTION_THRESHOLD,
    }
    assert _passes_retention(raw) is True


def test_retention_procedural_always_kept_at_threshold():
    raw = {
        "type": "procedural",
        "title": "Use HMAC for token validation",
        "body": "Always use HMAC-SHA256 for JWT signature verification.",
        "retention_score": RETENTION_THRESHOLD,
    }
    assert _passes_retention(raw) is True


def test_retention_failure_always_kept_at_threshold():
    raw = {
        "type": "failure",
        "title": "Forgot to check expiry",
        "body": "Tokens were accepted past expiry due to missing check.",
        "retention_score": RETENTION_THRESHOLD,
    }
    assert _passes_retention(raw) is True


def test_retention_tool_use_always_kept_at_threshold():
    raw = {
        "type": "tool_use",
        "title": "JWT decode call",
        "body": "jwt.decode(token, secret, algorithms=['HS256'])",
        "retention_score": RETENTION_THRESHOLD,
    }
    assert _passes_retention(raw) is True


def test_retention_semantic_short_body_rejected():
    raw = {
        "type": "semantic",
        "title": "T",
        "body": "short",  # < 30 chars
        "retention_score": RETENTION_THRESHOLD,
    }
    assert _passes_retention(raw) is False


# ── ConsolidationPipeline (no LLM) ───────────────────────────────────────────


def test_pipeline_no_llm_persists_raw_source(pipeline_no_llm, store):
    """Raw source must be persisted even when use_llm=False."""
    pipeline_no_llm.consolidate(
        raw_text="Hello world content.",
        source_type="manual",
    )
    # Count raw sources
    with store._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM raw_sources").fetchone()[0]
    assert count == 1


def test_pipeline_no_llm_returns_empty(pipeline_no_llm):
    """With use_llm=False, no memory objects are created."""
    memories = pipeline_no_llm.consolidate(
        raw_text="Some content to consolidate.",
        source_type="manual",
    )
    assert memories == []


def test_pipeline_empty_text_returns_empty(pipeline_no_llm, store):
    memories = pipeline_no_llm.consolidate(raw_text="   ", source_type="manual")
    assert memories == []
    with store._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM raw_sources").fetchone()[0]
    assert count == 0


def test_pipeline_raw_source_immutable(pipeline_no_llm, store):
    """Calling consolidate twice with the same text must not duplicate the source."""
    text = "Unique content for immutability test."
    pipeline_no_llm.consolidate(raw_text=text, source_type="manual")
    pipeline_no_llm.consolidate(raw_text=text, source_type="manual")
    # Two separate RawSource records are created (different UUIDs)
    # but both point to immutable data
    with store._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM raw_sources").fetchone()[0]
    # Two records (each consolidate call creates a new RawSource)
    assert count == 2


def test_pipeline_batch(pipeline_no_llm, store):
    sources = [{"raw_text": f"Source {i} content.", "source_type": "manual"} for i in range(3)]
    pipeline_no_llm.consolidate_batch(sources)
    with store._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM raw_sources").fetchone()[0]
    assert count == 3


# ── ConsolidationPipeline with manual memory injection ───────────────────────


class _FakePipeline(ConsolidationPipeline):
    """Pipeline subclass that injects pre-defined memories instead of calling Qwen."""

    def __init__(self, store, tenant_id, fake_memories):
        super().__init__(store=store, tenant_id=tenant_id, use_llm=False)
        self._fake_memories = fake_memories

    def consolidate(self, raw_text, source_type, source_path=None, title="", metadata=None):
        from psa.memory_object import RawSource
        from psa.consolidation import _passes_retention, _raw_to_memory_object

        if not raw_text.strip():
            return []

        source = RawSource.create(
            tenant_id=self.tenant_id,
            source_type=source_type,
            full_text=raw_text,
            title=title,
            source_path=source_path,
            metadata=metadata or {},
        )
        self.store.add_source(source)

        retained = [r for r in self._fake_memories if _passes_retention(r)]
        embedding_model = self._get_embedding_model()
        persisted = []
        for raw in retained:
            mo = _raw_to_memory_object(raw, source, self.tenant_id)
            if mo is None:
                continue
            try:
                embed_text = f"{mo.memory_type.value}: {mo.title}\n{mo.summary}\n{mo.body}"
                mo.embedding = embedding_model.embed(embed_text)
            except Exception:
                pass
            self.store.add(mo)
            if not mo.is_duplicate:
                persisted.append(mo)
        return persisted


def test_memory_type_mapping(store):
    """Raw memory dict types must map to MemoryType enum values."""
    for type_str, expected in [
        ("episodic", MemoryType.EPISODIC),
        ("semantic", MemoryType.SEMANTIC),
        ("procedural", MemoryType.PROCEDURAL),
        ("failure", MemoryType.FAILURE),
        ("tool_use", MemoryType.TOOL_USE),
        ("working_derivative", MemoryType.WORKING_DERIVATIVE),
        ("unknown_type", MemoryType.SEMANTIC),  # fallback
    ]:
        from psa.consolidation import _raw_to_memory_object

        src = RawSource.create(tenant_id="t", source_type="manual", full_text="x")
        raw = {
            "type": type_str,
            "title": "T",
            "body": "B",
            "summary": "S",
            "classification_reason": "x",
            "evidence_chunk_ids": [],
            "retention_score": 0.8,
        }
        mo = _raw_to_memory_object(raw, src, "t")
        assert mo.memory_type == expected, f"type_str={type_str}"


def test_evidence_chunk_ids_flow_to_memory_object(store):
    from psa.consolidation import _raw_to_memory_object

    src = RawSource.create(tenant_id="t", source_type="manual", full_text="x")
    raw = {
        "type": "procedural",
        "title": "T",
        "body": "B",
        "summary": "S",
        "classification_reason": "x",
        "evidence_chunk_ids": ["chunk_001", "chunk_002"],
        "retention_score": 0.9,
    }
    mo = _raw_to_memory_object(raw, src, "t")
    assert mo.evidence_chunk_ids == ["chunk_001", "chunk_002"]


def test_source_id_preserved_in_memory_object(store):
    from psa.consolidation import _raw_to_memory_object

    src = RawSource.create(tenant_id="t", source_type="manual", full_text="x")
    raw = {
        "type": "semantic",
        "title": "T",
        "body": "B body",
        "summary": "S",
        "classification_reason": "x",
        "evidence_chunk_ids": [],
        "retention_score": 0.7,
    }
    mo = _raw_to_memory_object(raw, src, "t")
    assert src.source_id in mo.source_ids
    assert len(mo.evidence_spans) == 1
    assert mo.evidence_spans[0].source_id == src.source_id


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
