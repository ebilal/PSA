"""
test_packer.py — Tests for psa.packer.EvidencePacker.

Covers:
- Budget enforcement (output stays within token_budget)
- Role sections ordered correctly (failure before procedural before raw)
- Untyped fallback (results with no MemoryObject go to RAW CONTEXT)
- Deduplication (same text appears only once)
- Empty input handling
- pack_memories_direct (Phase 3 path)
"""

from unittest.mock import MagicMock

import pytest

from psa.memory_object import EvidenceSpan, MemoryObject, MemoryType, RawSource
from psa.packer import EvidencePacker, PackedContext, _token_count


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_hit(text: str, similarity: float = 0.9, source: str = "test.py") -> dict:
    return {
        "text": text,
        "similarity": similarity,
        "source_file": source,
        "wing": "project",
        "room": "backend",
    }


def _make_memory(
    memory_type: MemoryType = MemoryType.SEMANTIC,
    title: str = "Test Memory",
    body: str = "Memory body content.",
    quality_score: float = 0.8,
) -> MemoryObject:
    return MemoryObject.create(
        tenant_id="test",
        memory_type=memory_type,
        title=title,
        body=body,
        summary="Summary.",
        source_ids=["src-001"],
        classification_reason="test",
        quality_score=quality_score,
    )


# ── EvidencePacker (raw hits path) ────────────────────────────────────────────


@pytest.fixture
def packer():
    return EvidencePacker()  # no store — all hits go to RAW CONTEXT in Phase 1


def test_empty_results(packer):
    result = packer.pack("test query", {"results": []})
    assert isinstance(result, PackedContext)
    assert result.text == "(no results)"
    assert result.token_count == 0
    assert result.untyped_count == 0


def test_missing_results_key(packer):
    result = packer.pack("test query", {})
    assert result.text == "(no results)"


def test_single_hit_in_raw_context(packer):
    results = {"results": [_make_hit("JWT tokens expire after 24 hours.")]}
    packed = packer.pack("auth patterns", results, token_budget=6000)
    assert "RAW CONTEXT" in packed.text
    assert "JWT tokens expire after 24 hours." in packed.text


def test_untyped_count(packer):
    hits = [_make_hit(f"Content {i}.") for i in range(3)]
    packed = packer.pack("query", {"results": hits})
    assert packed.untyped_count == 3


def test_token_budget_enforced(packer):
    # Create a large number of hits that would exceed budget
    long_text = "word " * 500  # ~500 tokens each
    hits = [_make_hit(long_text, source=f"file{i}.py") for i in range(20)]
    packed = packer.pack("query", {"results": hits}, token_budget=2000)
    assert packed.token_count <= 2000 + 100  # small tolerance for header


def test_deduplication(packer):
    # Same text in two hits — should appear only once
    same_text = "Identical content that should not be duplicated."
    hits = [_make_hit(same_text), _make_hit(same_text, source="other.py")]
    packed = packer.pack("query", {"results": hits})
    # Count occurrences of the text in the output
    assert packed.text.count("Identical content") == 1


def test_memory_ids_empty_when_no_store(packer):
    hits = [_make_hit("Some content.")]
    packed = packer.pack("query", {"results": hits})
    assert packed.memory_ids == []


def test_query_appears_in_header(packer):
    hits = [_make_hit("content")]
    packed = packer.pack("my specific query", {"results": hits})
    assert "my specific query" in packed.text


# ── pack_memories_direct (PSA path) ──────────────────────────────────────────


@pytest.fixture
def packer_direct():
    return EvidencePacker()


def test_pack_memories_direct_empty(packer_direct):
    result = packer_direct.pack_memories_direct("query", [])
    assert "no PSA memories" in result.text
    assert result.token_count == 0


def test_pack_memories_direct_role_sections(packer_direct):
    memories = [
        _make_memory(MemoryType.FAILURE, "Token expiry bug", "Forgot to validate exp claim."),
        _make_memory(MemoryType.PROCEDURAL, "HMAC validation", "Use HMAC-SHA256 for JWT."),
        _make_memory(MemoryType.SEMANTIC, "JWT structure", "Header.Payload.Signature format."),
    ]
    packed = packer_direct.pack_memories_direct("auth", memories)
    assert "FAILURE WARNINGS" in packed.text
    assert "PROCEDURAL GUIDANCE" in packed.text
    assert "FACTS & CONCEPTS" in packed.text


def test_pack_memories_direct_failure_before_procedural(packer_direct):
    """FAILURE WARNINGS must appear before PROCEDURAL GUIDANCE in the output."""
    memories = [
        _make_memory(MemoryType.PROCEDURAL, "Good pattern", "Use this approach."),
        _make_memory(MemoryType.FAILURE, "Bad pattern", "Do not do this."),
    ]
    packed = packer_direct.pack_memories_direct("query", memories)
    failure_pos = packed.text.find("FAILURE WARNINGS")
    procedural_pos = packed.text.find("PROCEDURAL GUIDANCE")
    assert failure_pos < procedural_pos


def test_pack_memories_direct_budget_enforcement(packer_direct):
    # Create many memories with long bodies
    memories = [_make_memory(MemoryType.SEMANTIC, f"Fact {i}", "word " * 200) for i in range(50)]
    packed = packer_direct.pack_memories_direct("query", memories, token_budget=1000)
    assert packed.token_count <= 1000 + 100


def test_pack_memories_direct_sorted_by_quality(packer_direct):
    """Higher quality memories should appear first within a section."""
    high_quality = _make_memory(MemoryType.SEMANTIC, "High quality fact", quality_score=0.95)
    low_quality = _make_memory(MemoryType.SEMANTIC, "Low quality fact", quality_score=0.40)
    packed = packer_direct.pack_memories_direct("query", [low_quality, high_quality])
    # High quality title should appear before low quality in the text
    hi_pos = packed.text.find("High quality fact")
    lo_pos = packed.text.find("Low quality fact")
    assert hi_pos != -1 and lo_pos != -1
    assert hi_pos < lo_pos


def test_pack_memories_direct_returns_memory_ids(packer_direct):
    memories = [
        _make_memory(MemoryType.EPISODIC, "Episode 1"),
        _make_memory(MemoryType.PROCEDURAL, "Pattern 1"),
    ]
    packed = packer_direct.pack_memories_direct("query", memories)
    assert len(packed.memory_ids) == 2
    for mo in memories:
        assert mo.memory_object_id in packed.memory_ids


def test_pack_memories_direct_all_memory_types(packer_direct):
    memories = [_make_memory(mtype, f"Memory {mtype.value}") for mtype in MemoryType]
    packed = packer_direct.pack_memories_direct("query", memories)
    # Should render at least some sections
    assert len(packed.sections) > 0
    assert packed.token_count > 0


# ── _token_count helper ───────────────────────────────────────────────────────


def test_token_count_empty():
    assert _token_count("") == 0


def test_token_count_approximate():
    # 400 chars ≈ 100 tokens
    text = "x" * 400
    assert _token_count(text) == 100


# ── _fetch_evidence_text ──────────────────────────────────────────────────────


def test_fetch_evidence_text_with_spans():
    from psa.packer import _fetch_evidence_text

    source = RawSource(
        source_id="src1",
        tenant_id="test",
        source_type="conversation",
        source_path="chat.jsonl",
        title="chat",
        full_text="Alice said she prefers PostgreSQL because of the JSONB support and extensibility.",
        created_at="2026-01-01T00:00:00Z",
    )
    span = EvidenceSpan(source_id="src1", start_offset=20, end_offset=80)
    store = MagicMock()
    store.get_source.return_value = source
    text = _fetch_evidence_text(store, [span], max_chars=500)
    assert "PostgreSQL" in text


def test_fetch_evidence_text_keyword_fallback():
    from psa.packer import _fetch_evidence_text

    source = RawSource(
        source_id="src1",
        tenant_id="test",
        source_type="conversation",
        source_path="chat.jsonl",
        title="chat",
        full_text="Some preamble. " * 20
        + "The PostgreSQL migration was decided. "
        + "Some epilogue. " * 20,
        created_at="2026-01-01T00:00:00Z",
    )
    store = MagicMock()
    store.get_source.return_value = source
    text = _fetch_evidence_text(
        store, [], max_chars=500, body_hint="PostgreSQL migration", source_ids=["src1"]
    )
    assert "PostgreSQL" in text


def test_fetch_evidence_text_no_store():
    from psa.packer import _fetch_evidence_text

    assert _fetch_evidence_text(None, []) is None
