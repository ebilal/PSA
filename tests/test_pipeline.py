"""Tests for psa.pipeline — PSAPipeline end-to-end with mocked components."""

from unittest.mock import MagicMock, patch

import pytest

from psa.anchor import AnchorCard, AnchorIndex
from psa.atlas import Atlas
from psa.embeddings import EmbeddingModel
from psa.memory_object import MemoryObject, MemoryStore, MemoryType
from psa.pipeline import PSAPipeline, PSAResult, QueryTiming
from psa.retriever import AnchorCandidate
from psa.selector import AnchorSelector, SelectedAnchor


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_card(anchor_id: int, name: str = "test") -> AnchorCard:
    return AnchorCard(
        anchor_id=anchor_id,
        name=name,
        meaning=f"Anchor about {name}",
        memory_types=["EPISODIC"],
        include_terms=[name.lower()],
        exclude_terms=[],
        prototype_examples=[],
        near_but_different=[],
        centroid=[1.0, 0.0],
        memory_count=2,
        is_novelty=False,
    )


def _make_candidate(anchor_id: int, dense_score: float = 0.9) -> AnchorCandidate:
    card = _make_card(anchor_id)
    return AnchorCandidate(
        anchor_id=anchor_id,
        card=card,
        dense_score=dense_score,
        bm25_score=0.5,
        rrf_score=dense_score,
        dense_rank=1,
        bm25_rank=1,
    )


@pytest.fixture
def mock_store():
    store = MagicMock(spec=MemoryStore)
    store.query_by_anchor.return_value = []
    return store


@pytest.fixture
def mock_atlas():
    card = _make_card(1, "auth")
    index = AnchorIndex(dim=2)
    card.centroid = [1.0, 0.0]
    index.build([card])

    atlas = MagicMock(spec=Atlas)
    atlas.cards = [card]
    atlas.anchor_index = index
    atlas.version = 1
    return atlas


@pytest.fixture
def mock_embedding_model():
    em = MagicMock(spec=EmbeddingModel)
    em.embed.return_value = [1.0, 0.0]
    return em


@pytest.fixture
def pipeline(mock_store, mock_atlas, mock_embedding_model):
    return PSAPipeline(
        store=mock_store,
        atlas=mock_atlas,
        embedding_model=mock_embedding_model,
        selector=AnchorSelector.cosine(max_k=2),
        token_budget=1000,
        tenant_id="test_tenant",
        psa_mode="side-by-side",
    )


# ── QueryTiming ───────────────────────────────────────────────────────────────


def test_query_timing_total():
    t = QueryTiming(embed_ms=10, retrieve_ms=5, select_ms=3, fetch_ms=2, pack_ms=1)
    assert t.total_ms == 21.0


# ── PSAPipeline.query ─────────────────────────────────────────────────────────


def test_pipeline_query_returns_result(pipeline):
    result = pipeline.query("What auth pattern did we use?")
    assert isinstance(result, PSAResult)
    assert result.query == "What auth pattern did we use?"
    assert result.tenant_id == "test_tenant"
    assert result.psa_mode == "side-by-side"


def test_pipeline_result_has_text(pipeline):
    result = pipeline.query("auth query")
    assert isinstance(result.text, str)


def test_pipeline_result_token_count(pipeline):
    result = pipeline.query("auth query")
    assert isinstance(result.token_count, int)
    assert result.token_count >= 0


def test_pipeline_timing_fields(pipeline):
    result = pipeline.query("auth query")
    assert result.timing.embed_ms >= 0
    assert result.timing.retrieve_ms >= 0
    assert result.timing.select_ms >= 0
    assert result.timing.pack_ms >= 0


def test_pipeline_empty_atlas_returns_empty_result(mock_store, mock_embedding_model):
    atlas = MagicMock(spec=Atlas)
    atlas.cards = []
    index = MagicMock()
    index.search.return_value = []
    atlas.anchor_index = index

    p = PSAPipeline(
        store=mock_store,
        atlas=atlas,
        embedding_model=mock_embedding_model,
        token_budget=1000,
        tenant_id="t",
    )
    result = p.query("anything")
    assert result.selected_anchors == []
    assert result.candidates == []
    assert "no anchor candidates" in result.text


def test_pipeline_to_dict(pipeline):
    result = pipeline.query("auth query")
    d = result.to_dict()
    assert "query" in d
    assert "text" in d
    assert "token_count" in d
    assert "selected_anchor_ids" in d
    assert "timing_ms" in d
    assert "psa_mode" in d
    assert "tenant_id" in d


# ── PSAPipeline.search ────────────────────────────────────────────────────────


def test_pipeline_search_returns_dict(pipeline):
    result = pipeline.search("auth query")
    assert "query" in result
    assert "results" in result
    assert "psa_context" in result


def test_pipeline_search_results_list(pipeline):
    result = pipeline.search("auth query", n_results=3)
    assert isinstance(result["results"], list)


# ── PSAPipeline._fetch_memories ────────────────────────────────────────────────


def test_pipeline_deduplicates_memories(mock_store, mock_atlas, mock_embedding_model):
    """Memories with the same ID should appear only once in the output."""
    mo = MagicMock(spec=MemoryObject)
    mo.memory_object_id = "dup-id"
    mo.quality_score = 0.9

    mock_store.query_by_anchor.return_value = [mo, mo]  # same object twice

    p = PSAPipeline(
        store=mock_store,
        atlas=mock_atlas,
        embedding_model=mock_embedding_model,
        token_budget=1000,
        tenant_id="t",
    )

    selected = [
        SelectedAnchor(anchor_id=1, selector_score=0.9, mode="cosine",
                       candidate=_make_candidate(1)),
    ]
    memories = p._fetch_memories(selected)
    assert len(memories) == 1


# ── from_tenant factory (smoke test with mocks) ───────────────────────────────


def test_from_tenant_raises_without_atlas(tmp_path):
    """from_tenant raises FileNotFoundError if no atlas is built."""
    with patch("psa.pipeline.TenantManager") as mock_tm, \
         patch("psa.pipeline.MemoryStore"), \
         patch("psa.pipeline.EmbeddingModel"), \
         patch("psa.pipeline.AtlasManager") as mock_am:

        mock_tenant = MagicMock()
        mock_tenant.root_dir = str(tmp_path)
        mock_tenant.memory_db_path = str(tmp_path / "memory.sqlite3")
        mock_tm.return_value.get_or_create.return_value = mock_tenant

        mock_am_inst = MagicMock()
        mock_am_inst.get_atlas.return_value = None
        mock_am.return_value = mock_am_inst

        with pytest.raises(FileNotFoundError, match="No atlas found"):
            PSAPipeline.from_tenant("test_tenant", base_dir=str(tmp_path))
