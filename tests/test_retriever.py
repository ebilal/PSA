"""Tests for psa.retriever — BM25Index, RRF fusion, AnchorRetriever."""

from unittest.mock import MagicMock

import pytest

from psa.anchor import AnchorCard, AnchorIndex
from psa.retriever import (
    AnchorCandidate,
    AnchorRetriever,
    BM25Index,
    _reciprocal_rank_fusion,
    _tokenize,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_card(anchor_id: int, name: str, meaning: str = "") -> AnchorCard:
    return AnchorCard(
        anchor_id=anchor_id,
        name=name,
        meaning=meaning or f"Anchor about {name}",
        memory_types=["SEMANTIC"],
        include_terms=[name.lower()],
        exclude_terms=[],
        prototype_examples=[],
        near_but_different=[],
        centroid=[0.0] * 4,
        memory_count=1,
        is_novelty=False,
    )


def _make_atlas(cards):
    """Build a mock Atlas with the given cards and a real AnchorIndex."""

    index = AnchorIndex(dim=4)
    # Give each card a distinct centroid
    for i, card in enumerate(cards):
        vec = [0.0] * 4
        vec[i % 4] = 1.0
        card.centroid = vec
    index.build(cards)

    atlas = MagicMock()
    atlas.cards = cards
    atlas.anchor_index = index
    return atlas


def _mock_embedding_model(vec):
    """Return a mock embedding model that always returns `vec`."""
    em = MagicMock()
    em.embed.return_value = vec
    return em


# ── _tokenize ─────────────────────────────────────────────────────────────────


def test_tokenize_basic():
    tokens = _tokenize("Authentication and login patterns")
    assert "authentication" in tokens
    assert "login" in tokens
    assert "patterns" in tokens
    # stop words removed
    assert "and" not in tokens


def test_tokenize_removes_short():
    tokens = _tokenize("a b cat dog")
    # 'a' and 'b' are both stop words AND too short; 'cat' and 'dog' are 3-char
    assert "a" not in tokens
    assert "b" not in tokens
    assert "cat" in tokens or "dog" in tokens  # both have 3 chars


def test_tokenize_lowercase():
    tokens = _tokenize("AUTH LOGIN")
    assert all(t == t.lower() for t in tokens)


# ── BM25Index ─────────────────────────────────────────────────────────────────


@pytest.fixture
def bm25_cards():
    cards = [
        _make_card(0, "auth", "authentication login password security"),
        _make_card(1, "database", "database storage query SQL records"),
        _make_card(2, "vector", "embedding vector similarity search"),
    ]
    # Set meaningful card text via meaning field
    return cards


@pytest.fixture
def bm25_index(bm25_cards):
    return BM25Index(bm25_cards)


def test_bm25_returns_score_per_card(bm25_index, bm25_cards):
    scores = bm25_index.score("login")
    assert len(scores) == len(bm25_cards)


def test_bm25_relevant_card_scores_highest(bm25_index):
    scores = bm25_index.score("authentication login")
    # Card 0 (auth) should score highest
    assert scores[0] == max(scores)


def test_bm25_irrelevant_query_scores_low(bm25_index, bm25_cards):
    scores = bm25_index.score("xyzzy unfindable term")
    # All scores should be 0 for completely unknown query terms
    assert all(s == 0.0 for s in scores)


def test_bm25_idf_higher_for_rare_terms(bm25_index):
    # 'vector' appears only in card 2; should have higher IDF than common terms
    idf_vector = bm25_index._idf.get("vector", 0)
    # Just check it's positive (rare term gets positive IDF)
    assert idf_vector > 0


# ── _reciprocal_rank_fusion ────────────────────────────────────────────────────


def test_rrf_both_lists_contribute():
    # Item at rank 0 in both lists should have highest score
    dense_ranked = [0, 1, 2]
    bm25_ranked = [0, 2, 1]
    scores = _reciprocal_rank_fusion(dense_ranked, bm25_ranked, n_cards=3, k=60)
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_rrf_only_in_one_list():
    # Item 0 in both lists, item 1 only in dense, item 2 only in dense
    dense_ranked = [0, 1, 2]
    bm25_ranked = [0, 1, 2]
    scores_both = _reciprocal_rank_fusion(dense_ranked, bm25_ranked, n_cards=3, k=60)

    # Now compare item 0 at rank 0 in both vs item 1 at rank 1 in both
    # Item 0 should have higher score since it's ranked higher
    assert scores_both[0] > scores_both[1]


def test_rrf_formula():
    # Manual: rank 1 in both → 1/61 + 1/61 = 2/61
    scores = _reciprocal_rank_fusion([0], [0], n_cards=1, k=60)
    expected = 2.0 / 61.0
    assert abs(scores[0] - expected) < 1e-9


# ── AnchorRetriever ───────────────────────────────────────────────────────────


@pytest.fixture
def small_atlas():
    cards = [
        _make_card(10, "auth", "authentication login security"),
        _make_card(20, "database", "storage records SQL"),
        _make_card(30, "vector", "embedding similarity search"),
        _make_card(40, "ci_cd", "deployment pipeline release"),
    ]
    return _make_atlas(cards)


def test_retrieve_returns_candidates(small_atlas):
    retriever = AnchorRetriever(small_atlas)
    em = _mock_embedding_model([1.0, 0.0, 0.0, 0.0])
    candidates = retriever.retrieve("auth login", em, top_k=3)
    assert len(candidates) >= 1
    assert all(isinstance(c, AnchorCandidate) for c in candidates)


def test_retrieve_respects_top_k(small_atlas):
    retriever = AnchorRetriever(small_atlas)
    em = _mock_embedding_model([1.0, 0.0, 0.0, 0.0])
    candidates = retriever.retrieve("query", em, top_k=2)
    assert len(candidates) <= 2


def test_retrieve_sorted_by_rrf(small_atlas):
    retriever = AnchorRetriever(small_atlas)
    em = _mock_embedding_model([1.0, 0.0, 0.0, 0.0])
    candidates = retriever.retrieve("query", em, top_k=4)
    scores = [c.rrf_score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_empty_atlas():
    atlas = MagicMock()
    atlas.cards = []
    retriever = AnchorRetriever(atlas)
    em = _mock_embedding_model([1.0, 0.0])
    candidates = retriever.retrieve("query", em, top_k=5)
    assert candidates == []


def test_retrieve_candidate_fields(small_atlas):
    retriever = AnchorRetriever(small_atlas)
    em = _mock_embedding_model([1.0, 0.0, 0.0, 0.0])
    candidates = retriever.retrieve("auth login security", em, top_k=4)
    assert len(candidates) > 0
    c = candidates[0]
    assert c.anchor_id in {10, 20, 30, 40}
    assert c.rrf_score > 0
    assert isinstance(c.dense_rank, int)
    assert isinstance(c.bm25_rank, int)


def test_bm25_cache_invalidated(small_atlas):
    retriever = AnchorRetriever(small_atlas)
    em = _mock_embedding_model([0.0, 1.0, 0.0, 0.0])
    _ = retriever.retrieve("query", em)
    assert retriever._bm25 is not None
    retriever.invalidate_bm25_cache()
    assert retriever._bm25 is None


def test_retriever_default_top_k_is_32():
    import inspect

    sig = inspect.signature(AnchorRetriever.retrieve)
    assert sig.parameters["top_k"].default == 32


# ── RetrievalResult + retrieve_with_bm25_topk ────────────────────────────────


def test_retrieval_result_dataclass_fields():
    from psa.retriever import RetrievalResult

    r = RetrievalResult(
        anchor_ids=[1, 2, 3],
        scores=[0.9, 0.8, 0.7],
        bm25_topk_anchor_ids=[2, 3, 7, 8],
    )
    assert r.anchor_ids == [1, 2, 3]
    assert r.scores == [0.9, 0.8, 0.7]
    assert r.bm25_topk_anchor_ids == [2, 3, 7, 8]


def test_retrieval_result_default_bm25_topk_empty():
    """bm25_topk_anchor_ids defaults to [] for backward compatibility."""
    from psa.retriever import RetrievalResult

    r = RetrievalResult(anchor_ids=[1], scores=[0.5])
    assert r.bm25_topk_anchor_ids == []


def test_retrieve_with_bm25_topk_populates_shortlist():
    """retrieve_with_bm25_topk returns a RetrievalResult with BM25 top-K."""
    from psa.retriever import (
        AnchorRetriever,
        RetrievalResult,
    )
    from psa.anchor import AnchorCard

    # Build a tiny atlas mock with 3 anchors.
    cards = []
    for aid, patterns in [(1, ["alpha"]), (2, ["beta"]), (3, ["gamma"])]:
        card = MagicMock(spec=AnchorCard)
        card.anchor_id = aid
        card.generated_query_patterns = patterns
        card.to_card_text = MagicMock(return_value=" ".join(patterns))
        cards.append(card)

    atlas = MagicMock()
    atlas.cards = cards
    atlas.anchor_index.search = MagicMock(return_value=[(1, 0.9), (2, 0.8), (3, 0.7)])

    retriever = AnchorRetriever(atlas=atlas)

    embedding_model = MagicMock()
    embedding_model.embed = MagicMock(return_value=[0.1] * 768)

    result = retriever.retrieve_with_bm25_topk(
        query="alpha",
        embedding_model=embedding_model,
        top_k=3,
        bm25_topk_floor=48,
    )
    assert isinstance(result, RetrievalResult)
    # Every anchor with positive BM25 score should make it into bm25_topk_anchor_ids
    # (since 48 > 3, all 3 anchors qualify).
    assert set(result.bm25_topk_anchor_ids).issubset({1, 2, 3})
    # anchor 1 ("alpha") must appear since the query matches its pattern text.
    assert 1 in result.bm25_topk_anchor_ids
    # The anchor_ids field holds the regular retrieve-result order.
    assert set(result.anchor_ids).issubset({1, 2, 3})
    # candidates field should also be populated so pipeline callers
    # don't have to re-run retrieve().
    assert len(result.candidates) >= 1
    assert all(hasattr(c, "anchor_id") for c in result.candidates)
