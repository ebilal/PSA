"""Tests for psa.full_atlas_scorer — FullAtlasScorer."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from psa.full_atlas_scorer import AnchorScore, FullAtlasScorer


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mock_card(anchor_id: int) -> MagicMock:
    """Create a mock AnchorCard with to_stable_card_text() and centroid."""
    card = MagicMock()
    card.anchor_id = anchor_id
    card.to_stable_card_text.return_value = f"Anchor {anchor_id}: some description"
    # 768-dim L2-normalized centroid
    vec = np.random.default_rng(anchor_id).random(768).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    card.centroid = vec
    return card


def _make_mock_atlas(n_cards: int = 8) -> MagicMock:
    """Create a mock Atlas with `cards` list of mock AnchorCards."""
    atlas = MagicMock()
    atlas.cards = [_make_mock_card(i) for i in range(n_cards)]
    return atlas


def _make_mock_cross_encoder(n_cards: int = 8) -> MagicMock:
    """Create a mock cross-encoder whose predict returns descending scores."""
    ce = MagicMock()
    ce.predict.return_value = np.array([float(n_cards - i) for i in range(n_cards)])
    return ce


def _make_query_vec(dim: int = 768) -> np.ndarray:
    rng = np.random.default_rng(42)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_score_all_returns_all_anchors():
    """score_all() returns one AnchorScore per atlas card."""
    n = 12
    atlas = _make_mock_atlas(n)
    ce = _make_mock_cross_encoder(n)
    scorer = FullAtlasScorer(cross_encoder=ce, atlas=atlas)

    results = scorer.score_all(query="what did I learn about auth?")

    assert len(results) == n
    for res in results:
        assert isinstance(res, AnchorScore)
        assert isinstance(res.anchor_id, int)
        assert isinstance(res.ce_score, float)
        assert isinstance(res.centroid, np.ndarray)
        assert res.centroid.shape == (768,)


def test_score_all_batches_pairs():
    """predict is called ONCE with all N pairs, not N times."""
    n = 16
    atlas = _make_mock_atlas(n)
    ce = _make_mock_cross_encoder(n)
    scorer = FullAtlasScorer(cross_encoder=ce, atlas=atlas)

    scorer.score_all(query="what tools did I use?")

    ce.predict.assert_called_once()
    call_args = ce.predict.call_args[0][0]  # first positional arg
    assert len(call_args) == n
    # Each element should be a (query, card_text) pair
    for query_str, card_text in call_args:
        assert isinstance(query_str, str)
        assert isinstance(card_text, str)


def test_score_all_sorted_by_score_desc():
    """Results are sorted by ce_score descending."""
    n = 8
    atlas = _make_mock_atlas(n)
    ce = _make_mock_cross_encoder(n)
    scorer = FullAtlasScorer(cross_encoder=ce, atlas=atlas)

    results = scorer.score_all(query="test query")

    scores = [r.ce_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_cosine_fallback_when_no_cross_encoder():
    """Without cross-encoder, uses cosine similarity to query_vec."""
    n = 8
    atlas = _make_mock_atlas(n)
    scorer = FullAtlasScorer(cross_encoder=None, atlas=atlas)
    query_vec = _make_query_vec()

    results = scorer.score_all(query="test query", query_vec=query_vec)

    assert len(results) == n
    for res in results:
        assert isinstance(res, AnchorScore)
        # Score should be a valid cosine similarity (between -1 and 1)
        assert -1.01 <= res.ce_score <= 1.01

    # Results should be sorted descending
    scores = [r.ce_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_cosine_fallback_requires_query_vec():
    """Without cross-encoder and without query_vec, raises ValueError."""
    atlas = _make_mock_atlas(4)
    scorer = FullAtlasScorer(cross_encoder=None, atlas=atlas)

    with pytest.raises((ValueError, TypeError)):
        scorer.score_all(query="no vec provided")


def test_from_model_path_loads_cross_encoder():
    """from_model_path() factory loads CrossEncoder via _load_cross_encoder."""
    atlas = _make_mock_atlas(4)
    mock_ce = MagicMock()

    with patch("psa.full_atlas_scorer._load_cross_encoder", return_value=mock_ce) as mock_load:
        scorer = FullAtlasScorer.from_model_path(model_path="/fake/model/path", atlas=atlas)

    mock_load.assert_called_once_with("/fake/model/path", device="mps")
    assert scorer.atlas is atlas
    assert scorer._cross_encoder is mock_ce


def test_override_used_when_provided():
    """Verify that override text is passed to cross_encoder.predict instead of card text."""
    n = 2
    atlas = _make_mock_atlas(n)
    ce = _make_mock_cross_encoder(n)

    # Override anchor_id=0 only
    override_dict = {0: "Override text for anchor 0"}

    scorer = FullAtlasScorer(
        cross_encoder=ce,
        atlas=atlas,
        card_text_override_by_anchor_id=override_dict,
    )

    query = "test query"
    results = scorer.score_all(query)

    # Verify cross_encoder.predict was called with override for anchor 0
    call_args = ce.predict.call_args[0][0]  # first positional argument
    assert len(call_args) == n

    # First pair should use override text
    assert call_args[0] == (query, "Override text for anchor 0")
    # Second pair should use card's original text
    assert call_args[1] == (query, "Anchor 1: some description")

    # Verify results are correct
    assert len(results) == n
    assert results[0].anchor_id == 0


def test_fallback_to_card_text_when_no_override():
    """Verify that card.to_stable_card_text() is used when no override provided."""
    n = 2
    atlas = _make_mock_atlas(n)
    ce = _make_mock_cross_encoder(n)

    # Create scorer WITHOUT override
    scorer = FullAtlasScorer(cross_encoder=ce, atlas=atlas)

    query = "test query"
    results = scorer.score_all(query)

    # Verify cross_encoder.predict was called with card text for both
    call_args = ce.predict.call_args[0][0]
    assert len(call_args) == n
    assert call_args[0] == (query, "Anchor 0: some description")
    assert call_args[1] == (query, "Anchor 1: some description")

    # Verify results are correct
    assert len(results) == n


def test_partial_override():
    """Verify that only specified anchor_ids use override, others fall back."""
    n = 3
    atlas = _make_mock_atlas(n)
    ce = _make_mock_cross_encoder(n)

    # Override only anchor_id 1
    override_dict = {1: "Override text 1"}

    scorer = FullAtlasScorer(
        cross_encoder=ce,
        atlas=atlas,
        card_text_override_by_anchor_id=override_dict,
    )

    query = "test query"
    scorer.score_all(query)

    call_args = ce.predict.call_args[0][0]

    assert call_args[0] == (query, "Anchor 0: some description")
    assert call_args[1] == (query, "Override text 1")
    assert call_args[2] == (query, "Anchor 2: some description")
