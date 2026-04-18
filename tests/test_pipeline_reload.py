"""Tests for PSAPipeline.reload_atlas and retriever.reindex_from_cards."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_reindex_from_cards_invalidates_bm25_cache():
    """After reindex_from_cards, retriever uses the new card list for BM25."""
    from psa.retriever import AnchorRetriever

    atlas = MagicMock()
    card_a = MagicMock()
    card_a.anchor_id = 1
    card_a.to_card_text = MagicMock(return_value="alpha beta gamma")
    atlas.cards = [card_a]

    retriever = AnchorRetriever(atlas=atlas)
    # Force BM25 build by accessing _get_bm25 once.
    _ = retriever._get_bm25()
    assert retriever._bm25 is not None

    card_b = MagicMock()
    card_b.anchor_id = 2
    card_b.to_card_text = MagicMock(return_value="delta epsilon zeta")
    new_cards = [card_b]

    retriever.reindex_from_cards(new_cards)
    # After reindex: BM25 cache must be invalidated so next access rebuilds.
    assert retriever._bm25 is None
    # atlas.cards should reflect the new list.
    assert retriever.atlas.cards == new_cards


def test_reload_atlas_method_exists_and_calls_retriever():
    """PSAPipeline.reload_atlas() re-reads the atlas and reindexes the retriever."""
    from psa.pipeline import PSAPipeline

    # Build a MagicMock-based pipeline to isolate reload_atlas behavior from
    # the full pipeline boot path.
    pipeline = MagicMock(spec=PSAPipeline)
    pipeline.atlas = MagicMock()
    pipeline.atlas.anchor_dir = "/fake/path"
    pipeline._retriever = MagicMock()

    reloaded_atlas = MagicMock()
    reloaded_atlas.cards = ["new-card"]

    # Use the real unbound method so we exercise PSAPipeline.reload_atlas.
    from unittest.mock import patch

    with patch("psa.atlas.Atlas.load", return_value=reloaded_atlas):
        PSAPipeline.reload_atlas(pipeline)

    # After reload: atlas replaced, retriever reindexed with new cards.
    assert pipeline.atlas is reloaded_atlas
    pipeline._retriever.reindex_from_cards.assert_called_once_with(["new-card"])
