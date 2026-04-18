"""Tests for psa.advertisement.ledger — record_query_signals inline writer."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock


def _make_atlas():
    atlas = MagicMock()
    card1 = MagicMock()
    card1.anchor_id = 1
    card1.generated_query_patterns = ["cat vs dog behavior", "unrelated pattern"]
    card2 = MagicMock()
    card2.anchor_id = 2
    card2.generated_query_patterns = ["system architecture design"]
    atlas.cards = [card1, card2]
    by_id = {1: card1, 2: card2}
    atlas.get_anchor = lambda aid: by_id[aid]
    return atlas


def _config(tracking=True):
    from psa.advertisement.config import AdvertisementDecayConfig, ShadowConfig

    return AdvertisementDecayConfig(
        tracking_enabled=tracking, shadow=ShadowConfig()
    )


def test_record_query_signals_skips_when_tracking_disabled(tmp_path):
    from psa.advertisement.ledger import (
        compute_attribution,
        create_schema,
        record_query_signals,
    )

    db = sqlite3.connect(tmp_path / "t.sqlite3")
    create_schema(db)
    atlas = _make_atlas()
    attribution = compute_attribution(
        query="cat vs dog behavior",
        retrieved_anchor_ids=[1],
        atlas=atlas,
        bm25_topk_anchor_ids={1},
        epsilon=0.05,
    )
    record_query_signals(
        db=db,
        attribution=attribution,
        selected_anchor_ids={1},
        config=_config(tracking=False),
    )
    rows = db.execute("SELECT COUNT(*) FROM pattern_ledger").fetchone()
    assert rows[0] == 0


def test_record_query_signals_applies_retrieval_plus_pick(tmp_path):
    from psa.advertisement.ledger import (
        compute_attribution,
        create_schema,
        record_query_signals,
    )

    db = sqlite3.connect(tmp_path / "t.sqlite3")
    create_schema(db)
    atlas = _make_atlas()
    attribution = compute_attribution(
        query="cat vs dog behavior",
        retrieved_anchor_ids=[1],
        atlas=atlas,
        bm25_topk_anchor_ids={1},
        epsilon=0.05,
    )
    record_query_signals(
        db=db,
        attribution=attribution,
        selected_anchor_ids={1},  # anchor was picked
        config=_config(),
    )
    row = db.execute(
        "SELECT ledger FROM pattern_ledger WHERE pattern_text=?",
        ("cat vs dog behavior",),
    ).fetchone()
    # retrieval +1.0, pick +2.0, no ε-tie so full credit to argmax
    assert row[0] == 3.0


def test_record_query_signals_applies_decline(tmp_path):
    from psa.advertisement.ledger import (
        compute_attribution,
        create_schema,
        record_query_signals,
    )

    db = sqlite3.connect(tmp_path / "t.sqlite3")
    create_schema(db)
    atlas = _make_atlas()
    attribution = compute_attribution(
        query="cat vs dog behavior",
        retrieved_anchor_ids=[1],
        atlas=atlas,
        bm25_topk_anchor_ids={1},
        epsilon=0.05,
    )
    record_query_signals(
        db=db,
        attribution=attribution,
        selected_anchor_ids=set(),  # retrieved but not picked
        config=_config(),
    )
    row = db.execute(
        "SELECT ledger FROM pattern_ledger WHERE pattern_text=?",
        ("cat vs dog behavior",),
    ).fetchone()
    # retrieval +1.0, decline −0.25
    assert abs(row[0] - 0.75) < 1e-9


def test_record_query_signals_skips_when_bm25_floor_excludes_anchor(tmp_path):
    from psa.advertisement.ledger import (
        compute_attribution,
        create_schema,
        record_query_signals,
    )

    db = sqlite3.connect(tmp_path / "t.sqlite3")
    create_schema(db)
    atlas = _make_atlas()
    # Anchor 1 retrieved, but NOT on BM25-side shortlist — means lexical
    # contribution was negligible, so no template credit.
    attribution = compute_attribution(
        query="cat vs dog behavior",
        retrieved_anchor_ids=[1],
        atlas=atlas,
        bm25_topk_anchor_ids=set(),
        epsilon=0.05,
    )
    record_query_signals(
        db=db,
        attribution=attribution,
        selected_anchor_ids={1},
        config=_config(),
    )
    rows = db.execute("SELECT COUNT(*) FROM pattern_ledger").fetchone()
    assert rows[0] == 0


def test_credited_set_weight_division_with_epsilon_tie(tmp_path):
    from psa.advertisement.ledger import (
        compute_attribution,
        create_schema,
        record_query_signals,
    )

    atlas = MagicMock()
    card = MagicMock()
    card.anchor_id = 7
    # Two near-duplicates — likely ε-tied under BM25
    card.generated_query_patterns = [
        "token refresh flow",
        "token refresh flow step",
    ]
    atlas.cards = [card]
    atlas.get_anchor = lambda aid: card

    db = sqlite3.connect(tmp_path / "t.sqlite3")
    create_schema(db)
    attribution = compute_attribution(
        query="token refresh flow",
        retrieved_anchor_ids=[7],
        atlas=atlas,
        bm25_topk_anchor_ids={7},
        epsilon=0.5,  # generous tie window
    )
    record_query_signals(
        db=db,
        attribution=attribution,
        selected_anchor_ids={7},
        config=_config(),
    )
    rows = db.execute(
        "SELECT pattern_text, ledger FROM pattern_ledger ORDER BY pattern_text"
    ).fetchall()
    # Both patterns split the +3.0 credit (+1 retrieval, +2 pick)
    assert len(rows) == 2
    for _, lgr in rows:
        assert abs(lgr - 1.5) < 1e-9
