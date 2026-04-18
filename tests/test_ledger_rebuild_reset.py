"""Atlas rebuild: archive all active rows, insert fresh rows for new patterns."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock


def _seed_active(db, n=3):
    from psa.advertisement.ledger import create_schema, upsert_ledger
    create_schema(db)
    for i in range(n):
        upsert_ledger(
            db, f"pid-{i}", i, f"pattern {i}", 1.0, 1.0, grace_days=21
        )


def test_reset_ledger_on_rebuild_archives_all_and_inserts_fresh(tmp_path):
    from psa.advertisement.ledger import reset_ledger_on_rebuild

    db = sqlite3.connect(tmp_path / "t.sqlite3")
    _seed_active(db, n=3)

    card1 = MagicMock()
    card1.anchor_id = 10
    card1.generated_query_patterns = ["new pattern alpha", "new pattern beta"]
    card2 = MagicMock()
    card2.anchor_id = 11
    card2.generated_query_patterns = ["fresh pattern gamma"]
    new_atlas = MagicMock(cards=[card1, card2])

    reset_ledger_on_rebuild(db=db, new_atlas=new_atlas, grace_days=21)

    # Old rows archived
    archived = db.execute(
        "SELECT COUNT(*) FROM pattern_ledger WHERE removal_reason = 'atlas_rebuild_reset'"
    ).fetchone()[0]
    assert archived == 3

    # New rows inserted
    active = db.execute(
        "SELECT anchor_id, pattern_text FROM pattern_ledger WHERE removed_at IS NULL ORDER BY anchor_id, pattern_text"
    ).fetchall()
    assert active == [
        (10, "new pattern alpha"),
        (10, "new pattern beta"),
        (11, "fresh pattern gamma"),
    ]


def test_reset_ledger_on_rebuild_handles_empty_old_state(tmp_path):
    """First-ever rebuild has no active rows — should not error."""
    from psa.advertisement.ledger import create_schema, reset_ledger_on_rebuild

    db = sqlite3.connect(tmp_path / "t.sqlite3")
    create_schema(db)

    card = MagicMock()
    card.anchor_id = 1
    card.generated_query_patterns = ["pattern x"]
    new_atlas = MagicMock(cards=[card])

    reset_ledger_on_rebuild(db=db, new_atlas=new_atlas, grace_days=21)

    active = db.execute(
        "SELECT anchor_id, pattern_text FROM pattern_ledger WHERE removed_at IS NULL"
    ).fetchall()
    assert active == [(1, "pattern x")]
