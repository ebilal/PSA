"""Tests for psa.advertisement.ledger — schema, CRUD, archival."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _conn(tmp_path):
    return sqlite3.connect(tmp_path / "ledger.sqlite3")


def test_pattern_id_is_deterministic():
    from psa.advertisement.ledger import pattern_id_for

    a = pattern_id_for(42, "how does the token refresh flow")
    b = pattern_id_for(42, "how does the token refresh flow")
    assert a == b
    # Normalization: different formatting yields same id
    c = pattern_id_for(42, "  How Does The Token REFRESH flow  ")
    assert c == a
    # Different anchor → different id
    d = pattern_id_for(43, "how does the token refresh flow")
    assert d != a


def test_create_schema_idempotent(tmp_path):
    from psa.advertisement.ledger import create_schema

    with _conn(tmp_path) as db:
        create_schema(db)
        create_schema(db)  # second call must not raise
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pattern_ledger'"
        ).fetchall()
        assert rows == [("pattern_ledger",)]


def test_upsert_inserts_new_row_with_grace_expires(tmp_path):
    from psa.advertisement.ledger import create_schema, upsert_ledger

    with _conn(tmp_path) as db:
        create_schema(db)
        upsert_ledger(
            db,
            pattern_id="pid-1",
            anchor_id=5,
            pattern_text="test pattern",
            ledger_delta=1.0,
            shadow_delta=1.0,
            grace_days=21,
        )
        row = db.execute("SELECT * FROM pattern_ledger").fetchone()
        cols = [c[0] for c in db.execute("SELECT * FROM pattern_ledger").description]
        d = dict(zip(cols, row))
        assert d["pattern_id"] == "pid-1"
        assert d["anchor_id"] == 5
        assert d["pattern_text"] == "test pattern"
        assert d["ledger"] == 1.0
        assert d["shadow_ledger"] == 1.0
        assert d["consecutive_negative_cycles"] == 0
        assert d["removed_at"] is None
        assert d["grace_expires_at"] is not None
        # grace should be ~21 days in the future
        g = datetime.fromisoformat(d["grace_expires_at"])
        delta = (g - datetime.now(timezone.utc)).days
        assert 19 <= delta <= 22


def test_upsert_accumulates_on_existing_row(tmp_path):
    from psa.advertisement.ledger import create_schema, upsert_ledger

    with _conn(tmp_path) as db:
        create_schema(db)
        upsert_ledger(db, "pid-1", 5, "p", 1.0, 0.5, grace_days=21)
        upsert_ledger(db, "pid-1", 5, "p", 2.5, 0.5, grace_days=21)
        row = db.execute(
            "SELECT ledger, shadow_ledger FROM pattern_ledger"
        ).fetchone()
        assert row[0] == 3.5
        assert row[1] == 1.0


def test_active_index_excludes_archived(tmp_path):
    from psa.advertisement.ledger import create_schema, upsert_ledger

    with _conn(tmp_path) as db:
        create_schema(db)
        upsert_ledger(db, "pid-1", 5, "p1", 1.0, 1.0, grace_days=21)
        upsert_ledger(db, "pid-2", 5, "p2", 1.0, 1.0, grace_days=21)
        db.execute(
            "UPDATE pattern_ledger SET removed_at=?, removal_reason=? WHERE pattern_id=?",
            ("2026-04-17T00:00:00+00:00", "test", "pid-1"),
        )
        active = db.execute(
            "SELECT pattern_id FROM pattern_ledger WHERE removed_at IS NULL"
        ).fetchall()
        assert active == [("pid-2",)]
