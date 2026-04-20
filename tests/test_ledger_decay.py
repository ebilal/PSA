"""Tests for psa.advertisement.ledger — decay + removal eligibility."""

# ruff: noqa: E731  -- terse lambdas are readable here; they mock stage-1 callbacks.

from __future__ import annotations

import math
import sqlite3


def _seed(tmp_path, ledger=2.0, cnc=0, grace_days_ago=30):
    from psa.advertisement.ledger import create_schema
    from datetime import datetime, timedelta, timezone

    db = sqlite3.connect(tmp_path / "t.sqlite3")
    create_schema(db)
    now = datetime.now(timezone.utc)
    grace_expires = (now - timedelta(days=grace_days_ago)).isoformat()
    db.execute(
        """
        INSERT INTO pattern_ledger
        (pattern_id, anchor_id, pattern_text, ledger, shadow_ledger,
         consecutive_negative_cycles, shadow_consecutive,
         grace_expires_at, created_at, last_updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "pid-x",
            1,
            "pattern",
            ledger,
            ledger,
            cnc,
            cnc,
            grace_expires,
            now.isoformat(),
            now.isoformat(),
        ),
    )
    db.commit()
    return db


def test_exponential_decay_multiplies_ledger(tmp_path):
    from psa.advertisement.ledger import apply_decay

    db = _seed(tmp_path, ledger=2.0)
    apply_decay(db, tau_days=45)
    row = db.execute("SELECT ledger FROM pattern_ledger").fetchone()
    expected = 2.0 * math.exp(-1 / 45)
    assert abs(row[0] - expected) < 1e-6


def test_decay_does_not_advance_negative_cycles_without_new_signals(tmp_path):
    from psa.advertisement.ledger import apply_decay

    db = _seed(tmp_path, ledger=-0.1, cnc=3)
    apply_decay(db, tau_days=45, removal_threshold=0.0)
    row = db.execute(
        "SELECT consecutive_negative_cycles FROM pattern_ledger"
    ).fetchone()
    assert row[0] == 3


def test_decay_does_not_reset_negative_cycles_without_new_signals(tmp_path):
    from psa.advertisement.ledger import apply_decay

    db = _seed(tmp_path, ledger=1.0, cnc=5)
    apply_decay(db, tau_days=45, removal_threshold=0.0)
    row = db.execute(
        "SELECT consecutive_negative_cycles FROM pattern_ledger"
    ).fetchone()
    assert row[0] == 5


def test_decay_applies_shadow_independently(tmp_path):
    from psa.advertisement.ledger import apply_decay

    db = _seed(tmp_path, ledger=2.0, cnc=0)
    # Set shadow to a different starting value to verify independent tracking.
    db.execute("UPDATE pattern_ledger SET shadow_ledger=-0.5, shadow_consecutive=2")
    db.commit()
    apply_decay(db, tau_days=45, removal_threshold=0.0)
    row = db.execute(
        "SELECT shadow_ledger, shadow_consecutive FROM pattern_ledger"
    ).fetchone()
    assert row[0] < 0  # still negative after decay
    assert row[1] == 2  # decay alone does not increment


def test_evaluate_removal_requires_grace_expired(tmp_path):
    """Pattern past negative threshold but inside grace → not eligible."""
    from psa.advertisement.ledger import evaluate_removal
    from datetime import datetime, timedelta, timezone
    from unittest.mock import MagicMock

    db = _seed(tmp_path, ledger=-2.0, cnc=20)
    # Override grace to be in the future
    future = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
    db.execute("UPDATE pattern_ledger SET grace_expires_at=?", (future,))
    db.commit()

    config = MagicMock(
        removal_threshold=0.0, sustained_cycles=14, min_patterns_floor=3,
        shadow=MagicMock(sustained_cycles=7),
    )
    atlas = MagicMock()
    atlas.get_anchor = lambda aid: MagicMock(
        generated_query_patterns=["p1", "p2", "p3", "pattern"]
    )
    shielded_fn = lambda tenant_id, anchor_ids: set()
    pinned_fn = lambda anchor_id, pattern_text: False

    result = evaluate_removal(
        db=db,
        atlas=atlas,
        tenant_id="default",
        config=config,
        shielded_anchor_fn=shielded_fn,
        pinned_fn=pinned_fn,
    )
    assert result.removal_candidates == []
    assert result.shadow_candidates == []


def test_evaluate_removal_requires_sustained_cycles(tmp_path):
    """Pattern just dipped negative → still eligible under sustained_cycles."""
    from psa.advertisement.ledger import evaluate_removal
    from unittest.mock import MagicMock

    db = _seed(tmp_path, ledger=-2.0, cnc=5)
    config = MagicMock(
        removal_threshold=0.0, sustained_cycles=14, min_patterns_floor=3,
        shadow=MagicMock(sustained_cycles=7),
    )
    atlas = MagicMock()
    atlas.get_anchor = lambda aid: MagicMock(
        generated_query_patterns=["p1", "p2", "p3", "pattern"]
    )
    shielded_fn = lambda tenant_id, anchor_ids: set()
    pinned_fn = lambda anchor_id, pattern_text: False

    result = evaluate_removal(
        db=db, atlas=atlas, tenant_id="default", config=config,
        shielded_anchor_fn=shielded_fn, pinned_fn=pinned_fn,
    )
    assert result.removal_candidates == []
    # shadow sustained_cycles=7 also not met
    assert result.shadow_candidates == []


def test_dormancy_alone_does_not_create_removal_candidate(tmp_path):
    """A negative ledger without sustained negative signal history is retained."""
    from psa.advertisement.ledger import evaluate_removal
    from unittest.mock import MagicMock

    db = _seed(tmp_path, ledger=-2.0, cnc=0)
    config = MagicMock(
        removal_threshold=0.0, sustained_cycles=21, min_patterns_floor=5,
        shadow=MagicMock(sustained_cycles=7),
    )
    atlas = MagicMock()
    atlas.get_anchor = lambda aid: MagicMock(
        generated_query_patterns=["p1", "p2", "p3", "p4", "p5", "pattern"]
    )
    shielded_fn = lambda tenant_id, anchor_ids: set()
    pinned_fn = lambda anchor_id, pattern_text: False

    result = evaluate_removal(
        db=db, atlas=atlas, tenant_id="default", config=config,
        shielded_anchor_fn=shielded_fn, pinned_fn=pinned_fn,
    )
    assert result.removal_candidates == []


def test_evaluate_removal_respects_min_patterns_floor(tmp_path):
    """Anchor with only 3 patterns → removal would drop below floor → refused."""
    from psa.advertisement.ledger import evaluate_removal
    from unittest.mock import MagicMock

    db = _seed(tmp_path, ledger=-2.0, cnc=20)
    config = MagicMock(
        removal_threshold=0.0, sustained_cycles=14, min_patterns_floor=3,
        shadow=MagicMock(sustained_cycles=7),
    )
    atlas = MagicMock()
    atlas.get_anchor = lambda aid: MagicMock(
        generated_query_patterns=["p1", "p2", "pattern"]  # only 3
    )
    shielded_fn = lambda tenant_id, anchor_ids: set()
    pinned_fn = lambda anchor_id, pattern_text: False

    result = evaluate_removal(
        db=db, atlas=atlas, tenant_id="default", config=config,
        shielded_anchor_fn=shielded_fn, pinned_fn=pinned_fn,
    )
    assert result.removal_candidates == []


def test_evaluate_removal_respects_p1_shield(tmp_path):
    """Anchor shielded by stage 1 P1 → eligible pattern excluded."""
    from psa.advertisement.ledger import evaluate_removal
    from unittest.mock import MagicMock

    db = _seed(tmp_path, ledger=-2.0, cnc=20)
    config = MagicMock(
        removal_threshold=0.0, sustained_cycles=14, min_patterns_floor=3,
        shadow=MagicMock(sustained_cycles=7),
    )
    atlas = MagicMock()
    atlas.get_anchor = lambda aid: MagicMock(
        generated_query_patterns=["p1", "p2", "p3", "p4", "pattern"]
    )
    shielded_fn = lambda tenant_id, anchor_ids: {1}  # anchor 1 is shielded
    pinned_fn = lambda anchor_id, pattern_text: False

    result = evaluate_removal(
        db=db, atlas=atlas, tenant_id="default", config=config,
        shielded_anchor_fn=shielded_fn, pinned_fn=pinned_fn,
    )
    assert result.removal_candidates == []


def test_evaluate_removal_happy_path(tmp_path):
    from psa.advertisement.ledger import evaluate_removal
    from unittest.mock import MagicMock

    db = _seed(tmp_path, ledger=-2.0, cnc=20)
    config = MagicMock(
        removal_threshold=0.0, sustained_cycles=14, min_patterns_floor=3,
        shadow=MagicMock(sustained_cycles=7),
    )
    atlas = MagicMock()
    atlas.get_anchor = lambda aid: MagicMock(
        generated_query_patterns=["p1", "p2", "p3", "p4", "pattern"]
    )
    shielded_fn = lambda tenant_id, anchor_ids: set()
    pinned_fn = lambda anchor_id, pattern_text: False

    result = evaluate_removal(
        db=db, atlas=atlas, tenant_id="default", config=config,
        shielded_anchor_fn=shielded_fn, pinned_fn=pinned_fn,
    )
    assert len(result.removal_candidates) == 1
    assert result.removal_candidates[0].pattern_text == "pattern"
    assert result.removal_candidates[0].anchor_id == 1


def test_apply_removals_mutates_refined_and_archives_row(tmp_path, monkeypatch):
    """End-to-end: refined file gets pattern removed, row soft-archived, marker written."""
    import json
    from psa.advertisement.ledger import (
        apply_removals,
        RemovalCandidate,
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    atlas_dir = tmp_path / ".psa" / "tenants" / "default" / "atlas_v1"
    atlas_dir.mkdir(parents=True)
    # Seed a refined file with two patterns; one will be removed.
    cards = [
        {
            "anchor_id": 1,
            "generated_query_patterns": ["keep this pattern", "remove this pattern"],
        }
    ]
    (atlas_dir / "anchor_cards_refined.json").write_text(json.dumps(cards))

    db = _seed(tmp_path, ledger=-2.0, cnc=20)
    candidate = RemovalCandidate(
        pattern_id="pid-x",
        anchor_id=1,
        pattern_text="remove this pattern",
        ledger=-2.0,
        consecutive_negative_cycles=20,
        shadow_ledger=-2.0,
        shadow_consecutive=20,
    )
    apply_removals(
        db=db,
        tenant_id="default",
        atlas_dir=str(atlas_dir),
        candidates=[candidate],
    )

    # Refined file updated
    after = json.loads((atlas_dir / "anchor_cards_refined.json").read_text())
    assert after[0]["generated_query_patterns"] == ["keep this pattern"]
    # Row archived
    row = db.execute(
        "SELECT removed_at, removal_reason, final_ledger FROM pattern_ledger"
    ).fetchone()
    assert row[0] is not None
    assert row[1] == "stage2_sustained_negative"
    assert row[2] == -2.0
    # Marker written
    marker = tmp_path / ".psa" / "tenants" / "default" / "atlas_reload_requested"
    assert marker.exists()


def test_apply_removals_creates_refined_from_base_when_absent(tmp_path, monkeypatch):
    """No refined file yet: read base, write refined, leave base untouched."""
    import json
    from psa.advertisement.ledger import apply_removals, RemovalCandidate

    monkeypatch.setenv("HOME", str(tmp_path))
    atlas_dir = tmp_path / ".psa" / "tenants" / "default" / "atlas_v1"
    atlas_dir.mkdir(parents=True)
    base_cards = [{"anchor_id": 1, "generated_query_patterns": ["keep", "remove"]}]
    (atlas_dir / "anchor_cards.json").write_text(json.dumps(base_cards))

    db = _seed(tmp_path, ledger=-2.0, cnc=20)
    candidate = RemovalCandidate(
        pattern_id="pid-x",
        anchor_id=1,
        pattern_text="remove",
        ledger=-2.0,
        consecutive_negative_cycles=20,
        shadow_ledger=-2.0,
        shadow_consecutive=20,
    )
    apply_removals(
        db=db, tenant_id="default", atlas_dir=str(atlas_dir), candidates=[candidate]
    )

    # Base untouched
    base_after = json.loads((atlas_dir / "anchor_cards.json").read_text())
    assert base_after == base_cards
    # Refined created with the removal applied
    refined = json.loads(
        (atlas_dir / "anchor_cards_refined.json").read_text()
    )
    assert refined[0]["generated_query_patterns"] == ["keep"]
