"""Lifecycle fast-path includes advertisement_decay_pass when tracking enabled."""

from __future__ import annotations

import sqlite3
import sys
from unittest.mock import MagicMock


def test_decay_pass_skipped_when_tracking_disabled(tmp_path, monkeypatch):
    from psa.lifecycle import advertisement_decay_pass
    from psa.advertisement.config import AdvertisementDecayConfig, ShadowConfig

    config = AdvertisementDecayConfig(tracking_enabled=False, shadow=ShadowConfig())
    summary = advertisement_decay_pass(
        tenant_id="default", config=config, atlas_or_loader=None
    )
    assert summary["skipped"] is True


def test_decay_pass_logs_summary(tmp_path, monkeypatch):
    from psa.lifecycle import advertisement_decay_pass
    from psa.advertisement.config import AdvertisementDecayConfig, ShadowConfig
    from psa.advertisement.ledger import create_schema, upsert_ledger

    monkeypatch.setenv("HOME", str(tmp_path))
    db_path = tmp_path / ".psa" / "tenants" / "default" / "memory.sqlite3"
    db_path.parent.mkdir(parents=True)
    db = sqlite3.connect(db_path)
    create_schema(db)
    upsert_ledger(db, "pid-1", 1, "some pattern", 1.0, 1.0, grace_days=21)
    db.close()

    atlas = MagicMock()
    atlas.get_anchor = lambda aid: MagicMock(
        generated_query_patterns=["some pattern", "x", "y", "z", "w"]
    )

    def shielded(tenant_id, anchor_ids):
        return set()

    def pinned(aid, text):
        return False

    config = AdvertisementDecayConfig(
        tracking_enabled=True, removal_enabled=False, shadow=ShadowConfig()
    )
    summary = advertisement_decay_pass(
        tenant_id="default",
        config=config,
        atlas_or_loader=atlas,
        shielded_anchor_fn=shielded,
        pinned_fn=pinned,
    )
    assert "n_active" in summary
    assert summary["n_active"] == 1
    assert summary["n_actually_removed_under_B"] == 0


def test_decay_pass_uses_lifecycle_guard_module_by_default(tmp_path, monkeypatch):
    from psa.lifecycle import advertisement_decay_pass
    from psa.advertisement.config import AdvertisementDecayConfig, ShadowConfig
    from psa.advertisement.ledger import create_schema, upsert_ledger

    monkeypatch.setenv("HOME", str(tmp_path))
    db_path = tmp_path / ".psa" / "tenants" / "default" / "memory.sqlite3"
    db_path.parent.mkdir(parents=True)
    db = sqlite3.connect(db_path)
    create_schema(db)
    upsert_ledger(db, "pid-1", 1, "some pattern", 1.0, 1.0, grace_days=21)
    db.close()

    atlas = MagicMock()
    atlas.anchor_dir = str(tmp_path / "atlas")
    atlas.get_anchor = lambda aid: MagicMock(
        generated_query_patterns=["some pattern", "x", "y", "z", "w", "v"]
    )

    # If lifecycle still imports psa.advertisement.decay directly for shielding,
    # this poisoned module will break the call.
    monkeypatch.setitem(sys.modules, "psa.advertisement.decay", object())

    import psa.advertisement.guards as guards

    monkeypatch.setattr(guards, "shielded_anchor_ids", lambda **kwargs: set())
    monkeypatch.setattr(guards, "is_pattern_pinned", lambda *args, **kwargs: False)

    config = AdvertisementDecayConfig(
        tracking_enabled=True, removal_enabled=False, shadow=ShadowConfig()
    )
    summary = advertisement_decay_pass(
        tenant_id="default",
        config=config,
        atlas_or_loader=atlas,
    )
    assert summary["n_active"] == 1
