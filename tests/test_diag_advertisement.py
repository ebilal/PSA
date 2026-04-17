"""Tests for psa.diag.advertisement — latent-capability gap."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def _write_trace(tenant_dir: Path, records: list[dict]) -> None:
    path = tenant_dir / "query_trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _fake_atlas(anchor_specs: list[tuple[int, int]]) -> MagicMock:
    """anchor_specs is [(anchor_id, memory_count), ...]."""
    atlas = MagicMock()
    cards = []
    for aid, mc in anchor_specs:
        c = MagicMock()
        c.anchor_id = aid
        c.name = f"anchor-{aid}"
        c.memory_count = mc
        cards.append(c)
    atlas.cards = cards
    return atlas


def test_advertisement_gap_positive_for_heavy_rarely_used_anchor(tmp_path, monkeypatch):
    """Anchor with lots of memories but low activation should have high positive gap."""
    from psa.diag.advertisement import advertisement_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"

    # Anchor 1: 100 memories, rarely activated.
    # Anchor 2: 2 memories, activated most queries.
    _write_trace(tenant_dir, [
        {"query_origin": "interactive", "selected_anchor_ids": [2]},
        {"query_origin": "interactive", "selected_anchor_ids": [2]},
        {"query_origin": "interactive", "selected_anchor_ids": [2, 1]},  # 1 activates once
        {"query_origin": "interactive", "selected_anchor_ids": [2]},
    ])

    with patch(
        "psa.diag.advertisement._load_atlas_for_tenant",
        return_value=_fake_atlas([(1, 100), (2, 2)]),
    ):
        rows = advertisement_report("default", origins={"interactive"})

    row_1 = next(r for r in rows if r.anchor_id == 1)
    row_2 = next(r for r in rows if r.anchor_id == 2)
    # Anchor 1: high memory, low activation → positive gap
    assert row_1.advertisement_gap > 0
    # Anchor 2: low memory, high activation → negative gap
    assert row_2.advertisement_gap < 0


def test_advertisement_memory_count_from_atlas_not_sqlite(tmp_path, monkeypatch):
    """Metric reads card.memory_count verbatim, does not recount from SQLite."""
    from psa.diag.advertisement import advertisement_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir, [
        {"query_origin": "interactive", "selected_anchor_ids": [1]},
    ])

    fake_atlas = _fake_atlas([(1, 42)])
    with patch("psa.diag.advertisement._load_atlas_for_tenant", return_value=fake_atlas):
        rows = advertisement_report("default", origins={"interactive"})

    assert next(r for r in rows if r.anchor_id == 1).memory_count == 42
