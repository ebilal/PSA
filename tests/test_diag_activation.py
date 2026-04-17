"""Tests for psa.diag.activation — carry-rate computation."""

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


def _fake_atlas(anchor_ids: list[int]) -> MagicMock:
    atlas = MagicMock()
    cards = []
    for aid in anchor_ids:
        c = MagicMock()
        c.anchor_id = aid
        c.name = f"anchor-{aid}"
        c.memory_count = 1
        cards.append(c)
    atlas.cards = cards
    return atlas


def test_activation_carry_rate_perfect(tmp_path, monkeypatch):
    """Anchor selected and always carries → carry_rate 1.0."""
    from psa.diag.activation import activation_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir, [
        {
            "query_origin": "interactive",
            "selected_anchor_ids": [1],
            "packed_memories": [{"memory_id": "m1", "source_anchor_id": 1}],
        },
        {
            "query_origin": "interactive",
            "selected_anchor_ids": [1],
            "packed_memories": [{"memory_id": "m2", "source_anchor_id": 1}],
        },
    ])

    with patch("psa.diag.activation._load_atlas_for_tenant", return_value=_fake_atlas([1])):
        rows = activation_report("default", origins={"interactive"})

    assert len(rows) == 1
    assert rows[0].anchor_id == 1
    assert rows[0].n_selected == 2
    assert rows[0].n_carried == 2
    assert rows[0].carry_rate == 1.0


def test_activation_carry_rate_zero(tmp_path, monkeypatch):
    """Anchor selected but never carried → carry_rate 0.0."""
    from psa.diag.activation import activation_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir, [
        {
            "query_origin": "interactive",
            "selected_anchor_ids": [1],
            "packed_memories": [{"memory_id": "m1", "source_anchor_id": 2}],
        },
    ])

    with patch("psa.diag.activation._load_atlas_for_tenant", return_value=_fake_atlas([1, 2])):
        rows = activation_report("default", origins={"interactive"})

    row_1 = next(r for r in rows if r.anchor_id == 1)
    assert row_1.n_selected == 1
    assert row_1.n_carried == 0
    assert row_1.carry_rate == 0.0


def test_activation_origins_filter_excludes_benchmark(tmp_path, monkeypatch):
    from psa.diag.activation import activation_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir, [
        {
            "query_origin": "interactive",
            "selected_anchor_ids": [1],
            "packed_memories": [{"memory_id": "m", "source_anchor_id": 1}],
        },
        {
            "query_origin": "benchmark",
            "selected_anchor_ids": [1],
            "packed_memories": [{"memory_id": "m", "source_anchor_id": 1}],
        },
    ])

    with patch("psa.diag.activation._load_atlas_for_tenant", return_value=_fake_atlas([1])):
        rows = activation_report("default", origins={"interactive"})

    row_1 = next(r for r in rows if r.anchor_id == 1)
    assert row_1.n_selected == 1  # only the interactive record
