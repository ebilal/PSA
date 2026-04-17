"""Tests for psa.diag.misses — below-threshold misses + near-miss anchors."""

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
        cards.append(c)
    atlas.cards = cards
    return atlas


def test_miss_report_counts_empty_selection_only(tmp_path, monkeypatch):
    from psa.diag.misses import miss_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir, [
        {"query_origin": "interactive", "result_kind": "synthesized",
         "top_anchor_scores": [{"anchor_id": 1, "score": 2.0, "selected": True, "rank": 1}]},
        {"query_origin": "interactive", "result_kind": "empty_selection",
         "top_anchor_scores": [{"anchor_id": 2, "score": 0.9, "selected": False, "rank": 1}]},
        {"query_origin": "interactive", "result_kind": "empty_selection",
         "top_anchor_scores": [{"anchor_id": 2, "score": 0.8, "selected": False, "rank": 1}]},
    ])

    with patch("psa.diag.misses._load_atlas_for_tenant", return_value=_fake_atlas([1, 2])):
        report = miss_report("default", origins={"interactive"})

    assert report.total_queries == 3
    assert report.empty_queries == 2
    assert abs(report.empty_rate - 2 / 3) < 1e-6


def test_near_miss_only_counts_rank_leq_3_in_empty_records(tmp_path, monkeypatch):
    """Near-miss requires rank ≤ 3 AND result_kind == 'empty_selection'."""
    from psa.diag.misses import miss_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir, [
        # Anchor 7 at rank 1 in a SYNTHESIZED record → NOT a near-miss.
        {"query_origin": "interactive", "result_kind": "synthesized",
         "top_anchor_scores": [{"anchor_id": 7, "score": 3.0, "selected": True, "rank": 1}]},
        # Anchor 7 at rank 4 in an empty record → NOT a near-miss (rank too high).
        {"query_origin": "interactive", "result_kind": "empty_selection",
         "top_anchor_scores": [
             {"anchor_id": 8, "score": 1.0, "selected": False, "rank": 1},
             {"anchor_id": 9, "score": 0.9, "selected": False, "rank": 2},
             {"anchor_id": 10, "score": 0.8, "selected": False, "rank": 3},
             {"anchor_id": 7, "score": 0.7, "selected": False, "rank": 4},
         ]},
        # Anchor 7 at rank 2 in an empty record → IS a near-miss.
        {"query_origin": "interactive", "result_kind": "empty_selection",
         "top_anchor_scores": [{"anchor_id": 7, "score": 0.95, "selected": False, "rank": 2}]},
    ])

    with patch("psa.diag.misses._load_atlas_for_tenant", return_value=_fake_atlas(list(range(1, 20)))):
        report = miss_report("default", origins={"interactive"})

    anchor_7 = next((t for t in report.near_miss_anchors if t[0] == 7), None)
    assert anchor_7 is not None, "anchor 7 must be a near-miss"
    assert anchor_7[1] == 1  # exactly one qualifying near-miss
