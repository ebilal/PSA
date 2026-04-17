"""Tests for psa.advertisement.reinforcement — derived per-run reinforcement map."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock


def _write_trace(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _fake_atlas_with_patterns(patterns_by_anchor: dict[int, list[str]]) -> MagicMock:
    atlas = MagicMock()
    cards = []
    for aid, patterns in patterns_by_anchor.items():
        card = MagicMock()
        card.anchor_id = aid
        card.generated_query_patterns = patterns
        cards.append(card)
    atlas.cards = cards
    return atlas


def test_reinforcement_substring_match_activates(tmp_path):
    from psa.advertisement.reinforcement import compute_reinforcement

    trace_path = tmp_path / "query_trace.jsonl"
    _write_trace(
        trace_path,
        [
            {
                "timestamp": "2026-04-15T10:00:00+00:00",
                "query": "how does the token refresh flow work in prod",
                "query_origin": "interactive",
                "selected_anchor_ids": [1],
            },
        ],
    )
    atlas = _fake_atlas_with_patterns({1: ["token refresh flow"]})

    window_start = datetime.fromisoformat("2026-04-01T00:00:00+00:00")
    rmap = compute_reinforcement(
        atlas,
        str(trace_path),
        origins={"interactive"},
        window_start=window_start,
    )
    assert "1::token refresh flow" in rmap


def test_reinforcement_without_activation_does_not_count(tmp_path):
    """Anchor didn't activate — pattern is not reinforced even if substring matches."""
    from psa.advertisement.reinforcement import compute_reinforcement

    trace_path = tmp_path / "query_trace.jsonl"
    _write_trace(
        trace_path,
        [
            {
                "timestamp": "2026-04-15T10:00:00+00:00",
                "query": "token refresh flow details",
                "query_origin": "interactive",
                "selected_anchor_ids": [2],  # anchor 2 selected, not 1
            },
        ],
    )
    atlas = _fake_atlas_with_patterns({1: ["token refresh flow"]})

    rmap = compute_reinforcement(
        atlas,
        str(trace_path),
        origins={"interactive"},
        window_start=datetime.fromisoformat("2026-04-01T00:00:00+00:00"),
    )
    assert "1::token refresh flow" not in rmap


def test_reinforcement_origin_filter_excludes_benchmark(tmp_path):
    from psa.advertisement.reinforcement import compute_reinforcement

    trace_path = tmp_path / "query_trace.jsonl"
    _write_trace(
        trace_path,
        [
            {
                "timestamp": "2026-04-15T10:00:00+00:00",
                "query": "token refresh flow",
                "query_origin": "benchmark",
                "selected_anchor_ids": [1],
            },
        ],
    )
    atlas = _fake_atlas_with_patterns({1: ["token refresh flow"]})

    rmap = compute_reinforcement(
        atlas,
        str(trace_path),
        origins={"interactive"},
        window_start=datetime.fromisoformat("2026-04-01T00:00:00+00:00"),
    )
    assert "1::token refresh flow" not in rmap


def test_reinforcement_window_bound_excludes_old_records(tmp_path):
    from psa.advertisement.reinforcement import compute_reinforcement

    trace_path = tmp_path / "query_trace.jsonl"
    _write_trace(
        trace_path,
        [
            {
                "timestamp": "2020-01-01T00:00:00+00:00",  # way before window
                "query": "token refresh flow",
                "query_origin": "interactive",
                "selected_anchor_ids": [1],
            },
        ],
    )
    atlas = _fake_atlas_with_patterns({1: ["token refresh flow"]})

    rmap = compute_reinforcement(
        atlas,
        str(trace_path),
        origins={"interactive"},
        window_start=datetime.fromisoformat("2026-04-01T00:00:00+00:00"),
    )
    assert "1::token refresh flow" not in rmap


def test_reinforcement_takes_most_recent_match(tmp_path):
    from psa.advertisement.reinforcement import compute_reinforcement

    trace_path = tmp_path / "query_trace.jsonl"
    _write_trace(
        trace_path,
        [
            {
                "timestamp": "2026-04-10T00:00:00+00:00",
                "query": "token refresh flow",
                "query_origin": "interactive",
                "selected_anchor_ids": [1],
            },
            {
                "timestamp": "2026-04-15T00:00:00+00:00",
                "query": "refresh flow token",  # still substring-matches "token refresh flow"? No.
                "query_origin": "interactive",
                "selected_anchor_ids": [1],
            },
            {
                "timestamp": "2026-04-20T00:00:00+00:00",
                "query": "the token refresh flow in prod",
                "query_origin": "interactive",
                "selected_anchor_ids": [1],
            },
        ],
    )
    atlas = _fake_atlas_with_patterns({1: ["token refresh flow"]})

    rmap = compute_reinforcement(
        atlas,
        str(trace_path),
        origins={"interactive"},
        window_start=datetime.fromisoformat("2026-04-01T00:00:00+00:00"),
    )
    # First and third records match. Should hold the LATER timestamp.
    got = rmap["1::token refresh flow"]
    assert got == datetime.fromisoformat("2026-04-20T00:00:00+00:00")


def test_reinforcement_missing_trace_file_returns_empty(tmp_path):
    from psa.advertisement.reinforcement import compute_reinforcement

    atlas = _fake_atlas_with_patterns({1: ["anything"]})
    rmap = compute_reinforcement(
        atlas,
        str(tmp_path / "does_not_exist.jsonl"),
        origins={"interactive"},
        window_start=datetime.fromisoformat("2026-04-01T00:00:00+00:00"),
    )
    assert rmap == {}
