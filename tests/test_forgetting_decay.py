"""Tests for psa.forgetting.decay — D1 rule + P1 shield + P3 pin."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch


def _fake_atlas(patterns_by_anchor: dict[int, list[str]]) -> MagicMock:
    atlas = MagicMock()
    cards = []
    for aid, patterns in patterns_by_anchor.items():
        card = MagicMock()
        card.anchor_id = aid
        card.name = f"anchor-{aid}"
        card.generated_query_patterns = list(patterns)
        cards.append(card)
    atlas.cards = cards
    atlas.anchor_dir = "/unused/in/tests"
    return atlas


def _write_trace(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


NOW = datetime.fromisoformat("2026-04-17T12:00:00+00:00")


def test_d1_pattern_too_young_is_not_candidate(tmp_path, monkeypatch):
    """Age < grace_days → pattern is not a decay candidate, even without reinforcement."""
    from psa.forgetting.decay import DecayParams, decay_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir / "query_trace.jsonl", [])

    atlas = _fake_atlas({1: ["young pattern"]})
    from psa.forgetting.metadata import PatternMetadata, metadata_key, save_metadata
    atlas.anchor_dir = str(tmp_path / "atlas_dir")
    (Path(atlas.anchor_dir)).mkdir(parents=True, exist_ok=True)
    recently = (NOW - timedelta(days=5)).isoformat()
    save_metadata(atlas.anchor_dir, {
        metadata_key(1, "young pattern"): PatternMetadata(
            anchor_id=1, pattern="young pattern", source="manual",
            created_at=recently,
        )
    })

    params = DecayParams(grace_days=30, decay_window_days=60,
                         low_activation_percentile=0, min_anchor_activations=0)
    with patch("psa.forgetting.decay._load_atlas_for_tenant", return_value=atlas):
        with patch("psa.forgetting.decay._now_utc", return_value=NOW):
            report = decay_report("default", params=params, origins={"interactive"})
    assert report.n_patterns_removed == 0


def test_d1_old_unreinforced_is_candidate(tmp_path, monkeypatch):
    from psa.forgetting.decay import DecayParams, decay_report
    from psa.forgetting.metadata import PatternMetadata, metadata_key, save_metadata

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir / "query_trace.jsonl", [])

    atlas = _fake_atlas({1: ["stale pattern"]})
    atlas.anchor_dir = str(tmp_path / "atlas_dir")
    (Path(atlas.anchor_dir)).mkdir(parents=True, exist_ok=True)
    long_ago = (NOW - timedelta(days=200)).isoformat()
    save_metadata(atlas.anchor_dir, {
        metadata_key(1, "stale pattern"): PatternMetadata(
            anchor_id=1, pattern="stale pattern", source="manual",
            created_at=long_ago,
        )
    })

    params = DecayParams(grace_days=30, decay_window_days=60,
                         low_activation_percentile=0, min_anchor_activations=0)
    with patch("psa.forgetting.decay._load_atlas_for_tenant", return_value=atlas):
        with patch("psa.forgetting.decay._now_utc", return_value=NOW):
            report = decay_report("default", params=params, origins={"interactive"})
    assert report.n_patterns_removed == 1
    assert report.removed_patterns[0].anchor_id == 1
    assert report.removed_patterns[0].pattern == "stale pattern"
    assert report.removed_patterns[0].reason == "stale_unreinforced"


def test_p1_shield_below_absolute_floor(tmp_path, monkeypatch):
    """Anchor below min_anchor_activations is shielded — patterns NOT candidates."""
    from psa.forgetting.decay import DecayParams, decay_report
    from psa.forgetting.metadata import PatternMetadata, metadata_key, save_metadata

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    # Anchor 1 activated only 3 times in the window.
    _write_trace(tenant_dir / "query_trace.jsonl", [
        {"timestamp": "2026-04-10T00:00:00+00:00", "query": "q",
         "query_origin": "interactive", "selected_anchor_ids": [1]}
        for _ in range(3)
    ])

    atlas = _fake_atlas({1: ["stale pattern"]})
    atlas.anchor_dir = str(tmp_path / "atlas_dir")
    (Path(atlas.anchor_dir)).mkdir(parents=True, exist_ok=True)
    long_ago = (NOW - timedelta(days=200)).isoformat()
    save_metadata(atlas.anchor_dir, {
        metadata_key(1, "stale pattern"): PatternMetadata(
            anchor_id=1, pattern="stale pattern", source="manual", created_at=long_ago,
        )
    })

    # Floor is 10 activations; anchor 1 has 3. Shielded.
    params = DecayParams(grace_days=30, decay_window_days=60,
                         low_activation_percentile=0, min_anchor_activations=10)
    with patch("psa.forgetting.decay._load_atlas_for_tenant", return_value=atlas):
        with patch("psa.forgetting.decay._now_utc", return_value=NOW):
            report = decay_report("default", params=params, origins={"interactive"})
    assert report.n_patterns_removed == 0
    assert any(s.anchor_id == 1 for s in report.shielded_anchors)


def test_p3_pinned_pattern_not_pruned(tmp_path, monkeypatch):
    from psa.forgetting.decay import DecayParams, decay_report
    from psa.forgetting.metadata import PatternMetadata, metadata_key, save_metadata

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir / "query_trace.jsonl", [])

    atlas = _fake_atlas({1: ["pinned stale"]})
    atlas.anchor_dir = str(tmp_path / "atlas_dir")
    (Path(atlas.anchor_dir)).mkdir(parents=True, exist_ok=True)
    long_ago = (NOW - timedelta(days=200)).isoformat()
    save_metadata(atlas.anchor_dir, {
        metadata_key(1, "pinned stale"): PatternMetadata(
            anchor_id=1, pattern="pinned stale", source="manual",
            created_at=long_ago, pinned=True,
        )
    })

    params = DecayParams(grace_days=30, decay_window_days=60,
                         low_activation_percentile=0, min_anchor_activations=0)
    with patch("psa.forgetting.decay._load_atlas_for_tenant", return_value=atlas):
        with patch("psa.forgetting.decay._now_utc", return_value=NOW):
            report = decay_report("default", params=params, origins={"interactive"})
    assert report.n_patterns_removed == 0
    assert report.n_patterns_pinned_exempt == 1


def test_source_grouped_counts(tmp_path, monkeypatch):
    """n_patterns_by_source_removed groups correctly."""
    from psa.forgetting.decay import DecayParams, decay_report
    from psa.forgetting.metadata import PatternMetadata, metadata_key, save_metadata

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir / "query_trace.jsonl", [])

    atlas = _fake_atlas({1: ["a1", "a2", "b1"]})
    atlas.anchor_dir = str(tmp_path / "atlas_dir")
    (Path(atlas.anchor_dir)).mkdir(parents=True, exist_ok=True)
    long_ago = (NOW - timedelta(days=200)).isoformat()
    save_metadata(atlas.anchor_dir, {
        metadata_key(1, "a1"): PatternMetadata(
            anchor_id=1, pattern="a1", source="atlas_build", created_at=long_ago),
        metadata_key(1, "a2"): PatternMetadata(
            anchor_id=1, pattern="a2", source="atlas_build", created_at=long_ago),
        metadata_key(1, "b1"): PatternMetadata(
            anchor_id=1, pattern="b1", source="refinement", created_at=long_ago),
    })

    params = DecayParams(grace_days=30, decay_window_days=60,
                         low_activation_percentile=0, min_anchor_activations=0)
    with patch("psa.forgetting.decay._load_atlas_for_tenant", return_value=atlas):
        with patch("psa.forgetting.decay._now_utc", return_value=NOW):
            report = decay_report("default", params=params, origins={"interactive"})
    assert report.n_patterns_removed == 3
    assert report.n_patterns_by_source_removed["atlas_build"] == 2
    assert report.n_patterns_by_source_removed["refinement"] == 1


def test_backfill_happens_on_decay_run(tmp_path, monkeypatch):
    """Patterns without metadata get stamped source='unknown' + now, and persisted."""
    from psa.forgetting.decay import DecayParams, decay_report
    from psa.forgetting.metadata import load_metadata

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir / "query_trace.jsonl", [])

    atlas = _fake_atlas({1: ["orphan pattern"]})
    atlas.anchor_dir = str(tmp_path / "atlas_dir")
    (Path(atlas.anchor_dir)).mkdir(parents=True, exist_ok=True)
    # No metadata file on disk.

    params = DecayParams(grace_days=30, decay_window_days=60,
                         low_activation_percentile=0, min_anchor_activations=0)
    with patch("psa.forgetting.decay._load_atlas_for_tenant", return_value=atlas):
        with patch("psa.forgetting.decay._now_utc", return_value=NOW):
            report = decay_report("default", params=params, origins={"interactive"})
    assert report.n_patterns_backfilled_this_run == 1
    # Because created_at == now, age is 0 → within grace → no removal.
    assert report.n_patterns_removed == 0
    # Metadata file persisted.
    loaded = load_metadata(atlas.anchor_dir)
    assert "1::orphan pattern" in loaded
    assert loaded["1::orphan pattern"].source == "unknown"
