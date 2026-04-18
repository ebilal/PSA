"""Tests for psa.advertisement.writer — candidate + meta + detail file writes."""

from __future__ import annotations

import json


def _make_report(tmp_path, n_removed=1):
    from psa.advertisement.decay import (
        DecayParams,
        DecayReport,
        RemovedPattern,
    )

    atlas_dir = tmp_path / "atlas_v1"
    atlas_dir.mkdir()
    removed = []
    if n_removed >= 1:
        removed = [
            RemovedPattern(
                anchor_id=1,
                pattern="stale pattern",
                source="manual",
                created_at="2020-01-01T00:00:00+00:00",
                last_reinforced_at=None,
                reason="stale_unreinforced",
            )
        ]
    return atlas_dir, DecayReport(
        tenant_id="default",
        atlas_version=1,
        created_at="2026-04-17T12:00:00+00:00",
        params=DecayParams(),
        origins={"interactive"},
        n_patterns_scanned=5,
        n_patterns_removed=n_removed,
        n_patterns_by_source_removed={"manual": n_removed} if n_removed else {},
        n_anchors_touched=1 if n_removed else 0,
        n_anchors_shielded=0,
        n_patterns_shielded=0,
        n_patterns_pinned_exempt=0,
        n_patterns_backfilled_this_run=0,
        pruning_by_reason={"stale_unreinforced": n_removed} if n_removed else {},
        removed_patterns=removed,
        shielded_anchors=[],
        new_cards=[{"anchor_id": 1, "generated_query_patterns": []}],
    )


def test_write_decay_candidate_writes_three_files(tmp_path):
    from psa.advertisement.writer import write_decay_candidate

    atlas_dir, report = _make_report(tmp_path)
    wrote = write_decay_candidate(str(atlas_dir), report)

    assert wrote is True
    assert (atlas_dir / "anchor_cards_candidate.json").exists()
    assert (atlas_dir / "anchor_cards_candidate.meta.json").exists()
    assert (atlas_dir / "anchor_cards_candidate.decay_report.json").exists()


def test_candidate_cards_has_refined_patterns(tmp_path):
    from psa.advertisement.writer import write_decay_candidate

    atlas_dir, report = _make_report(tmp_path)
    write_decay_candidate(str(atlas_dir), report)

    cards = json.loads((atlas_dir / "anchor_cards_candidate.json").read_text())
    assert cards[0]["generated_query_patterns"] == []  # patterns pruned


def test_meta_is_summary_no_removed_patterns(tmp_path):
    """anchor_cards_candidate.meta.json must NOT include removed_patterns
    (that belongs in the decay_report file, not the meta summary)."""
    from psa.advertisement.writer import write_decay_candidate

    atlas_dir, report = _make_report(tmp_path)
    write_decay_candidate(str(atlas_dir), report)

    meta = json.loads((atlas_dir / "anchor_cards_candidate.meta.json").read_text())
    assert "removed_patterns" not in meta
    assert meta["source"] == "decay"
    assert meta["n_patterns_removed"] == 1
    assert meta["pruning_by_reason"] == {"stale_unreinforced": 1}


def test_decay_report_has_full_detail(tmp_path):
    from psa.advertisement.writer import write_decay_candidate

    atlas_dir, report = _make_report(tmp_path)
    write_decay_candidate(str(atlas_dir), report)

    detail = json.loads((atlas_dir / "anchor_cards_candidate.decay_report.json").read_text())
    assert detail["tenant_id"] == "default"
    assert len(detail["removed_patterns"]) == 1
    assert detail["removed_patterns"][0]["pattern"] == "stale pattern"


def test_empty_run_skips_all_writes(tmp_path):
    """When n_patterns_removed == 0, no candidate files are written."""
    from psa.advertisement.writer import write_decay_candidate

    atlas_dir, report = _make_report(tmp_path, n_removed=0)
    wrote = write_decay_candidate(str(atlas_dir), report)

    assert wrote is False
    assert not (atlas_dir / "anchor_cards_candidate.json").exists()
    assert not (atlas_dir / "anchor_cards_candidate.meta.json").exists()
    assert not (atlas_dir / "anchor_cards_candidate.decay_report.json").exists()


def test_origins_serialized_sorted(tmp_path):
    """origins set becomes a sorted list in the meta for stable JSON output."""
    from psa.advertisement.writer import write_decay_candidate

    atlas_dir, report = _make_report(tmp_path)
    report.origins = {"benchmark", "interactive"}
    write_decay_candidate(str(atlas_dir), report)

    meta = json.loads((atlas_dir / "anchor_cards_candidate.meta.json").read_text())
    assert meta["origins"] == ["benchmark", "interactive"]


def test_stamp_refined_hash_with_existing_file(tmp_path):
    from psa.advertisement.writer import stamp_refined_hash

    atlas_dir = tmp_path / "atlas_v1"
    atlas_dir.mkdir()
    refined = atlas_dir / "anchor_cards_refined.json"
    refined.write_text('{"cards": []}')

    meta = {"source": "decay"}
    stamp_refined_hash(meta, atlas_dir)

    assert meta["refined_existed_at_generation"] is True
    assert meta["refined_path_at_generation"].endswith("anchor_cards_refined.json")
    assert meta["refined_hash_at_generation"].startswith("sha256:")
    # SHA-256 of '{"cards": []}' is deterministic
    assert len(meta["refined_hash_at_generation"]) == len("sha256:") + 64


def test_stamp_refined_hash_without_existing_file(tmp_path):
    from psa.advertisement.writer import stamp_refined_hash

    atlas_dir = tmp_path / "atlas_v1"
    atlas_dir.mkdir()

    meta = {"source": "decay"}
    stamp_refined_hash(meta, atlas_dir)

    assert meta["refined_existed_at_generation"] is False
    assert meta["refined_hash_at_generation"] is None
    assert meta["refined_path_at_generation"].endswith("anchor_cards_refined.json")
