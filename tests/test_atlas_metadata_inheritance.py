"""Tests for atlas rebuild carrying pattern_metadata.json forward."""

from __future__ import annotations

import json
from pathlib import Path


def _seed_metadata(atlas_dir: Path, entries: list[dict]) -> None:
    atlas_dir.mkdir(parents=True, exist_ok=True)
    raw = {f"{e['anchor_id']}::{e['pattern']}": e for e in entries}
    (atlas_dir / "pattern_metadata.json").write_text(json.dumps(raw))


def test_inherit_matched_anchor_and_pattern_copies_metadata(tmp_path):
    """If new atlas has the same (anchor_id, normalized pattern) as old, the
    entry carries forward verbatim."""
    from psa.atlas import inherit_pattern_metadata

    old_dir = tmp_path / "atlas_v1"
    new_dir = tmp_path / "atlas_v2"
    new_dir.mkdir()

    _seed_metadata(old_dir, [
        {
            "anchor_id": 1,
            "pattern": "old pattern",
            "source": "manual",
            "created_at": "2020-01-01T00:00:00+00:00",
        }
    ])

    # New atlas has the same pattern on the same anchor.
    new_cards = [
        {"anchor_id": 1, "generated_query_patterns": ["Old Pattern"]},  # case drift intentional
    ]
    inherit_pattern_metadata(str(old_dir), str(new_dir), new_cards)

    new_meta = json.loads((new_dir / "pattern_metadata.json").read_text())
    assert "1::old pattern" in new_meta
    assert new_meta["1::old pattern"]["created_at"] == "2020-01-01T00:00:00+00:00"


def test_inherit_drops_orphans(tmp_path):
    """Old metadata entries without a matching new pattern are NOT copied."""
    from psa.atlas import inherit_pattern_metadata

    old_dir = tmp_path / "atlas_v1"
    new_dir = tmp_path / "atlas_v2"
    new_dir.mkdir()

    _seed_metadata(old_dir, [
        {"anchor_id": 1, "pattern": "retired pattern", "source": "manual",
         "created_at": "2020-01-01T00:00:00+00:00"},
    ])
    # New atlas has a different pattern for anchor 1.
    new_cards = [
        {"anchor_id": 1, "generated_query_patterns": ["fresh pattern"]},
    ]
    inherit_pattern_metadata(str(old_dir), str(new_dir), new_cards)

    new_meta = json.loads((new_dir / "pattern_metadata.json").read_text())
    assert "1::retired pattern" not in new_meta
    assert "1::fresh pattern" not in new_meta  # new patterns wait for backfill


def test_inherit_skips_missing_old_metadata(tmp_path):
    """If old atlas had no pattern_metadata.json, new file is empty (or absent)."""
    from psa.atlas import inherit_pattern_metadata

    old_dir = tmp_path / "atlas_v1"
    old_dir.mkdir()
    new_dir = tmp_path / "atlas_v2"
    new_dir.mkdir()

    new_cards = [
        {"anchor_id": 1, "generated_query_patterns": ["any"]},
    ]
    # Should not raise.
    inherit_pattern_metadata(str(old_dir), str(new_dir), new_cards)

    path = new_dir / "pattern_metadata.json"
    if path.exists():
        assert json.loads(path.read_text()) == {}


def test_inherit_preserves_pinned_flag(tmp_path):
    from psa.atlas import inherit_pattern_metadata

    old_dir = tmp_path / "atlas_v1"
    new_dir = tmp_path / "atlas_v2"
    new_dir.mkdir()

    _seed_metadata(old_dir, [
        {
            "anchor_id": 1,
            "pattern": "pinned one",
            "source": "manual",
            "created_at": "2020-01-01T00:00:00+00:00",
            "pinned": True,
        }
    ])
    new_cards = [
        {"anchor_id": 1, "generated_query_patterns": ["pinned one"]},
    ]
    inherit_pattern_metadata(str(old_dir), str(new_dir), new_cards)

    new_meta = json.loads((new_dir / "pattern_metadata.json").read_text())
    assert new_meta["1::pinned one"]["pinned"] is True
