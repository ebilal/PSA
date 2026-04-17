"""Tests for psa.advertisement.metadata — pattern metadata storage."""

from __future__ import annotations

import json


def test_normalize_pattern_strips_and_lowercases():
    from psa.advertisement.metadata import normalize_pattern

    assert normalize_pattern("  HELLO World  ") == "hello world"


def test_normalize_pattern_collapses_whitespace():
    from psa.advertisement.metadata import normalize_pattern

    assert normalize_pattern("foo   bar\t baz") == "foo bar baz"


def test_metadata_key_format():
    from psa.advertisement.metadata import metadata_key

    assert metadata_key(227, "What Is This?") == "227::what is this?"


def test_load_metadata_missing_file_returns_empty_dict(tmp_path):
    from psa.advertisement.metadata import load_metadata

    assert load_metadata(str(tmp_path)) == {}


def test_save_and_load_roundtrip(tmp_path):
    from psa.advertisement.metadata import PatternMetadata, load_metadata, save_metadata

    meta = {
        "1::hello world": PatternMetadata(
            anchor_id=1,
            pattern="hello world",
            source="atlas_build",
            created_at="2026-04-17T00:00:00+00:00",
        )
    }
    save_metadata(str(tmp_path), meta)
    loaded = load_metadata(str(tmp_path))
    assert set(loaded.keys()) == {"1::hello world"}
    assert loaded["1::hello world"].anchor_id == 1
    assert loaded["1::hello world"].source == "atlas_build"
    assert loaded["1::hello world"].created_at == "2026-04-17T00:00:00+00:00"


def test_save_is_atomic_via_tmp_file(tmp_path):
    """The writer uses os.replace — no half-written state if interrupted."""
    from psa.advertisement.metadata import PatternMetadata, save_metadata

    meta = {
        "1::x": PatternMetadata(
            anchor_id=1, pattern="x", source="manual", created_at="2026-04-17T00:00:00+00:00"
        ),
    }
    save_metadata(str(tmp_path), meta)
    # After save, the tmp file should not remain.
    assert not (tmp_path / "pattern_metadata.json.tmp").exists()
    assert (tmp_path / "pattern_metadata.json").exists()


def test_load_tolerates_unknown_future_fields(tmp_path):
    """Extra keys in the JSON entries (e.g., reserved pinned, promoted_at) don't crash load."""
    from psa.advertisement.metadata import load_metadata

    raw = {
        "1::x": {
            "anchor_id": 1,
            "pattern": "x",
            "source": "manual",
            "created_at": "2026-04-17T00:00:00+00:00",
            "pinned": True,
            "promoted_at": "2026-04-18T00:00:00+00:00",
        }
    }
    (tmp_path / "pattern_metadata.json").write_text(json.dumps(raw))
    loaded = load_metadata(str(tmp_path))
    assert loaded["1::x"].pinned is True  # Supported in schema even if unused.


def test_backfill_unknown_stamps_new_patterns(tmp_path):
    """Patterns absent from metadata get source='unknown' + created_at=<now> entries.

    Does NOT mutate entries that already exist.
    """
    from psa.advertisement.metadata import (
        PatternMetadata,
        backfill_unknown,
        metadata_key,
    )

    existing = {
        metadata_key(1, "old pattern"): PatternMetadata(
            anchor_id=1,
            pattern="old pattern",
            source="manual",
            created_at="2020-01-01T00:00:00+00:00",
        )
    }
    # Anchor 1 now has two patterns; anchor 2 is new.
    patterns_by_anchor = {
        1: ["old pattern", "fresh pattern"],
        2: ["brand new"],
    }
    now_iso = "2026-04-17T12:00:00+00:00"
    n_backfilled = backfill_unknown(existing, patterns_by_anchor, now_iso)

    # Two new entries stamped; one pre-existing entry untouched.
    assert n_backfilled == 2
    assert existing[metadata_key(1, "old pattern")].created_at == "2020-01-01T00:00:00+00:00"
    assert existing[metadata_key(1, "fresh pattern")].source == "unknown"
    assert existing[metadata_key(1, "fresh pattern")].created_at == now_iso
    assert existing[metadata_key(2, "brand new")].source == "unknown"
