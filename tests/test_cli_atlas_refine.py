"""Tests for `psa atlas refine` CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def _write_atlas(atlas_dir: Path, patterns: list[str]) -> None:
    """Write a minimal, valid atlas_vN directory."""
    atlas_dir.mkdir(parents=True, exist_ok=True)
    cards = [
        {
            "anchor_id": 1,
            "name": "anchor-1",
            "meaning": "test anchor",
            "memory_types": ["semantic"],
            "include_terms": [],
            "exclude_terms": [],
            "prototype_examples": [],
            "near_but_different": [],
            "centroid": [0.0] * 768,
            "memory_count": 1,
            "is_novelty": False,
            "status": "active",
            "metadata": {},
            "generated_query_patterns": patterns,
            "query_fingerprint": [],
        }
    ]
    (atlas_dir / "anchor_cards.json").write_text(json.dumps(cards))
    np.save(atlas_dir / "centroids.npy", np.zeros((1, 768), dtype=np.float32))
    (atlas_dir / "atlas_meta.json").write_text(
        json.dumps(
            {
                "version": 1,
                "tenant_id": "test",
                "stats": {
                    "n_learned": 1,
                    "n_novelty": 0,
                    "n_total": 1,
                    "coverage": 1.0,
                    "novelty_rate": 0.0,
                    "utilization_skew": 0.0,
                },
            }
        )
    )


def test_atlas_refine_writes_refined_cards_file(tmp_path, monkeypatch, capsys):
    """`psa atlas refine` writes anchor_cards_refined.json into the latest atlas dir."""
    from psa.cli import main

    # Redirect HOME so TenantManager / AtlasManager see our tmp tree
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original pattern"])

    miss_log = tmp_path / "misses.jsonl"
    miss_log.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "query": "how does the authentication token refresh flow work",
                "gold_anchor_ids": [1],
                "miss_reason": "scoring_rank",
            }
        )
        + "\n"
    )

    # Minimal memory.sqlite3 so MemoryStore() init in TenantManager does not fail
    tenant_dir.mkdir(parents=True, exist_ok=True)

    with patch("sys.argv", ["psa", "atlas", "refine", "--miss-log", str(miss_log)]):
        main()

    candidate_path = atlas_dir / "anchor_cards_candidate.json"
    assert candidate_path.exists(), "candidate cards file must be written"

    # Refined is NOT written by `refine`; only `promote-refinement` touches it.
    assert not (atlas_dir / "anchor_cards_refined.json").exists()

    refined_cards = json.loads(candidate_path.read_text())
    assert len(refined_cards) == 1
    patterns = refined_cards[0]["generated_query_patterns"]
    assert "original pattern" in patterns
    assert len(patterns) > 1


def test_atlas_refine_errors_when_no_atlas(tmp_path, monkeypatch, capsys):
    """`psa atlas refine` prints a clear error and exits non-zero when no atlas exists."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    miss_log = tmp_path / "misses.jsonl"
    miss_log.write_text("")

    with patch("sys.argv", ["psa", "atlas", "refine", "--miss-log", str(miss_log)]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code != 0
    out = capsys.readouterr().out
    assert "atlas" in out.lower()


def test_atlas_refine_errors_when_miss_log_missing(tmp_path, monkeypatch, capsys):
    """`psa atlas refine` errors when the provided miss log path does not exist."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original"])

    with patch(
        "sys.argv", ["psa", "atlas", "refine", "--miss-log", str(tmp_path / "does_not_exist.jsonl")]
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code != 0


def test_atlas_refine_skips_write_on_empty_miss_log(tmp_path, monkeypatch, capsys):
    """`psa atlas refine` warns and skips write when miss log has 0 valid entries."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original"])

    miss_log = tmp_path / "misses.jsonl"
    miss_log.write_text("\n\n  \n")  # blank lines only

    with patch("sys.argv", ["psa", "atlas", "refine", "--miss-log", str(miss_log)]):
        main()

    out = capsys.readouterr().out
    assert "0 valid entries" in out.lower() or "0 valid entries" in out
    assert not (atlas_dir / "anchor_cards_candidate.json").exists()
    assert not (atlas_dir / "anchor_cards_refined.json").exists()


def test_atlas_refine_writes_candidate_metadata(tmp_path, monkeypatch, capsys):
    """`psa atlas refine` writes anchor_cards_candidate.meta.json with provenance."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original"])

    miss_log = tmp_path / "misses.jsonl"
    miss_log.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "query": "how does token refresh work in the auth flow",
                "gold_anchor_ids": [1],
            }
        )
        + "\n"
    )

    with patch("sys.argv", ["psa", "atlas", "refine", "--miss-log", str(miss_log)]):
        main()

    meta_path = atlas_dir / "anchor_cards_candidate.meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["source"] == "manual"
    assert meta["tenant_id"] == "default"
    assert meta["atlas_version"] == 1
    assert meta["promoted"] is False
    assert meta["promoted_at"] is None
    assert meta["miss_log_path"] == str(miss_log)
    assert meta["n_anchors_touched"] >= 1
    assert meta["n_patterns_added"] >= 1
    assert "created_at" in meta and meta["created_at"]


def test_atlas_refine_records_source_from_flag(tmp_path, monkeypatch, capsys):
    """`--source benchmark` is recorded in the candidate metadata."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original"])

    miss_log = tmp_path / "misses.jsonl"
    miss_log.write_text(json.dumps({"query": "baking cookies", "gold_anchor_ids": [1]}) + "\n")

    with patch(
        "sys.argv",
        [
            "psa",
            "atlas",
            "refine",
            "--miss-log",
            str(miss_log),
            "--source",
            "benchmark",
        ],
    ):
        main()

    meta = json.loads((atlas_dir / "anchor_cards_candidate.meta.json").read_text())
    assert meta["source"] == "benchmark"


def test_promote_refinement_creates_refined_and_meta(tmp_path, monkeypatch, capsys):
    """promote-refinement copies candidate → refined and marks promoted=true."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original"])

    # First, refine to create a candidate
    miss_log = tmp_path / "misses.jsonl"
    miss_log.write_text(
        json.dumps({"query": "auth token refresh flow", "gold_anchor_ids": [1]}) + "\n"
    )
    with patch(
        "sys.argv",
        [
            "psa",
            "atlas",
            "refine",
            "--miss-log",
            str(miss_log),
            "--source",
            "benchmark",
        ],
    ):
        main()

    # Now promote
    with patch("sys.argv", ["psa", "atlas", "promote-refinement"]):
        main()

    refined_path = atlas_dir / "anchor_cards_refined.json"
    refined_meta_path = atlas_dir / "anchor_cards_refined.meta.json"
    assert refined_path.exists()
    assert refined_meta_path.exists()

    # Candidate remains in place (not deleted)
    assert (atlas_dir / "anchor_cards_candidate.json").exists()
    assert (atlas_dir / "anchor_cards_candidate.meta.json").exists()

    meta = json.loads(refined_meta_path.read_text())
    assert meta["promoted"] is True
    assert meta["promoted_at"] is not None
    assert meta["source"] == "benchmark"  # preserved from candidate
    out = capsys.readouterr().out
    assert "psa train --coactivation --force" in out, (
        "promote output must name the recalibration command"
    )


def test_promote_refinement_errors_when_no_candidate(tmp_path, monkeypatch, capsys):
    """promote-refinement with no candidate exits non-zero."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original"])

    with patch("sys.argv", ["psa", "atlas", "promote-refinement"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code != 0

    out = capsys.readouterr().out
    assert "candidate" in out.lower()


def test_promote_refinement_overwrites_previous_promotion(tmp_path, monkeypatch, capsys):
    """Running promote twice replaces the refined artifact cleanly."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original"])

    miss_log = tmp_path / "misses.jsonl"
    miss_log.write_text(json.dumps({"query": "sample query text", "gold_anchor_ids": [1]}) + "\n")
    with patch("sys.argv", ["psa", "atlas", "refine", "--miss-log", str(miss_log)]):
        main()

    with patch("sys.argv", ["psa", "atlas", "promote-refinement"]):
        main()
    first_meta = json.loads((atlas_dir / "anchor_cards_refined.meta.json").read_text())

    # Re-refine with different source, re-promote
    with patch(
        "sys.argv",
        [
            "psa",
            "atlas",
            "refine",
            "--miss-log",
            str(miss_log),
            "--source",
            "oracle",
        ],
    ):
        main()
    with patch("sys.argv", ["psa", "atlas", "promote-refinement"]):
        main()
    second_meta = json.loads((atlas_dir / "anchor_cards_refined.meta.json").read_text())

    assert second_meta["source"] == "oracle"
    assert second_meta["promoted_at"] != first_meta["promoted_at"]


def test_atlas_refine_stamps_refined_hash(tmp_path, monkeypatch, capsys):
    """Refine candidate meta must carry refined_hash_at_generation so
    promote-refinement can gate against stage 2 stale-candidate hazard."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original pattern"])

    # Seed an existing refined file so the stamp records a non-null hash.
    # Refine does NOT parse anchor_cards_refined.json — it only reads base
    # cards — so arbitrary JSON content is safe here; the stamp helper just
    # hashes the bytes.
    (atlas_dir / "anchor_cards_refined.json").write_text('{"v": 1}')

    miss_log = tmp_path / "misses.jsonl"
    miss_log.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "query": "how does the authentication token refresh flow work",
                "gold_anchor_ids": [1],
                "miss_reason": "scoring_rank",
            }
        )
        + "\n"
    )

    tenant_dir.mkdir(parents=True, exist_ok=True)

    with patch("sys.argv", ["psa", "atlas", "refine", "--miss-log", str(miss_log)]):
        main()

    meta_path = atlas_dir / "anchor_cards_candidate.meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["refined_existed_at_generation"] is True
    assert meta["refined_hash_at_generation"].startswith("sha256:")
    assert meta["refined_path_at_generation"].endswith("anchor_cards_refined.json")
