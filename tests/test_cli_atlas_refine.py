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

    refined_path = atlas_dir / "anchor_cards_refined.json"
    assert refined_path.exists(), "refined cards file must be written"

    refined_cards = json.loads(refined_path.read_text())
    assert len(refined_cards) == 1
    # Original pattern preserved, at least one new pattern from the miss log added
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
    assert not (atlas_dir / "anchor_cards_refined.json").exists()
