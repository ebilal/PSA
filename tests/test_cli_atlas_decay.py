"""End-to-end CLI tests for `psa atlas decay`."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _write_atlas(atlas_dir: Path, anchor_ids: list[int], patterns: dict[int, list[str]]) -> None:
    atlas_dir.mkdir(parents=True, exist_ok=True)
    cards = [
        {
            "anchor_id": aid,
            "name": f"anchor-{aid}",
            "meaning": "m",
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
            "generated_query_patterns": patterns.get(aid, []),
            "query_fingerprint": [],
        }
        for aid in anchor_ids
    ]
    (atlas_dir / "anchor_cards.json").write_text(json.dumps(cards))
    np.save(atlas_dir / "centroids.npy", np.zeros((len(anchor_ids), 768), dtype=np.float32))
    (atlas_dir / "atlas_meta.json").write_text(
        json.dumps(
            {
                "version": 1,
                "tenant_id": "test",
                "stats": {
                    "n_memories": len(anchor_ids),
                    "n_anchors_learned": len(anchor_ids),
                    "n_anchors_novelty": 0,
                    "mean_cluster_size": 1.0,
                    "min_cluster_size": 1,
                    "max_cluster_size": 1,
                    "stability_score": 1.0,
                    "built_at": "2026-04-17T00:00:00+00:00",
                },
            }
        )
    )


def test_cli_decay_dry_run_no_candidate_files(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [1], patterns={1: ["stale pattern"]})

    # Seed metadata: pattern is old and unreinforced.
    long_ago = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    (atlas_dir / "pattern_metadata.json").write_text(
        json.dumps(
            {
                "1::stale pattern": {
                    "anchor_id": 1,
                    "pattern": "stale pattern",
                    "source": "manual",
                    "created_at": long_ago,
                }
            }
        )
    )

    # Empty trace.
    (tenant_dir / "query_trace.jsonl").write_text("")

    with patch(
        "sys.argv",
        [
            "psa",
            "atlas",
            "decay",
            "--dry-run",
            "--min-anchor-activations",
            "0",
            "--low-activation-percentile",
            "0",
        ],
    ):
        main()

    assert not (atlas_dir / "anchor_cards_candidate.json").exists()
    assert not (atlas_dir / "anchor_cards_candidate.meta.json").exists()
    assert not (atlas_dir / "anchor_cards_candidate.decay_report.json").exists()


def test_cli_decay_real_writes_three_files(tmp_path, monkeypatch):
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [1], patterns={1: ["stale pattern"]})

    long_ago = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    (atlas_dir / "pattern_metadata.json").write_text(
        json.dumps(
            {
                "1::stale pattern": {
                    "anchor_id": 1,
                    "pattern": "stale pattern",
                    "source": "manual",
                    "created_at": long_ago,
                }
            }
        )
    )
    (tenant_dir / "query_trace.jsonl").write_text("")

    with patch(
        "sys.argv",
        [
            "psa",
            "atlas",
            "decay",
            "--min-anchor-activations",
            "0",
            "--low-activation-percentile",
            "0",
        ],
    ):
        main()

    assert (atlas_dir / "anchor_cards_candidate.json").exists()
    assert (atlas_dir / "anchor_cards_candidate.meta.json").exists()
    assert (atlas_dir / "anchor_cards_candidate.decay_report.json").exists()

    meta = json.loads((atlas_dir / "anchor_cards_candidate.meta.json").read_text())
    assert meta["source"] == "decay"
    assert meta["n_patterns_removed"] == 1


def test_cli_decay_json_mode_emits_envelope(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [1], patterns={1: ["stale"]})
    long_ago = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    (atlas_dir / "pattern_metadata.json").write_text(
        json.dumps(
            {
                "1::stale": {
                    "anchor_id": 1,
                    "pattern": "stale",
                    "source": "manual",
                    "created_at": long_ago,
                }
            }
        )
    )
    (tenant_dir / "query_trace.jsonl").write_text("")

    with patch(
        "sys.argv",
        [
            "psa",
            "atlas",
            "decay",
            "--dry-run",
            "--json",
            "--min-anchor-activations",
            "0",
            "--low-activation-percentile",
            "0",
        ],
    ):
        main()

    out = capsys.readouterr().out
    envelope = json.loads(out)
    assert envelope["source"] == "decay"
    assert envelope["n_patterns_removed"] == 1
    # Dry-run JSON embeds the detail as well.
    assert "removed_patterns" in envelope


def test_cli_decay_backfills_on_dry_run(tmp_path, monkeypatch):
    """Dry-run writes no candidate files, but DOES persist metadata backfill."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [1], patterns={1: ["unstamped"]})
    # No pre-existing pattern_metadata.json.
    (tenant_dir / "query_trace.jsonl").write_text("")

    with patch("sys.argv", ["psa", "atlas", "decay", "--dry-run"]):
        main()

    assert (atlas_dir / "pattern_metadata.json").exists()
    meta = json.loads((atlas_dir / "pattern_metadata.json").read_text())
    assert "1::unstamped" in meta
    assert meta["1::unstamped"]["source"] == "unknown"
