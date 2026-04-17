"""Tests for psa.curation.curator — orchestrator with empty-run guard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_atlas_dir(
    atlas_dir: Path,
    anchor_ids: list[int],
    patterns: dict[int, list[str]],
) -> None:
    """Write a minimal valid atlas_vN directory that AnchorIndex.load can read."""
    atlas_dir.mkdir(parents=True, exist_ok=True)
    cards = [
        {
            "anchor_id": aid,
            "name": f"anchor-{aid}",
            "meaning": "test",
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
                    "built_at": "2026-04-16T00:00:00+00:00",
                },
            }
        )
    )


def _write_oracle_labels(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_fingerprints(atlas_dir: Path, data: dict[int, list[str]]) -> None:
    with open(atlas_dir / "fingerprints.json", "w") as f:
        json.dump({str(k): v for k, v in data.items()}, f)


def test_curate_happy_path_writes_candidate_and_meta(tmp_path, monkeypatch):
    from psa.curation.curator import curate

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas_dir(atlas_dir, [1, 2], patterns={1: [], 2: []})

    labels_path = tenant_dir / "training" / "oracle_labels.jsonl"
    _write_oracle_labels(
        labels_path,
        [
            {"query": "how does the token refresh flow work", "winning_oracle_set": [1]},
            {"query": "what is the user session expiry", "winning_oracle_set": [2]},
        ],
    )
    _write_fingerprints(atlas_dir, {1: ["how does the token refresh flow work"], 2: []})

    summary = curate(tenant_id="default", extractor_name="heuristic")

    cand_path = atlas_dir / "anchor_cards_candidate.json"
    meta_path = atlas_dir / "anchor_cards_candidate.meta.json"
    assert cand_path.exists()
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text())
    assert meta["source"] == "production_signal"
    assert meta["extractor"] == "heuristic"
    assert meta["support_semantics"] == "anchor_level_oracle_endorsement"
    assert meta["promoted"] is False
    assert meta["oracle_labels_read"] == 2
    assert meta["n_anchors_with_oracle_support"] == 2
    assert meta["n_anchors_touched"] >= 1
    assert meta["n_patterns_added"] >= 1

    # Summary mirrors what was written.
    assert summary["n_anchors_touched"] == meta["n_anchors_touched"]
    assert summary["n_patterns_added"] == meta["n_patterns_added"]


def test_curate_skips_write_when_no_oracle_labels(tmp_path, monkeypatch, capsys):
    """Empty-run guard: no labels → no candidate written, no in-flight clobber."""
    from psa.curation.curator import curate

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas_dir(atlas_dir, [1], patterns={1: []})

    # Deliberately: no oracle_labels.jsonl and no fingerprints.json.

    # Simulate an in-flight miss-log candidate already sitting in the atlas dir.
    (atlas_dir / "anchor_cards_candidate.json").write_text('[{"in_flight": true}]')
    (atlas_dir / "anchor_cards_candidate.meta.json").write_text('{"source": "manual"}')

    summary = curate(tenant_id="default", extractor_name="heuristic")

    # In-flight candidate is untouched.
    assert json.loads((atlas_dir / "anchor_cards_candidate.json").read_text()) == [
        {"in_flight": True}
    ]
    assert json.loads((atlas_dir / "anchor_cards_candidate.meta.json").read_text()) == {
        "source": "manual"
    }

    assert summary["wrote_candidate"] is False
    assert summary["n_anchors_touched"] == 0
    assert summary["n_patterns_added"] == 0


def test_curate_skips_write_when_all_patterns_duplicate(tmp_path, monkeypatch):
    """Empty-run guard fires when oracle labels exist but ngrams all duplicate existing patterns."""
    from psa.curation.curator import curate

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"

    # Pre-seed generated_query_patterns with ALL ngrams the extractor would
    # produce from the oracle query below, so extractor output fully
    # duplicates. Compute seeded list by calling extract_ngrams directly.
    from psa.curation.ngrams import extract_ngrams

    query_text = "how does the token refresh flow work"
    seeded = extract_ngrams(query_text)

    _write_atlas_dir(atlas_dir, [1], patterns={1: seeded})

    labels_path = tenant_dir / "training" / "oracle_labels.jsonl"
    _write_oracle_labels(
        labels_path,
        [
            {"query": query_text, "winning_oracle_set": [1]},
        ],
    )

    summary = curate(tenant_id="default", extractor_name="heuristic")

    # No candidate file created; all extractor output duplicated existing patterns.
    assert not (atlas_dir / "anchor_cards_candidate.json").exists()
    assert not (atlas_dir / "anchor_cards_candidate.meta.json").exists()
    assert summary["wrote_candidate"] is False


def test_curate_rejects_unknown_extractor(tmp_path, monkeypatch):
    from psa.curation.curator import curate

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas_dir(atlas_dir, [1], patterns={1: []})

    import pytest

    with pytest.raises(ValueError, match="extractor"):
        curate(tenant_id="default", extractor_name="bogus")


def test_curate_llm_extractor_raises_notimplemented(tmp_path, monkeypatch):
    from psa.curation.curator import curate

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas_dir(atlas_dir, [1], patterns={1: []})

    labels_path = tenant_dir / "training" / "oracle_labels.jsonl"
    _write_oracle_labels(
        labels_path,
        [
            {"query": "a query", "winning_oracle_set": [1]},
        ],
    )

    import pytest

    with pytest.raises(NotImplementedError):
        curate(tenant_id="default", extractor_name="llm")
