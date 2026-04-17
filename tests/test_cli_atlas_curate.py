"""End-to-end CLI test for `psa atlas curate`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def _write_atlas(atlas_dir: Path, anchor_ids: list[int], patterns: dict[int, list[str]]) -> None:
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
    # Match the real AtlasStats dataclass field names (see psa/atlas.py).
    (atlas_dir / "atlas_meta.json").write_text(
        json.dumps({
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
        })
    )


def test_cli_atlas_curate_writes_candidate(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [1], patterns={1: []})

    labels = tenant_dir / "training" / "oracle_labels.jsonl"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text(
        json.dumps({"query": "how does the auth token refresh flow work", "winning_oracle_set": [1]}) + "\n"
    )
    with open(atlas_dir / "fingerprints.json", "w") as f:
        json.dump({"1": ["how does the auth token refresh flow work"]}, f)

    with patch("sys.argv", ["psa", "atlas", "curate"]):
        main()

    cand = atlas_dir / "anchor_cards_candidate.json"
    meta = atlas_dir / "anchor_cards_candidate.meta.json"
    assert cand.exists()
    meta_obj = json.loads(meta.read_text())
    assert meta_obj["source"] == "production_signal"
    assert meta_obj["extractor"] == "heuristic"
    assert meta_obj["support_semantics"] == "anchor_level_oracle_endorsement"


def test_cli_atlas_curate_llm_flag_errors_in_mvp(tmp_path, monkeypatch, capsys):
    """Passing --extractor llm in MVP exits non-zero with a pointer to the stub file."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [1], patterns={1: []})

    labels = tenant_dir / "training" / "oracle_labels.jsonl"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text(json.dumps({"query": "q", "winning_oracle_set": [1]}) + "\n")

    with patch("sys.argv", ["psa", "atlas", "curate", "--extractor", "llm"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code != 0
    out = capsys.readouterr().out
    assert "extractor_llm.py" in out


def test_cli_atlas_curate_empty_run_skips_write(tmp_path, monkeypatch, capsys):
    """No oracle labels → no candidate written; existing in-flight candidate preserved."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [1], patterns={1: []})

    # In-flight miss-log candidate sitting there.
    (atlas_dir / "anchor_cards_candidate.json").write_text('[{"sentinel": true}]')
    (atlas_dir / "anchor_cards_candidate.meta.json").write_text('{"source": "manual"}')

    with patch("sys.argv", ["psa", "atlas", "curate"]):
        main()

    # Untouched.
    assert json.loads((atlas_dir / "anchor_cards_candidate.json").read_text()) == [
        {"sentinel": True}
    ]
    out = capsys.readouterr().out
    assert "skip" in out.lower() or "no oracle" in out.lower()
