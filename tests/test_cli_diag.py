"""End-to-end CLI tests for `psa diag {activation,advertisement,misses}`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _write_atlas(atlas_dir: Path, anchor_specs: list[tuple[int, int]]) -> None:
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
            "memory_count": mc,
            "is_novelty": False,
            "status": "active",
            "metadata": {},
            "generated_query_patterns": [],
            "query_fingerprint": [],
        }
        for aid, mc in anchor_specs
    ]
    (atlas_dir / "anchor_cards.json").write_text(json.dumps(cards))
    np.save(atlas_dir / "centroids.npy", np.zeros((len(anchor_specs), 768), dtype=np.float32))
    (atlas_dir / "atlas_meta.json").write_text(
        json.dumps(
            {
                "version": 1,
                "tenant_id": "test",
                "stats": {
                    "n_memories": sum(mc for _, mc in anchor_specs),
                    "n_anchors_learned": len(anchor_specs),
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


def _write_trace(tenant_dir: Path, records: list[dict]) -> None:
    path = tenant_dir / "query_trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_cli_diag_activation_json_envelope(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [(1, 5), (2, 3)])
    _write_trace(
        tenant_dir,
        [
            {
                "query_origin": "interactive",
                "selected_anchor_ids": [1],
                "packed_memories": [{"memory_id": "m", "source_anchor_id": 1}],
            },
        ],
    )

    with patch("sys.argv", ["psa", "diag", "activation", "--json"]):
        main()

    out = capsys.readouterr().out
    envelope = json.loads(out)
    assert envelope["tenant_id"] == "default"
    assert envelope["atlas_version"] == 1
    assert envelope["trace_records"] == 1
    assert envelope["origins"] == ["interactive"]
    assert isinstance(envelope["rows"], list)
    assert any(row["anchor_id"] == 1 for row in envelope["rows"])


def test_cli_diag_advertisement_default_sort_is_gap_desc(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [(1, 100), (2, 1)])
    _write_trace(
        tenant_dir,
        [
            {"query_origin": "interactive", "selected_anchor_ids": [2]},
        ],
    )

    with patch("sys.argv", ["psa", "diag", "advertisement", "--json"]):
        main()

    envelope = json.loads(capsys.readouterr().out)
    rows = envelope["rows"]
    # Anchor 1 has high memory, never activated → largest positive gap → top row.
    assert rows[0]["anchor_id"] == 1


def test_cli_diag_misses_includes_empty_rate_and_near_misses(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [(1, 5)])
    _write_trace(
        tenant_dir,
        [
            {
                "query_origin": "interactive",
                "result_kind": "empty_selection",
                "top_anchor_scores": [{"anchor_id": 1, "score": 0.9, "selected": False, "rank": 1}],
            },
            {
                "query_origin": "interactive",
                "result_kind": "synthesized",
                "top_anchor_scores": [{"anchor_id": 1, "score": 3.0, "selected": True, "rank": 1}],
                "selected_anchor_ids": [1],
            },
        ],
    )

    with patch("sys.argv", ["psa", "diag", "misses", "--json"]):
        main()

    envelope = json.loads(capsys.readouterr().out)
    assert envelope["total_queries"] == 2
    assert envelope["empty_queries"] == 1
    assert abs(envelope["empty_rate"] - 0.5) < 1e-6
    # Anchor 1 appeared at rank 1 in the empty record → near-miss.
    assert any(nm["anchor_id"] == 1 for nm in envelope["rows"])


def test_cli_diag_default_origins_excludes_benchmark(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [(1, 1)])
    _write_trace(
        tenant_dir,
        [
            {
                "query_origin": "interactive",
                "selected_anchor_ids": [1],
                "packed_memories": [{"memory_id": "m", "source_anchor_id": 1}],
            },
            {
                "query_origin": "benchmark",
                "selected_anchor_ids": [1],
                "packed_memories": [{"memory_id": "m", "source_anchor_id": 1}],
            },
        ],
    )

    with patch("sys.argv", ["psa", "diag", "activation", "--json"]):
        main()

    envelope = json.loads(capsys.readouterr().out)
    row_1 = next(r for r in envelope["rows"] if r["anchor_id"] == 1)
    assert row_1["n_selected"] == 1  # benchmark excluded by default


def test_cli_diag_include_origin_widens_filter(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [(1, 1)])
    _write_trace(
        tenant_dir,
        [
            {
                "query_origin": "interactive",
                "selected_anchor_ids": [1],
                "packed_memories": [{"memory_id": "m", "source_anchor_id": 1}],
            },
            {
                "query_origin": "benchmark",
                "selected_anchor_ids": [1],
                "packed_memories": [{"memory_id": "m", "source_anchor_id": 1}],
            },
        ],
    )

    with patch(
        "sys.argv", ["psa", "diag", "activation", "--include-origin", "benchmark", "--json"]
    ):
        main()

    envelope = json.loads(capsys.readouterr().out)
    row_1 = next(r for r in envelope["rows"] if r["anchor_id"] == 1)
    assert row_1["n_selected"] == 2  # both records counted
