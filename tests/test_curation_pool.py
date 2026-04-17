"""Tests for psa.curation.pool — anchor-level oracle-endorsement filter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from psa.curation.pool import Pool, build_pool


def _write_oracle_labels(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _fake_atlas(anchor_ids: list[int]) -> MagicMock:
    atlas = MagicMock()
    cards = []
    for aid in anchor_ids:
        c = MagicMock()
        c.anchor_id = aid
        cards.append(c)
    atlas.cards = cards
    atlas.fingerprint_store = MagicMock()
    return atlas


def test_pool_is_a_dataclass_with_two_list_fields():
    p = Pool(oracle_queries=["a", "b"], endorsed_fingerprint_queries=["c"])
    assert p.oracle_queries == ["a", "b"]
    assert p.endorsed_fingerprint_queries == ["c"]


def test_build_pool_collects_oracle_queries_per_winning_anchor(tmp_path):
    atlas = _fake_atlas([1, 2])
    atlas.fingerprint_store.get.return_value = []
    labels_path = tmp_path / "oracle_labels.jsonl"
    _write_oracle_labels(
        labels_path,
        [
            {"query": "q1", "winning_oracle_set": [1, 2]},
            {"query": "q2", "winning_oracle_set": [2]},
        ],
    )

    pools = build_pool(atlas, str(labels_path))
    assert pools[1].oracle_queries == ["q1"]
    assert sorted(pools[2].oracle_queries) == ["q1", "q2"]


def test_build_pool_includes_fingerprints_only_when_anchor_is_oracle_endorsed(tmp_path):
    """Anchor-level rule: fingerprints flow in only if the anchor appears in ≥1 winning_oracle_set."""
    atlas = _fake_atlas([1, 2])

    # Anchor 1 has one oracle endorsement; anchor 2 has zero.
    labels_path = tmp_path / "oracle_labels.jsonl"
    _write_oracle_labels(
        labels_path,
        [
            {"query": "oracle_q", "winning_oracle_set": [1]},
        ],
    )

    # Fingerprints exist for BOTH anchors.
    def _get(aid):
        return {1: ["fp_for_1"], 2: ["fp_for_2"]}[aid]

    atlas.fingerprint_store.get.side_effect = _get

    pools = build_pool(atlas, str(labels_path))
    assert pools[1].endorsed_fingerprint_queries == ["fp_for_1"]
    assert pools[2].endorsed_fingerprint_queries == []
    assert pools[2].oracle_queries == []


def test_build_pool_empty_when_no_signal(tmp_path):
    """No oracle labels, no fingerprints → every anchor has an empty Pool."""
    atlas = _fake_atlas([1, 2])
    atlas.fingerprint_store.get.return_value = []

    labels_path = tmp_path / "oracle_labels.jsonl"
    _write_oracle_labels(labels_path, [])

    pools = build_pool(atlas, str(labels_path))
    for aid in [1, 2]:
        assert pools[aid].oracle_queries == []
        assert pools[aid].endorsed_fingerprint_queries == []


def test_build_pool_returns_entry_for_every_atlas_anchor(tmp_path):
    """Even anchors with no signal get a Pool key (with empty lists)."""
    atlas = _fake_atlas([1, 2, 3])
    atlas.fingerprint_store.get.return_value = []
    labels_path = tmp_path / "oracle_labels.jsonl"
    _write_oracle_labels(labels_path, [{"query": "q", "winning_oracle_set": [1]}])

    pools = build_pool(atlas, str(labels_path))
    assert set(pools.keys()) == {1, 2, 3}


def test_build_pool_missing_labels_file_returns_empty_pools(tmp_path):
    """Soft: missing oracle_labels.jsonl → every anchor's Pool is empty."""
    atlas = _fake_atlas([1])
    atlas.fingerprint_store.get.return_value = ["fp_q"]

    pools = build_pool(atlas, str(tmp_path / "does_not_exist.jsonl"))
    assert pools[1].oracle_queries == []
    # Fingerprints require oracle endorsement; without any oracle labels at
    # all, the anchor is NOT endorsed, so fingerprints don't flow in.
    assert pools[1].endorsed_fingerprint_queries == []
