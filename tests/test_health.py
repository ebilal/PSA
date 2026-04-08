"""Tests for psa.health — AtlasHealthMonitor and HealthReport."""

from unittest.mock import MagicMock, patch

import pytest

from psa.anchor import AnchorCard
from psa.atlas import Atlas
from psa.health import (
    NOVELTY_RATE_THRESHOLD,
    UTILIZATION_SKEW_THRESHOLD,
    AtlasHealthMonitor,
    HealthReport,
    _utilization_skew,
)
from psa.memory_object import MemoryStore


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_card(anchor_id: int, is_novelty: bool = False, name: str = "anchor") -> AnchorCard:
    return AnchorCard(
        anchor_id=anchor_id,
        name=f"{name}_{anchor_id}",
        meaning=f"Anchor {anchor_id}",
        memory_types=["SEMANTIC"],
        include_terms=[],
        exclude_terms=[],
        prototype_examples=[],
        near_but_different=[],
        centroid=[0.0] * 4,
        memory_count=0,
        is_novelty=is_novelty,
    )


def _make_atlas(cards) -> Atlas:
    atlas = MagicMock(spec=Atlas)
    atlas.version = 1
    atlas.cards = cards
    return atlas


def _make_store(memories_per_anchor: dict) -> MemoryStore:
    """Mock store with count_by_anchor and count_by_type for health checks."""
    store = MagicMock(spec=MemoryStore)
    store.count_by_anchor.return_value = memories_per_anchor
    store.count_by_type.return_value = {}
    # Keep legacy mocks for any code that still uses them
    store.query_by_anchor.side_effect = lambda tenant_id, anchor_id, limit: (
        [MagicMock()] * memories_per_anchor.get(anchor_id, 0)
    )
    store.query_by_type.return_value = []
    return store


# ── _utilization_skew ─────────────────────────────────────────────────────────


def test_utilization_skew_uniform():
    counts = [10, 10, 10, 10]
    assert _utilization_skew(counts) == pytest.approx(1.0)


def test_utilization_skew_heavy_skew():
    counts = [1, 1, 1, 100]
    skew = _utilization_skew(counts)
    assert skew > 3.0


def test_utilization_skew_single():
    assert _utilization_skew([5]) == 1.0


def test_utilization_skew_empty():
    assert _utilization_skew([]) == 1.0


def test_utilization_skew_all_zero():
    assert _utilization_skew([0, 0, 0]) == 1.0


def test_utilization_skew_formula():
    # counts = [2, 4, 6, 8] → median = (4+6)/2 = 5, max = 8 → skew = 8/5 = 1.6
    counts = [2, 4, 6, 8]
    assert _utilization_skew(counts) == pytest.approx(8.0 / 5.0)


# ── HealthReport.summary / to_dict ────────────────────────────────────────────


def test_health_report_summary_healthy():
    report = HealthReport(
        tenant_id="t1",
        atlas_version=1,
        total_memories=100,
        total_anchors=12,
        novelty_anchors=2,
        novelty_rate=0.04,
        utilization_skew=1.5,
        memory_type_distribution={"EPISODIC": 50, "SEMANTIC": 50},
        anchor_stats=[],
        embedding_drift=None,
        should_rebuild=False,
        rebuild_reasons=[],
    )
    s = report.summary()
    assert "healthy" in s
    assert "REBUILD" not in s


def test_health_report_summary_rebuild():
    report = HealthReport(
        tenant_id="t1",
        atlas_version=1,
        total_memories=100,
        total_anchors=12,
        novelty_anchors=2,
        novelty_rate=0.12,
        utilization_skew=5.0,
        memory_type_distribution={},
        anchor_stats=[],
        embedding_drift=None,
        should_rebuild=True,
        rebuild_reasons=["novelty_rate 12.0% > 8%"],
    )
    s = report.summary()
    assert "REBUILD" in s


def test_health_report_to_dict():
    report = HealthReport(
        tenant_id="t1",
        atlas_version=2,
        total_memories=50,
        total_anchors=8,
        novelty_anchors=1,
        novelty_rate=0.02,
        utilization_skew=1.2,
        memory_type_distribution={"SEMANTIC": 50},
        anchor_stats=[],
        embedding_drift=0.05,
        should_rebuild=False,
        rebuild_reasons=[],
    )
    d = report.to_dict()
    assert d["atlas_version"] == 2
    assert d["novelty_rate"] == pytest.approx(0.02)
    assert d["embedding_drift"] == pytest.approx(0.05)
    assert d["should_rebuild"] is False


# ── AtlasHealthMonitor.check_health ───────────────────────────────────────────


def test_check_health_healthy():
    # 10 learned + 2 novelty anchors; novelty each has 1 memory, learned each has 10
    # novelty_rate = 2/102 ≈ 2% (well below 8% threshold)
    cards = [_make_card(i, is_novelty=(i >= 10)) for i in range(12)]
    atlas = _make_atlas(cards)
    counts = {c.anchor_id: (1 if c.is_novelty else 10) for c in cards}
    store = _make_store(counts)

    monitor = AtlasHealthMonitor()
    report = monitor.check_health(atlas, store, tenant_id="default")

    assert report.total_anchors == 12
    assert report.novelty_anchors == 2
    assert report.novelty_rate == pytest.approx(2 / 102)
    assert report.should_rebuild is False


def test_check_health_high_novelty():
    # 8 learned (10 each) + 4 novelty (40 each) → novelty_rate = 160/240 > 0.08
    cards = [_make_card(i, is_novelty=(i >= 8)) for i in range(12)]
    atlas = _make_atlas(cards)
    counts = {c.anchor_id: (40 if c.is_novelty else 10) for c in cards}
    store = _make_store(counts)

    monitor = AtlasHealthMonitor()
    report = monitor.check_health(atlas, store, tenant_id="default")

    assert report.novelty_rate > NOVELTY_RATE_THRESHOLD
    assert report.should_rebuild is True
    assert any("novelty_rate" in r for r in report.rebuild_reasons)


def test_check_health_high_skew():
    # 8 learned anchors: one has 100 memories, others have 1 each → skew > 3
    cards = [_make_card(i) for i in range(8)]
    atlas = _make_atlas(cards)
    counts = {cards[0].anchor_id: 100}
    for c in cards[1:]:
        counts[c.anchor_id] = 1
    store = _make_store(counts)

    monitor = AtlasHealthMonitor()
    report = monitor.check_health(atlas, store, tenant_id="default")

    assert report.utilization_skew > UTILIZATION_SKEW_THRESHOLD
    assert report.should_rebuild is True
    assert any("utilization_skew" in r for r in report.rebuild_reasons)


def test_check_health_empty_atlas():
    atlas = _make_atlas([])
    store = _make_store({})

    monitor = AtlasHealthMonitor()
    report = monitor.check_health(atlas, store, tenant_id="t")

    assert report.total_memories == 0
    assert report.total_anchors == 0
    assert report.should_rebuild is False


def test_check_health_anchor_stats():
    cards = [_make_card(i) for i in range(3)]
    atlas = _make_atlas(cards)
    store = _make_store({0: 5, 1: 10, 2: 3})

    monitor = AtlasHealthMonitor()
    report = monitor.check_health(atlas, store, tenant_id="t")

    assert len(report.anchor_stats) == 3
    counts = {s.anchor_id: s.memory_count for s in report.anchor_stats}
    assert counts[0] == 5
    assert counts[1] == 10
    assert counts[2] == 3
