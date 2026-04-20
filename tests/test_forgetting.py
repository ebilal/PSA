"""Tests for psa.forgetting: score behavior and pruning order."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from psa.forgetting import forgetting_score, low_usage_pressure, prune_anchor
from psa.memory_object import MemoryObject, MemoryStore, MemoryType


def make_memory(
    *,
    pack_count: int = 0,
    select_count: int = 0,
    quality_score: float = 0.5,
    created_at: Optional[datetime] = None,
    last_packed: Optional[datetime] = None,
    primary_anchor_id: int = 1,
    tenant_id: str = "t",
) -> MemoryObject:
    """Build a MemoryObject with just the fields the forgetting system reads."""
    mo = MemoryObject.create(
        tenant_id=tenant_id,
        memory_type=MemoryType.SEMANTIC,
        title="t",
        body="b",
        summary="s",
        source_ids=[],
        classification_reason="",
        pack_count=pack_count,
        select_count=select_count,
        quality_score=quality_score,
        primary_anchor_id=primary_anchor_id,
    )
    if created_at is not None:
        mo.created_at = created_at.isoformat()
    if last_packed is not None:
        mo.last_packed = last_packed.isoformat()
    return mo


def now_utc() -> datetime:
    return datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)


def old(days: int) -> datetime:
    return now_utc() - timedelta(days=days)


def _seed_anchor(store, tenant, anchor_id, *, pack_counts):
    import numpy as np
    ids = []
    for pc in pack_counts:
        m = make_memory(
            pack_count=pc,
            quality_score=0.5,
            created_at=old(10),
            primary_anchor_id=anchor_id,
            tenant_id=tenant,
        )
        m.embedding = np.zeros(8, dtype="float32").tolist()
        store.add(m)
        ids.append(m.memory_object_id)
    return ids


def test_fixture_builds_memory():
    m = make_memory(pack_count=3, created_at=old(10))
    assert m.pack_count == 3
    assert m.primary_anchor_id == 1


def test_age_alone_does_not_raise_score():
    """Two memories identical except created_at must score identically."""
    young = make_memory(pack_count=0, quality_score=0.5, created_at=old(30))
    ancient = make_memory(pack_count=0, quality_score=0.5, created_at=old(365))

    s_young = forgetting_score(young, anchor_size=10, now=now_utc(), usage_pressure=0.5)
    s_ancient = forgetting_score(ancient, anchor_size=10, now=now_utc(), usage_pressure=0.5)

    assert s_young == pytest.approx(s_ancient)


def test_low_usage_pressure_small_anchor_returns_zero():
    peers = [make_memory(pack_count=i) for i in range(4)]  # N=4 < 5
    assert low_usage_pressure(peers[0], peers) == 0.0
    assert low_usage_pressure(peers[-1], peers) == 0.0


def test_low_usage_pressure_all_zero_returns_zero():
    peers = [make_memory(pack_count=0, select_count=0) for _ in range(10)]
    for p in peers:
        assert low_usage_pressure(p, peers) == 0.0


def test_low_usage_pressure_ranks_bottom_highest():
    peers = [make_memory(pack_count=i) for i in range(10)]  # 0..9
    bottom = peers[0]
    top = peers[-1]
    p_bottom = low_usage_pressure(bottom, peers)
    p_top = low_usage_pressure(top, peers)
    assert p_bottom == pytest.approx(1.0)
    assert p_top == pytest.approx(0.0)
    assert p_bottom > p_top


def test_low_usage_pressure_monotonic_with_rank():
    peers = [make_memory(pack_count=i) for i in range(10)]
    pressures = [low_usage_pressure(p, peers) for p in peers]
    assert pressures == sorted(pressures, reverse=True)


def test_low_usage_pressure_tiebreak_by_quality_then_age():
    """Identical pack/select: older (older created_at) sorts below → higher pressure."""
    older = make_memory(pack_count=5, quality_score=0.3, created_at=old(100))
    newer = make_memory(pack_count=5, quality_score=0.3, created_at=old(1))
    filler = [make_memory(pack_count=i + 10) for i in range(8)]
    peers = [older, newer] + filler
    assert low_usage_pressure(older, peers) > low_usage_pressure(newer, peers)


def test_grace_period_preserved():
    fresh = make_memory(created_at=now_utc() - timedelta(hours=1))
    assert forgetting_score(fresh, anchor_size=1, now=now_utc()) == -10.0


def test_heavy_usage_reduces_score():
    light = make_memory(pack_count=0, created_at=old(10))
    heavy = make_memory(pack_count=50, created_at=old(10))
    s_light = forgetting_score(light, anchor_size=10, now=now_utc(), usage_pressure=1.0)
    s_heavy = forgetting_score(heavy, anchor_size=10, now=now_utc(), usage_pressure=0.0)
    assert s_heavy < s_light


def test_high_quality_reduces_score():
    low_q = make_memory(pack_count=0, quality_score=0.1, created_at=old(10))
    high_q = make_memory(pack_count=0, quality_score=0.9, created_at=old(10))
    s_low = forgetting_score(low_q, anchor_size=10, now=now_utc(), usage_pressure=1.0)
    s_high = forgetting_score(high_q, anchor_size=10, now=now_utc(), usage_pressure=1.0)
    assert s_high < s_low


def test_score_range_bounds():
    m = make_memory(pack_count=0, quality_score=0.0, created_at=old(10))
    upper = forgetting_score(m, anchor_size=10_000, now=now_utc(), usage_pressure=1.0)
    assert upper <= 2.0 + 1e-9
    strong = make_memory(pack_count=500, quality_score=1.0, created_at=old(10))
    lower = forgetting_score(strong, anchor_size=1, now=now_utc(), usage_pressure=0.0)
    assert lower >= -2.0 - 1e-9


def test_prune_anchor_archives_local_worst(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = MemoryStore(str(tmp_path / "memory.sqlite3"))
    tenant = "t"
    pack_counts = [0, 0, 5, 5, 10, 10, 20, 20, 30, 30, 40, 40]
    ids = _seed_anchor(store, tenant, 42, pack_counts=pack_counts)

    n_archived = prune_anchor(store, tenant, 42, budget=10, now=now_utc())

    assert n_archived == 2
    surviving = {m.memory_object_id for m in store.query_by_anchor_for_pruning(tenant, 42)}
    assert ids[0] not in surviving
    assert ids[1] not in surviving
