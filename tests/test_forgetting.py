"""Tests for psa.forgetting: score behavior and pruning order."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from psa.forgetting import forgetting_score
from psa.memory_object import MemoryObject, MemoryType


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


def test_fixture_builds_memory():
    m = make_memory(pack_count=3, created_at=old(10))
    assert m.pack_count == 3
    assert m.primary_anchor_id == 1
