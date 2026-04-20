"""
forgetting.py — Memory forgetting: scoring, per-anchor pruning, global cap.

Implements a simple 4-term forgetting score and two pruning strategies:
  1. Per-anchor pruning: keep each anchor within a memory budget
  2. Global cap: hard limit on total active memories

Forgetting is always soft (archive) first, hard (delete) only for memories
that have been archived for > 90 days.
"""

import logging
from datetime import datetime, timezone
from math import log
from typing import Optional

from .memory_object import MemoryObject, MemoryStore

logger = logging.getLogger("psa.forgetting")

# ── Defaults ─────────────────────────────────────────────────────────────────

ANCHOR_MEMORY_BUDGET = 100
MAX_MEMORIES = 50_000
GLOBAL_CAP_ANCHOR_WEIGHT = 0.7
GLOBAL_CAP_TENANT_WEIGHT = 0.3


# ── Forgetting score ─────────────────────────────────────────────────────────


def _days_since(iso_timestamp: Optional[str], now: datetime) -> float:
    """Return days between an ISO timestamp and now. Returns 0 if timestamp is None."""
    if not iso_timestamp:
        return 0.0
    try:
        then = datetime.fromisoformat(iso_timestamp)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        delta = now - then
        return max(delta.total_seconds() / 86400.0, 0.0)
    except (ValueError, TypeError):
        return 0.0


# ── Low-usage pressure (anchor-relative) ─────────────────────────────────────

SMALL_ANCHOR_THRESHOLD = 5


def _timestamp_for_sort(iso: Optional[str]) -> float:
    if not iso:
        return 0.0
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def _usage_sort_key(m: MemoryObject) -> tuple:
    """Ascending sort key: least-used (and weakest on tiebreaks) sorts first.

    Tiebreak order: pack_count, select_count, quality_score, then older created_at first.
    A memory that sorts earlier gets HIGHER pressure.
    """
    return (
        m.pack_count,
        m.select_count,
        m.quality_score,
        _timestamp_for_sort(m.created_at),
    )


def low_usage_pressure(memory: MemoryObject, peers: list) -> float:
    """
    Rank-based pressure within a peer set. Higher = more disposable.

    - Small anchors (< SMALL_ANCHOR_THRESHOLD peers): returns 0.0 (too noisy).
    - All-zero usage population: returns 0.0 (no signal; defer to crowding/quality).
    - Otherwise: bottom-ranked returns 1.0; top returns 0.0; linear between.
    """
    n = len(peers)
    if n < SMALL_ANCHOR_THRESHOLD:
        return 0.0
    if all(p.pack_count == 0 and p.select_count == 0 for p in peers):
        return 0.0

    ordered = sorted(peers, key=_usage_sort_key)
    try:
        rank = next(i for i, p in enumerate(ordered) if p.memory_object_id == memory.memory_object_id)
    except StopIteration:
        return 0.0
    if n == 1:
        return 0.0
    return 1.0 - rank / (n - 1)


def forgetting_score(
    memory: MemoryObject,
    anchor_size: int,
    target_per_anchor: int = ANCHOR_MEMORY_BUDGET,
    now: Optional[datetime] = None,
    *,
    usage_pressure: float = 0.0,
) -> float:
    """
    Compute a forgetting score for a memory. Higher = more disposable.

    Four terms, no tunable weights:
      + usage_pressure:   rank-based, precomputed by caller (range [0, 1])
      + crowding pressure:    min(overflow / target, 1.0)
      - usage protection:     min(log(1 + pack_count) / 3.0, 1.0)
      - quality protection:   quality_score

    Range is roughly [-2, 2]. Dormancy alone is NOT a disposal signal.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    age_days = _days_since(memory.created_at, now)
    if age_days < 1.0:
        return -10.0

    overflow = max(0, anchor_size - target_per_anchor) / max(target_per_anchor, 1)
    usage = log(1 + memory.pack_count) / 3.0

    return (
        usage_pressure
        + min(overflow, 1.0)
        - min(usage, 1.0)
        - memory.quality_score
    )


# ── Per-anchor pruning ───────────────────────────────────────────────────────


def prune_anchor(
    store: MemoryStore,
    tenant_id: str,
    anchor_id: int,
    budget: int = ANCHOR_MEMORY_BUDGET,
    now: Optional[datetime] = None,
) -> int:
    """
    If an anchor exceeds its memory budget, archive the lowest-value memories
    using anchor-relative low-usage pressure as the primary disposal signal.

    Returns the number of memories archived.
    """
    memories = store.query_by_anchor_for_pruning(tenant_id, anchor_id)
    if len(memories) <= budget:
        return 0

    if now is None:
        now = datetime.now(timezone.utc)

    anchor_size = len(memories)
    pressure_by_id = {
        m.memory_object_id: low_usage_pressure(m, memories) for m in memories
    }

    def score(m: MemoryObject) -> float:
        return forgetting_score(
            m,
            anchor_size,
            budget,
            now,
            usage_pressure=pressure_by_id[m.memory_object_id],
        )

    scored = sorted(memories, key=score, reverse=True)
    to_archive = scored[: len(memories) - budget]
    archive_ids = [m.memory_object_id for m in to_archive]
    store.archive_memories(archive_ids)

    logger.info(
        "Pruned anchor %d: archived %d memories (was %d, budget %d)",
        anchor_id,
        len(archive_ids),
        anchor_size,
        budget,
    )
    return len(archive_ids)


# ── Global cap ───────────────────────────────────────────────────────────────


def enforce_global_cap(
    store: MemoryStore,
    tenant_id: str,
    max_memories: int = MAX_MEMORIES,
    archived_ttl_days: int = 90,
) -> dict:
    """
    Enforce a global memory cap for a tenant.

    Phase 1: hard-delete memories archived > archived_ttl_days ago.
    Phase 2: if still over cap, archive the lowest-scoring active memories
             using a hybrid (anchor-relative + tenant-wide) pressure score.

    Returns a dict with counts of actions taken.
    """
    result = {"hard_deleted": 0, "archived": 0}

    deleted = store.delete_old_archived(tenant_id, older_than_days=archived_ttl_days)
    result["hard_deleted"] = deleted
    if deleted:
        logger.info("Global cap: hard-deleted %d old archived memories", deleted)

    active_count = store.count(tenant_id)
    if active_count <= max_memories:
        return result

    excess = active_count - max_memories
    now = datetime.now(timezone.utc)

    all_memories = store.get_all_with_embeddings(tenant_id)
    if not all_memories:
        return result

    by_anchor: dict = {}
    for m in all_memories:
        by_anchor.setdefault(m.primary_anchor_id, []).append(m)

    anchor_pressure = {
        m.memory_object_id: low_usage_pressure(m, by_anchor[m.primary_anchor_id])
        for m in all_memories
    }
    tenant_pressure = {
        m.memory_object_id: low_usage_pressure(m, all_memories) for m in all_memories
    }

    def score(m: MemoryObject) -> float:
        hybrid = (
            GLOBAL_CAP_ANCHOR_WEIGHT * anchor_pressure[m.memory_object_id]
            + GLOBAL_CAP_TENANT_WEIGHT * tenant_pressure[m.memory_object_id]
        )
        return forgetting_score(
            m,
            active_count,
            ANCHOR_MEMORY_BUDGET,
            now,
            usage_pressure=hybrid,
        )

    scored = sorted(all_memories, key=score, reverse=True)
    to_archive = scored[:excess]
    archive_ids = [m.memory_object_id for m in to_archive]
    store.archive_memories(archive_ids)
    result["archived"] = len(archive_ids)

    logger.info(
        "Global cap: archived %d memories (was %d, cap %d)",
        len(archive_ids),
        active_count,
        max_memories,
    )
    return result
