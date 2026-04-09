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
from typing import List, Optional

from .memory_object import MemoryObject, MemoryStore

logger = logging.getLogger("psa.forgetting")

# ── Defaults ─────────────────────────────────────────────────────────────────

ANCHOR_MEMORY_BUDGET = 100
MAX_MEMORIES = 50_000


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


def forgetting_score(
    memory: MemoryObject,
    anchor_size: int,
    target_per_anchor: int = ANCHOR_MEMORY_BUDGET,
    now: Optional[datetime] = None,
) -> float:
    """
    Compute a forgetting score for a memory. Higher = more disposable.

    Four terms, no tunable weights:
      + idle pressure:    min(idle_days / 90, 1.0)
      + crowding pressure: min(overflow / target, 1.0)
      - usage protection:  min(log(1 + pack_count) / 3.0, 1.0)
      - quality protection: quality_score

    Range is roughly [-1, 2].
    """
    if now is None:
        now = datetime.now(timezone.utc)

    idle_days = _days_since(memory.last_packed or memory.created_at, now)
    overflow = max(0, anchor_size - target_per_anchor) / max(target_per_anchor, 1)
    usage = log(1 + memory.pack_count) / 3.0  # 20 packs ~ 1.0

    return (
        min(idle_days / 90.0, 1.0)       # idle pressure (caps at 90 days)
        + min(overflow, 1.0)             # crowding pressure
        - min(usage, 1.0)               # usage protection
        - memory.quality_score           # quality protection
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
    If an anchor exceeds its memory budget, archive the lowest-value memories.

    Returns the number of memories archived.
    """
    memories = store.query_by_anchor_for_pruning(tenant_id, anchor_id)
    if len(memories) <= budget:
        return 0

    if now is None:
        now = datetime.now(timezone.utc)

    anchor_size = len(memories)
    scored = sorted(
        memories,
        key=lambda m: forgetting_score(m, anchor_size, budget, now),
        reverse=True,
    )
    to_archive = scored[: len(memories) - budget]
    archive_ids = [m.memory_object_id for m in to_archive]
    store.archive_memories(archive_ids)

    logger.info(
        "Pruned anchor %d: archived %d memories (was %d, budget %d)",
        anchor_id, len(archive_ids), anchor_size, budget,
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
    Phase 2: if still over cap, archive the lowest-scoring active memories.

    Returns a dict with counts of actions taken.
    """
    result = {"hard_deleted": 0, "archived": 0}

    # Phase 1: hard-delete old archived memories
    deleted = store.delete_old_archived(tenant_id, older_than_days=archived_ttl_days)
    result["hard_deleted"] = deleted
    if deleted:
        logger.info("Global cap: hard-deleted %d old archived memories", deleted)

    # Phase 2: check if still over cap
    active_count = store.count(tenant_id)
    if active_count <= max_memories:
        return result

    excess = active_count - max_memories
    now = datetime.now(timezone.utc)

    # Get all active memories with embeddings, score them, archive the worst
    all_memories = store.get_all_with_embeddings(tenant_id)
    if not all_memories:
        return result

    scored = sorted(
        all_memories,
        key=lambda m: forgetting_score(m, len(all_memories), max_memories // 256, now),
        reverse=True,
    )
    to_archive = scored[:excess]
    archive_ids = [m.memory_object_id for m in to_archive]
    store.archive_memories(archive_ids)
    result["archived"] = len(archive_ids)

    logger.info(
        "Global cap: archived %d memories (was %d, cap %d)",
        len(archive_ids), active_count, max_memories,
    )
    return result
