"""
health.py — Atlas health monitoring for PSA.

Tracks key health signals for each atlas version:
  - novelty_rate: fraction of recent memories landing in novelty anchors
  - anchor_utilization: distribution of memory counts across anchors
  - memory_type_distribution: breakdown of memory types
  - embedding_drift: cosine distance between atlas centroid and recent memory centroids

Rebuild triggers (logged as warnings):
  - novelty_rate > 8%  — new semantic territory not covered by learned anchors
  - utilization skew > 3:1  — a few anchors dominate, most are empty

Usage::

    monitor = AtlasHealthMonitor()
    report = monitor.check_health(atlas, memory_store, tenant_id="default")
    if report.should_rebuild:
        # trigger AtlasManager.rebuild()
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .atlas import Atlas
from .memory_object import MemoryStore

logger = logging.getLogger("psa.health")

# ── Rebuild thresholds ────────────────────────────────────────────────────────

NOVELTY_RATE_THRESHOLD = 0.08  # >8% novelty → rebuild
UTILIZATION_SKEW_THRESHOLD = 3.0  # max/median utilization > 3× → rebuild


# ── HealthReport ──────────────────────────────────────────────────────────────


@dataclass
class AnchorStats:
    """Per-anchor utilization statistics."""

    anchor_id: int
    name: str
    memory_count: int
    is_novelty: bool


@dataclass
class HealthReport:
    """Complete atlas health report for a tenant."""

    tenant_id: str
    atlas_version: int
    total_memories: int
    total_anchors: int
    novelty_anchors: int
    novelty_rate: float  # fraction of memories in novelty anchors
    utilization_skew: float  # max_count / median_count (learned anchors only)
    memory_type_distribution: Dict[str, int]
    anchor_stats: List[AnchorStats]
    embedding_drift: Optional[float]  # mean cosine distance from centroids (None if not computed)
    should_rebuild: bool
    rebuild_reasons: List[str]
    # Lifecycle metrics
    never_packed_count: int = 0  # memories with pack_count == 0
    archived_count: int = 0  # memories with is_archived = 1
    capacity_pct: float = 0.0  # total / MAX_MEMORIES

    def summary(self) -> str:
        lines = [
            f"Atlas v{self.atlas_version} health report ({self.tenant_id}):",
            f"  memories: {self.total_memories}",
            f"  anchors: {self.total_anchors} ({self.novelty_anchors} novelty)",
            f"  novelty_rate: {self.novelty_rate:.1%} (threshold {NOVELTY_RATE_THRESHOLD:.0%})",
            f"  utilization_skew: {self.utilization_skew:.2f}x (threshold {UTILIZATION_SKEW_THRESHOLD:.0f}x)",
        ]
        if self.memory_type_distribution:
            lines.append(
                "  memory types: "
                + ", ".join(f"{t}={n}" for t, n in sorted(self.memory_type_distribution.items()))
            )
        if self.never_packed_count > 0:
            lines.append(f"  never_packed: {self.never_packed_count}")
        if self.archived_count > 0:
            lines.append(f"  archived: {self.archived_count}")
        if self.capacity_pct > 0:
            lines.append(f"  capacity: {self.capacity_pct:.1%}")
        if self.should_rebuild:
            lines.append(f"  ⚠ REBUILD RECOMMENDED: {'; '.join(self.rebuild_reasons)}")
        else:
            lines.append("  ✓ atlas is healthy")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "atlas_version": self.atlas_version,
            "total_memories": self.total_memories,
            "total_anchors": self.total_anchors,
            "novelty_anchors": self.novelty_anchors,
            "novelty_rate": round(self.novelty_rate, 4),
            "utilization_skew": round(self.utilization_skew, 4),
            "memory_type_distribution": self.memory_type_distribution,
            "embedding_drift": (
                round(self.embedding_drift, 4) if self.embedding_drift is not None else None
            ),
            "should_rebuild": self.should_rebuild,
            "rebuild_reasons": self.rebuild_reasons,
            "never_packed_count": self.never_packed_count,
            "archived_count": self.archived_count,
            "capacity_pct": round(self.capacity_pct, 4),
        }


# ── AtlasHealthMonitor ────────────────────────────────────────────────────────


class AtlasHealthMonitor:
    """
    Monitor atlas health and report rebuild signals.

    Usage::

        monitor = AtlasHealthMonitor()
        report = monitor.check_health(atlas, store, tenant_id="default")
    """

    def check_health(
        self,
        atlas: Atlas,
        store: MemoryStore,
        tenant_id: str = "default",
    ) -> HealthReport:
        """
        Compute health metrics for the given atlas.

        Parameters
        ----------
        atlas:
            The loaded Atlas to evaluate.
        store:
            MemoryStore for the tenant (used to count memories per anchor).
        tenant_id:
            Tenant identifier.

        Returns
        -------
        HealthReport with rebuild recommendation.
        """
        cards = atlas.cards
        total_anchors = len(cards)
        novelty_anchors = sum(1 for c in cards if c.is_novelty)

        # Per-anchor memory counts via single GROUP BY query
        counts_by_anchor = store.count_by_anchor(tenant_id)
        anchor_stats: List[AnchorStats] = []
        novelty_memories = 0
        total_memories = 0

        for card in cards:
            count = counts_by_anchor.get(card.anchor_id, 0)
            total_memories += count
            if card.is_novelty:
                novelty_memories += count
            anchor_stats.append(
                AnchorStats(
                    anchor_id=card.anchor_id,
                    name=card.name,
                    memory_count=count,
                    is_novelty=card.is_novelty,
                )
            )

        # Novelty rate
        novelty_rate = novelty_memories / max(total_memories, 1)

        # Utilization skew: max / median for learned anchors only
        learned_counts = [s.memory_count for s in anchor_stats if not s.is_novelty]
        utilization_skew = _utilization_skew(learned_counts)

        # Memory type distribution via single GROUP BY query
        memory_type_distribution = store.count_by_type(tenant_id)

        # Rebuild decision
        rebuild_reasons: List[str] = []
        if novelty_rate > NOVELTY_RATE_THRESHOLD:
            rebuild_reasons.append(
                f"novelty_rate {novelty_rate:.1%} > {NOVELTY_RATE_THRESHOLD:.0%}"
            )
        if utilization_skew > UTILIZATION_SKEW_THRESHOLD:
            rebuild_reasons.append(
                f"utilization_skew {utilization_skew:.2f}x > {UTILIZATION_SKEW_THRESHOLD:.0f}x"
            )

        # Lifecycle metrics
        forgetting = store.forgetting_stats(tenant_id)
        never_packed_count = forgetting.get("never_packed", 0)
        archived_count = forgetting.get("archived", 0)
        from .forgetting import MAX_MEMORIES

        capacity_pct = total_memories / max(MAX_MEMORIES, 1)

        should_rebuild = len(rebuild_reasons) > 0
        if should_rebuild:
            logger.warning(
                "Atlas v%d health check: REBUILD RECOMMENDED for tenant '%s'. Reasons: %s",
                atlas.version,
                tenant_id,
                "; ".join(rebuild_reasons),
            )
        else:
            logger.info(
                "Atlas v%d health check: healthy (novelty=%.1f%%, skew=%.2fx) for tenant '%s'",
                atlas.version,
                novelty_rate * 100,
                utilization_skew,
                tenant_id,
            )

        return HealthReport(
            tenant_id=tenant_id,
            atlas_version=atlas.version,
            total_memories=total_memories,
            total_anchors=total_anchors,
            novelty_anchors=novelty_anchors,
            novelty_rate=novelty_rate,
            utilization_skew=utilization_skew,
            memory_type_distribution=memory_type_distribution,
            anchor_stats=anchor_stats,
            embedding_drift=None,  # V1: not computed (requires recent memory embeddings)
            should_rebuild=should_rebuild,
            rebuild_reasons=rebuild_reasons,
            never_packed_count=never_packed_count,
            archived_count=archived_count,
            capacity_pct=capacity_pct,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _utilization_skew(counts: List[int]) -> float:
    """
    Compute max / median utilization for a list of per-anchor memory counts.

    Returns 1.0 if the list has fewer than 2 elements.
    """
    if len(counts) < 2:
        return 1.0
    sorted_counts = sorted(counts)
    n = len(sorted_counts)
    mid = n // 2
    median = (sorted_counts[mid - 1] + sorted_counts[mid]) / 2 if n % 2 == 0 else sorted_counts[mid]
    max_count = sorted_counts[-1]
    if median == 0:
        return float(max_count) if max_count > 0 else 1.0
    return max_count / median


def _count_memory_types(store: MemoryStore, tenant_id: str) -> Dict[str, int]:
    """Count memories per MemoryType for a tenant."""
    from .memory_object import MemoryType

    distribution: Dict[str, int] = {}
    for mtype in MemoryType:
        memories = store.query_by_type(
            tenant_id=tenant_id,
            memory_type=mtype,
            limit=100_000,
        )
        if memories:
            distribution[mtype.value] = len(memories)
    return distribution
