"""misses.py — below-threshold query rollup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .trace_reader import iter_trace_records


@dataclass
class MissReport:
    total_queries: int
    empty_queries: int
    empty_rate: float
    recent_misses: list[dict] = field(default_factory=list)
    # Each tuple: (anchor_id, count_of_near_misses, mean_rank, mean_score)
    near_miss_anchors: list[tuple[int, int, float, float]] = field(default_factory=list)


def _load_atlas_for_tenant(tenant_id: str) -> Any:
    from ..tenant import TenantManager
    from ..atlas import AtlasManager

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    return AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id).get_atlas()


def miss_report(
    tenant_id: str,
    *,
    n_recent: int = 20,
    origins: Optional[set[str]] = None,
) -> MissReport:
    total_queries = 0
    empty_queries = 0
    recent_misses: list[dict] = []
    # anchor_id -> list of (rank, score) tuples from empty-selection records
    near_miss_accum: dict[int, list[tuple[int, float]]] = {}

    for rec in iter_trace_records(tenant_id, origins=origins):
        total_queries += 1
        if rec.get("result_kind") != "empty_selection":
            continue
        empty_queries += 1
        recent_misses.append(rec)
        for score_rec in rec.get("top_anchor_scores", []) or []:
            rank = score_rec.get("rank", 99)
            if rank > 3:
                continue
            aid = score_rec.get("anchor_id")
            if aid is None:
                continue
            near_miss_accum.setdefault(aid, []).append((rank, float(score_rec.get("score", 0.0))))

    near_miss_rows: list[tuple[int, int, float, float]] = []
    for aid, entries in near_miss_accum.items():
        count = len(entries)
        mean_rank = sum(r for r, _ in entries) / count
        mean_score = sum(s for _, s in entries) / count
        near_miss_rows.append((aid, count, mean_rank, mean_score))
    near_miss_rows.sort(key=lambda t: t[1], reverse=True)

    # Keep only the most recent N empty queries.
    recent_misses = recent_misses[-n_recent:]

    return MissReport(
        total_queries=total_queries,
        empty_queries=empty_queries,
        empty_rate=(empty_queries / total_queries) if total_queries else 0.0,
        recent_misses=recent_misses,
        near_miss_anchors=near_miss_rows,
    )
