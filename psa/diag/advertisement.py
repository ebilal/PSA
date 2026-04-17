"""advertisement.py — memory_count vs activation_rate gap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .trace_reader import iter_trace_records


@dataclass
class AnchorAdvertisement:
    anchor_id: int
    anchor_name: str
    memory_count: int
    n_selected: int
    activation_rate: float
    memory_percentile: float
    activation_percentile: float
    advertisement_gap: float


def _load_atlas_for_tenant(tenant_id: str) -> Any:
    from ..tenant import TenantManager
    from ..atlas import AtlasManager

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    return AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id).get_atlas()


def _percentile_rank(value: float, sorted_values: list[float]) -> float:
    """Return percentile rank (0-100) of `value` within `sorted_values`.

    Uses count of values <= `value` divided by total count. Simple and
    stable against ties.
    """
    if not sorted_values:
        return 0.0
    n_le = sum(1 for v in sorted_values if v <= value)
    return 100.0 * n_le / len(sorted_values)


def advertisement_report(
    tenant_id: str, *, origins: Optional[set[str]] = None
) -> list[AnchorAdvertisement]:
    atlas = _load_atlas_for_tenant(tenant_id)
    if atlas is None:
        return []

    n_selected: dict[int, int] = {}
    total_queries = 0
    for rec in iter_trace_records(tenant_id, origins=origins):
        total_queries += 1
        for aid in rec.get("selected_anchor_ids", []) or []:
            n_selected[aid] = n_selected.get(aid, 0) + 1

    memory_counts = [c.memory_count for c in atlas.cards]
    memory_counts_sorted = sorted(memory_counts)
    activation_rates = [
        (n_selected.get(c.anchor_id, 0) / total_queries) if total_queries else 0.0
        for c in atlas.cards
    ]
    activation_rates_sorted = sorted(activation_rates)

    rows: list[AnchorAdvertisement] = []
    for c in atlas.cards:
        sel = n_selected.get(c.anchor_id, 0)
        act_rate = (sel / total_queries) if total_queries else 0.0
        mem_pct = _percentile_rank(c.memory_count, memory_counts_sorted)
        act_pct = _percentile_rank(act_rate, activation_rates_sorted)
        rows.append(
            AnchorAdvertisement(
                anchor_id=c.anchor_id,
                anchor_name=c.name,
                memory_count=c.memory_count,
                n_selected=sel,
                activation_rate=act_rate,
                memory_percentile=mem_pct,
                activation_percentile=act_pct,
                advertisement_gap=mem_pct - act_pct,
            )
        )
    return rows
