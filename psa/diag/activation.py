"""activation.py — per-anchor activation / carry rollup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .trace_reader import iter_trace_records


@dataclass
class AnchorActivation:
    anchor_id: int
    anchor_name: str
    n_selected: int
    n_carried: int
    carry_rate: float


def _load_atlas_for_tenant(tenant_id: str) -> Any:
    """Load atlas via TenantManager + AtlasManager. Kept as a patchable helper."""
    from ..tenant import TenantManager
    from ..atlas import AtlasManager

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    return AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id).get_atlas()


def activation_report(
    tenant_id: str, *, origins: Optional[set[str]] = None
) -> list[AnchorActivation]:
    """Compute per-anchor activation/carry stats across the trace log."""
    atlas = _load_atlas_for_tenant(tenant_id)
    if atlas is None:
        return []
    name_by_id = {c.anchor_id: c.name for c in atlas.cards}

    n_selected: dict[int, int] = {}
    n_carried: dict[int, int] = {}

    for rec in iter_trace_records(tenant_id, origins=origins):
        selected = rec.get("selected_anchor_ids", []) or []
        packed = rec.get("packed_memories", []) or []
        carried_source_anchors = {pm.get("source_anchor_id") for pm in packed}

        for aid in selected:
            n_selected[aid] = n_selected.get(aid, 0) + 1
            if aid in carried_source_anchors:
                n_carried[aid] = n_carried.get(aid, 0) + 1

    result: list[AnchorActivation] = []
    for aid, sel in n_selected.items():
        carried = n_carried.get(aid, 0)
        result.append(
            AnchorActivation(
                anchor_id=aid,
                anchor_name=name_by_id.get(aid, f"anchor-{aid}"),
                n_selected=sel,
                n_carried=carried,
                carry_rate=(carried / sel) if sel else 0.0,
            )
        )
    return result
