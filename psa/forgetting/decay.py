"""
decay.py — orchestrate an advertisement forgetting pass.

D1: grace period + reinforcement window.
P1: shield anchors below absolute activation floor OR below percentile.
P3: pinned patterns are exempt.

Entry point: decay_report(tenant_id, params, origins) → DecayReport.
No disk side effects in this module EXCEPT metadata backfill persistence
(non-destructive provenance establishment).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .metadata import (
    backfill_unknown,
    load_metadata,
    metadata_key,
    save_metadata,
)
from .reinforcement import compute_reinforcement

logger = logging.getLogger("psa.forgetting.decay")

REASON_STALE = "stale_unreinforced"


@dataclass
class DecayParams:
    grace_days: int = 30
    decay_window_days: int = 60
    low_activation_percentile: float = 25.0
    min_anchor_activations: int = 10


@dataclass
class RemovedPattern:
    anchor_id: int
    pattern: str
    source: str
    created_at: str
    last_reinforced_at: str | None
    reason: str


@dataclass
class ShieldedAnchor:
    anchor_id: int
    activation_count: int
    patterns_held: int


@dataclass
class DecayReport:
    tenant_id: str
    atlas_version: int
    created_at: str
    params: DecayParams
    origins: set[str]
    n_patterns_scanned: int
    n_patterns_removed: int
    n_patterns_by_source_removed: dict[str, int]
    n_anchors_touched: int
    n_anchors_shielded: int
    n_patterns_shielded: int
    n_patterns_pinned_exempt: int
    n_patterns_backfilled_this_run: int
    pruning_by_reason: dict[str, int]
    removed_patterns: list[RemovedPattern] = field(default_factory=list)
    shielded_anchors: list[ShieldedAnchor] = field(default_factory=list)
    new_cards: list[dict] = field(default_factory=list)  # refined card list post-decay


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _load_atlas_for_tenant(tenant_id: str) -> Any:
    """Load atlas via TenantManager + AtlasManager (patchable helper for tests)."""
    from ..atlas import AtlasManager
    from ..tenant import TenantManager

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    return AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id).get_atlas()


def _trace_path(tenant_id: str) -> str:
    home = os.path.expanduser("~")
    return os.path.join(home, ".psa", "tenants", tenant_id, "query_trace.jsonl")


def _parse_iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _compute_activation_counts(
    atlas: Any,
    trace_path: str,
    origins: set[str],
    window_start: datetime,
) -> dict[int, int]:
    """Count activations per anchor within window, respecting origin filter."""
    import json as _json

    counts: dict[int, int] = {c.anchor_id: 0 for c in atlas.cards}
    if not os.path.exists(trace_path):
        return counts
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            ts = _parse_iso(rec.get("timestamp", ""))
            if ts is None or ts < window_start:
                continue
            if rec.get("query_origin", "interactive") not in origins:
                continue
            for aid in rec.get("selected_anchor_ids") or []:
                if aid in counts:
                    counts[aid] += 1
    return counts


def _percentile(value: float, sorted_values: list[float]) -> float:
    """Percentile rank of value within sorted_values (0..100)."""
    if not sorted_values:
        return 0.0
    n_le = sum(1 for v in sorted_values if v <= value)
    return 100.0 * n_le / len(sorted_values)


def _shielded_anchors(
    activation_counts: dict[int, int],
    params: DecayParams,
) -> set[int]:
    """Apply P1 two-part shield.

    Anchor is shielded when it's below the absolute floor OR below the
    percentile cutoff, computed over all anchors' activation counts.
    """
    values = sorted(activation_counts.values())
    shielded: set[int] = set()
    for aid, count in activation_counts.items():
        if count < params.min_anchor_activations:
            shielded.add(aid)
            continue
        pct = _percentile(count, values)
        if pct < params.low_activation_percentile:
            shielded.add(aid)
    return shielded


def decay_report(
    tenant_id: str,
    *,
    params: DecayParams,
    origins: set[str],
) -> DecayReport:
    """Run a decay pass. Returns a DecayReport; persists only metadata backfill."""
    atlas = _load_atlas_for_tenant(tenant_id)
    if atlas is None:
        raise FileNotFoundError(
            f"No atlas for tenant {tenant_id!r}. Run 'psa atlas build' first."
        )

    now = _now_utc()
    now_iso = now.isoformat()
    window_start = now - timedelta(days=params.decay_window_days + params.grace_days)

    # Load + backfill metadata (persisted — provenance is non-destructive).
    metadata = load_metadata(atlas.anchor_dir)
    patterns_by_anchor: dict[int, list[str]] = {
        c.anchor_id: list(c.generated_query_patterns or []) for c in atlas.cards
    }
    n_backfilled = backfill_unknown(metadata, patterns_by_anchor, now_iso)
    if n_backfilled > 0:
        save_metadata(atlas.anchor_dir, metadata)

    # Derive reinforcement (ephemeral).
    trace_path = _trace_path(tenant_id)
    reinforcement = compute_reinforcement(
        atlas, trace_path, origins=origins, window_start=window_start,
    )

    # Activation counts + shielded anchors (P1).
    activation_counts = _compute_activation_counts(
        atlas, trace_path, origins, window_start,
    )
    shielded = _shielded_anchors(activation_counts, params)

    # Walk anchors, apply D1 + P1 + P3.
    n_patterns_scanned = 0
    n_patterns_pinned_exempt = 0
    removed: list[RemovedPattern] = []
    by_source: dict[str, int] = {}
    anchors_touched: set[int] = set()
    new_cards: list[dict] = []

    for card in atlas.cards:
        patterns = list(card.generated_query_patterns or [])
        n_patterns_scanned += len(patterns)
        new_patterns: list[str] = []

        for pattern in patterns:
            key = metadata_key(card.anchor_id, pattern)
            meta = metadata.get(key)
            if meta is None:
                # Shouldn't happen post-backfill, but be defensive.
                new_patterns.append(pattern)
                continue

            # P3: pinned exempts.
            if meta.pinned:
                new_patterns.append(pattern)
                n_patterns_pinned_exempt += 1
                continue

            # P1: shielded anchor retains all patterns.
            if card.anchor_id in shielded:
                new_patterns.append(pattern)
                continue

            # D1: grace + window.
            created = _parse_iso(meta.created_at)
            if created is None:
                new_patterns.append(pattern)
                continue
            age_days = (now - created).days
            if age_days < params.grace_days:
                new_patterns.append(pattern)
                continue

            last_ts = reinforcement.get(key)
            if last_ts is not None:
                days_since = (now - last_ts).days
                if days_since <= params.decay_window_days:
                    new_patterns.append(pattern)
                    continue

            # It's a decay candidate.
            removed.append(RemovedPattern(
                anchor_id=card.anchor_id,
                pattern=pattern,
                source=meta.source,
                created_at=meta.created_at,
                last_reinforced_at=last_ts.isoformat() if last_ts else None,
                reason=REASON_STALE,
            ))
            by_source[meta.source] = by_source.get(meta.source, 0) + 1
            anchors_touched.add(card.anchor_id)

        # Build refined card dict (existing card with pruned patterns).
        card_dict = _card_to_dict(card, new_patterns)
        new_cards.append(card_dict)

    shielded_list = [
        ShieldedAnchor(
            anchor_id=aid,
            activation_count=activation_counts.get(aid, 0),
            patterns_held=len(patterns_by_anchor.get(aid, [])),
        )
        for aid in shielded
    ]

    return DecayReport(
        tenant_id=tenant_id,
        atlas_version=getattr(atlas, "version", 0),
        created_at=now_iso,
        params=params,
        origins=origins,
        n_patterns_scanned=n_patterns_scanned,
        n_patterns_removed=len(removed),
        n_patterns_by_source_removed=by_source,
        n_anchors_touched=len(anchors_touched),
        n_anchors_shielded=len(shielded),
        n_patterns_shielded=sum(len(patterns_by_anchor.get(a, [])) for a in shielded),
        n_patterns_pinned_exempt=n_patterns_pinned_exempt,
        n_patterns_backfilled_this_run=n_backfilled,
        pruning_by_reason={REASON_STALE: len(removed)} if removed else {},
        removed_patterns=removed,
        shielded_anchors=shielded_list,
        new_cards=new_cards,
    )


def _card_to_dict(card: Any, new_patterns: list[str]) -> dict:
    """Serialize a card with updated generated_query_patterns.

    Reuses the card's to_dict when present; otherwise a field-wise fallback.
    Mirrors psa.curation.curator._card_to_dict pattern.
    """
    if hasattr(card, "to_dict"):
        d = card.to_dict()
    else:
        d = {
            "anchor_id": card.anchor_id,
            "name": getattr(card, "name", f"anchor-{card.anchor_id}"),
            "meaning": getattr(card, "meaning", ""),
            "memory_types": list(getattr(card, "memory_types", [])),
            "include_terms": list(getattr(card, "include_terms", [])),
            "exclude_terms": list(getattr(card, "exclude_terms", [])),
            "prototype_examples": list(getattr(card, "prototype_examples", [])),
            "near_but_different": list(getattr(card, "near_but_different", [])),
            "centroid": list(getattr(card, "centroid", [])),
            "memory_count": getattr(card, "memory_count", 0),
            "is_novelty": getattr(card, "is_novelty", False),
            "status": getattr(card, "status", "active"),
            "metadata": dict(getattr(card, "metadata", {})),
            "generated_query_patterns": list(getattr(card, "generated_query_patterns", [])),
            "query_fingerprint": list(getattr(card, "query_fingerprint", [])),
        }
    d["generated_query_patterns"] = new_patterns
    return d
