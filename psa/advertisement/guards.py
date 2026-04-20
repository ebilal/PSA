"""Lifecycle-owned advertisement guard helpers.

These protections are part of the supported lifecycle policy and should not
depend on the legacy candidate-side advertisement decay flow.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from .metadata import load_metadata, metadata_key

LOW_ACTIVATION_PERCENTILE = 25.0
MIN_ANCHOR_ACTIVATIONS = 10
LOOKBACK_DAYS = 90


def _parse_iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _percentile(value: float, sorted_values: list[float]) -> float:
    if not sorted_values:
        return 0.0
    n_le = sum(1 for v in sorted_values if v <= value)
    return 100.0 * n_le / len(sorted_values)


def shielded_anchor_ids(
    *,
    tenant_id: str,
    atlas,
    anchor_ids,
    lookback_days: int = LOOKBACK_DAYS,
    low_activation_percentile: float = LOW_ACTIVATION_PERCENTILE,
    min_anchor_activations: int = MIN_ANCHOR_ACTIVATIONS,
) -> set[int]:
    """Return low-signal anchors whose patterns should not be removed."""
    active_anchor_ids = set(anchor_ids or [])
    if not active_anchor_ids:
        return set()

    counts = {aid: 0 for aid in active_anchor_ids}
    trace_path = os.path.expanduser(f"~/.psa/tenants/{tenant_id}/query_trace.jsonl")
    if not os.path.exists(trace_path):
        return active_anchor_ids

    window_start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_iso(rec.get("timestamp", ""))
            if ts is None or ts < window_start:
                continue
            for aid in rec.get("selected_anchor_ids") or []:
                if aid in counts:
                    counts[aid] += 1

    values = sorted(counts.values())
    shielded = set()
    for aid, count in counts.items():
        if count < min_anchor_activations:
            shielded.add(aid)
            continue
        if _percentile(count, values) < low_activation_percentile:
            shielded.add(aid)
    return shielded


def is_pattern_pinned(atlas_dir: str | None, anchor_id: int, pattern_text: str) -> bool:
    """Return whether the pattern is explicitly pinned in metadata."""
    if not atlas_dir:
        return False
    meta = load_metadata(atlas_dir)
    entry = meta.get(metadata_key(anchor_id, pattern_text))
    return bool(entry.pinned) if entry is not None else False
