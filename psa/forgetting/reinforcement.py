"""
reinforcement.py — derive last_reinforced_at per pattern from the trace log.

Principle: reinforcement is computed per-run into an in-memory dict and
never written to disk. The trace log (Branch 4) is the authoritative source
of dynamic signal; re-deriving every run keeps persistent metadata minimal
and drift-free.

Reinforcement rule (R1, see spec §2):
    A trace record reinforces pattern P on anchor A when
        - record.selected_anchor_ids contains A, AND
        - record.query_origin is in the active origins set, AND
        - normalize(P) is a substring of normalize(record.query).
    last_reinforced_at[P] = latest qualifying record timestamp.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from .metadata import metadata_key, normalize_pattern

logger = logging.getLogger("psa.forgetting.reinforcement")


def compute_reinforcement(
    atlas: Any,
    trace_path: str,
    *,
    origins: set[str],
    window_start: datetime,
) -> dict[str, datetime]:
    """Return {metadata_key: last_reinforced_at} for every reinforced pattern.

    The window_start bound is a conservative optimization: records older than
    (decay_window_days + grace_days) cannot affect the decay decision. Widening
    it later does not change the rule's semantics, just scans more records.

    Missing trace file → returns {}.
    """
    if not os.path.exists(trace_path):
        return {}

    # Build a pattern index: anchor_id -> list of (normalized_pattern, raw_pattern)
    pattern_index: dict[int, list[tuple[str, str]]] = {}
    for card in atlas.cards:
        patterns = getattr(card, "generated_query_patterns", []) or []
        if patterns:
            pattern_index[card.anchor_id] = [
                (normalize_pattern(p), p) for p in patterns
            ]

    rmap: dict[str, datetime] = {}
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed trace line")
                continue

            ts_str = rec.get("timestamp")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue
            if ts < window_start:
                continue

            if rec.get("query_origin", "interactive") not in origins:
                continue

            selected_ids = rec.get("selected_anchor_ids") or []
            if not selected_ids:
                continue

            q_norm = normalize_pattern(rec.get("query", ""))
            if not q_norm:
                continue

            for aid in selected_ids:
                for norm_p, raw_p in pattern_index.get(aid, []):
                    if norm_p and norm_p in q_norm:
                        key = metadata_key(aid, raw_p)
                        if key not in rmap or ts > rmap[key]:
                            rmap[key] = ts

    return rmap
