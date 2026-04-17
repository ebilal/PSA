"""
metadata.py — Per-pattern provenance storage in pattern_metadata.json.

Sibling file to anchor_cards.json. Holds only provenance (source, created_at,
optional pinned flag) — never dynamic counters. Dynamic reinforcement signal
is derived per-run from the trace log.

Key format: "{anchor_id}::{normalized_pattern}" where normalize_pattern
collapses whitespace, strips, and lowercases. Stable across formatting drift.

Writes are atomic (tmp + os.replace) so concurrent readers never observe
a half-written file.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Optional

logger = logging.getLogger("psa.advertisement.metadata")

FILENAME = "pattern_metadata.json"


def normalize_pattern(text: str) -> str:
    """Normalize pattern text for key-stable storage.

    Collapses internal whitespace, strips leading/trailing whitespace, lowercases.
    Logical identity: cases and spacing drift don't create fake new entries.
    """
    return " ".join(text.strip().lower().split())


def metadata_key(anchor_id: int, pattern: str) -> str:
    """Compose the metadata key for an (anchor_id, pattern) pair."""
    return f"{anchor_id}::{normalize_pattern(pattern)}"


@dataclass
class PatternMetadata:
    """Provenance for a single anchor-card pattern.

    Stored on disk; reinforcement state (last_reinforced_at) is computed per-run
    and never written here — see psa/forgetting/reinforcement.py.

    Reserved fields (schema-forward, not consumed in this branch):
        promoted_at — when the pattern first landed on anchor_cards_refined.json.
                      Absent/None until the creation-time stamping follow-up lands.
        pinned — P3 operator-pinned flag. Supported in schema; no CLI to set yet.
    """

    anchor_id: int
    pattern: str
    source: str  # "atlas_build" | "refinement" | "production_signal" | "manual" | "unknown"
    created_at: str  # ISO-8601 UTC timestamp
    promoted_at: Optional[str] = None
    pinned: bool = False


def load_metadata(atlas_dir: str) -> dict[str, PatternMetadata]:
    """Load pattern_metadata.json from atlas_dir. Missing file → empty dict.

    Tolerates unknown keys in entries (forward compat). Malformed entries are
    skipped with a debug log.
    """
    path = os.path.join(atlas_dir, FILENAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return {}

    out: dict[str, PatternMetadata] = {}
    known_fields = {"anchor_id", "pattern", "source", "created_at", "promoted_at", "pinned"}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        try:
            filtered = {k: v for k, v in entry.items() if k in known_fields}
            out[key] = PatternMetadata(**filtered)
        except TypeError as e:
            logger.debug("Skipping malformed metadata entry %s: %s", key, e)
    return out


def save_metadata(atlas_dir: str, metadata: dict[str, PatternMetadata]) -> None:
    """Persist metadata atomically (write tmp, os.replace to final)."""
    os.makedirs(atlas_dir, exist_ok=True)
    final_path = os.path.join(atlas_dir, FILENAME)
    tmp_path = final_path + ".tmp"
    serialized = {key: asdict(meta) for key, meta in metadata.items()}
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)
    os.replace(tmp_path, final_path)


def backfill_unknown(
    metadata: dict[str, PatternMetadata],
    patterns_by_anchor: dict[int, list[str]],
    now_iso: str,
) -> int:
    """Stamp missing entries with source='unknown', created_at=now.

    Conservative migration policy — see spec §5.2. Does NOT mutate entries
    that already exist.

    Returns the count of entries newly added.
    """
    n_added = 0
    for anchor_id, patterns in patterns_by_anchor.items():
        for pattern in patterns:
            key = metadata_key(anchor_id, pattern)
            if key in metadata:
                continue
            metadata[key] = PatternMetadata(
                anchor_id=anchor_id,
                pattern=pattern,
                source="unknown",
                created_at=now_iso,
            )
            n_added += 1
    return n_added
