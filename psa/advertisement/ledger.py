"""
ledger.py — persistent pattern_ledger table + CRUD + decay arithmetic.

Schema is created lazily via create_schema() on first write. The table
lives in the tenant SQLite at ~/.psa/tenants/{tenant}/memory.sqlite3
alongside memory_objects.

Keyed by a content hash of (anchor_id, normalized_pattern_text) so the
id is stable across process restarts and changes naturally when the
pattern text changes (regeneration at atlas rebuild).
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone

from psa.advertisement.metadata import normalize_pattern


def pattern_id_for(anchor_id: int, pattern_text: str) -> str:
    """Stable content-hash id. Matches stage 1's metadata_key semantics."""
    norm = normalize_pattern(pattern_text)
    raw = f"{anchor_id}::{norm}".encode("utf-8")
    return "lp_" + hashlib.sha256(raw).hexdigest()[:24]


def create_schema(db: sqlite3.Connection) -> None:
    """Idempotent schema creation."""
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS pattern_ledger (
            pattern_id                     TEXT PRIMARY KEY,
            anchor_id                      INTEGER NOT NULL,
            pattern_text                   TEXT NOT NULL,

            ledger                         REAL NOT NULL DEFAULT 0.0,
            consecutive_negative_cycles    INTEGER NOT NULL DEFAULT 0,

            shadow_ledger                  REAL NOT NULL DEFAULT 0.0,
            shadow_consecutive             INTEGER NOT NULL DEFAULT 0,

            grace_expires_at               TEXT NOT NULL,
            created_at                     TEXT NOT NULL,
            last_updated_at                TEXT NOT NULL,

            removed_at                     TEXT,
            removal_reason                 TEXT,
            final_ledger                   REAL,
            final_shadow_ledger            REAL
        );
        CREATE INDEX IF NOT EXISTS idx_pattern_ledger_anchor
            ON pattern_ledger(anchor_id);
        CREATE INDEX IF NOT EXISTS idx_pattern_ledger_active
            ON pattern_ledger(anchor_id) WHERE removed_at IS NULL;
        """
    )
    db.commit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_ledger(
    db: sqlite3.Connection,
    pattern_id: str,
    anchor_id: int,
    pattern_text: str,
    ledger_delta: float,
    shadow_delta: float,
    grace_days: int,
) -> None:
    """Insert a fresh row or add the deltas to an existing row.

    On insert, stamps `grace_expires_at = now + grace_days`.
    `created_at` is set once; `last_updated_at` is always refreshed.
    """
    now = _now_iso()
    grace = (datetime.now(timezone.utc) + timedelta(days=grace_days)).isoformat()
    db.execute(
        """
        INSERT INTO pattern_ledger
            (pattern_id, anchor_id, pattern_text,
             ledger, shadow_ledger,
             grace_expires_at, created_at, last_updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pattern_id) DO UPDATE SET
            ledger = ledger + excluded.ledger,
            shadow_ledger = shadow_ledger + excluded.shadow_ledger,
            last_updated_at = excluded.last_updated_at
        """,
        (
            pattern_id,
            anchor_id,
            pattern_text,
            ledger_delta,
            shadow_delta,
            grace,
            now,
            now,
        ),
    )
    db.commit()
