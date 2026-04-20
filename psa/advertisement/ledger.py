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
import json
import math
import os
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from psa.advertisement.attribution import attribute_bm25_argmax
from psa.advertisement.metadata import normalize_pattern
from psa.advertisement.reload import write_reload_marker


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
    Negative-cycle counters advance only when fresh query signals arrive.
    """
    now = _now_iso()
    grace = (datetime.now(timezone.utc) + timedelta(days=grace_days)).isoformat()
    db.execute(
        """
        INSERT INTO pattern_ledger
            (pattern_id, anchor_id, pattern_text,
             ledger, shadow_ledger,
             consecutive_negative_cycles, shadow_consecutive,
             grace_expires_at, created_at, last_updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pattern_id) DO UPDATE SET
            ledger = ledger + excluded.ledger,
            shadow_ledger = shadow_ledger + excluded.shadow_ledger,
            consecutive_negative_cycles = CASE
                WHEN ledger + excluded.ledger < 0 THEN consecutive_negative_cycles + 1
                ELSE 0
            END,
            shadow_consecutive = CASE
                WHEN shadow_ledger + excluded.shadow_ledger < 0 THEN shadow_consecutive + 1
                ELSE 0
            END,
            last_updated_at = excluded.last_updated_at
        """,
        (
            pattern_id,
            anchor_id,
            pattern_text,
            ledger_delta,
            shadow_delta,
            1 if ledger_delta < 0 else 0,
            1 if shadow_delta < 0 else 0,
            grace,
            now,
            now,
        ),
    )
    db.commit()


@dataclass
class AnchorAttribution:
    """Per-retrieved-anchor attribution record.

    credited is [argmax] + eps_tied when BM25 floor passes and argmax exists;
    empty otherwise. Each credited pattern shares the per-query weight.
    """

    anchor_id: int
    argmax_pattern: str | None
    eps_tied_patterns: list[str] = field(default_factory=list)
    credited: list[str] = field(default_factory=list)
    bm25_floor_passed: bool = True


def compute_attribution(
    *,
    query: str,
    retrieved_anchor_ids: list[int],
    atlas,
    bm25_topk_anchor_ids,
    epsilon: float,
) -> list[AnchorAttribution]:
    """Run BM25 argmax per retrieved anchor, gate on BM25 top-K membership."""
    bm25_set = set(bm25_topk_anchor_ids)
    out: list[AnchorAttribution] = []
    for aid in retrieved_anchor_ids:
        if aid not in bm25_set:
            out.append(
                AnchorAttribution(
                    anchor_id=aid,
                    argmax_pattern=None,
                    bm25_floor_passed=False,
                )
            )
            continue
        anchor = atlas.get_anchor(aid)
        patterns = anchor.generated_query_patterns if anchor is not None else []
        argmax, tied = attribute_bm25_argmax(
            query, patterns, epsilon=epsilon
        )
        credited = [argmax] + list(tied) if argmax is not None else []
        out.append(
            AnchorAttribution(
                anchor_id=aid,
                argmax_pattern=argmax,
                eps_tied_patterns=list(tied),
                credited=credited,
                bm25_floor_passed=True,
            )
        )
    return out


def record_query_signals(
    *,
    db,
    attribution: list[AnchorAttribution],
    selected_anchor_ids,
    config,
) -> None:
    """Apply retrieval/pick/decline credit to every credited template."""
    if not config.tracking_enabled:
        return

    selected = set(selected_anchor_ids)
    for attr in attribution:
        if not attr.credited:
            continue
        n = len(attr.credited)
        per = 1.0 / n

        base = config.retrieval_credit
        shadow = config.retrieval_credit
        if attr.anchor_id in selected:
            base += config.selector_pick_credit
            shadow += config.selector_pick_credit
        else:
            base -= config.selector_decline_penalty
            shadow -= config.shadow.selector_decline_penalty

        for pat in attr.credited:
            upsert_ledger(
                db=db,
                pattern_id=pattern_id_for(attr.anchor_id, pat),
                anchor_id=attr.anchor_id,
                pattern_text=pat,
                ledger_delta=base * per,
                shadow_delta=shadow * per,
                grace_days=config.grace_days,
            )


def apply_decay(db: sqlite3.Connection, tau_days: int, removal_threshold: float = 0.0) -> None:
    """Apply exponential decay to active rows.

    ledger ← ledger · exp(−1/tau)   (shadow_ledger likewise)

    Negative-cycle counters are intentionally NOT updated here. Dormancy alone
    is not enough to advance a pattern toward removal; only fresh query signals
    can advance or reset the streak counters.

    Operates on unarchived rows only.
    """
    factor = math.exp(-1.0 / tau_days)
    db.execute(
        """
        UPDATE pattern_ledger
        SET
            ledger = ledger * ?,
            shadow_ledger = shadow_ledger * ?,
            last_updated_at = ?
        WHERE removed_at IS NULL
        """,
        (factor, factor, _now_iso()),
    )
    db.commit()


@dataclass
class RemovalCandidate:
    pattern_id: str
    anchor_id: int
    pattern_text: str
    ledger: float
    consecutive_negative_cycles: int
    shadow_ledger: float
    shadow_consecutive: int


@dataclass
class EvaluationResult:
    removal_candidates: list[RemovalCandidate] = field(default_factory=list)
    shadow_candidates: list[RemovalCandidate] = field(default_factory=list)
    n_active: int = 0
    n_in_grace: int = 0
    n_at_risk: int = 0


def evaluate_removal(
    *,
    db,
    atlas,
    tenant_id: str,
    config,
    shielded_anchor_fn,
    pinned_fn,
) -> EvaluationResult:
    """Evaluate active rows against the removal rule.

    `shielded_anchor_fn(tenant_id, anchor_ids) -> set[int]` re-uses stage 1 P1.
    `pinned_fn(anchor_id, pattern_text) -> bool` re-uses stage 1 P3.

    Does not mutate the DB. Callers decide whether to apply removals.
    """
    now_iso = _now_iso()
    rows = db.execute(
        """
        SELECT pattern_id, anchor_id, pattern_text, ledger,
               consecutive_negative_cycles, shadow_ledger, shadow_consecutive,
               grace_expires_at
        FROM pattern_ledger
        WHERE removed_at IS NULL
        """
    ).fetchall()

    n_active = len(rows)
    n_in_grace = 0
    n_at_risk = 0
    anchor_ids = {r[1] for r in rows}
    shielded = shielded_anchor_fn(tenant_id, anchor_ids) or set()

    # Count per-anchor active patterns AFTER potential removal
    # (we need this to enforce min_patterns_floor).
    per_anchor_counts = {aid: 0 for aid in anchor_ids}
    for aid in anchor_ids:
        anchor = atlas.get_anchor(aid)
        per_anchor_counts[aid] = len(anchor.generated_query_patterns)

    removal: list[RemovalCandidate] = []
    shadow: list[RemovalCandidate] = []

    for (
        pid,
        aid,
        text,
        lgr,
        cnc,
        shadow_lgr,
        shadow_cnc,
        grace_expires_at,
    ) in rows:
        in_grace = grace_expires_at > now_iso
        if in_grace:
            n_in_grace += 1
            continue

        if cnc >= max(1, config.sustained_cycles // 2):
            n_at_risk += 1

        candidate = RemovalCandidate(
            pattern_id=pid,
            anchor_id=aid,
            pattern_text=text,
            ledger=lgr,
            consecutive_negative_cycles=cnc,
            shadow_ledger=shadow_lgr,
            shadow_consecutive=shadow_cnc,
        )

        # Shared protections
        if aid in shielded:
            continue
        if pinned_fn(aid, text):
            continue
        if per_anchor_counts.get(aid, 0) - 1 < config.min_patterns_floor:
            continue

        if cnc >= config.sustained_cycles:
            removal.append(candidate)
            per_anchor_counts[aid] -= 1

        if shadow_cnc >= config.shadow.sustained_cycles:
            shadow.append(candidate)

    return EvaluationResult(
        removal_candidates=removal,
        shadow_candidates=shadow,
        n_active=n_active,
        n_in_grace=n_in_grace,
        n_at_risk=n_at_risk,
    )


def _atomic_write_json(path: str, obj) -> None:
    """Write JSON via tmp + os.replace for durability."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".cards-", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_live_cards(atlas_dir: str):
    """Prefer refined; fall back to base. Returns (cards, source_file_name)."""
    refined = os.path.join(atlas_dir, "anchor_cards_refined.json")
    base = os.path.join(atlas_dir, "anchor_cards.json")
    if os.path.exists(refined):
        with open(refined) as f:
            return json.load(f), "anchor_cards_refined.json"
    with open(base) as f:
        return json.load(f), "anchor_cards.json"


def apply_removals(
    *,
    db,
    tenant_id: str,
    atlas_dir: str,
    candidates: list,
) -> int:
    """Drop candidate patterns from refined cards; archive ledger rows; write marker.

    Returns the number of successfully-applied removals.
    """
    if not candidates:
        return 0

    cards, _ = _load_live_cards(atlas_dir)
    # Index cards by anchor_id for quick lookup
    by_anchor: dict[int, dict] = {}
    for c in cards:
        aid = c.get("anchor_id") if isinstance(c, dict) else getattr(c, "anchor_id", None)
        if aid is not None:
            by_anchor[aid] = c

    changed_anchors: set[int] = set()
    applied: list = []
    for cand in candidates:
        card = by_anchor.get(cand.anchor_id)
        if card is None:
            continue
        patterns = card.get("generated_query_patterns", [])
        # Remove by normalized-text match so format drift doesn't defeat us.
        target = normalize_pattern(cand.pattern_text)
        new_patterns = [p for p in patterns if normalize_pattern(p) != target]
        if len(new_patterns) == len(patterns):
            continue
        card["generated_query_patterns"] = new_patterns
        changed_anchors.add(cand.anchor_id)
        applied.append(cand)

    if not applied:
        return 0

    refined_path = os.path.join(atlas_dir, "anchor_cards_refined.json")
    _atomic_write_json(refined_path, cards)

    now = _now_iso()
    for cand in applied:
        db.execute(
            """
            UPDATE pattern_ledger
            SET removed_at = ?,
                removal_reason = ?,
                final_ledger = ?,
                final_shadow_ledger = ?
            WHERE pattern_id = ?
            """,
            (
                now,
                "stage2_sustained_negative",
                cand.ledger,
                cand.shadow_ledger,
                cand.pattern_id,
            ),
        )
    db.commit()

    write_reload_marker(tenant_id=tenant_id, changed_anchor_ids=changed_anchors)
    return len(applied)


def reset_ledger_on_rebuild(*, db, new_atlas, grace_days: int) -> None:
    """Archive every active row; insert fresh rows for new atlas's patterns.

    Called from AtlasBuilder.build_atlas() AFTER the metadata-inheritance pass,
    so stage 1 and stage 2 both see a consistent rebuild event.
    """
    now = _now_iso()
    db.execute(
        """
        UPDATE pattern_ledger
        SET removed_at = ?,
            removal_reason = 'atlas_rebuild_reset',
            final_ledger = ledger,
            final_shadow_ledger = shadow_ledger
        WHERE removed_at IS NULL
        """,
        (now,),
    )
    for card in new_atlas.cards:
        for pattern in getattr(card, "generated_query_patterns", []):
            upsert_ledger(
                db=db,
                pattern_id=pattern_id_for(card.anchor_id, pattern),
                anchor_id=card.anchor_id,
                pattern_text=pattern,
                ledger_delta=0.0,
                shadow_delta=0.0,
                grace_days=grace_days,
            )
    db.commit()
