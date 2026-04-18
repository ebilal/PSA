"""
cli.py — psa advertisement subcommands.

  status          Ledger distribution + shadow-disagreement counts.
  diff            B-vs-A counterfactual report.
  rebuild-ledger  Canonical regenerator (reads trace + atlas).
  purge           Hard-delete archived rows past retention.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone


def _db_path(tenant_id: str) -> str:
    return os.path.expanduser(
        f"~/.psa/tenants/{tenant_id}/memory.sqlite3"
    )


def _histogram(values, bins: int = 20):
    if not values:
        return {"bins": [], "counts": []}
    lo, hi = min(values), max(values)
    if lo == hi:
        return {"bins": [lo, hi], "counts": [len(values)]}
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    return {"bins": edges, "counts": counts}


def cmd_status(args) -> int:
    tenant_id = getattr(args, "tenant", None) or "default"
    as_json = getattr(args, "json", False)

    db_path = _db_path(tenant_id)
    if not os.path.exists(db_path):
        if as_json:
            print(json.dumps({"tenant_id": tenant_id, "n_active": 0, "histogram": {"bins": [], "counts": []}}))
        else:
            print(f"No ledger DB at {db_path}")
        return 0
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            """
            SELECT pattern_id, anchor_id, pattern_text, ledger,
                   shadow_ledger, consecutive_negative_cycles,
                   shadow_consecutive, grace_expires_at
            FROM pattern_ledger
            WHERE removed_at IS NULL
            """
        ).fetchall()
    now_iso = datetime.now(timezone.utc).isoformat()
    in_grace = sum(1 for r in rows if r[7] > now_iso)
    at_risk = sum(1 for r in rows if r[5] >= 7)
    shadow_only_risk = sum(
        1 for r in rows if r[6] >= 7 and r[5] < 14
    )
    values = [r[3] for r in rows]
    hist = _histogram(values)
    data = {
        "tenant_id": tenant_id,
        "n_active": len(rows),
        "n_in_grace": in_grace,
        "n_at_risk": at_risk,
        "n_shadow_only_at_risk": shadow_only_risk,
        "histogram": hist,
    }
    if as_json:
        print(json.dumps(data, indent=2))
        return 0
    print(f"tenant: {tenant_id}")
    print(f"  active patterns:         {data['n_active']}")
    print(f"  in grace:                {data['n_in_grace']}")
    print(f"  at risk (cnc >= 7):      {data['n_at_risk']}")
    print(f"  shadow-only at risk:     {data['n_shadow_only_at_risk']}")
    return 0


def cmd_diff(args) -> int:
    tenant_id = getattr(args, "tenant", None) or "default"
    as_json = getattr(args, "json", False)

    db_path = _db_path(tenant_id)
    if not os.path.exists(db_path):
        if as_json:
            print(json.dumps({"tenant_id": tenant_id, "disagreements": []}))
        else:
            print(f"No ledger DB at {db_path}")
        return 0
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            """
            SELECT pattern_id, anchor_id, pattern_text, ledger,
                   consecutive_negative_cycles, shadow_ledger, shadow_consecutive
            FROM pattern_ledger
            WHERE removed_at IS NULL
            """
        ).fetchall()
    sustained_B, sustained_A = 14, 7
    B_would = {r[0] for r in rows if r[4] >= sustained_B}
    A_would = {r[0] for r in rows if r[6] >= sustained_A}
    only_A = A_would - B_would
    only_B = B_would - A_would
    agree = A_would & B_would
    disagreements = [
        {
            "pattern_id": r[0],
            "anchor_id": r[1],
            "pattern_text": r[2],
            "ledger": r[3],
            "consecutive_negative_cycles": r[4],
            "shadow_ledger": r[5],
            "shadow_consecutive": r[6],
            "B_would_remove": r[0] in B_would,
            "A_would_remove": r[0] in A_would,
        }
        for r in rows
        if r[0] in only_A or r[0] in only_B
    ]
    data = {
        "tenant_id": tenant_id,
        "both_would_remove": len(agree),
        "shadow_only_would_remove": len(only_A),
        "primary_only_would_remove": len(only_B),
        "disagreements": disagreements,
    }
    if as_json:
        print(json.dumps(data, indent=2))
        return 0
    print(f"tenant: {tenant_id}")
    print(f"  both (B AND A) would remove:  {data['both_would_remove']}")
    print(f"  shadow-only (A not B):        {data['shadow_only_would_remove']}")
    print(f"  primary-only (B not A):       {data['primary_only_would_remove']}")
    for d in disagreements[:20]:
        print(
            f"    - anchor={d['anchor_id']} "
            f"B={'Y' if d['B_would_remove'] else 'N'} "
            f"A={'Y' if d['A_would_remove'] else 'N'} "
            f"cnc={d['consecutive_negative_cycles']} "
            f"scc={d['shadow_consecutive']} "
            f"{d['pattern_text'][:60]!r}"
        )
    return 0


def cmd_rebuild_ledger(args) -> int:
    import math

    from datetime import timedelta

    tenant_id = getattr(args, "tenant", None) or "default"
    dry_run = getattr(args, "dry_run", False)
    as_json = getattr(args, "json", False)

    trace_path = os.path.expanduser(
        f"~/.psa/tenants/{tenant_id}/query_trace.jsonl"
    )
    if not os.path.exists(trace_path):
        print(f"No trace at {trace_path}")
        return 1

    from psa.advertisement.config import AdvertisementDecayConfig
    from psa.advertisement.ledger import pattern_id_for
    from psa.config import MempalaceConfig

    ad = AdvertisementDecayConfig.from_mempalace(MempalaceConfig())

    records = 0
    derived: dict[str, dict] = {}

    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records += 1
            attrs = rec.get("retrieval_attribution") or []
            selected = set(rec.get("selected_anchor_ids") or [])
            ts_str = rec.get("timestamp")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue
            age_days = max(0, (datetime.now(timezone.utc) - ts).days)
            decay = math.exp(-age_days / ad.tau_days)
            for a in attrs:
                if not a.get("bm25_floor_passed"):
                    continue
                argmax = a.get("bm25_argmax_pattern")
                tied = a.get("bm25_epsilon_tied") or []
                credited = [argmax] + list(tied) if argmax else []
                if not credited:
                    continue
                per = 1.0 / len(credited)
                base = ad.retrieval_credit
                if a["anchor_id"] in selected:
                    base += ad.selector_pick_credit
                else:
                    base -= ad.selector_decline_penalty
                for pat in credited:
                    key = pattern_id_for(a["anchor_id"], pat)
                    slot = derived.setdefault(
                        key,
                        {"anchor_id": a["anchor_id"], "pattern_text": pat, "ledger": 0.0},
                    )
                    slot["ledger"] += base * per * decay

    data = {
        "tenant_id": tenant_id,
        "records_processed": records,
        "derived_patterns": len(derived),
    }
    if dry_run:
        if as_json:
            print(json.dumps(data, indent=2))
        else:
            print(
                f"dry-run: processed {records} records, "
                f"derived {len(derived)} patterns"
            )
        return 0

    # Non-dry: write derived values into the table.
    from psa.advertisement.ledger import create_schema
    db_path = _db_path(tenant_id)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    grace_iso = (datetime.now(timezone.utc) + timedelta(days=ad.grace_days)).isoformat()
    with sqlite3.connect(db_path) as db:
        create_schema(db)
        for pid, slot in derived.items():
            db.execute(
                """
                INSERT INTO pattern_ledger
                    (pattern_id, anchor_id, pattern_text,
                     ledger, shadow_ledger,
                     grace_expires_at, created_at, last_updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pattern_id) DO UPDATE SET
                    ledger = excluded.ledger,
                    last_updated_at = excluded.last_updated_at
                """,
                (
                    pid,
                    slot["anchor_id"],
                    slot["pattern_text"],
                    slot["ledger"],
                    slot["ledger"],
                    grace_iso,
                    now_iso,
                    now_iso,
                ),
            )
        db.commit()

    if as_json:
        print(json.dumps(data, indent=2))
    else:
        print(f"Rebuilt ledger: {len(derived)} patterns from {records} records")
    return 0


def build_parser(subparsers):
    """Register the `advertisement` subparser. Returns the parser for chaining."""
    p = subparsers.add_parser("advertisement", help="Advertisement decay CLIs")
    sub = p.add_subparsers(dest="advertisement_action", required=False)

    p_status = sub.add_parser("status", help="Ledger distribution + risk counts")
    p_status.add_argument("--tenant")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    p_diff = sub.add_parser("diff", help="B vs A counterfactual report")
    p_diff.add_argument("--tenant")
    p_diff.add_argument("--json", action="store_true")
    p_diff.set_defaults(func=cmd_diff)

    p_rebuild = sub.add_parser("rebuild-ledger", help="Recompute ledger from trace")
    p_rebuild.add_argument("--tenant")
    p_rebuild.add_argument("--dry-run", action="store_true")
    p_rebuild.add_argument("--json", action="store_true")
    p_rebuild.set_defaults(func=cmd_rebuild_ledger)

    return p


def dispatch(args) -> int:
    """Dispatch advertisement subcommands. Returns exit code."""
    func = getattr(args, "func", None)
    if func is None:
        return 1  # caller should print help
    return func(args)
