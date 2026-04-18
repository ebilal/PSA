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


def build_parser(subparsers):
    """Register the `advertisement` subparser. Returns the parser for chaining."""
    p = subparsers.add_parser("advertisement", help="Advertisement decay CLIs")
    sub = p.add_subparsers(dest="advertisement_action", required=False)

    p_status = sub.add_parser("status", help="Ledger distribution + risk counts")
    p_status.add_argument("--tenant")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    return p


def dispatch(args) -> int:
    """Dispatch advertisement subcommands. Returns exit code."""
    func = getattr(args, "func", None)
    if func is None:
        return 1  # caller should print help
    return func(args)
