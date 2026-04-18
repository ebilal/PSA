"""
reload.py — marker file for long-lived pipeline consumers (MCP server).

Stage 2 writes ~/.psa/tenants/{tenant}/atlas_reload_requested atomically
after removing patterns from anchor_cards_refined.json. Long-lived
processes check the marker's mtime between queries and call
PSAPipeline.reload_atlas() when it advances.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone


def _marker_path(tenant_id: str) -> str:
    home = os.path.expanduser("~")
    return os.path.join(home, ".psa", "tenants", tenant_id, "atlas_reload_requested")


def write_reload_marker(*, tenant_id: str, changed_anchor_ids) -> None:
    """Atomically (tmp + rename) update the marker with current timestamp."""
    path = _marker_path(tenant_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "changed_anchor_ids": sorted(set(changed_anchor_ids)),
    }
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".reload-", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(body, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def marker_mtime(*, tenant_id: str) -> float:
    path = _marker_path(tenant_id)
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def should_reload(*, tenant_id: str, last_reload_mtime: float) -> bool:
    """True when the marker exists and its mtime is newer than last_reload_mtime."""
    m = marker_mtime(tenant_id=tenant_id)
    return m > last_reload_mtime
