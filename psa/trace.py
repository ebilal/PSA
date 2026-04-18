"""
trace.py — per-query trace writer for `psa diag` rollups.

Every `pipeline.query()` call emits one JSONL record to
~/.psa/tenants/{tenant_id}/query_trace.jsonl unless disabled.

Disable via:
    - PSA_TRACE=0 in environment, or
    - trace_queries=false in ~/.psa/config.json

The writer is side-effect-only: no in-memory buffering, no rotation.
Records are small (~1-3 KB) and the file is a simple append target.

Query text is stored in plaintext for MVP. Redaction/hashing mode is
a follow-up; the writer will gain a redactor hook when it lands.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("psa.trace")


def _config_trace_enabled() -> bool:
    """Read trace_queries from ~/.psa/config.json (default True)."""
    home = os.path.expanduser("~")
    cfg_path = os.path.join(home, ".psa", "config.json")
    if not os.path.exists(cfg_path):
        return True
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        value = cfg.get("trace_queries", True)
        return bool(value)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read trace_queries from %s: %s", cfg_path, e)
        return True


def _trace_disabled() -> bool:
    """Combined env + config gate. Env wins."""
    env = os.environ.get("PSA_TRACE")
    if env is not None:
        return env == "0"
    return not _config_trace_enabled()


def write_trace(record: dict[str, Any], *, tenant_id: str) -> bool:
    """Append one JSONL record. Returns True on success, False on failure or when tracing is disabled."""
    if _trace_disabled():
        return False
    home = os.path.expanduser("~")
    path = os.path.join(home, ".psa", "tenants", tenant_id, "query_trace.jsonl")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return True
    except OSError as e:
        logger.warning("Could not write trace to %s: %s", path, e)
        return False


def new_trace_record(
    *,
    run_id: str,
    timestamp: str,
    tenant_id: str,
    atlas_version: int,
    query: str,
    query_origin: str = "interactive",
) -> dict[str, Any]:
    """Seed a trace record with the fields known at query start.

    Callers populate the rest (selection_mode, result_kind,
    top_anchor_scores, selected_anchor_ids, packed_memories,
    tokens_used, token_budget, timing_ms) as pipeline processing
    proceeds.
    """
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "tenant_id": tenant_id,
        "atlas_version": atlas_version,
        "query": query,
        "query_origin": query_origin,
        "selection_mode": None,
        "result_kind": None,
        "top_anchor_scores": [],
        "selected_anchor_ids": [],
        "empty_selection": False,
        "packed_memories": [],
        "tokens_used": 0,
        "token_budget": 0,
        "timing_ms": {},
        "retrieval_attribution": [],
    }
