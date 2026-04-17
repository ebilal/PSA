"""
trace_reader.py — shared generator over query_trace.jsonl.

Origins contract (locked in the spec):
    - origins=None          → no filter; every record yielded.
    - origins=set[str]      → only records whose query_origin is in the set.
    - origins=set()         → filter-everything-out; yields nothing.

The CLI passes origins={"interactive"} as its own default. The library
function stays neutral — it doesn't assume any default.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Iterator, Optional

logger = logging.getLogger("psa.diag.trace_reader")


def iter_trace_records(
    tenant_id: str,
    *,
    origins: Optional[set[str]] = None,
) -> Iterator[dict]:
    """Yield trace records for `tenant_id`, optionally filtered by origin."""
    home = os.path.expanduser("~")
    path = os.path.join(home, ".psa", "tenants", tenant_id, "query_trace.jsonl")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.debug("Skipping malformed trace line: %s", e)
                continue
            if origins is None:
                yield record
            elif record.get("query_origin", "interactive") in origins:
                yield record
