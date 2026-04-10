"""
inspect.py — PSA query inspection and observability.

Usage::

    from psa.inspect import inspect_query
    result = inspect_query("What auth pattern did we use?")
    print(result.render_brief())
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from .pipeline import PSAPipeline, QueryTiming

logger = logging.getLogger("psa.inspect")


@dataclass
class CandidateTrace:
    """Scored anchor candidate from the retriever."""
    anchor_id: int
    anchor_name: str
    bm25_score: float
    dense_score: float
    rrf_score: float
    selected: bool
    selector_score: float   # from SelectedAnchor if selected; rrf_score otherwise


@dataclass
class PackerSectionTrace:
    """One role section that made it into the packed context."""
    role: str               # "FAILURE WARNINGS", "EPISODIC", etc. or "RAW CONTEXT"
    memory_count: int
    token_cost: int
    items: List[str]        # actual text items packed


@dataclass
class InspectResult:
    """Full inspection result for one PSA query."""

    run_id: str
    query: str
    tenant_id: str
    context_text: str
    tokens_used: int
    token_budget: int
    sections: List[PackerSectionTrace]
    selected_anchor_ids: List[int]
    candidates: List[CandidateTrace]
    timing: QueryTiming
    timestamp: str

    def render_brief(self) -> str:
        """One-page summary suitable for terminal output."""
        lines = [
            f"Query: {self.query!r}",
            f"Tenant: {self.tenant_id}  |  Run: {self.run_id}",
            "",
            f"CONTEXT INJECTED ({self.tokens_used:,} tokens / {self.token_budget:,} budget)",
            "─" * 60,
            self.context_text,
            "─" * 60,
        ]
        selected_names = [c.anchor_name for c in self.candidates if c.selected]
        lines.append(
            f"Anchors selected ({len(self.selected_anchor_ids)}/{len(self.candidates)}): "
            + ", ".join(selected_names or ["none"])
        )
        t = self.timing
        lines.append(
            f"Timing: embed {t.embed_ms:.0f}ms | retrieve {t.retrieve_ms:.0f}ms | "
            f"select {t.select_ms:.0f}ms | fetch {t.fetch_ms:.0f}ms | "
            f"pack {t.pack_ms:.0f}ms | total {t.total_ms:.0f}ms"
        )
        return "\n".join(lines)

    def render_verbose(self) -> str:
        """Full trace: all candidates + scores, per-section breakdown."""
        lines = [self.render_brief(), ""]
        lines.append(f"ANCHOR CANDIDATES ({len(self.candidates)} total, {len(self.selected_anchor_ids)} selected)")
        for c in sorted(self.candidates, key=lambda x: x.rrf_score, reverse=True):
            mark = "Y" if c.selected else "N"
            lines.append(
                f"  [{mark}] {c.anchor_name:<40} "
                f"rrf={c.rrf_score:.2f}  bm25={c.bm25_score:.2f}  "
                f"dense={c.dense_score:.2f}  selector={c.selector_score:.2f}"
            )
        lines.append("")
        lines.append("PACKER SECTIONS")
        for sec in self.sections:
            lines.append(f"  {sec.role:<30} {sec.memory_count} item(s)  {sec.token_cost} tokens")
            for item in sec.items:
                preview = item[:120] + "..." if len(item) > 120 else item
                lines.append(f"    - {preview}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a JSON-serializable dict for the query log."""
        return {
            "run_id": self.run_id,
            "query": self.query,
            "tenant_id": self.tenant_id,
            "context_text": self.context_text,
            "tokens_used": self.tokens_used,
            "token_budget": self.token_budget,
            "sections": [
                {"role": s.role, "memory_count": s.memory_count,
                 "token_cost": s.token_cost, "items": s.items}
                for s in self.sections
            ],
            "selected_anchor_ids": self.selected_anchor_ids,
            "candidates": [
                {"anchor_id": c.anchor_id, "anchor_name": c.anchor_name,
                 "bm25_score": c.bm25_score, "dense_score": c.dense_score,
                 "rrf_score": c.rrf_score, "selected": c.selected,
                 "selector_score": c.selector_score}
                for c in self.candidates
            ],
            "timing": {
                "embed_ms": round(self.timing.embed_ms, 1),
                "retrieve_ms": round(self.timing.retrieve_ms, 1),
                "select_ms": round(self.timing.select_ms, 1),
                "fetch_ms": round(self.timing.fetch_ms, 1),
                "pack_ms": round(self.timing.pack_ms, 1),
                "total_ms": round(self.timing.total_ms, 1),
            },
            "timestamp": self.timestamp,
        }


def _run_id(query: str, now: Optional[datetime] = None) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%S")
    q_hash = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()[:6]
    return f"{ts}_{q_hash}"


def _log_path(tenant_id: str, base_dir: Optional[str] = None) -> str:
    if base_dir:
        tenant_dir = os.path.join(base_dir, "tenants", tenant_id)
    else:
        tenant_dir = os.path.expanduser(f"~/.psa/tenants/{tenant_id}")
    return os.path.join(tenant_dir, "query_log.jsonl")


def _append_log(result: "InspectResult", base_dir: Optional[str] = None) -> None:
    path = _log_path(result.tenant_id, base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict()) + "\n")


def inspect_query(
    query: str,
    tenant_id: str = "default",
    token_budget: int = 6000,
    base_dir: Optional[str] = None,
    write_log: bool = True,
) -> "InspectResult":
    """
    Run a PSA pipeline query and return a rich InspectResult.

    Parameters
    ----------
    query:
        The query string.
    tenant_id:
        PSA tenant to query (default: "default").
    token_budget:
        Token budget passed to the packer (default: 6000).
    base_dir:
        Override the PSA home directory (used in tests).
    write_log:
        If True, append the result to query_log.jsonl.

    Returns
    -------
    InspectResult with full trace including candidates, sections, and context text.

    Raises
    ------
    FileNotFoundError:
        If no atlas has been built for the tenant.
    """
    pipeline = PSAPipeline.from_tenant(
        tenant_id=tenant_id,
        token_budget=token_budget,
        base_dir=base_dir,
    )
    result = pipeline.query(query)

    selected_ids = {s.anchor_id for s in result.selected_anchors}
    selector_score_by_id = {s.anchor_id: s.selector_score for s in result.selected_anchors}

    candidates = [
        CandidateTrace(
            anchor_id=c.anchor_id,
            anchor_name=c.card.name,
            bm25_score=c.bm25_score,
            dense_score=c.dense_score,
            rrf_score=c.rrf_score,
            selected=c.anchor_id in selected_ids,
            selector_score=selector_score_by_id.get(c.anchor_id, 0.0),
        )
        for c in result.candidates
    ]

    sections = [
        PackerSectionTrace(
            role=sec.header,
            memory_count=len(sec.items),
            token_cost=sec.token_count,
            items=list(sec.items),
        )
        for sec in result.packed_context.sections
    ]

    now = datetime.now(timezone.utc)
    inspect_result = InspectResult(
        run_id=_run_id(query, now),
        query=query,
        tenant_id=tenant_id,
        context_text=result.text,
        tokens_used=result.token_count,
        token_budget=token_budget,
        sections=sections,
        selected_anchor_ids=list(selected_ids),
        candidates=candidates,
        timing=result.timing,
        timestamp=now.isoformat(),
    )

    if write_log:
        _append_log(inspect_result, base_dir)

    return inspect_result


def load_log(
    tenant_id: str = "default",
    base_dir: Optional[str] = None,
) -> List[dict]:
    """Load all entries from query_log.jsonl, newest first."""
    path = _log_path(tenant_id, base_dir)
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line in query log: %s", path)
    return list(reversed(entries))
