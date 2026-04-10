# PSA Observability + LongMemEval Benchmarking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `psa inspect` observability CLI (Part 1) and a LongMemEval benchmark harness (Part 2) to the PSA system.

**Architecture:** Part 1 wraps the existing `PSAPipeline.query()` in a new `psa/inspect.py` module that formats it for human inspection and logs it to JSONL. Part 2 adds `psa/benchmarks/longmemeval.py` that ingests HuggingFace sessions, runs PSA, scores answers, and writes oracle labels in the existing `oracle_labels.jsonl` format.

**Tech Stack:** Python 3.9+, existing PSA pipeline, `datasets` (HuggingFace, optional), `uv run pytest`, `uv run ruff`.

**Key facts:**
- `PackedContext` already has `sections: List[PackedSection]` — no packer change needed.
- `PSAResult.candidates` = all 24 `AnchorCandidate` objects (each has `dense_score`, `bm25_score`, `rrf_score`).
- `PSAResult.selected_anchors` = `List[SelectedAnchor]` — each has `anchor_id` and `selector_score`.
- Tests live in flat `tests/` directory. Use `MagicMock(spec=ClassName)` for PSA objects.
- `conftest.py` redirects HOME to tmp dir — tests never touch real `~/.psa`.
- Run tests: `uv run pytest tests/ -v`


---

## Part 1 — Observability

### Task 1: Create `psa/inspect.py` — data model and `inspect_query()`

**Files:**
- Create: `psa/inspect.py`

- [ ] **Step 1: Write the file**

```python
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from .pipeline import PSAPipeline, QueryTiming
from .packer import PackedSection

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


def _run_id(query: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    q_hash = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()[:6]
    return f"{ts}_{q_hash}"


def _log_path(tenant_id: str, base_dir: Optional[str] = None) -> str:
    if base_dir:
        tenant_dir = os.path.join(base_dir, "tenants", tenant_id)
    else:
        tenant_dir = os.path.expanduser(f"~/.psa/tenants/{tenant_id}")
    os.makedirs(tenant_dir, exist_ok=True)
    return os.path.join(tenant_dir, "query_log.jsonl")


def _append_log(result: "InspectResult", base_dir: Optional[str] = None) -> None:
    path = _log_path(result.tenant_id, base_dir)
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

    inspect_result = InspectResult(
        run_id=_run_id(query),
        query=query,
        tenant_id=tenant_id,
        context_text=result.text,
        tokens_used=result.token_count,
        token_budget=token_budget,
        sections=sections,
        selected_anchor_ids=list(selected_ids),
        candidates=candidates,
        timing=result.timing,
        timestamp=datetime.now(timezone.utc).isoformat(),
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
                    pass
    return list(reversed(entries))
```

- [ ] **Step 2: Verify import works**

```bash
uv run python -c "from psa.inspect import inspect_query, load_log; print('OK')"
```

Expected: `OK`


---

### Task 2: Write tests for `psa/inspect.py`

**Files:**
- Create: `tests/test_inspect.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for psa.inspect — InspectResult construction and query_log."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from psa.anchor import AnchorCard, AnchorIndex
from psa.atlas import Atlas
from psa.embeddings import EmbeddingModel
from psa.inspect import (
    CandidateTrace,
    InspectResult,
    PackerSectionTrace,
    _run_id,
    inspect_query,
    load_log,
)
from psa.memory_object import MemoryStore
from psa.pipeline import PSAPipeline, QueryTiming
from psa.retriever import AnchorCandidate
from psa.selector import AnchorSelector, SelectedAnchor


def _make_card(anchor_id: int, name: str = "auth") -> AnchorCard:
    return AnchorCard(
        anchor_id=anchor_id,
        name=name,
        meaning=f"Anchor about {name}",
        memory_types=["EPISODIC"],
        include_terms=[name],
        exclude_terms=[],
        prototype_examples=[],
        near_but_different=[],
        centroid=[1.0, 0.0],
        memory_count=2,
        is_novelty=False,
    )


def _make_pipeline(tmp_path):
    card = _make_card(1)
    index = AnchorIndex(dim=2)
    card.centroid = [1.0, 0.0]
    index.build([card])

    atlas = MagicMock(spec=Atlas)
    atlas.cards = [card]
    atlas.anchor_index = index
    atlas.version = 1

    store = MagicMock(spec=MemoryStore)
    store.query_by_anchor.return_value = []

    em = MagicMock(spec=EmbeddingModel)
    em.embed.return_value = [1.0, 0.0]

    return PSAPipeline(
        store=store,
        atlas=atlas,
        embedding_model=em,
        selector=AnchorSelector.cosine(max_k=2),
        token_budget=1000,
        tenant_id="test",
    )


def test_run_id_format():
    rid = _run_id("my query")
    parts = rid.split("_")
    assert len(parts) == 2
    assert len(parts[0]) == 15   # YYYYMMDDTHHmmSS
    assert len(parts[1]) == 6


def test_run_id_same_hash_for_same_query():
    rid1 = _run_id("hello world")
    rid2 = _run_id("hello world")
    assert rid1.split("_")[1] == rid2.split("_")[1]


def test_inspect_query_returns_inspect_result(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        result = inspect_query("test query", base_dir=str(tmp_path), write_log=False)
    assert isinstance(result, InspectResult)
    assert result.query == "test query"
    assert result.tenant_id == "default"


def test_inspect_result_has_context_text(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        result = inspect_query("test query", base_dir=str(tmp_path), write_log=False)
    assert isinstance(result.context_text, str)
    assert len(result.context_text) > 0


def test_inspect_result_selected_is_subset_of_candidates(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        result = inspect_query("test query", base_dir=str(tmp_path), write_log=False)
    selected_ids = set(result.selected_anchor_ids)
    candidate_ids = {c.anchor_id for c in result.candidates}
    assert selected_ids.issubset(candidate_ids)


def test_inspect_result_to_dict_has_required_keys(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        result = inspect_query("test query", base_dir=str(tmp_path), write_log=False)
    d = result.to_dict()
    for key in ("run_id", "query", "tenant_id", "context_text", "tokens_used",
                 "token_budget", "sections", "selected_anchor_ids", "candidates", "timing"):
        assert key in d, f"Missing key: {key}"


def test_render_brief_contains_query(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        result = inspect_query("auth query", base_dir=str(tmp_path), write_log=False)
    brief = result.render_brief()
    assert "auth query" in brief
    assert "tokens" in brief.lower()


def test_render_verbose_contains_candidates_section(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        result = inspect_query("auth query", base_dir=str(tmp_path), write_log=False)
    verbose = result.render_verbose()
    assert "ANCHOR CANDIDATES" in verbose


def test_inspect_writes_to_log(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        inspect_query("log test query", base_dir=str(tmp_path), write_log=True)
    log_path = tmp_path / "tenants" / "default" / "query_log.jsonl"
    assert log_path.exists()
    line = log_path.read_text().strip()
    entry = json.loads(line)
    assert entry["query"] == "log test query"


def test_load_log_returns_newest_first(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        inspect_query("first query", base_dir=str(tmp_path), write_log=True)
        inspect_query("second query", base_dir=str(tmp_path), write_log=True)
    entries = load_log(tenant_id="default", base_dir=str(tmp_path))
    assert len(entries) == 2
    assert entries[0]["query"] == "second query"
    assert entries[1]["query"] == "first query"


def test_load_log_empty_if_no_file(tmp_path):
    entries = load_log(tenant_id="no_such_tenant", base_dir=str(tmp_path))
    assert entries == []
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/test_inspect.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add psa/inspect.py tests/test_inspect.py
git commit -m "feat: add psa/inspect.py — InspectResult + inspect_query() + query log"
```


---

### Task 3: Add `psa inspect` and `psa log` CLI commands

**Files:**
- Modify: `psa/cli.py`

- [ ] **Step 1: Add `cmd_inspect()` and `cmd_log()` functions**

Add the following functions to `psa/cli.py` immediately before `cmd_benchmark` (around line 651):

```python
def cmd_inspect(args):
    """Inspect what context PSA injects for a query."""
    from .inspect import inspect_query

    query = args.query
    tenant_id = getattr(args, "tenant", "default")
    token_budget = getattr(args, "token_budget", 6000)
    verbose = getattr(args, "verbose", False)

    try:
        result = inspect_query(query, tenant_id=tenant_id, token_budget=token_budget)
        if verbose:
            print(result.render_verbose())
        else:
            print(result.render_brief())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'psa atlas build' to build an atlas first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def _print_log_diff(old: dict, new: dict) -> None:
    """Print a human-readable diff between two log entries."""
    print(f"DIFF: {old['run_id']} -> {new['run_id']}")
    print(f"Query: {new['query']!r}")
    old_tok = old.get("tokens_used", 0)
    new_tok = new.get("tokens_used", 0)
    print(f"Tokens: {old_tok} -> {new_tok} ({new_tok - old_tok:+d})")
    old_sel = set(old.get("selected_anchor_ids", []))
    new_sel = set(new.get("selected_anchor_ids", []))
    added = new_sel - old_sel
    removed = old_sel - new_sel
    if added:
        print(f"Anchors added:   {sorted(added)}")
    if removed:
        print(f"Anchors removed: {sorted(removed)}")
    if not added and not removed:
        print("Anchors: unchanged")


def cmd_log(args):
    """Manage the PSA query inspection log."""
    from .inspect import load_log

    tenant_id = getattr(args, "tenant", "default")
    action = getattr(args, "log_action", None)

    if action == "list" or action is None:
        n = getattr(args, "n", 20)
        entries = load_log(tenant_id=tenant_id)
        if not entries:
            print(f"No log entries for tenant '{tenant_id}'.")
            return
        print(f"Recent queries for tenant '{tenant_id}' (newest first):")
        for e in entries[:n]:
            ts = e.get("timestamp", "?")[:19].replace("T", " ")
            tokens = e.get("tokens_used", 0)
            budget = e.get("token_budget", 0)
            run_id = e.get("run_id", "?")
            query = e.get("query", "?")
            print(f"  [{ts}] {run_id}  {tokens}/{budget} tok  {query!r}")

    elif action == "show":
        run_id = args.run_id
        entries = load_log(tenant_id=tenant_id)
        match = next((e for e in entries if e.get("run_id") == run_id), None)
        if match is None:
            print(f"No log entry with run_id '{run_id}'.")
            sys.exit(1)
        import json as _json
        print(_json.dumps(match, indent=2))

    elif action == "diff":
        entries = load_log(tenant_id=tenant_id)
        query_filter = getattr(args, "query", None)
        run_id_a = getattr(args, "run_id_a", None)
        run_id_b = getattr(args, "run_id_b", None)

        if query_filter:
            matches = [e for e in entries if e.get("query") == query_filter]
            if len(matches) < 2:
                print(f"Need at least 2 log entries for query {query_filter!r}. Found {len(matches)}.")
                sys.exit(1)
            e_new, e_old = matches[0], matches[1]
        elif run_id_a and run_id_b:
            e_old = next((e for e in entries if e.get("run_id") == run_id_a), None)
            e_new = next((e for e in entries if e.get("run_id") == run_id_b), None)
            if not e_old or not e_new:
                print("Could not find both run IDs.")
                sys.exit(1)
        else:
            print("Provide --query or two run IDs.")
            sys.exit(1)

        _print_log_diff(e_old, e_new)
```

- [ ] **Step 2: Add argparser entries for `inspect` and `log`**

Find the `# benchmark` comment block (around line 1036) in the argparser. Add these two blocks immediately before it:

```python
    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect what context PSA injects for a query")
    p_inspect.add_argument("query", help="Query string to inspect")
    p_inspect.add_argument("--tenant", default="default")
    p_inspect.add_argument("--token-budget", dest="token_budget", type=int, default=6000)
    p_inspect.add_argument("--verbose", action="store_true", help="Show full trace with all candidates")

    # log
    p_log = sub.add_parser("log", help="Manage the PSA query inspection log")
    p_log.add_argument("--tenant", default="default")
    log_sub = p_log.add_subparsers(dest="log_action")
    p_log_list = log_sub.add_parser("list", help="List recent logged queries")
    p_log_list.add_argument("-n", type=int, default=20)
    p_log_show = log_sub.add_parser("show", help="Show a log entry by run ID")
    p_log_show.add_argument("run_id")
    p_log_diff = log_sub.add_parser("diff", help="Diff two log entries")
    p_log_diff.add_argument("--query", default=None)
    p_log_diff.add_argument("run_id_a", nargs="?", default=None)
    p_log_diff.add_argument("run_id_b", nargs="?", default=None)
```

- [ ] **Step 3: Add `inspect` and `log` to the dispatch dict**

Find the dispatch dict near the bottom of `main()` (around line 1149) and add the two new commands:

```python
    dispatch = {
        "init": cmd_init,
        "mine": cmd_mine,
        "split": cmd_split,
        "search": cmd_search,
        "wake-up": cmd_wakeup,
        "repair": cmd_repair,
        "status": cmd_status,
        "inspect": cmd_inspect,
        "log": cmd_log,
        "benchmark": cmd_benchmark,
        "migrate": cmd_migrate,
    }
```

- [ ] **Step 4: Verify CLI help works**

```bash
uv run python -m psa inspect --help
uv run python -m psa log --help
```

Expected: both print help without errors.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -v --ignore=tests/test_llm_integration.py -x
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add psa/cli.py
git commit -m "feat: add psa inspect + psa log CLI commands"
```

---

### Task 4: Update README files

**Files:**
- Modify: `README.md` (top-level)
- Modify: `psa/README.md` (replace entirely)

- [ ] **Step 1: Add MCP Server and Observability sections to `README.md`**

Find the existing `## MCP Server` section in `README.md`. If it exists, extend it; if not, add after the Commands section. The content to add/replace:

```markdown
## MCP Server — Claude CLI Integration

PSA integrates with Claude Code as a memory server via MCP. When Claude calls
`psa_atlas_search`, it receives a formatted context block drawn from your typed
memory objects — identical to what `psa inspect` shows you.

### Install

```
claude mcp add psa -- python -m psa.mcp_server
```

This registers PSA as a persistent MCP server. Claude calls `psa_status` on
session start, which returns the PALACE_PROTOCOL instructing Claude to call
`psa_atlas_search` before answering about any past decision or event.

### What Claude receives

`psa_atlas_search` runs the full pipeline and returns formatted context like:

```
PSA MEMORY CONTEXT — what auth pattern did we use?
============================================================

FAILURE WARNINGS
- Auth token stored in localStorage — caused XSS exposure in March

PROCEDURAL GUIDANCE
- Always validate JWT expiry server-side before trusting claims

FACTS & CONCEPTS
- RS256 signed JWTs with 15-minute expiry and refresh tokens
```

### Available MCP Tools

| Tool | Purpose |
|------|---------|
| `psa_atlas_search` | Full pipeline query — retrieves and packs relevant memories |
| `psa_store_memory` | Store a typed memory object directly |
| `psa_status` | Palace overview + wake-up protocol |
| `psa_atlas_status` | Atlas version, anchor count, memory count |
| `psa_atlas_health` | Health report (novelty rate, utilization skew) |
| `psa_list_anchors` | All anchors with utilization stats |
| `psa_rebuild_atlas` | Trigger atlas rebuild |
| `psa_search` | Legacy ChromaDB semantic search |
| `psa_add_drawer` | Store raw content in the legacy palace |

## Observability — Inspect What Claude Sees

See exactly what context PSA injects into Claude's context window for any query:

```bash
psa inspect "what auth pattern did we use?"
```

Output:
```
Query: 'what auth pattern did we use?'
Tenant: default  |  Run: 20260410T143201_a3f91c

CONTEXT INJECTED (847 tokens / 6,000 budget)
────────────────────────────────────────────────────────────
PSA MEMORY CONTEXT — what auth pattern did we use?
...
────────────────────────────────────────────────────────────
Anchors selected (2/24): auth-jwt-pattern, session-security-incidents
Timing: embed 12ms | retrieve 34ms | select 8ms | fetch 5ms | pack 3ms | total 62ms
```

Add `--verbose` to see all 24 anchor candidates with BM25/dense/selector scores.

Every `psa inspect` call is logged to `~/.psa/tenants/{tenant_id}/query_log.jsonl`:

```bash
psa log list                              # recent queries
psa log show 20260410T143201_a3f91c       # full entry as JSON
psa log diff --query "auth pattern"       # diff last 2 runs for same query
```
```

- [ ] **Step 2: Replace `psa/README.md` with accurate module table**

Overwrite `psa/README.md` with:

```markdown
# psa/ — Core Package

The Python package that powers PSA.

## Modules

| Module | What it does |
|--------|-------------|
| `cli.py` | CLI entry point — routes mine, search, inspect, atlas, train, benchmark |
| `config.py` | Configuration — env vars, `~/.psa/config.json`, defaults |
| `memory_object.py` | `MemoryObject` dataclass and `MemoryStore` SQLite backend |
| `pipeline.py` | `PSAPipeline` — retriever → selector → packer |
| `retriever.py` | `AnchorRetriever` — BM25 + dense hybrid, RRF fusion |
| `selector.py` | `AnchorSelector` — cosine baseline or trained cross-encoder |
| `packer.py` | `EvidencePacker` — role-organized sections under token budget |
| `atlas.py` | `AtlasBuilder` + `AtlasManager` — spherical k-means clustering |
| `anchor.py` | `AnchorCard` + `AnchorIndex` — FAISS index over anchor centroids |
| `embeddings.py` | `EmbeddingModel` — BAAI/bge-base-en-v1.5, 768-dim, L2-normalized |
| `inspect.py` | `inspect_query()` — full pipeline trace + query log for observability |
| `mcp_server.py` | MCP server for Claude and other MCP-compatible agents |
| `miner.py` | Project file ingest — scans dirs, chunks, stores to SQLite |
| `convo_miner.py` | Conversation ingest — chunks by exchange pair, stores to SQLite |
| `consolidation.py` | LLM-driven memory extraction — chunks, calls LLM, filters, deduplicates |
| `normalize.py` | Converts chat formats (Claude Code, Claude.ai, ChatGPT, Slack, plain text) |
| `searcher.py` | ChromaDB semantic search (legacy palace path) |
| `tenant.py` | Tenant directory management |
| `llm.py` | Unified LLM caller — local (Ollama) + cloud (LiteLLM) with fallback |
| `entity_registry.py` | Entity code registry |
| `entity_detector.py` | Auto-detects people and projects from file content |
| `health.py` | Atlas health checks |
| `lifecycle.py` | Nightly lifecycle pipeline — label, train, rebuild |
| `forgetting.py` | Retention scoring and memory expiry |
| `spellcheck.py` | Name-aware spellcheck |
| `split_mega_files.py` | Splits concatenated transcript files |
| `migrate.py` | Migrates ChromaDB palace to PSA MemoryStore |
| `hooks_cli.py` | Claude Code hook handlers |
| `version.py` | Package version |

## Sub-packages

| Sub-package | What it does |
|-------------|-------------|
| `training/` | Selector training: oracle labeler, data generator, train script |
| `benchmarks/` | Benchmark harnesses: LongMemEval ingest / run / score |
```

- [ ] **Step 3: Run lint**

```bash
uv run ruff check psa/cli.py psa/inspect.py
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add README.md psa/README.md
git commit -m "docs: update README with MCP server integration and observability sections"
```


---

## Part 2 — LongMemEval Benchmarking

### Task 5: Create `psa/benchmarks/` package and `longmemeval.py`

**Files:**
- Create: `psa/benchmarks/__init__.py`
- Create: `psa/benchmarks/longmemeval.py`

- [ ] **Step 1: Create `psa/benchmarks/__init__.py`**

```python
"""PSA benchmark harnesses."""
```

- [ ] **Step 2: Write `psa/benchmarks/longmemeval.py`**

```python
"""
longmemeval.py — LongMemEval benchmark harness for PSA.

Three sub-commands:

  psa benchmark longmemeval ingest   — download dataset, mine sessions, build atlas
  psa benchmark longmemeval run      — query PSA for each question, generate answers
  psa benchmark longmemeval score    — score answers, write oracle labels

Results: ~/.psa/benchmarks/longmemeval/
Oracle labels: existing oracle_labels.jsonl format, picked up by 'psa train'.

Benchmarks run against an isolated tenant (default: 'longmemeval_bench').
"""

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("psa.benchmarks.longmemeval")

BENCH_TENANT = "longmemeval_bench"
HF_DATASET = "xiaowu0162/longmemeval"
RESULTS_DIR_DEFAULT = os.path.expanduser("~/.psa/benchmarks/longmemeval")


# ── Ingest ─────────────────────────────────────────────────────────────────────


def ingest(tenant_id: str = BENCH_TENANT, results_dir: str = RESULTS_DIR_DEFAULT) -> None:
    """Download LongMemEval and ingest all sessions into PSA."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for LongMemEval ingestion.\n"
            "Install with: pip install datasets"
        )

    from ..convo_miner import ConvoMiner
    from ..tenant import TenantManager

    logger.info("Loading LongMemEval dataset from HuggingFace...")
    ds = load_dataset(HF_DATASET, split="train")

    sessions: Dict[str, List[Dict]] = {}
    for example in ds:
        for session_id, messages in zip(
            example.get("session_ids", []), example.get("sessions", [])
        ):
            if session_id not in sessions:
                sessions[session_id] = messages

    logger.info("Found %d unique sessions to ingest.", len(sessions))
    os.makedirs(results_dir, exist_ok=True)

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        for session_id, messages in sessions.items():
            session_path = os.path.join(tmpdir, f"{session_id}.jsonl")
            _write_session_jsonl(session_path, session_id, messages)

        miner = ConvoMiner(tenant_id=tenant_id, use_llm=True, tenant_dir=tenant.root_dir)
        miner.mine_directory(tmpdir)

    logger.info("Ingestion complete. Building atlas...")
    _build_atlas(tenant_id)
    logger.info("Done. Tenant '%s' is ready for benchmarking.", tenant_id)


def _write_session_jsonl(path: str, session_id: str, messages: List[Dict]) -> None:
    """Write one session as Claude Code JSONL format for convo_miner."""
    with open(path, "w", encoding="utf-8") as f:
        for msg in messages:
            record = {
                "type": "message",
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
                "session_id": session_id,
            }
            f.write(json.dumps(record) + "\n")


def _build_atlas(tenant_id: str) -> None:
    from ..atlas import AtlasBuilder, AtlasManager
    from ..memory_object import MemoryStore
    from ..tenant import TenantManager
    from ..embeddings import EmbeddingModel

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    store = MemoryStore(tenant.memory_db_path)
    em = EmbeddingModel()
    mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
    builder = AtlasBuilder(store=store, embedding_model=em, tenant_id=tenant_id)
    atlas = builder.build_atlas()
    mgr.save_atlas(atlas)
    logger.info("Atlas built: version %s, %d anchors.", atlas.version, len(atlas.cards))


# ── Run ────────────────────────────────────────────────────────────────────────


def run(
    split: str = "val",
    limit: Optional[int] = None,
    tenant_id: str = BENCH_TENANT,
    results_dir: str = RESULTS_DIR_DEFAULT,
    token_budget: int = 6000,
) -> str:
    """
    Run each LongMemEval question through PSA and generate answers.

    Returns the path to the results JSONL file.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    from ..pipeline import PSAPipeline
    from ..llm import call_llm

    logger.info("Loading LongMemEval dataset (split=%s)...", split)
    ds = load_dataset(HF_DATASET, split="train")
    examples = [e for e in ds if e.get("split", "val") == split]
    if limit:
        examples = examples[:limit]

    logger.info("Running %d questions (tenant=%s)...", len(examples), tenant_id)

    try:
        pipeline = PSAPipeline.from_tenant(tenant_id=tenant_id, token_budget=token_budget)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No atlas for tenant '{tenant_id}'. Run 'psa benchmark longmemeval ingest' first."
        )

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(results_dir, f"results_{split}_{ts}.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(examples):
            question_id = example.get("question_id", f"q_{i:04d}")
            question = example.get("question", "")
            gold_answer = example.get("answer", "")

            result = pipeline.query(question)
            context_text = result.text

            prompt = (
                f"Answer the following question based only on the provided context.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            generated = call_llm(prompt, max_tokens=256)

            record = {
                "question_id": question_id,
                "question": question,
                "context_text": context_text,
                "answer_generated": generated,
                "answer_gold": gold_answer,
                "tokens_used": result.token_count,
                "token_budget": token_budget,
                "selected_anchor_ids": [a.anchor_id for a in result.selected_anchors],
                "timing_ms": {
                    "embed": round(result.timing.embed_ms, 1),
                    "retrieve": round(result.timing.retrieve_ms, 1),
                    "select": round(result.timing.select_ms, 1),
                    "fetch": round(result.timing.fetch_ms, 1),
                    "pack": round(result.timing.pack_ms, 1),
                    "total": round(result.timing.total_ms, 1),
                },
            }
            f.write(json.dumps(record) + "\n")

            if (i + 1) % 50 == 0:
                logger.info("  %d / %d questions complete", i + 1, len(examples))

    logger.info("Results written to %s", out_path)
    return out_path


# ── Score ──────────────────────────────────────────────────────────────────────


def score(
    results_path: str,
    method: str = "both",
    tenant_id: str = BENCH_TENANT,
) -> Dict[str, Any]:
    """
    Score benchmark results and write oracle labels for failures.

    Parameters
    ----------
    results_path:
        Path to results JSONL from run().
    method:
        "exact" — F1 token overlap; "llm" — LLM-as-judge; "both" — both.
    tenant_id:
        PSA tenant for oracle label output path.
    """
    records = _load_results(results_path)
    if not records:
        raise ValueError(f"No records found in {results_path}")

    exact_f1_scores = [_f1_score(r["answer_gold"], r["answer_generated"]) for r in records]
    avg_exact_f1 = sum(exact_f1_scores) / len(exact_f1_scores)

    avg_llm = None
    if method in ("llm", "both"):
        from ..llm import call_llm
        llm_scores = [
            _llm_judge(r["question"], r["answer_gold"], r["answer_generated"], call_llm)
            for r in records
        ]
        valid = [s for s in llm_scores if s is not None]
        avg_llm = sum(valid) / len(valid) if valid else 0.0

    FAILURE_THRESHOLD = 0.3
    oracle_labels_written = 0
    oracle_path = _oracle_labels_path(tenant_id)
    os.makedirs(os.path.dirname(oracle_path), exist_ok=True)

    with open(oracle_path, "a", encoding="utf-8") as f:
        for record, f1 in zip(records, exact_f1_scores):
            if f1 < FAILURE_THRESHOLD:
                label = _make_oracle_label(record, f1)
                f.write(json.dumps(label) + "\n")
                oracle_labels_written += 1

    result: Dict[str, Any] = {
        "exact_f1": round(avg_exact_f1, 4),
        "n_questions": len(records),
        "oracle_labels_written": oracle_labels_written,
        "oracle_labels_path": oracle_path,
    }
    if avg_llm is not None:
        result["llm_score"] = round(avg_llm, 4)
    return result


def _load_results(path: str) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _normalize_text(text: str) -> List[str]:
    import re
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _f1_score(gold: str, pred: str) -> float:
    gold_tokens = set(_normalize_text(gold))
    pred_tokens = set(_normalize_text(pred))
    if not gold_tokens or not pred_tokens:
        return 0.0
    common = gold_tokens & pred_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _llm_judge(question: str, gold: str, generated: str, call_llm) -> Optional[float]:
    prompt = (
        f"Judge whether the answer correctly addresses the question.\n"
        f"Question: {question}\nGold answer: {gold}\nGenerated answer: {generated}\n\n"
        f"Reply with exactly one word: PASS or FAIL."
    )
    try:
        response = call_llm(prompt, max_tokens=10).strip().upper()
        if "PASS" in response:
            return 1.0
        if "FAIL" in response:
            return 0.0
        return None
    except Exception:
        return None


def _oracle_labels_path(tenant_id: str) -> str:
    return os.path.expanduser(f"~/.psa/tenants/{tenant_id}/training/oracle_labels.jsonl")


def _make_oracle_label(record: Dict, f1: float) -> Dict:
    """Create an oracle label for a failed question (failure-signal-only label)."""
    now = datetime.now(timezone.utc).isoformat()
    q_hash = hashlib.md5(record["question"].encode(), usedforsecurity=False).hexdigest()[:8]
    return {
        "query_id": f"lme_{q_hash}",
        "query": record["question"],
        "atlas_version": -1,
        "runtime_model_id": "longmemeval",
        "candidate_anchor_ids": record.get("selected_anchor_ids", []),
        "all_sets": [],
        "winning_oracle_set": [],
        "winning_oracle_score": f1,
        "labeled_at": now,
        "is_high_complexity": False,
        "metadata": {
            "source": "longmemeval",
            "question_id": record.get("question_id", ""),
            "exact_f1": f1,
        },
    }
```

- [ ] **Step 3: Verify import**

```bash
uv run python -c "from psa.benchmarks.longmemeval import ingest, run, score; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add psa/benchmarks/__init__.py psa/benchmarks/longmemeval.py
git commit -m "feat: add psa/benchmarks/longmemeval.py — ingest/run/score harness"
```


---

### Task 6: Write tests for `longmemeval.py`

**Files:**
- Create: `tests/test_longmemeval.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for psa.benchmarks.longmemeval — unit tests with mocks."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from psa.benchmarks.longmemeval import (
    _f1_score,
    _llm_judge,
    _make_oracle_label,
    _normalize_text,
    _write_session_jsonl,
    score,
)


def test_normalize_text_lowercases():
    tokens = _normalize_text("Hello World")
    assert "hello" in tokens
    assert "world" in tokens


def test_normalize_text_strips_punctuation():
    tokens = _normalize_text("hello, world!")
    assert "hello" in tokens
    assert "world" in tokens


def test_f1_score_perfect_match():
    assert _f1_score("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_f1_score_no_overlap():
    assert _f1_score("hello world", "foo bar") == 0.0


def test_f1_score_partial_overlap():
    f1 = _f1_score("the cat sat", "the cat ran")
    assert 0.0 < f1 < 1.0


def test_f1_score_empty_gold():
    assert _f1_score("", "some answer") == 0.0


def test_f1_score_empty_pred():
    assert _f1_score("some gold", "") == 0.0


def test_llm_judge_pass():
    mock_llm = MagicMock(return_value="PASS")
    result = _llm_judge("q", "gold", "generated", mock_llm)
    assert result == 1.0


def test_llm_judge_fail():
    mock_llm = MagicMock(return_value="FAIL")
    result = _llm_judge("q", "gold", "generated", mock_llm)
    assert result == 0.0


def test_llm_judge_unexpected_response():
    mock_llm = MagicMock(return_value="I am not sure")
    result = _llm_judge("q", "gold", "generated", mock_llm)
    assert result is None


def test_llm_judge_exception_returns_none():
    mock_llm = MagicMock(side_effect=RuntimeError("LLM error"))
    result = _llm_judge("q", "gold", "generated", mock_llm)
    assert result is None


def test_make_oracle_label_structure():
    record = {
        "question": "What did I say?",
        "question_id": "q_001",
        "selected_anchor_ids": [1, 2],
    }
    label = _make_oracle_label(record, f1=0.1)
    assert label["query"] == "What did I say?"
    assert label["winning_oracle_score"] == 0.1
    assert label["runtime_model_id"] == "longmemeval"
    assert label["candidate_anchor_ids"] == [1, 2]
    assert label["metadata"]["source"] == "longmemeval"


def test_make_oracle_label_stable_query_id():
    """Same question produces same query_id hash."""
    record = {"question": "stable question", "question_id": "q_001", "selected_anchor_ids": []}
    l1 = _make_oracle_label(record, 0.0)
    l2 = _make_oracle_label(record, 0.0)
    assert l1["query_id"] == l2["query_id"]


def test_write_session_jsonl(tmp_path):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    out = str(tmp_path / "session.jsonl")
    _write_session_jsonl(out, "session_001", messages)
    lines = open(out).readlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["role"] == "user"
    assert record["content"] == "Hello"
    assert record["session_id"] == "session_001"


def test_score_writes_oracle_labels_for_failures(tmp_path):
    results_path = str(tmp_path / "results.jsonl")
    with open(results_path, "w") as f:
        f.write(json.dumps({
            "question_id": "q_001",
            "question": "What color is the sky?",
            "answer_gold": "the sky is blue",
            "answer_generated": "the sky is blue",
            "tokens_used": 100,
            "token_budget": 6000,
            "selected_anchor_ids": [1],
        }) + "\n")
        f.write(json.dumps({
            "question_id": "q_002",
            "question": "What did I eat for lunch?",
            "answer_gold": "a sandwich with turkey",
            "answer_generated": "I do not know",
            "tokens_used": 50,
            "token_budget": 6000,
            "selected_anchor_ids": [2],
        }) + "\n")

    oracle_path = str(tmp_path / "oracle_labels.jsonl")
    with patch("psa.benchmarks.longmemeval._oracle_labels_path", return_value=oracle_path):
        result = score(results_path, method="exact", tenant_id="test")

    assert result["oracle_labels_written"] == 1
    assert result["n_questions"] == 2
    assert 0.0 <= result["exact_f1"] <= 1.0
    labels = [json.loads(l) for l in open(oracle_path).readlines()]
    assert len(labels) == 1
    assert labels[0]["query"] == "What did I eat for lunch?"


def test_score_exact_f1_perfect(tmp_path):
    results_path = str(tmp_path / "results.jsonl")
    with open(results_path, "w") as f:
        f.write(json.dumps({
            "question_id": "q_001",
            "question": "q",
            "answer_gold": "the answer is yes",
            "answer_generated": "the answer is yes",
            "tokens_used": 10,
            "token_budget": 6000,
            "selected_anchor_ids": [],
        }) + "\n")

    oracle_path = str(tmp_path / "oracle_labels.jsonl")
    with patch("psa.benchmarks.longmemeval._oracle_labels_path", return_value=oracle_path):
        result = score(results_path, method="exact", tenant_id="test")

    assert result["exact_f1"] == 1.0
    assert result["oracle_labels_written"] == 0
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/test_longmemeval.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_longmemeval.py
git commit -m "test: add unit tests for longmemeval benchmark harness"
```

---

### Task 7: Extend `psa benchmark` CLI with LongMemEval subcommands

**Files:**
- Modify: `psa/cli.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `_cmd_longmemeval()` to `cli.py`**

Add this function to `psa/cli.py` immediately before `cmd_benchmark`:

```python
def _cmd_longmemeval(args):
    """Handle 'psa benchmark longmemeval <action>' subcommands."""
    from .benchmarks.longmemeval import ingest, run, score

    lme_action = getattr(args, "lme_action", None)
    tenant_id = getattr(args, "tenant", "longmemeval_bench")

    if lme_action == "ingest":
        print(f"Ingesting LongMemEval sessions into tenant '{tenant_id}'...")
        ingest(tenant_id=tenant_id)
        print("Done. Run 'psa benchmark longmemeval run' to benchmark.")

    elif lme_action == "run":
        split = getattr(args, "split", "val")
        limit = getattr(args, "limit", None)
        print(f"Running LongMemEval ({split} split, {'all' if not limit else limit} questions)...")
        out_path = run(split=split, limit=limit, tenant_id=tenant_id)
        print(f"Results: {out_path}")
        print("Run 'psa benchmark longmemeval score' next.")

    elif lme_action == "score":
        results_file = getattr(args, "results", None)
        method = getattr(args, "method", "both")

        if not results_file:
            import glob
            results_dir = os.path.expanduser("~/.psa/benchmarks/longmemeval")
            files = sorted(glob.glob(os.path.join(results_dir, "results_*.jsonl")))
            if not files:
                print("No results files found. Run 'psa benchmark longmemeval run' first.")
                sys.exit(1)
            results_file = files[-1]
            print(f"Scoring: {results_file}")

        stats = score(results_file, method=method, tenant_id=tenant_id)
        print(f"\nLongMemEval Score")
        print(f"  Questions:     {stats['n_questions']}")
        print(f"  Exact F1:      {stats['exact_f1']:.3f}")
        if "llm_score" in stats:
            print(f"  LLM-as-judge:  {stats['llm_score']:.3f}")
        print(f"\nOracle labels written: {stats['oracle_labels_written']}")
        print(f"  -> {stats['oracle_labels_path']}")
        print("\nRun 'psa train' to train the selector on these labels.")

    else:
        print("Usage: psa benchmark longmemeval ingest|run|score")
        sys.exit(1)
```

- [ ] **Step 2: Replace `cmd_benchmark` with a version that routes to longmemeval**

Find `def cmd_benchmark(args):` (around line 651) and replace its entire body:

```python
def cmd_benchmark(args):
    """Benchmark commands — longmemeval harness or quick PSA vs ChromaDB comparison."""
    bench_cmd = getattr(args, "bench_cmd", None)
    if bench_cmd == "longmemeval":
        _cmd_longmemeval(args)
        return

    query = getattr(args, "query", None)
    if not query:
        print("Usage: psa benchmark --query 'your query here'")
        print("       psa benchmark longmemeval ingest|run|score")
        return

    print("PSA benchmark mode — comparing PSA pipeline vs raw ChromaDB search.")
    print("(Requires a populated palace and atlas. Run 'psa mine' and 'psa atlas build' first.)")

    tenant_id = getattr(args, "tenant", "default")
    try:
        from .searcher import search_memories
        from .config import MempalaceConfig
        cfg = MempalaceConfig()

        print(f"\n--- Raw ChromaDB search ---")
        raw_results = search_memories(query, n_results=5, palace_path=cfg.palace_path)
        for i, r in enumerate(raw_results.get("results", []), 1):
            print(f"  [{i}] {r.get('title', '?')} ({r.get('similarity', 0):.3f})")

        print(f"\n--- PSA pipeline search ---")
        from .pipeline import PSAPipeline
        try:
            pipeline = PSAPipeline.from_tenant(tenant_id=tenant_id, psa_mode=cfg.psa_mode)
            result = pipeline.query(query)
            print(f"  Packed context: {result.token_count} tokens")
            print(f"  Selected anchors: {[a.anchor_id for a in result.selected_anchors]}")
            print(f"  Pipeline timing: {result.timing.total_ms:.1f}ms total")
        except FileNotFoundError:
            print(f"  (No atlas for tenant '{tenant_id}' — run 'psa atlas build' first)")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
```

- [ ] **Step 3: Replace the benchmark argparser block**

Find `# benchmark` in the argparser section (around line 1036) and replace the block with:

```python
    # benchmark
    p_benchmark = sub.add_parser(
        "benchmark", help="Benchmark commands (longmemeval harness or --query for quick comparison)"
    )
    p_benchmark.add_argument("--query", default=None, help="Quick query comparison (PSA vs ChromaDB)")
    p_benchmark.add_argument("--tenant", default="default")
    bench_sub = p_benchmark.add_subparsers(dest="bench_cmd")

    p_lme = bench_sub.add_parser("longmemeval", help="LongMemEval benchmark harness")
    p_lme.add_argument("--tenant", default="longmemeval_bench")
    lme_sub = p_lme.add_subparsers(dest="lme_action")
    lme_sub.add_parser("ingest", help="Download dataset and ingest sessions into PSA")
    p_lme_run = lme_sub.add_parser("run", help="Run questions through PSA, generate answers")
    p_lme_run.add_argument("--split", default="val", choices=["val", "test"])
    p_lme_run.add_argument("--limit", type=int, default=None)
    p_lme_score = lme_sub.add_parser("score", help="Score answers and write oracle labels")
    p_lme_score.add_argument("--results", default=None)
    p_lme_score.add_argument("--method", default="both", choices=["exact", "llm", "both"])
```

- [ ] **Step 4: Add `benchmark` extra to `pyproject.toml`**

In `[project.optional-dependencies]`, add:

```toml
# LongMemEval and other HuggingFace benchmarks
benchmark = ["datasets>=2.0"]
```

- [ ] **Step 5: Verify CLI parses correctly**

```bash
uv run python -m psa benchmark longmemeval --help
uv run python -m psa benchmark longmemeval run --help
uv run python -m psa benchmark longmemeval score --help
```

Expected: all print help without errors.

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest tests/ -v --ignore=tests/test_llm_integration.py -x
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add psa/cli.py pyproject.toml
git commit -m "feat: extend psa benchmark with longmemeval ingest/run/score subcommands"
```

---

### Task 8: Final lint, test run, and wrap-up

- [ ] **Step 1: Run ruff on all new and modified files**

```bash
uv run ruff check psa/inspect.py psa/benchmarks/longmemeval.py psa/benchmarks/__init__.py psa/cli.py
uv run ruff format --check psa/inspect.py psa/benchmarks/longmemeval.py psa/cli.py
```

Fix any errors with `uv run ruff format <file>`.

- [ ] **Step 2: Run full test suite with coverage**

```bash
uv run pytest tests/ -v --ignore=tests/test_llm_integration.py --cov=psa --cov-report=term-missing
```

Expected: all tests pass, coverage above 30% threshold.

- [ ] **Step 3: Smoke-test CLI help**

```bash
uv run python -m psa inspect --help
uv run python -m psa log --help
uv run python -m psa benchmark longmemeval --help
```

Expected: clean help output, no import errors.

- [ ] **Step 4: Commit if any fixes were made**

```bash
git add -u
git commit -m "fix: ruff lint/format fixes across new files"
```
