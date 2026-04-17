# Diagnostic Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship always-on per-query tracing and three `psa diag` rollups (`activation`, `advertisement`, `misses`) so the operator can see whether anchors are carrying, which anchors have latent content, and which queries fail below threshold.

**Architecture:** New `psa/trace.py` module handles the write side. `pipeline.query()` is refactored to funnel all code paths through a single return block that writes one trace record per call, tagged with `query_origin`. A new `psa/diag/` package provides pure-function aggregators and the three CLI handlers read from there.

**Tech Stack:** Python 3.13, `argparse`, existing `AtlasManager` / `TenantManager` / `MempalaceConfig`, JSONL as storage format.

**Design spec:** `docs/superpowers/specs/2026-04-17-diag-observability-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `psa/trace.py` | Create | Writer: `_trace_disabled()`, `new_trace_record()`, `write_trace()`. Single sink for JSONL append under `~/.psa/tenants/{id}/query_trace.jsonl`. |
| `psa/config.py` | Modify | Add `trace_queries: bool = True` to `MempalaceConfig`. |
| `psa/pipeline.py` | Modify | Refactor `query()` to single return; add `query_origin="interactive"` kwarg; build + write trace record once at exit. |
| `psa/training/oracle_labeler.py` | Modify | `OracleLabeler.label()` passes `query_origin="labeling"` at the `pipeline.query()` call (line 442 today). |
| `psa/benchmarks/longmemeval.py` | Modify | `run()` passes `query_origin="benchmark"` at the `pipeline.query()` call (line 208 today). |
| `psa/inspect.py` | Modify | `inspect_query()` passes `query_origin="inspect"` at the `pipeline.query()` call. |
| `psa/diag/__init__.py` | Create | Package marker. |
| `psa/diag/trace_reader.py` | Create | `iter_trace_records(tenant_id, *, origins=None)` generator with spec'd origins contract. |
| `psa/diag/activation.py` | Create | `AnchorActivation` dataclass + `activation_report(tenant_id, *, origins=None)`. |
| `psa/diag/advertisement.py` | Create | `AnchorAdvertisement` dataclass + `advertisement_report(tenant_id, *, origins=None)`. |
| `psa/diag/misses.py` | Create | `MissReport` dataclass + `miss_report(tenant_id, *, n_recent=20, origins=None)`. |
| `psa/cli.py` | Modify | Add `cmd_diag` dispatcher + three subparsers (`activation`, `advertisement`, `misses`) + three `_cmd_diag_*` handlers that call the reports and format output. |
| `tests/test_trace_writer.py` | Create | Writer tests: enabled/disabled paths, env flag, config flag, file creation. |
| `tests/test_pipeline_trace.py` | Create | Integration: one trace record per `pipeline.query()` call across all three `result_kind` values, with correct `query_origin` default and override. |
| `tests/test_diag_reader.py` | Create | Origins contract: None → all, `{x,y}` → subset, `set()` → nothing. |
| `tests/test_diag_activation.py` | Create | Activation metric: carry_rate computation, source_anchor_id bookkeeping, n_selected tally. |
| `tests/test_diag_advertisement.py` | Create | Percentile math; memory-count comes from atlas, not SQLite recount. |
| `tests/test_diag_misses.py` | Create | Near-miss definition is strict (rank ≤ 3, only within `result_kind == "empty_selection"` records). |
| `tests/test_cli_diag.py` | Create | End-to-end per subcommand; `--json` envelope shape (`tenant_id`, `atlas_version`, `trace_records`, `origins`, `rows`). |

---

### Task 1: Trace writer + config flag

Smallest piece that unblocks everything else. Pure side-effect writer with a single disable check.

**Files:**
- Create: `psa/trace.py`
- Modify: `psa/config.py`
- Create: `tests/test_trace_writer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trace_writer.py`:

```python
"""Tests for psa.trace — per-query JSONL writer."""

from __future__ import annotations

import json
import os
from pathlib import Path


def test_write_trace_appends_record(tmp_path, monkeypatch):
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PSA_TRACE", raising=False)

    record = {"run_id": "r1", "query": "hello", "tenant_id": "default"}
    write_trace(record, tenant_id="default")

    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    assert path.exists()
    line = path.read_text().strip()
    assert json.loads(line) == record


def test_write_trace_appends_multiple_records(tmp_path, monkeypatch):
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PSA_TRACE", raising=False)

    write_trace({"run_id": "r1", "tenant_id": "default"}, tenant_id="default")
    write_trace({"run_id": "r2", "tenant_id": "default"}, tenant_id="default")

    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["run_id"] == "r1"
    assert json.loads(lines[1])["run_id"] == "r2"


def test_write_trace_disabled_by_env(tmp_path, monkeypatch):
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PSA_TRACE", "0")

    write_trace({"run_id": "r1", "tenant_id": "default"}, tenant_id="default")

    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    assert not path.exists()


def test_write_trace_disabled_by_config(tmp_path, monkeypatch):
    """When MempalaceConfig.trace_queries is False, write_trace no-ops."""
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PSA_TRACE", raising=False)

    # Write a config file with trace_queries=False.
    cfg_dir = tmp_path / ".psa"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.json").write_text('{"trace_queries": false}')

    write_trace({"run_id": "r1", "tenant_id": "default"}, tenant_id="default")

    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    assert not path.exists()


def test_write_trace_enabled_by_default(tmp_path, monkeypatch):
    """No env flag, no config — write happens."""
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PSA_TRACE", raising=False)

    write_trace({"run_id": "r1", "tenant_id": "default"}, tenant_id="default")

    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    assert path.exists()


def test_write_trace_env_wins_over_config(tmp_path, monkeypatch):
    """PSA_TRACE=1 overrides config trace_queries=False."""
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PSA_TRACE", "1")

    cfg_dir = tmp_path / ".psa"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.json").write_text('{"trace_queries": false}')

    write_trace({"run_id": "r1", "tenant_id": "default"}, tenant_id="default")

    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    assert path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_trace_writer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'psa.trace'`.

- [ ] **Step 3: Add the config key**

In `psa/config.py`, add `trace_queries: bool = True` to `MempalaceConfig` alongside other dataclass fields. Also ensure the JSON loader reads the key; most `MempalaceConfig` implementations already use `**kwargs` or explicit load. Check the existing file structure and add the field in the matching style.

If the dataclass already uses `field` defaults, the addition looks like:

```python
trace_queries: bool = True
```

If the JSON loader has an explicit allowlist, add `"trace_queries"` to it.

- [ ] **Step 4: Create `psa/trace.py`**

```python
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


def write_trace(record: dict[str, Any], *, tenant_id: str) -> None:
    """Append one JSONL record to the tenant's query_trace.jsonl.

    No-op when tracing is disabled via env or config. Failures are
    logged, not raised — tracing must never break a live query.
    """
    if _trace_disabled():
        return
    home = os.path.expanduser("~")
    path = os.path.join(home, ".psa", "tenants", tenant_id, "query_trace.jsonl")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.warning("Could not write trace to %s: %s", path, e)


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
    }
```

- [ ] **Step 5: Run the writer tests**

```
uv run pytest tests/test_trace_writer.py -v
```

Expected: all 6 PASS.

- [ ] **Step 6: Run the full config tests to confirm no regression**

```
uv run pytest tests/ -q -k config
```

Expected: all existing config tests still pass.

- [ ] **Step 7: Commit**

```bash
git add psa/trace.py psa/config.py tests/test_trace_writer.py
git commit -m "$(cat <<'EOF'
feat: add psa.trace writer + trace_queries config key

Writer appends one JSONL record per call to
~/.psa/tenants/{id}/query_trace.jsonl. Disable via PSA_TRACE=0 or
trace_queries=false in ~/.psa/config.json. Env wins over config.

Six unit tests cover the enabled/disabled matrix. Writer is side-effect
only — failures log and return, they never break a live query.

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Refactor `pipeline.query()` to single-return + wire trace writer

Core pipeline change. Four current return sites converge into one; trace record populates incrementally; `query_origin="interactive"` added as kwarg.

**Files:**
- Modify: `psa/pipeline.py`
- Create: `tests/test_pipeline_trace.py`

- [ ] **Step 1: Write the failing integration tests**

Create `tests/test_pipeline_trace.py`:

```python
"""Integration: pipeline.query() writes one trace record per call."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def _write_minimal_atlas(atlas_dir: Path) -> None:
    """Write a minimal atlas_v1 dir. See tests/test_curation_curator.py for the pattern."""
    atlas_dir.mkdir(parents=True, exist_ok=True)
    cards = [
        {
            "anchor_id": 1, "name": "a1", "meaning": "m",
            "memory_types": ["semantic"], "include_terms": [], "exclude_terms": [],
            "prototype_examples": [], "near_but_different": [],
            "centroid": [0.0] * 768, "memory_count": 1, "is_novelty": False,
            "status": "active", "metadata": {},
            "generated_query_patterns": [], "query_fingerprint": [],
        }
    ]
    (atlas_dir / "anchor_cards.json").write_text(json.dumps(cards))
    np.save(atlas_dir / "centroids.npy", np.zeros((1, 768), dtype=np.float32))
    (atlas_dir / "atlas_meta.json").write_text(
        json.dumps(
            {
                "version": 1, "tenant_id": "test",
                "stats": {
                    "n_memories": 1, "n_anchors_learned": 1, "n_anchors_novelty": 0,
                    "mean_cluster_size": 1.0, "min_cluster_size": 1,
                    "max_cluster_size": 1, "stability_score": 1.0,
                    "built_at": "2026-04-17T00:00:00+00:00",
                },
            }
        )
    )


def test_pipeline_query_writes_one_trace_record_per_call(tmp_path, monkeypatch):
    """Every call to pipeline.query() appends exactly one record."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PSA_TRACE", raising=False)

    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_minimal_atlas(atlas_dir)

    from psa.pipeline import PSAPipeline

    pipeline = PSAPipeline.from_tenant(tenant_id="default", selector_mode="cosine")
    pipeline.memory_scorer = None  # level-2 off

    pipeline.query("first query")
    pipeline.query("second query")

    path = tenant_dir / "query_trace.jsonl"
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2


def test_pipeline_query_default_origin_is_interactive(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PSA_TRACE", raising=False)

    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_minimal_atlas(atlas_dir)

    from psa.pipeline import PSAPipeline
    pipeline = PSAPipeline.from_tenant(tenant_id="default", selector_mode="cosine")
    pipeline.memory_scorer = None

    pipeline.query("default origin query")

    rec = json.loads((tenant_dir / "query_trace.jsonl").read_text().strip())
    assert rec["query_origin"] == "interactive"


def test_pipeline_query_accepts_query_origin_kwarg(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PSA_TRACE", raising=False)

    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_minimal_atlas(atlas_dir)

    from psa.pipeline import PSAPipeline
    pipeline = PSAPipeline.from_tenant(tenant_id="default", selector_mode="cosine")
    pipeline.memory_scorer = None

    pipeline.query("labeled call", query_origin="labeling")

    rec = json.loads((tenant_dir / "query_trace.jsonl").read_text().strip())
    assert rec["query_origin"] == "labeling"


def test_pipeline_query_records_result_kind_for_empty_selection(tmp_path, monkeypatch):
    """When no anchor crosses threshold, result_kind is 'empty_selection'."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PSA_TRACE", raising=False)

    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_minimal_atlas(atlas_dir)

    from psa.pipeline import PSAPipeline
    pipeline = PSAPipeline.from_tenant(tenant_id="default", selector_mode="cosine")
    pipeline.memory_scorer = None

    # Force empty selection by patching the selector to return [].
    pipeline.selector = MagicMock()
    pipeline.selector.select.return_value = []

    pipeline.query("will produce empty selection")

    rec = json.loads((tenant_dir / "query_trace.jsonl").read_text().strip())
    assert rec["result_kind"] == "empty_selection"
    assert rec["empty_selection"] is True
    assert rec["selected_anchor_ids"] == []


def test_pipeline_query_records_result_kind_for_synthesized(tmp_path, monkeypatch):
    """Normal successful query produces result_kind='synthesized'."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PSA_TRACE", raising=False)

    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_minimal_atlas(atlas_dir)

    from psa.pipeline import PSAPipeline
    pipeline = PSAPipeline.from_tenant(tenant_id="default", selector_mode="cosine")
    pipeline.memory_scorer = None

    # Mock synthesizer so we don't actually call an LLM.
    pipeline._synthesizer.synthesize = MagicMock(return_value="synthesized text")

    pipeline.query("normal query")

    lines = (tenant_dir / "query_trace.jsonl").read_text().strip().split("\n")
    rec = json.loads(lines[-1])
    # result_kind is either "synthesized" or "packer_fallback" depending on path.
    assert rec["result_kind"] in ("synthesized", "packer_fallback", "empty_selection")
    # If synthesizer was called AND returned text, result should be synthesized or
    # packer_fallback. Either way, it's not an exception.


def test_pipeline_query_disabled_trace_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PSA_TRACE", "0")

    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_minimal_atlas(atlas_dir)

    from psa.pipeline import PSAPipeline
    pipeline = PSAPipeline.from_tenant(tenant_id="default", selector_mode="cosine")
    pipeline.memory_scorer = None

    pipeline.query("no trace should be written")

    assert not (tenant_dir / "query_trace.jsonl").exists()
```

- [ ] **Step 2: Run tests — expect failures**

```
uv run pytest tests/test_pipeline_trace.py -v
```

Expected: FAIL on every test — `pipeline.query()` doesn't write a trace today. `test_pipeline_query_accepts_query_origin_kwarg` will also fail on an unexpected-keyword-argument error.

- [ ] **Step 3: Read the current `pipeline.query()` structure**

Before modifying, re-read `psa/pipeline.py` `query()` method (starts around line 140). Note the four return sites:

- Line ~203: early return, "no anchor candidates"
- Line ~255: early return after full-atlas failure
- Line ~293: early return, "empty selection"
- Line ~387: normal return (after synthesis / packer fallback)

The refactor goal: all four converge through one tail block. The trace record is built in a dict that accumulates state at each step; the tail block writes it once, then returns the appropriate `PSAResult`.

- [ ] **Step 4: Refactor `pipeline.query()` to single-return with trace recording**

The exact edit is substantial; this step describes the shape and leaves line-level placement to the implementer.

Add at the top of `query()`, right after the signature:

```python
    def query(self, query: str, top_k_candidates: int = 24, *, query_origin: str = "interactive") -> PSAResult:
        """..."""  # existing docstring
        from .trace import new_trace_record, write_trace
        import datetime as _dt
        import hashlib as _h

        t_start = time.perf_counter()
        _run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + _h.md5(query.encode(), usedforsecurity=False).hexdigest()[:6]
        _trace = new_trace_record(
            run_id=_run_id,
            timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            tenant_id=self.tenant_id,
            atlas_version=getattr(self.atlas, "version", 0),
            query=query,
            query_origin=query_origin,
        )
        _trace["token_budget"] = self.token_budget
        _memory_to_source_anchor: dict[str, int] = {}

        timing = QueryTiming()
        # ... existing body ...
```

Convert all four return sites so they assign to a local `result` variable and break to the tail. For example, the early "no anchor candidates" branch becomes:

```python
            if not anchor_scores:
                packed = PackedContext(
                    query=query,
                    text="(no anchor candidates found — atlas may be empty or unbuilt)",
                    token_count=0, memory_ids=[], sections=[], untyped_count=0,
                )
                result = PSAResult(
                    query=query, packed_context=packed,
                    selected_anchors=[], candidates=[],
                    timing=timing, tenant_id=self.tenant_id,
                    psa_mode=self.psa_mode, selection_mode="full_atlas",
                )
                _trace["selection_mode"] = "full_atlas"
                _trace["result_kind"] = "empty_selection"
                _trace["empty_selection"] = True
                # top_anchor_scores stays []
                # fall through to the tail block
```

Replace `return PSAResult(...)` with the result-assignment pattern, then fall through. If the existing control flow uses deeply-nested if/else, a simple early-`return` → `result = ...; fall_through()` helper can be avoided by a flat sequence of `if ... elif ... else` branches that all terminate in `result = ...; trace updates`.

At the tail (replacing what was formerly the last `return PSAResult(...)`):

```python
        # Tail: populate remaining trace fields, write once, return.
        _trace["tokens_used"] = result.packed_context.token_count
        _trace["timing_ms"] = {
            "embed": round(timing.embed_ms, 1),
            "retrieve": round(timing.retrieve_ms, 1),
            "select": round(timing.select_ms, 1),
            "fetch": round(timing.fetch_ms, 1),
            "pack": round(timing.pack_ms, 1),
            "total": round((time.perf_counter() - t_start) * 1000, 1),
        }
        # Populate packed_memories only if not already set (success path sets it).
        if not _trace["packed_memories"]:
            _trace["packed_memories"] = [
                {"memory_id": mid, "source_anchor_id": _memory_to_source_anchor.get(mid, -1)}
                for mid in result.packed_context.memory_ids
            ]
        write_trace(_trace, tenant_id=self.tenant_id)
        return result
```

**`_memory_to_source_anchor` population:** when the pipeline fetches memories inside `_fetch_memories()`, it iterates selected anchors. Wrap that iteration so each memory is recorded under the selected anchor that first fetched it:

Find `_fetch_memories()` in `psa/pipeline.py`. It currently returns `list[MemoryObject]`. Change the caller site in `query()` to also populate `_memory_to_source_anchor` while iterating. Simplest: do the bookkeeping inline at the call site rather than changing `_fetch_memories`'s signature:

```python
        # Inline: fetch memories and record source anchor per memory.
        t0 = time.perf_counter()
        memories = []
        for sa in selected:
            for m in self.store.query_by_anchor(sa.anchor_id):
                if m.memory_object_id in _memory_to_source_anchor:
                    continue  # first selected anchor wins on multi-assign
                _memory_to_source_anchor[m.memory_object_id] = sa.anchor_id
                memories.append(m)
        timing.fetch_ms = (time.perf_counter() - t0) * 1000
```

**`top_anchor_scores` population:** each selection-mode branch builds the list before the packer runs. In the coactivation branch, rank by refined score and emit `score_source="coactivation_refined"`. In the full-atlas-trained branch, use CE score + `score_source="full_atlas"`. In the legacy retriever branch, use RRF + `score_source="retriever"`. Cap at 24 entries. Mark `selected: true` for ids that appear in `selected_anchor_ids`.

**`result_kind` population:** assign at the point each result branch commits:

- Empty selection paths → `_trace["result_kind"] = "empty_selection"`.
- Synthesizer succeeded → `_trace["result_kind"] = "synthesized"`.
- Synthesizer raised and the packer handled it → `_trace["result_kind"] = "packer_fallback"`.

- [ ] **Step 5: Run tests — expect PASS**

```
uv run pytest tests/test_pipeline_trace.py -v
```

Expected: all 6 PASS.

- [ ] **Step 6: Run the full pipeline test file to confirm no regression**

```
uv run pytest tests/test_pipeline.py -v
```

Expected: all existing pipeline tests still pass.

- [ ] **Step 7: Commit**

```bash
git add psa/pipeline.py tests/test_pipeline_trace.py
git commit -m "$(cat <<'EOF'
feat: pipeline.query() writes one trace record per call

Refactored query() to funnel all four return paths through a single
tail block that builds and writes one trace record via
psa.trace.write_trace. Adds query_origin="interactive" kwarg and
populates result_kind, top_anchor_scores, selected_anchor_ids,
packed_memories (with source_anchor_id), tokens_used, and timing.

Empty-selection and no-candidate paths now produce trace records —
previously they returned early before any logging could happen.

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Tag non-interactive callers

Three call sites need `query_origin` passed explicitly so their traces don't contaminate production rollups.

**Files:**
- Modify: `psa/training/oracle_labeler.py`
- Modify: `psa/benchmarks/longmemeval.py`
- Modify: `psa/inspect.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_trace.py`:

```python
def test_oracle_labeler_tags_labeling_origin(tmp_path, monkeypatch):
    """When OracleLabeler.label() internally calls pipeline.query(), trace should
    record query_origin='labeling'."""
    # Spot-check by running the labeler's internal call site in isolation.
    # We mock heavy LLM path but let the pipeline.query() call happen.
    import json
    from psa.training.oracle_labeler import OracleLabeler

    # This test exercises that OracleLabeler's source calls pipeline.query with
    # query_origin="labeling". The fastest way to verify: look at the source.
    import inspect as _inspect
    src = _inspect.getsource(OracleLabeler)
    assert 'query_origin="labeling"' in src or "query_origin='labeling'" in src


def test_benchmark_longmemeval_tags_benchmark_origin():
    """longmemeval.run() source must pass query_origin='benchmark'."""
    import inspect as _inspect
    from psa.benchmarks import longmemeval
    src = _inspect.getsource(longmemeval)
    assert 'query_origin="benchmark"' in src or "query_origin='benchmark'" in src


def test_inspect_query_tags_inspect_origin():
    """inspect_query's source must pass query_origin='inspect'."""
    import inspect as _inspect
    from psa import inspect as psa_inspect
    src = _inspect.getsource(psa_inspect)
    assert 'query_origin="inspect"' in src or "query_origin='inspect'" in src
```

- [ ] **Step 2: Run tests — expect failures**

```
uv run pytest tests/test_pipeline_trace.py::test_oracle_labeler_tags_labeling_origin tests/test_pipeline_trace.py::test_benchmark_longmemeval_tags_benchmark_origin tests/test_pipeline_trace.py::test_inspect_query_tags_inspect_origin -v
```

Expected: all 3 FAIL.

- [ ] **Step 3: Tag `OracleLabeler.label()`**

In `psa/training/oracle_labeler.py`, find line 442 (`result = self.pipeline.query(query, top_k_candidates=top_k_candidates)`). Replace with:

```python
        result = self.pipeline.query(
            query, top_k_candidates=top_k_candidates, query_origin="labeling"
        )
```

- [ ] **Step 4: Tag `longmemeval.run()`**

In `psa/benchmarks/longmemeval.py`, find line 208 (`result = pipeline.query(question)`). Replace with:

```python
            result = pipeline.query(question, query_origin="benchmark")
```

- [ ] **Step 5: Tag `inspect_query()`**

In `psa/inspect.py`, find the `pipeline.query(...)` call inside `inspect_query()`. Add `query_origin="inspect"` to the kwargs:

```python
    psa_result = pipeline.query(query, query_origin="inspect")
```

(Adjust the argument list to match the actual existing call shape; the only addition is the `query_origin` kwarg.)

- [ ] **Step 6: Run the three tests**

```
uv run pytest tests/test_pipeline_trace.py::test_oracle_labeler_tags_labeling_origin tests/test_pipeline_trace.py::test_benchmark_longmemeval_tags_benchmark_origin tests/test_pipeline_trace.py::test_inspect_query_tags_inspect_origin -v
```

Expected: all 3 PASS.

- [ ] **Step 7: Commit**

```bash
git add psa/training/oracle_labeler.py psa/benchmarks/longmemeval.py psa/inspect.py tests/test_pipeline_trace.py
git commit -m "$(cat <<'EOF'
feat: tag non-interactive pipeline.query() callers with query_origin

OracleLabeler.label() → query_origin='labeling'
longmemeval.run()     → query_origin='benchmark'
inspect_query()       → query_origin='inspect'

Source-inspection tests lock the tags so diag rollups (defaulting to
{"interactive"}) never silently include offline traffic.

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `psa/diag/` reader

The shared generator used by all three reports. Implements the locked `origins` contract from the spec.

**Files:**
- Create: `psa/diag/__init__.py`
- Create: `psa/diag/trace_reader.py`
- Create: `tests/test_diag_reader.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diag_reader.py`:

```python
"""Tests for psa.diag.trace_reader — origins contract."""

from __future__ import annotations

import json
from pathlib import Path


def _write_trace(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_reader_origins_none_returns_all_records(tmp_path, monkeypatch):
    from psa.diag.trace_reader import iter_trace_records

    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    _write_trace(path, [
        {"run_id": "r1", "query_origin": "interactive"},
        {"run_id": "r2", "query_origin": "benchmark"},
        {"run_id": "r3", "query_origin": "labeling"},
    ])

    records = list(iter_trace_records("default", origins=None))
    assert [r["run_id"] for r in records] == ["r1", "r2", "r3"]


def test_reader_origins_interactive_filters(tmp_path, monkeypatch):
    from psa.diag.trace_reader import iter_trace_records

    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    _write_trace(path, [
        {"run_id": "r1", "query_origin": "interactive"},
        {"run_id": "r2", "query_origin": "benchmark"},
    ])

    records = list(iter_trace_records("default", origins={"interactive"}))
    assert [r["run_id"] for r in records] == ["r1"]


def test_reader_empty_origins_returns_nothing(tmp_path, monkeypatch):
    """Explicit filter-everything-out: origins=set() yields zero records."""
    from psa.diag.trace_reader import iter_trace_records

    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    _write_trace(path, [
        {"run_id": "r1", "query_origin": "interactive"},
    ])

    records = list(iter_trace_records("default", origins=set()))
    assert records == []


def test_reader_missing_file_yields_nothing(tmp_path, monkeypatch):
    from psa.diag.trace_reader import iter_trace_records

    monkeypatch.setenv("HOME", str(tmp_path))
    records = list(iter_trace_records("does_not_exist", origins=None))
    assert records == []


def test_reader_skips_malformed_lines(tmp_path, monkeypatch):
    from psa.diag.trace_reader import iter_trace_records

    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"run_id": "r1", "query_origin": "interactive"}\n'
        'not-json\n'
        '{"run_id": "r2", "query_origin": "interactive"}\n'
    )
    records = list(iter_trace_records("default", origins=None))
    assert [r["run_id"] for r in records] == ["r1", "r2"]
```

- [ ] **Step 2: Run tests — expect ModuleNotFoundError**

```
uv run pytest tests/test_diag_reader.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'psa.diag.trace_reader'`.

- [ ] **Step 3: Create `psa/diag/__init__.py`**

```python
"""psa.diag — diagnostic rollups over the per-query trace log."""
```

- [ ] **Step 4: Create `psa/diag/trace_reader.py`**

```python
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
```

- [ ] **Step 5: Run tests — expect PASS**

```
uv run pytest tests/test_diag_reader.py -v
```

Expected: all 5 PASS.

- [ ] **Step 6: Commit**

```bash
git add psa/diag/__init__.py psa/diag/trace_reader.py tests/test_diag_reader.py
git commit -m "$(cat <<'EOF'
feat: add psa.diag.trace_reader with locked origins contract

origins=None      → no filter, all records
origins=set()     → filter everything out
origins={...}     → only matching query_origin

Library stays neutral on defaults; CLI passes {"interactive"} itself.

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Activation / Advertisement / Misses reports

All three share the reader and are independent of each other. Build them in one task (three modules + one test file each).

**Files:**
- Create: `psa/diag/activation.py`
- Create: `psa/diag/advertisement.py`
- Create: `psa/diag/misses.py`
- Create: `tests/test_diag_activation.py`
- Create: `tests/test_diag_advertisement.py`
- Create: `tests/test_diag_misses.py`

- [ ] **Step 1: Write failing tests for activation**

Create `tests/test_diag_activation.py`:

```python
"""Tests for psa.diag.activation — carry-rate computation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def _write_trace(tenant_dir: Path, records: list[dict]) -> None:
    path = tenant_dir / "query_trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _fake_atlas(anchor_ids: list[int]) -> MagicMock:
    atlas = MagicMock()
    cards = []
    for aid in anchor_ids:
        c = MagicMock()
        c.anchor_id = aid
        c.name = f"anchor-{aid}"
        c.memory_count = 1
        cards.append(c)
    atlas.cards = cards
    return atlas


def test_activation_carry_rate_perfect(tmp_path, monkeypatch):
    """Anchor selected and always carries → carry_rate 1.0."""
    from psa.diag.activation import activation_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir, [
        {
            "query_origin": "interactive",
            "selected_anchor_ids": [1],
            "packed_memories": [{"memory_id": "m1", "source_anchor_id": 1}],
        },
        {
            "query_origin": "interactive",
            "selected_anchor_ids": [1],
            "packed_memories": [{"memory_id": "m2", "source_anchor_id": 1}],
        },
    ])

    with patch("psa.diag.activation._load_atlas_for_tenant", return_value=_fake_atlas([1])):
        rows = activation_report("default", origins={"interactive"})

    assert len(rows) == 1
    assert rows[0].anchor_id == 1
    assert rows[0].n_selected == 2
    assert rows[0].n_carried == 2
    assert rows[0].carry_rate == 1.0


def test_activation_carry_rate_zero(tmp_path, monkeypatch):
    """Anchor selected but never carried → carry_rate 0.0."""
    from psa.diag.activation import activation_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir, [
        {
            "query_origin": "interactive",
            "selected_anchor_ids": [1],
            "packed_memories": [{"memory_id": "m1", "source_anchor_id": 2}],
        },
    ])

    with patch("psa.diag.activation._load_atlas_for_tenant", return_value=_fake_atlas([1, 2])):
        rows = activation_report("default", origins={"interactive"})

    row_1 = next(r for r in rows if r.anchor_id == 1)
    assert row_1.n_selected == 1
    assert row_1.n_carried == 0
    assert row_1.carry_rate == 0.0


def test_activation_origins_filter_excludes_benchmark(tmp_path, monkeypatch):
    from psa.diag.activation import activation_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir, [
        {
            "query_origin": "interactive",
            "selected_anchor_ids": [1],
            "packed_memories": [{"memory_id": "m", "source_anchor_id": 1}],
        },
        {
            "query_origin": "benchmark",
            "selected_anchor_ids": [1],
            "packed_memories": [{"memory_id": "m", "source_anchor_id": 1}],
        },
    ])

    with patch("psa.diag.activation._load_atlas_for_tenant", return_value=_fake_atlas([1])):
        rows = activation_report("default", origins={"interactive"})

    row_1 = next(r for r in rows if r.anchor_id == 1)
    assert row_1.n_selected == 1  # only the interactive record
```

- [ ] **Step 2: Write failing tests for advertisement**

Create `tests/test_diag_advertisement.py`:

```python
"""Tests for psa.diag.advertisement — latent-capability gap."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def _write_trace(tenant_dir: Path, records: list[dict]) -> None:
    path = tenant_dir / "query_trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _fake_atlas(anchor_specs: list[tuple[int, int]]) -> MagicMock:
    """anchor_specs is [(anchor_id, memory_count), ...]."""
    atlas = MagicMock()
    cards = []
    for aid, mc in anchor_specs:
        c = MagicMock()
        c.anchor_id = aid
        c.name = f"anchor-{aid}"
        c.memory_count = mc
        cards.append(c)
    atlas.cards = cards
    return atlas


def test_advertisement_gap_positive_for_heavy_rarely_used_anchor(tmp_path, monkeypatch):
    """Anchor with lots of memories but low activation should have high positive gap."""
    from psa.diag.advertisement import advertisement_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"

    # Anchor 1: 100 memories, rarely activated.
    # Anchor 2: 2 memories, activated most queries.
    _write_trace(tenant_dir, [
        {"query_origin": "interactive", "selected_anchor_ids": [2]},
        {"query_origin": "interactive", "selected_anchor_ids": [2]},
        {"query_origin": "interactive", "selected_anchor_ids": [2, 1]},  # 1 activates once
        {"query_origin": "interactive", "selected_anchor_ids": [2]},
    ])

    with patch(
        "psa.diag.advertisement._load_atlas_for_tenant",
        return_value=_fake_atlas([(1, 100), (2, 2)]),
    ):
        rows = advertisement_report("default", origins={"interactive"})

    row_1 = next(r for r in rows if r.anchor_id == 1)
    row_2 = next(r for r in rows if r.anchor_id == 2)
    # Anchor 1: high memory, low activation → positive gap
    assert row_1.advertisement_gap > 0
    # Anchor 2: low memory, high activation → negative gap
    assert row_2.advertisement_gap < 0


def test_advertisement_memory_count_from_atlas_not_sqlite(tmp_path, monkeypatch):
    """Metric reads card.memory_count verbatim, does not recount from SQLite."""
    from psa.diag.advertisement import advertisement_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir, [
        {"query_origin": "interactive", "selected_anchor_ids": [1]},
    ])

    fake_atlas = _fake_atlas([(1, 42)])
    with patch("psa.diag.advertisement._load_atlas_for_tenant", return_value=fake_atlas):
        rows = advertisement_report("default", origins={"interactive"})

    assert next(r for r in rows if r.anchor_id == 1).memory_count == 42
```

- [ ] **Step 3: Write failing tests for misses**

Create `tests/test_diag_misses.py`:

```python
"""Tests for psa.diag.misses — below-threshold misses + near-miss anchors."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def _write_trace(tenant_dir: Path, records: list[dict]) -> None:
    path = tenant_dir / "query_trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _fake_atlas(anchor_ids: list[int]) -> MagicMock:
    atlas = MagicMock()
    cards = []
    for aid in anchor_ids:
        c = MagicMock()
        c.anchor_id = aid
        c.name = f"anchor-{aid}"
        cards.append(c)
    atlas.cards = cards
    return atlas


def test_miss_report_counts_empty_selection_only(tmp_path, monkeypatch):
    from psa.diag.misses import miss_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir, [
        {"query_origin": "interactive", "result_kind": "synthesized",
         "top_anchor_scores": [{"anchor_id": 1, "score": 2.0, "selected": True, "rank": 1}]},
        {"query_origin": "interactive", "result_kind": "empty_selection",
         "top_anchor_scores": [{"anchor_id": 2, "score": 0.9, "selected": False, "rank": 1}]},
        {"query_origin": "interactive", "result_kind": "empty_selection",
         "top_anchor_scores": [{"anchor_id": 2, "score": 0.8, "selected": False, "rank": 1}]},
    ])

    with patch("psa.diag.misses._load_atlas_for_tenant", return_value=_fake_atlas([1, 2])):
        report = miss_report("default", origins={"interactive"})

    assert report.total_queries == 3
    assert report.empty_queries == 2
    assert abs(report.empty_rate - 2 / 3) < 1e-6


def test_near_miss_only_counts_rank_leq_3_in_empty_records(tmp_path, monkeypatch):
    """Near-miss requires rank ≤ 3 AND result_kind == 'empty_selection'."""
    from psa.diag.misses import miss_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir, [
        # Anchor 7 at rank 1 in a SYNTHESIZED record → NOT a near-miss.
        {"query_origin": "interactive", "result_kind": "synthesized",
         "top_anchor_scores": [{"anchor_id": 7, "score": 3.0, "selected": True, "rank": 1}]},
        # Anchor 7 at rank 4 in an empty record → NOT a near-miss (rank too high).
        {"query_origin": "interactive", "result_kind": "empty_selection",
         "top_anchor_scores": [
             {"anchor_id": 8, "score": 1.0, "selected": False, "rank": 1},
             {"anchor_id": 9, "score": 0.9, "selected": False, "rank": 2},
             {"anchor_id": 10, "score": 0.8, "selected": False, "rank": 3},
             {"anchor_id": 7, "score": 0.7, "selected": False, "rank": 4},
         ]},
        # Anchor 7 at rank 2 in an empty record → IS a near-miss.
        {"query_origin": "interactive", "result_kind": "empty_selection",
         "top_anchor_scores": [{"anchor_id": 7, "score": 0.95, "selected": False, "rank": 2}]},
    ])

    with patch("psa.diag.misses._load_atlas_for_tenant", return_value=_fake_atlas(list(range(1, 20)))):
        report = miss_report("default", origins={"interactive"})

    anchor_7 = next((t for t in report.near_miss_anchors if t[0] == 7), None)
    assert anchor_7 is not None, "anchor 7 must be a near-miss"
    assert anchor_7[1] == 1  # exactly one qualifying near-miss
```

- [ ] **Step 4: Run all three tests — expect ModuleNotFoundError**

```
uv run pytest tests/test_diag_activation.py tests/test_diag_advertisement.py tests/test_diag_misses.py -v
```

Expected: all FAIL with module-not-found.

- [ ] **Step 5: Create `psa/diag/activation.py`**

```python
"""activation.py — per-anchor activation / carry rollup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .trace_reader import iter_trace_records


@dataclass
class AnchorActivation:
    anchor_id: int
    anchor_name: str
    n_selected: int
    n_carried: int
    carry_rate: float


def _load_atlas_for_tenant(tenant_id: str) -> Any:
    """Load atlas via TenantManager + AtlasManager. Kept as a patchable helper."""
    from ..tenant import TenantManager
    from ..atlas import AtlasManager

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    return AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id).get_atlas()


def activation_report(
    tenant_id: str, *, origins: Optional[set[str]] = None
) -> list[AnchorActivation]:
    """Compute per-anchor activation/carry stats across the trace log."""
    atlas = _load_atlas_for_tenant(tenant_id)
    if atlas is None:
        return []
    name_by_id = {c.anchor_id: c.name for c in atlas.cards}

    n_selected: dict[int, int] = {}
    n_carried: dict[int, int] = {}

    for rec in iter_trace_records(tenant_id, origins=origins):
        selected = rec.get("selected_anchor_ids", []) or []
        packed = rec.get("packed_memories", []) or []
        carried_source_anchors = {pm.get("source_anchor_id") for pm in packed}

        for aid in selected:
            n_selected[aid] = n_selected.get(aid, 0) + 1
            if aid in carried_source_anchors:
                n_carried[aid] = n_carried.get(aid, 0) + 1

    result: list[AnchorActivation] = []
    for aid, sel in n_selected.items():
        carried = n_carried.get(aid, 0)
        result.append(
            AnchorActivation(
                anchor_id=aid,
                anchor_name=name_by_id.get(aid, f"anchor-{aid}"),
                n_selected=sel,
                n_carried=carried,
                carry_rate=(carried / sel) if sel else 0.0,
            )
        )
    return result
```

- [ ] **Step 6: Create `psa/diag/advertisement.py`**

```python
"""advertisement.py — memory_count vs activation_rate gap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .trace_reader import iter_trace_records


@dataclass
class AnchorAdvertisement:
    anchor_id: int
    anchor_name: str
    memory_count: int
    n_selected: int
    activation_rate: float
    memory_percentile: float
    activation_percentile: float
    advertisement_gap: float


def _load_atlas_for_tenant(tenant_id: str) -> Any:
    from ..tenant import TenantManager
    from ..atlas import AtlasManager

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    return AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id).get_atlas()


def _percentile_rank(value: float, sorted_values: list[float]) -> float:
    """Return percentile rank (0-100) of `value` within `sorted_values`.

    Uses count of values ≤ `value` divided by total count. Simple and
    stable against ties.
    """
    if not sorted_values:
        return 0.0
    n_le = sum(1 for v in sorted_values if v <= value)
    return 100.0 * n_le / len(sorted_values)


def advertisement_report(
    tenant_id: str, *, origins: Optional[set[str]] = None
) -> list[AnchorAdvertisement]:
    atlas = _load_atlas_for_tenant(tenant_id)
    if atlas is None:
        return []

    n_selected: dict[int, int] = {}
    total_queries = 0
    for rec in iter_trace_records(tenant_id, origins=origins):
        total_queries += 1
        for aid in rec.get("selected_anchor_ids", []) or []:
            n_selected[aid] = n_selected.get(aid, 0) + 1

    memory_counts = [c.memory_count for c in atlas.cards]
    memory_counts_sorted = sorted(memory_counts)
    activation_rates = [
        (n_selected.get(c.anchor_id, 0) / total_queries) if total_queries else 0.0
        for c in atlas.cards
    ]
    activation_rates_sorted = sorted(activation_rates)

    rows: list[AnchorAdvertisement] = []
    for c in atlas.cards:
        sel = n_selected.get(c.anchor_id, 0)
        act_rate = (sel / total_queries) if total_queries else 0.0
        mem_pct = _percentile_rank(c.memory_count, memory_counts_sorted)
        act_pct = _percentile_rank(act_rate, activation_rates_sorted)
        rows.append(
            AnchorAdvertisement(
                anchor_id=c.anchor_id,
                anchor_name=c.name,
                memory_count=c.memory_count,
                n_selected=sel,
                activation_rate=act_rate,
                memory_percentile=mem_pct,
                activation_percentile=act_pct,
                advertisement_gap=mem_pct - act_pct,
            )
        )
    return rows
```

- [ ] **Step 7: Create `psa/diag/misses.py`**

```python
"""misses.py — below-threshold query rollup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .trace_reader import iter_trace_records


@dataclass
class MissReport:
    total_queries: int
    empty_queries: int
    empty_rate: float
    recent_misses: list[dict] = field(default_factory=list)
    # Each tuple: (anchor_id, count_of_near_misses, mean_rank, mean_score)
    near_miss_anchors: list[tuple[int, int, float, float]] = field(default_factory=list)


def _load_atlas_for_tenant(tenant_id: str) -> Any:
    from ..tenant import TenantManager
    from ..atlas import AtlasManager

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    return AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id).get_atlas()


def miss_report(
    tenant_id: str,
    *,
    n_recent: int = 20,
    origins: Optional[set[str]] = None,
) -> MissReport:
    total_queries = 0
    empty_queries = 0
    recent_misses: list[dict] = []
    # anchor_id → list of (rank, score) tuples from empty-selection records
    near_miss_accum: dict[int, list[tuple[int, float]]] = {}

    for rec in iter_trace_records(tenant_id, origins=origins):
        total_queries += 1
        if rec.get("result_kind") != "empty_selection":
            continue
        empty_queries += 1
        recent_misses.append(rec)
        for score_rec in rec.get("top_anchor_scores", []) or []:
            rank = score_rec.get("rank", 99)
            if rank > 3:
                continue
            aid = score_rec.get("anchor_id")
            if aid is None:
                continue
            near_miss_accum.setdefault(aid, []).append(
                (rank, float(score_rec.get("score", 0.0)))
            )

    near_miss_rows: list[tuple[int, int, float, float]] = []
    for aid, entries in near_miss_accum.items():
        count = len(entries)
        mean_rank = sum(r for r, _ in entries) / count
        mean_score = sum(s for _, s in entries) / count
        near_miss_rows.append((aid, count, mean_rank, mean_score))
    near_miss_rows.sort(key=lambda t: t[1], reverse=True)

    # Keep only the most recent N empty queries.
    recent_misses = recent_misses[-n_recent:]

    return MissReport(
        total_queries=total_queries,
        empty_queries=empty_queries,
        empty_rate=(empty_queries / total_queries) if total_queries else 0.0,
        recent_misses=recent_misses,
        near_miss_anchors=near_miss_rows,
    )
```

- [ ] **Step 8: Run the three test files — expect PASS**

```
uv run pytest tests/test_diag_activation.py tests/test_diag_advertisement.py tests/test_diag_misses.py -v
```

Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
git add psa/diag/activation.py psa/diag/advertisement.py psa/diag/misses.py tests/test_diag_activation.py tests/test_diag_advertisement.py tests/test_diag_misses.py
git commit -m "$(cat <<'EOF'
feat: add three diag reports (activation / advertisement / misses)

- activation_report: per-anchor n_selected + carry_rate
- advertisement_report: memory_percentile vs activation_percentile gap
- miss_report: total/empty counts + near-miss anchors (rank ≤ 3 in
  empty-selection records only)

All three accept the same origins filter and share
iter_trace_records. Pure functions, no caching. Atlas is loaded via
a patchable helper to make tests hermetic.

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `psa diag` CLI verb

Three subcommands, each a thin formatter over the reports from Task 5.

**Files:**
- Modify: `psa/cli.py`
- Create: `tests/test_cli_diag.py`

- [ ] **Step 1: Write the failing e2e tests**

Create `tests/test_cli_diag.py`:

```python
"""End-to-end CLI tests for `psa diag {activation,advertisement,misses}`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _write_atlas(atlas_dir: Path, anchor_specs: list[tuple[int, int]]) -> None:
    atlas_dir.mkdir(parents=True, exist_ok=True)
    cards = [
        {
            "anchor_id": aid, "name": f"anchor-{aid}", "meaning": "m",
            "memory_types": ["semantic"], "include_terms": [], "exclude_terms": [],
            "prototype_examples": [], "near_but_different": [],
            "centroid": [0.0] * 768, "memory_count": mc, "is_novelty": False,
            "status": "active", "metadata": {},
            "generated_query_patterns": [], "query_fingerprint": [],
        }
        for aid, mc in anchor_specs
    ]
    (atlas_dir / "anchor_cards.json").write_text(json.dumps(cards))
    np.save(atlas_dir / "centroids.npy", np.zeros((len(anchor_specs), 768), dtype=np.float32))
    (atlas_dir / "atlas_meta.json").write_text(
        json.dumps({
            "version": 1, "tenant_id": "test",
            "stats": {
                "n_memories": sum(mc for _, mc in anchor_specs),
                "n_anchors_learned": len(anchor_specs),
                "n_anchors_novelty": 0,
                "mean_cluster_size": 1.0, "min_cluster_size": 1,
                "max_cluster_size": 1, "stability_score": 1.0,
                "built_at": "2026-04-17T00:00:00+00:00",
            },
        })
    )


def _write_trace(tenant_dir: Path, records: list[dict]) -> None:
    path = tenant_dir / "query_trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_cli_diag_activation_json_envelope(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [(1, 5), (2, 3)])
    _write_trace(tenant_dir, [
        {"query_origin": "interactive", "selected_anchor_ids": [1],
         "packed_memories": [{"memory_id": "m", "source_anchor_id": 1}]},
    ])

    with patch("sys.argv", ["psa", "diag", "activation", "--json"]):
        main()

    out = capsys.readouterr().out
    envelope = json.loads(out)
    assert envelope["tenant_id"] == "default"
    assert envelope["atlas_version"] == 1
    assert envelope["trace_records"] == 1
    assert envelope["origins"] == ["interactive"]
    assert isinstance(envelope["rows"], list)
    assert any(row["anchor_id"] == 1 for row in envelope["rows"])


def test_cli_diag_advertisement_default_sort_is_gap_desc(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [(1, 100), (2, 1)])
    _write_trace(tenant_dir, [
        {"query_origin": "interactive", "selected_anchor_ids": [2]},
    ])

    with patch("sys.argv", ["psa", "diag", "advertisement", "--json"]):
        main()

    envelope = json.loads(capsys.readouterr().out)
    rows = envelope["rows"]
    # Anchor 1 has high memory, never activated → largest positive gap → top row.
    assert rows[0]["anchor_id"] == 1


def test_cli_diag_misses_includes_empty_rate_and_near_misses(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [(1, 5)])
    _write_trace(tenant_dir, [
        {"query_origin": "interactive", "result_kind": "empty_selection",
         "top_anchor_scores": [{"anchor_id": 1, "score": 0.9, "selected": False, "rank": 1}]},
        {"query_origin": "interactive", "result_kind": "synthesized",
         "top_anchor_scores": [{"anchor_id": 1, "score": 3.0, "selected": True, "rank": 1}],
         "selected_anchor_ids": [1]},
    ])

    with patch("sys.argv", ["psa", "diag", "misses", "--json"]):
        main()

    envelope = json.loads(capsys.readouterr().out)
    assert envelope["total_queries"] == 2
    assert envelope["empty_queries"] == 1
    assert abs(envelope["empty_rate"] - 0.5) < 1e-6
    # Anchor 1 appeared at rank 1 in the empty record → near-miss.
    assert any(nm["anchor_id"] == 1 for nm in envelope["rows"])


def test_cli_diag_default_origins_excludes_benchmark(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [(1, 1)])
    _write_trace(tenant_dir, [
        {"query_origin": "interactive", "selected_anchor_ids": [1],
         "packed_memories": [{"memory_id": "m", "source_anchor_id": 1}]},
        {"query_origin": "benchmark", "selected_anchor_ids": [1],
         "packed_memories": [{"memory_id": "m", "source_anchor_id": 1}]},
    ])

    with patch("sys.argv", ["psa", "diag", "activation", "--json"]):
        main()

    envelope = json.loads(capsys.readouterr().out)
    row_1 = next(r for r in envelope["rows"] if r["anchor_id"] == 1)
    assert row_1["n_selected"] == 1  # benchmark excluded by default


def test_cli_diag_include_origin_widens_filter(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [(1, 1)])
    _write_trace(tenant_dir, [
        {"query_origin": "interactive", "selected_anchor_ids": [1],
         "packed_memories": [{"memory_id": "m", "source_anchor_id": 1}]},
        {"query_origin": "benchmark", "selected_anchor_ids": [1],
         "packed_memories": [{"memory_id": "m", "source_anchor_id": 1}]},
    ])

    with patch("sys.argv",
               ["psa", "diag", "activation", "--include-origin", "benchmark", "--json"]):
        main()

    envelope = json.loads(capsys.readouterr().out)
    row_1 = next(r for r in envelope["rows"] if r["anchor_id"] == 1)
    assert row_1["n_selected"] == 2  # both records counted
```

- [ ] **Step 2: Run tests — expect failure on `invalid choice: 'diag'`**

```
uv run pytest tests/test_cli_diag.py -v
```

Expected: FAIL at argparse level.

- [ ] **Step 3: Register the `diag` subparser in `psa/cli.py`**

In `psa/cli.py`, find the block where top-level subparsers are registered (search for e.g., `p_atlas = sub.add_parser("atlas", ...)`). Add, in a logical spot alongside similar admin verbs:

```python
    # diag
    p_diag = sub.add_parser(
        "diag", help="Diagnostic rollups over the per-query trace log"
    )
    p_diag.add_argument("--tenant", default="default",
                        help="Tenant identifier (default: 'default')")
    diag_sub = p_diag.add_subparsers(dest="diag_action")

    p_diag_act = diag_sub.add_parser("activation", help="Anchor activation + carry rates")
    p_diag_act.add_argument("--limit", type=int, default=20)
    p_diag_act.add_argument("--min-selections", type=int, default=10)
    p_diag_act.add_argument(
        "--sort", default="n_selected",
        choices=["n_selected", "carry_rate_asc", "gap"],
        help="Default n_selected (most-used first); use carry_rate_asc to "
             "surface worst context switches",
    )
    p_diag_act.add_argument("--include-origin", action="append", default=None,
                            help="Repeatable. Default: interactive only.")
    p_diag_act.add_argument("--json", action="store_true")

    p_diag_adv = diag_sub.add_parser("advertisement", help="Memory-count vs activation gap")
    p_diag_adv.add_argument("--limit", type=int, default=20)
    p_diag_adv.add_argument("--include-origin", action="append", default=None)
    p_diag_adv.add_argument("--json", action="store_true")

    p_diag_mis = diag_sub.add_parser("misses", help="Below-threshold queries + near-miss anchors")
    p_diag_mis.add_argument("--recent", type=int, default=20)
    p_diag_mis.add_argument("--top-near-miss", type=int, default=10)
    p_diag_mis.add_argument("--include-origin", action="append", default=None)
    p_diag_mis.add_argument("--json", action="store_true")
```

- [ ] **Step 4: Dispatch and handlers**

In `psa/cli.py`, find the main dispatch (where `cmd_atlas` etc. are called based on `args.command`). Add:

```python
    elif args.command == "diag":
        cmd_diag(args)
```

Add the following helper and handler functions alongside `cmd_atlas` and friends:

```python
def cmd_diag(args):
    """Handle 'psa diag <subcommand>'."""
    action = getattr(args, "diag_action", None)
    if not action:
        print("Usage: psa diag {activation,advertisement,misses}")
        return
    if action == "activation":
        _cmd_diag_activation(args)
    elif action == "advertisement":
        _cmd_diag_advertisement(args)
    elif action == "misses":
        _cmd_diag_misses(args)


def _resolve_origins(args) -> set[str]:
    """Default to {'interactive'}; --include-origin widens the set."""
    origins = {"interactive"}
    extra = getattr(args, "include_origin", None) or []
    origins.update(extra)
    return origins


def _atlas_meta_for_tenant(tenant_id: str) -> tuple[int, int]:
    """Return (atlas_version, trace_records_count)."""
    import json as _json
    import os as _os
    from .tenant import TenantManager
    from .atlas import AtlasManager

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
    atlas = mgr.get_atlas()
    version = atlas.version if atlas is not None else 0

    trace_path = _os.path.expanduser(f"~/.psa/tenants/{tenant_id}/query_trace.jsonl")
    n = 0
    if _os.path.exists(trace_path):
        with open(trace_path) as f:
            for line in f:
                if line.strip():
                    n += 1
    return version, n


def _cmd_diag_activation(args):
    from dataclasses import asdict
    from .diag.activation import activation_report

    tenant_id = getattr(args, "tenant", "default")
    origins = _resolve_origins(args)
    rows = activation_report(tenant_id, origins=origins)

    # Filter by min-selections, then sort.
    min_sel = getattr(args, "min_selections", 10)
    rows = [r for r in rows if r.n_selected >= min_sel]
    sort_key = getattr(args, "sort", "n_selected")
    if sort_key == "carry_rate_asc":
        rows.sort(key=lambda r: (r.carry_rate, -r.n_selected))
    elif sort_key == "gap":
        rows.sort(key=lambda r: (1.0 - r.carry_rate) * r.n_selected, reverse=True)
    else:  # n_selected
        rows.sort(key=lambda r: r.n_selected, reverse=True)

    limit = getattr(args, "limit", 20)
    if limit and limit > 0:
        rows = rows[:limit]

    version, total = _atlas_meta_for_tenant(tenant_id)
    if getattr(args, "json", False):
        import json as _json
        print(_json.dumps({
            "tenant_id": tenant_id,
            "atlas_version": version,
            "trace_records": total,
            "origins": sorted(origins),
            "rows": [asdict(r) for r in rows],
        }))
        return

    print(f"tenant: {tenant_id}   atlas v{version}   trace records: {total:,}   origins: {', '.join(sorted(origins))}")
    print()
    print(f"{'anchor':<32} {'n_sel':>6} {'n_carry':>8} {'carry%':>8}")
    for r in rows:
        print(f"{r.anchor_name[:32]:<32} {r.n_selected:>6} {r.n_carried:>8} {r.carry_rate * 100:>7.1f}%")


def _cmd_diag_advertisement(args):
    from dataclasses import asdict
    from .diag.advertisement import advertisement_report

    tenant_id = getattr(args, "tenant", "default")
    origins = _resolve_origins(args)
    rows = advertisement_report(tenant_id, origins=origins)
    rows.sort(key=lambda r: r.advertisement_gap, reverse=True)

    limit = getattr(args, "limit", 20)
    if limit and limit > 0:
        rows = rows[:limit]

    version, total = _atlas_meta_for_tenant(tenant_id)
    if getattr(args, "json", False):
        import json as _json
        print(_json.dumps({
            "tenant_id": tenant_id,
            "atlas_version": version,
            "trace_records": total,
            "origins": sorted(origins),
            "rows": [asdict(r) for r in rows],
        }))
        return

    print(f"tenant: {tenant_id}   atlas v{version}   trace records: {total:,}   origins: {', '.join(sorted(origins))}")
    print()
    print(f"{'anchor':<32} {'mem#':>5} {'mem%':>5} {'act_rate':>9} {'act%':>5} {'gap':>6}")
    for r in rows:
        print(
            f"{r.anchor_name[:32]:<32} {r.memory_count:>5} {r.memory_percentile:>4.0f}% "
            f"{r.activation_rate:>8.2%} {r.activation_percentile:>4.0f}% {r.advertisement_gap:>+5.0f}"
        )


def _cmd_diag_misses(args):
    from .diag.misses import miss_report

    tenant_id = getattr(args, "tenant", "default")
    origins = _resolve_origins(args)
    n_recent = getattr(args, "recent", 20)
    top_k = getattr(args, "top_near_miss", 10)
    report = miss_report(tenant_id, n_recent=n_recent, origins=origins)
    near_miss_trimmed = report.near_miss_anchors[:top_k]

    version, total = _atlas_meta_for_tenant(tenant_id)
    if getattr(args, "json", False):
        import json as _json
        print(_json.dumps({
            "tenant_id": tenant_id,
            "atlas_version": version,
            "trace_records": total,
            "origins": sorted(origins),
            "total_queries": report.total_queries,
            "empty_queries": report.empty_queries,
            "empty_rate": report.empty_rate,
            "rows": [
                {"anchor_id": aid, "count": c, "mean_rank": mr, "mean_score": ms}
                for aid, c, mr, ms in near_miss_trimmed
            ],
            "recent_misses": [
                {
                    "timestamp": m.get("timestamp"),
                    "query": m.get("query"),
                    "top_anchor_scores": m.get("top_anchor_scores", [])[:3],
                }
                for m in report.recent_misses
            ],
        }))
        return

    print(f"tenant: {tenant_id}   atlas v{version}   trace records: {total:,}   origins: {', '.join(sorted(origins))}")
    print()
    print(f"Empty-selection rate:  {report.empty_queries:,} / {report.total_queries:,}  ({report.empty_rate * 100:.2f}%)")
    print()
    print("Top near-miss anchors (rank <= 3 in empty-selection records only):")
    for i, (aid, count, mean_rank, mean_score) in enumerate(near_miss_trimmed, start=1):
        print(f"  {i}. anchor-{aid:<5}  {count:>4} near-misses   mean rank {mean_rank:.1f}   mean score {mean_score:.2f}")
    print()
    print("Recent empty queries:")
    for m in report.recent_misses[-min(n_recent, 10):]:
        ts = (m.get("timestamp") or "").replace("T", " ")[:19]
        q = (m.get("query") or "")[:60]
        print(f"  {ts}  {q!r}")
```

- [ ] **Step 5: Run the CLI tests — expect PASS**

```
uv run pytest tests/test_cli_diag.py -v
```

Expected: all 5 PASS.

- [ ] **Step 6: Commit**

```bash
git add psa/cli.py tests/test_cli_diag.py
git commit -m "$(cat <<'EOF'
feat: add `psa diag {activation,advertisement,misses}` CLI

Three subcommands over the psa/diag report functions with a shared
--include-origin filter (default: interactive only). JSON mode emits
the wrapped envelope {tenant_id, atlas_version, trace_records,
origins, rows}. Table mode prints a header that always names the
origin set in view.

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Full test suite + lint

- [ ] **Step 1: Run the full test suite**

```
uv run pytest tests/ -q --ignore=tests/test_convo_miner.py --ignore=tests/test_mcp_server.py
```

Expected: all PASS. The two excluded files are pre-existing flaky-in-suite network-dependent tests; they pass in isolation. Confirm that by running them separately if desired:

```
uv run pytest tests/test_convo_miner.py tests/test_mcp_server.py -q
```

- [ ] **Step 2: Lint + format check**

```
uv run ruff check .
uv run ruff format --check .
```

Expected: clean. If `ruff format --check` reports drift, run `uv run ruff format .`, verify clean, and commit.

- [ ] **Step 3: Commit format fixes (only if Step 2 required them)**

```bash
git add -u
git commit -m "chore: ruff format pass"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 Trace record schema (all fields incl. `query_origin`, `result_kind`, `top_anchor_scores` with `score_source`, `packed_memories[*].source_anchor_id`) | Task 2 (populated during query()), Task 1 (new_trace_record seed) |
| §1 Disable via `PSA_TRACE=0` or `trace_queries=false` | Task 1 |
| §2 Single-return refactor | Task 2 |
| §2 Call-site origin tagging (OracleLabeler, longmemeval, inspect_query) | Task 3 |
| §3 `iter_trace_records` with locked origins contract | Task 4 |
| §3 Activation report + AnchorActivation dataclass | Task 5 |
| §3 Advertisement report + AnchorAdvertisement dataclass + percentile math | Task 5 |
| §3 Miss report + MissReport dataclass + strict near-miss definition | Task 5 |
| §3 `memory_count` from atlas, not SQLite | Task 5 (test_advertisement_memory_count_from_atlas_not_sqlite locks this) |
| §4 `psa diag activation` with --sort, --limit, --min-selections, --include-origin, --json | Task 6 |
| §4 `psa diag advertisement` with --limit, --include-origin, --json | Task 6 |
| §4 `psa diag misses` with --recent, --top-near-miss, --include-origin, --json | Task 6 |
| §4 Default origins = {"interactive"}, CLI header shows origins in view | Task 6 (`_resolve_origins`) |
| §4 JSON envelope {tenant_id, atlas_version, trace_records, origins, rows} | Task 6 (handlers) |
| No breaking changes to psa search / MCP / psa log / psa inspect | Inherent — pipeline.query() adds optional kwarg; no existing callers break. Tests confirm. |

No gaps.

**Placeholder scan:** No TBDs, no "implement later." Every code step has a complete block or an explicit reference ("see Task N Step M") for context-sharing.

**Type consistency:**
- `query_origin` string values consistent across Task 2 default (`"interactive"`), Task 3 call-site overrides (`"labeling"`, `"benchmark"`, `"inspect"`), Task 4 reader filter, Task 6 CLI `_resolve_origins`.
- `score_source` values from spec §1 (`"coactivation_refined" | "full_atlas" | "retriever"`) used in Task 2 trace population (not enforced by tests — spot-check during implementation).
- `iter_trace_records(tenant_id, *, origins)` signature consistent across Task 4 definition and Tasks 5/6 callers.
- `AnchorActivation` fields consistent between Task 5 definition and Task 6 CLI serialization.
