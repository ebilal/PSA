# Diagnostic Observability

**Date:** 2026-04-17
**Branch:** `feat/diag-observability` (off `main`)

## Problem

Branches 1–3 established the research/production boundary and wired a feedback loop for card evolution. None of them can yet answer the philosophical questions that decide whether the system is becoming competent or just verbose:

- **Which anchors activate but don't carry?** An anchor selected often that contributes nothing to packed context is a context switch, not a skill.
- **Which anchors have memories but don't advertise?** A cluster with rich content but weak card text is latent capability the system can't surface.
- **Which queries fail to activate anything at all?** Below-threshold misses are the loudest negative signal against "memory = computation" and are currently invisible — `psa search` returns "no relevant memories" and we keep no trace.

Today the only query-level record is `query_log.jsonl`, written by `psa inspect`. `psa search` and the MCP server call `pipeline.query()` directly and log nothing. Any aggregate view built from that would reflect operator debugging sessions, not production use.

Branch 4 adds always-on per-query tracing and three rollup commands that answer those three questions directly.

## Goals

1. **Always-on trace.** Every `pipeline.query()` call appends one lean JSONL record to `~/.psa/tenants/{id}/query_trace.jsonl`. Disable via `PSA_TRACE=0` or config key.
2. **Three rollup commands** under a new `psa diag` verb: `activation`, `advertisement`, `misses`.
3. **Pure-function aggregators** in a new `psa/diag/` package — no persistent aggregates, no rotation, no caches. Every call re-scans the trace. MVP.
4. **Handle early returns.** `pipeline.query()` funnels through a single return block so empty-selection and no-candidate cases are counted, not dropped.
5. **No breaking changes** to `psa log`, `psa inspect`, the pipeline's public result shape, or existing CLI commands.

## Non-goals

- Trace rotation / retention policy.
- Aggregate caching or incremental rollups.
- Fingerprint-contribution rollup (secondary #4 from the architecture assessment — deferred).
- Promotion-diff rollup (secondary #5 — deferred; needs a "before" snapshot at promote time).
- Time-series plots, dashboards, charts. Tables only; `--json` feeds notebooks for anyone who wants more.
- Query redaction / hashing. Plaintext query for MVP; acknowledged as a later concern.
- Per-anchor memory-count recount from SQLite. `advertisement` uses atlas `card.memory_count` as-is. When forgetting/archive enters the picture, revisit.

---

## §1 — Per-query trace record schema

One JSON line per `pipeline.query()` call, appended to `~/.psa/tenants/{id}/query_trace.jsonl`.

```json
{
  "run_id": "20260417T123045_a3f2c1",
  "timestamp": "2026-04-17T12:30:45.123+00:00",
  "tenant_id": "default",
  "atlas_version": 14,
  "query": "how does token refresh work",
  "selection_mode": "coactivation",
  "result_kind": "synthesized",
  "top_anchor_scores": [
    {"anchor_id": 227, "score": 3.21, "selected": true, "score_source": "coactivation_input", "rank": 1},
    {"anchor_id": 316, "score": 2.87, "selected": true, "score_source": "coactivation_input", "rank": 2},
    {"anchor_id": 415, "score": 1.04, "selected": false, "score_source": "coactivation_input", "rank": 3}
  ],
  "selected_anchor_ids": [227, 316],
  "empty_selection": false,
  "packed_memories": [
    {"memory_id": "mem_abc", "source_anchor_id": 227, "token_cost": 142},
    {"memory_id": "mem_def", "source_anchor_id": 316, "token_cost": 98}
  ],
  "tokens_used": 240,
  "token_budget": 6000,
  "timing_ms": {"embed": 35, "retrieve": 80, "select": 120, "fetch": 12, "pack": 8, "total": 255}
}
```

### Field semantics (the load-bearing ones)

- **`result_kind`**: `"empty_selection" | "synthesized" | "packer_fallback"`. Derived at pipeline exit:
  - `empty_selection` when `selected_anchor_ids == []` (nothing cleared threshold)
  - `packer_fallback` when the synthesizer raised and the packer's `pack_memories_direct` produced the context
  - `synthesized` in the normal path
- **`empty_selection`** (bool): redundant with `result_kind == "empty_selection"` but kept for grep-ability in operational contexts.
- **`selection_mode`**: `"coactivation" | "trained" | "cosine" | "legacy"`. Records which selector path produced the scores.
- **`top_anchor_scores`**: up to 24 entries, ranked by the score the live selector actually decided over. For coactivation mode that's the refined score (not raw CE). `score_source` names which score is stored: `"coactivation_input" | "full_atlas" | "retriever"`. This uniformity means rollups can filter by mode cleanly.
- **`packed_memories[*].source_anchor_id`**: the selected anchor through which this memory entered the fetched set — NOT the memory's stored `primary_anchor_id`. Implemented by keeping `dict[memory_id → selected_anchor_id]` during `_fetch_memories()`. On multi-assignment (rare), record the first selected anchor that fetched the memory.
- **`packed_memories[*].token_cost`**: per-memory token contribution to the packed context. Sourced from the packer's per-section breakdown.
- **`query`**: plaintext. Redaction/hashing is a later concern; single comment in the writer acknowledges it.

### Size budget

~1.5 KB per typical query, ~3 KB at the upper end. 10,000 queries ≈ 15–30 MB per tenant. No rotation in MVP.

### Opt-out

Disable with either:
- `PSA_TRACE=0` in environment
- `"trace_queries": false` in `~/.psa/config.json` (or `MempalaceConfig`)

Default: enabled. Laptop-first, single-operator default.

---

## §2 — Collection layer

### Where the write happens

`pipeline.query()` today has multiple early returns (no candidates, empty selection, normal path). Writing the trace at the end would miss exactly the failure cases Branch 4 cares about most.

**The fix:** refactor `query()` so every code path funnels through one final return block. The trace record is built incrementally during query processing (selection_mode, selected_anchor_ids, packed_memories accumulate), then written once before return.

Concrete refactor shape:

```python
def query(self, ...) -> PSAResult:
    trace = _new_trace_record(query, tenant_id, atlas_version, ...)
    try:
        # ... selection, fetch, pack, synthesize ...
        # populate trace.top_anchor_scores, selected_anchor_ids,
        # packed_memories, result_kind, etc. as we go
        result = PSAResult(...)
    except Exception:
        # Any mid-pipeline exception: trace still gets written with
        # whatever partial state we have; re-raise after write.
        _write_trace(trace, tenant_id)
        raise
    _write_trace(trace, tenant_id)
    return result
```

Exceptions inside `query()` are currently rare; the refactor ensures they don't silently swallow trace emission if they happen. The `_write_trace` call is guarded by `_trace_disabled()` inside the function.

### Files

| File | Action |
|---|---|
| `psa/pipeline.py` | Refactor `query()` to single-return; populate trace incrementally; call `_write_trace` once at exit. |
| `psa/trace.py` (new) | `_trace_disabled()` (env + config check), `_new_trace_record(...)`, `_write_trace(record, tenant_id)`, `_append_jsonl(path, record)`. Writer uses `os.makedirs(dirname, exist_ok=True)` + append-open with `encoding="utf-8"`. |
| `psa/config.py` | One new optional key `trace_queries: bool = True`. |
| `tests/test_trace_writer.py` | Writer tests: enabled/disabled paths, file created, records appended, disable-flag variants. |
| `tests/test_pipeline_trace.py` | Integration: run `pipeline.query()` under a tmp_path HOME, assert a trace record lands with the expected fields for each of the three `result_kind` values. |

### What this does NOT change

- Public `PSAResult` dataclass shape. The trace is a side-channel artifact; consumers of `query()`'s return value see no new fields.
- `psa log`, `psa inspect`, `query_log.jsonl`. Those continue to work exactly as today. Branch 4 adds a *separate* trace stream.

---

## §3 — Aggregation layer

Pure-function aggregators in a new `psa/diag/` package. Each call re-scans the trace file; no caches, no background jobs.

### Package layout

```
psa/diag/
  __init__.py
  trace_reader.py       # iter_trace_records(tenant_id) → generator[dict]
  activation.py         # AnchorActivation + activation_report()
  advertisement.py      # AnchorAdvertisement + advertisement_report()
  misses.py             # MissReport + miss_report()
```

### Shared reader

```python
def iter_trace_records(tenant_id: str) -> Iterator[dict]:
    """Yield every record from ~/.psa/tenants/{id}/query_trace.jsonl.
    Missing file yields nothing (no error). Malformed JSON lines are skipped
    with a debug log.
    """
```

### Activation report

Answers "which anchors activate but don't carry?"

```python
@dataclass
class AnchorActivation:
    anchor_id: int
    anchor_name: str
    n_selected: int                    # in selected_anchor_ids
    n_carried: int                     # ≥1 packed_memory had source_anchor_id == this
    carry_rate: float                  # n_carried / n_selected
    tokens_contributed: int            # sum across all queries
    mean_tokens_per_selection: float

def activation_report(tenant_id: str) -> list[AnchorActivation]:
    """One entry per anchor that was selected in ≥1 trace record.
    Ordering is up to the caller; CLI applies its own sort.
    """
```

### Advertisement report

Answers "which anchors have memories but don't advertise?"

```python
@dataclass
class AnchorAdvertisement:
    anchor_id: int
    anchor_name: str
    memory_count: int                  # from atlas card (NOT a live SQLite recount)
    n_selected: int
    activation_rate: float             # n_selected / total_queries
    memory_percentile: float           # among all anchors in atlas
    activation_percentile: float
    advertisement_gap: float           # memory_percentile - activation_percentile

def advertisement_report(tenant_id: str) -> list[AnchorAdvertisement]: ...
```

**Memory count source:** `card.memory_count` from the current atlas. When forgetting/archive lands, this may diverge from actual active-memory counts; at that point we'll revisit whether this metric should recount from SQLite. For now, atlas-as-of-build is the right reference.

### Miss report

Answers "which queries fail to activate anything, and what almost matched?"

```python
@dataclass
class MissReport:
    total_queries: int
    empty_queries: int
    empty_rate: float
    recent_misses: list[dict]          # records where result_kind == "empty_selection"
    near_miss_anchors: list[tuple[int, int, float, float]]
    # Each tuple: (anchor_id, count_of_near_misses, mean_rank, mean_score)

def miss_report(tenant_id: str, n_recent: int = 20) -> MissReport: ...
```

**Near-miss definition (strict):** an anchor appears in the `top_anchor_scores` list of a trace record whose `result_kind == "empty_selection"`, at `rank ≤ 3`. Counted ONLY in empty-selection records — not across all queries. This keeps the "near-miss" signal about what was almost selected under the live selector, not any general scoring closeness.

### What this deliberately does NOT provide

- No time-series per anchor.
- No before/after diffs against promotions.
- No fingerprint-vs-oracle contribution split.
- No query clustering for the miss log (could be useful — anchors that miss on similar queries — but adds dependency on embeddings and is clearly follow-up).

---

## §4 — CLI surface

New top-level verb `psa diag`. Distinct from `psa log` (per-query inspection) and `psa inspect` (single-query debug). Three subcommands.

### `psa diag activation`

```
psa diag activation [--tenant T] [--limit N] [--min-selections N]
                    [--sort n_selected|carry_rate_asc|gap] [--json]
```

Default: `--sort n_selected` descending (matches "most-used first"), `--limit 20`, `--min-selections 10`.

**Recommended operator-facing sort: `--sort carry_rate_asc`.** Surfaces anchors with the worst contribute-rate first, filtered to those with meaningful activation volume. That's the "worst context switches" view. Default was left as `n_selected` because it's the most intuitive first view; `carry_rate_asc` is the diagnostic follow-up. Both documented in `--help`.

Default tabular output:

```
tenant: default   atlas v14   trace records: 8,412   unique anchors seen: 187/256

anchor                          n_sel   n_carry  carry%  mean_tok
auth-jwt-patterns                 412      389    94.4%    128.3
schema-migration-decisions        387      341    88.1%    112.7
recipe-meal-prep                  298      141    47.3%     34.2
timezone-handling                 276      251    91.0%     98.4
```

### `psa diag advertisement`

```
psa diag advertisement [--tenant T] [--limit N] [--json]
```

Default: sorted by `advertisement_gap` descending. `--limit 20`.

```
tenant: default   atlas v14   trace records: 8,412

anchor                          mem#   mem%   act_rate  act%  gap
deployment-runbooks-2024          84    97     0.4%       12    +85
legacy-auth-migrations            62    88     1.1%       22    +66
weekly-standups                  108    99    14.8%       91     +8
auth-jwt-patterns                 47    72     4.9%       68     +4
recipe-meal-prep                  31    48     3.5%       64    -16
```

### `psa diag misses`

```
psa diag misses [--tenant T] [--recent N] [--top-near-miss K] [--json]
```

Defaults: `--recent 20`, `--top-near-miss 10`.

```
tenant: default   atlas v14   trace records: 8,412

Empty-selection rate:  342 / 8,412  (4.07%)

Top near-miss anchors (rank ≤ 3 in empty-selection records only):
  1. legacy-auth-migrations         87 near-misses    mean rank 2.1   mean score 1.04
  2. deployment-runbooks-2024       54 near-misses    mean rank 1.8   mean score 0.92
  3. recipe-meal-prep               41 near-misses    mean rank 2.4   mean score 0.87

Recent empty queries (last 20):
  2026-04-17 11:42  "how do i rotate the jwt secret in prod"    near-miss: legacy-auth-migrations (0.97)
  2026-04-17 11:28  "what did we decide about rollback policy"  near-miss: deployment-runbooks-2024 (0.88)
```

### `--json` output shape

All three subcommands emit the same envelope under `--json`:

```json
{
  "tenant_id": "default",
  "atlas_version": 14,
  "trace_records": 8412,
  "rows": [ ... ]
}
```

Where `rows` is the list of the subcommand's dataclass serialized to dicts. For `psa diag misses` the envelope also carries top-level fields (`empty_queries`, `empty_rate`) alongside `rows` (the near-miss anchors).

Wrapping with `tenant_id` + `atlas_version` + `trace_records` makes notebook use cleaner: one object carries context, not a bare array. Any downstream tooling (if built) can key on the metadata to compare tenants or versions.

### Files

| File | Action |
|---|---|
| `psa/cli.py` | Add `cmd_diag` dispatcher + three subparsers (`activation`, `advertisement`, `misses`) + three `_cmd_diag_*` handler functions that call `psa/diag/` and format output. |
| `tests/test_cli_diag.py` | End-to-end: fixture atlas + synthetic trace, run each subcommand, assert expected rows and `--json` envelope shape. |

Handlers are thin formatters; metric logic lives in `psa/diag/*.py` (§3), each with its own pytest module.

---

## Files — full list

### New

| File | Responsibility |
|---|---|
| `psa/trace.py` | Writer: `_trace_disabled()`, `_new_trace_record()`, `_write_trace()`. |
| `psa/diag/__init__.py` | Package marker. |
| `psa/diag/trace_reader.py` | `iter_trace_records(tenant_id)` generator. |
| `psa/diag/activation.py` | `AnchorActivation` + `activation_report()`. |
| `psa/diag/advertisement.py` | `AnchorAdvertisement` + `advertisement_report()`. |
| `psa/diag/misses.py` | `MissReport` + `miss_report()`. |
| `tests/test_trace_writer.py` | Writer tests. |
| `tests/test_pipeline_trace.py` | Pipeline integration — trace lands for each `result_kind`. |
| `tests/test_diag_activation.py` | Activation metric tests (synthetic trace fixtures). |
| `tests/test_diag_advertisement.py` | Advertisement metric tests. |
| `tests/test_diag_misses.py` | Miss-report tests, including strict near-miss semantics. |
| `tests/test_cli_diag.py` | CLI end-to-end for all three subcommands + `--json`. |

### Modified

| File | Change |
|---|---|
| `psa/pipeline.py` | Refactor `query()` to single-return; accumulate trace fields incrementally; `_write_trace` once at exit. |
| `psa/config.py` | Add `trace_queries: bool = True` key. |
| `psa/cli.py` | New `cmd_diag` + three subparsers. |

### Not touched

- `psa/anchor.py`, `psa/atlas.py`, `psa/fingerprints.py` — unchanged.
- `psa/inspect.py`, `psa/log.py` (if it exists) — `query_log.jsonl` behavior unchanged; `psa log` and `psa inspect` commands unchanged.
- `psa/curation/*` — unchanged.
- Branch 1/2/3 artifacts and promotion flow — unchanged.

---

## Success criteria

- Every `pipeline.query()` call writes one trace record unless `PSA_TRACE=0` or `trace_queries=False`.
- All three `result_kind` values (`synthesized`, `empty_selection`, `packer_fallback`) produce trace records — empty-selection especially.
- `psa diag activation` surfaces anchors with low `carry_rate` + high `n_selected` in the `carry_rate_asc` sort.
- `psa diag advertisement` surfaces anchors with high `memory_percentile` + low `activation_percentile` (positive `advertisement_gap`) in default sort.
- `psa diag misses` reports empty-selection rate and near-miss anchors counted only within empty-selection records.
- `--json` output wraps rows in `{tenant_id, atlas_version, trace_records, rows}`.
- Full test suite green; `ruff check` and `ruff format --check` clean.
- No breaking changes to `psa search`, `psa inspect`, `psa log`, MCP server, or the promotion flow.

## Follow-ups (explicit deferrals)

- Trace rotation / retention (flagged as future operational concern).
- Fingerprint-contribution rollup (`psa diag fingerprint-contribution`).
- Promotion-diff rollup (`psa diag promotion-diff`) — requires "before" snapshot at promote time.
- Query clustering for miss-log (which queries miss together).
- Query redaction / hashing mode.
- Per-anchor active-memory recount from SQLite (once forgetting/archive matures).
- Time-series / trend views.
- **Card-vs-weights gate asymmetry** (flagged in architecture assessment, §1). Trained models (`selector_latest`, `coactivation_latest`) are auto-written on lifecycle runs; anchor cards gate through candidate/promote. Philosophically inconsistent if both are "adaptive cognitive surface." Branch 4 telemetry will show how often promotions happen in practice — that data informs whether the asymmetry is worth resolving and how.
- **Card text only grows** (flagged in architecture assessment, §4b). `generated_query_patterns` accumulates on promote; nothing prunes patterns whose source memories were forgotten. Cards become historical residue rather than live self-descriptions. Major future design item; depends on how forgetting matures.
