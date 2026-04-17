# Lifecycle Benchmark Leakage Removal

**Date:** 2026-04-17
**Branch:** `fix/lifecycle-benchmark-leakage` (off `main`)

## Problem

`LifecyclePipeline.run()` silently trains the live `memory_scorer_latest` model from any benchmark results file it finds under `~/.psa/benchmarks/longmemeval/results_*.jsonl`. That's the concrete violation of the training policy ordering (real query telemetry → oracle labels → benchmarks-for-regression-only): benchmark artifacts should never write back to the live atlas, and certainly not automatically.

Today: `psa lifecycle run` → finds a benchmark result file → calls `generate_memory_scorer_data(..., mode="benchmark")` → writes weights to `models/memory_scorer_latest`. Operator has no say; the file's presence alone triggers it.

## Goal

Remove the automatic, benchmark-driven memory-scorer training from the lifecycle slow path. Keep the explicit `psa train --memory-scorer` CLI command in place for operator-invoked research use, with clearer help-text provenance. Revisit the CLI path when Branch 3 establishes a production-signal training source.

## Non-goals

- Candidate/promote mirror for the memory scorer (would preempt Branch 3's design).
- Removing or disabling `psa train --memory-scorer` (operator opt-in; not the automatic-leakage class of problem).
- Any change to `psa/training/memory_scorer_data.py` — `mode="benchmark"` stays as a valid mode for the research CLI path.
- Any change to `lifecycle.run()`'s return signature (the `memory_scorer_trained` summary key simply never gets set — existing tests don't assert on it).

## Design

### Change 1: delete the benchmark-training block in `lifecycle.py`

Remove `psa/lifecycle.py:212-242` (the `try`/`except` wrapping `glob.glob(...results_*.jsonl)`, `generate_memory_scorer_data(..., mode="benchmark")`, and `MemoryScorerTrainer(...).train(...)`). Replace with a 4-line comment naming the design and the operator escape hatch:

```python
# Memory scorer is NOT trained in the lifecycle slow path. Benchmark
# artifacts must never write back to the live atlas (see
# docs/superpowers/specs/2026-04-17-lifecycle-benchmark-leakage-design.md).
# Operators can still run `psa train --memory-scorer` explicitly.
```

Because the block is inside the `if retrained:` branch of `_retrain_selector`, removing it has no knock-on effect on surrounding control flow — the coactivation block above it stays untouched, and the `# 6. Save state` comment below it still marks the next stage.

### Change 2: update `--memory-scorer` help text in the CLI

In `psa/cli.py` (around line 1700), change:

```python
help="Also train memory-level re-ranker (requires benchmark results)"
```

to:

```python
help=(
    "Train memory-level re-ranker from benchmark results at "
    "~/.psa/benchmarks/longmemeval/results_*.jsonl. "
    "Research-only; produces a benchmark-derived model. "
    "Revisit in Branch 3 once a production-signal path exists."
)
```

No behavioral change; the warning just moves from silent to visible in `psa train --help`.

### Change 3: regression test

Add to `tests/test_lifecycle.py`:

```python
def test_lifecycle_source_does_not_reference_benchmark_path():
    """Regression: lifecycle.py must not auto-train from benchmark artifacts."""
    import psa.lifecycle as _mod
    import inspect

    src = inspect.getsource(_mod)
    assert "benchmarks/longmemeval" not in src
    assert "MemoryScorerTrainer" not in src
    assert 'mode="benchmark"' not in src and "mode='benchmark'" not in src
```

Structural (source-inspection) rather than behavioral because the invariant is structural: *this file must not reach into the benchmark directory.* The alternative (mocking `lifecycle.run()` end-to-end to assert no call occurs) would require extensive fixture scaffolding and test a symptom rather than the root shape.

## Files touched

| File | Action |
|------|--------|
| `psa/lifecycle.py` | Remove lines 212-242 (the benchmark→memory-scorer block). Replace with a comment. |
| `psa/cli.py` | Expand the `--memory-scorer` help text (line ~1700). |
| `tests/test_lifecycle.py` | Add one source-inspection regression test. |

## Tests to run

- `uv run pytest tests/ -q` — full suite green. The deleted block was inside `try`/`except`, so when the benchmark glob was empty (the usual case) it silently no-op'd; existing tests don't exercise it.
- `uv run ruff check . && uv run ruff format --check .` — clean.

## Success criteria

- `lifecycle.run()` never writes to `models/memory_scorer_latest/` or reads from `~/.psa/benchmarks/longmemeval/`.
- The new test passes and locks that invariant.
- `psa train --memory-scorer` still works and its help text now names its research/benchmark orientation explicitly.
- Nothing else changes.

## Follow-ups (explicitly deferred)

- Branch 3 (production-signal refinement / memory scorer training from oracle labels or query fingerprints) will revisit `psa train --memory-scorer` — likely replacing `mode="benchmark"` with a production-signal mode or routing output to a candidate directory.
- Branch 4 (observability) will add regression-detection metrics reporting for benchmarks without write-back.
