# Lifecycle Benchmark Leakage Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the silent benchmark-driven memory-scorer training block from `LifecyclePipeline.run()` so benchmark artifacts never auto-write back to the live atlas. Keep the explicit `psa train --memory-scorer` CLI path in place with clearer research-only help text.

**Architecture:** Delete a 31-line `try`/`except` block in `psa/lifecycle.py`, replace with a 4-line comment, update one CLI help string, lock the removal with one source-inspection regression test. No behavior changes anywhere except: lifecycle no longer reads from `~/.psa/benchmarks/` or writes to `models/memory_scorer_latest/`.

**Tech Stack:** Python 3.13, `inspect.getsource`, existing pytest conventions.

**Design spec:** `docs/superpowers/specs/2026-04-17-lifecycle-benchmark-leakage-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `psa/lifecycle.py` | Modify | Remove the benchmark→memory-scorer training block (lines 212-242). Replace with a 4-line comment pointing at the design spec and the operator escape hatch. |
| `psa/cli.py` | Modify | Expand the `--memory-scorer` help text so the benchmark provenance is visible in `psa train --help` instead of silent. |
| `tests/test_lifecycle.py` | Modify | Add one source-inspection regression test that locks the removal. |

---

### Task 1: Remove benchmark-training block + update help + regression test

**Files:**
- Modify: `psa/lifecycle.py` (lines 212-242)
- Modify: `psa/cli.py` (around line 1700 — the `--memory-scorer` argparse flag)
- Modify: `tests/test_lifecycle.py`

- [ ] **Step 1: Write the failing regression test**

Append to `tests/test_lifecycle.py`:

```python
def test_lifecycle_source_does_not_reference_benchmark_path():
    """Regression: lifecycle.py must not auto-train from benchmark artifacts.

    Benchmark-derived memory scorer training used to run in the slow path.
    That's a research/production boundary violation; the block was removed
    in Branch 2. This test locks the removal.
    """
    import inspect

    import psa.lifecycle as _mod

    src = inspect.getsource(_mod)
    assert "benchmarks/longmemeval" not in src, (
        "lifecycle.py must not reference the benchmark results directory"
    )
    assert "MemoryScorerTrainer" not in src, (
        "lifecycle.py must not import or call MemoryScorerTrainer"
    )
    assert 'mode="benchmark"' not in src and "mode='benchmark'" not in src, (
        "lifecycle.py must not invoke any trainer with mode=benchmark"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
uv run pytest tests/test_lifecycle.py::test_lifecycle_source_does_not_reference_benchmark_path -v
```

Expected: FAIL. All three assertions should fire because the current `psa/lifecycle.py` contains `benchmarks/longmemeval`, `MemoryScorerTrainer`, and `mode="benchmark"`.

- [ ] **Step 3: Remove the benchmark block from `psa/lifecycle.py`**

In `psa/lifecycle.py`, find and replace this exact block:

```python
                try:
                    import glob as _glob
                    from .training.memory_scorer_data import generate_memory_scorer_data
                    from .training.train_memory_scorer import MemoryScorerTrainer

                    results_dir = os.path.expanduser("~/.psa/benchmarks/longmemeval")
                    results_files = sorted(_glob.glob(os.path.join(results_dir, "results_*.jsonl")))
                    if results_files:
                        scorer_data_path = os.path.join(
                            tenant.root_dir, "training", "memory_scorer_train.jsonl"
                        )
                        from .pipeline import PSAPipeline

                        _pipeline = PSAPipeline.from_tenant(tenant_id=tenant_id)
                        n_ex = generate_memory_scorer_data(
                            results_path=results_files[-1],
                            output_path=scorer_data_path,
                            pipeline=_pipeline,
                            mode="benchmark",
                        )
                        if n_ex >= 200:
                            scorer_output = os.path.join(
                                tenant.root_dir, "models", "memory_scorer_latest"
                            )
                            MemoryScorerTrainer(output_dir=scorer_output).train(
                                data_path=scorer_data_path
                            )
                            summary["memory_scorer_trained"] = True
                            print("        Memory scorer trained.")
                except Exception as e:
                    logger.warning("Memory scorer training failed: %s", e)
```

With:

```python
                # Memory scorer is NOT trained in the lifecycle slow path.
                # Benchmark artifacts must never write back to the live atlas
                # (see docs/superpowers/specs/2026-04-17-lifecycle-benchmark-leakage-design.md).
                # Operators can still run `psa train --memory-scorer` explicitly.
```

The replacement sits at the same indentation level (16 spaces) as the removed `try:`. The surrounding control flow is untouched: the coactivation block above it stays, and the `else:` branch below it (attached to `if atlas is not None:`) is still reachable.

- [ ] **Step 4: Update the `--memory-scorer` help text in `psa/cli.py`**

In `psa/cli.py`, find this block (around line 1697):

```python
    p_train.add_argument(
        "--memory-scorer",
        action="store_true",
        help="Also train memory-level re-ranker (requires benchmark results)",
    )
```

Replace with:

```python
    p_train.add_argument(
        "--memory-scorer",
        action="store_true",
        help=(
            "Train memory-level re-ranker from benchmark results at "
            "~/.psa/benchmarks/longmemeval/results_*.jsonl. "
            "Research-only; produces a benchmark-derived model. "
            "Revisit in Branch 3 once a production-signal path exists."
        ),
    )
```

- [ ] **Step 5: Run the regression test**

Run:
```
uv run pytest tests/test_lifecycle.py::test_lifecycle_source_does_not_reference_benchmark_path -v
```

Expected: PASS.

- [ ] **Step 6: Run the full `test_lifecycle.py` to confirm no regressions**

Run:
```
uv run pytest tests/test_lifecycle.py -v
```

Expected: all PASS. The existing two tests (`test_retrain_selector_mutates_state_dict`, `test_retrain_selector_gates_not_met_leaves_state`) target `_retrain_selector` directly and don't touch the removed block.

- [ ] **Step 7: Commit**

```bash
git add psa/lifecycle.py psa/cli.py tests/test_lifecycle.py
git commit -m "$(cat <<'EOF'
fix: remove silent benchmark-driven memory scorer training from lifecycle

Lifecycle slow path used to glob ~/.psa/benchmarks/longmemeval/results_*.jsonl
and train models/memory_scorer_latest from it with mode="benchmark" —
a research artifact auto-writing back to the live atlas. Removed.

The explicit `psa train --memory-scorer` CLI path is unchanged; its help
text now names its benchmark-research orientation visibly. A
source-inspection regression test locks the removal so a future diff
can't silently re-introduce the leak.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Full test suite + lint

- [ ] **Step 1: Run the full test suite**

Run:
```
uv run pytest tests/ -q
```

Expected: all PASS. The deleted block was inside a `try`/`except`, so when the benchmark glob was empty (the usual case in CI and during tests) it silently no-op'd; existing tests don't exercise it, so no regressions.

If `tests/test_convo_miner.py::test_convo_mining` times out due to network/httpx (it's pre-existing flaky behavior unrelated to this branch), that's acceptable — re-run it in isolation (`uv run pytest tests/test_convo_miner.py::test_convo_mining`) to confirm it's the lone failure and unrelated.

- [ ] **Step 2: Lint + format check**

Run:
```
uv run ruff check .
uv run ruff format --check .
```

Expected: both clean. If `ruff format --check` reports drift, run `uv run ruff format .`, verify clean again, and commit the format fixes separately.

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
| Remove benchmark-training block in `lifecycle.py` (lines 212-242) | Task 1 Step 3 |
| Replace with 4-line comment pointing at spec + CLI escape hatch | Task 1 Step 3 |
| Expand `--memory-scorer` help text to name benchmark provenance | Task 1 Step 4 |
| Source-inspection regression test | Task 1 Step 1 (written first, TDD) + Step 5 (verified) |
| No changes to `psa/training/memory_scorer_data.py` | Inherent — no task touches it |
| No changes to `cmd_train` memory scorer body (`psa/cli.py:813-845`) | Inherent — Task 1 Step 4 only modifies the argparse block at ~line 1697 |
| `lifecycle.run()` return signature unchanged | Inherent — Task 1 Step 3 deletes the `summary["memory_scorer_trained"] = True` assignment, but the key was never guaranteed to be set and existing tests don't assert on it |
| Full suite + lint pass | Task 2 |

No gaps.

**Placeholder scan:** No TBDs, no "TODO" / "implement later" patterns. Every step has complete replacement text or exact commands.

**Type consistency:**
- Regression-test assertion strings (`"benchmarks/longmemeval"`, `"MemoryScorerTrainer"`, `'mode="benchmark"'`, `"mode='benchmark'"`) appear verbatim in the current `psa/lifecycle.py` — removing the block removes all four. Checked by reading the current source: line 214 imports `MemoryScorerTrainer`, line 217 references `benchmarks/longmemeval`, line 230 passes `mode="benchmark"`. After removal, none remain.
- The CLI help-text replacement leaves `action="store_true"` and the `--memory-scorer` flag name intact, so `getattr(args, "memory_scorer", False)` downstream in `cmd_train` continues to work unchanged.
