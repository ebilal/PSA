# Memory Forgetting — Relative Usage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the absolute wall-clock `idle_days` term from the memory forgetting score and replace it with anchor-relative low-usage pressure so dormant-but-useful memories are not archived purely because time passed.

**Architecture:** Single-file behavioral change in `psa/forgetting.py` plus a new dedicated test file. `low_usage_pressure` is computed once per pruning batch (by the caller that knows the peer set) and passed into `forgetting_score` as a keyword argument. Per-anchor pruning uses anchor-local percentile; global-cap pruning uses a weighted hybrid of anchor-local and tenant-wide percentile. Grace period, archive-first semantics, and DB schema are unchanged.

**Tech Stack:** Python 3.13, SQLite (WAL), pytest, ruff, uv. No new dependencies.

**Companion spec:** `docs/superpowers/specs/2026-04-20-memory-forgetting-relative-usage-design.md`.

---

## File Structure

**Modify:**
- `psa/forgetting.py` — replace idle term, new helper `low_usage_pressure`, update `forgetting_score` signature, update `prune_anchor` and `enforce_global_cap`, add shadow log.
- `README.md:402-414` — rewrite the forgetting-score formula block.

**Create:**
- `tests/test_forgetting.py` — 12 test cases covering score behavior, pruning order, edge cases, grace period, score range.

**No migrations.** Schema columns `pack_count`, `select_count`, `last_packed`, `created_at`, `quality_score`, `is_archived` already exist (`psa/memory_object.py:83-86, 212-215`). Not touched: `psa/lifecycle.py` (callers of `prune_anchor`/`enforce_global_cap` use only the module-level function, so no call-site change), `psa/health.py` (only imports `MAX_MEMORIES`).

---

## Task 1: Scaffold `tests/test_forgetting.py` with a MemoryObject factory

**Files:**
- Create: `tests/test_forgetting.py`

- [ ] **Step 1: Create the test file with imports and a fixture helper**

Create `tests/test_forgetting.py`:

```python
"""Tests for psa.forgetting: score behavior and pruning order."""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pytest

from psa.forgetting import forgetting_score
from psa.memory_object import MemoryObject, MemoryType


def make_memory(
    *,
    pack_count: int = 0,
    select_count: int = 0,
    quality_score: float = 0.5,
    created_at: Optional[datetime] = None,
    last_packed: Optional[datetime] = None,
    primary_anchor_id: int = 1,
    tenant_id: str = "t",
) -> MemoryObject:
    """Build a MemoryObject with just the fields the forgetting system reads."""
    mo = MemoryObject.create(
        tenant_id=tenant_id,
        memory_type=MemoryType.SEMANTIC,
        title="t",
        body="b",
        summary="s",
        source_ids=[],
        classification_reason="",
        pack_count=pack_count,
        select_count=select_count,
        quality_score=quality_score,
        primary_anchor_id=primary_anchor_id,
    )
    if created_at is not None:
        mo.created_at = created_at.isoformat()
    if last_packed is not None:
        mo.last_packed = last_packed.isoformat()
    return mo


def now_utc() -> datetime:
    return datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)


def old(days: int) -> datetime:
    return now_utc() - timedelta(days=days)
```

- [ ] **Step 2: Add a smoke test that imports the helper**

Append:

```python
def test_fixture_builds_memory():
    m = make_memory(pack_count=3, created_at=old(10))
    assert m.pack_count == 3
    assert m.primary_anchor_id == 1
```

- [ ] **Step 3: Run it**

Run: `uv run pytest tests/test_forgetting.py -v`
Expected: PASS (1 test).

- [ ] **Step 4: Commit**

```bash
git add tests/test_forgetting.py
git commit -m "test(forgetting): scaffold test file with memory factory helper"
```

---

## Task 2: Write the failing "age alone does not raise score" test

**Files:**
- Modify: `tests/test_forgetting.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_forgetting.py`:

```python
def test_age_alone_does_not_raise_score():
    """Two memories identical except created_at must score identically."""
    young = make_memory(pack_count=0, quality_score=0.5, created_at=old(30))
    ancient = make_memory(pack_count=0, quality_score=0.5, created_at=old(365))

    s_young = forgetting_score(young, anchor_size=10, now=now_utc(), low_usage_pressure=0.5)
    s_ancient = forgetting_score(ancient, anchor_size=10, now=now_utc(), low_usage_pressure=0.5)

    assert s_young == pytest.approx(s_ancient)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/test_forgetting.py::test_age_alone_does_not_raise_score -v`
Expected: FAIL — current `forgetting_score` has no `low_usage_pressure` kwarg AND returns different values because of the `idle_days/90` term.

- [ ] **Step 3: No commit yet** — implementation fix arrives in Task 4.

---

## Task 3: Implement `low_usage_pressure` helper (TDD)

**Files:**
- Modify: `psa/forgetting.py`
- Modify: `tests/test_forgetting.py`

- [ ] **Step 1: Write the failing tests for the helper**

Append to `tests/test_forgetting.py`:

```python
from psa.forgetting import low_usage_pressure  # noqa: E402


def test_low_usage_pressure_small_anchor_returns_zero():
    peers = [make_memory(pack_count=i) for i in range(4)]  # N=4 < 5
    assert low_usage_pressure(peers[0], peers) == 0.0
    assert low_usage_pressure(peers[-1], peers) == 0.0


def test_low_usage_pressure_all_zero_returns_zero():
    peers = [make_memory(pack_count=0, select_count=0) for _ in range(10)]
    for p in peers:
        assert low_usage_pressure(p, peers) == 0.0


def test_low_usage_pressure_ranks_bottom_highest():
    peers = [make_memory(pack_count=i) for i in range(10)]  # 0..9
    bottom = peers[0]
    top = peers[-1]
    p_bottom = low_usage_pressure(bottom, peers)
    p_top = low_usage_pressure(top, peers)
    assert p_bottom == pytest.approx(1.0)
    assert p_top == pytest.approx(0.0)
    assert p_bottom > p_top


def test_low_usage_pressure_monotonic_with_rank():
    peers = [make_memory(pack_count=i) for i in range(10)]
    pressures = [low_usage_pressure(p, peers) for p in peers]
    assert pressures == sorted(pressures, reverse=True)


def test_low_usage_pressure_tiebreak_by_quality_then_age():
    """Identical pack/select: lower quality sorts to bottom (higher pressure)."""
    older = make_memory(pack_count=5, quality_score=0.3, created_at=old(100))
    newer = make_memory(pack_count=5, quality_score=0.3, created_at=old(1))
    filler = [make_memory(pack_count=i + 10) for i in range(8)]
    peers = [older, newer] + filler
    # older has lower pressure than newer when quality is tied (older sorts above newer in
    # the ascending usage list because created_at desc means newer sorts below).
    # Concretely: we defined "older created_at sorts below → higher pressure".
    assert low_usage_pressure(older, peers) > low_usage_pressure(newer, peers)
```

- [ ] **Step 2: Run tests to confirm failure (ImportError)**

Run: `uv run pytest tests/test_forgetting.py -v`
Expected: FAIL at collection — `ImportError: cannot import name 'low_usage_pressure'`.

- [ ] **Step 3: Implement `low_usage_pressure`**

Edit `psa/forgetting.py`. Add after the `_days_since` helper and before `forgetting_score`:

```python
# ── Low-usage pressure (anchor-relative) ─────────────────────────────────────

SMALL_ANCHOR_THRESHOLD = 5


def _usage_sort_key(m: MemoryObject) -> tuple:
    """Ascending sort key: least-used (and weakest on tiebreaks) sorts first.

    Tiebreak order: pack_count, select_count, quality_score, created_at desc.
    A memory that sorts earlier in this list gets HIGHER pressure.
    """
    return (
        m.pack_count,
        m.select_count,
        m.quality_score,
        # Older created_at sorts below newer when everything else ties:
        # negate via string reversal isn't reliable, so we invert by using -ts.
        -_timestamp_for_sort(m.created_at),
    )


def _timestamp_for_sort(iso: Optional[str]) -> float:
    if not iso:
        return 0.0
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def low_usage_pressure(memory: MemoryObject, peers: list) -> float:
    """
    Rank-based pressure within a peer set. Higher = more disposable.

    - Small anchors (< SMALL_ANCHOR_THRESHOLD peers): returns 0.0 (too noisy).
    - All-zero usage population: returns 0.0 (no signal; defer to crowding/quality).
    - Otherwise: bottom-ranked memory returns 1.0; top returns 0.0; linear between.
    """
    n = len(peers)
    if n < SMALL_ANCHOR_THRESHOLD:
        return 0.0
    if all(p.pack_count == 0 and p.select_count == 0 for p in peers):
        return 0.0

    ordered = sorted(peers, key=_usage_sort_key)
    try:
        rank = next(i for i, p in enumerate(ordered) if p.memory_object_id == memory.memory_object_id)
    except StopIteration:
        return 0.0
    if n == 1:
        return 0.0
    return 1.0 - rank / (n - 1)
```

- [ ] **Step 4: Run the five helper tests**

Run: `uv run pytest tests/test_forgetting.py -v -k low_usage_pressure`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add psa/forgetting.py tests/test_forgetting.py
git commit -m "feat(forgetting): add anchor-relative low_usage_pressure helper"
```

---

## Task 4: Replace `idle_days` term in `forgetting_score` with `low_usage_pressure` kwarg

**Files:**
- Modify: `psa/forgetting.py:44-78`

- [ ] **Step 1: Update `forgetting_score` signature and body**

In `psa/forgetting.py`, replace the `forgetting_score` function (lines 44-78) with:

```python
def forgetting_score(
    memory: MemoryObject,
    anchor_size: int,
    target_per_anchor: int = ANCHOR_MEMORY_BUDGET,
    now: Optional[datetime] = None,
    *,
    low_usage_pressure: float = 0.0,
) -> float:
    """
    Compute a forgetting score for a memory. Higher = more disposable.

    Four terms, no tunable weights:
      + low_usage_pressure:   rank-based, precomputed by caller (range [0, 1])
      + crowding pressure:    min(overflow / target, 1.0)
      - usage protection:     min(log(1 + pack_count) / 3.0, 1.0)
      - quality protection:   quality_score

    Range is roughly [-1, 2]. Dormancy alone is NOT a disposal signal.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Grace period: memories created in the last 24 hours are never pruned
    age_days = _days_since(memory.created_at, now)
    if age_days < 1.0:
        return -10.0

    overflow = max(0, anchor_size - target_per_anchor) / max(target_per_anchor, 1)
    usage = log(1 + memory.pack_count) / 3.0

    return (
        low_usage_pressure
        + min(overflow, 1.0)
        - min(usage, 1.0)
        - memory.quality_score
    )
```

- [ ] **Step 2: Add score-behavior tests**

Append to `tests/test_forgetting.py`:

```python
def test_grace_period_preserved():
    fresh = make_memory(created_at=now_utc() - timedelta(hours=1))
    assert forgetting_score(fresh, anchor_size=1, now=now_utc()) == -10.0


def test_heavy_usage_reduces_score():
    light = make_memory(pack_count=0, created_at=old(10))
    heavy = make_memory(pack_count=50, created_at=old(10))
    s_light = forgetting_score(light, anchor_size=10, now=now_utc(), low_usage_pressure=1.0)
    s_heavy = forgetting_score(heavy, anchor_size=10, now=now_utc(), low_usage_pressure=0.0)
    assert s_heavy < s_light


def test_high_quality_reduces_score():
    low_q = make_memory(pack_count=0, quality_score=0.1, created_at=old(10))
    high_q = make_memory(pack_count=0, quality_score=0.9, created_at=old(10))
    s_low = forgetting_score(low_q, anchor_size=10, now=now_utc(), low_usage_pressure=1.0)
    s_high = forgetting_score(high_q, anchor_size=10, now=now_utc(), low_usage_pressure=1.0)
    assert s_high < s_low


def test_score_range_bounds():
    m = make_memory(pack_count=0, quality_score=0.0, created_at=old(10))
    # Upper envelope: pressure=1, overflow=1, usage=0, quality=0 → 2.0
    upper = forgetting_score(m, anchor_size=10_000, now=now_utc(), low_usage_pressure=1.0)
    assert upper <= 2.0 + 1e-9
    # Lower envelope (non-grace): pressure=0, overflow=0, usage=1, quality=1 → -2.0
    strong = make_memory(pack_count=500, quality_score=1.0, created_at=old(10))
    lower = forgetting_score(strong, anchor_size=1, now=now_utc(), low_usage_pressure=0.0)
    assert lower >= -2.0 - 1e-9
```

- [ ] **Step 3: Run the age-insensitivity test plus new score tests**

Run: `uv run pytest tests/test_forgetting.py -v`
Expected: all tests PASS including `test_age_alone_does_not_raise_score`.

- [ ] **Step 4: Commit**

```bash
git add psa/forgetting.py tests/test_forgetting.py
git commit -m "feat(forgetting): remove idle_days term; accept low_usage_pressure kwarg"
```

---

## Task 5: Update `prune_anchor` to compute and pass anchor-local pressure

**Files:**
- Modify: `psa/forgetting.py:84-120`

- [ ] **Step 1: Write a failing pruning-order test**

Append to `tests/test_forgetting.py`:

```python
from psa.forgetting import prune_anchor  # noqa: E402
from psa.memory_object import MemoryStore  # noqa: E402


def _seed_anchor(store: MemoryStore, tenant: str, anchor_id: int, *, count: int, pack_counts: list):
    assert len(pack_counts) == count
    import numpy as np
    ids = []
    for i, pc in enumerate(pack_counts):
        m = make_memory(
            pack_count=pc,
            quality_score=0.5,
            created_at=old(10),
            primary_anchor_id=anchor_id,
            tenant_id=tenant,
        )
        m.embedding = np.zeros(8, dtype="float32").tolist()
        store.add(m)
        ids.append(m.memory_object_id)
    return ids


def test_prune_anchor_archives_local_worst(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = MemoryStore()
    tenant = "t"
    # 12 memories on anchor 42; budget=10; bottom 2 usage should be archived.
    pack_counts = [0, 0, 5, 5, 10, 10, 20, 20, 30, 30, 40, 40]
    ids = _seed_anchor(store, tenant, 42, count=12, pack_counts=pack_counts)

    n_archived = prune_anchor(store, tenant, 42, budget=10, now=now_utc())

    assert n_archived == 2
    surviving = {m.memory_object_id for m in store.query_by_anchor_for_pruning(tenant, 42)}
    # The two pack_count=0 memories are ids[0] and ids[1].
    assert ids[0] not in surviving
    assert ids[1] not in surviving
```

- [ ] **Step 2: Run the test — expect failure from signature mismatch**

Run: `uv run pytest tests/test_forgetting.py::test_prune_anchor_archives_local_worst -v`
Expected: FAIL — `prune_anchor` is calling `forgetting_score` without the new kwarg, so rankings use pressure=0 everywhere and ordering falls back to `(pack_count asc)` from `query_by_anchor_for_pruning`'s ORDER BY. **The test may coincidentally pass on the happy path because the DB pre-sorts by `pack_count ASC`; if so, strengthen the assertion by scrambling `pack_counts` or add a test that relies on `low_usage_pressure` explicitly.**

(If it passes, continue — Step 3 is still required for correctness and the score-regression tests.)

- [ ] **Step 3: Update `prune_anchor`**

Replace the body of `prune_anchor` (lines 84-120) with:

```python
def prune_anchor(
    store: MemoryStore,
    tenant_id: str,
    anchor_id: int,
    budget: int = ANCHOR_MEMORY_BUDGET,
    now: Optional[datetime] = None,
) -> int:
    """
    If an anchor exceeds its memory budget, archive the lowest-value memories.

    Uses anchor-relative low-usage pressure (computed over the anchor's own
    memories) as the primary disposal signal.

    Returns the number of memories archived.
    """
    memories = store.query_by_anchor_for_pruning(tenant_id, anchor_id)
    if len(memories) <= budget:
        return 0

    if now is None:
        now = datetime.now(timezone.utc)

    anchor_size = len(memories)
    pressure_by_id = {
        m.memory_object_id: low_usage_pressure(m, memories) for m in memories
    }

    def score(m: MemoryObject) -> float:
        return forgetting_score(
            m,
            anchor_size,
            budget,
            now,
            low_usage_pressure=pressure_by_id[m.memory_object_id],
        )

    scored = sorted(memories, key=score, reverse=True)
    to_archive = scored[: len(memories) - budget]
    archive_ids = [m.memory_object_id for m in to_archive]
    store.archive_memories(archive_ids)

    logger.info(
        "Pruned anchor %d: archived %d memories (was %d, budget %d)",
        anchor_id,
        len(archive_ids),
        anchor_size,
        budget,
    )
    return len(archive_ids)
```

- [ ] **Step 4: Run the anchor pruning test**

Run: `uv run pytest tests/test_forgetting.py::test_prune_anchor_archives_local_worst -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add psa/forgetting.py tests/test_forgetting.py
git commit -m "feat(forgetting): use anchor-relative pressure in prune_anchor"
```

---

## Task 6: Rewrite `enforce_global_cap` with hybrid anchor+tenant pressure

**Files:**
- Modify: `psa/forgetting.py:126-177`

- [ ] **Step 1: Write the failing cross-anchor test**

Append to `tests/test_forgetting.py`:

```python
from psa.forgetting import enforce_global_cap  # noqa: E402


def test_global_cap_uses_hybrid_cross_anchor_score(tmp_path, monkeypatch):
    """Two anchors. Anchor A is busy (high pack_counts); anchor B is quiet.
    Cap forces 2 archives. The two least-used memories overall must be
    archived, not two from one anchor alone."""
    monkeypatch.setenv("HOME", str(tmp_path))
    store = MemoryStore()
    tenant = "t"

    a_ids = _seed_anchor(store, tenant, 1, count=6, pack_counts=[50, 60, 70, 80, 90, 100])
    b_ids = _seed_anchor(store, tenant, 2, count=6, pack_counts=[0, 1, 2, 3, 4, 5])

    result = enforce_global_cap(store, tenant, max_memories=10)

    # 12 total, cap 10 → 2 archives.
    assert result["archived"] == 2
    # The two lowest-usage overall are b_ids[0] (pc=0) and b_ids[1] (pc=1).
    from psa.memory_object import MemoryStore as _MS  # noqa
    surviving_a = {m.memory_object_id for m in store.query_by_anchor_for_pruning(tenant, 1)}
    surviving_b = {m.memory_object_id for m in store.query_by_anchor_for_pruning(tenant, 2)}
    assert a_ids[0] in surviving_a  # busy anchor's weakest still survives
    assert b_ids[0] not in surviving_b
    assert b_ids[1] not in surviving_b
```

- [ ] **Step 2: Run the test — expect failure**

Run: `uv run pytest tests/test_forgetting.py::test_global_cap_uses_hybrid_cross_anchor_score -v`
Expected: FAIL — current `enforce_global_cap` passes `low_usage_pressure=0` implicitly, so ranking falls back to `-usage_protection - quality` which favors low-`pack_count` but still passes coincidentally in some runs. If it passes coincidentally, Step 3 is still required for the documented hybrid behavior.

- [ ] **Step 3: Update `enforce_global_cap` with hybrid score**

Replace `enforce_global_cap` (lines 126-177) with:

```python
GLOBAL_CAP_ANCHOR_WEIGHT = 0.7
GLOBAL_CAP_TENANT_WEIGHT = 0.3


def enforce_global_cap(
    store: MemoryStore,
    tenant_id: str,
    max_memories: int = MAX_MEMORIES,
    archived_ttl_days: int = 90,
) -> dict:
    """
    Enforce a global memory cap for a tenant.

    Phase 1: hard-delete memories archived > archived_ttl_days ago.
    Phase 2: if still over cap, archive the lowest-scoring active memories
             using a hybrid (anchor-relative + tenant-wide) pressure score.

    Returns a dict with counts of actions taken.
    """
    result = {"hard_deleted": 0, "archived": 0}

    deleted = store.delete_old_archived(tenant_id, older_than_days=archived_ttl_days)
    result["hard_deleted"] = deleted
    if deleted:
        logger.info("Global cap: hard-deleted %d old archived memories", deleted)

    active_count = store.count(tenant_id)
    if active_count <= max_memories:
        return result

    excess = active_count - max_memories
    now = datetime.now(timezone.utc)

    all_memories = store.get_all_with_embeddings(tenant_id)
    if not all_memories:
        return result

    # Group by primary_anchor_id for anchor-local pressure.
    by_anchor: dict = {}
    for m in all_memories:
        by_anchor.setdefault(m.primary_anchor_id, []).append(m)

    anchor_pressure = {
        m.memory_object_id: low_usage_pressure(m, by_anchor[m.primary_anchor_id])
        for m in all_memories
    }
    tenant_pressure = {
        m.memory_object_id: low_usage_pressure(m, all_memories) for m in all_memories
    }

    def score(m: MemoryObject) -> float:
        hybrid = (
            GLOBAL_CAP_ANCHOR_WEIGHT * anchor_pressure[m.memory_object_id]
            + GLOBAL_CAP_TENANT_WEIGHT * tenant_pressure[m.memory_object_id]
        )
        return forgetting_score(
            m,
            active_count,
            ANCHOR_MEMORY_BUDGET,
            now,
            low_usage_pressure=hybrid,
        )

    scored = sorted(all_memories, key=score, reverse=True)
    to_archive = scored[:excess]
    archive_ids = [m.memory_object_id for m in to_archive]
    store.archive_memories(archive_ids)
    result["archived"] = len(archive_ids)

    logger.info(
        "Global cap: archived %d memories (was %d, cap %d)",
        len(archive_ids),
        active_count,
        max_memories,
    )
    return result
```

- [ ] **Step 4: Run the hybrid test plus the whole file**

Run: `uv run pytest tests/test_forgetting.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add psa/forgetting.py tests/test_forgetting.py
git commit -m "feat(forgetting): hybrid anchor+tenant pressure in enforce_global_cap"
```

---

## Task 7: Add shadow-mode logging for one-release observation

**Files:**
- Modify: `psa/forgetting.py`

- [ ] **Step 1: Add shadow log in `prune_anchor`**

In `psa/forgetting.py`, inside `prune_anchor`, after `archive_ids = ...` and before the existing `logger.info`, add:

```python
    for rank, m in enumerate(scored[: len(memories) - budget]):
        logger.info(
            "forgetting.shadow anchor=%d mo=%s archived=1 pressure=%.3f score=%.3f",
            anchor_id,
            m.memory_object_id,
            pressure_by_id[m.memory_object_id],
            score(m),
        )
```

- [ ] **Step 2: Add shadow log in `enforce_global_cap`**

Inside `enforce_global_cap`, after `archive_ids = ...`:

```python
    for m in to_archive:
        logger.info(
            "forgetting.shadow global mo=%s anchor=%d anchor_p=%.3f tenant_p=%.3f score=%.3f",
            m.memory_object_id,
            m.primary_anchor_id,
            anchor_pressure[m.memory_object_id],
            tenant_pressure[m.memory_object_id],
            score(m),
        )
```

- [ ] **Step 3: Run full test file to confirm no regression**

Run: `uv run pytest tests/test_forgetting.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add psa/forgetting.py
git commit -m "chore(forgetting): add per-archive shadow logging for rollout validation"
```

---

## Task 8: Update README to describe the new forgetting model

**Files:**
- Modify: `README.md:402-414`

- [ ] **Step 1: Replace the forgetting-score block**

In `README.md`, replace lines 402-414 with:

```
### Memory forgetting

`psa/forgetting.py` scores each memory for disposability. **Time alone is not a disposal signal.** A memory that has gone a year without being packed is not more disposable than one packed yesterday — what matters is how it compares to its peers under pressure.

Per-anchor pruning uses:

```
forgetting_score =
  + low_usage_pressure_within_anchor    # bottom of anchor's usage distribution → 1.0; top → 0.0
  + min(overflow / budget, 1.0)          # crowding pressure — anchor over its 100-memory budget
  - min(log(1 + pack_count) / 3.0, 1.0)  # usage protection — 20 packs ≈ full protection
  - quality_score                        # quality protection — set once at extraction
```

Global-cap enforcement uses the same formula but substitutes a hybrid pressure: `0.7 * anchor_local + 0.3 * tenant_wide`, so scores remain comparable across anchors.

Range is roughly −2 to 2. Score > 0 is a candidate for pruning. Newly ingested memories get a 24-hour absolute grace period (score forced to −10). Low-scoring memories are archived (soft delete) and hard-deleted after 90 days in archive. The global cap is 50k memories; the per-anchor budget is 100.

Small-anchor stability rule: anchors with fewer than 5 active memories get `low_usage_pressure = 0`, so pruning there is driven entirely by crowding and quality (percentiles on tiny samples are too noisy to trust).
```

- [ ] **Step 2: Verify no other README references to `idle_days`**

Run: (use Grep tool) search for `idle_days` in `README.md`.
Expected: no matches remain outside the replaced block.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite forgetting-score README section for relative-usage model"
```

---

## Task 9: Lint, full test suite, manual lifecycle smoke

**Files:** none.

- [ ] **Step 1: Ruff check**

Run: `uv run ruff check psa/forgetting.py tests/test_forgetting.py`
Expected: `All checks passed.`

- [ ] **Step 2: Ruff format check**

Run: `uv run ruff format --check psa/forgetting.py tests/test_forgetting.py`
Expected: clean. If not, run `uv run ruff format psa/forgetting.py tests/test_forgetting.py` and amend.

- [ ] **Step 3: Full test suite**

Run: `uv run pytest tests/ -v`
Expected: all PASS. Pay attention to `tests/test_lifecycle.py` in particular.

- [ ] **Step 4: Manual lifecycle smoke on default tenant**

Run: `uv run psa lifecycle run 2>&1 | grep forgetting.shadow | head -20`
Expected: shadow log lines print with plausible pressure/score values. If no archives happened (tenant under pressure thresholds), the grep will be empty — that's fine; archive-pressure isn't always hit.

- [ ] **Step 5: Final commit if anything was touched by ruff format**

```bash
git status
# If files modified by ruff format:
git add -u
git commit -m "style: apply ruff format"
```

---

## Success Criteria

- `idle_days` no longer appears in `psa/forgetting.py` (verify with Grep).
- `tests/test_forgetting.py` exists with all 12+ tests passing.
- `uv run ruff check .` and `uv run pytest tests/` green.
- `README.md` forgetting section reflects the new model.
- Shadow log lines emitted by `psa lifecycle run` on a tenant with pruning activity.
- No DB schema migration required; no config flag added; no other module modified.

## Rollback

Single `git revert` of the commit range produces a clean reversion. The only state mutation done during the new code's operation is `is_archived = 1` on some memory rows — the old code reads the same column, so previously-archived rows stay archived. No data loss path.

## Open items left by the spec

These are deliberately deferred — the plan does not implement them:

- `select_count` as an independent protection term (it is used here only as a lexicographic tiebreak inside `_usage_sort_key`).
- Removing the shadow log after a one-release observation window (a follow-up PR, not part of this plan).
- Any tuning of `GLOBAL_CAP_ANCHOR_WEIGHT` / `GLOBAL_CAP_TENANT_WEIGHT` from the 0.7/0.3 starting point — the shadow log informs that decision.
