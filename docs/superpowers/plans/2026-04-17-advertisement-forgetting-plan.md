# Advertisement Forgetting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `psa atlas decay` — a candidate-side pass that removes stale `generated_query_patterns` from anchor cards based on derived reinforcement from the query trace, subject to a grace period, low-activation shielding, and operator pinning. Atlas rebuild carries pattern metadata forward so the forgetting clock survives rebuilds.

**Architecture:** New `psa/forgetting/` package split into four small modules (metadata storage, reinforcement derivation, decay orchestration, candidate writer). `psa/atlas.py`'s rebuild gains a pattern-metadata inheritance step mirroring the existing `FingerprintStore.inherit_from()` pattern. CLI exposes `psa atlas decay` with `--dry-run`. Output reuses Branch 1's candidate/promote flow; detailed removals go to a sibling `.decay_report.json` so `.meta.json` stays summary-oriented.

**Tech Stack:** Python 3.13, existing `AtlasManager` / `FingerprintStore` patterns, JSONL trace reader from Branch 4.

**Design spec:** `docs/superpowers/specs/2026-04-17-advertisement-forgetting-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `psa/forgetting/__init__.py` | Create | Package marker. |
| `psa/forgetting/metadata.py` | Create | `PatternMetadata` dataclass, `normalize_pattern`, `metadata_key`, atomic `load_metadata` / `save_metadata`, `backfill_unknown`. |
| `psa/forgetting/reinforcement.py` | Create | `compute_reinforcement(atlas, trace_path, origins, window_start) -> dict[str, datetime]`. Ephemeral per-run map. |
| `psa/forgetting/decay.py` | Create | `DecayParams`, `DecayReport`, `RemovedPattern`, `ShieldedAnchor` dataclasses. `decay_report(tenant_id, params, origins) -> DecayReport` orchestrator. D1 + P1 + P3 logic. |
| `psa/forgetting/writer.py` | Create | `write_decay_candidate(atlas_dir, report)` — writes candidate + summary meta + detailed report. Empty-run guard. |
| `psa/atlas.py` | Modify | `AtlasBuilder.build_atlas()` gains a pattern-metadata inheritance pass right after the fingerprint inheritance (around line 755). |
| `psa/cli.py` | Modify | Add `_cmd_atlas_decay` handler + `decay` subparser + dispatch. |
| `tests/test_forgetting_metadata.py` | Create | Normalization, key format, atomic write, load missing file, backfill behavior, reserved fields tolerance. |
| `tests/test_forgetting_reinforcement.py` | Create | Substring + activation semantics, origin filtering, window bound, ephemeral-only (no write-back). |
| `tests/test_forgetting_decay.py` | Create | D1 rule, P1 two-part shield, P3 pin respect, source-grouped counts, empty-run detection. |
| `tests/test_forgetting_writer.py` | Create | Real write produces 3 files; dry-run writes none (metadata backfill is caller's job, not writer's). Empty-run skips all writes. |
| `tests/test_atlas_metadata_inheritance.py` | Create | Rebuild copies matched patterns, drops orphans, leaves new patterns unstamped. |
| `tests/test_cli_atlas_decay.py` | Create | End-to-end CLI: dry-run prints summary, real writes all three files, JSON envelope shape, backfill happens in both modes, coexistence with refine/curate candidates. |

---

### Task 1: metadata module

Foundation. All downstream tasks import from here.

**Files:**
- Create: `psa/forgetting/__init__.py`
- Create: `psa/forgetting/metadata.py`
- Create: `tests/test_forgetting_metadata.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_forgetting_metadata.py`:

```python
"""Tests for psa.forgetting.metadata — pattern metadata storage."""

from __future__ import annotations

import json
from pathlib import Path


def test_normalize_pattern_strips_and_lowercases():
    from psa.forgetting.metadata import normalize_pattern

    assert normalize_pattern("  HELLO World  ") == "hello world"


def test_normalize_pattern_collapses_whitespace():
    from psa.forgetting.metadata import normalize_pattern

    assert normalize_pattern("foo   bar\t baz") == "foo bar baz"


def test_metadata_key_format():
    from psa.forgetting.metadata import metadata_key

    assert metadata_key(227, "What Is This?") == "227::what is this?"


def test_load_metadata_missing_file_returns_empty_dict(tmp_path):
    from psa.forgetting.metadata import load_metadata

    assert load_metadata(str(tmp_path)) == {}


def test_save_and_load_roundtrip(tmp_path):
    from psa.forgetting.metadata import PatternMetadata, load_metadata, save_metadata

    meta = {
        "1::hello world": PatternMetadata(
            anchor_id=1,
            pattern="hello world",
            source="atlas_build",
            created_at="2026-04-17T00:00:00+00:00",
        )
    }
    save_metadata(str(tmp_path), meta)
    loaded = load_metadata(str(tmp_path))
    assert set(loaded.keys()) == {"1::hello world"}
    assert loaded["1::hello world"].anchor_id == 1
    assert loaded["1::hello world"].source == "atlas_build"
    assert loaded["1::hello world"].created_at == "2026-04-17T00:00:00+00:00"


def test_save_is_atomic_via_tmp_file(tmp_path):
    """The writer uses os.replace — no half-written state if interrupted."""
    from psa.forgetting.metadata import PatternMetadata, save_metadata

    meta = {
        "1::x": PatternMetadata(anchor_id=1, pattern="x", source="manual",
                                created_at="2026-04-17T00:00:00+00:00"),
    }
    save_metadata(str(tmp_path), meta)
    # After save, the tmp file should not remain.
    assert not (tmp_path / "pattern_metadata.json.tmp").exists()
    assert (tmp_path / "pattern_metadata.json").exists()


def test_load_tolerates_unknown_future_fields(tmp_path):
    """Extra keys in the JSON entries (e.g., reserved pinned, promoted_at) don't crash load."""
    from psa.forgetting.metadata import load_metadata

    raw = {
        "1::x": {
            "anchor_id": 1,
            "pattern": "x",
            "source": "manual",
            "created_at": "2026-04-17T00:00:00+00:00",
            "pinned": True,
            "promoted_at": "2026-04-18T00:00:00+00:00",
        }
    }
    (tmp_path / "pattern_metadata.json").write_text(json.dumps(raw))
    loaded = load_metadata(str(tmp_path))
    assert loaded["1::x"].pinned is True  # Supported in schema even if unused.


def test_backfill_unknown_stamps_new_patterns(tmp_path):
    """Patterns absent from metadata get source='unknown' + created_at=<now> entries.

    Does NOT mutate entries that already exist.
    """
    from psa.forgetting.metadata import (
        PatternMetadata,
        backfill_unknown,
        metadata_key,
    )

    existing = {
        metadata_key(1, "old pattern"): PatternMetadata(
            anchor_id=1, pattern="old pattern", source="manual",
            created_at="2020-01-01T00:00:00+00:00",
        )
    }
    # Anchor 1 now has two patterns; anchor 2 is new.
    patterns_by_anchor = {
        1: ["old pattern", "fresh pattern"],
        2: ["brand new"],
    }
    now_iso = "2026-04-17T12:00:00+00:00"
    n_backfilled = backfill_unknown(existing, patterns_by_anchor, now_iso)

    # Two new entries stamped; one pre-existing entry untouched.
    assert n_backfilled == 2
    assert existing[metadata_key(1, "old pattern")].created_at == "2020-01-01T00:00:00+00:00"
    assert existing[metadata_key(1, "fresh pattern")].source == "unknown"
    assert existing[metadata_key(1, "fresh pattern")].created_at == now_iso
    assert existing[metadata_key(2, "brand new")].source == "unknown"
```

- [ ] **Step 2: Run tests — expect ModuleNotFoundError**

```
uv run pytest tests/test_forgetting_metadata.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'psa.forgetting'`.

- [ ] **Step 3: Create `psa/forgetting/__init__.py`**

```python
"""psa.forgetting — advertisement forgetting (pattern-level card decay)."""
```

- [ ] **Step 4: Create `psa/forgetting/metadata.py`**

```python
"""
metadata.py — Per-pattern provenance storage in pattern_metadata.json.

Sibling file to anchor_cards.json. Holds only provenance (source, created_at,
optional pinned flag) — never dynamic counters. Dynamic reinforcement signal
is derived per-run from the trace log.

Key format: "{anchor_id}::{normalized_pattern}" where normalize_pattern
collapses whitespace, strips, and lowercases. Stable across formatting drift.

Writes are atomic (tmp + os.replace) so concurrent readers never observe
a half-written file.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Optional

logger = logging.getLogger("psa.forgetting.metadata")

FILENAME = "pattern_metadata.json"


def normalize_pattern(text: str) -> str:
    """Normalize pattern text for key-stable storage.

    Collapses internal whitespace, strips leading/trailing whitespace, lowercases.
    Logical identity: cases and spacing drift don't create fake new entries.
    """
    return " ".join(text.strip().lower().split())


def metadata_key(anchor_id: int, pattern: str) -> str:
    """Compose the metadata key for an (anchor_id, pattern) pair."""
    return f"{anchor_id}::{normalize_pattern(pattern)}"


@dataclass
class PatternMetadata:
    """Provenance for a single anchor-card pattern.

    Stored on disk; reinforcement state (last_reinforced_at) is computed per-run
    and never written here — see psa/forgetting/reinforcement.py.

    Reserved fields (schema-forward, not consumed in this branch):
        promoted_at — when the pattern first landed on anchor_cards_refined.json.
                      Absent/None until the creation-time stamping follow-up lands.
        pinned — P3 operator-pinned flag. Supported in schema; no CLI to set yet.
    """

    anchor_id: int
    pattern: str
    source: str  # "atlas_build" | "refinement" | "production_signal" | "manual" | "unknown"
    created_at: str  # ISO-8601 UTC timestamp
    promoted_at: Optional[str] = None
    pinned: bool = False


def load_metadata(atlas_dir: str) -> dict[str, PatternMetadata]:
    """Load pattern_metadata.json from atlas_dir. Missing file → empty dict.

    Tolerates unknown keys in entries (forward compat). Malformed entries are
    skipped with a debug log.
    """
    path = os.path.join(atlas_dir, FILENAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return {}

    out: dict[str, PatternMetadata] = {}
    known_fields = {"anchor_id", "pattern", "source", "created_at", "promoted_at", "pinned"}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        try:
            filtered = {k: v for k, v in entry.items() if k in known_fields}
            out[key] = PatternMetadata(**filtered)
        except TypeError as e:
            logger.debug("Skipping malformed metadata entry %s: %s", key, e)
    return out


def save_metadata(atlas_dir: str, metadata: dict[str, PatternMetadata]) -> None:
    """Persist metadata atomically (write tmp, os.replace to final)."""
    os.makedirs(atlas_dir, exist_ok=True)
    final_path = os.path.join(atlas_dir, FILENAME)
    tmp_path = final_path + ".tmp"
    serialized = {key: asdict(meta) for key, meta in metadata.items()}
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)
    os.replace(tmp_path, final_path)


def backfill_unknown(
    metadata: dict[str, PatternMetadata],
    patterns_by_anchor: dict[int, list[str]],
    now_iso: str,
) -> int:
    """Stamp missing entries with source='unknown', created_at=now.

    Conservative migration policy — see spec §5.2. Does NOT mutate entries
    that already exist.

    Returns the count of entries newly added.
    """
    n_added = 0
    for anchor_id, patterns in patterns_by_anchor.items():
        for pattern in patterns:
            key = metadata_key(anchor_id, pattern)
            if key in metadata:
                continue
            metadata[key] = PatternMetadata(
                anchor_id=anchor_id,
                pattern=pattern,
                source="unknown",
                created_at=now_iso,
            )
            n_added += 1
    return n_added
```

- [ ] **Step 5: Run tests — expect PASS**

```
uv run pytest tests/test_forgetting_metadata.py -v
```

Expected: all 8 PASS.

- [ ] **Step 6: Commit**

```bash
git add psa/forgetting/__init__.py psa/forgetting/metadata.py tests/test_forgetting_metadata.py
git commit -m "$(cat <<'EOF'
feat: psa.forgetting.metadata — pattern provenance storage

PatternMetadata dataclass (anchor_id, pattern, source, created_at,
promoted_at reserved, pinned reserved). Sibling file
pattern_metadata.json per atlas version. Atomic writes via os.replace.
backfill_unknown stamps previously-unstamped patterns with
source='unknown' + created_at=now (conservative migration policy).

Reinforcement state is intentionally absent from this module — it is
derived per-run from the trace log, never persisted.

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: reinforcement derivation

Reads the trace log, returns an ephemeral map. No persistence.

**Files:**
- Create: `psa/forgetting/reinforcement.py`
- Create: `tests/test_forgetting_reinforcement.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_forgetting_reinforcement.py`:

```python
"""Tests for psa.forgetting.reinforcement — derived per-run reinforcement map."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock


def _write_trace(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _fake_atlas_with_patterns(patterns_by_anchor: dict[int, list[str]]) -> MagicMock:
    atlas = MagicMock()
    cards = []
    for aid, patterns in patterns_by_anchor.items():
        card = MagicMock()
        card.anchor_id = aid
        card.generated_query_patterns = patterns
        cards.append(card)
    atlas.cards = cards
    return atlas


def test_reinforcement_substring_match_activates(tmp_path):
    from psa.forgetting.reinforcement import compute_reinforcement

    trace_path = tmp_path / "query_trace.jsonl"
    _write_trace(trace_path, [
        {
            "timestamp": "2026-04-15T10:00:00+00:00",
            "query": "how does the token refresh flow work in prod",
            "query_origin": "interactive",
            "selected_anchor_ids": [1],
        },
    ])
    atlas = _fake_atlas_with_patterns({1: ["token refresh flow"]})

    window_start = datetime.fromisoformat("2026-04-01T00:00:00+00:00")
    rmap = compute_reinforcement(
        atlas, str(trace_path),
        origins={"interactive"},
        window_start=window_start,
    )
    assert "1::token refresh flow" in rmap


def test_reinforcement_without_activation_does_not_count(tmp_path):
    """Anchor didn't activate — pattern is not reinforced even if substring matches."""
    from psa.forgetting.reinforcement import compute_reinforcement

    trace_path = tmp_path / "query_trace.jsonl"
    _write_trace(trace_path, [
        {
            "timestamp": "2026-04-15T10:00:00+00:00",
            "query": "token refresh flow details",
            "query_origin": "interactive",
            "selected_anchor_ids": [2],  # anchor 2 selected, not 1
        },
    ])
    atlas = _fake_atlas_with_patterns({1: ["token refresh flow"]})

    rmap = compute_reinforcement(
        atlas, str(trace_path),
        origins={"interactive"},
        window_start=datetime.fromisoformat("2026-04-01T00:00:00+00:00"),
    )
    assert "1::token refresh flow" not in rmap


def test_reinforcement_origin_filter_excludes_benchmark(tmp_path):
    from psa.forgetting.reinforcement import compute_reinforcement

    trace_path = tmp_path / "query_trace.jsonl"
    _write_trace(trace_path, [
        {
            "timestamp": "2026-04-15T10:00:00+00:00",
            "query": "token refresh flow",
            "query_origin": "benchmark",
            "selected_anchor_ids": [1],
        },
    ])
    atlas = _fake_atlas_with_patterns({1: ["token refresh flow"]})

    rmap = compute_reinforcement(
        atlas, str(trace_path),
        origins={"interactive"},
        window_start=datetime.fromisoformat("2026-04-01T00:00:00+00:00"),
    )
    assert "1::token refresh flow" not in rmap


def test_reinforcement_window_bound_excludes_old_records(tmp_path):
    from psa.forgetting.reinforcement import compute_reinforcement

    trace_path = tmp_path / "query_trace.jsonl"
    _write_trace(trace_path, [
        {
            "timestamp": "2020-01-01T00:00:00+00:00",   # way before window
            "query": "token refresh flow",
            "query_origin": "interactive",
            "selected_anchor_ids": [1],
        },
    ])
    atlas = _fake_atlas_with_patterns({1: ["token refresh flow"]})

    rmap = compute_reinforcement(
        atlas, str(trace_path),
        origins={"interactive"},
        window_start=datetime.fromisoformat("2026-04-01T00:00:00+00:00"),
    )
    assert "1::token refresh flow" not in rmap


def test_reinforcement_takes_most_recent_match(tmp_path):
    from psa.forgetting.reinforcement import compute_reinforcement

    trace_path = tmp_path / "query_trace.jsonl"
    _write_trace(trace_path, [
        {
            "timestamp": "2026-04-10T00:00:00+00:00",
            "query": "token refresh flow",
            "query_origin": "interactive",
            "selected_anchor_ids": [1],
        },
        {
            "timestamp": "2026-04-15T00:00:00+00:00",
            "query": "refresh flow token",  # still substring-matches "token refresh flow"? No.
            "query_origin": "interactive",
            "selected_anchor_ids": [1],
        },
        {
            "timestamp": "2026-04-20T00:00:00+00:00",
            "query": "the token refresh flow in prod",
            "query_origin": "interactive",
            "selected_anchor_ids": [1],
        },
    ])
    atlas = _fake_atlas_with_patterns({1: ["token refresh flow"]})

    rmap = compute_reinforcement(
        atlas, str(trace_path),
        origins={"interactive"},
        window_start=datetime.fromisoformat("2026-04-01T00:00:00+00:00"),
    )
    # First and third records match. Should hold the LATER timestamp.
    got = rmap["1::token refresh flow"]
    assert got == datetime.fromisoformat("2026-04-20T00:00:00+00:00")


def test_reinforcement_missing_trace_file_returns_empty(tmp_path):
    from psa.forgetting.reinforcement import compute_reinforcement

    atlas = _fake_atlas_with_patterns({1: ["anything"]})
    rmap = compute_reinforcement(
        atlas, str(tmp_path / "does_not_exist.jsonl"),
        origins={"interactive"},
        window_start=datetime.fromisoformat("2026-04-01T00:00:00+00:00"),
    )
    assert rmap == {}
```

- [ ] **Step 2: Run tests — expect ModuleNotFoundError**

```
uv run pytest tests/test_forgetting_reinforcement.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `psa/forgetting/reinforcement.py`**

```python
"""
reinforcement.py — derive last_reinforced_at per pattern from the trace log.

Principle: reinforcement is computed per-run into an in-memory dict and
never written to disk. The trace log (Branch 4) is the authoritative source
of dynamic signal; re-deriving every run keeps persistent metadata minimal
and drift-free.

Reinforcement rule (R1, see spec §2):
    A trace record reinforces pattern P on anchor A when
        - record.selected_anchor_ids contains A, AND
        - record.query_origin is in the active origins set, AND
        - normalize(P) is a substring of normalize(record.query).
    last_reinforced_at[P] = latest qualifying record timestamp.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from .metadata import metadata_key, normalize_pattern

logger = logging.getLogger("psa.forgetting.reinforcement")


def compute_reinforcement(
    atlas: Any,
    trace_path: str,
    *,
    origins: set[str],
    window_start: datetime,
) -> dict[str, datetime]:
    """Return {metadata_key: last_reinforced_at} for every reinforced pattern.

    The window_start bound is a conservative optimization: records older than
    (decay_window_days + grace_days) cannot affect the decay decision. Widening
    it later does not change the rule's semantics, just scans more records.

    Missing trace file → returns {}.
    """
    if not os.path.exists(trace_path):
        return {}

    # Build a pattern index: anchor_id -> list of (normalized_pattern, raw_pattern)
    pattern_index: dict[int, list[tuple[str, str]]] = {}
    for card in atlas.cards:
        patterns = getattr(card, "generated_query_patterns", []) or []
        if patterns:
            pattern_index[card.anchor_id] = [
                (normalize_pattern(p), p) for p in patterns
            ]

    rmap: dict[str, datetime] = {}
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed trace line")
                continue

            ts_str = rec.get("timestamp")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue
            if ts < window_start:
                continue

            if rec.get("query_origin", "interactive") not in origins:
                continue

            selected_ids = rec.get("selected_anchor_ids") or []
            if not selected_ids:
                continue

            q_norm = normalize_pattern(rec.get("query", ""))
            if not q_norm:
                continue

            for aid in selected_ids:
                for norm_p, raw_p in pattern_index.get(aid, []):
                    if norm_p and norm_p in q_norm:
                        key = metadata_key(aid, raw_p)
                        if key not in rmap or ts > rmap[key]:
                            rmap[key] = ts

    return rmap
```

- [ ] **Step 4: Run tests — expect PASS**

```
uv run pytest tests/test_forgetting_reinforcement.py -v
```

Expected: all 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add psa/forgetting/reinforcement.py tests/test_forgetting_reinforcement.py
git commit -m "$(cat <<'EOF'
feat: psa.forgetting.reinforcement — derive reinforcement from trace

compute_reinforcement() reads query_trace.jsonl, returns an ephemeral
{key: datetime} map. R1 rule: activation + substring within origin set
+ window. Ephemeral by design — never persisted, avoids drift from
the trace. Missing trace file is soft (returns empty map).

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: decay orchestrator

The heart of the feature. D1 rule, P1 shield (two-part), P3 pin respect.

**Files:**
- Create: `psa/forgetting/decay.py`
- Create: `tests/test_forgetting_decay.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_forgetting_decay.py`:

```python
"""Tests for psa.forgetting.decay — D1 rule + P1 shield + P3 pin."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch


def _fake_atlas(patterns_by_anchor: dict[int, list[str]]) -> MagicMock:
    atlas = MagicMock()
    cards = []
    for aid, patterns in patterns_by_anchor.items():
        card = MagicMock()
        card.anchor_id = aid
        card.name = f"anchor-{aid}"
        card.generated_query_patterns = list(patterns)
        cards.append(card)
    atlas.cards = cards
    atlas.anchor_dir = "/unused/in/tests"
    return atlas


def _write_trace(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


NOW = datetime.fromisoformat("2026-04-17T12:00:00+00:00")


def test_d1_pattern_too_young_is_not_candidate(tmp_path, monkeypatch):
    """Age < grace_days → pattern is not a decay candidate, even without reinforcement."""
    from psa.forgetting.decay import DecayParams, decay_report

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir / "query_trace.jsonl", [])

    atlas = _fake_atlas({1: ["young pattern"]})
    from psa.forgetting.metadata import PatternMetadata, metadata_key, save_metadata
    atlas.anchor_dir = str(tmp_path / "atlas_dir")
    (Path(atlas.anchor_dir)).mkdir(parents=True, exist_ok=True)
    recently = (NOW - timedelta(days=5)).isoformat()
    save_metadata(atlas.anchor_dir, {
        metadata_key(1, "young pattern"): PatternMetadata(
            anchor_id=1, pattern="young pattern", source="manual",
            created_at=recently,
        )
    })

    params = DecayParams(grace_days=30, decay_window_days=60,
                         low_activation_percentile=0, min_anchor_activations=0)
    with patch("psa.forgetting.decay._load_atlas_for_tenant", return_value=atlas):
        with patch("psa.forgetting.decay._now_utc", return_value=NOW):
            report = decay_report("default", params=params, origins={"interactive"})
    assert report.n_patterns_removed == 0


def test_d1_old_unreinforced_is_candidate(tmp_path, monkeypatch):
    from psa.forgetting.decay import DecayParams, decay_report
    from psa.forgetting.metadata import PatternMetadata, metadata_key, save_metadata

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir / "query_trace.jsonl", [])

    atlas = _fake_atlas({1: ["stale pattern"]})
    atlas.anchor_dir = str(tmp_path / "atlas_dir")
    (Path(atlas.anchor_dir)).mkdir(parents=True, exist_ok=True)
    long_ago = (NOW - timedelta(days=200)).isoformat()
    save_metadata(atlas.anchor_dir, {
        metadata_key(1, "stale pattern"): PatternMetadata(
            anchor_id=1, pattern="stale pattern", source="manual",
            created_at=long_ago,
        )
    })

    params = DecayParams(grace_days=30, decay_window_days=60,
                         low_activation_percentile=0, min_anchor_activations=0)
    with patch("psa.forgetting.decay._load_atlas_for_tenant", return_value=atlas):
        with patch("psa.forgetting.decay._now_utc", return_value=NOW):
            report = decay_report("default", params=params, origins={"interactive"})
    assert report.n_patterns_removed == 1
    assert report.removed_patterns[0].anchor_id == 1
    assert report.removed_patterns[0].pattern == "stale pattern"
    assert report.removed_patterns[0].reason == "stale_unreinforced"


def test_p1_shield_below_absolute_floor(tmp_path, monkeypatch):
    """Anchor below min_anchor_activations is shielded — patterns NOT candidates."""
    from psa.forgetting.decay import DecayParams, decay_report
    from psa.forgetting.metadata import PatternMetadata, metadata_key, save_metadata

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    # Anchor 1 activated only 3 times in the window.
    _write_trace(tenant_dir / "query_trace.jsonl", [
        {"timestamp": "2026-04-10T00:00:00+00:00", "query": "q",
         "query_origin": "interactive", "selected_anchor_ids": [1]}
        for _ in range(3)
    ])

    atlas = _fake_atlas({1: ["stale pattern"]})
    atlas.anchor_dir = str(tmp_path / "atlas_dir")
    (Path(atlas.anchor_dir)).mkdir(parents=True, exist_ok=True)
    long_ago = (NOW - timedelta(days=200)).isoformat()
    save_metadata(atlas.anchor_dir, {
        metadata_key(1, "stale pattern"): PatternMetadata(
            anchor_id=1, pattern="stale pattern", source="manual", created_at=long_ago,
        )
    })

    # Floor is 10 activations; anchor 1 has 3. Shielded.
    params = DecayParams(grace_days=30, decay_window_days=60,
                         low_activation_percentile=0, min_anchor_activations=10)
    with patch("psa.forgetting.decay._load_atlas_for_tenant", return_value=atlas):
        with patch("psa.forgetting.decay._now_utc", return_value=NOW):
            report = decay_report("default", params=params, origins={"interactive"})
    assert report.n_patterns_removed == 0
    assert any(s.anchor_id == 1 for s in report.shielded_anchors)


def test_p3_pinned_pattern_not_pruned(tmp_path, monkeypatch):
    from psa.forgetting.decay import DecayParams, decay_report
    from psa.forgetting.metadata import PatternMetadata, metadata_key, save_metadata

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir / "query_trace.jsonl", [])

    atlas = _fake_atlas({1: ["pinned stale"]})
    atlas.anchor_dir = str(tmp_path / "atlas_dir")
    (Path(atlas.anchor_dir)).mkdir(parents=True, exist_ok=True)
    long_ago = (NOW - timedelta(days=200)).isoformat()
    save_metadata(atlas.anchor_dir, {
        metadata_key(1, "pinned stale"): PatternMetadata(
            anchor_id=1, pattern="pinned stale", source="manual",
            created_at=long_ago, pinned=True,
        )
    })

    params = DecayParams(grace_days=30, decay_window_days=60,
                         low_activation_percentile=0, min_anchor_activations=0)
    with patch("psa.forgetting.decay._load_atlas_for_tenant", return_value=atlas):
        with patch("psa.forgetting.decay._now_utc", return_value=NOW):
            report = decay_report("default", params=params, origins={"interactive"})
    assert report.n_patterns_removed == 0
    assert report.n_patterns_pinned_exempt == 1


def test_source_grouped_counts(tmp_path, monkeypatch):
    """n_patterns_by_source_removed groups correctly."""
    from psa.forgetting.decay import DecayParams, decay_report
    from psa.forgetting.metadata import PatternMetadata, metadata_key, save_metadata

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir / "query_trace.jsonl", [])

    atlas = _fake_atlas({1: ["a1", "a2", "b1"]})
    atlas.anchor_dir = str(tmp_path / "atlas_dir")
    (Path(atlas.anchor_dir)).mkdir(parents=True, exist_ok=True)
    long_ago = (NOW - timedelta(days=200)).isoformat()
    save_metadata(atlas.anchor_dir, {
        metadata_key(1, "a1"): PatternMetadata(
            anchor_id=1, pattern="a1", source="atlas_build", created_at=long_ago),
        metadata_key(1, "a2"): PatternMetadata(
            anchor_id=1, pattern="a2", source="atlas_build", created_at=long_ago),
        metadata_key(1, "b1"): PatternMetadata(
            anchor_id=1, pattern="b1", source="refinement", created_at=long_ago),
    })

    params = DecayParams(grace_days=30, decay_window_days=60,
                         low_activation_percentile=0, min_anchor_activations=0)
    with patch("psa.forgetting.decay._load_atlas_for_tenant", return_value=atlas):
        with patch("psa.forgetting.decay._now_utc", return_value=NOW):
            report = decay_report("default", params=params, origins={"interactive"})
    assert report.n_patterns_removed == 3
    assert report.n_patterns_by_source_removed["atlas_build"] == 2
    assert report.n_patterns_by_source_removed["refinement"] == 1


def test_backfill_happens_on_decay_run(tmp_path, monkeypatch):
    """Patterns without metadata get stamped source='unknown' + now, and persisted."""
    from psa.forgetting.decay import DecayParams, decay_report
    from psa.forgetting.metadata import load_metadata

    monkeypatch.setenv("HOME", str(tmp_path))
    tenant_dir = tmp_path / ".psa" / "tenants" / "default"
    _write_trace(tenant_dir / "query_trace.jsonl", [])

    atlas = _fake_atlas({1: ["orphan pattern"]})
    atlas.anchor_dir = str(tmp_path / "atlas_dir")
    (Path(atlas.anchor_dir)).mkdir(parents=True, exist_ok=True)
    # No metadata file on disk.

    params = DecayParams(grace_days=30, decay_window_days=60,
                         low_activation_percentile=0, min_anchor_activations=0)
    with patch("psa.forgetting.decay._load_atlas_for_tenant", return_value=atlas):
        with patch("psa.forgetting.decay._now_utc", return_value=NOW):
            report = decay_report("default", params=params, origins={"interactive"})
    assert report.n_patterns_backfilled_this_run == 1
    # Because created_at == now, age is 0 → within grace → no removal.
    assert report.n_patterns_removed == 0
    # Metadata file persisted.
    loaded = load_metadata(atlas.anchor_dir)
    assert "1::orphan pattern" in loaded
    assert loaded["1::orphan pattern"].source == "unknown"
```

- [ ] **Step 2: Run tests — expect ModuleNotFoundError**

```
uv run pytest tests/test_forgetting_decay.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `psa/forgetting/decay.py`**

```python
"""
decay.py — orchestrate an advertisement forgetting pass.

D1: grace period + reinforcement window.
P1: shield anchors below absolute activation floor OR below percentile.
P3: pinned patterns are exempt.

Entry point: decay_report(tenant_id, params, origins) → DecayReport.
No disk side effects in this module EXCEPT metadata backfill persistence
(non-destructive provenance establishment).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .metadata import (
    PatternMetadata,
    backfill_unknown,
    load_metadata,
    metadata_key,
    save_metadata,
)
from .reinforcement import compute_reinforcement

logger = logging.getLogger("psa.forgetting.decay")

REASON_STALE = "stale_unreinforced"


@dataclass
class DecayParams:
    grace_days: int = 30
    decay_window_days: int = 60
    low_activation_percentile: float = 25.0
    min_anchor_activations: int = 10


@dataclass
class RemovedPattern:
    anchor_id: int
    pattern: str
    source: str
    created_at: str
    last_reinforced_at: str | None
    reason: str


@dataclass
class ShieldedAnchor:
    anchor_id: int
    activation_count: int
    patterns_held: int


@dataclass
class DecayReport:
    tenant_id: str
    atlas_version: int
    created_at: str
    params: DecayParams
    origins: set[str]
    n_patterns_scanned: int
    n_patterns_removed: int
    n_patterns_by_source_removed: dict[str, int]
    n_anchors_touched: int
    n_anchors_shielded: int
    n_patterns_shielded: int
    n_patterns_pinned_exempt: int
    n_patterns_backfilled_this_run: int
    pruning_by_reason: dict[str, int]
    removed_patterns: list[RemovedPattern] = field(default_factory=list)
    shielded_anchors: list[ShieldedAnchor] = field(default_factory=list)
    new_cards: list[dict] = field(default_factory=list)  # refined card list post-decay


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _load_atlas_for_tenant(tenant_id: str) -> Any:
    """Load atlas via TenantManager + AtlasManager (patchable helper for tests)."""
    from ..atlas import AtlasManager
    from ..tenant import TenantManager

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    return AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id).get_atlas()


def _trace_path(tenant_id: str) -> str:
    home = os.path.expanduser("~")
    return os.path.join(home, ".psa", "tenants", tenant_id, "query_trace.jsonl")


def _parse_iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _compute_activation_counts(
    atlas: Any, trace_path: str, origins: set[str], window_start: datetime,
) -> dict[int, int]:
    """Count activations per anchor within window, respecting origin filter."""
    import json as _json

    counts: dict[int, int] = {c.anchor_id: 0 for c in atlas.cards}
    if not os.path.exists(trace_path):
        return counts
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            ts = _parse_iso(rec.get("timestamp", ""))
            if ts is None or ts < window_start:
                continue
            if rec.get("query_origin", "interactive") not in origins:
                continue
            for aid in rec.get("selected_anchor_ids") or []:
                if aid in counts:
                    counts[aid] += 1
    return counts


def _percentile(value: float, sorted_values: list[float]) -> float:
    """Percentile rank of value within sorted_values (0..100)."""
    if not sorted_values:
        return 0.0
    n_le = sum(1 for v in sorted_values if v <= value)
    return 100.0 * n_le / len(sorted_values)


def _shielded_anchors(
    activation_counts: dict[int, int], params: DecayParams,
) -> set[int]:
    """Apply P1 two-part shield.

    Anchor is shielded when it's below the absolute floor OR below the
    percentile cutoff, computed over all anchors' activation counts.
    """
    values = sorted(activation_counts.values())
    shielded: set[int] = set()
    for aid, count in activation_counts.items():
        if count < params.min_anchor_activations:
            shielded.add(aid)
            continue
        pct = _percentile(count, values)
        if pct < params.low_activation_percentile:
            shielded.add(aid)
    return shielded


def decay_report(
    tenant_id: str,
    *,
    params: DecayParams,
    origins: set[str],
) -> DecayReport:
    """Run a decay pass. Returns a DecayReport; persists only metadata backfill."""
    atlas = _load_atlas_for_tenant(tenant_id)
    if atlas is None:
        raise FileNotFoundError(
            f"No atlas for tenant {tenant_id!r}. Run 'psa atlas build' first."
        )

    now = _now_utc()
    now_iso = now.isoformat()
    window_start = now - timedelta(days=params.decay_window_days + params.grace_days)

    # Load + backfill metadata (persisted — provenance is non-destructive).
    metadata = load_metadata(atlas.anchor_dir)
    patterns_by_anchor: dict[int, list[str]] = {
        c.anchor_id: list(c.generated_query_patterns or []) for c in atlas.cards
    }
    n_backfilled = backfill_unknown(metadata, patterns_by_anchor, now_iso)
    if n_backfilled > 0:
        save_metadata(atlas.anchor_dir, metadata)

    # Derive reinforcement (ephemeral).
    trace_path = _trace_path(tenant_id)
    reinforcement = compute_reinforcement(
        atlas, trace_path, origins=origins, window_start=window_start,
    )

    # Activation counts + shielded anchors (P1).
    activation_counts = _compute_activation_counts(
        atlas, trace_path, origins, window_start,
    )
    shielded = _shielded_anchors(activation_counts, params)

    # Walk anchors, apply D1 + P1 + P3.
    n_patterns_scanned = 0
    n_patterns_pinned_exempt = 0
    removed: list[RemovedPattern] = []
    by_source: dict[str, int] = {}
    anchors_touched: set[int] = set()
    new_cards: list[dict] = []

    for card in atlas.cards:
        patterns = list(card.generated_query_patterns or [])
        n_patterns_scanned += len(patterns)
        new_patterns: list[str] = []

        for pattern in patterns:
            key = metadata_key(card.anchor_id, pattern)
            meta = metadata.get(key)
            if meta is None:
                # Shouldn't happen post-backfill, but be defensive.
                new_patterns.append(pattern)
                continue

            # P3: pinned exempts.
            if meta.pinned:
                new_patterns.append(pattern)
                n_patterns_pinned_exempt += 1
                continue

            # P1: shielded anchor retains all patterns.
            if card.anchor_id in shielded:
                new_patterns.append(pattern)
                continue

            # D1: grace + window.
            created = _parse_iso(meta.created_at)
            if created is None:
                new_patterns.append(pattern)
                continue
            age_days = (now - created).days
            if age_days < params.grace_days:
                new_patterns.append(pattern)
                continue

            last_ts = reinforcement.get(key)
            if last_ts is not None:
                days_since = (now - last_ts).days
                if days_since <= params.decay_window_days:
                    new_patterns.append(pattern)
                    continue

            # It's a decay candidate.
            removed.append(RemovedPattern(
                anchor_id=card.anchor_id,
                pattern=pattern,
                source=meta.source,
                created_at=meta.created_at,
                last_reinforced_at=last_ts.isoformat() if last_ts else None,
                reason=REASON_STALE,
            ))
            by_source[meta.source] = by_source.get(meta.source, 0) + 1
            anchors_touched.add(card.anchor_id)

        # Build refined card dict (existing card with pruned patterns).
        card_dict = _card_to_dict(card, new_patterns)
        new_cards.append(card_dict)

    shielded_list = [
        ShieldedAnchor(
            anchor_id=aid,
            activation_count=activation_counts.get(aid, 0),
            patterns_held=len(patterns_by_anchor.get(aid, [])),
        )
        for aid in shielded
    ]

    return DecayReport(
        tenant_id=tenant_id,
        atlas_version=getattr(atlas, "version", 0),
        created_at=now_iso,
        params=params,
        origins=origins,
        n_patterns_scanned=n_patterns_scanned,
        n_patterns_removed=len(removed),
        n_patterns_by_source_removed=by_source,
        n_anchors_touched=len(anchors_touched),
        n_anchors_shielded=len(shielded),
        n_patterns_shielded=sum(len(patterns_by_anchor.get(a, [])) for a in shielded),
        n_patterns_pinned_exempt=n_patterns_pinned_exempt,
        n_patterns_backfilled_this_run=n_backfilled,
        pruning_by_reason={REASON_STALE: len(removed)} if removed else {},
        removed_patterns=removed,
        shielded_anchors=shielded_list,
        new_cards=new_cards,
    )


def _card_to_dict(card: Any, new_patterns: list[str]) -> dict:
    """Serialize a card with updated generated_query_patterns.

    Reuses the card's to_dict when present; otherwise a field-wise fallback.
    Mirrors psa.curation.curator._card_to_dict pattern.
    """
    if hasattr(card, "to_dict"):
        d = card.to_dict()
    else:
        d = {
            "anchor_id": card.anchor_id,
            "name": getattr(card, "name", f"anchor-{card.anchor_id}"),
            "meaning": getattr(card, "meaning", ""),
            "memory_types": list(getattr(card, "memory_types", [])),
            "include_terms": list(getattr(card, "include_terms", [])),
            "exclude_terms": list(getattr(card, "exclude_terms", [])),
            "prototype_examples": list(getattr(card, "prototype_examples", [])),
            "near_but_different": list(getattr(card, "near_but_different", [])),
            "centroid": list(getattr(card, "centroid", [])),
            "memory_count": getattr(card, "memory_count", 0),
            "is_novelty": getattr(card, "is_novelty", False),
            "status": getattr(card, "status", "active"),
            "metadata": dict(getattr(card, "metadata", {})),
            "generated_query_patterns": list(getattr(card, "generated_query_patterns", [])),
            "query_fingerprint": list(getattr(card, "query_fingerprint", [])),
        }
    d["generated_query_patterns"] = new_patterns
    return d
```

- [ ] **Step 4: Run tests — expect PASS**

```
uv run pytest tests/test_forgetting_decay.py -v
```

Expected: all 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add psa/forgetting/decay.py tests/test_forgetting_decay.py
git commit -m "$(cat <<'EOF'
feat: psa.forgetting.decay — D1 rule, P1 shield, P3 pin

decay_report(tenant_id, params, origins) orchestrates a decay pass:
loads metadata + backfills missing provenance, derives reinforcement
per-run, computes activation counts + P1 shielded anchors, applies
D1 grace+window rule with P3 pinning exemption. Returns a structured
DecayReport carrying per-pattern removals, source-grouped counts,
shielded anchor list, and refined cards.

Persists only metadata backfill; candidate/report writes live in
psa.forgetting.writer (Task 4).

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: candidate writer

Writes the three output files (candidate cards, summary meta, detail report) — only in real mode, only when there are removals.

**Files:**
- Create: `psa/forgetting/writer.py`
- Create: `tests/test_forgetting_writer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_forgetting_writer.py`:

```python
"""Tests for psa.forgetting.writer — candidate + meta + detail file writes."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path


def _make_report(tmp_path, n_removed=1):
    from psa.forgetting.decay import (
        DecayParams,
        DecayReport,
        RemovedPattern,
        ShieldedAnchor,
    )

    atlas_dir = tmp_path / "atlas_v1"
    atlas_dir.mkdir()
    removed = []
    if n_removed >= 1:
        removed = [RemovedPattern(
            anchor_id=1, pattern="stale pattern", source="manual",
            created_at="2020-01-01T00:00:00+00:00",
            last_reinforced_at=None, reason="stale_unreinforced",
        )]
    return atlas_dir, DecayReport(
        tenant_id="default",
        atlas_version=1,
        created_at="2026-04-17T12:00:00+00:00",
        params=DecayParams(),
        origins={"interactive"},
        n_patterns_scanned=5,
        n_patterns_removed=n_removed,
        n_patterns_by_source_removed={"manual": n_removed} if n_removed else {},
        n_anchors_touched=1 if n_removed else 0,
        n_anchors_shielded=0,
        n_patterns_shielded=0,
        n_patterns_pinned_exempt=0,
        n_patterns_backfilled_this_run=0,
        pruning_by_reason={"stale_unreinforced": n_removed} if n_removed else {},
        removed_patterns=removed,
        shielded_anchors=[],
        new_cards=[{"anchor_id": 1, "generated_query_patterns": []}],
    )


def test_write_decay_candidate_writes_three_files(tmp_path):
    from psa.forgetting.writer import write_decay_candidate

    atlas_dir, report = _make_report(tmp_path)
    wrote = write_decay_candidate(str(atlas_dir), report)

    assert wrote is True
    assert (atlas_dir / "anchor_cards_candidate.json").exists()
    assert (atlas_dir / "anchor_cards_candidate.meta.json").exists()
    assert (atlas_dir / "anchor_cards_candidate.decay_report.json").exists()


def test_candidate_cards_has_refined_patterns(tmp_path):
    from psa.forgetting.writer import write_decay_candidate

    atlas_dir, report = _make_report(tmp_path)
    write_decay_candidate(str(atlas_dir), report)

    cards = json.loads((atlas_dir / "anchor_cards_candidate.json").read_text())
    assert cards[0]["generated_query_patterns"] == []  # patterns pruned


def test_meta_is_summary_no_removed_patterns(tmp_path):
    """anchor_cards_candidate.meta.json must NOT include removed_patterns
    (that belongs in the decay_report file, not the meta summary)."""
    from psa.forgetting.writer import write_decay_candidate

    atlas_dir, report = _make_report(tmp_path)
    write_decay_candidate(str(atlas_dir), report)

    meta = json.loads((atlas_dir / "anchor_cards_candidate.meta.json").read_text())
    assert "removed_patterns" not in meta
    assert meta["source"] == "decay"
    assert meta["n_patterns_removed"] == 1
    assert meta["pruning_by_reason"] == {"stale_unreinforced": 1}


def test_decay_report_has_full_detail(tmp_path):
    from psa.forgetting.writer import write_decay_candidate

    atlas_dir, report = _make_report(tmp_path)
    write_decay_candidate(str(atlas_dir), report)

    detail = json.loads((atlas_dir / "anchor_cards_candidate.decay_report.json").read_text())
    assert detail["tenant_id"] == "default"
    assert len(detail["removed_patterns"]) == 1
    assert detail["removed_patterns"][0]["pattern"] == "stale pattern"


def test_empty_run_skips_all_writes(tmp_path):
    """When n_patterns_removed == 0, no candidate files are written."""
    from psa.forgetting.writer import write_decay_candidate

    atlas_dir, report = _make_report(tmp_path, n_removed=0)
    wrote = write_decay_candidate(str(atlas_dir), report)

    assert wrote is False
    assert not (atlas_dir / "anchor_cards_candidate.json").exists()
    assert not (atlas_dir / "anchor_cards_candidate.meta.json").exists()
    assert not (atlas_dir / "anchor_cards_candidate.decay_report.json").exists()


def test_origins_serialized_sorted(tmp_path):
    """origins set becomes a sorted list in the meta for stable JSON output."""
    from psa.forgetting.writer import write_decay_candidate

    atlas_dir, report = _make_report(tmp_path)
    report.origins = {"benchmark", "interactive"}
    write_decay_candidate(str(atlas_dir), report)

    meta = json.loads((atlas_dir / "anchor_cards_candidate.meta.json").read_text())
    assert meta["origins"] == ["benchmark", "interactive"]
```

- [ ] **Step 2: Run tests — expect ModuleNotFoundError**

```
uv run pytest tests/test_forgetting_writer.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `psa/forgetting/writer.py`**

```python
"""
writer.py — persist a DecayReport as three files.

Output files (spec §4):
    anchor_cards_candidate.json           — refined cards (Branch 1 candidate slot)
    anchor_cards_candidate.meta.json      — summary (bounded size)
    anchor_cards_candidate.decay_report.json — full removed-patterns detail

Empty-run guard: when n_patterns_removed == 0, nothing is written
(matches Branch 3 curate behavior — protects any in-flight candidate).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any

logger = logging.getLogger("psa.forgetting.writer")


def write_decay_candidate(atlas_dir: str, report: Any) -> bool:
    """Write candidate + meta + detail. Returns True if files were written.

    Empty-run: when no patterns are removed, skips all writes (preserves
    any unrelated in-flight candidate from refine/curate).
    """
    if report.n_patterns_removed == 0:
        logger.info("Decay produced no removals — skipping candidate write.")
        return False

    os.makedirs(atlas_dir, exist_ok=True)

    cards_path = os.path.join(atlas_dir, "anchor_cards_candidate.json")
    meta_path = os.path.join(atlas_dir, "anchor_cards_candidate.meta.json")
    detail_path = os.path.join(atlas_dir, "anchor_cards_candidate.decay_report.json")

    # 1. Candidate cards.
    with open(cards_path, "w", encoding="utf-8") as f:
        json.dump(report.new_cards, f, indent=2)

    # 2. Summary meta (bounded — no per-pattern array).
    meta = {
        "source": "decay",
        "created_at": report.created_at,
        "tenant_id": report.tenant_id,
        "atlas_version": report.atlas_version,
        "promoted": False,
        "promoted_at": None,
        "decay_parameters": asdict(report.params),
        "origins": sorted(report.origins),
        "n_patterns_scanned": report.n_patterns_scanned,
        "n_patterns_removed": report.n_patterns_removed,
        "n_patterns_by_source_removed": report.n_patterns_by_source_removed,
        "n_anchors_touched": report.n_anchors_touched,
        "n_anchors_shielded": report.n_anchors_shielded,
        "n_patterns_shielded": report.n_patterns_shielded,
        "n_patterns_pinned_exempt": report.n_patterns_pinned_exempt,
        "n_patterns_backfilled_this_run": report.n_patterns_backfilled_this_run,
        "pruning_by_reason": report.pruning_by_reason,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 3. Detail report (full per-pattern provenance).
    detail = {
        "tenant_id": report.tenant_id,
        "atlas_version": report.atlas_version,
        "created_at": report.created_at,
        "removed_patterns": [asdict(r) for r in report.removed_patterns],
        "shielded_anchors": [asdict(s) for s in report.shielded_anchors],
    }
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2)

    return True
```

- [ ] **Step 4: Run tests — expect PASS**

```
uv run pytest tests/test_forgetting_writer.py -v
```

Expected: all 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add psa/forgetting/writer.py tests/test_forgetting_writer.py
git commit -m "$(cat <<'EOF'
feat: psa.forgetting.writer — three-file decay candidate output

write_decay_candidate() persists candidate cards + summary .meta.json
+ detailed .decay_report.json. Empty-run guard skips all writes when
no patterns would be removed (preserves in-flight candidates from
refine/curate). Origins serialized as sorted list for stable output.

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Atlas rebuild metadata inheritance

Teaches `AtlasBuilder.build_atlas()` to copy `pattern_metadata.json` forward for matched `(anchor_id, normalized_pattern)` pairs. Without this, every rebuild resets the forgetting clock.

**Files:**
- Modify: `psa/atlas.py`
- Create: `tests/test_atlas_metadata_inheritance.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_atlas_metadata_inheritance.py`:

```python
"""Tests for atlas rebuild carrying pattern_metadata.json forward."""

from __future__ import annotations

import json
from pathlib import Path


def _seed_metadata(atlas_dir: Path, entries: list[dict]) -> None:
    atlas_dir.mkdir(parents=True, exist_ok=True)
    raw = {f"{e['anchor_id']}::{e['pattern']}": e for e in entries}
    (atlas_dir / "pattern_metadata.json").write_text(json.dumps(raw))


def test_inherit_matched_anchor_and_pattern_copies_metadata(tmp_path):
    """If new atlas has the same (anchor_id, normalized pattern) as old, the
    entry carries forward verbatim."""
    from psa.atlas import inherit_pattern_metadata

    old_dir = tmp_path / "atlas_v1"
    new_dir = tmp_path / "atlas_v2"
    new_dir.mkdir()

    _seed_metadata(old_dir, [
        {
            "anchor_id": 1,
            "pattern": "old pattern",
            "source": "manual",
            "created_at": "2020-01-01T00:00:00+00:00",
        }
    ])

    # New atlas has the same pattern on the same anchor.
    new_cards = [
        {"anchor_id": 1, "generated_query_patterns": ["Old Pattern"]},  # case drift intentional
    ]
    inherit_pattern_metadata(str(old_dir), str(new_dir), new_cards)

    new_meta = json.loads((new_dir / "pattern_metadata.json").read_text())
    assert "1::old pattern" in new_meta
    assert new_meta["1::old pattern"]["created_at"] == "2020-01-01T00:00:00+00:00"


def test_inherit_drops_orphans(tmp_path):
    """Old metadata entries without a matching new pattern are NOT copied."""
    from psa.atlas import inherit_pattern_metadata

    old_dir = tmp_path / "atlas_v1"
    new_dir = tmp_path / "atlas_v2"
    new_dir.mkdir()

    _seed_metadata(old_dir, [
        {"anchor_id": 1, "pattern": "retired pattern", "source": "manual",
         "created_at": "2020-01-01T00:00:00+00:00"},
    ])
    # New atlas has a different pattern for anchor 1.
    new_cards = [
        {"anchor_id": 1, "generated_query_patterns": ["fresh pattern"]},
    ]
    inherit_pattern_metadata(str(old_dir), str(new_dir), new_cards)

    new_meta = json.loads((new_dir / "pattern_metadata.json").read_text())
    assert "1::retired pattern" not in new_meta
    assert "1::fresh pattern" not in new_meta  # new patterns wait for backfill


def test_inherit_skips_missing_old_metadata(tmp_path):
    """If old atlas had no pattern_metadata.json, new file is empty (or absent)."""
    from psa.atlas import inherit_pattern_metadata

    old_dir = tmp_path / "atlas_v1"
    old_dir.mkdir()
    new_dir = tmp_path / "atlas_v2"
    new_dir.mkdir()

    new_cards = [
        {"anchor_id": 1, "generated_query_patterns": ["any"]},
    ]
    # Should not raise.
    inherit_pattern_metadata(str(old_dir), str(new_dir), new_cards)

    path = new_dir / "pattern_metadata.json"
    if path.exists():
        assert json.loads(path.read_text()) == {}


def test_inherit_preserves_pinned_flag(tmp_path):
    from psa.atlas import inherit_pattern_metadata

    old_dir = tmp_path / "atlas_v1"
    new_dir = tmp_path / "atlas_v2"
    new_dir.mkdir()

    _seed_metadata(old_dir, [
        {
            "anchor_id": 1,
            "pattern": "pinned one",
            "source": "manual",
            "created_at": "2020-01-01T00:00:00+00:00",
            "pinned": True,
        }
    ])
    new_cards = [
        {"anchor_id": 1, "generated_query_patterns": ["pinned one"]},
    ]
    inherit_pattern_metadata(str(old_dir), str(new_dir), new_cards)

    new_meta = json.loads((new_dir / "pattern_metadata.json").read_text())
    assert new_meta["1::pinned one"]["pinned"] is True
```

- [ ] **Step 2: Run tests — expect AttributeError / ImportError**

```
uv run pytest tests/test_atlas_metadata_inheritance.py -v
```

Expected: FAIL (`inherit_pattern_metadata` doesn't exist yet).

- [ ] **Step 3: Add `inherit_pattern_metadata` to `psa/atlas.py`**

In `psa/atlas.py`, add this function near the top-level helpers (before `AtlasBuilder`). Find a natural spot near the other `_match_anchors` / helper functions:

```python
def inherit_pattern_metadata(
    old_atlas_dir: str, new_atlas_dir: str, new_cards: list[dict]
) -> None:
    """Copy pattern_metadata.json entries from old atlas to new for matched patterns.

    Walks `new_cards` (the just-built atlas) and looks up each
    (anchor_id, normalize_pattern(pattern)) in the old metadata file. When
    found, carries the entry forward verbatim. Orphan entries (old patterns
    absent from new atlas) are dropped. New patterns (not in old metadata)
    are left unstamped — they'll be backfilled on the next decay run.

    See spec §1.1.
    """
    from .forgetting.metadata import FILENAME, normalize_pattern

    old_path = os.path.join(old_atlas_dir, FILENAME)
    if not os.path.exists(old_path):
        return
    try:
        with open(old_path, encoding="utf-8") as f:
            old_meta = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read %s during inheritance: %s", old_path, e)
        return

    new_meta: dict[str, dict] = {}
    for card in new_cards:
        anchor_id = card.get("anchor_id")
        if anchor_id is None:
            continue
        for pattern in card.get("generated_query_patterns") or []:
            key = f"{anchor_id}::{normalize_pattern(pattern)}"
            if key in old_meta:
                new_meta[key] = old_meta[key]

    os.makedirs(new_atlas_dir, exist_ok=True)
    new_path = os.path.join(new_atlas_dir, FILENAME)
    tmp_path = new_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(new_meta, f, indent=2)
    os.replace(tmp_path, new_path)
```

- [ ] **Step 4: Wire the inheritance into `AtlasBuilder.build_atlas()`**

In `psa/atlas.py`, find the block that inherits fingerprints (around line 744-755). After `new_fingerprints.save()`, add the metadata inheritance call. The context around the insertion should be:

```python
        atlas.fingerprint_store = new_fingerprints
        for card in atlas.cards:
            card.query_fingerprint = atlas.fingerprint_store.get(card.anchor_id)
        new_fingerprints.save()

        # Carry pattern_metadata.json forward for matched (anchor_id, pattern)
        # pairs. See spec §1.1 — without this, the advertisement-forgetting
        # clock resets on every atlas rebuild.
        if previous_atlas is not None:
            inherit_pattern_metadata(
                old_atlas_dir=previous_atlas.anchor_dir,
                new_atlas_dir=output_dir,
                new_cards=[card.to_dict() for card in atlas.cards],
            )

        logger.info(
            "Atlas v%d built: ...",
```

- [ ] **Step 5: Run the inheritance tests — expect PASS**

```
uv run pytest tests/test_atlas_metadata_inheritance.py -v
```

Expected: all 4 PASS.

- [ ] **Step 6: Run the existing atlas tests to confirm no regression**

```
uv run pytest tests/test_atlas.py -v
```

Expected: all existing atlas tests still pass (no behavior change for atlases without pattern_metadata.json).

- [ ] **Step 7: Commit**

```bash
git add psa/atlas.py tests/test_atlas_metadata_inheritance.py
git commit -m "$(cat <<'EOF'
feat: atlas rebuild inherits pattern_metadata.json

AtlasBuilder.build_atlas() now carries pattern_metadata.json entries
forward for matched (anchor_id, normalize_pattern(pattern)) pairs.
Orphan entries drop; new patterns wait for backfill on the next decay
run. Mirrors the existing FingerprintStore.inherit_from pattern.

Without this, every rebuild would reset created_at on all patterns,
defeating the purpose of persisting pattern provenance for
advertisement forgetting.

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `psa atlas decay` CLI

Wires the orchestrator + writer behind a CLI verb with `--dry-run` and param overrides.

**Files:**
- Modify: `psa/cli.py`
- Create: `tests/test_cli_atlas_decay.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_atlas_decay.py`:

```python
"""End-to-end CLI tests for `psa atlas decay`."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _write_atlas(atlas_dir: Path, anchor_ids: list[int], patterns: dict[int, list[str]]) -> None:
    atlas_dir.mkdir(parents=True, exist_ok=True)
    cards = [
        {
            "anchor_id": aid, "name": f"anchor-{aid}", "meaning": "m",
            "memory_types": ["semantic"], "include_terms": [], "exclude_terms": [],
            "prototype_examples": [], "near_but_different": [],
            "centroid": [0.0] * 768, "memory_count": 1, "is_novelty": False,
            "status": "active", "metadata": {},
            "generated_query_patterns": patterns.get(aid, []),
            "query_fingerprint": [],
        }
        for aid in anchor_ids
    ]
    (atlas_dir / "anchor_cards.json").write_text(json.dumps(cards))
    np.save(atlas_dir / "centroids.npy", np.zeros((len(anchor_ids), 768), dtype=np.float32))
    (atlas_dir / "atlas_meta.json").write_text(
        json.dumps({
            "version": 1, "tenant_id": "test",
            "stats": {
                "n_memories": len(anchor_ids),
                "n_anchors_learned": len(anchor_ids),
                "n_anchors_novelty": 0,
                "mean_cluster_size": 1.0, "min_cluster_size": 1,
                "max_cluster_size": 1, "stability_score": 1.0,
                "built_at": "2026-04-17T00:00:00+00:00",
            },
        })
    )


def test_cli_decay_dry_run_no_candidate_files(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [1], patterns={1: ["stale pattern"]})

    # Seed metadata: pattern is old and unreinforced.
    long_ago = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    (atlas_dir / "pattern_metadata.json").write_text(json.dumps({
        f"1::stale pattern": {
            "anchor_id": 1, "pattern": "stale pattern",
            "source": "manual", "created_at": long_ago,
        }
    }))

    # Empty trace.
    (tenant_dir / "query_trace.jsonl").write_text("")

    with patch("sys.argv", ["psa", "atlas", "decay", "--dry-run",
                            "--min-anchor-activations", "0",
                            "--low-activation-percentile", "0"]):
        main()

    assert not (atlas_dir / "anchor_cards_candidate.json").exists()
    assert not (atlas_dir / "anchor_cards_candidate.meta.json").exists()
    assert not (atlas_dir / "anchor_cards_candidate.decay_report.json").exists()


def test_cli_decay_real_writes_three_files(tmp_path, monkeypatch):
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [1], patterns={1: ["stale pattern"]})

    long_ago = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    (atlas_dir / "pattern_metadata.json").write_text(json.dumps({
        f"1::stale pattern": {
            "anchor_id": 1, "pattern": "stale pattern",
            "source": "manual", "created_at": long_ago,
        }
    }))
    (tenant_dir / "query_trace.jsonl").write_text("")

    with patch("sys.argv", ["psa", "atlas", "decay",
                            "--min-anchor-activations", "0",
                            "--low-activation-percentile", "0"]):
        main()

    assert (atlas_dir / "anchor_cards_candidate.json").exists()
    assert (atlas_dir / "anchor_cards_candidate.meta.json").exists()
    assert (atlas_dir / "anchor_cards_candidate.decay_report.json").exists()

    meta = json.loads((atlas_dir / "anchor_cards_candidate.meta.json").read_text())
    assert meta["source"] == "decay"
    assert meta["n_patterns_removed"] == 1


def test_cli_decay_json_mode_emits_envelope(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [1], patterns={1: ["stale"]})
    long_ago = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    (atlas_dir / "pattern_metadata.json").write_text(json.dumps({
        f"1::stale": {"anchor_id": 1, "pattern": "stale",
                     "source": "manual", "created_at": long_ago}
    }))
    (tenant_dir / "query_trace.jsonl").write_text("")

    with patch("sys.argv", ["psa", "atlas", "decay", "--dry-run", "--json",
                            "--min-anchor-activations", "0",
                            "--low-activation-percentile", "0"]):
        main()

    out = capsys.readouterr().out
    envelope = json.loads(out)
    assert envelope["source"] == "decay"
    assert envelope["n_patterns_removed"] == 1
    # Dry-run JSON embeds the detail as well.
    assert "removed_patterns" in envelope


def test_cli_decay_backfills_on_dry_run(tmp_path, monkeypatch):
    """Dry-run writes no candidate files, but DOES persist metadata backfill."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [1], patterns={1: ["unstamped"]})
    # No pre-existing pattern_metadata.json.
    (tenant_dir / "query_trace.jsonl").write_text("")

    with patch("sys.argv", ["psa", "atlas", "decay", "--dry-run"]):
        main()

    assert (atlas_dir / "pattern_metadata.json").exists()
    meta = json.loads((atlas_dir / "pattern_metadata.json").read_text())
    assert "1::unstamped" in meta
    assert meta["1::unstamped"]["source"] == "unknown"
```

- [ ] **Step 2: Run tests — expect failures**

```
uv run pytest tests/test_cli_atlas_decay.py -v
```

Expected: FAIL on argparse (`invalid choice: 'decay'`).

- [ ] **Step 3: Register the `decay` subparser in `psa/cli.py`**

Find the block that registers `curate` under `atlas_sub` (added in Branch 3). Add the `decay` subparser alongside:

```python
    p_atlas_decay = atlas_sub.add_parser(
        "decay",
        help=(
            "Propose removal of stale generated_query_patterns based on trace "
            "reinforcement. Candidate-side only; run `psa atlas "
            "promote-refinement` to apply."
        ),
    )
    p_atlas_decay.add_argument("--dry-run", action="store_true",
                               help="Print the report; write no candidate files. "
                                    "Metadata backfill is still persisted (non-destructive).")
    p_atlas_decay.add_argument("--grace-days", type=int, default=None)
    p_atlas_decay.add_argument("--decay-window-days", type=int, default=None)
    p_atlas_decay.add_argument("--low-activation-percentile", type=float, default=None)
    p_atlas_decay.add_argument("--min-anchor-activations", type=int, default=None)
    p_atlas_decay.add_argument("--include-origin", action="append", default=None,
                               help="Repeatable. Default: interactive only.")
    p_atlas_decay.add_argument("--verbose", action="store_true")
    p_atlas_decay.add_argument("--json", action="store_true")
```

- [ ] **Step 4: Add dispatch in `cmd_atlas()`**

Extend the dispatch in `cmd_atlas()`:

```python
def cmd_atlas(args):
    """Handle 'psa atlas <subcommand>'."""
    action = getattr(args, "atlas_action", None)
    if not action:
        print("Usage: psa atlas {build,status,health,refine,promote-refinement,curate,decay}")
        return

    if action in ("build", "rebuild"):
        _cmd_atlas_build(args)
    elif action == "status":
        _cmd_atlas_status(args)
    elif action == "health":
        _cmd_atlas_health(args)
    elif action == "refine":
        _cmd_atlas_refine(args)
    elif action == "promote-refinement":
        _cmd_atlas_refine_promote(args)
    elif action == "curate":
        _cmd_atlas_curate(args)
    elif action == "decay":
        _cmd_atlas_decay(args)
```

- [ ] **Step 5: Implement `_cmd_atlas_decay`**

Add this function at `psa/cli.py` immediately after `_cmd_atlas_curate`:

```python
def _cmd_atlas_decay(args):
    """Run an advertisement-forgetting pass for the current atlas."""
    import json as _json
    from dataclasses import asdict
    from pathlib import Path

    from .forgetting.decay import DecayParams, decay_report
    from .forgetting.writer import write_decay_candidate

    tenant_id = getattr(args, "tenant", "default")

    # Load config defaults from ~/.psa/config.json.decay (if present), then
    # apply CLI overrides.
    config_defaults = _load_decay_config()

    def _pick(flag_val, key, default):
        if flag_val is not None:
            return flag_val
        return config_defaults.get(key, default)

    params = DecayParams(
        grace_days=_pick(args.grace_days, "grace_days", 30),
        decay_window_days=_pick(args.decay_window_days, "decay_window_days", 60),
        low_activation_percentile=_pick(args.low_activation_percentile,
                                        "low_activation_percentile", 25.0),
        min_anchor_activations=_pick(args.min_anchor_activations,
                                     "min_anchor_activations", 10),
    )

    origins = _resolve_origins(args)

    try:
        report = decay_report(tenant_id, params=params, origins=origins)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        sys.exit(1)

    # Resolve atlas dir for write.
    from .atlas import AtlasManager
    from .tenant import TenantManager
    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
    atlas = mgr.get_atlas()
    atlas_dir = Path(atlas.anchor_dir) if atlas is not None else None

    wrote = False
    if not args.dry_run and atlas_dir is not None:
        wrote = write_decay_candidate(str(atlas_dir), report)

    if args.json:
        envelope = {
            "source": "decay",
            "tenant_id": tenant_id,
            "atlas_version": report.atlas_version,
            "n_patterns_scanned": report.n_patterns_scanned,
            "n_patterns_removed": report.n_patterns_removed,
            "n_patterns_by_source_removed": report.n_patterns_by_source_removed,
            "n_anchors_touched": report.n_anchors_touched,
            "n_anchors_shielded": report.n_anchors_shielded,
            "n_patterns_pinned_exempt": report.n_patterns_pinned_exempt,
            "n_patterns_backfilled_this_run": report.n_patterns_backfilled_this_run,
            "pruning_by_reason": report.pruning_by_reason,
            "origins": sorted(report.origins),
            "dry_run": args.dry_run,
            "wrote_candidate": wrote,
            "removed_patterns": [asdict(r) for r in report.removed_patterns],
            "shielded_anchors": [asdict(s) for s in report.shielded_anchors],
        }
        print(_json.dumps(envelope, indent=2))
        return

    # Tabular output.
    print(f"tenant: {tenant_id}   atlas v{report.atlas_version}   "
          f"origins: {', '.join(sorted(origins))}")
    print(f"params: grace={params.grace_days}d  "
          f"decay_window={params.decay_window_days}d  "
          f"activation_floor: <p{params.low_activation_percentile:g} "
          f"or <{params.min_anchor_activations} activations")
    print()
    print(f"Summary:")
    print(f"  Total patterns scanned: {report.n_patterns_scanned:>6}")
    print(f"  Decay candidates:       {report.n_patterns_removed:>6}")
    for src, count in sorted(report.n_patterns_by_source_removed.items()):
        print(f"    - {src}: {count}")
    print(f"  Anchors shielded (P1):  {report.n_anchors_shielded:>6}")
    print(f"    - patterns held:      {report.n_patterns_shielded:>6}")
    print(f"  Pinned (P3) exempt:     {report.n_patterns_pinned_exempt:>6}")
    print(f"  Backfilled this run:    {report.n_patterns_backfilled_this_run:>6}")

    if args.verbose and report.removed_patterns:
        print("\nDecay candidates:")
        for r in report.removed_patterns[:100]:
            print(f"  anchor {r.anchor_id:>4}  [{r.source}]  {r.pattern!r}")

    if args.dry_run:
        print("\n(dry-run — no candidate files written)")
    elif wrote:
        print(f"\nCandidate written to {atlas_dir / 'anchor_cards_candidate.json'}")
        print(f"Detail report at     {atlas_dir / 'anchor_cards_candidate.decay_report.json'}")
        print("Run 'psa atlas promote-refinement' to make this candidate inference-visible.")
    else:
        print("\nNo patterns removed — no candidate written.")


def _load_decay_config() -> dict:
    """Load the `decay` block from ~/.psa/config.json. Missing file → {}."""
    import json as _json
    import os as _os
    path = _os.path.expanduser("~/.psa/config.json")
    if not _os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return _json.load(f).get("decay", {})
    except (OSError, _json.JSONDecodeError):
        return {}
```

- [ ] **Step 6: Run CLI tests — expect PASS**

```
uv run pytest tests/test_cli_atlas_decay.py -v
```

Expected: all 4 PASS.

- [ ] **Step 7: Commit**

```bash
git add psa/cli.py tests/test_cli_atlas_decay.py
git commit -m "$(cat <<'EOF'
feat: `psa atlas decay` CLI — advertisement-forgetting pass

New subcommand wires the forgetting package together:
- --dry-run prints a report, writes no candidate files (backfill still
  persists metadata provenance).
- Real mode writes anchor_cards_candidate.json + .meta.json +
  .decay_report.json through the Branch 1 promote gate.
- --include-origin respects Branch 4's origin convention.
- --json emits the full envelope (summary + detail) for notebook use.
- Params come from ~/.psa/config.json decay block with CLI overrides.

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: full suite + lint

- [ ] **Step 1: Run the full test suite**

```
uv run pytest tests/ -q --ignore=tests/test_convo_miner.py --ignore=tests/test_mcp_server.py --ignore=tests/test_pipeline.py
```

Expected: all PASS. The three excluded files have pre-existing flakiness (network / FAISS abort) unrelated to this branch.

- [ ] **Step 2: Lint + format**

```
uv run ruff check .
uv run ruff format --check .
```

If `ruff format --check` reports drift, run `uv run ruff format .`, verify, commit.

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
| §1 Metadata model (normalized key, atomic write, source enum, reserved fields) | Task 1 |
| §1.1 Atlas rebuild inheritance | Task 5 |
| §2 R1 reinforcement (activation + substring, ephemeral) | Task 2 |
| §3 D1 grace + window rule | Task 3 |
| §3 P1 two-part shield (percentile OR absolute floor) | Task 3 (`_shielded_anchors`) |
| §3 P3 pinned exemption | Task 3 |
| §3 Config defaults + per-source reserved | Task 6 (`_load_decay_config`), params + CLI overrides |
| §4 `psa atlas decay` CLI + flags | Task 6 |
| §4 Dry-run writes no candidate files | Task 6 (test coverage) |
| §4 Three output files: candidate / summary .meta / detail .decay_report | Task 4 |
| §4 Summary excludes unbounded removed_patterns array | Task 4 (test_meta_is_summary_no_removed_patterns) |
| §4 Empty-run guard skips all candidate writes | Task 4 (test_empty_run_skips_all_writes) |
| §4 JSON envelope shape (includes detail in dry-run JSON mode) | Task 6 |
| §5.1–§5.2 Lazy backfill with conservative created_at=now | Task 1 (`backfill_unknown`), Task 3 (call site + persistence) |
| §5.3 Ephemeral reinforcement; never persisted | Task 2 (returns map; no disk write) |
| §5.4 Origin filter default `{"interactive"}` | Task 6 (`_resolve_origins` from Branch 4) |

No gaps.

**Placeholder scan:** every step carries complete code. No TBDs.

**Type consistency:**
- `PatternMetadata` signature in Task 1 matches usage in Tasks 3, 5.
- `DecayReport` fields in Task 3 match the writer's reads in Task 4.
- `metadata_key`, `normalize_pattern` imported consistently across modules.
- `_resolve_origins` is the Branch 4 helper (already in `psa/cli.py`); Task 6 reuses it.
- `_shielded_anchors` returns `set[int]`; consumers in `decay_report` use it as membership set.
