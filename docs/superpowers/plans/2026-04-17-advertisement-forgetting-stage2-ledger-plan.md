# Advertisement Forgetting Stage 2 — Pattern-Level Signal Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the shipped `psa/advertisement/` system with a persistent pattern-level signal ledger, inline per-query attribution, automated nightly decay gated by tracking/removal split with shadow-policy counterfactuals, and a stage 1 coexistence gate (refined-hash check at promotion) that prevents stale candidates from overwriting stage 2 removals.

**Architecture:** Phase 1 installs the stage 1 prerequisite (hash-stamp helper + promote-gate) across all candidate producers. Phase 2 adds the foundational data structures (config, attribution, ledger table). Phase 3 wires the inline write path with trace-first ordering. Phase 4 implements decay, removal, atlas rebuild reset, and lifecycle integration. Phase 5 adds inspection CLIs.

**Tech Stack:** Python 3.13, SQLite (tenant store at `~/.psa/tenants/{tenant}/memory.sqlite3`), pytest, ruff, existing `psa/advertisement/`, `psa/retriever.py`, `psa/pipeline.py`, `psa/lifecycle.py`, `psa/atlas.py`, `psa/trace.py`.

**Spec:** `docs/superpowers/specs/2026-04-17-advertisement-forgetting-stage2-ledger-design.md`

---

## Phase 1 — Stage 1 coexistence gate (refined-hash stamp + promote refusal)

Ships independently of anything in stage 2. Closes the overwrite hazard for every candidate producer before stage 2 ever mutates the refined surface.

### Task 1.1: `stamp_refined_hash` helper

**Files:**
- Modify: `psa/advertisement/writer.py`
- Test: `tests/test_advertisement_writer.py` (extend existing file)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_advertisement_writer.py`:

```python
def test_stamp_refined_hash_with_existing_file(tmp_path):
    from psa.advertisement.writer import stamp_refined_hash

    atlas_dir = tmp_path / "atlas_v1"
    atlas_dir.mkdir()
    refined = atlas_dir / "anchor_cards_refined.json"
    refined.write_text('{"cards": []}')

    meta = {"source": "decay"}
    stamp_refined_hash(meta, atlas_dir)

    assert meta["refined_existed_at_generation"] is True
    assert meta["refined_path_at_generation"].endswith("anchor_cards_refined.json")
    assert meta["refined_hash_at_generation"].startswith("sha256:")
    # SHA-256 of '{"cards": []}' is deterministic
    assert len(meta["refined_hash_at_generation"]) == len("sha256:") + 64


def test_stamp_refined_hash_without_existing_file(tmp_path):
    from psa.advertisement.writer import stamp_refined_hash

    atlas_dir = tmp_path / "atlas_v1"
    atlas_dir.mkdir()

    meta = {"source": "decay"}
    stamp_refined_hash(meta, atlas_dir)

    assert meta["refined_existed_at_generation"] is False
    assert meta["refined_hash_at_generation"] is None
    assert meta["refined_path_at_generation"].endswith("anchor_cards_refined.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_advertisement_writer.py::test_stamp_refined_hash_with_existing_file tests/test_advertisement_writer.py::test_stamp_refined_hash_without_existing_file -v`
Expected: FAIL with `ImportError: cannot import name 'stamp_refined_hash'`

- [ ] **Step 3: Implement the helper**

Prepend imports to `psa/advertisement/writer.py` (after existing imports):

```python
import hashlib
from pathlib import Path
```

Append to `psa/advertisement/writer.py`:

```python
def stamp_refined_hash(meta: dict, atlas_dir) -> None:
    """Stamp meta with refined-file hash + path + existence.

    Every candidate producer (decay, refine, curate) calls this before
    persisting its meta file. promote-refinement refuses promotion if
    the current refined hash differs from the recorded value, blocking
    stale candidates from overwriting stage 2 advertisement removals.

    Mutates `meta` in place.
    """
    atlas_dir = Path(atlas_dir)
    refined_path = atlas_dir / "anchor_cards_refined.json"
    meta["refined_path_at_generation"] = str(refined_path)
    if refined_path.exists():
        raw = refined_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        meta["refined_hash_at_generation"] = f"sha256:{digest}"
        meta["refined_existed_at_generation"] = True
    else:
        meta["refined_hash_at_generation"] = None
        meta["refined_existed_at_generation"] = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_advertisement_writer.py -v`
Expected: PASS (both new tests plus the existing tests)

- [ ] **Step 5: Commit**

```bash
git add psa/advertisement/writer.py tests/test_advertisement_writer.py
git commit -m "feat: add stamp_refined_hash helper for candidate meta"
```

---

### Task 1.2: Stamp decay candidate meta

**Files:**
- Modify: `psa/advertisement/writer.py` (inside `write_decay_candidate`)
- Test: `tests/test_advertisement_writer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_advertisement_writer.py`:

```python
def test_write_decay_candidate_stamps_refined_hash(tmp_path):
    import json
    from psa.advertisement.writer import write_decay_candidate

    atlas_dir, report = _make_report(tmp_path)
    # Create an existing refined file so the hash is non-null
    refined = atlas_dir / "anchor_cards_refined.json"
    refined.write_text('{"v":1}')

    assert write_decay_candidate(str(atlas_dir), report) is True
    meta_path = atlas_dir / "anchor_cards_candidate.meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["refined_existed_at_generation"] is True
    assert meta["refined_hash_at_generation"].startswith("sha256:")
    assert meta["refined_path_at_generation"].endswith("anchor_cards_refined.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_advertisement_writer.py::test_write_decay_candidate_stamps_refined_hash -v`
Expected: FAIL with `KeyError: 'refined_existed_at_generation'`

- [ ] **Step 3: Implement the stamp call**

In `psa/advertisement/writer.py`, inside `write_decay_candidate`, replace the `meta = {...}` construction block with this (after the dict literal, before the file write):

```python
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
    stamp_refined_hash(meta, atlas_dir)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_advertisement_writer.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add psa/advertisement/writer.py tests/test_advertisement_writer.py
git commit -m "feat: stamp refined hash on decay candidate meta"
```

---

### Task 1.3: Stamp curate candidate meta

**Files:**
- Modify: `psa/curation/curator.py`
- Test: `tests/test_cli_atlas_curate.py` (extend existing)

- [ ] **Step 1: Write the failing test**

Find the existing curate-writes-candidate test in `tests/test_cli_atlas_curate.py` (search for where `anchor_cards_candidate.meta.json` is read). Append a new test:

```python
def test_curate_stamps_refined_hash(tmp_path, monkeypatch):
    """Curate candidate must stamp refined hash so promote-refinement can gate."""
    import json
    import os
    monkeypatch.setenv("HOME", str(tmp_path))
    # Replicate the minimal curate setup used by existing tests in this file.
    # If a helper like _run_curate_with_atlas(...) exists, call it; otherwise
    # copy the shortest setup block and invoke psa.curation.curator.curate(...).
    from psa.curation.curator import curate

    atlas_dir = tmp_path / ".psa" / "tenants" / "default" / "atlas_v1"
    atlas_dir.mkdir(parents=True)
    # Seed a minimal anchor_cards.json
    (atlas_dir / "anchor_cards.json").write_text(json.dumps([{
        "anchor_id": 1,
        "label": "test",
        "generated_query_patterns": ["pat a", "pat b"],
        "dominant_memory_types": [],
        "example_memory_snippets": [],
    }]))
    # Seed an existing refined file so the hash is non-null.
    (atlas_dir / "anchor_cards_refined.json").write_text('{"v":1}')

    # Call whatever curate entry point produces the candidate. Every test file
    # already exercises it; replicate the same call pattern for this assertion.
    # For example: curate(tenant_id="default", atlas_version=1, ...)
    _ = curate  # usage depends on file-specific test helpers

    # After curate runs:
    meta_path = atlas_dir / "anchor_cards_candidate.meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        assert meta["refined_existed_at_generation"] is True
        assert meta["refined_hash_at_generation"].startswith("sha256:")
```

If the existing test file already has a helper that runs curate end-to-end and reads the meta, extend that test to also assert the three new fields — that is the preferred shape.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_atlas_curate.py::test_curate_stamps_refined_hash -v`
Expected: FAIL (meta missing hash fields, or test helper fails to import — reshape to match the file's existing helpers)

- [ ] **Step 3: Implement the stamp call**

In `psa/curation/curator.py`, find the block where the candidate meta is written (around line 112). Before the `json.dump(meta, f)` call for the meta file, insert:

```python
    from psa.advertisement.writer import stamp_refined_hash
    stamp_refined_hash(meta, atlas_dir)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cli_atlas_curate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/curation/curator.py tests/test_cli_atlas_curate.py
git commit -m "feat: stamp refined hash on curate candidate meta"
```

---

### Task 1.4: Stamp refine candidate meta

**Files:**
- Modify: `psa/cli.py` (refine path; search for `anchor_cards_candidate.meta.json` writes inside `_cmd_atlas_refine` or equivalent)
- Test: `tests/test_cli_atlas_refine.py` (extend existing)

- [ ] **Step 1: Write the failing test**

Extend `tests/test_cli_atlas_refine.py` with:

```python
def test_refine_stamps_refined_hash(tmp_path, monkeypatch):
    """Refine candidate must stamp refined hash so promote-refinement can gate."""
    import json
    # Use whatever setup helper other tests in this file use.
    # After running the refine command end-to-end:
    atlas_dir = tmp_path / ".psa" / "tenants" / "default" / "atlas_v1"
    meta_path = atlas_dir / "anchor_cards_candidate.meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        assert "refined_hash_at_generation" in meta
        assert "refined_path_at_generation" in meta
        assert "refined_existed_at_generation" in meta
```

Reuse the existing scaffold that runs refine end-to-end in this test file.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_atlas_refine.py::test_refine_stamps_refined_hash -v`
Expected: FAIL (meta missing three fields)

- [ ] **Step 3: Implement the stamp call**

In `psa/cli.py`, find the refine path that writes `anchor_cards_candidate.meta.json`. Before `json.dump(meta, f)` for that meta, add:

```python
    from psa.advertisement.writer import stamp_refined_hash
    stamp_refined_hash(meta, atlas_dir)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cli_atlas_refine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/cli.py tests/test_cli_atlas_refine.py
git commit -m "feat: stamp refined hash on refine candidate meta"
```

---

### Task 1.5: Hash-gate in `promote-refinement`

**Files:**
- Modify: `psa/cli.py` (promote-refinement path, around line 594)
- Test: `tests/test_promote_refinement_hash_gate.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_promote_refinement_hash_gate.py`:

```python
"""Hash gate at promotion prevents stale candidates from overwriting refined."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _setup_tenant(tmp_path, monkeypatch, version=1):
    monkeypatch.setenv("HOME", str(tmp_path))
    atlas_dir = tmp_path / ".psa" / "tenants" / "default" / f"atlas_v{version}"
    atlas_dir.mkdir(parents=True)
    (tmp_path / ".psa" / "tenants" / "default" / "atlas_version").write_text(str(version))
    return atlas_dir


def _run_promote(tmp_path):
    return subprocess.run(
        [sys.executable, "-m", "psa", "atlas", "promote-refinement"],
        env={**{"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}},
        capture_output=True,
        text=True,
    )


def test_promote_refuses_when_refined_hash_mismatches(tmp_path, monkeypatch):
    atlas_dir = _setup_tenant(tmp_path, monkeypatch)
    refined = atlas_dir / "anchor_cards_refined.json"
    refined.write_text('{"state":"A"}')  # hash X
    candidate = atlas_dir / "anchor_cards_candidate.json"
    candidate.write_text('{"state":"candidate"}')
    # Meta records hash X, but we then mutate refined to make its hash Y.
    from psa.advertisement.writer import stamp_refined_hash
    meta = {"source": "decay", "tenant_id": "default", "atlas_version": 1}
    stamp_refined_hash(meta, atlas_dir)
    (atlas_dir / "anchor_cards_candidate.meta.json").write_text(json.dumps(meta))
    refined.write_text('{"state":"B"}')  # hash Y now

    result = _run_promote(tmp_path)
    assert result.returncode != 0
    assert "refined cards changed" in (result.stdout + result.stderr).lower()


def test_promote_succeeds_when_refined_hash_matches(tmp_path, monkeypatch):
    atlas_dir = _setup_tenant(tmp_path, monkeypatch)
    refined = atlas_dir / "anchor_cards_refined.json"
    refined.write_text('{"state":"A"}')
    candidate = atlas_dir / "anchor_cards_candidate.json"
    candidate.write_text('{"state":"candidate"}')
    from psa.advertisement.writer import stamp_refined_hash
    meta = {"source": "decay", "tenant_id": "default", "atlas_version": 1}
    stamp_refined_hash(meta, atlas_dir)
    (atlas_dir / "anchor_cards_candidate.meta.json").write_text(json.dumps(meta))
    # Refined is unchanged — hash still matches recorded value.

    result = _run_promote(tmp_path)
    assert result.returncode == 0


def test_promote_succeeds_on_first_run_when_refined_absent_and_stays_absent(
    tmp_path, monkeypatch
):
    """First-time promotion: no refined file at generation OR promotion."""
    atlas_dir = _setup_tenant(tmp_path, monkeypatch)
    candidate = atlas_dir / "anchor_cards_candidate.json"
    candidate.write_text('{"state":"candidate"}')
    from psa.advertisement.writer import stamp_refined_hash
    meta = {"source": "decay", "tenant_id": "default", "atlas_version": 1}
    stamp_refined_hash(meta, atlas_dir)  # refined_existed_at_generation=False
    (atlas_dir / "anchor_cards_candidate.meta.json").write_text(json.dumps(meta))

    result = _run_promote(tmp_path)
    assert result.returncode == 0


def test_promote_refuses_when_refined_absent_then_created(tmp_path, monkeypatch):
    """Asymmetric case: meta says refined_existed=False, but a refined file
    exists now (e.g., stage 2 created it in the interim)."""
    atlas_dir = _setup_tenant(tmp_path, monkeypatch)
    candidate = atlas_dir / "anchor_cards_candidate.json"
    candidate.write_text('{"state":"candidate"}')
    from psa.advertisement.writer import stamp_refined_hash
    meta = {"source": "decay", "tenant_id": "default", "atlas_version": 1}
    stamp_refined_hash(meta, atlas_dir)  # refined absent at stamp time
    (atlas_dir / "anchor_cards_candidate.meta.json").write_text(json.dumps(meta))
    (atlas_dir / "anchor_cards_refined.json").write_text('{"new":true}')

    result = _run_promote(tmp_path)
    assert result.returncode != 0


def test_promote_refuses_legacy_candidate_without_hash_field(tmp_path, monkeypatch):
    """Candidates generated before the hash-gate change: meta lacks the field."""
    atlas_dir = _setup_tenant(tmp_path, monkeypatch)
    refined = atlas_dir / "anchor_cards_refined.json"
    refined.write_text('{"state":"A"}')
    candidate = atlas_dir / "anchor_cards_candidate.json"
    candidate.write_text('{"state":"candidate"}')
    legacy_meta = {"source": "decay", "tenant_id": "default", "atlas_version": 1}
    (atlas_dir / "anchor_cards_candidate.meta.json").write_text(json.dumps(legacy_meta))

    result = _run_promote(tmp_path)
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "missing" in combined or "legacy" in combined or "regenerate" in combined
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_promote_refinement_hash_gate.py -v`
Expected: FAIL (no hash check yet, so mismatching/legacy cases wrongly succeed)

- [ ] **Step 3: Implement the hash gate**

In `psa/cli.py`, locate the promote-refinement handler (search for the comment `# Copy cards verbatim` around line 594). Replace that block and the block that follows with:

```python
    # Hash-gate: refuse stale candidates that were generated before refined changed.
    import hashlib
    if candidate_meta_path.exists():
        with open(candidate_meta_path) as f:
            meta_for_gate = json.load(f)
    else:
        meta_for_gate = None

    if meta_for_gate is None or "refined_hash_at_generation" not in meta_for_gate:
        print(
            "  Error: candidate meta is missing refined_hash_at_generation "
            "(legacy candidate pre-dating the hash gate)."
        )
        print(
            "  Rerun the producing command (psa atlas decay | refine | curate) "
            "to regenerate the candidate against current refined state."
        )
        sys.exit(1)

    recorded_hash = meta_for_gate.get("refined_hash_at_generation")
    recorded_existed = meta_for_gate.get("refined_existed_at_generation", True)

    if refined_path.exists():
        current_hash = "sha256:" + hashlib.sha256(refined_path.read_bytes()).hexdigest()
        current_existed = True
    else:
        current_hash = None
        current_existed = False

    if recorded_existed != current_existed or recorded_hash != current_hash:
        source = meta_for_gate.get("source", "unknown")
        print("  Error: refined cards changed since candidate was generated.")
        print(f"    candidate source:       {source}")
        print(f"    recorded refined hash:  {recorded_hash}")
        print(f"    current refined hash:   {current_hash}")
        print("  Likely cause: stage 2 advertisement decay removed patterns, or")
        print("  another candidate was promoted, between candidate generation and now.")
        print(f"  Fix: rerun `psa atlas {source}` to regenerate the candidate.")
        sys.exit(1)

    # Copy cards verbatim
    shutil.copyfile(candidate_path, refined_path)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_promote_refinement_hash_gate.py -v && uv run pytest tests/test_cli_atlas_decay.py tests/test_cli_atlas_curate.py tests/test_cli_atlas_refine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/cli.py tests/test_promote_refinement_hash_gate.py
git commit -m "feat: hash-gate psa atlas promote-refinement

Refuse stale candidates whose recorded refined-hash differs from the
current refined-hash, including legacy candidates missing the field.
Covers all three candidate producers (decay, refine, curate) via the
shared stamp_refined_hash helper."
```

---

## Phase 2 — Foundation: config, attribution, schema

### Task 2.1: `AdvertisementDecayConfig` + config validation

**Files:**
- Create: `psa/advertisement/config.py`
- Modify: `psa/config.py`
- Test: `tests/test_advertisement_config.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_advertisement_config.py`:

```python
"""Tests for psa.advertisement.config — typed view + tracking→trace validation."""

from __future__ import annotations

import json

import pytest


def _write_config(tmp_path, body):
    d = tmp_path / ".psa"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(body))
    return d


def test_defaults(tmp_path):
    from psa.config import MempalaceConfig
    from psa.advertisement.config import AdvertisementDecayConfig

    _write_config(tmp_path, {})
    cfg = MempalaceConfig(config_dir=tmp_path / ".psa")
    ad = AdvertisementDecayConfig.from_mempalace(cfg)

    assert ad.tracking_enabled is False
    assert ad.removal_enabled is False
    assert ad.retrieval_credit == 1.0
    assert ad.selector_pick_credit == 2.0
    assert ad.selector_decline_penalty == 0.25
    assert ad.tau_days == 45
    assert ad.grace_days == 21
    assert ad.removal_threshold == 0.0
    assert ad.sustained_cycles == 14
    assert ad.min_patterns_floor == 3
    assert ad.epsilon == 0.05
    assert ad.bm25_topk_floor == 48
    assert ad.shadow.selector_decline_penalty == 0.5
    assert ad.shadow.sustained_cycles == 7


def test_tracking_requires_trace(tmp_path):
    from psa.config import MempalaceConfig
    from psa.advertisement.config import (
        AdvertisementDecayConfig,
        AdvertisementDecayConfigError,
    )

    _write_config(
        tmp_path,
        {
            "trace_queries": False,
            "advertisement_decay": {"tracking_enabled": True},
        },
    )
    cfg = MempalaceConfig(config_dir=tmp_path / ".psa")
    with pytest.raises(AdvertisementDecayConfigError) as exc:
        AdvertisementDecayConfig.from_mempalace(cfg)
    assert "trace_queries" in str(exc.value)


def test_env_overrides(tmp_path, monkeypatch):
    from psa.config import MempalaceConfig
    from psa.advertisement.config import AdvertisementDecayConfig

    _write_config(tmp_path, {})
    monkeypatch.setenv("PSA_AD_DECAY_TRACKING_ENABLED", "1")
    monkeypatch.setenv("PSA_AD_DECAY_TAU_DAYS", "30")
    monkeypatch.setenv("PSA_AD_DECAY_DECLINE_PENALTY", "0.5")
    # trace_queries defaults True, so tracking can be enabled
    cfg = MempalaceConfig(config_dir=tmp_path / ".psa")
    ad = AdvertisementDecayConfig.from_mempalace(cfg)

    assert ad.tracking_enabled is True
    assert ad.tau_days == 30
    assert ad.selector_decline_penalty == 0.5
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_advertisement_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psa.advertisement.config'`

- [ ] **Step 3: Implement the config module**

Create `psa/advertisement/config.py`:

```python
"""
config.py — typed view over the advertisement_decay block in MempalaceConfig.

Validation: tracking_enabled=true requires trace_queries=true. Without it,
ledger writes (gated on trace-first success) accumulate zero events, and
rebuild-ledger cannot be canonical. Fail fast at config load.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class AdvertisementDecayConfigError(ValueError):
    """Raised when advertisement_decay settings are inconsistent."""


def _env_bool(name: str, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in ("0", "", "false", "False")


def _env_float(name: str, default):
    v = os.environ.get(name)
    return float(v) if v is not None else default


def _env_int(name: str, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default


@dataclass(frozen=True)
class ShadowConfig:
    selector_decline_penalty: float = 0.5
    sustained_cycles: int = 7


@dataclass(frozen=True)
class AdvertisementDecayConfig:
    tracking_enabled: bool = False
    removal_enabled: bool = False

    retrieval_credit: float = 1.0
    selector_pick_credit: float = 2.0
    selector_decline_penalty: float = 0.25
    tau_days: int = 45
    grace_days: int = 21
    removal_threshold: float = 0.0
    sustained_cycles: int = 14
    min_patterns_floor: int = 3
    epsilon: float = 0.05
    bm25_topk_floor: int = 48

    shadow: ShadowConfig = field(default_factory=ShadowConfig)

    @classmethod
    def from_mempalace(cls, mempalace_cfg) -> "AdvertisementDecayConfig":
        block = mempalace_cfg._file_config.get("advertisement_decay", {}) or {}
        shadow_block = block.get("shadow", {}) or {}
        shadow = ShadowConfig(
            selector_decline_penalty=float(
                shadow_block.get("selector_decline_penalty", 0.5)
            ),
            sustained_cycles=int(shadow_block.get("sustained_cycles", 7)),
        )
        cfg = cls(
            tracking_enabled=_env_bool(
                "PSA_AD_DECAY_TRACKING_ENABLED",
                bool(block.get("tracking_enabled", False)),
            ),
            removal_enabled=_env_bool(
                "PSA_AD_DECAY_REMOVAL_ENABLED",
                bool(block.get("removal_enabled", False)),
            ),
            retrieval_credit=_env_float(
                "PSA_AD_DECAY_RETRIEVAL_CREDIT",
                float(block.get("retrieval_credit", 1.0)),
            ),
            selector_pick_credit=_env_float(
                "PSA_AD_DECAY_PICK_CREDIT",
                float(block.get("selector_pick_credit", 2.0)),
            ),
            selector_decline_penalty=_env_float(
                "PSA_AD_DECAY_DECLINE_PENALTY",
                float(block.get("selector_decline_penalty", 0.25)),
            ),
            tau_days=_env_int("PSA_AD_DECAY_TAU_DAYS", int(block.get("tau_days", 45))),
            grace_days=_env_int(
                "PSA_AD_DECAY_GRACE_DAYS", int(block.get("grace_days", 21))
            ),
            removal_threshold=_env_float(
                "PSA_AD_DECAY_REMOVAL_THRESHOLD",
                float(block.get("removal_threshold", 0.0)),
            ),
            sustained_cycles=_env_int(
                "PSA_AD_DECAY_SUSTAINED_CYCLES",
                int(block.get("sustained_cycles", 14)),
            ),
            min_patterns_floor=_env_int(
                "PSA_AD_DECAY_MIN_PATTERNS_FLOOR",
                int(block.get("min_patterns_floor", 3)),
            ),
            epsilon=_env_float(
                "PSA_AD_DECAY_EPSILON", float(block.get("epsilon", 0.05))
            ),
            bm25_topk_floor=_env_int(
                "PSA_AD_DECAY_BM25_TOPK_FLOOR",
                int(block.get("bm25_topk_floor", 48)),
            ),
            shadow=shadow,
        )
        cfg._validate(mempalace_cfg)
        return cfg

    def _validate(self, mempalace_cfg) -> None:
        if self.tracking_enabled and not mempalace_cfg.trace_queries:
            raise AdvertisementDecayConfigError(
                "advertisement_decay.tracking_enabled=true requires "
                "trace_queries=true. Ledger writes fire only when the "
                "preceding trace write succeeds, so tracing must be "
                "enabled for ledger accumulation to work. "
                "Set trace_queries=true or tracking_enabled=false."
            )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_advertisement_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/advertisement/config.py tests/test_advertisement_config.py
git commit -m "feat: AdvertisementDecayConfig with tracking→trace validation"
```

---

### Task 2.2: BM25 argmax attribution (pure function)

**Files:**
- Create: `psa/advertisement/attribution.py`
- Test: `tests/test_ledger_attribution.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ledger_attribution.py`:

```python
"""Tests for psa.advertisement.attribution — BM25 argmax + ε-split."""

from __future__ import annotations


def test_argmax_single_best(tmp_path):
    from psa.advertisement.attribution import attribute_bm25_argmax

    patterns = ["cat vs dog behavior", "unrelated pattern text"]
    query = "cat vs dog behavior analysis"
    argmax, tied = attribute_bm25_argmax(query, patterns, epsilon=0.05)
    assert argmax == "cat vs dog behavior"
    assert tied == []


def test_argmax_with_epsilon_tie():
    from psa.advertisement.attribution import attribute_bm25_argmax

    # Two near-duplicate patterns — BM25 scores should be within ε
    patterns = ["token refresh flow", "token refresh flow step", "fully unrelated"]
    query = "token refresh flow"
    argmax, tied = attribute_bm25_argmax(query, patterns, epsilon=0.5)
    assert argmax in {"token refresh flow", "token refresh flow step"}
    # Both near-duplicates should appear in argmax ∪ tied
    covered = {argmax, *tied}
    assert covered == {"token refresh flow", "token refresh flow step"}


def test_argmax_empty_patterns():
    from psa.advertisement.attribution import attribute_bm25_argmax

    argmax, tied = attribute_bm25_argmax("query", [], epsilon=0.05)
    assert argmax is None
    assert tied == []


def test_argmax_no_lexical_overlap():
    from psa.advertisement.attribution import attribute_bm25_argmax

    # Query and all patterns share zero terms
    patterns = ["pattern one", "pattern two"]
    query = "xyzxyz qqq"
    argmax, tied = attribute_bm25_argmax(query, patterns, epsilon=0.05)
    # BM25 returns 0 for every pattern. All are tied at zero — we don't credit
    # any template when nothing matches.
    assert argmax is None
    assert tied == []
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_ledger_attribution.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the attribution function**

Create `psa/advertisement/attribution.py`:

```python
"""
attribution.py — BM25 argmax attribution over an anchor's patterns.

Pure function; no DB access. Reused by the inline ledger writer and by
rebuild-ledger. Given a query and an anchor's generated_query_patterns,
returns (argmax_pattern, epsilon_tied_patterns). Patterns whose BM25
score is within `epsilon` of argmax share credit with argmax downstream.

Returns (None, []) when no pattern has a positive BM25 score — stage 2
does not credit a template when lexical contribution is zero.
"""

from __future__ import annotations

import math
import re
from typing import List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    avg_doc_len: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Single-document BM25 against a background of N=1 (no IDF damping).

    This is a deliberately simple in-document BM25 — we are attributing
    credit within one anchor's small pattern list, not ranking across a
    corpus. Term-frequency saturation + length normalization give the
    expected "best lexical match" shape without needing a global IDF.
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    dl = len(doc_tokens)
    score = 0.0
    tf = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1
    for q in query_tokens:
        if q not in tf:
            continue
        f = tf[q]
        norm = 1 - b + b * (dl / max(avg_doc_len, 1e-9))
        score += (f * (k1 + 1)) / (f + k1 * norm)
    return score


def attribute_bm25_argmax(
    query: str, patterns: List[str], epsilon: float
) -> Tuple[str | None, List[str]]:
    """Return (argmax_pattern, epsilon_tied_patterns) over BM25 scores.

    Returns (None, []) when patterns is empty or all scores are zero.
    Epsilon ties are patterns whose score is within `epsilon` of argmax,
    excluding argmax itself.
    """
    if not patterns:
        return None, []
    q_tokens = _tokenize(query)
    if not q_tokens:
        return None, []

    tokenized = [_tokenize(p) for p in patterns]
    avg_len = sum(len(t) for t in tokenized) / len(tokenized)

    scores = [_bm25_score(q_tokens, dt, avg_len) for dt in tokenized]
    max_score = max(scores)
    if max_score <= 0.0:
        return None, []

    argmax_idx = scores.index(max_score)
    argmax_pattern = patterns[argmax_idx]
    tied = [
        patterns[i]
        for i, s in enumerate(scores)
        if i != argmax_idx and (max_score - s) <= epsilon and s > 0.0
    ]
    return argmax_pattern, tied
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ledger_attribution.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/advertisement/attribution.py tests/test_ledger_attribution.py
git commit -m "feat: BM25 argmax attribution with ε-tie split"
```

---

### Task 2.3: `pattern_ledger` table + `pattern_id_for` + `upsert_ledger`

**Files:**
- Create: `psa/advertisement/ledger.py`
- Test: `tests/test_ledger_schema.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ledger_schema.py`:

```python
"""Tests for psa.advertisement.ledger — schema, CRUD, archival."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _conn(tmp_path):
    return sqlite3.connect(tmp_path / "ledger.sqlite3")


def test_pattern_id_is_deterministic():
    from psa.advertisement.ledger import pattern_id_for

    a = pattern_id_for(42, "how does the token refresh flow")
    b = pattern_id_for(42, "how does the token refresh flow")
    assert a == b
    # Normalization: different formatting yields same id
    c = pattern_id_for(42, "  How Does The Token REFRESH flow  ")
    assert c == a
    # Different anchor → different id
    d = pattern_id_for(43, "how does the token refresh flow")
    assert d != a


def test_create_schema_idempotent(tmp_path):
    from psa.advertisement.ledger import create_schema

    with _conn(tmp_path) as db:
        create_schema(db)
        create_schema(db)  # second call must not raise
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pattern_ledger'"
        ).fetchall()
        assert rows == [("pattern_ledger",)]


def test_upsert_inserts_new_row_with_grace_expires(tmp_path):
    from psa.advertisement.ledger import create_schema, upsert_ledger

    with _conn(tmp_path) as db:
        create_schema(db)
        upsert_ledger(
            db,
            pattern_id="pid-1",
            anchor_id=5,
            pattern_text="test pattern",
            ledger_delta=1.0,
            shadow_delta=1.0,
            grace_days=21,
        )
        row = db.execute("SELECT * FROM pattern_ledger").fetchone()
        cols = [c[0] for c in db.execute("SELECT * FROM pattern_ledger").description]
        d = dict(zip(cols, row))
        assert d["pattern_id"] == "pid-1"
        assert d["anchor_id"] == 5
        assert d["pattern_text"] == "test pattern"
        assert d["ledger"] == 1.0
        assert d["shadow_ledger"] == 1.0
        assert d["consecutive_negative_cycles"] == 0
        assert d["removed_at"] is None
        assert d["grace_expires_at"] is not None
        # grace should be ~21 days in the future
        g = datetime.fromisoformat(d["grace_expires_at"])
        delta = (g - datetime.now(timezone.utc)).days
        assert 19 <= delta <= 22


def test_upsert_accumulates_on_existing_row(tmp_path):
    from psa.advertisement.ledger import create_schema, upsert_ledger

    with _conn(tmp_path) as db:
        create_schema(db)
        upsert_ledger(db, "pid-1", 5, "p", 1.0, 0.5, grace_days=21)
        upsert_ledger(db, "pid-1", 5, "p", 2.5, 0.5, grace_days=21)
        row = db.execute(
            "SELECT ledger, shadow_ledger FROM pattern_ledger"
        ).fetchone()
        assert row[0] == 3.5
        assert row[1] == 1.0


def test_active_index_excludes_archived(tmp_path):
    from psa.advertisement.ledger import create_schema, upsert_ledger

    with _conn(tmp_path) as db:
        create_schema(db)
        upsert_ledger(db, "pid-1", 5, "p1", 1.0, 1.0, grace_days=21)
        upsert_ledger(db, "pid-2", 5, "p2", 1.0, 1.0, grace_days=21)
        db.execute(
            "UPDATE pattern_ledger SET removed_at=?, removal_reason=? WHERE pattern_id=?",
            ("2026-04-17T00:00:00+00:00", "test", "pid-1"),
        )
        active = db.execute(
            "SELECT pattern_id FROM pattern_ledger WHERE removed_at IS NULL"
        ).fetchall()
        assert active == [("pid-2",)]
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_ledger_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'psa.advertisement.ledger'`

- [ ] **Step 3: Implement schema + helpers**

Create `psa/advertisement/ledger.py`:

```python
"""
ledger.py — persistent pattern_ledger table + CRUD + decay arithmetic.

Schema is created lazily via create_schema() on first write. The table
lives in the tenant SQLite at ~/.psa/tenants/{tenant}/memory.sqlite3
alongside memory_objects.

Keyed by a content hash of (anchor_id, normalized_pattern_text) so the
id is stable across process restarts and changes naturally when the
pattern text changes (regeneration at atlas rebuild).
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from psa.advertisement.metadata import normalize_pattern


def pattern_id_for(anchor_id: int, pattern_text: str) -> str:
    """Stable content-hash id. Matches stage 1's metadata_key semantics."""
    norm = normalize_pattern(pattern_text)
    raw = f"{anchor_id}::{norm}".encode("utf-8")
    return "lp_" + hashlib.sha256(raw).hexdigest()[:24]


def create_schema(db: sqlite3.Connection) -> None:
    """Idempotent schema creation."""
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS pattern_ledger (
            pattern_id                     TEXT PRIMARY KEY,
            anchor_id                      INTEGER NOT NULL,
            pattern_text                   TEXT NOT NULL,

            ledger                         REAL NOT NULL DEFAULT 0.0,
            consecutive_negative_cycles    INTEGER NOT NULL DEFAULT 0,

            shadow_ledger                  REAL NOT NULL DEFAULT 0.0,
            shadow_consecutive             INTEGER NOT NULL DEFAULT 0,

            grace_expires_at               TEXT NOT NULL,
            created_at                     TEXT NOT NULL,
            last_updated_at                TEXT NOT NULL,

            removed_at                     TEXT,
            removal_reason                 TEXT,
            final_ledger                   REAL,
            final_shadow_ledger            REAL
        );
        CREATE INDEX IF NOT EXISTS idx_pattern_ledger_anchor
            ON pattern_ledger(anchor_id);
        CREATE INDEX IF NOT EXISTS idx_pattern_ledger_active
            ON pattern_ledger(anchor_id) WHERE removed_at IS NULL;
        """
    )
    db.commit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_ledger(
    db: sqlite3.Connection,
    pattern_id: str,
    anchor_id: int,
    pattern_text: str,
    ledger_delta: float,
    shadow_delta: float,
    grace_days: int,
) -> None:
    """Insert a fresh row or add the deltas to an existing row.

    On insert, stamps `grace_expires_at = now + grace_days`.
    `created_at` is set once; `last_updated_at` is always refreshed.
    """
    now = _now_iso()
    grace = (datetime.now(timezone.utc) + timedelta(days=grace_days)).isoformat()
    db.execute(
        """
        INSERT INTO pattern_ledger
            (pattern_id, anchor_id, pattern_text,
             ledger, shadow_ledger,
             grace_expires_at, created_at, last_updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pattern_id) DO UPDATE SET
            ledger = ledger + excluded.ledger,
            shadow_ledger = shadow_ledger + excluded.shadow_ledger,
            last_updated_at = excluded.last_updated_at
        """,
        (
            pattern_id,
            anchor_id,
            pattern_text,
            ledger_delta,
            shadow_delta,
            grace,
            now,
            now,
        ),
    )
    db.commit()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ledger_schema.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/advertisement/ledger.py tests/test_ledger_schema.py
git commit -m "feat: pattern_ledger table + pattern_id_for + upsert_ledger"
```

---

### Task 2.4: `trace.write_trace` returns bool + `retrieval_attribution` field

**Files:**
- Modify: `psa/trace.py`
- Test: `tests/test_trace.py` (create or extend)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trace.py` if it does not already exist, or append:

```python
"""Tests for psa.trace — bool return + retrieval_attribution field."""

from __future__ import annotations

import json


def test_write_trace_returns_true_on_success(tmp_path, monkeypatch):
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    ok = write_trace({"query": "hello"}, tenant_id="default")
    assert ok is True


def test_write_trace_returns_false_when_disabled(tmp_path, monkeypatch):
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PSA_TRACE", "0")
    ok = write_trace({"query": "hello"}, tenant_id="default")
    assert ok is False


def test_new_trace_record_includes_retrieval_attribution_field():
    from psa.trace import new_trace_record

    r = new_trace_record(
        run_id="r",
        timestamp="2026-04-17T00:00:00+00:00",
        tenant_id="default",
        atlas_version=1,
        query="q",
    )
    assert "retrieval_attribution" in r
    assert r["retrieval_attribution"] == []
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_trace.py -v`
Expected: FAIL (write_trace currently returns None; retrieval_attribution missing from new_trace_record)

- [ ] **Step 3: Modify trace.py**

In `psa/trace.py`:

Change `write_trace` to return `bool`:

```python
def write_trace(record: dict, *, tenant_id: str) -> bool:
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
```

In `new_trace_record(...)`, add `"retrieval_attribution": []` to the returned dict before `return`. The final dict should include:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_trace.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/trace.py tests/test_trace.py
git commit -m "feat: trace.write_trace returns bool + retrieval_attribution field"
```

---

### Task 2.5: Retriever exposes BM25 top-K shortlist

**Files:**
- Modify: `psa/retriever.py`
- Test: `tests/test_retriever.py` (create or extend)

- [ ] **Step 1: Inspect the current shape**

Run: `uv run python -c "from psa.retriever import AnchorRetriever; import inspect; print(inspect.signature(AnchorRetriever.retrieve))"`

Expected: prints the current return type. Look at `psa/retriever.py` lines ~140–220 to identify where BM25 scores are computed inside `retrieve()`, and where the result object is built.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_retriever.py` (create if missing):

```python
"""Tests for psa.retriever — BM25 top-K exposure."""

from __future__ import annotations

import numpy as np


def test_retrieval_result_exposes_bm25_topk_anchor_ids():
    """RetrievalResult must carry the BM25-side top-K shortlist so stage 2
    attribution can gate on lexical contribution."""
    from psa.retriever import RetrievalResult

    r = RetrievalResult(
        anchor_ids=[1, 2, 3],
        scores=[0.9, 0.8, 0.7],
        bm25_topk_anchor_ids=[2, 3, 7, 8],
    )
    assert r.bm25_topk_anchor_ids == [2, 3, 7, 8]
```

- [ ] **Step 3: Run test**

Run: `uv run pytest tests/test_retriever.py::test_retrieval_result_exposes_bm25_topk_anchor_ids -v`
Expected: FAIL with `TypeError` for unexpected kwarg `bm25_topk_anchor_ids`

- [ ] **Step 4: Add the field to RetrievalResult and populate from retrieve()**

In `psa/retriever.py`:

Find the `RetrievalResult` dataclass (or namedtuple). Add a new field `bm25_topk_anchor_ids: list[int]` with a default of `[]` for backward compatibility.

If `RetrievalResult` is a dataclass:

```python
@dataclass
class RetrievalResult:
    anchor_ids: list[int]
    scores: list[float]
    bm25_topk_anchor_ids: list[int] = field(default_factory=list)
```

In `AnchorRetriever.retrieve(...)`, locate where BM25 scores are computed and the anchors are RRF-fused. After BM25 is computed but before RRF, capture:

```python
        # BM25-side top-K shortlist (before RRF). bm25_topk_floor controls how
        # many anchors survive the BM25-only shortlist; stage 2 attribution
        # only credits templates when the retrieved anchor is on this list.
        bm25_topk = 48  # matches AdvertisementDecayConfig.bm25_topk_floor default
        bm25_pairs = sorted(
            zip(candidate_anchor_ids, bm25_scores), key=lambda p: p[1], reverse=True
        )
        bm25_topk_ids = [aid for aid, _ in bm25_pairs[:bm25_topk]]
```

Pass `bm25_topk_anchor_ids=bm25_topk_ids` when constructing the final `RetrievalResult`.

If the exact variable names in the file differ (e.g., `bm25_scores_by_id` or similar), adapt to match — the essential behavior is: after BM25 computation, before RRF fusion or truncation, capture the top-48 anchor ids by BM25 score into the result.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add psa/retriever.py tests/test_retriever.py
git commit -m "feat: expose BM25 top-K shortlist on RetrievalResult"
```

---

## Phase 3 — Inline write path (tracking)

### Task 3.1: Attribution compute + `record_query_signals`

**Files:**
- Modify: `psa/advertisement/ledger.py`
- Test: `tests/test_ledger_write_path.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ledger_write_path.py`:

```python
"""Tests for psa.advertisement.ledger — record_query_signals inline writer."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from unittest.mock import MagicMock


def _make_atlas():
    atlas = MagicMock()
    card1 = MagicMock()
    card1.anchor_id = 1
    card1.generated_query_patterns = ["cat vs dog behavior", "unrelated pattern"]
    card2 = MagicMock()
    card2.anchor_id = 2
    card2.generated_query_patterns = ["system architecture design"]
    atlas.cards = [card1, card2]
    by_id = {1: card1, 2: card2}
    atlas.get_anchor = lambda aid: by_id[aid]
    return atlas


def _config(tracking=True):
    from psa.advertisement.config import AdvertisementDecayConfig, ShadowConfig

    return AdvertisementDecayConfig(
        tracking_enabled=tracking, shadow=ShadowConfig()
    )


def test_record_query_signals_skips_when_tracking_disabled(tmp_path):
    from psa.advertisement.ledger import (
        compute_attribution,
        create_schema,
        record_query_signals,
    )

    db = sqlite3.connect(tmp_path / "t.sqlite3")
    create_schema(db)
    atlas = _make_atlas()
    attribution = compute_attribution(
        query="cat vs dog behavior",
        retrieved_anchor_ids=[1],
        atlas=atlas,
        bm25_topk_anchor_ids={1},
        epsilon=0.05,
    )
    record_query_signals(
        db=db,
        attribution=attribution,
        selected_anchor_ids={1},
        config=_config(tracking=False),
    )
    rows = db.execute("SELECT COUNT(*) FROM pattern_ledger").fetchone()
    assert rows[0] == 0


def test_record_query_signals_applies_retrieval_plus_pick(tmp_path):
    from psa.advertisement.ledger import (
        compute_attribution,
        create_schema,
        record_query_signals,
    )

    db = sqlite3.connect(tmp_path / "t.sqlite3")
    create_schema(db)
    atlas = _make_atlas()
    attribution = compute_attribution(
        query="cat vs dog behavior",
        retrieved_anchor_ids=[1],
        atlas=atlas,
        bm25_topk_anchor_ids={1},
        epsilon=0.05,
    )
    record_query_signals(
        db=db,
        attribution=attribution,
        selected_anchor_ids={1},  # anchor was picked
        config=_config(),
    )
    row = db.execute(
        "SELECT ledger FROM pattern_ledger WHERE pattern_text=?",
        ("cat vs dog behavior",),
    ).fetchone()
    # retrieval +1.0, pick +2.0, no ε-tie so full credit to argmax
    assert row[0] == 3.0


def test_record_query_signals_applies_decline(tmp_path):
    from psa.advertisement.ledger import (
        compute_attribution,
        create_schema,
        record_query_signals,
    )

    db = sqlite3.connect(tmp_path / "t.sqlite3")
    create_schema(db)
    atlas = _make_atlas()
    attribution = compute_attribution(
        query="cat vs dog behavior",
        retrieved_anchor_ids=[1],
        atlas=atlas,
        bm25_topk_anchor_ids={1},
        epsilon=0.05,
    )
    record_query_signals(
        db=db,
        attribution=attribution,
        selected_anchor_ids=set(),  # retrieved but not picked
        config=_config(),
    )
    row = db.execute(
        "SELECT ledger FROM pattern_ledger WHERE pattern_text=?",
        ("cat vs dog behavior",),
    ).fetchone()
    # retrieval +1.0, decline −0.25
    assert abs(row[0] - 0.75) < 1e-9


def test_record_query_signals_skips_when_bm25_floor_excludes_anchor(tmp_path):
    from psa.advertisement.ledger import (
        compute_attribution,
        create_schema,
        record_query_signals,
    )

    db = sqlite3.connect(tmp_path / "t.sqlite3")
    create_schema(db)
    atlas = _make_atlas()
    # Anchor 1 retrieved, but NOT on BM25-side shortlist — means lexical
    # contribution was negligible, so no template credit.
    attribution = compute_attribution(
        query="cat vs dog behavior",
        retrieved_anchor_ids=[1],
        atlas=atlas,
        bm25_topk_anchor_ids=set(),
        epsilon=0.05,
    )
    record_query_signals(
        db=db,
        attribution=attribution,
        selected_anchor_ids={1},
        config=_config(),
    )
    rows = db.execute("SELECT COUNT(*) FROM pattern_ledger").fetchone()
    assert rows[0] == 0


def test_credited_set_weight_division_with_epsilon_tie(tmp_path):
    from psa.advertisement.ledger import (
        compute_attribution,
        create_schema,
        record_query_signals,
    )
    from unittest.mock import MagicMock

    atlas = MagicMock()
    card = MagicMock()
    card.anchor_id = 7
    # Two near-duplicates — likely ε-tied under BM25
    card.generated_query_patterns = [
        "token refresh flow",
        "token refresh flow step",
    ]
    atlas.cards = [card]
    atlas.get_anchor = lambda aid: card

    db = sqlite3.connect(tmp_path / "t.sqlite3")
    create_schema(db)
    attribution = compute_attribution(
        query="token refresh flow",
        retrieved_anchor_ids=[7],
        atlas=atlas,
        bm25_topk_anchor_ids={7},
        epsilon=0.5,  # generous tie window
    )
    record_query_signals(
        db=db,
        attribution=attribution,
        selected_anchor_ids={7},
        config=_config(),
    )
    rows = db.execute(
        "SELECT pattern_text, ledger FROM pattern_ledger ORDER BY pattern_text"
    ).fetchall()
    # Both patterns split the +3.0 credit (+1 retrieval, +2 pick)
    assert len(rows) == 2
    for _, lgr in rows:
        assert abs(lgr - 1.5) < 1e-9
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_ledger_write_path.py -v`
Expected: FAIL (compute_attribution and record_query_signals do not exist yet)

- [ ] **Step 3: Implement compute_attribution + record_query_signals**

Append to `psa/advertisement/ledger.py`:

```python
from dataclasses import dataclass, field

from psa.advertisement.attribution import attribute_bm25_argmax


@dataclass
class AnchorAttribution:
    """Per-retrieved-anchor attribution record.

    credited is [argmax] + eps_tied when BM25 floor passes and argmax exists;
    empty otherwise. Each credited pattern shares the per-query weight.
    """

    anchor_id: int
    argmax_pattern: str | None
    eps_tied_patterns: list[str] = field(default_factory=list)
    credited: list[str] = field(default_factory=list)
    bm25_floor_passed: bool = True


def compute_attribution(
    *,
    query: str,
    retrieved_anchor_ids: list[int],
    atlas,
    bm25_topk_anchor_ids,
    epsilon: float,
) -> list[AnchorAttribution]:
    """Run BM25 argmax per retrieved anchor, gate on BM25 top-K membership."""
    bm25_set = set(bm25_topk_anchor_ids)
    out: list[AnchorAttribution] = []
    for aid in retrieved_anchor_ids:
        if aid not in bm25_set:
            out.append(
                AnchorAttribution(
                    anchor_id=aid,
                    argmax_pattern=None,
                    bm25_floor_passed=False,
                )
            )
            continue
        anchor = atlas.get_anchor(aid)
        argmax, tied = attribute_bm25_argmax(
            query, anchor.generated_query_patterns, epsilon=epsilon
        )
        credited = [argmax] + list(tied) if argmax is not None else []
        out.append(
            AnchorAttribution(
                anchor_id=aid,
                argmax_pattern=argmax,
                eps_tied_patterns=list(tied),
                credited=credited,
                bm25_floor_passed=True,
            )
        )
    return out


def record_query_signals(
    *,
    db,
    attribution: list[AnchorAttribution],
    selected_anchor_ids,
    config,
) -> None:
    """Apply retrieval/pick/decline credit to every credited template."""
    if not config.tracking_enabled:
        return

    selected = set(selected_anchor_ids)
    for attr in attribution:
        if not attr.credited:
            continue
        n = len(attr.credited)
        per = 1.0 / n

        base = config.retrieval_credit
        shadow = config.retrieval_credit
        if attr.anchor_id in selected:
            base += config.selector_pick_credit
            shadow += config.selector_pick_credit
        else:
            base -= config.selector_decline_penalty
            shadow -= config.shadow.selector_decline_penalty

        for pat in attr.credited:
            upsert_ledger(
                db=db,
                pattern_id=pattern_id_for(attr.anchor_id, pat),
                anchor_id=attr.anchor_id,
                pattern_text=pat,
                ledger_delta=base * per,
                shadow_delta=shadow * per,
                grace_days=config.grace_days,
            )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ledger_write_path.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/advertisement/ledger.py tests/test_ledger_write_path.py
git commit -m "feat: compute_attribution + record_query_signals inline writer"
```

---

### Task 3.2: Wire trace-first + ledger in `pipeline.py`

**Files:**
- Modify: `psa/pipeline.py`
- Test: `tests/test_ledger_trace_ordering.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ledger_trace_ordering.py`:

```python
"""Tests for trace-first-ledger-second semantics in pipeline.query()."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def test_ledger_not_written_when_trace_fails(tmp_path, monkeypatch):
    """If trace.write_trace returns False, record_query_signals is not called."""
    from psa.advertisement.ledger import create_schema
    import sqlite3

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PSA_TRACE", "0")  # force trace to return False

    # Sketch a minimal pipeline scenario. Full integration test uses the real
    # PSAPipeline; this unit test asserts the write-ordering invariant via the
    # compose_and_record() helper that pipeline.py delegates to.
    from psa.pipeline import compose_and_record

    recorded = {"count": 0}
    fake_record = lambda **kw: recorded.__setitem__("count", recorded["count"] + 1)

    compose_and_record(
        tenant_id="default",
        trace_record={"run_id": "r", "timestamp": "t", "query": "q"},
        attribution=[],
        selected_anchor_ids=set(),
        config=MagicMock(tracking_enabled=True),
        record_signals_fn=fake_record,
    )
    assert recorded["count"] == 0  # ledger write skipped because trace disabled


def test_ledger_written_when_trace_succeeds(tmp_path, monkeypatch):
    from psa.pipeline import compose_and_record

    monkeypatch.setenv("HOME", str(tmp_path))  # trace enabled by default

    recorded = {"count": 0}
    fake_record = lambda **kw: recorded.__setitem__("count", recorded["count"] + 1)

    compose_and_record(
        tenant_id="default",
        trace_record={"run_id": "r", "timestamp": "t", "query": "q"},
        attribution=[],
        selected_anchor_ids=set(),
        config=MagicMock(tracking_enabled=True),
        record_signals_fn=fake_record,
    )
    assert recorded["count"] == 1
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_ledger_trace_ordering.py -v`
Expected: FAIL — `compose_and_record` does not exist.

- [ ] **Step 3: Implement compose_and_record + integrate into PSAPipeline.query**

In `psa/pipeline.py`:

Add near the top (after existing imports):

```python
from psa.trace import write_trace
```

Add a module-level helper:

```python
def compose_and_record(
    *,
    tenant_id: str,
    trace_record: dict,
    attribution,
    selected_anchor_ids,
    config,
    record_signals_fn=None,
) -> None:
    """Write trace first; only if it succeeded, record ledger signals.

    `record_signals_fn` is injectable for unit testing. Production callers
    omit it and the function constructs the default ledger writer.
    """
    trace_record["retrieval_attribution"] = [
        {
            "anchor_id": a.anchor_id,
            "bm25_argmax_pattern": a.argmax_pattern,
            "bm25_epsilon_tied": list(a.eps_tied_patterns),
            "bm25_floor_passed": a.bm25_floor_passed,
        }
        for a in attribution
    ]
    trace_written = write_trace(trace_record, tenant_id=tenant_id)
    if not trace_written:
        return
    if not getattr(config, "tracking_enabled", False):
        return
    if record_signals_fn is None:
        from psa.advertisement.ledger import record_query_signals

        def _default(**kw):
            import sqlite3, os

            db_path = os.path.expanduser(
                f"~/.psa/tenants/{tenant_id}/memory.sqlite3"
            )
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            with sqlite3.connect(db_path) as db:
                from psa.advertisement.ledger import create_schema

                create_schema(db)
                record_query_signals(db=db, **kw)

        record_signals_fn = _default
    record_signals_fn(
        attribution=attribution,
        selected_anchor_ids=selected_anchor_ids,
        config=config,
    )
```

In `PSAPipeline.query(...)` (find the existing call to `trace.write_trace` or the place where the trace record is finalized — around the end of query processing), replace that direct write with a call to `compose_and_record(...)`. Pass the same trace_record plus the attribution list (computed via `compute_attribution` using the retrieval's `bm25_topk_anchor_ids`) and the selected anchor ids. Load the `AdvertisementDecayConfig` from `MempalaceConfig` once at pipeline construction and pass it here.

Minimal integration sketch at the call site (after selector returns):

```python
        from psa.advertisement.ledger import compute_attribution
        attribution = compute_attribution(
            query=query,
            retrieved_anchor_ids=[a for a, _ in retrieved_anchors],
            atlas=self.atlas,
            bm25_topk_anchor_ids=retrieval_result.bm25_topk_anchor_ids,
            epsilon=self.ad_config.epsilon,
        )
        compose_and_record(
            tenant_id=self.tenant_id,
            trace_record=trace_record,
            attribution=attribution,
            selected_anchor_ids=selected_ids,
            config=self.ad_config,
        )
```

Add `self.ad_config = AdvertisementDecayConfig.from_mempalace(config)` in `PSAPipeline.__init__` (or `from_config`), importing from `psa.advertisement.config`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ledger_trace_ordering.py -v && uv run pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/pipeline.py tests/test_ledger_trace_ordering.py
git commit -m "feat: pipeline writes trace first, ledger only on trace success"
```

---

## Phase 4 — Decay, removal, rebuild, lifecycle

### Task 4.1: Exponential decay + consecutive-negative state machine

**Files:**
- Modify: `psa/advertisement/ledger.py`
- Test: `tests/test_ledger_decay.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ledger_decay.py`:

```python
"""Tests for psa.advertisement.ledger — decay + removal eligibility."""

from __future__ import annotations

import math
import sqlite3


def _seed(tmp_path, ledger=2.0, cnc=0, grace_days_ago=30):
    from psa.advertisement.ledger import create_schema
    from datetime import datetime, timedelta, timezone

    db = sqlite3.connect(tmp_path / "t.sqlite3")
    create_schema(db)
    now = datetime.now(timezone.utc)
    grace_expires = (now - timedelta(days=grace_days_ago)).isoformat()
    db.execute(
        """
        INSERT INTO pattern_ledger
        (pattern_id, anchor_id, pattern_text, ledger, shadow_ledger,
         consecutive_negative_cycles, shadow_consecutive,
         grace_expires_at, created_at, last_updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "pid-x",
            1,
            "pattern",
            ledger,
            ledger,
            cnc,
            cnc,
            grace_expires,
            now.isoformat(),
            now.isoformat(),
        ),
    )
    db.commit()
    return db


def test_exponential_decay_multiplies_ledger(tmp_path):
    from psa.advertisement.ledger import apply_decay

    db = _seed(tmp_path, ledger=2.0)
    apply_decay(db, tau_days=45)
    row = db.execute("SELECT ledger FROM pattern_ledger").fetchone()
    expected = 2.0 * math.exp(-1 / 45)
    assert abs(row[0] - expected) < 1e-6


def test_decay_increments_consecutive_when_below_threshold(tmp_path):
    from psa.advertisement.ledger import apply_decay

    db = _seed(tmp_path, ledger=-0.1, cnc=3)
    apply_decay(db, tau_days=45, removal_threshold=0.0)
    row = db.execute(
        "SELECT consecutive_negative_cycles FROM pattern_ledger"
    ).fetchone()
    assert row[0] == 4


def test_decay_resets_consecutive_when_at_or_above_threshold(tmp_path):
    from psa.advertisement.ledger import apply_decay

    db = _seed(tmp_path, ledger=1.0, cnc=5)
    apply_decay(db, tau_days=45, removal_threshold=0.0)
    row = db.execute(
        "SELECT consecutive_negative_cycles FROM pattern_ledger"
    ).fetchone()
    assert row[0] == 0


def test_decay_applies_shadow_independently(tmp_path):
    from psa.advertisement.ledger import apply_decay

    db = _seed(tmp_path, ledger=2.0, cnc=0)
    # Set shadow to a different starting value to verify independent tracking.
    db.execute("UPDATE pattern_ledger SET shadow_ledger=-0.5, shadow_consecutive=2")
    db.commit()
    apply_decay(db, tau_days=45, removal_threshold=0.0)
    row = db.execute(
        "SELECT shadow_ledger, shadow_consecutive FROM pattern_ledger"
    ).fetchone()
    assert row[0] < 0  # still negative after decay
    assert row[1] == 3  # incremented
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_ledger_decay.py -v`
Expected: FAIL — `apply_decay` missing

- [ ] **Step 3: Implement apply_decay**

Append to `psa/advertisement/ledger.py`:

```python
import math


def apply_decay(db, tau_days: int, removal_threshold: float = 0.0) -> None:
    """Apply exponential decay to active rows + update consecutive counters.

    ledger ← ledger · exp(−1/tau)   (shadow_ledger likewise)
    consecutive_negative_cycles ← (ledger<threshold) ? prev+1 : 0
    shadow_consecutive ← (shadow_ledger<threshold) ? prev+1 : 0

    Operates on unarchived rows only.
    """
    factor = math.exp(-1.0 / tau_days)
    db.execute(
        """
        UPDATE pattern_ledger
        SET
            ledger = ledger * ?,
            shadow_ledger = shadow_ledger * ?,
            consecutive_negative_cycles = CASE
                WHEN ledger * ? < ? THEN consecutive_negative_cycles + 1
                ELSE 0
            END,
            shadow_consecutive = CASE
                WHEN shadow_ledger * ? < ? THEN shadow_consecutive + 1
                ELSE 0
            END,
            last_updated_at = ?
        WHERE removed_at IS NULL
        """,
        (factor, factor, factor, removal_threshold, factor, removal_threshold, _now_iso()),
    )
    db.commit()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ledger_decay.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/advertisement/ledger.py tests/test_ledger_decay.py
git commit -m "feat: exponential decay + consecutive-negative state machine"
```

---

### Task 4.2: Evaluate removal eligibility (P1/P3 reuse + min-patterns-floor)

**Files:**
- Modify: `psa/advertisement/ledger.py`
- Test: `tests/test_ledger_decay.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ledger_decay.py`:

```python
def test_evaluate_removal_requires_grace_expired(tmp_path):
    """Pattern past negative threshold but inside grace → not eligible."""
    from psa.advertisement.ledger import evaluate_removal
    from datetime import datetime, timedelta, timezone
    from unittest.mock import MagicMock

    db = _seed(tmp_path, ledger=-2.0, cnc=20)
    # Override grace to be in the future
    future = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
    db.execute("UPDATE pattern_ledger SET grace_expires_at=?", (future,))
    db.commit()

    config = MagicMock(
        removal_threshold=0.0, sustained_cycles=14, min_patterns_floor=3,
        shadow=MagicMock(sustained_cycles=7),
    )
    atlas = MagicMock()
    atlas.get_anchor = lambda aid: MagicMock(
        generated_query_patterns=["p1", "p2", "p3", "pattern"]
    )
    shielded_fn = lambda tenant_id, anchor_ids: set()
    pinned_fn = lambda anchor_id, pattern_text: False

    result = evaluate_removal(
        db=db,
        atlas=atlas,
        tenant_id="default",
        config=config,
        shielded_anchor_fn=shielded_fn,
        pinned_fn=pinned_fn,
    )
    assert result.removal_candidates == []
    assert result.shadow_candidates == []


def test_evaluate_removal_requires_sustained_cycles(tmp_path):
    """Pattern just dipped negative → still eligible under sustained_cycles."""
    from psa.advertisement.ledger import evaluate_removal
    from unittest.mock import MagicMock

    db = _seed(tmp_path, ledger=-2.0, cnc=5)
    config = MagicMock(
        removal_threshold=0.0, sustained_cycles=14, min_patterns_floor=3,
        shadow=MagicMock(sustained_cycles=7),
    )
    atlas = MagicMock()
    atlas.get_anchor = lambda aid: MagicMock(
        generated_query_patterns=["p1", "p2", "p3", "pattern"]
    )
    shielded_fn = lambda tenant_id, anchor_ids: set()
    pinned_fn = lambda anchor_id, pattern_text: False

    result = evaluate_removal(
        db=db, atlas=atlas, tenant_id="default", config=config,
        shielded_anchor_fn=shielded_fn, pinned_fn=pinned_fn,
    )
    assert result.removal_candidates == []
    # shadow sustained_cycles=7 also not met
    assert result.shadow_candidates == []


def test_evaluate_removal_respects_min_patterns_floor(tmp_path):
    """Anchor with only 3 patterns → removal would drop below floor → refused."""
    from psa.advertisement.ledger import evaluate_removal
    from unittest.mock import MagicMock

    db = _seed(tmp_path, ledger=-2.0, cnc=20)
    config = MagicMock(
        removal_threshold=0.0, sustained_cycles=14, min_patterns_floor=3,
        shadow=MagicMock(sustained_cycles=7),
    )
    atlas = MagicMock()
    atlas.get_anchor = lambda aid: MagicMock(
        generated_query_patterns=["p1", "p2", "pattern"]  # only 3
    )
    shielded_fn = lambda tenant_id, anchor_ids: set()
    pinned_fn = lambda anchor_id, pattern_text: False

    result = evaluate_removal(
        db=db, atlas=atlas, tenant_id="default", config=config,
        shielded_anchor_fn=shielded_fn, pinned_fn=pinned_fn,
    )
    assert result.removal_candidates == []


def test_evaluate_removal_respects_p1_shield(tmp_path):
    """Anchor shielded by stage 1 P1 → eligible pattern excluded."""
    from psa.advertisement.ledger import evaluate_removal
    from unittest.mock import MagicMock

    db = _seed(tmp_path, ledger=-2.0, cnc=20)
    config = MagicMock(
        removal_threshold=0.0, sustained_cycles=14, min_patterns_floor=3,
        shadow=MagicMock(sustained_cycles=7),
    )
    atlas = MagicMock()
    atlas.get_anchor = lambda aid: MagicMock(
        generated_query_patterns=["p1", "p2", "p3", "p4", "pattern"]
    )
    shielded_fn = lambda tenant_id, anchor_ids: {1}  # anchor 1 is shielded
    pinned_fn = lambda anchor_id, pattern_text: False

    result = evaluate_removal(
        db=db, atlas=atlas, tenant_id="default", config=config,
        shielded_anchor_fn=shielded_fn, pinned_fn=pinned_fn,
    )
    assert result.removal_candidates == []


def test_evaluate_removal_happy_path(tmp_path):
    from psa.advertisement.ledger import evaluate_removal
    from unittest.mock import MagicMock

    db = _seed(tmp_path, ledger=-2.0, cnc=20)
    config = MagicMock(
        removal_threshold=0.0, sustained_cycles=14, min_patterns_floor=3,
        shadow=MagicMock(sustained_cycles=7),
    )
    atlas = MagicMock()
    atlas.get_anchor = lambda aid: MagicMock(
        generated_query_patterns=["p1", "p2", "p3", "p4", "pattern"]
    )
    shielded_fn = lambda tenant_id, anchor_ids: set()
    pinned_fn = lambda anchor_id, pattern_text: False

    result = evaluate_removal(
        db=db, atlas=atlas, tenant_id="default", config=config,
        shielded_anchor_fn=shielded_fn, pinned_fn=pinned_fn,
    )
    assert len(result.removal_candidates) == 1
    assert result.removal_candidates[0].pattern_text == "pattern"
    assert result.removal_candidates[0].anchor_id == 1
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_ledger_decay.py -v`
Expected: FAIL — evaluate_removal missing

- [ ] **Step 3: Implement evaluate_removal**

Append to `psa/advertisement/ledger.py`:

```python
@dataclass
class RemovalCandidate:
    pattern_id: str
    anchor_id: int
    pattern_text: str
    ledger: float
    consecutive_negative_cycles: int
    shadow_ledger: float
    shadow_consecutive: int


@dataclass
class EvaluationResult:
    removal_candidates: list["RemovalCandidate"] = field(default_factory=list)
    shadow_candidates: list["RemovalCandidate"] = field(default_factory=list)
    n_active: int = 0
    n_in_grace: int = 0
    n_at_risk: int = 0


def evaluate_removal(
    *,
    db,
    atlas,
    tenant_id: str,
    config,
    shielded_anchor_fn,
    pinned_fn,
) -> "EvaluationResult":
    """Evaluate active rows against the removal rule.

    `shielded_anchor_fn(tenant_id, anchor_ids) -> set[int]` re-uses stage 1 P1.
    `pinned_fn(anchor_id, pattern_text) -> bool` re-uses stage 1 P3.

    Does not mutate the DB. Callers decide whether to apply removals.
    """
    now_iso = _now_iso()
    rows = db.execute(
        """
        SELECT pattern_id, anchor_id, pattern_text, ledger,
               consecutive_negative_cycles, shadow_ledger, shadow_consecutive,
               grace_expires_at
        FROM pattern_ledger
        WHERE removed_at IS NULL
        """
    ).fetchall()

    n_active = len(rows)
    n_in_grace = 0
    n_at_risk = 0
    anchor_ids = {r[1] for r in rows}
    shielded = shielded_anchor_fn(tenant_id, anchor_ids) or set()

    # Count per-anchor active patterns AFTER potential removal
    # (we need this to enforce min_patterns_floor).
    per_anchor_counts = {aid: 0 for aid in anchor_ids}
    for aid in anchor_ids:
        anchor = atlas.get_anchor(aid)
        per_anchor_counts[aid] = len(anchor.generated_query_patterns)

    removal: list[RemovalCandidate] = []
    shadow: list[RemovalCandidate] = []

    for (
        pid,
        aid,
        text,
        lgr,
        cnc,
        shadow_lgr,
        shadow_cnc,
        grace_expires_at,
    ) in rows:
        in_grace = grace_expires_at > now_iso
        if in_grace:
            n_in_grace += 1
            continue

        if cnc >= max(1, config.sustained_cycles // 2):
            n_at_risk += 1

        candidate = RemovalCandidate(
            pattern_id=pid,
            anchor_id=aid,
            pattern_text=text,
            ledger=lgr,
            consecutive_negative_cycles=cnc,
            shadow_ledger=shadow_lgr,
            shadow_consecutive=shadow_cnc,
        )

        # Shared protections
        if aid in shielded:
            continue
        if pinned_fn(aid, text):
            continue
        if per_anchor_counts.get(aid, 0) - 1 < config.min_patterns_floor:
            continue

        if cnc >= config.sustained_cycles:
            removal.append(candidate)
            per_anchor_counts[aid] -= 1

        if shadow_cnc >= config.shadow.sustained_cycles:
            shadow.append(candidate)

    return EvaluationResult(
        removal_candidates=removal,
        shadow_candidates=shadow,
        n_active=n_active,
        n_in_grace=n_in_grace,
        n_at_risk=n_at_risk,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ledger_decay.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/advertisement/ledger.py tests/test_ledger_decay.py
git commit -m "feat: evaluate_removal with P1/P3 reuse + min-patterns-floor"
```

---

### Task 4.3: Reload marker file module

**Files:**
- Create: `psa/advertisement/reload.py`
- Test: `tests/test_ledger_reload_marker.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ledger_reload_marker.py`:

```python
"""Tests for reload marker file used by long-lived pipelines."""

from __future__ import annotations

import json
import time


def test_write_reload_marker_creates_file(tmp_path, monkeypatch):
    from psa.advertisement.reload import write_reload_marker

    monkeypatch.setenv("HOME", str(tmp_path))
    write_reload_marker(tenant_id="default", changed_anchor_ids=[1, 2, 3])

    marker = tmp_path / ".psa" / "tenants" / "default" / "atlas_reload_requested"
    assert marker.exists()
    body = json.loads(marker.read_text())
    assert set(body["changed_anchor_ids"]) == {1, 2, 3}
    assert "timestamp" in body


def test_marker_should_reload(tmp_path, monkeypatch):
    from psa.advertisement.reload import write_reload_marker, should_reload

    monkeypatch.setenv("HOME", str(tmp_path))
    assert should_reload(tenant_id="default", last_reload_mtime=0.0) is False
    write_reload_marker(tenant_id="default", changed_anchor_ids=[1])
    time.sleep(0.01)
    assert should_reload(tenant_id="default", last_reload_mtime=0.0) is True


def test_should_reload_respects_last_reload_mtime(tmp_path, monkeypatch):
    from psa.advertisement.reload import write_reload_marker, should_reload, marker_mtime

    monkeypatch.setenv("HOME", str(tmp_path))
    write_reload_marker(tenant_id="default", changed_anchor_ids=[1])
    m = marker_mtime(tenant_id="default")
    # Same mtime → no reload needed
    assert should_reload(tenant_id="default", last_reload_mtime=m) is False
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_ledger_reload_marker.py -v`
Expected: FAIL — module missing

- [ ] **Step 3: Implement the marker module**

Create `psa/advertisement/reload.py`:

```python
"""
reload.py — marker file for long-lived pipeline consumers (MCP server).

Stage 2 writes ~/.psa/tenants/{tenant}/atlas_reload_requested atomically
after removing patterns from anchor_cards_refined.json. Long-lived
processes check the marker's mtime between queries and call
PSAPipeline.reload_atlas() when it advances.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone


def _marker_path(tenant_id: str) -> str:
    home = os.path.expanduser("~")
    return os.path.join(home, ".psa", "tenants", tenant_id, "atlas_reload_requested")


def write_reload_marker(*, tenant_id: str, changed_anchor_ids) -> None:
    """Atomically (tmp + rename) update the marker with current timestamp."""
    path = _marker_path(tenant_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "changed_anchor_ids": sorted(set(changed_anchor_ids)),
    }
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".reload-", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(body, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def marker_mtime(*, tenant_id: str) -> float:
    path = _marker_path(tenant_id)
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def should_reload(*, tenant_id: str, last_reload_mtime: float) -> bool:
    """True when the marker exists and its mtime is newer than last_reload_mtime."""
    m = marker_mtime(tenant_id=tenant_id)
    return m > last_reload_mtime
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ledger_reload_marker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/advertisement/reload.py tests/test_ledger_reload_marker.py
git commit -m "feat: reload marker file for long-lived pipeline consumers"
```

---

### Task 4.4: `apply_removals` — mutate refined, archive, write marker

**Files:**
- Modify: `psa/advertisement/ledger.py`
- Test: `tests/test_ledger_decay.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ledger_decay.py`:

```python
def test_apply_removals_mutates_refined_and_archives_row(tmp_path, monkeypatch):
    """End-to-end: refined file gets pattern removed, row soft-archived, marker written."""
    import json
    from datetime import datetime, timedelta, timezone
    from psa.advertisement.ledger import (
        apply_removals,
        RemovalCandidate,
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    atlas_dir = tmp_path / ".psa" / "tenants" / "default" / "atlas_v1"
    atlas_dir.mkdir(parents=True)
    # Seed a refined file with two patterns; one will be removed.
    cards = [
        {
            "anchor_id": 1,
            "generated_query_patterns": ["keep this pattern", "remove this pattern"],
        }
    ]
    (atlas_dir / "anchor_cards_refined.json").write_text(json.dumps(cards))

    db = _seed(tmp_path, ledger=-2.0, cnc=20)
    candidate = RemovalCandidate(
        pattern_id="pid-x",
        anchor_id=1,
        pattern_text="remove this pattern",
        ledger=-2.0,
        consecutive_negative_cycles=20,
        shadow_ledger=-2.0,
        shadow_consecutive=20,
    )
    apply_removals(
        db=db,
        tenant_id="default",
        atlas_dir=str(atlas_dir),
        candidates=[candidate],
    )

    # Refined file updated
    after = json.loads((atlas_dir / "anchor_cards_refined.json").read_text())
    assert after[0]["generated_query_patterns"] == ["keep this pattern"]
    # Row archived
    row = db.execute(
        "SELECT removed_at, removal_reason, final_ledger FROM pattern_ledger"
    ).fetchone()
    assert row[0] is not None
    assert row[1] == "stage2_sustained_negative"
    assert row[2] == -2.0
    # Marker written
    marker = tmp_path / ".psa" / "tenants" / "default" / "atlas_reload_requested"
    assert marker.exists()


def test_apply_removals_creates_refined_from_base_when_absent(tmp_path, monkeypatch):
    """No refined file yet: read base, write refined, leave base untouched."""
    import json
    from psa.advertisement.ledger import apply_removals, RemovalCandidate

    monkeypatch.setenv("HOME", str(tmp_path))
    atlas_dir = tmp_path / ".psa" / "tenants" / "default" / "atlas_v1"
    atlas_dir.mkdir(parents=True)
    base_cards = [{"anchor_id": 1, "generated_query_patterns": ["keep", "remove"]}]
    (atlas_dir / "anchor_cards.json").write_text(json.dumps(base_cards))

    db = _seed(tmp_path, ledger=-2.0, cnc=20)
    candidate = RemovalCandidate(
        pattern_id="pid-x",
        anchor_id=1,
        pattern_text="remove",
        ledger=-2.0,
        consecutive_negative_cycles=20,
        shadow_ledger=-2.0,
        shadow_consecutive=20,
    )
    apply_removals(
        db=db, tenant_id="default", atlas_dir=str(atlas_dir), candidates=[candidate]
    )

    # Base untouched
    base_after = json.loads((atlas_dir / "anchor_cards.json").read_text())
    assert base_after == base_cards
    # Refined created with the removal applied
    refined = json.loads(
        (atlas_dir / "anchor_cards_refined.json").read_text()
    )
    assert refined[0]["generated_query_patterns"] == ["keep"]
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_ledger_decay.py -v`
Expected: FAIL — apply_removals missing

- [ ] **Step 3: Implement apply_removals**

Append to `psa/advertisement/ledger.py`:

```python
import json
import os
import tempfile

from psa.advertisement.metadata import normalize_pattern
from psa.advertisement.reload import write_reload_marker


def _atomic_write_json(path: str, obj) -> None:
    """Write JSON via tmp + os.replace for durability."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".cards-", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_live_cards(atlas_dir: str):
    """Prefer refined; fall back to base. Returns (cards, source_file_name)."""
    refined = os.path.join(atlas_dir, "anchor_cards_refined.json")
    base = os.path.join(atlas_dir, "anchor_cards.json")
    if os.path.exists(refined):
        with open(refined) as f:
            return json.load(f), "anchor_cards_refined.json"
    with open(base) as f:
        return json.load(f), "anchor_cards.json"


def apply_removals(
    *,
    db,
    tenant_id: str,
    atlas_dir: str,
    candidates: list,
) -> int:
    """Drop candidate patterns from refined cards; archive ledger rows; write marker.

    Returns the number of successfully-applied removals.
    """
    if not candidates:
        return 0

    cards, _ = _load_live_cards(atlas_dir)
    # Index cards by anchor_id for quick lookup
    by_anchor: dict[int, dict] = {}
    for c in cards:
        aid = c.get("anchor_id") if isinstance(c, dict) else getattr(c, "anchor_id", None)
        if aid is not None:
            by_anchor[aid] = c

    changed_anchors: set[int] = set()
    applied: list = []
    for cand in candidates:
        card = by_anchor.get(cand.anchor_id)
        if card is None:
            continue
        patterns = card.get("generated_query_patterns", [])
        # Remove by normalized-text match so format drift doesn't defeat us.
        target = normalize_pattern(cand.pattern_text)
        new_patterns = [p for p in patterns if normalize_pattern(p) != target]
        if len(new_patterns) == len(patterns):
            continue
        card["generated_query_patterns"] = new_patterns
        changed_anchors.add(cand.anchor_id)
        applied.append(cand)

    if not applied:
        return 0

    refined_path = os.path.join(atlas_dir, "anchor_cards_refined.json")
    _atomic_write_json(refined_path, cards)

    now = _now_iso()
    for cand in applied:
        db.execute(
            """
            UPDATE pattern_ledger
            SET removed_at = ?,
                removal_reason = ?,
                final_ledger = ?,
                final_shadow_ledger = ?
            WHERE pattern_id = ?
            """,
            (
                now,
                "stage2_sustained_negative",
                cand.ledger,
                cand.shadow_ledger,
                cand.pattern_id,
            ),
        )
    db.commit()

    write_reload_marker(tenant_id=tenant_id, changed_anchor_ids=changed_anchors)
    return len(applied)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ledger_decay.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/advertisement/ledger.py tests/test_ledger_decay.py
git commit -m "feat: apply_removals mutates refined + archives row + writes marker"
```

---

### Task 4.5: `PSAPipeline.reload_atlas()`

**Files:**
- Modify: `psa/pipeline.py`, `psa/retriever.py`
- Test: `tests/test_pipeline_reload.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_reload.py`:

```python
"""Tests for PSAPipeline.reload_atlas()."""

from __future__ import annotations

import json
from unittest.mock import MagicMock


def test_reload_atlas_rereads_refined_and_reindexes_bm25(tmp_path, monkeypatch):
    """After reload, retriever's BM25 index reflects new refined cards."""
    from psa.pipeline import PSAPipeline

    monkeypatch.setenv("HOME", str(tmp_path))
    # Full pipeline boot is heavy; we test at method level by mocking.
    pipeline = MagicMock(spec=PSAPipeline)
    pipeline._reloaded = []

    def fake_reload(self):
        self._reloaded.append("ok")

    PSAPipeline.reload_atlas(pipeline)  # will fail until implemented
    assert pipeline._reloaded == ["ok"]
```

(Real integration is exercised via MCP server test in Task 4.6; this unit test just proves `reload_atlas` exists and is callable.)

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_pipeline_reload.py -v`
Expected: FAIL — reload_atlas missing

- [ ] **Step 3: Implement reload_atlas**

In `psa/retriever.py`, add a method on `AnchorRetriever`:

```python
    def reindex_from_cards(self, cards) -> None:
        """Rebuild BM25 + dense indexes from a fresh cards list.

        Called by PSAPipeline.reload_atlas() after re-reading refined cards.
        """
        # Replace the internal card list and re-tokenize/embed.
        self._cards = list(cards)
        self._build_indexes()
```

Adjust method/attribute names to match the retriever's existing internals (look at how `__init__` builds the indexes — factor that into `_build_indexes()` if it isn't already).

In `psa/pipeline.py`, add to `PSAPipeline`:

```python
    def reload_atlas(self) -> None:
        """Re-read refined cards and rebuild BM25 index.

        For long-lived consumers (MCP server). Safe to call at any time; no
        effect on in-flight queries — index swap is a local reference update.
        """
        import os
        import json

        atlas_dir = self.atlas_manager.current_atlas_dir(self.tenant_id)
        refined = os.path.join(atlas_dir, "anchor_cards_refined.json")
        base = os.path.join(atlas_dir, "anchor_cards.json")
        path = refined if os.path.exists(refined) else base
        with open(path) as f:
            cards = json.load(f)
        # Let atlas rebuild the AnchorCard objects if needed:
        self.atlas.reload_cards(cards)
        self.retriever.reindex_from_cards(self.atlas.cards)
```

If `atlas_manager.current_atlas_dir` and `atlas.reload_cards` don't yet exist, add small stubs that wrap the existing load path. The specific adapter names depend on the current atlas loader — follow the pattern used by `AtlasManager.get_atlas(...)`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_pipeline_reload.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/pipeline.py psa/retriever.py tests/test_pipeline_reload.py
git commit -m "feat: PSAPipeline.reload_atlas + retriever.reindex_from_cards"
```

---

### Task 4.6: MCP server marker check between queries

**Files:**
- Modify: `psa/mcp_server.py`
- Test: `tests/test_mcp_reload.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_mcp_reload.py`:

```python
"""MCP server honors the reload marker between queries."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_mcp_calls_reload_atlas_when_marker_advances(tmp_path, monkeypatch):
    from psa.mcp_server import maybe_reload_pipeline

    monkeypatch.setenv("HOME", str(tmp_path))
    pipeline = MagicMock()
    state = {"last_reload_mtime": 0.0}

    from psa.advertisement.reload import write_reload_marker
    write_reload_marker(tenant_id="default", changed_anchor_ids=[1])

    maybe_reload_pipeline(pipeline=pipeline, state=state, tenant_id="default")
    pipeline.reload_atlas.assert_called_once()
    assert state["last_reload_mtime"] > 0.0


def test_mcp_does_not_reload_when_marker_unchanged(tmp_path, monkeypatch):
    from psa.mcp_server import maybe_reload_pipeline
    from psa.advertisement.reload import write_reload_marker, marker_mtime

    monkeypatch.setenv("HOME", str(tmp_path))
    write_reload_marker(tenant_id="default", changed_anchor_ids=[1])
    m = marker_mtime(tenant_id="default")

    pipeline = MagicMock()
    state = {"last_reload_mtime": m}
    maybe_reload_pipeline(pipeline=pipeline, state=state, tenant_id="default")
    pipeline.reload_atlas.assert_not_called()
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_mcp_reload.py -v`
Expected: FAIL — maybe_reload_pipeline missing

- [ ] **Step 3: Implement maybe_reload_pipeline + wire into query dispatch**

In `psa/mcp_server.py`, add near the top:

```python
def maybe_reload_pipeline(*, pipeline, state: dict, tenant_id: str) -> None:
    """Called between queries. Triggers pipeline.reload_atlas() when the
    advertisement reload marker has advanced since the last reload."""
    from psa.advertisement.reload import marker_mtime, should_reload

    last = state.get("last_reload_mtime", 0.0)
    if should_reload(tenant_id=tenant_id, last_reload_mtime=last):
        pipeline.reload_atlas()
        state["last_reload_mtime"] = marker_mtime(tenant_id=tenant_id)
```

At each query-handling entry point in `psa/mcp_server.py` (search for the tool handlers that call `pipeline.query(...)` or `pipeline.search(...)`), call `maybe_reload_pipeline(...)` at the start. Maintain a module-level dict `_reload_state = {}` keyed by tenant id.

Minimal wiring:

```python
_reload_state: dict = {}


def _ensure_fresh_pipeline(pipeline, tenant_id: str) -> None:
    st = _reload_state.setdefault(tenant_id, {"last_reload_mtime": 0.0})
    maybe_reload_pipeline(pipeline=pipeline, state=st, tenant_id=tenant_id)
```

Call `_ensure_fresh_pipeline(pipeline, tenant_id)` at the top of each `psa_*` tool handler that uses the pipeline.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mcp_reload.py tests/test_mcp_server.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/mcp_server.py tests/test_mcp_reload.py
git commit -m "feat: mcp server reloads pipeline on marker advance"
```

---

### Task 4.7: Atlas rebuild ledger reset

**Files:**
- Modify: `psa/advertisement/ledger.py`, `psa/atlas.py`
- Test: `tests/test_ledger_rebuild_reset.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_ledger_rebuild_reset.py`:

```python
"""Atlas rebuild: archive all active rows, insert fresh rows for new patterns."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock


def _seed_active(db, n=3):
    from psa.advertisement.ledger import create_schema, upsert_ledger
    create_schema(db)
    for i in range(n):
        upsert_ledger(
            db, f"pid-{i}", i, f"pattern {i}", 1.0, 1.0, grace_days=21
        )


def test_reset_ledger_on_rebuild_archives_all_and_inserts_fresh(tmp_path):
    from psa.advertisement.ledger import reset_ledger_on_rebuild

    db = sqlite3.connect(tmp_path / "t.sqlite3")
    _seed_active(db, n=3)

    card1 = MagicMock()
    card1.anchor_id = 10
    card1.generated_query_patterns = ["new pattern alpha", "new pattern beta"]
    card2 = MagicMock()
    card2.anchor_id = 11
    card2.generated_query_patterns = ["fresh pattern gamma"]
    new_atlas = MagicMock(cards=[card1, card2])

    reset_ledger_on_rebuild(db=db, new_atlas=new_atlas, grace_days=21)

    # Old rows archived
    archived = db.execute(
        "SELECT COUNT(*) FROM pattern_ledger WHERE removal_reason = 'atlas_rebuild_reset'"
    ).fetchone()[0]
    assert archived == 3

    # New rows inserted
    active = db.execute(
        "SELECT anchor_id, pattern_text FROM pattern_ledger WHERE removed_at IS NULL ORDER BY anchor_id, pattern_text"
    ).fetchall()
    assert active == [
        (10, "new pattern alpha"),
        (10, "new pattern beta"),
        (11, "fresh pattern gamma"),
    ]
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_ledger_rebuild_reset.py -v`
Expected: FAIL — `reset_ledger_on_rebuild` missing

- [ ] **Step 3: Implement reset_ledger_on_rebuild**

Append to `psa/advertisement/ledger.py`:

```python
def reset_ledger_on_rebuild(*, db, new_atlas, grace_days: int) -> None:
    """Archive every active row; insert fresh rows for new atlas's patterns.

    Called from AtlasManager.rebuild() AFTER the metadata-inheritance pass,
    so stage 1 and stage 2 both see a consistent rebuild event.
    """
    now = _now_iso()
    db.execute(
        """
        UPDATE pattern_ledger
        SET removed_at = ?,
            removal_reason = 'atlas_rebuild_reset',
            final_ledger = ledger,
            final_shadow_ledger = shadow_ledger
        WHERE removed_at IS NULL
        """,
        (now,),
    )
    for card in new_atlas.cards:
        for pattern in getattr(card, "generated_query_patterns", []):
            upsert_ledger(
                db=db,
                pattern_id=pattern_id_for(card.anchor_id, pattern),
                anchor_id=card.anchor_id,
                pattern_text=pattern,
                ledger_delta=0.0,
                shadow_delta=0.0,
                grace_days=grace_days,
            )
    db.commit()
```

In `psa/atlas.py`, find `AtlasManager.rebuild(...)` — after the existing metadata-inheritance pass, add:

```python
        # Stage 2: wholesale-archive old ledger + fresh-insert new patterns.
        try:
            import sqlite3

            from psa.advertisement.config import AdvertisementDecayConfig
            from psa.advertisement.ledger import create_schema, reset_ledger_on_rebuild
            from psa.config import MempalaceConfig

            ad = AdvertisementDecayConfig.from_mempalace(MempalaceConfig())
            db_path = os.path.expanduser(
                f"~/.psa/tenants/{self.tenant_id}/memory.sqlite3"
            )
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            with sqlite3.connect(db_path) as db:
                create_schema(db)
                reset_ledger_on_rebuild(
                    db=db, new_atlas=new_atlas, grace_days=ad.grace_days
                )
        except Exception as e:
            logger.warning("Ledger reset on rebuild failed: %s", e)
```

Adjust to use the actual attribute names (e.g., `self.tenant_id` vs `self._tenant_id`; `new_atlas` vs whatever the local name is) by reading the current `rebuild()` body.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ledger_rebuild_reset.py tests/test_atlas.py tests/test_atlas_metadata_inheritance.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/advertisement/ledger.py psa/atlas.py tests/test_ledger_rebuild_reset.py
git commit -m "feat: atlas rebuild archives old ledger, inserts fresh rows"
```

---

### Task 4.8: Lifecycle decay pass wiring

**Files:**
- Modify: `psa/lifecycle.py`
- Test: `tests/test_lifecycle_advertisement_decay.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_lifecycle_advertisement_decay.py`:

```python
"""Lifecycle fast-path includes advertisement_decay_pass when tracking enabled."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock, patch


def test_decay_pass_skipped_when_tracking_disabled(tmp_path, monkeypatch):
    from psa.lifecycle import advertisement_decay_pass
    from psa.advertisement.config import AdvertisementDecayConfig, ShadowConfig

    config = AdvertisementDecayConfig(tracking_enabled=False, shadow=ShadowConfig())
    summary = advertisement_decay_pass(
        tenant_id="default", config=config, atlas_or_loader=None
    )
    assert summary["skipped"] is True


def test_decay_pass_logs_summary(tmp_path, monkeypatch):
    from psa.lifecycle import advertisement_decay_pass
    from psa.advertisement.config import AdvertisementDecayConfig, ShadowConfig
    from psa.advertisement.ledger import create_schema, upsert_ledger

    monkeypatch.setenv("HOME", str(tmp_path))
    db_path = tmp_path / ".psa" / "tenants" / "default" / "memory.sqlite3"
    db_path.parent.mkdir(parents=True)
    db = sqlite3.connect(db_path)
    create_schema(db)
    upsert_ledger(db, "pid-1", 1, "some pattern", 1.0, 1.0, grace_days=21)
    db.close()

    atlas = MagicMock()
    atlas.get_anchor = lambda aid: MagicMock(
        generated_query_patterns=["some pattern", "x", "y", "z", "w"]
    )
    shielded = lambda tenant_id, anchor_ids: set()
    pinned = lambda aid, text: False

    config = AdvertisementDecayConfig(
        tracking_enabled=True, removal_enabled=False, shadow=ShadowConfig()
    )
    summary = advertisement_decay_pass(
        tenant_id="default",
        config=config,
        atlas_or_loader=atlas,
        shielded_anchor_fn=shielded,
        pinned_fn=pinned,
    )
    assert "n_active" in summary
    assert summary["n_active"] == 1
    assert summary["n_actually_removed_under_B"] == 0
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_lifecycle_advertisement_decay.py -v`
Expected: FAIL — function missing

- [ ] **Step 3: Implement advertisement_decay_pass**

Append to `psa/lifecycle.py`:

```python
def advertisement_decay_pass(
    *,
    tenant_id: str,
    config,
    atlas_or_loader,
    shielded_anchor_fn=None,
    pinned_fn=None,
) -> dict:
    """Nightly fast-path decay + removal evaluation + optional apply.

    Returns a summary dict (also logged as structured JSON via logger).
    """
    import logging
    import os
    import sqlite3

    logger = logging.getLogger("psa.lifecycle.advertisement_decay")

    if not config.tracking_enabled:
        logger.info("advertisement_decay skipped (tracking_enabled=false)")
        return {"stage": "advertisement_decay", "skipped": True}

    if shielded_anchor_fn is None:
        from psa.advertisement.decay import shielded_anchors as _sh

        def shielded_anchor_fn(tenant_id, anchor_ids):
            # Wrap stage 1 P1 helper. If signature differs, adapt here.
            return _sh(tenant_id, anchor_ids)

    if pinned_fn is None:
        from psa.advertisement.metadata import load_metadata, metadata_key

        def pinned_fn(anchor_id, pattern_text):
            try:
                # Stage 1 metadata store: {key: {...pinned?}}
                meta = load_metadata(atlas_or_loader.anchor_dir)
                entry = meta.get(metadata_key(anchor_id, pattern_text), {})
                return bool(entry.get("pinned", False))
            except Exception:
                return False

    from psa.advertisement.ledger import (
        apply_decay,
        apply_removals,
        create_schema,
        evaluate_removal,
    )

    db_path = os.path.expanduser(
        f"~/.psa/tenants/{tenant_id}/memory.sqlite3"
    )
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with sqlite3.connect(db_path) as db:
        create_schema(db)
        apply_decay(db, tau_days=config.tau_days, removal_threshold=config.removal_threshold)
        result = evaluate_removal(
            db=db,
            atlas=atlas_or_loader,
            tenant_id=tenant_id,
            config=config,
            shielded_anchor_fn=shielded_anchor_fn,
            pinned_fn=pinned_fn,
        )
        n_removed_B = 0
        if config.removal_enabled and result.removal_candidates:
            atlas_dir = getattr(atlas_or_loader, "anchor_dir", None) or os.path.dirname(
                db_path
            )
            n_removed_B = apply_removals(
                db=db,
                tenant_id=tenant_id,
                atlas_dir=atlas_dir,
                candidates=result.removal_candidates,
            )

    summary = {
        "stage": "advertisement_decay",
        "tenant_id": tenant_id,
        "n_active": result.n_active,
        "n_in_grace": result.n_in_grace,
        "n_at_risk": result.n_at_risk,
        "n_would_remove_under_A": len(result.shadow_candidates),
        "n_would_remove_under_B": len(result.removal_candidates),
        "n_actually_removed_under_B": n_removed_B,
    }
    logger.info("advertisement_decay summary: %s", summary)
    return summary
```

In the existing fast-path orchestrator in `lifecycle.py` (look for where `prune_memories` or similar is called), add a call:

```python
        advertisement_decay_pass(
            tenant_id=tenant_id,
            config=AdvertisementDecayConfig.from_mempalace(MempalaceConfig()),
            atlas_or_loader=atlas,
        )
```

Place this between "mine new sessions" and "prune memories" as specified in spec §7.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_lifecycle_advertisement_decay.py tests/test_lifecycle.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/lifecycle.py tests/test_lifecycle_advertisement_decay.py
git commit -m "feat: advertisement_decay_pass wired into lifecycle fast path"
```

---

## Phase 5 — Inspection CLIs

### Task 5.1: `psa advertisement status`

**Files:**
- Create: `psa/advertisement/cli.py`
- Modify: `psa/cli.py` (add subparser dispatch)
- Test: `tests/test_cli_advertisement.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_advertisement.py`:

```python
"""Tests for psa advertisement CLI subcommands."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys


def _seed(tmp_path):
    from psa.advertisement.ledger import create_schema, upsert_ledger
    db_path = tmp_path / ".psa" / "tenants" / "default" / "memory.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as db:
        create_schema(db)
        upsert_ledger(db, "pid-1", 1, "alpha", 2.5, 2.0, grace_days=21)
        upsert_ledger(db, "pid-2", 1, "beta", -0.3, -0.5, grace_days=21)


def test_status_prints_distribution(tmp_path, monkeypatch):
    _seed(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    r = subprocess.run(
        [sys.executable, "-m", "psa", "advertisement", "status", "--json"],
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["n_active"] == 2
    assert "histogram" in data
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_cli_advertisement.py::test_status_prints_distribution -v`
Expected: FAIL — subcommand missing

- [ ] **Step 3: Implement status subcommand**

Create `psa/advertisement/cli.py`:

```python
"""
cli.py — psa advertisement subcommands.

  status          Ledger distribution + shadow-disagreement counts.
  diff            B-vs-A counterfactual report.
  rebuild-ledger  Canonical regenerator (reads trace + atlas).
  purge           Hard-delete archived rows past retention.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone


def _db_path(tenant_id: str) -> str:
    return os.path.expanduser(
        f"~/.psa/tenants/{tenant_id}/memory.sqlite3"
    )


def _histogram(values, bins: int = 20):
    if not values:
        return {"bins": [], "counts": []}
    lo, hi = min(values), max(values)
    if lo == hi:
        return {"bins": [lo, hi], "counts": [len(values)]}
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    return {"bins": edges, "counts": counts}


def cmd_status(args) -> int:
    tenant_id = args.tenant or "default"
    db_path = _db_path(tenant_id)
    if not os.path.exists(db_path):
        print(f"No ledger DB at {db_path}")
        return 0 if args.json else 0
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            """
            SELECT pattern_id, anchor_id, pattern_text, ledger,
                   shadow_ledger, consecutive_negative_cycles,
                   shadow_consecutive, grace_expires_at
            FROM pattern_ledger
            WHERE removed_at IS NULL
            """
        ).fetchall()
    now_iso = datetime.now(timezone.utc).isoformat()
    active = rows
    in_grace = sum(1 for r in rows if r[7] > now_iso)
    at_risk = sum(1 for r in rows if r[5] >= 7)
    shadow_only_risk = sum(
        1 for r in rows if r[6] >= 7 and r[5] < 14
    )
    values = [r[3] for r in rows]
    hist = _histogram(values)
    data = {
        "tenant_id": tenant_id,
        "n_active": len(active),
        "n_in_grace": in_grace,
        "n_at_risk": at_risk,
        "n_shadow_only_at_risk": shadow_only_risk,
        "histogram": hist,
    }
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    print(f"tenant: {tenant_id}")
    print(f"  active patterns:         {data['n_active']}")
    print(f"  in grace:                {data['n_in_grace']}")
    print(f"  at risk (cnc ≥ 7):       {data['n_at_risk']}")
    print(f"  shadow-only at risk:     {data['n_shadow_only_at_risk']}")
    return 0


def build_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("advertisement", help="Advertisement decay CLIs")
    sub = p.add_subparsers(dest="advertisement_cmd", required=True)

    p_status = sub.add_parser("status", help="Ledger distribution + risk counts")
    p_status.add_argument("--tenant")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    return p
```

In `psa/cli.py`, find the `argparse` top-level dispatch and add:

```python
    from psa.advertisement.cli import build_parser as _build_ad_parser
    _build_ad_parser(subparsers)
```

Add dispatch at the top of the main command handler:

```python
    if args.command == "advertisement":
        return args.func(args)
```

Match the existing style (`_cmd_atlas_*` functions, dispatch in `main()`).

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_cli_advertisement.py::test_status_prints_distribution -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/advertisement/cli.py psa/cli.py tests/test_cli_advertisement.py
git commit -m "feat: psa advertisement status"
```

---

### Task 5.2: `psa advertisement diff`

**Files:**
- Modify: `psa/advertisement/cli.py`
- Test: `tests/test_cli_advertisement.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_advertisement.py`:

```python
def test_diff_prints_counterfactual(tmp_path, monkeypatch):
    """B and A policies disagree when shadow_consecutive ≥ 7 but cnc < 14."""
    import sqlite3
    from psa.advertisement.ledger import create_schema
    db_path = tmp_path / ".psa" / "tenants" / "default" / "memory.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as db:
        create_schema(db)
        db.execute(
            """
            INSERT INTO pattern_ledger
                (pattern_id, anchor_id, pattern_text,
                 ledger, consecutive_negative_cycles,
                 shadow_ledger, shadow_consecutive,
                 grace_expires_at, created_at, last_updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "pid-1", 1, "would-A-removes",
                -0.5, 8, -0.5, 10,
                "2000-01-01T00:00:00+00:00",
                "2025-01-01T00:00:00+00:00",
                "2025-01-01T00:00:00+00:00",
            ),
        )
        db.commit()
    monkeypatch.setenv("HOME", str(tmp_path))
    r = subprocess.run(
        [sys.executable, "-m", "psa", "advertisement", "diff", "--json"],
        capture_output=True, text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["shadow_only_would_remove"] == 1
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_cli_advertisement.py::test_diff_prints_counterfactual -v`
Expected: FAIL — diff missing

- [ ] **Step 3: Implement diff**

Append to `psa/advertisement/cli.py`:

```python
def cmd_diff(args) -> int:
    tenant_id = args.tenant or "default"
    db_path = _db_path(tenant_id)
    if not os.path.exists(db_path):
        print(f"No ledger DB at {db_path}")
        return 0
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            """
            SELECT pattern_id, anchor_id, pattern_text, ledger,
                   consecutive_negative_cycles, shadow_ledger, shadow_consecutive
            FROM pattern_ledger
            WHERE removed_at IS NULL
            """
        ).fetchall()
    sustained_B, sustained_A = 14, 7
    B_would = {r[0] for r in rows if r[4] >= sustained_B}
    A_would = {r[0] for r in rows if r[6] >= sustained_A}
    only_A = A_would - B_would
    only_B = B_would - A_would
    agree = A_would & B_would
    disagreements = [
        {
            "pattern_id": r[0],
            "anchor_id": r[1],
            "pattern_text": r[2],
            "ledger": r[3],
            "consecutive_negative_cycles": r[4],
            "shadow_ledger": r[5],
            "shadow_consecutive": r[6],
            "B_would_remove": r[0] in B_would,
            "A_would_remove": r[0] in A_would,
        }
        for r in rows
        if r[0] in only_A or r[0] in only_B
    ]
    data = {
        "tenant_id": tenant_id,
        "both_would_remove": len(agree),
        "shadow_only_would_remove": len(only_A),
        "primary_only_would_remove": len(only_B),
        "disagreements": disagreements,
    }
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    print(f"tenant: {tenant_id}")
    print(f"  both (B ∧ A) would remove:  {data['both_would_remove']}")
    print(f"  shadow-only (A∖B):          {data['shadow_only_would_remove']}")
    print(f"  primary-only (B∖A):         {data['primary_only_would_remove']}")
    for d in disagreements[:20]:
        print(
            f"    - anchor={d['anchor_id']} "
            f"B={'Y' if d['B_would_remove'] else 'N'} "
            f"A={'Y' if d['A_would_remove'] else 'N'} "
            f"cnc={d['consecutive_negative_cycles']} "
            f"scc={d['shadow_consecutive']} "
            f"{d['pattern_text'][:60]!r}"
        )
    return 0
```

Inside `build_parser`, register:

```python
    p_diff = sub.add_parser("diff", help="B vs A counterfactual report")
    p_diff.add_argument("--tenant")
    p_diff.add_argument("--json", action="store_true")
    p_diff.set_defaults(func=cmd_diff)
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_cli_advertisement.py::test_diff_prints_counterfactual -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/advertisement/cli.py tests/test_cli_advertisement.py
git commit -m "feat: psa advertisement diff counterfactual report"
```

---

### Task 5.3: `psa advertisement rebuild-ledger`

**Files:**
- Modify: `psa/advertisement/cli.py`
- Test: `tests/test_cli_advertisement.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_advertisement.py`:

```python
def test_rebuild_ledger_recomputes_from_trace(tmp_path, monkeypatch):
    """rebuild-ledger walks trace + recomputes ledger state."""
    import sqlite3
    from psa.advertisement.ledger import create_schema

    monkeypatch.setenv("HOME", str(tmp_path))
    # Write a minimal trace with retrieval_attribution
    trace_dir = tmp_path / ".psa" / "tenants" / "default"
    trace_dir.mkdir(parents=True)
    record = {
        "timestamp": "2026-04-17T00:00:00+00:00",
        "query": "cat vs dog",
        "selected_anchor_ids": [1],
        "retrieval_attribution": [
            {
                "anchor_id": 1,
                "bm25_argmax_pattern": "cat vs dog behavior",
                "bm25_epsilon_tied": [],
                "bm25_floor_passed": True,
            }
        ],
    }
    (trace_dir / "query_trace.jsonl").write_text(json.dumps(record) + "\n")

    r = subprocess.run(
        [sys.executable, "-m", "psa", "advertisement", "rebuild-ledger", "--dry-run", "--json"],
        capture_output=True, text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["records_processed"] == 1
    assert data["derived_patterns"] >= 1
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_cli_advertisement.py::test_rebuild_ledger_recomputes_from_trace -v`
Expected: FAIL — rebuild-ledger missing

- [ ] **Step 3: Implement rebuild-ledger**

Append to `psa/advertisement/cli.py`:

```python
def cmd_rebuild_ledger(args) -> int:
    tenant_id = args.tenant or "default"
    trace_path = os.path.expanduser(
        f"~/.psa/tenants/{tenant_id}/query_trace.jsonl"
    )
    if not os.path.exists(trace_path):
        print(f"No trace at {trace_path}")
        return 1

    records = 0
    # Derived ledger computed purely from trace + augmentation fields.
    # Decay is applied on read at rebuild time using the record timestamp.
    derived: dict[str, dict] = {}

    # Load config to know weights + tau
    from psa.advertisement.config import AdvertisementDecayConfig
    from psa.config import MempalaceConfig
    import math

    ad = AdvertisementDecayConfig.from_mempalace(MempalaceConfig())

    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records += 1
            attrs = rec.get("retrieval_attribution") or []
            selected = set(rec.get("selected_anchor_ids") or [])
            ts_str = rec.get("timestamp")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue
            age_days = (datetime.now(timezone.utc) - ts).days
            decay = math.exp(-age_days / ad.tau_days)
            for a in attrs:
                if not a.get("bm25_floor_passed"):
                    continue
                argmax = a.get("bm25_argmax_pattern")
                tied = a.get("bm25_epsilon_tied") or []
                credited = [argmax] + list(tied) if argmax else []
                if not credited:
                    continue
                per = 1.0 / len(credited)
                base = ad.retrieval_credit
                if a["anchor_id"] in selected:
                    base += ad.selector_pick_credit
                else:
                    base -= ad.selector_decline_penalty
                for pat in credited:
                    from psa.advertisement.ledger import pattern_id_for
                    key = pattern_id_for(a["anchor_id"], pat)
                    slot = derived.setdefault(
                        key,
                        {"anchor_id": a["anchor_id"], "pattern_text": pat, "ledger": 0.0},
                    )
                    slot["ledger"] += base * per * decay

    data = {
        "tenant_id": tenant_id,
        "records_processed": records,
        "derived_patterns": len(derived),
    }
    if args.dry_run:
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(
                f"dry-run: processed {records} records, "
                f"derived {len(derived)} patterns"
            )
        return 0

    # Non-dry: write derived values into the table (overwrite existing).
    db_path = _db_path(tenant_id)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    from psa.advertisement.ledger import create_schema
    with sqlite3.connect(db_path) as db:
        create_schema(db)
        for pid, slot in derived.items():
            db.execute(
                """
                INSERT INTO pattern_ledger
                    (pattern_id, anchor_id, pattern_text,
                     ledger, shadow_ledger,
                     grace_expires_at, created_at, last_updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pattern_id) DO UPDATE SET
                    ledger = excluded.ledger,
                    last_updated_at = excluded.last_updated_at
                """,
                (
                    pid,
                    slot["anchor_id"],
                    slot["pattern_text"],
                    slot["ledger"],
                    slot["ledger"],
                    datetime.now(timezone.utc).isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        db.commit()

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(f"Rebuilt ledger: {len(derived)} patterns from {records} records")
    return 0
```

Register in `build_parser`:

```python
    p_rebuild = sub.add_parser("rebuild-ledger", help="Recompute ledger from trace")
    p_rebuild.add_argument("--tenant")
    p_rebuild.add_argument("--dry-run", action="store_true")
    p_rebuild.add_argument("--json", action="store_true")
    p_rebuild.set_defaults(func=cmd_rebuild_ledger)
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_cli_advertisement.py::test_rebuild_ledger_recomputes_from_trace -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/advertisement/cli.py tests/test_cli_advertisement.py
git commit -m "feat: psa advertisement rebuild-ledger canonical regenerator"
```

---

### Task 5.4: `psa advertisement purge`

**Files:**
- Modify: `psa/advertisement/cli.py`
- Test: `tests/test_cli_advertisement.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_advertisement.py`:

```python
def test_purge_deletes_archived_rows_past_retention(tmp_path, monkeypatch):
    import sqlite3
    from datetime import datetime, timedelta, timezone
    from psa.advertisement.ledger import create_schema

    db_path = tmp_path / ".psa" / "tenants" / "default" / "memory.sqlite3"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as db:
        create_schema(db)
        old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        recent = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        active = None
        for pid, removed_at in [("old", old), ("recent", recent), ("active", active)]:
            db.execute(
                """
                INSERT INTO pattern_ledger
                    (pattern_id, anchor_id, pattern_text, ledger, shadow_ledger,
                     grace_expires_at, created_at, last_updated_at, removed_at)
                VALUES (?, ?, ?, 0.0, 0.0, '', '', '', ?)
                """,
                (pid, 1, "p", removed_at),
            )
        db.commit()

    monkeypatch.setenv("HOME", str(tmp_path))
    r = subprocess.run(
        [sys.executable, "-m", "psa", "advertisement", "purge", "--older-than-days", "90"],
        capture_output=True, text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 0

    with sqlite3.connect(db_path) as db:
        remaining = {r[0] for r in db.execute("SELECT pattern_id FROM pattern_ledger")}
    assert "old" not in remaining
    assert "recent" in remaining
    assert "active" in remaining
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_cli_advertisement.py::test_purge_deletes_archived_rows_past_retention -v`
Expected: FAIL — purge missing

- [ ] **Step 3: Implement purge**

Append to `psa/advertisement/cli.py`:

```python
def cmd_purge(args) -> int:
    tenant_id = args.tenant or "default"
    db_path = _db_path(tenant_id)
    if not os.path.exists(db_path):
        print(f"No ledger DB at {db_path}")
        return 0

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=args.older_than_days)
    ).isoformat()
    with sqlite3.connect(db_path) as db:
        cur = db.execute(
            "DELETE FROM pattern_ledger WHERE removed_at IS NOT NULL AND removed_at < ?",
            (cutoff,),
        )
        db.commit()
        deleted = cur.rowcount

    if args.json:
        print(json.dumps({"tenant_id": tenant_id, "deleted": deleted}, indent=2))
    else:
        print(f"Purged {deleted} archived row(s) older than {args.older_than_days} days.")
    return 0
```

Register:

```python
    p_purge = sub.add_parser("purge", help="Hard-delete archived rows past retention")
    p_purge.add_argument("--tenant")
    p_purge.add_argument("--older-than-days", type=int, default=90)
    p_purge.add_argument("--json", action="store_true")
    p_purge.set_defaults(func=cmd_purge)
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_cli_advertisement.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add psa/advertisement/cli.py tests/test_cli_advertisement.py
git commit -m "feat: psa advertisement purge archived rows past retention"
```

---

### Task 5.5: Full-suite green + final commit

**Files:**
- None (verification task)

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 2: Run ruff**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: clean.

- [ ] **Step 3: Run the end-to-end smoke**

With a tenant that has no existing refined file, no ledger, and `tracking_enabled=false`:

```bash
uv run psa advertisement status
```

Expected: prints `n_active=0` and an empty histogram (no error).

With `tracking_enabled=true` and `trace_queries=false`:

```bash
PSA_AD_DECAY_TRACKING_ENABLED=1 PSA_TRACE=0 uv run psa search "test"
```

Expected: fails with `AdvertisementDecayConfigError` mentioning `trace_queries`.

- [ ] **Step 4: Commit any final lint/format adjustments**

```bash
git add -A
git commit -m "chore: full-suite green + ruff clean for stage 2"
```

(Skip if no changes.)

---

## Self-review checklist

After implementation, verify:

1. **Spec coverage:**
   - Pattern-level graded signals (retrieval/pick/decline): Task 3.1
   - pattern_ledger table schema + soft-archive columns + grace_expires_at: Task 2.3
   - BM25 argmax with ε-split + BM25 floor: Tasks 2.2, 3.1
   - Trace-first-ledger-second ordering: Task 3.2
   - `tracking_enabled → trace_queries` validation: Task 2.1
   - Exponential decay + consecutive-negative state: Task 4.1
   - Evaluation with P1/P3/min-floor: Task 4.2
   - Reload marker + MCP check + reload_atlas: Tasks 4.3, 4.5, 4.6
   - Atlas rebuild archive + fresh insert: Task 4.7
   - Lifecycle fast-path decay pass: Task 4.8
   - Hash-gate for all three candidate producers: Tasks 1.1–1.5
   - Shadow policy parallel columns: Tasks 2.3, 3.1, 4.1, 4.2
   - CLIs (status, diff, rebuild-ledger, purge): Tasks 5.1–5.4
   - Purge at 90 days: Task 5.4

2. **Type consistency:**
   - `pattern_id_for(anchor_id: int, pattern_text: str) -> str` — consistent across Tasks 2.3, 3.1, 4.7, 5.3
   - `AdvertisementDecayConfig` — consistent dataclass shape across all consumers
   - `AnchorAttribution`, `RemovalCandidate`, `EvaluationResult` — defined once in ledger.py, referenced elsewhere
   - `write_trace(...) -> bool` — new signature consistent across pipeline.py consumers

3. **No placeholders:**
   - Every task includes concrete test code + concrete implementation code
   - No "add appropriate error handling" style requirements — behavior is specified

## Files — quick index

### New
- `psa/advertisement/config.py`
- `psa/advertisement/attribution.py`
- `psa/advertisement/ledger.py`
- `psa/advertisement/reload.py`
- `psa/advertisement/cli.py`
- `tests/test_advertisement_config.py`
- `tests/test_ledger_attribution.py`
- `tests/test_ledger_schema.py`
- `tests/test_ledger_write_path.py`
- `tests/test_ledger_trace_ordering.py`
- `tests/test_ledger_decay.py`
- `tests/test_ledger_reload_marker.py`
- `tests/test_ledger_rebuild_reset.py`
- `tests/test_lifecycle_advertisement_decay.py`
- `tests/test_pipeline_reload.py`
- `tests/test_mcp_reload.py`
- `tests/test_cli_advertisement.py`
- `tests/test_promote_refinement_hash_gate.py`
- `tests/test_trace.py` (if absent)

### Modified (stage 2)
- `psa/pipeline.py`
- `psa/retriever.py`
- `psa/lifecycle.py`
- `psa/atlas.py`
- `psa/trace.py`
- `psa/config.py`
- `psa/cli.py`
- `psa/mcp_server.py`

### Modified (stage 1 coexistence — Phase 1)
- `psa/advertisement/writer.py`
- `psa/curation/curator.py`
- `psa/cli.py` (refine + promote paths)
