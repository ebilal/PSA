# Phase 3C — Refined-Atlas Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `AnchorIndex.load()` prefer `anchor_cards_refined.json` when present, and add a `psa atlas refine` CLI wrapper that drives `scripts/refine_anchor_cards.py` against the current atlas.

**Architecture:** Two surgical changes. The loader change makes training data generation and inference both see refined cards automatically — no override plumbing needed. The CLI wrapper resolves the atlas path for the tenant, invokes the existing `refine_cards()` function from the script (loaded via `importlib.util` the same way the existing tests load it), and writes the output back into the atlas directory.

**Tech Stack:** Python 3.13, argparse, `importlib.util`, existing `psa.atlas.AtlasManager` and `psa.tenant.TenantManager`.

**Design spec:** `docs/superpowers/specs/2026-04-16-phase3c-refined-atlas-loader-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `psa/anchor.py` | Modify | `AnchorIndex.load()` prefers `anchor_cards_refined.json` when it exists in the directory; falls back to `anchor_cards.json`. |
| `tests/test_anchor.py` | Modify | New tests for the loader preference + fallback. |
| `psa/cli.py` | Modify | Add `_cmd_atlas_refine()` and register `refine` subcommand under `psa atlas`. |
| `tests/test_cli_atlas_refine.py` | Create | Smoke test for argparse wiring + end-to-end refinement against a fixture. |

---

### Task 1: `AnchorIndex.load()` prefers refined cards file

**Files:**
- Modify: `psa/anchor.py:226-232`
- Modify: `tests/test_anchor.py`

- [ ] **Step 1: Write the failing test for the refined-cards preference**

Add to `tests/test_anchor.py`:

```python
def test_anchor_index_load_prefers_refined_cards(tmp_path):
    """When both files exist, AnchorIndex.load() loads anchor_cards_refined.json."""
    import json
    import numpy as np
    from psa.anchor import AnchorIndex

    raw = [
        {
            "anchor_id": 1, "name": "raw-name", "meaning": "raw meaning",
            "memory_types": ["semantic"], "include_terms": [], "exclude_terms": [],
            "prototype_examples": [], "near_but_different": [], "centroid": [0.0] * 768,
            "memory_count": 1, "is_novelty": False, "status": "active", "metadata": {},
            "generated_query_patterns": ["raw pattern"],
            "query_fingerprint": [],
        }
    ]
    refined = [
        {
            "anchor_id": 1, "name": "raw-name", "meaning": "raw meaning",
            "memory_types": ["semantic"], "include_terms": [], "exclude_terms": [],
            "prototype_examples": [], "near_but_different": [], "centroid": [0.0] * 768,
            "memory_count": 1, "is_novelty": False, "status": "active", "metadata": {},
            "generated_query_patterns": ["raw pattern", "refined pattern"],
            "query_fingerprint": [],
        }
    ]
    (tmp_path / "anchor_cards.json").write_text(json.dumps(raw))
    (tmp_path / "anchor_cards_refined.json").write_text(json.dumps(refined))
    np.save(tmp_path / "centroids.npy", np.zeros((1, 768), dtype=np.float32))

    idx = AnchorIndex.load(str(tmp_path))

    assert len(idx._cards) == 1
    assert "refined pattern" in idx._cards[0].generated_query_patterns


def test_anchor_index_load_falls_back_to_raw_when_no_refined(tmp_path):
    """When only anchor_cards.json exists, it is loaded as before."""
    import json
    import numpy as np
    from psa.anchor import AnchorIndex

    raw = [
        {
            "anchor_id": 1, "name": "raw-name", "meaning": "raw meaning",
            "memory_types": ["semantic"], "include_terms": [], "exclude_terms": [],
            "prototype_examples": [], "near_but_different": [], "centroid": [0.0] * 768,
            "memory_count": 1, "is_novelty": False, "status": "active", "metadata": {},
            "generated_query_patterns": ["raw pattern"],
            "query_fingerprint": [],
        }
    ]
    (tmp_path / "anchor_cards.json").write_text(json.dumps(raw))
    np.save(tmp_path / "centroids.npy", np.zeros((1, 768), dtype=np.float32))

    idx = AnchorIndex.load(str(tmp_path))

    assert len(idx._cards) == 1
    assert idx._cards[0].generated_query_patterns == ["raw pattern"]


def test_anchor_index_load_raises_when_neither_file_exists(tmp_path):
    """If neither raw nor refined cards file exists, raise FileNotFoundError."""
    import numpy as np
    from psa.anchor import AnchorIndex

    np.save(tmp_path / "centroids.npy", np.zeros((1, 768), dtype=np.float32))

    with pytest.raises(FileNotFoundError):
        AnchorIndex.load(str(tmp_path))
```

Note: `pytest` is already imported in `tests/test_anchor.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
uv run pytest tests/test_anchor.py::test_anchor_index_load_prefers_refined_cards tests/test_anchor.py::test_anchor_index_load_falls_back_to_raw_when_no_refined tests/test_anchor.py::test_anchor_index_load_raises_when_neither_file_exists -v
```

Expected: first test FAILS (refined file ignored; "refined pattern" missing); the other two should PASS since current behavior already covers them. That's OK — still commit them to lock behavior.

- [ ] **Step 3: Modify `AnchorIndex.load()` in `psa/anchor.py`**

Replace the block at `psa/anchor.py:228-232`:

```python
        cards_path = os.path.join(path, "anchor_cards.json")
        centroids_path = os.path.join(path, "centroids.npy")

        if not os.path.exists(cards_path):
            raise FileNotFoundError(f"No anchor_cards.json at {path}")
```

With:

```python
        refined_path = os.path.join(path, "anchor_cards_refined.json")
        raw_path = os.path.join(path, "anchor_cards.json")
        centroids_path = os.path.join(path, "centroids.npy")

        if os.path.exists(refined_path):
            cards_path = refined_path
        elif os.path.exists(raw_path):
            cards_path = raw_path
        else:
            raise FileNotFoundError(
                f"No anchor_cards.json or anchor_cards_refined.json at {path}"
            )
```

- [ ] **Step 4: Run the three new tests**

Run:
```
uv run pytest tests/test_anchor.py::test_anchor_index_load_prefers_refined_cards tests/test_anchor.py::test_anchor_index_load_falls_back_to_raw_when_no_refined tests/test_anchor.py::test_anchor_index_load_raises_when_neither_file_exists -v
```

Expected: all three PASS.

- [ ] **Step 5: Run the full `test_anchor.py` to confirm no regressions**

Run:
```
uv run pytest tests/test_anchor.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add psa/anchor.py tests/test_anchor.py
git commit -m "$(cat <<'EOF'
feat: AnchorIndex.load() prefers anchor_cards_refined.json

When a refined cards file sits in the atlas directory, prefer it
over anchor_cards.json. Lets the selector see refined card text
at both training data generation and inference time without any
override plumbing in FullAtlasScorer or the pipeline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `psa atlas refine` CLI command

**Files:**
- Modify: `psa/cli.py`
- Create: `tests/test_cli_atlas_refine.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_atlas_refine.py`:

```python
"""Tests for `psa atlas refine` CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def _write_atlas(atlas_dir: Path, patterns: list[str]) -> None:
    """Write a minimal, valid atlas_vN directory."""
    atlas_dir.mkdir(parents=True, exist_ok=True)
    cards = [
        {
            "anchor_id": 1,
            "name": "anchor-1",
            "meaning": "test anchor",
            "memory_types": ["semantic"],
            "include_terms": [],
            "exclude_terms": [],
            "prototype_examples": [],
            "near_but_different": [],
            "centroid": [0.0] * 768,
            "memory_count": 1,
            "is_novelty": False,
            "status": "active",
            "metadata": {},
            "generated_query_patterns": patterns,
            "query_fingerprint": [],
        }
    ]
    (atlas_dir / "anchor_cards.json").write_text(json.dumps(cards))
    np.save(atlas_dir / "centroids.npy", np.zeros((1, 768), dtype=np.float32))
    (atlas_dir / "atlas_meta.json").write_text(json.dumps({
        "version": 1,
        "tenant_id": "test",
        "stats": {
            "n_learned": 1, "n_novelty": 0, "n_total": 1,
            "coverage": 1.0, "novelty_rate": 0.0, "utilization_skew": 0.0,
        },
    }))


def test_atlas_refine_writes_refined_cards_file(tmp_path, monkeypatch, capsys):
    """`psa atlas refine` writes anchor_cards_refined.json into the latest atlas dir."""
    from psa.cli import main

    # Redirect HOME so TenantManager / AtlasManager see our tmp tree
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original pattern"])

    miss_log = tmp_path / "misses.jsonl"
    miss_log.write_text(json.dumps({
        "question_id": "q1",
        "query": "how does the authentication token refresh flow work",
        "gold_anchor_ids": [1],
        "miss_reason": "scoring_rank",
    }) + "\n")

    # Minimal memory.sqlite3 so MemoryStore() init in TenantManager does not fail
    tenant_dir.mkdir(parents=True, exist_ok=True)

    with patch("sys.argv", ["psa", "atlas", "refine", "--miss-log", str(miss_log)]):
        main()

    refined_path = atlas_dir / "anchor_cards_refined.json"
    assert refined_path.exists(), "refined cards file must be written"

    refined_cards = json.loads(refined_path.read_text())
    assert len(refined_cards) == 1
    # Original pattern preserved, at least one new pattern from the miss log added
    patterns = refined_cards[0]["generated_query_patterns"]
    assert "original pattern" in patterns
    assert len(patterns) > 1


def test_atlas_refine_errors_when_no_atlas(tmp_path, monkeypatch, capsys):
    """`psa atlas refine` prints a clear error and exits non-zero when no atlas exists."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    miss_log = tmp_path / "misses.jsonl"
    miss_log.write_text("")

    with patch("sys.argv", ["psa", "atlas", "refine", "--miss-log", str(miss_log)]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code != 0
    out = capsys.readouterr().out
    assert "atlas" in out.lower()


def test_atlas_refine_errors_when_miss_log_missing(tmp_path, monkeypatch, capsys):
    """`psa atlas refine` errors when the provided miss log path does not exist."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original"])

    with patch("sys.argv", ["psa", "atlas", "refine", "--miss-log", str(tmp_path / "does_not_exist.jsonl")]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
uv run pytest tests/test_cli_atlas_refine.py -v
```

Expected: all three FAIL — likely with "argument atlas_action: invalid choice: 'refine'" or equivalent, since the subcommand does not exist yet.

- [ ] **Step 3: Register the `refine` subparser in `psa/cli.py`**

Find the block starting at `psa/cli.py:1436` that adds atlas subcommands:

```python
    atlas_sub = p_atlas.add_subparsers(dest="atlas_action")
    atlas_sub.add_parser("build", help="Build or rebuild the PSA atlas for the tenant")
    atlas_sub.add_parser("status", help="Show atlas version and anchor count")
    atlas_sub.add_parser("health", help="Show health report (novelty rate, utilization skew)")
    atlas_sub.add_parser("rebuild", help="Force rebuild the atlas (same as build)")
```

Add a new subparser after `rebuild`:

```python
    p_atlas_refine = atlas_sub.add_parser(
        "refine",
        help="Refine anchor card generated_query_patterns from a miss log",
    )
    p_atlas_refine.add_argument(
        "--miss-log",
        required=True,
        help="Path to a JSONL miss log produced by benchmark runs",
    )
    p_atlas_refine.add_argument(
        "--max-patterns",
        type=int,
        default=20,
        help="Max total generated_query_patterns per anchor after refinement (default: 20)",
    )
```

- [ ] **Step 4: Add dispatch in `cmd_atlas()`**

Find `cmd_atlas()` at `psa/cli.py:345`:

```python
def cmd_atlas(args):
    """Handle 'psa atlas <subcommand>'."""
    action = getattr(args, "atlas_action", None)
    if not action:
        print("Usage: psa atlas {build,status,health}")
        return

    if action in ("build", "rebuild"):
        _cmd_atlas_build(args)
    elif action == "status":
        _cmd_atlas_status(args)
    elif action == "health":
        _cmd_atlas_health(args)
```

Replace with:

```python
def cmd_atlas(args):
    """Handle 'psa atlas <subcommand>'."""
    action = getattr(args, "atlas_action", None)
    if not action:
        print("Usage: psa atlas {build,status,health,refine}")
        return

    if action in ("build", "rebuild"):
        _cmd_atlas_build(args)
    elif action == "status":
        _cmd_atlas_status(args)
    elif action == "health":
        _cmd_atlas_health(args)
    elif action == "refine":
        _cmd_atlas_refine(args)
```

- [ ] **Step 5: Implement `_cmd_atlas_refine()`**

Add this function at `psa/cli.py` immediately after `_cmd_atlas_health()` (around line 480):

```python
def _cmd_atlas_refine(args):
    """Refine anchor cards for the current atlas using a miss log."""
    import importlib.util
    import json
    import os
    from pathlib import Path

    from .tenant import TenantManager
    from .atlas import AtlasManager

    tenant_id = getattr(args, "tenant", "default")
    miss_log_path = args.miss_log
    max_patterns = getattr(args, "max_patterns", 20)

    if not os.path.exists(miss_log_path):
        print(f"  Error: miss log not found: {miss_log_path}")
        sys.exit(1)

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
    version = mgr.latest_version()
    if version is None:
        print(f"  Error: no atlas for tenant '{tenant_id}'. Run 'psa atlas build' first.")
        sys.exit(1)

    atlas_dir = Path(tenant.root_dir) / f"atlas_v{version}"
    base_cards_path = atlas_dir / "anchor_cards.json"
    output_path = atlas_dir / "anchor_cards_refined.json"

    if not base_cards_path.exists():
        print(f"  Error: {base_cards_path} not found.")
        sys.exit(1)

    # Load the refinement script as a module (it lives outside the psa package)
    script_path = Path(__file__).parent.parent / "scripts" / "refine_anchor_cards.py"
    if not script_path.exists():
        print(f"  Error: refinement script not found at {script_path}.")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("refine_anchor_cards", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with open(base_cards_path) as f:
        base_cards = json.load(f)

    misses = []
    with open(miss_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                misses.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    refined = mod.refine_cards(base_cards, misses, max_patterns=max_patterns)

    with open(output_path, "w") as f:
        json.dump(refined, f, indent=2)

    print(f"  Refined cards written to {output_path}")
    print(f"  Anchors: {len(refined)}, miss-log entries consumed: {len(misses)}")
```

- [ ] **Step 6: Run the CLI tests**

Run:
```
uv run pytest tests/test_cli_atlas_refine.py -v
```

Expected: all three PASS.

- [ ] **Step 7: Commit**

```bash
git add psa/cli.py tests/test_cli_atlas_refine.py
git commit -m "$(cat <<'EOF'
feat: add psa atlas refine CLI command

Thin wrapper over scripts/refine_anchor_cards.py that resolves the
current atlas for a tenant, takes a miss-log path, and writes
anchor_cards_refined.json into the atlas directory. With the
AnchorIndex loader change, the next psa train --coactivation run
will regenerate training data against refined CE scores.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Full test suite + lint

- [ ] **Step 1: Run the full test suite**

Run:
```
uv run pytest tests/ -q
```

Expected: all PASS (527+ tests, including the new loader + CLI tests).

- [ ] **Step 2: Run lint**

Run:
```
uv run ruff check .
uv run ruff format --check .
```

Expected: both clean. If `ruff format` reports changes, run `uv run ruff format .` and stage the result.

- [ ] **Step 3: Commit any format fixes (only if needed)**

```bash
git add -u
git commit -m "chore: ruff format pass"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `AnchorIndex.load()` prefers refined cards file | Task 1 |
| Fallback to raw when refined absent | Task 1 |
| Clear error when neither file exists | Task 1 |
| `psa atlas refine` subcommand registered | Task 2 (Step 3) |
| `--miss-log` required argument | Task 2 (Step 3) |
| `--max-patterns` optional argument | Task 2 (Step 3) |
| Resolves current atlas via `AtlasManager` | Task 2 (Step 5) |
| Writes `anchor_cards_refined.json` into atlas dir | Task 2 (Step 5) |
| Clear error when no atlas exists | Task 2 (Step 5, test covers) |
| Clear error when miss log missing | Task 2 (Step 5, test covers) |
| Full test suite + lint pass | Task 3 |

No gaps.

**Placeholder scan:** No TBDs. Every code step has a complete code block.

**Type consistency:**
- `_cmd_atlas_refine(args)` defined Task 2 Step 5, registered Task 2 Step 4 — consistent.
- `p_atlas_refine.add_argument("--miss-log", ...)` matches `args.miss_log` usage in the command — consistent.
- `mod.refine_cards(base_cards, misses, max_patterns=...)` matches the signature in `scripts/refine_anchor_cards.py:130` (verified) — consistent.
