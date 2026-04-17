# Research / Production Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce a hard research/production boundary on anchor-card refinement through write paths: `psa atlas refine` writes a never-auto-loaded candidate, `psa atlas promote-refinement` is the only way for a refinement to affect inference.

**Architecture:** Two artifacts under each `atlas_v{N}/`: `anchor_cards_candidate.json` (+ sibling `.meta.json`) from `refine`, and `anchor_cards_refined.json` (+ sibling `.meta.json`) only from `promote-refinement`. The loader keeps its current shape; the boundary lives on the write side. `source` in metadata is immutable across promotion.

**Tech Stack:** Python 3.13, argparse, `importlib.util`, existing `psa.atlas.AtlasManager` / `psa.tenant.TenantManager`.

**Design spec:** `docs/superpowers/specs/2026-04-17-research-prod-boundary-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `psa/cli.py` | Modify | Rewrite `_cmd_atlas_refine` output path, write metadata sibling, accept `--source`. Add `_cmd_atlas_refine_promote` + subparser + dispatch. |
| `psa/anchor.py` | Modify | When `AnchorIndex.load()` picks the refined file, emit one INFO log line with provenance from the sibling `.meta.json` (or `source=unknown` when absent). |
| `tests/test_cli_atlas_refine.py` | Modify | Update existing tests for candidate rename; add tests for metadata content, `--source`, and the full promote flow. |
| `tests/test_anchor.py` | Modify | Add tests for the `source=unknown` log path and confirmation that a candidate-alone atlas dir does NOT auto-load. |

---

### Task 1: `psa atlas refine` writes candidate + metadata

The current command writes `anchor_cards_refined.json` directly, which is exactly the boundary violation we're fixing. Rename the output file, add a sibling metadata JSON, and accept `--source`.

**Files:**
- Modify: `psa/cli.py` — `_cmd_atlas_refine()` (lines 452-521) + subparser (lines 1515-1529)
- Modify: `tests/test_cli_atlas_refine.py`

- [ ] **Step 1: Update the existing `test_atlas_refine_writes_refined_cards_file` test to assert the candidate path**

Edit `tests/test_cli_atlas_refine.py`. Find `test_atlas_refine_writes_refined_cards_file` and replace its body (the portion after `main()`) with:

```python
    candidate_path = atlas_dir / "anchor_cards_candidate.json"
    assert candidate_path.exists(), "candidate cards file must be written"

    # Refined is NOT written by `refine`; only `promote-refinement` touches it.
    assert not (atlas_dir / "anchor_cards_refined.json").exists()

    refined_cards = json.loads(candidate_path.read_text())
    assert len(refined_cards) == 1
    patterns = refined_cards[0]["generated_query_patterns"]
    assert "original pattern" in patterns
    assert len(patterns) > 1
```

- [ ] **Step 2: Update `test_atlas_refine_skips_write_on_empty_miss_log` to check the candidate path**

Find the assertion `assert not (atlas_dir / "anchor_cards_refined.json").exists()` in that test and replace with:

```python
    assert not (atlas_dir / "anchor_cards_candidate.json").exists()
    assert not (atlas_dir / "anchor_cards_refined.json").exists()
```

- [ ] **Step 3: Add the metadata-assertion test**

Append to `tests/test_cli_atlas_refine.py`:

```python
def test_atlas_refine_writes_candidate_metadata(tmp_path, monkeypatch, capsys):
    """`psa atlas refine` writes anchor_cards_candidate.meta.json with provenance."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original"])

    miss_log = tmp_path / "misses.jsonl"
    miss_log.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "query": "how does token refresh work in the auth flow",
                "gold_anchor_ids": [1],
            }
        )
        + "\n"
    )

    with patch("sys.argv", ["psa", "atlas", "refine", "--miss-log", str(miss_log)]):
        main()

    meta_path = atlas_dir / "anchor_cards_candidate.meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["source"] == "manual"
    assert meta["tenant_id"] == "default"
    assert meta["atlas_version"] == 1
    assert meta["promoted"] is False
    assert meta["promoted_at"] is None
    assert meta["miss_log_path"] == str(miss_log)
    assert meta["n_anchors_touched"] >= 1
    assert meta["n_patterns_added"] >= 1
    assert "created_at" in meta and meta["created_at"]
```

- [ ] **Step 4: Add the `--source` test**

Append to `tests/test_cli_atlas_refine.py`:

```python
def test_atlas_refine_records_source_from_flag(tmp_path, monkeypatch, capsys):
    """`--source benchmark` is recorded in the candidate metadata."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original"])

    miss_log = tmp_path / "misses.jsonl"
    miss_log.write_text(
        json.dumps({"query": "baking cookies", "gold_anchor_ids": [1]}) + "\n"
    )

    with patch(
        "sys.argv",
        [
            "psa",
            "atlas",
            "refine",
            "--miss-log",
            str(miss_log),
            "--source",
            "benchmark",
        ],
    ):
        main()

    meta = json.loads((atlas_dir / "anchor_cards_candidate.meta.json").read_text())
    assert meta["source"] == "benchmark"
```

- [ ] **Step 5: Run the failing tests**

```
uv run pytest tests/test_cli_atlas_refine.py -v
```

Expected: `test_atlas_refine_writes_refined_cards_file` FAIL (still writes to refined path), `test_atlas_refine_writes_candidate_metadata` FAIL (no .meta.json), `test_atlas_refine_records_source_from_flag` FAIL (unrecognized `--source`).

- [ ] **Step 6: Add `--source` to the argparse subparser**

In `psa/cli.py`, find the block starting at line 1515 (`p_atlas_refine = atlas_sub.add_parser("refine", ...`). After the `--max-patterns` argument, add:

```python
    p_atlas_refine.add_argument(
        "--source",
        default="manual",
        help=(
            "Provenance marker for this refinement — one of "
            "'manual', 'benchmark', 'oracle', 'query_fingerprint'. "
            "Defaults to 'manual'. Stored in the candidate's .meta.json and "
            "preserved verbatim on promotion."
        ),
    )
```

- [ ] **Step 7: Rewrite `_cmd_atlas_refine()` to produce candidate + metadata**

Replace the entire `_cmd_atlas_refine()` function at `psa/cli.py:452-521` with:

```python
def _cmd_atlas_refine(args):
    """Refine anchor cards into a candidate artifact (not inference-visible)."""
    import datetime as _dt
    import importlib.util
    import json
    import os
    from pathlib import Path

    from .tenant import TenantManager
    from .atlas import AtlasManager

    tenant_id = getattr(args, "tenant", "default")
    miss_log_path = args.miss_log
    max_patterns = getattr(args, "max_patterns", 20)
    source = getattr(args, "source", "manual") or "manual"

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
    candidate_path = atlas_dir / "anchor_cards_candidate.json"
    candidate_meta_path = atlas_dir / "anchor_cards_candidate.meta.json"

    if not base_cards_path.exists():
        print(f"  Error: {base_cards_path} not found.")
        sys.exit(1)

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

    if not misses:
        print(f"  Warning: miss log {miss_log_path} has 0 valid entries.")
        print("  Nothing to refine — skipping write (no candidate produced).")
        return

    refined = mod.refine_cards(base_cards, misses, max_patterns=max_patterns)

    # Diff stats for metadata
    base_by_id = {c.get("anchor_id"): c for c in base_cards}
    n_anchors_touched = 0
    n_patterns_added = 0
    for r in refined:
        b = base_by_id.get(r.get("anchor_id"), {})
        before = set(b.get("generated_query_patterns", []) or [])
        after = set(r.get("generated_query_patterns", []) or [])
        delta = after - before
        if delta:
            n_anchors_touched += 1
            n_patterns_added += len(delta)

    with open(candidate_path, "w") as f:
        json.dump(refined, f, indent=2)

    meta = {
        "source": source,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "tenant_id": tenant_id,
        "atlas_version": version,
        "promoted": False,
        "promoted_at": None,
        "miss_log_path": str(miss_log_path),
        "n_anchors_touched": n_anchors_touched,
        "n_patterns_added": n_patterns_added,
    }
    with open(candidate_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Candidate written to {candidate_path}")
    print(f"  Metadata written to {candidate_meta_path}")
    print(
        f"  Source: {source}, anchors touched: {n_anchors_touched}, "
        f"patterns added: {n_patterns_added}"
    )
    print("  Run 'psa atlas promote-refinement' to make this candidate inference-visible.")
```

- [ ] **Step 8: Run tests**

```
uv run pytest tests/test_cli_atlas_refine.py -v
```

Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add psa/cli.py tests/test_cli_atlas_refine.py
git commit -m "$(cat <<'EOF'
feat: psa atlas refine writes candidate artifact + metadata

Candidate cards land at anchor_cards_candidate.json with a sibling
.meta.json carrying source (default: manual; override via --source),
created_at, tenant_id, atlas_version, promoted=false, miss_log_path,
and diff stats. Never writes anchor_cards_refined.json — that path
is reserved for the promote-refinement step in the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `psa atlas promote-refinement`

Promotion is the only write path to `anchor_cards_refined.json`. It reads the candidate + its metadata, copies both to the refined paths, preserves the original `source`, and flips `promoted` to true.

**Files:**
- Modify: `psa/cli.py`
- Modify: `tests/test_cli_atlas_refine.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli_atlas_refine.py`:

```python
def test_promote_refinement_creates_refined_and_meta(tmp_path, monkeypatch, capsys):
    """promote-refinement copies candidate → refined and marks promoted=true."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original"])

    # First, refine to create a candidate
    miss_log = tmp_path / "misses.jsonl"
    miss_log.write_text(
        json.dumps({"query": "auth token refresh flow", "gold_anchor_ids": [1]}) + "\n"
    )
    with patch(
        "sys.argv",
        [
            "psa",
            "atlas",
            "refine",
            "--miss-log",
            str(miss_log),
            "--source",
            "benchmark",
        ],
    ):
        main()

    # Now promote
    with patch("sys.argv", ["psa", "atlas", "promote-refinement"]):
        main()

    refined_path = atlas_dir / "anchor_cards_refined.json"
    refined_meta_path = atlas_dir / "anchor_cards_refined.meta.json"
    assert refined_path.exists()
    assert refined_meta_path.exists()

    # Candidate remains in place (not deleted)
    assert (atlas_dir / "anchor_cards_candidate.json").exists()
    assert (atlas_dir / "anchor_cards_candidate.meta.json").exists()

    meta = json.loads(refined_meta_path.read_text())
    assert meta["promoted"] is True
    assert meta["promoted_at"] is not None
    assert meta["source"] == "benchmark"  # preserved from candidate


def test_promote_refinement_errors_when_no_candidate(tmp_path, monkeypatch, capsys):
    """promote-refinement with no candidate exits non-zero."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original"])

    with patch("sys.argv", ["psa", "atlas", "promote-refinement"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code != 0

    out = capsys.readouterr().out
    assert "candidate" in out.lower()


def test_promote_refinement_overwrites_previous_promotion(tmp_path, monkeypatch, capsys):
    """Running promote twice replaces the refined artifact cleanly."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, patterns=["original"])

    miss_log = tmp_path / "misses.jsonl"
    miss_log.write_text(
        json.dumps({"query": "sample query text", "gold_anchor_ids": [1]}) + "\n"
    )
    with patch("sys.argv", ["psa", "atlas", "refine", "--miss-log", str(miss_log)]):
        main()

    with patch("sys.argv", ["psa", "atlas", "promote-refinement"]):
        main()
    first_meta = json.loads((atlas_dir / "anchor_cards_refined.meta.json").read_text())

    # Re-refine with different source, re-promote
    with patch(
        "sys.argv",
        [
            "psa",
            "atlas",
            "refine",
            "--miss-log",
            str(miss_log),
            "--source",
            "oracle",
        ],
    ):
        main()
    with patch("sys.argv", ["psa", "atlas", "promote-refinement"]):
        main()
    second_meta = json.loads((atlas_dir / "anchor_cards_refined.meta.json").read_text())

    assert second_meta["source"] == "oracle"
    assert second_meta["promoted_at"] != first_meta["promoted_at"]
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_cli_atlas_refine.py::test_promote_refinement_creates_refined_and_meta tests/test_cli_atlas_refine.py::test_promote_refinement_errors_when_no_candidate tests/test_cli_atlas_refine.py::test_promote_refinement_overwrites_previous_promotion -v
```

Expected: all FAIL with "argument atlas_action: invalid choice: 'promote-refinement'" or equivalent.

- [ ] **Step 3: Register the `promote-refinement` subparser**

In `psa/cli.py`, immediately after the `p_atlas_refine.add_argument("--source", ...)` block added in Task 1 Step 6 (around line 1530), add:

```python
    atlas_sub.add_parser(
        "promote-refinement",
        help=(
            "Promote the current atlas candidate (anchor_cards_candidate.json) to "
            "the live refined artifact (anchor_cards_refined.json). This is the "
            "only write path to the live refined file."
        ),
    )
```

- [ ] **Step 4: Add dispatch in `cmd_atlas()`**

In `psa/cli.py`, find `cmd_atlas()` (around line 345). Replace with:

```python
def cmd_atlas(args):
    """Handle 'psa atlas <subcommand>'."""
    action = getattr(args, "atlas_action", None)
    if not action:
        print("Usage: psa atlas {build,status,health,refine,promote-refinement}")
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
```

- [ ] **Step 5: Implement `_cmd_atlas_refine_promote()`**

Add this function at `psa/cli.py` immediately after `_cmd_atlas_refine()`:

```python
def _cmd_atlas_refine_promote(args):
    """Promote the current candidate refinement into the live refined artifact."""
    import datetime as _dt
    import json
    import os
    import shutil
    from pathlib import Path

    from .tenant import TenantManager
    from .atlas import AtlasManager

    tenant_id = getattr(args, "tenant", "default")
    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
    version = mgr.latest_version()
    if version is None:
        print(f"  Error: no atlas for tenant '{tenant_id}'. Run 'psa atlas build' first.")
        sys.exit(1)

    atlas_dir = Path(tenant.root_dir) / f"atlas_v{version}"
    candidate_path = atlas_dir / "anchor_cards_candidate.json"
    candidate_meta_path = atlas_dir / "anchor_cards_candidate.meta.json"
    refined_path = atlas_dir / "anchor_cards_refined.json"
    refined_meta_path = atlas_dir / "anchor_cards_refined.meta.json"

    if not candidate_path.exists():
        print(f"  Error: no candidate to promote at {candidate_path}.")
        print("  Run 'psa atlas refine --miss-log PATH' first.")
        sys.exit(1)

    # Copy cards verbatim
    shutil.copyfile(candidate_path, refined_path)

    # Rewrite meta with promoted=true / promoted_at=<now>, preserving source.
    if candidate_meta_path.exists():
        with open(candidate_meta_path) as f:
            meta = json.load(f)
    else:
        # Candidate without meta — rare but handle gracefully.
        meta = {
            "source": "unknown",
            "tenant_id": tenant_id,
            "atlas_version": version,
        }

    meta["promoted"] = True
    meta["promoted_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    with open(refined_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Promoted: {candidate_path} → {refined_path}")
    print(f"  Metadata: {refined_meta_path}")
    print(f"  Source: {meta.get('source', 'unknown')}, promoted_at: {meta['promoted_at']}")
```

- [ ] **Step 6: Run tests**

```
uv run pytest tests/test_cli_atlas_refine.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add psa/cli.py tests/test_cli_atlas_refine.py
git commit -m "$(cat <<'EOF'
feat: psa atlas promote-refinement

The only write path to anchor_cards_refined.json. Copies the current
candidate + sibling metadata, preserves source verbatim, flips promoted
to true and stamps promoted_at. Idempotent: running twice simply
replaces the refined artifact.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Loader provenance logging + candidate-never-loads regression

`AnchorIndex.load()`'s selection mechanism stays the same — refined preferred over raw. What changes: when the refined file is chosen, read the sibling `.meta.json` (if present) and log one INFO line with `source`. When absent, log `source=unknown`. Adds a regression test that a candidate file alone never causes auto-load.

**Files:**
- Modify: `psa/anchor.py` — `AnchorIndex.load()` (around line 225-255)
- Modify: `tests/test_anchor.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_anchor.py`:

```python
def test_anchor_index_load_logs_source_unknown_when_meta_missing(tmp_path, caplog):
    """Loading a refined file with no sibling .meta.json logs source=unknown at INFO."""
    import json
    import logging
    import numpy as np
    from psa.anchor import AnchorIndex

    cards = [
        {
            "anchor_id": 1,
            "name": "a",
            "meaning": "m",
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
            "generated_query_patterns": [],
            "query_fingerprint": [],
        }
    ]
    (tmp_path / "anchor_cards_refined.json").write_text(json.dumps(cards))
    np.save(tmp_path / "centroids.npy", np.zeros((1, 768), dtype=np.float32))

    caplog.set_level(logging.INFO, logger="psa.anchor")
    AnchorIndex.load(str(tmp_path))

    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("source=unknown" in m for m in msgs), msgs


def test_anchor_index_load_logs_source_from_meta_file(tmp_path, caplog):
    """Loading a refined file with a sibling .meta.json logs its source."""
    import json
    import logging
    import numpy as np
    from psa.anchor import AnchorIndex

    cards = [
        {
            "anchor_id": 1,
            "name": "a",
            "meaning": "m",
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
            "generated_query_patterns": [],
            "query_fingerprint": [],
        }
    ]
    (tmp_path / "anchor_cards_refined.json").write_text(json.dumps(cards))
    (tmp_path / "anchor_cards_refined.meta.json").write_text(
        json.dumps({"source": "oracle", "promoted": True})
    )
    np.save(tmp_path / "centroids.npy", np.zeros((1, 768), dtype=np.float32))

    caplog.set_level(logging.INFO, logger="psa.anchor")
    AnchorIndex.load(str(tmp_path))

    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("source=oracle" in m for m in msgs), msgs


def test_anchor_index_load_ignores_candidate_when_no_refined(tmp_path):
    """A candidate file alone does NOT cause auto-load; falls back to anchor_cards.json."""
    import json
    import numpy as np
    from psa.anchor import AnchorIndex

    raw = [
        {
            "anchor_id": 1,
            "name": "raw",
            "meaning": "m",
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
            "generated_query_patterns": ["raw pattern"],
            "query_fingerprint": [],
        }
    ]
    candidate = [
        {
            **raw[0],
            "generated_query_patterns": ["raw pattern", "candidate pattern"],
        }
    ]
    (tmp_path / "anchor_cards.json").write_text(json.dumps(raw))
    (tmp_path / "anchor_cards_candidate.json").write_text(json.dumps(candidate))
    np.save(tmp_path / "centroids.npy", np.zeros((1, 768), dtype=np.float32))

    idx = AnchorIndex.load(str(tmp_path))
    assert idx._cards[0].generated_query_patterns == ["raw pattern"]
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_anchor.py::test_anchor_index_load_logs_source_unknown_when_meta_missing tests/test_anchor.py::test_anchor_index_load_logs_source_from_meta_file tests/test_anchor.py::test_anchor_index_load_ignores_candidate_when_no_refined -v
```

Expected: the third test PASSES (loader already ignores unknown files); the two log-assertion tests FAIL (no such log line).

- [ ] **Step 3: Add the INFO log emission in `AnchorIndex.load()`**

In `psa/anchor.py`, replace the `AnchorIndex.load()` block at lines 225-250 with:

```python
    @classmethod
    def load(cls, path: str, dim: int = 768) -> "AnchorIndex":
        """Load a previously saved AnchorIndex from a directory."""
        refined_path = os.path.join(path, "anchor_cards_refined.json")
        raw_path = os.path.join(path, "anchor_cards.json")
        centroids_path = os.path.join(path, "centroids.npy")

        if os.path.exists(refined_path):
            cards_path = refined_path
            # Log provenance for the live refined artifact.
            meta_path = os.path.join(path, "anchor_cards_refined.meta.json")
            source = "unknown"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as _f:
                        source = json.load(_f).get("source", "unknown") or "unknown"
                except (OSError, json.JSONDecodeError):
                    source = "unknown"
            logger.info(
                "AnchorIndex loading refined cards from %s (source=%s)",
                refined_path,
                source,
            )
        elif os.path.exists(raw_path):
            cards_path = raw_path
        else:
            raise FileNotFoundError(
                f"No anchor_cards.json or anchor_cards_refined.json at {path}"
            )

        with open(cards_path) as f:
            cards_data = json.load(f)
        cards = [AnchorCard.from_dict(d) for d in cards_data]

        idx = cls(dim=dim)
        idx._cards = cards

        if os.path.exists(centroids_path):
            idx._centroids = np.load(centroids_path)
            if idx._centroids.ndim == 2:
                idx.dim = idx._centroids.shape[1]
            if idx._use_faiss:
                idx._build_faiss()
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_anchor.py -v
```

Expected: all PASS (including the three new tests and all previously-existing tests).

- [ ] **Step 5: Commit**

```bash
git add psa/anchor.py tests/test_anchor.py
git commit -m "$(cat <<'EOF'
feat: log refinement provenance at atlas load

When AnchorIndex.load() picks up anchor_cards_refined.json, read the
sibling .meta.json and log one INFO line naming its source. Absent
or unreadable meta logs source=unknown — this is the pre-boundary
migration path for refined files that predate the candidate/promote
workflow.

Also adds a regression test that a candidate file alone never causes
auto-load.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Full test suite + lint

- [ ] **Step 1: Run the full test suite**

```
uv run pytest tests/ -q
```

Expected: all PASS. The one pre-existing flaky `test_convo_miner::test_convo_mining` (httpx timeout on network-dependent Qwen call) is orthogonal — if it's the only failure, ignore.

- [ ] **Step 2: Lint + format check**

```
uv run ruff check .
uv run ruff format --check .
```

Expected: both clean. If `ruff format --check` fails, run `uv run ruff format .`, verify clean, and commit.

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
| `anchor_cards_candidate.json` as non-auto-loaded output | Task 1 (renamed output path) |
| `anchor_cards_refined.json` only written by promote | Task 2 (new command, Task 1 removes the previous write) |
| Sibling `.meta.json` with `source`, `created_at`, `tenant_id`, `atlas_version`, `promoted`, `promoted_at`, `miss_log_path`, `n_anchors_touched`, `n_patterns_added` | Task 1 (metadata dict), Task 1 Step 3 (test) |
| `source` immutable across promotion | Task 2 Step 5 (preserves `meta["source"]`); tested at Task 2 Step 1 |
| `psa atlas refine --source` flag, default `manual` | Task 1 Step 6 (subparser), Task 1 Step 4 (test) |
| `psa atlas promote-refinement` command | Task 2 |
| Promote errors cleanly when no candidate exists | Task 2 Step 1 (test), Step 5 (implementation) |
| `AnchorIndex.load()` mechanism unchanged | Task 3 — preference order untouched; only an INFO log added |
| Refined without meta logs `source=unknown` | Task 3 |
| Candidate alone does not auto-load | Task 3 (regression test) |
| Existing `anchor_cards_refined.json` pre-branch still loads | Inherent: loader preference unchanged; verified by Task 3 `source=unknown` test |

No gaps.

**Placeholder scan:** No TBDs. Every code step has a complete code block.

**Type consistency:**
- `_cmd_atlas_refine` / `_cmd_atlas_refine_promote` function names are consistent between the function definitions (Task 1, Task 2 Step 5) and the dispatch table (Task 2 Step 4).
- Metadata keys are identical across Task 1 (refine-side write) and Task 2 (promote-side read + mutation).
- `mod.refine_cards(base_cards, misses, max_patterns=...)` matches the signature in `scripts/refine_anchor_cards.py:129` (same pattern used in the current `_cmd_atlas_refine`).
