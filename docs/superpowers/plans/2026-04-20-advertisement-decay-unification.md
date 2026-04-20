# Advertisement Decay Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make advertisement decay a single lifecycle-integrated mechanism with JSON-configured conservative defaults, remove the legacy `psa atlas decay` CLI path, and align docs/tests with the new behavior.

**Architecture:** Keep the ledger-based lifecycle path as the only supported advertisement-decay implementation. Update runtime defaults and generated config to be authoritative via `~/.psa/config.json`, remove advertisement-specific legacy CLI entry points, and decouple lifecycle removal checks from Stage 1 helper imports while preserving shielding, pinning, and “dormancy alone is insufficient” behavior.

**Tech Stack:** Python 3.13, pytest, existing PSA CLI/config/lifecycle modules, README docs

---

### Task 1: Make JSON Config Authoritative And Default-On

**Files:**
- Modify: `psa/advertisement/config.py`
- Modify: `psa/config.py`
- Modify: `tests/test_advertisement_config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add/update tests so they expect:
- `AdvertisementDecayConfig.from_mempalace()` defaults to `tracking_enabled=True`, `removal_enabled=True`, `grace_days=30`, `sustained_cycles=21`, `min_patterns_floor=5`
- `MempalaceConfig.init()` writes an explicit `advertisement_decay` block into `config.json`
- advertisement-decay env override behavior is no longer supported in the config tests

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_advertisement_config.py tests/test_config.py -q`
Expected: FAIL on the old `False` defaults / missing generated JSON block / env-override assumptions.

- [ ] **Step 3: Write the minimal implementation**

Update `psa/advertisement/config.py` to:
- remove `PSA_AD_DECAY_*` override reads
- keep `trace_queries` validation
- set runtime defaults to:
  - `tracking_enabled=True`
  - `removal_enabled=True`
  - `tau_days=45`
  - `grace_days=30`
  - `sustained_cycles=21`
  - `min_patterns_floor=5`

Update `psa/config.py` `init()` to write a complete default config including:
- `tenant_id`
- `psa_mode`
- `token_budget`
- `max_memories`
- `anchor_memory_budget`
- `trace_queries`
- `nightly_hour`
- `advertisement_decay` block with the fixed defaults above

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_advertisement_config.py tests/test_config.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

Run:
```bash
git add psa/advertisement/config.py psa/config.py tests/test_advertisement_config.py tests/test_config.py
git commit -m "feat: make advertisement decay config default-on"
```

### Task 2: Remove Legacy Advertisement CLI And Update Docs Surface

**Files:**
- Modify: `psa/cli.py`
- Delete: `tests/test_cli_atlas_decay.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing tests**

Add/update assertions so the command surface no longer exposes `psa atlas decay`, and README documents advertisement decay as the lifecycle path with JSON config as the authoritative setup.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli_advertisement.py tests/test_config.py -q`
Expected: current docs/CLI assumptions are incomplete and the CLI still exposes `atlas decay`.

- [ ] **Step 3: Write the minimal implementation**

Update `psa/cli.py` to remove:
- the `atlas decay` parser entry
- any dispatch/help text that advertises advertisement decay through the atlas candidate flow

Delete `tests/test_cli_atlas_decay.py`.

Update `README.md` so it matches the implementation:
- full JSON config block in install guide
- advertisement decay described as lifecycle-integrated and default-on
- `psa atlas decay` removed from supported command reference
- remove leftover stage/calibration/default-off wording for advertisement decay

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_advertisement.py tests/test_config.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

Run:
```bash
git add psa/cli.py README.md tests/test_cli_advertisement.py tests/test_config.py
git rm tests/test_cli_atlas_decay.py
git commit -m "feat: remove legacy advertisement decay cli"
```

### Task 3: Decouple Lifecycle Removal Logic From Stage 1 Helpers

**Files:**
- Modify: `psa/lifecycle.py`
- Modify: `psa/advertisement/ledger.py`
- Modify: `psa/advertisement/metadata.py`
- Create or Modify: `psa/advertisement/guards.py`
- Modify: `tests/test_lifecycle_advertisement_decay.py`
- Modify: `tests/test_ledger_decay.py`

- [ ] **Step 1: Write the failing tests**

Add tests that assert:
- lifecycle removal no longer imports Stage 1 helper functions at runtime
- shielding and pinned-pattern behavior still work through the new lifecycle-owned guard path
- dormancy alone does not create removal candidates

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_lifecycle_advertisement_decay.py tests/test_ledger_decay.py -q`
Expected: FAIL because lifecycle still reaches into `psa.advertisement.decay` / `metadata` directly and dormancy-only behavior is not explicitly protected.

- [ ] **Step 3: Write the minimal implementation**

Create a lifecycle-owned guard surface, for example `psa/advertisement/guards.py`, that provides:
- anchor shielding lookup for ledger evaluation
- metadata-backed pinned-pattern checks

Update `psa/lifecycle.py` to use the new guard surface instead of importing Stage 1 helper functions.

Update `psa/advertisement/ledger.py` evaluation path so:
- time decay contributes to risk
- removal still requires sustained negative cycles
- lack of recent use by itself does not produce a removal candidate

Keep `pattern_metadata.json` support for provenance and `pinned`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lifecycle_advertisement_decay.py tests/test_ledger_decay.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

Run:
```bash
git add psa/lifecycle.py psa/advertisement/ledger.py psa/advertisement/metadata.py psa/advertisement/guards.py tests/test_lifecycle_advertisement_decay.py tests/test_ledger_decay.py
git commit -m "refactor: decouple advertisement lifecycle guards"
```

### Task 4: Final Verification

**Files:**
- Modify: `docs/superpowers/specs/2026-04-20-advertisement-decay-unification-design.md`
- Verify: changed files above

- [ ] **Step 1: Sync the updated spec into this branch**

Bring the approved spec changes into this worktree so the repository state matches the implemented behavior.

- [ ] **Step 2: Run the focused verification suite**

Run:
```bash
uv run pytest tests/test_advertisement_config.py tests/test_config.py tests/test_lifecycle_advertisement_decay.py tests/test_cli_advertisement.py tests/test_ledger_decay.py -q
```
Expected: PASS

- [ ] **Step 3: Run a CLI smoke check**

Run:
```bash
uv run python -m psa --help
uv run python -m psa atlas --help
uv run python -m psa advertisement --help
```
Expected:
- `psa atlas --help` does not list `decay`
- advertisement maintenance commands still appear

- [ ] **Step 4: Commit**

Run:
```bash
git add docs/superpowers/specs/2026-04-20-advertisement-decay-unification-design.md
git commit -m "docs: align advertisement decay spec with implementation"
```
