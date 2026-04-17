# Research / Production Boundary Design

**Date:** 2026-04-17
**Branch:** `fix/research-prod-boundary` (off `main`)

## Problem

Phase 3C made `AnchorIndex.load()` prefer `anchor_cards_refined.json` when present, and `psa atlas refine` writes that file directly. The result: any refinement — regardless of where its signal came from — silently becomes inference behavior. Benchmark-driven miss logs, ablation artifacts, and one-off experiments can leak into production state without review.

## Goal

Enforce a hard boundary between research artifacts and live inference state through *write paths*, not loader logic. Refinement still works; it just has a two-step workflow: produce a candidate, review, then explicitly promote.

## Non-goals

- Lifecycle audit / removal of benchmark-driven memory-scorer training (Branch 2).
- `query_fingerprint` → `to_stable_card_text()` wiring (Branch 3).
- Observability rollups, dual metric reporting (Branch 4).
- Auto-migration of existing `anchor_cards_refined.json` files — stay passive; log `source=unknown` on load.

## Design

### File layout under `atlas_v{N}/`

| File | Role | Auto-loaded by `AnchorIndex.load()` |
|---|---|---|
| `anchor_cards.json` | Canonical baseline from atlas build. | Yes (when no refined file present). |
| `anchor_cards_candidate.json` | Output of `psa atlas refine`. Research artifact. | **No** — never inference-visible. |
| `anchor_cards_refined.json` | Promoted, live. Only written by `psa atlas promote-refinement`. | Yes (preferred). |
| `<name>.meta.json` | Sibling metadata per candidate/refined artifact. | Read for provenance at load time; does not affect preference. |

### Loader behavior

`AnchorIndex.load()` is unchanged in mechanism — it still prefers `anchor_cards_refined.json` when present, falls back to `anchor_cards.json`, raises when neither exists. The boundary is enforced on the **write** side: `anchor_cards_refined.json` can only appear through `psa atlas promote-refinement`.

Addition: when loading, if the preferred file is `anchor_cards_refined.json` but the sibling `.meta.json` is absent, log one INFO line at `psa.anchor` level with `source=unknown`. That way pre-branch refined files keep working but their provenance is flagged as lost.

### Metadata schema (sibling JSON)

```json
{
  "source": "benchmark|oracle|query_fingerprint|manual",
  "created_at": "2026-04-17T...",
  "tenant_id": "default",
  "atlas_version": 1,
  "promoted": false,
  "promoted_at": null,
  "miss_log_path": "/tmp/...",
  "n_anchors_touched": 102,
  "n_patterns_added": 789
}
```

`source` is **immutable after creation** — promotion does not overwrite it. That way a promoted artifact retains "originated from benchmark" even after it becomes live.

`miss_log_path` is optional (only meaningful when refinement was driven by a miss log).

`n_anchors_touched` and `n_patterns_added` are computed at write time by diffing the candidate against its base atlas cards.

### CLI surface

**`psa atlas refine --miss-log PATH [--max-patterns N] [--source SOURCE]`** *(modified)*

- Writes `anchor_cards_candidate.json` (renamed from `anchor_cards_refined.json`).
- Writes sibling `anchor_cards_candidate.meta.json` with `source` (default: `"manual"`), `created_at`, `tenant_id`, `atlas_version`, `promoted=false`, `miss_log_path`, `n_anchors_touched`, `n_patterns_added`.
- Overwrites an existing candidate — candidates are transient by design.
- Never writes `anchor_cards_refined.json`.

**`psa atlas promote-refinement [--tenant TENANT]`** *(new)*

- Reads `anchor_cards_candidate.json` + its `.meta.json` from the current atlas directory.
- Copies to `anchor_cards_refined.json` + `anchor_cards_refined.meta.json`.
- Updates the new meta file: preserves `source` (immutable), sets `promoted=true`, sets `promoted_at=<now>`.
- Leaves the candidate in place (operator can re-promote later if they want).
- Errors cleanly with `sys.exit(1)` if no candidate exists. Message names the expected path.
- No force/overwrite flag — promoting twice simply rewrites the refined artifact.

### Constraint

`source` is set once at `refine` time, carried into `promote-refinement` verbatim. A single immutable field covers the "where did this originally come from?" question without adding `source_at_promotion` or a second field.

## Tests

New or modified:

1. `AnchorIndex.load()` — existing `anchor_cards_refined.json` without a sibling `.meta.json` still loads, logs `source=unknown` at INFO (existing test file).
2. `AnchorIndex.load()` — presence of `anchor_cards_candidate.json` without a refined file does NOT cause it to load; falls back to `anchor_cards.json`.
3. `psa atlas refine` writes `anchor_cards_candidate.json` + `anchor_cards_candidate.meta.json`. No `anchor_cards_refined.json` appears. Metadata has `source=manual` by default, `promoted=false`, the required fields, and correct `n_anchors_touched` / `n_patterns_added`.
4. `psa atlas refine --source benchmark` records `source=benchmark` in the metadata.
5. `psa atlas promote-refinement` with a candidate present: creates refined + refined.meta with `promoted=true`, `promoted_at` set, `source` preserved from the candidate.
6. `psa atlas promote-refinement` with no candidate: exits non-zero, stdout mentions the missing path.
7. After promote, the loader prefers the refined file (unchanged Phase 3C behavior).
8. Overwriting: running `refine` twice replaces the candidate cleanly; running `promote-refinement` twice replaces the refined file cleanly.

## Files touched

| File | Action |
|---|---|
| `psa/cli.py` | Modify `_cmd_atlas_refine` output path, write metadata sibling, accept `--source`. Add `_cmd_atlas_refine_promote` + subparser. Register dispatch. |
| `psa/anchor.py` | Add INFO log line in `AnchorIndex.load()` when refined file loaded but no sibling meta found. |
| `tests/test_cli_atlas_refine.py` | Update existing tests for the rename; add `--source` test, promote tests. |
| `tests/test_anchor.py` | Add the `source=unknown` log test and the candidate-never-loaded test. |
| `docs/superpowers/specs/2026-04-16-phase3c-refined-atlas-loader-design.md` | One-line note at top: "superseded by the Branch 1 boundary design on 2026-04-17". |

## Out of scope reminder

- No changes to `scripts/refine_anchor_cards.py` itself — it's a pure function that produces refined card JSON; the CLI wraps it.
- No changes to `AnchorIndex.load()` mechanism.
- No `refine-status` subcommand in this branch. Deferred.

## Success criteria

- Full test suite green.
- `psa atlas refine` alone produces no loader-visible change.
- `psa atlas promote-refinement` is the only write path to `anchor_cards_refined.json`.
- Existing refined files (pre-branch) still load; operator sees a log line naming their unknown provenance.
