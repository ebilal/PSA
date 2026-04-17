# Phase 3C — Refined-Atlas Loader Design

**Date:** 2026-04-16
**Branch:** `feat/phase3a-refined-cards` (continues from Phase 3A cleanup)
**Scope:** Plumbing only — loader + CLI helper. No lifecycle wiring, no training runs in this spec.

## Problem

The `CoActivationSelector` was trained on CE scores computed from pre-refinement card text (`card.to_stable_card_text()` before `generated_query_patterns` were expanded from the miss log). Inference now runs against refined cards. The selector's learned threshold is calibrated to a CE-score distribution that inference no longer produces.

This is the single most plausible cause of the 12 scoring_rank misses (gold anchor in CE top-24, selector scores it below threshold). Phase 3B's ablation confirmed `min_k`/`max_k` tuning can't rescue these — the selector is actively assigning low scores to the gold anchor, not just cutting too aggressively.

## Goal

Give the system one canonical way to present refined cards to every consumer (training data generation, inference, ablations). Then anyone regenerating `coactivation_train.npz` on a refined atlas gets the right distribution automatically.

## Design

### Atlas-level refinement

`AnchorIndex.load(atlas_dir)` (called transitively by `Atlas.load()`) currently reads `anchor_cards.json` from `atlas_dir`. The change: if `anchor_cards_refined.json` exists in the same directory, load that instead.

```python
@classmethod
def load(cls, path: str, dim: int = 768) -> "AnchorIndex":
    refined_path = os.path.join(path, "anchor_cards_refined.json")
    raw_path = os.path.join(path, "anchor_cards.json")
    if os.path.exists(refined_path):
        cards_path = refined_path
    elif os.path.exists(raw_path):
        cards_path = raw_path
    else:
        raise FileNotFoundError(
            f"No anchor_cards.json or anchor_cards_refined.json at {path}"
        )
    # ... existing load logic, reading from cards_path
```

Consequences:
- Training data generation (`coactivation_data.py:90`) sees refined card text when building CE scores. No code change in that module.
- Inference pipeline (`pipeline.py`) sees refined cards automatically — the `FullAtlasScorer` it constructs from the atlas uses them. No code change.
- Ablations that want to compare refined-vs-raw can still use the existing `card_text_override_by_anchor_id` mechanism on `FullAtlasScorer` (the Phase 3A surgical tool — we keep it).

### CLI helper

Add `psa atlas refine` — a thin wrapper over `scripts/refine_anchor_cards.py` that resolves the current atlas for a tenant, takes a miss-log path, and writes `anchor_cards_refined.json` into the atlas dir.

```bash
psa atlas refine --miss-log PATH [--tenant default] [--max-patterns 20]
```

The script `scripts/refine_anchor_cards.py` already does the heavy lifting; this wrapper just locates `--base-atlas` and `--output` from the tenant's atlas manager so the user doesn't hand-compose paths.

### What this does NOT change

- `scripts/refine_anchor_cards.py` — unchanged. It's the refinement algorithm; already tested (19 tests).
- `full_atlas_scorer.py` — unchanged. The `card_text_override_by_anchor_id` mechanism stays for ablations.
- `pipeline.py` — unchanged. Gets refined cards for free via `Atlas.load()`.
- `coactivation_data.py` — unchanged. Same reason.
- `lifecycle.py` — unchanged in this round. Wire-up comes later, after we've verified retrain actually moves R@5.

## Files

| File | Change |
|------|--------|
| `psa/atlas.py` | `Atlas.load()` prefers `anchor_cards_refined.json` when present. |
| `tests/test_atlas.py` | Test: refined file loads when present; falls back to raw when absent; shape unchanged. |
| `psa/cli.py` | New `cmd_atlas_refine()` + argparse wiring under `psa atlas refine`. |
| `tests/test_cli.py` | Smoke test for `psa atlas refine` arg parsing. |

## Non-goals

- Lifecycle integration (auto-refinement in slow path) — next round if this works.
- Continual refinement from `query_fingerprint` — future round.
- Regenerating training data, retraining selector, benchmarking — user-driven workflows outside this spec.
- Migration of refined cards from `.worktrees/phase2a/artifacts/`: those carry Phase 2 temporal fields this branch's `AnchorCard` doesn't accept. User regenerates via the new `psa atlas refine` against an existing miss log.

## Verification

- `uv run pytest tests/test_atlas.py tests/test_cli.py -v` passes.
- `uv run ruff check . && uv run ruff format --check .` clean.
- Manual sanity: after `psa atlas refine --miss-log PATH`, `psa atlas status` reports the atlas normally and `Atlas.load()` returns cards whose `generated_query_patterns` include refinement-added phrases for the anchors in the miss log.

## Miss-log source

Miss logs are not produced by main's benchmark runner — the miss-log generator lives in `scripts/run_phase3a_ablation.py` on the abandoned branch, which we intentionally did not cherry-pick. For this round the user supplies a miss log from one of:

- **Existing artifacts**: `.worktrees/phase2a/artifacts/phase3a_full/*/misses.jsonl` from the prior Phase 3A runs. Format: JSONL with `query`, `gold_anchor_ids`, `miss_reason` per line. `scripts/refine_anchor_cards.py` consumes this format.
- **A small standalone miss-log extractor** (future work, out of scope). Adding miss-log emission to `psa benchmark longmemeval run` itself is a separate, small change we can pick up next round if the retrain moves R@5.

## Workflow enabled (user-driven, outside this spec)

```bash
# 1. Pick a miss log (existing artifact or a new one via out-of-scope tooling)
MISS_LOG=./.worktrees/phase2a/artifacts/phase3a_full/256_atlas_temporal/misses.jsonl

# 2. Refine cards against the current atlas
psa atlas refine --miss-log "$MISS_LOG"

# 3. Regenerate coactivation training data (atlas now returns refined cards)
psa train --coactivation --regenerate-data

# 4. Retrain selector (existing command)
psa train --selector

# 5. Re-benchmark
psa benchmark longmemeval run
```
