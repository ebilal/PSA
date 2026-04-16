# Phase 3A Refined Cards — Cleanup Design

**Date:** 2026-04-16
**Branch:** `feat/phase3a-refined-cards` (off `main`)
**Supersedes:** `feature/phase2a-answer-bearing-labels` (abandoned — 69 commits, many dead)

## Goal

Carry forward only the changes that actually moved R@5 past the 0.932 Phase 1 baseline, reaching ~0.954. Leave the rest of `feature/phase2a-answer-bearing-labels` on that branch; do not merge.

## Design philosophy alignment

The winning mechanism is teaching anchors to *carry what they can answer* — `generated_query_patterns` seeded at atlas build, then refined offline from the miss log. Retrieval scores improve because anchors resonate with the query shape, not because we added more rerank machinery. This is the "memory = computation, activation not lookup, similarity ≠ usefulness" direction.

## What comes forward (three commits)

| # | Change | Source | Reason |
|---|---|---|---|
| 1 | `min_k` + `top_ce_budget` params on `CoActivationSelector` | `f50b89a` | Main's pipeline/CLI already pass `min_k`; the selector itself doesn't consume it. `min_k=10` rescues 6 scoring_rank misses — the Phase 1 baseline. |
| 2 | `card_text_override_by_anchor_id` on `FullAtlasScorer` | `f0f17c6` | Lets refined-card runs swap text in a single scorer instance without mutating on-disk anchors. Necessary plumbing for the refinement workflow. |
| 3 | `scripts/refine_anchor_cards.py` + tests | `4c2985d`, `9841a33` | The +0.088 R@5 lever. Offline ngram expansion of `generated_query_patterns` driven by a miss log. |

## What stays dead

Deliberately not cherry-picked:

- **Phase 2A graded labeler** (`psa/training/graded_labeler.py`, 379 lines, Phase 2A). Built to improve Level 2 scoring. R@5 stayed at 0.920.
- **16-dim `MemoryReRanker` expansion + weighted MSE + `supports_constraints` + `constraint_penalty`**. Same program as above; same null result.
- **`ConstraintScorer` beyond what's already on main** — Phase 1's hard-rule scorer earned its keep; the Phase 2A learned-constraint extensions did not.
- **Phase 3B ablation** (`scripts/run_phase3b_selector_ablation.py` + tests). Conclusively showed `min_k`/`max_k` tuning has zero effect past min_k=10 — research artifact.
- **Phase 3A ablation runner** (`scripts/run_phase3a_ablation.py` + tests). One-shot research harness; its finding is the refinement script, which is coming forward.
- **Phase 2 temporal metadata** (`source_event_at` threading, `date_earliest/latest/months` on `AnchorCard`, time-span card text). +0.006 R@5 standalone; the refinement already absorbs date signal through the query patterns it surfaces.

## What's already on main (and stays)

Confirmed present:
- `AnchorCard.generated_query_patterns` + `query_fingerprint`
- `atlas.py` Qwen prompt extracts `query_patterns`
- `psa/fingerprints.py` (FingerprintStore)
- `psa/synthesizer.py` (AnchorSynthesizer) wired into `pipeline.py`
- `min_k` plumbing through `pipeline.py`, `cli.py`, `benchmarks/longmemeval.py`
- Phase 1 query frames + facets + ConstraintScorer

## Non-goals

- No new model training in this cleanup.
- No new feature work — that lands on a follow-up branch.
- No rebuild of atlas in this branch — the refinement is offline against any existing atlas.

## Verification

- Full test suite passes: `uv run pytest tests/ -v`
- Lint clean: `uv run ruff check . && uv run ruff format --check .`
- R@5 reproduction is out of scope for this branch (expensive; the previous Phase 3A/3B artifacts already establish it). Benchmarking happens separately when needed.
