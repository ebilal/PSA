# Memory Forgetting Relative Usage Design

**Date:** 2026-04-20

## Goal

Replace the absolute time-based idle component in memory forgetting with a relative usage-based signal so that memories are not penalized simply because time passed.

The new design should preserve automatic pruning under pressure, but make one product principle explicit:

- dormancy alone is not evidence that a memory stopped being useful

## Current State

`psa/forgetting.py` currently computes a forgetting score with four terms:

- idle pressure: `min(idle_days / 90, 1.0)`
- crowding pressure: `min(overflow / budget, 1.0)`
- usage protection: `min(log(1 + pack_count) / 3.0, 1.0)`
- quality protection: `quality_score`

This has an undesirable property:

- a memory becomes more disposable as wall-clock time passes even if nothing in the system indicates that it became less true, less useful, or less important

That is especially problematic for:

- niche procedural memories
- rare but critical failure memories
- infrequently referenced project facts
- long-lived knowledge on machines that are idle for long periods

## Product Decision

Memory forgetting should not include an absolute elapsed-time penalty.

Instead:

- pruning happens only under pressure
- ranking under pressure is based on relative usage and quality
- time is not used as a direct disposability signal

Pressure still comes from the same two places:

- anchor overflow
- global tenant cap

## Scope

In scope:

- the forgetting formula in `psa/forgetting.py`
- a new `tests/test_forgetting.py` covering score behavior and pruning order
- README updates describing the new forgetting model

Out of scope:

- changing archive TTL for already-archived memories
- redesigning the lifecycle fast path
- redesigning advertisement decay
- changing the global cap or anchor budget defaults

No database migration is required for this work. The necessary telemetry fields already exist on `MemoryObject` and in SQLite:

- `pack_count`
- `select_count`
- `last_packed`
- `last_selected`

## Recommended Approach

Use **anchor-relative usage pressure** as the replacement for the idle-time term.

This means each memory competes primarily with the other memories in its anchor, because that is where pruning pressure is most meaningful.

## Target Behavior

### 1. No absolute time penalty

The forgetting score must not include:

- memory age
- days since last packed
- any direct wall-clock decay term

A memory that has not been used in a year should not become more disposable solely because a year passed.

### 2. Relative usage pressure within the anchor

Each memory should receive a usage-pressure value based on its standing within the anchor’s usage distribution.

Primary signal:

- `pack_count`

V1 decision:

- `select_count` is not a separate protection term in V1
- `select_count` is used only as a deterministic tie-breaker when `pack_count` is equal

V1 pressure must use a percentile-based rule with an explicit definition:

- memories near the bottom of the anchor’s usage distribution get higher pressure
- memories near the top get little or no usage pressure
- ties are resolved deterministically

### 3. Crowding remains the trigger

The system should still only archive active memories when one of these is true:

- an anchor is over its budget
- the tenant is over the global cap

This preserves the operational model that forgetting is pressure-driven, not age-driven.

### 4. Quality still protects memories

`quality_score` remains part of the score so memories judged stronger at ingestion retain protection even if they are not used often.

### 5. New-memory grace remains

The existing 24-hour absolute grace period should remain unchanged.

That grace is not an age-based forgetting penalty. It is a protection window for newly-created memories before the system has any opportunity to observe their value.

## Detailed Design

### A. Replace idle pressure with low-usage pressure

Current term:

```text
+ min(idle_days / 90, 1.0)
```

Replacement concept:

```text
+ low_usage_pressure(memory, anchor_peer_usage_distribution)
```

V1 shape:

- compute a usage key for each memory:
  - primary: `pack_count`
  - secondary tie-break: `select_count`
  - tertiary tie-break: `quality_score` ascending
  - final deterministic tie-break: `created_at` ascending (older is more disposable)
- rank memories within the anchor by that usage key
- assign higher disposal pressure to lower-ranked memories

Exact percentile formulation:

- if anchor size `< 5`, `low_usage_pressure_within_anchor = 0.0`
- if all memories in the anchor have `pack_count == 0` and `select_count == 0`, `low_usage_pressure_within_anchor = 0.0`
- otherwise, let `N` be the number of active memories in the anchor and `rank` be the zero-based index after ascending sort by the usage key above
- pressure is:

```text
low_usage_pressure_within_anchor = 1.0 - (rank / (N - 1))
```

This yields:

- `1.0` for the least-used memory
- `0.0` for the most-used memory
- a linear interpolation for the memories in between

This keeps the signal relative and local instead of absolute and time-driven.

### B. Keep the formula simple

The replacement should stay operationally simple and interpretable.

Recommended score shape:

```text
forgetting_score =
  + low_usage_pressure_within_anchor
  + crowding_pressure
  - usage_protection
  - quality_protection
```

Where:

- `low_usage_pressure_within_anchor` identifies locally underused memories
- `usage_protection` remains an explicit positive-retention term derived from real usage

Recommended protection term:

- keep the existing `pack_count` protection term, or a close equivalent, so heavily used memories remain strongly protected under overflow

V1 decision:

- keep the existing `pack_count` protection term unchanged:

```text
usage_protection = min(log(1 + pack_count) / 3.0, 1.0)
```

Not recommended for the first revision:

- tenant-wide normalization as the primary signal
- recency weighting
- time-decayed counters
- learned forgetting models

### C. Use anchor-relative pressure for per-anchor pruning and a comparable hybrid score for global-cap pruning

There are two pruning modes today:

- per-anchor pruning
- global cap enforcement

For per-anchor pruning, anchor-relative usage pressure is the correct primary signal.

Reason:

- it keeps pruning consistent with the local semantic neighborhood
- it avoids letting very busy anchors distort the importance ranking of quiet anchors

For global-cap enforcement, scores must remain comparable across anchors. The recommended approach is a hybrid with fixed weights:

- keep anchor-relative low-usage pressure as the primary component
- keep usage protection as an explicit term
- compute a weaker tenant-wide low-usage term for cross-anchor comparability

Product rule:

- global-cap enforcement must not rank memories solely by anchor-local percentile, because that makes cross-anchor comparisons unstable

Recommended shape for global-cap pruning:

```text
global_forgetting_score =
  + 0.7 * anchor_relative_low_usage_pressure
  + 0.3 * tenant_wide_low_usage_pressure
  + crowding_pressure
  - usage_protection
  - quality_protection
```

The tenant-wide term is weaker by design. Its purpose is comparability, not replacing the local signal.

Exact tenant-wide pressure formulation follows the same ranking rule as the anchor-local one:

- compute the same usage key over all active tenant memories
- if total active memories `< 5`, tenant-wide pressure is `0.0`
- if all active tenant memories have `pack_count == 0` and `select_count == 0`, tenant-wide pressure is `0.0`
- otherwise:

```text
tenant_wide_low_usage_pressure = 1.0 - (tenant_rank / (tenant_N - 1))
```

Implementation note:

- this requires building per-anchor peer groups and one tenant-wide peer group during `enforce_global_cap`
- the current `max_memories // 256` target hack should be removed from global-cap ranking and replaced with `ANCHOR_MEMORY_BUDGET` for crowding pressure consistency

### D. Archive-first behavior stays

The archive lifecycle remains unchanged:

- active memories may be archived under pressure
- archived memories may later be hard-deleted by archived TTL

This design changes how memories become archive candidates, not what happens after they are archived.

### E. Small-anchor stability rule

Very small anchors do not have enough distributional structure for a noisy percentile-only rule.

Implementation constraint:

- anchors with fewer than 5 active memories must not use raw percentile pressure alone

Recommended handling:

- if anchor size < 5, set `low_usage_pressure_within_anchor` to `0`
- rely on crowding pressure, explicit usage protection, and quality protection
- if a tie-break is still needed, use absolute usage counts (`pack_count`, then `select_count`) rather than percentile rank

This keeps pruning stable when anchors are small and avoids arbitrary ordering caused by tiny local samples.

This rule applies based on the memory’s own anchor size even during global-cap enforcement.

## Policy Constraints

These are fixed constraints for implementation:

- dormancy alone must not increase forgetting pressure
- elapsed wall-clock time must not appear in the forgetting score
- pruning must remain pressure-driven
- relative usage must be computed primarily within the anchor
- explicit usage protection must remain in the score
- global-cap ranking must use a cross-anchor comparable score
- anchors with fewer than 5 active memories must not rely on raw percentile-only pressure
- `quality_score` remains protective
- the 24-hour new-memory grace period remains

## Testing Strategy

Required coverage in a new file `tests/test_forgetting.py`:

- a memory does not become more disposable solely because time passes
- a low-usage memory in an overfull anchor is ranked as more disposable than a high-usage peer
- two equally old memories with different usage levels are ordered by usage, not age
- a heavily used memory remains protected under overflow relative to weakly used peers
- high-quality low-usage memories retain more protection than low-quality low-usage memories
- per-anchor pruning still archives the worst local candidates first
- global-cap enforcement still archives memories when over cap, using a cross-anchor comparable scoring model
- anchors with fewer than 5 active memories do not exhibit unstable percentile-driven ordering
- an all-zero usage anchor produces `0.0` low-usage pressure for every memory
- score bounds remain sane after the formula change

Regression coverage to retain:

- new memories remain protected for 24 hours
- archive-first / hard-delete-later semantics remain unchanged
- no pruning occurs when anchors are within budget and tenant is within cap

Expected concrete tests for V1:

1. `test_age_alone_does_not_raise_score`
2. `test_low_usage_memory_ranked_above_high_usage_peer`
3. `test_equal_age_different_usage_ordered_by_usage`
4. `test_heavily_used_memory_protected_under_overflow`
5. `test_high_quality_low_usage_memory_beats_low_quality_peer`
6. `test_prune_anchor_archives_local_worst_candidates`
7. `test_global_cap_uses_hybrid_cross_anchor_score`
8. `test_small_anchor_pressure_is_zero`
9. `test_all_zero_usage_anchor_pressure_is_zero`
10. `test_grace_period_preserved`
11. `test_archive_first_semantics_preserved`
12. `test_score_range_bounds`

## Risks

### 1. Under-pruning cold anchors

Removing the time term may make quiet anchors retain weak memories longer.

This is acceptable as long as pruning remains effective under actual pressure.

### 2. Overfitting to `pack_count`

If `pack_count` is the only usage signal, some useful memories may still look weak when they are selected but not packed often.

This is why `select_count` may be useful as a future secondary signal, but it should not block the first revision.

### 3. Cross-anchor comparability

Global-cap pruning can become unfair if scores are not comparable across anchors.

This is why the design requires a hybrid score for global-cap enforcement rather than pure anchor-local percentile ranking.

### 4. Distribution edge cases

Very small anchors may have weak usage distributions.

This is why the design fixes a small-anchor rule instead of leaving that behavior implicit.

## Rollout

This change affects active pruning behavior for running tenants, so it needs an observation window before the old score is removed entirely.

Concrete rollout for V1:

1. Implement the new scoring path.
2. Add temporary shadow logging in `prune_anchor` and `enforce_global_cap` that records old-score vs new-score divergence without changing archive semantics for that log line.
3. Run `psa lifecycle run` against the default tenant for **3 lifecycle runs** and inspect the divergence.
4. If divergence is explainable and acceptable, keep the new score active and remove the temporary shadow log in the follow-up change.

This is not a feature flag rollout. It is a short observation window with explicit logging.

Recommended shadow log payload:

- tenant id
- anchor id
- memory id
- old score
- new score
- archive decision
- rank position within the batch

## Documentation

README sections that must be updated:

- the forgetting formula block in [README.md](/Users/erhanbilal/Work/Projects/memnexus/README.md:402)
- the paragraph immediately after it describing score range and pruning semantics

No change is required to lifecycle orchestration sections unless wording still implies time-based pruning.

## Success Criteria

This work is complete when:

- memory forgetting no longer contains an absolute time-decay term
- pruning remains automatic under anchor/global pressure
- low relative usage, not elapsed time, is the primary disposal signal
- explicit usage protection still prevents obviously valuable memories from being pruned too aggressively
- README explains the new model clearly
- tests prove dormancy alone does not increase forgetting pressure
- no DB migration is introduced
