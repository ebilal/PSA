# Advertisement Decay Unification Design

**Date:** 2026-04-20

## Goal

Replace the current dual-path advertisement decay model with a single, conservative, fully automated lifecycle-integrated mechanism that is enabled by default.

After this change, advertisement decay should behave like memory forgetting at the product level:

- one supported mechanism
- no human review or promotion step
- part of `psa lifecycle run`
- enabled by default

The system should remain more conservative than memory forgetting because advertisement patterns are low-cost, intentionally authored strings rather than passively accumulated memories.

## Current State

The repository currently has two advertisement-decay mechanisms:

1. Stage 1, operator-driven
   - `psa atlas decay`
   - `psa atlas promote-refinement`
   - trace-log reinforcement
   - candidate-file workflow
   - optional metadata and pinning concepts

2. Stage 2, lifecycle-integrated
   - `pattern_ledger` SQLite table
   - inline tracking during `psa search`
   - nightly evaluation during `psa lifecycle run`
   - optional direct mutation of `anchor_cards_refined.json`

This split creates three problems:

- conceptual duplication: users must understand two kinds of advertisement decay
- product asymmetry: memories have one lifecycle path while advertisements have two
- implementation coupling: Stage 2 still reaches back into Stage 1 for shielding and pinned-pattern checks

## Product Decision

Advertisement decay will have exactly one supported mechanism:

- the lifecycle-integrated ledger-based path

The operator-driven candidate/promotion path will no longer be part of the supported product surface for advertisement decay.

Specifically:

- `psa lifecycle run` becomes the only supported mutation path for advertisement decay
- `psa search` remains the signal-collection path
- advertisement decay is on by default
- removals are automatic
- conservatism comes from policy thresholds, grace, and floors rather than manual approval

## Scope

This design changes advertisement decay only. It does not redesign memory forgetting, atlas refinement in general, or the selector/retriever pipeline.

In scope:

- advertisement config defaults
- lifecycle behavior
- removal policy conservatism
- CLI and README surface
- tests for default-on automated behavior
- removal of advertisement-specific Stage 1 product paths

Out of scope:

- redesigning memory forgetting heuristics
- removing non-advertisement atlas refinement workflows such as refinement or curation
- changing the trace-first ordering of `psa search`
- changing the physical storage location of `anchor_cards_refined.json`

## Target Behavior

### 1. Default-on automated advertisement decay

Fresh installs and default config should behave as follows:

- advertisement tracking is enabled by default
- advertisement removal is enabled by default
- `trace_queries` remains required because ledger accumulation depends on trace-first success

This must be true in every default source the product exposes:

- the runtime defaults in `AdvertisementDecayConfig`
- the generated `~/.psa/config.json` written by config initialization
- the documented install/config examples in `README.md`

This means a default system will:

- collect advertisement signals during normal searches
- evaluate stale patterns during lifecycle
- remove sufficiently stale patterns automatically during lifecycle

### 2. One removal policy

There should be only one live policy for removal decisions: the ledger-based policy in lifecycle.

The current shadow-policy counterfactual may remain as an observability mechanism, but it must not imply a separate operator-gated rollout model. Its role becomes:

- debugging
- tuning
- safety visibility

It is no longer a prerequisite for enabling automatic removals.

### 3. Conservative by policy, not by workflow

Since advertisements are cheap and intentionally written, the system should bias toward keeping them unless there is sustained evidence they are stale.

A core policy constraint is:

- dormancy alone is not enough to remove a pattern
- lack of recent use is weak evidence, not dispositive evidence
- automated removal requires sustained negative evidence, not just elapsed time
- absence of queries for a niche but valid pattern must bias toward retention

Conservatism should come from:

- longer grace periods
- longer sustained-negative windows
- non-zero minimum pattern floor per anchor
- anchor shielding for weak-signal anchors
- optional pinned-pattern exemptions only if they remain useful and do not require a human promotion workflow

The product guarantee should be:

- stale patterns disappear automatically
- borderline patterns stay longer than memories would
- no human intervention is needed
- long-unused but still plausible patterns are retained unless the system accumulates negative evidence against them

## Detailed Design

### A. Keep Stage 2 architecture as the foundation

The Stage 2 design is already the correct operational shape:

- signal collection inline in `psa search`
- durable ledger state in SQLite
- periodic decay in lifecycle
- direct mutation of the refined card surface

This path should remain and become the sole supported implementation.

Important behavioral constraint:

- exponential decay and age are ranking signals for risk, not sufficient removal conditions by themselves
- actual removal requires the negative-evidence threshold (`sustained_cycles`) to be crossed after grace and floor/shield checks

### B. Remove Stage 1 as an advertisement product feature

Advertisement-specific Stage 1 user-facing commands and documentation should be removed from the supported surface:

- `psa atlas decay`
- advertisement-oriented references to `psa atlas promote-refinement`

Important nuance:

- if `promote-refinement` is still required for non-advertisement refinement flows such as ngram refinement or curation, it should remain
- but it should no longer be presented as part of advertisement decay

The result is not necessarily deleting every Stage 1 source file immediately. The result is removing Stage 1 as a supported advertisement mechanism. Code deletion can be aggressive where safe, but the product surface must be singular.

### C. Decouple lifecycle Stage 2 from Stage 1 helpers

Today `advertisement_decay_pass()` still imports Stage 1 helpers for:

- low-activation shielding
- pinned-pattern metadata lookups

That coupling should be removed.

The lifecycle path should own its own protections directly. Two acceptable implementations:

1. Move the shared protections into neutral advertisement modules used by lifecycle only.
2. Reimplement the small required logic inside Stage 2-specific modules and stop importing from Stage 1.

Recommendation:

- extract or re-home only the protections that remain part of the single supported policy
- do not preserve Stage 1 abstractions just to keep old command flows alive

### D. Keep provenance metadata only if it serves autonomous policy

`pattern_metadata.json` currently exists to support provenance and pinning. It remains part of the single automated policy.

Required rule:

- keep provenance timestamps/source metadata for migration and diagnostics
- keep `pinned` as a narrow expert override, not as part of a review workflow
- do not keep metadata solely to support candidate-file promotion

If the metadata file remains, it becomes passive policy/provenance state, not part of a separate advertisement-decay mode.

### E. More conservative defaults

The new single default-on path should use safer defaults than the current Stage 2 rollout assumed.

Fixed changes:

- `tracking_enabled`: default `true`
- `removal_enabled`: default `true`
- `tau_days`: keep `45`
- `grace_days`: change from `21` to `30`
- `sustained_cycles`: change from `14` to `21`
- `min_patterns_floor`: change from `3` to `5`
- shadow policy: retain for diagnostics only; it does not gate whether automated removal is enabled

These values are part of the product behavior, not implementation-time tuning. The implementation should update code defaults, generated config, tests, and README to match these exact values.

Interpretation requirement:

- `tau_days` and `grace_days` may reduce ledger confidence over time
- but a pattern is not removable solely because enough time passed
- removal still requires sustained negative cycles plus all shared protections passing

### F. Lifecycle semantics

`psa lifecycle run` should describe advertisement decay the same way it describes memory forgetting:

- always part of the fast path
- skipped only if explicitly disabled in config
- no language implying manual sign-off
- summary output focused on observed/removal counts

This keeps the mental model simple:

- memories archive automatically
- stale advertisement patterns prune automatically

### G. CLI and docs cleanup

The command and documentation surface should reflect one supported model.

README changes:

- remove “Stage 1 vs Stage 2” framing from user-facing explanation
- remove calibration-flow guidance
- describe advertisement decay as a single lifecycle-integrated feature
- document conservative default-on behavior

CLI changes:

- remove `psa atlas decay` from help, dispatch, and supported CLI surface
- keep `psa advertisement status|diff|rebuild-ledger|purge` if they remain useful for introspection and maintenance
- update help text for `psa lifecycle run` and config defaults

This is an explicit product decision, not an implementation choice:

- `psa atlas decay` is removed as a user-facing command
- tests dedicated to `psa atlas decay` are removed
- `psa atlas promote-refinement` stays only for non-advertisement refinement flows such as `refine` and `curate`

## Migration

### Existing users

Existing tenants may have one of several states:

- no `advertisement_decay` block at all
- tracking on, removal off
- tracking off
- Stage 1 metadata present
- existing active ledger rows

Migration behavior must be safe and automatic:

- absent config should resolve to the new default-on settings
- if a tenant explicitly disabled advertisement decay, their explicit config should still win
- existing ledger rows should continue to work
- existing metadata should be tolerated
- environment variables are not part of the supported configuration model for this feature; the canonical source of truth is `~/.psa/config.json`

### Supported configuration model

The supported operator-facing configuration surface for advertisement decay is `~/.psa/config.json`.

Implementation requirements:

- all advertisement-decay settings used in normal operation must be represented in `~/.psa/config.json`
- the installation guide in `README.md` must document the full JSON block needed to run the system
- environment-variable overrides for advertisement-decay settings are removed; the JSON config is the only supported configuration surface
- documentation must treat the JSON config as authoritative

### Candidate artifacts

Any old candidate files for advertisement decay become unsupported leftovers. The system should ignore them unless they are still used by other atlas refinement features.

## Testing Strategy

Tests must cover the new product contract, not just preserve old behavior.

Required test coverage:

- config defaults now produce tracking enabled and removal enabled
- config initialization writes the advertisement-decay JSON block with the new default values
- lifecycle applies advertisement decay by default when using default config
- lifecycle still respects explicit config disablement
- removal path does not depend on Stage 1 helper imports
- CLI no longer exposes `psa atlas decay`
- README documents the complete JSON config required for a full install
- README/config docs match the new defaults and single-mechanism model

Regression coverage to retain:

- trace-first ordering for ledger writes
- graceful behavior when trace is disabled but tracking is requested
- min-pattern-floor and shielding protections
- dormancy alone does not create removal candidates
- rebuild-ledger and purge maintenance commands

## Risks

### 1. False removals

Making removal default-on increases the cost of policy mistakes. This is mitigated by conservative thresholds and floors.

### 2. Hidden coupling to Stage 1

There may be tests or code paths that assume Stage 2 reuses Stage 1 modules. The implementation must identify and sever these dependencies explicitly.

### 3. Documentation drift

The current README contains extensive Stage 1 and calibration language. Partial cleanup would leave the system internally simplified but externally confusing.

## Success Criteria

This work is complete when all of the following are true:

- there is one supported advertisement-decay mechanism in the product surface
- that mechanism runs automatically in lifecycle
- it is enabled by default
- it is more conservative than memory forgetting by policy, not by human review
- user-facing docs no longer describe advertisement decay as a staged rollout with manual sign-off
- tests enforce the new default-on and single-mechanism behavior

## Final Decisions

These decisions are fixed for implementation:

- advertisement decay is default-on in runtime defaults, generated config, and README examples
- the supported configuration surface is `~/.psa/config.json`
- `tracking_enabled=true`
- `removal_enabled=true`
- `tau_days=45`
- `grace_days=30`
- `sustained_cycles=21`
- `min_patterns_floor=5`
- shadow-policy reporting remains as diagnostics only
- `pinned` metadata remains as a narrow expert override and is not tied to any human review workflow
- `psa atlas decay` is removed from the supported CLI surface and its dedicated tests are removed
- `psa atlas promote-refinement` remains only for non-advertisement refinement flows
