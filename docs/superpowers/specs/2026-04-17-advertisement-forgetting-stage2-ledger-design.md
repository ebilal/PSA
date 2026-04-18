# Advertisement Forgetting Stage 2 — Pattern-Level Signal Ledger

**Date:** 2026-04-17
**Status:** Design
**Builds on:** `docs/superpowers/specs/2026-04-17-advertisement-forgetting-design.md` (shipped)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the shipped advertisement-forgetting system with a pattern-level signal ledger that captures per-query attribution (retrieval, selector pick, selector decline) and supports automated nightly decay gated by a tracking/removal split and a shadow-policy safety net.

**Architecture:** A new `pattern_ledger` SQLite sidecar table materialized from per-query inline writes. Attribution is BM25 argmax (with ε-tie split and a BM25 contribution floor) computed at query time. A nightly lifecycle pass applies exponential decay and a sustained-negative-cycles removal rule. A shadow policy (more aggressive knob set) runs in parallel columns for counterfactual calibration without altering live behaviour. The shipped candidate/promote flow (`psa atlas decay`) stays in place for operator-gated removal; the stage 2 path is a complementary automated lane.

**Tech Stack:** SQLite (tenant store), existing `AnchorRetriever` BM25 scoring for attribution, `MempalaceConfig` + env overrides for tuning, existing `psa/trace.py` schema (minor optional augmentation).

---

## Relationship to implemented advertisement-forgetting

Stage 1 (shipped in `psa/advertisement/`) established the first advertisement-fading mechanism:

- **Provenance store**: `pattern_metadata.json` sibling file per atlas (`source`, `created_at`, optional `pinned`).
- **Ephemeral reinforcement**: per-run dict derived from the query trace log via substring rule (R1).
- **Decay rule D1**: grace period + unreinforced-window. Binary: "reinforced within window or not."
- **Protections**: P1 low-activation anchor shield, P3 operator-pinned exemption.
- **Candidate/promote gate**: `psa atlas decay` writes `anchor_cards_candidate.json` + summary + detail report; operator promotes via `psa atlas promote-refinement`.
- **Atlas rebuild inheritance**: metadata copy-forward for matched `(anchor_id, normalized_pattern)` pairs.

**Stage 2 does not replace any of that.** Everything listed above stays, unchanged.

### What stays

- `pattern_metadata.json` remains the provenance record of truth.
- `psa atlas decay` remains the operator-driven, candidate-gated removal path.
- R1 substring reinforcement remains the signal used by `psa atlas decay`.
- P1 and P3 protections apply to both stage 1 and stage 2 removal paths.
- Atlas rebuild metadata inheritance is unchanged.

### What stage 2 adds

- **Pattern-level graded signals** captured per query at the moment retrieval and selection happen: retrieval hit (+), selector pick (+), selector decline (−). Stage 1 only has a binary "reinforced in window."
- **`pattern_ledger` SQLite table**: persistent per-pattern ledger scalar + shadow-policy columns + `consecutive_negative_cycles` state machine + `grace_expires_at`.
- **Automated nightly decay pass** in `lifecycle.py` fast path. Applies exponential decay, evaluates the sustained-negative-cycles removal rule, and either reports-only or removes depending on the `tracking_enabled`/`removal_enabled` split.
- **Shadow policy**: a more-aggressive knob set logged in parallel columns so B-vs-A disagreement rate is observable before any knob is tuned.
- **`psa advertisement status` / `psa advertisement diff`** CLIs for inspecting ledger distribution and shadow-policy counterfactuals.
- **Coexistence with candidate/promote flow**: when `removal_enabled=false`, stage 2's decisions surface in the status CLI and lifecycle logs but take no destructive action. When `removal_enabled=true`, stage 2 removes patterns directly from the live `anchor_cards.json` and soft-archives the ledger row. Operators can still run `psa atlas decay` at any time for candidate-gated batch review.

### Why persistent ledger + inline lifecycle logic is justified

The shipped design's load-bearing architectural principle is: **"reinforcement is ephemeral. Re-derive every run. Trace is the source of truth; no persistent counter that can drift."** This section engages that principle directly.

**Claim:** the stage 2 ledger does not violate the principle. It is a *materialised view* of events the trace already records (plus a small augmentation, see §2.2), with a canonical `rebuild-ledger` regenerator that reconstructs it from trace. Trace remains the source of truth; the ledger is a cache.

**Four reasons the materialised view is worth the code:**

1. **Exponential-weighted accumulation is expensive to re-derive on demand.** Stage 1's R1 rule is binary over a window — "was there any reinforcement in the last 60 days?" — which is cheap to recompute. Stage 2's arithmetic is an exponentially-decayed weighted sum over the full lifetime of the pattern with three distinct weights per event class. Walking years of trace records every time the status CLI is invoked is wasteful when the same number is sitting in a row.

2. **Attribution requires the retriever's runtime state.** BM25 argmax for a query is computed at query time using exactly the scores the retriever saw. Reconstructing it after the fact requires re-running BM25 against the query with the current pattern list — but the pattern list may have changed since the query was recorded (patterns added by curate/refine, removed by decay). Capturing argmax inline avoids the "attribute against which atlas version?" question. The trace schema augmentation in §2.2 records the argmax at query time so reconstruction remains possible, but the *primary* write path is inline.

3. **The sustained-negative-cycles state machine is genuinely time-serial.** "Has ledger been below threshold for N consecutive nightly evaluations?" is a property of the nightly evaluation cadence, not the query stream. Deriving it requires replaying every nightly eval point since pattern creation. Storing the counter as persistent state is the simplest correct representation of a state machine whose input is a scheduled nightly trigger.

4. **Shadow policy needs parallel state.** Computing both B (live) and A (shadow) ledgers from scratch on every inspection doubles the cost. More importantly, the whole point of the shadow column is to make "does A disagree with B" a simple read, not a multi-minute derivation.

**Drift mitigations:**

- Trace remains the source of truth for events. The ledger never gains signals not present in the trace.
- `psa advertisement rebuild-ledger` recomputes the ledger from trace + augmentation fields. Canonical regenerator; if drift is ever suspected, run this.
- On atlas rebuild, the ledger is archived wholesale and rebuilt from the new atlas's fresh patterns (§8). This bounds the drift window to one atlas version.
- The ledger is scoped to `generated_query_patterns` only. It does not hold or derive metadata (`source`, `created_at`, `pinned`) — those remain in `pattern_metadata.json`, single-source.

**What stage 2 deliberately does NOT do:**

- Does not change `pattern_metadata.json`'s contents or ownership.
- Does not change the R1 substring rule used by `psa atlas decay`.
- Does not change the candidate/promote gate for operator-driven batches.
- Does not compete with stage 1 for which-removes-first. If a pattern is removed by either path, the other path no-ops on it on the next run (stage 2 sees the pattern gone from the card; stage 1 sees the same).
- Does not introduce a second provenance store. The ledger does not record `source` — that's always read from `pattern_metadata.json`.

## Problem

Stage 1 is conservative and candidate-gated by design. After operating it for a period, two concrete gaps remain:

1. **Binary reinforcement is lossy.** R1 says "was there any substring match in the last 60 days?" A pattern that never wins a selector choice but brushes BM25 weakly looks identical to a pattern that wins selector picks every day. Stage 2's graded signal distinguishes these: an ad that gets attention but is always declined (clickbait) should decay faster than one that is simply niche.

2. **Operator-only removal doesn't scale.** `psa atlas decay` requires a human in the loop. For tenants with hundreds of anchors and thousands of patterns, batched operator review is the right *audit* surface but a bad *continuous* cleanup surface. Automated decay under a tracking/removal split, with shadow-policy observation before any knob moves, is the scalable path.

Stage 2 targets these two gaps without altering stage 1's surfaces.

## Goals

1. **Graded, pattern-specific signal** captured at the moment of retrieval/selection, attributed via BM25 argmax so credit lands on "the ad that earned this query's lexical attention."
2. **Persistent ledger** as a materialised view over inline-written events, scoped narrowly to dynamic decay arithmetic + shadow state.
3. **Tracking vs removal split** so the first deployment writes the ledger without acting on it. Flip removal on only after the shadow disagreement rate is well-characterised.
4. **Shadow policy** logged in parallel columns. Primary: B-vs-A disagreement rate over the first calibration window.
5. **Coexistence** with the shipped candidate/promote flow. Operators can still run `psa atlas decay` at any time.
6. **Configurable** via `MempalaceConfig` + env overrides. No separate `.env` file.
7. **Observable** via minimum-viable CLIs (`status`, `diff`) and nightly lifecycle log summaries.

## Non-goals

- **Dense/semantic attribution.** Stage 2 uses BM25 argmax only. Dense cosine per pattern (would require per-pattern embeddings) is a stage 3 consideration if substring-family attribution proves too lexical.
- **Replacing stage 1.** Stage 1 remains the operator-audit surface.
- **Carry-over of ledger state across atlas rebuilds.** Fresh slate at rebuild; matches the shipped behaviour that regenerates the candidate surface at rebuild time.
- **Ledger term in memory forgetting.** `psa/forgetting.py` is memory-level and unrelated.
- **Automatic tuning of weights/half-life.** Knobs are config; calibration is operator-driven using `psa advertisement diff` output.

---

## §1 — Data model: `pattern_ledger` table

New table in the tenant SQLite (same DB as `memory_objects`). No changes to existing tables.

```sql
CREATE TABLE pattern_ledger (
    pattern_id                     TEXT PRIMARY KEY,  -- hash(anchor_id + normalize_pattern(text))
    anchor_id                      INTEGER NOT NULL,
    pattern_text                   TEXT NOT NULL,     -- raw text, for debugging / reverse lookup

    -- primary (B) policy state
    ledger                         REAL NOT NULL DEFAULT 0.0,
    consecutive_negative_cycles    INTEGER NOT NULL DEFAULT 0,

    -- shadow (A) policy state
    shadow_ledger                  REAL NOT NULL DEFAULT 0.0,
    shadow_consecutive             INTEGER NOT NULL DEFAULT 0,

    -- lifecycle
    grace_expires_at               TEXT NOT NULL,     -- ISO8601; set at insert = now + grace_days
    created_at                     TEXT NOT NULL,
    last_updated_at                TEXT NOT NULL,

    -- removal / archival
    removed_at                     TEXT,              -- NULL while active; set at removal
    removal_reason                 TEXT,              -- NULL while active; e.g. "stage2_sustained_negative"
    final_ledger                   REAL,              -- snapshot at removal
    final_shadow_ledger            REAL
);

CREATE INDEX idx_pattern_ledger_anchor ON pattern_ledger(anchor_id);
CREATE INDEX idx_pattern_ledger_active ON pattern_ledger(anchor_id) WHERE removed_at IS NULL;
```

**Key design choices:**

- `pattern_id` is a content hash of `(anchor_id, normalized_pattern_text)`. Stable across restarts, changes only if the pattern text changes (which at atlas rebuild is fine — a reworded pattern is a new ad). Matches the metadata_key convention in stage 1.
- `pattern_text` is carried for human inspection. A single SELECT against the table is debuggable without joins to metadata or anchor cards.
- `grace_expires_at` is stamped at insert rather than derived from `created_at + grace_days`. Decouples per-row grace from config changes — tuning `grace_days` down doesn't shorten the grace of already-inserted rows.
- Archival is soft: `removed_at` is set, row is retained for analysis. Active-row index filters out archived rows for hot-path scans. A periodic `purge_archived_ledgers(older_than_days=90)` pass hard-deletes rows past retention, matching memory forgetting's 90-day cadence.
- `final_ledger` / `final_shadow_ledger` snapshot state at removal. Enables postmortem queries like "of the N patterns we removed in March, how many did the shadow policy disagree with?"

## §2 — Attribution: BM25 argmax with ε-split + BM25 floor

### §2.1 — Attribution rule

For each query, after retrieval produces the top-24 anchor list and before selector runs:

1. For each retrieved anchor A:
   - Compute BM25 score against the query for every pattern in A's `generated_query_patterns`.
   - Identify `argmax` = the pattern with the highest BM25 score.
   - Identify `ε_tied` = all patterns whose BM25 score is within `epsilon` of `argmax`'s score.
2. **BM25 contribution floor**: require that A was present on the BM25-side shortlist (top-K_bm25 before RRF fusion, K_bm25 = 48 by default). If A got into top-24 purely via dense similarity with negligible BM25 contribution, **no pattern is credited for this query**. This preserves the "best explains the *lexical* retrieval evidence" framing.

Credited set for anchor A = `[argmax] + ε_tied` if A passes the BM25 floor; otherwise `[]`.

### §2.2 — Trace schema augmentation (additive, optional)

For reconstructability, add one optional field per retrieved anchor in the trace record:

```json
{
  "timestamp": "...",
  "query": "...",
  "selected_anchor_ids": [...],
  "top_anchor_scores": [...],
  "retrieval_attribution": [
    {
      "anchor_id": 227,
      "bm25_argmax_pattern_id": "227::what-are-some-quick-healthy-lunch-ideas",
      "bm25_argmax_score": 4.31,
      "bm25_epsilon_tied_ids": ["227::quick-healthy-lunch-recipes"],
      "bm25_contribution_rank": 12
    }
  ]
}
```

This is additive — readers that don't know about `retrieval_attribution` ignore it. `rebuild-ledger` (§13) uses these fields to reconstruct the ledger exactly. Tracing remains an append-only JSONL file; no schema migration.

If the augmentation field is absent (older traces or trace disabled), `rebuild-ledger` falls back to recomputing BM25 attribution from the query text against the *current* pattern list — lossy if patterns have changed, acceptable for a regenerator of last resort.

## §3 — Reinforcement signals

Applied to each pattern in the credited set (§2.1) for each query. Credit divided by credited-set size:

| Event | Weight | Applied when |
|---|---|---|
| Retrieval hit | +1.0 | Anchor was in top-24 AND credited set is non-empty |
| Selector pick | +2.0 (additional) | Anchor was in `selected_anchor_ids` |
| Selector decline | −0.25 (flat) | Anchor was in credited set but not in `selected_anchor_ids` |

Per-credited-set division: if argmax had no ε-tied peers, the full weight goes to argmax. If two patterns ε-tie, each gets half. Prevents anchor-level wins from bleeding into patterns that didn't actually carry the lexical signal.

**Shadow policy uses the same events with different weights** (see §5).

## §4 — Decay rule

### §4.1 — Nightly decay + evaluation

Once per lifecycle fast-path run:

```python
# For each active (non-archived) ledger row:
ledger          *= exp(-1 / tau_days)
shadow_ledger   *= exp(-1 / tau_days)

if ledger < removal_threshold:           # default 0.0
    consecutive_negative_cycles += 1
else:
    consecutive_negative_cycles = 0

# same for shadow_ledger → shadow_consecutive
```

### §4.2 — Removal criteria

A pattern is eligible for stage 2 removal when **all** hold:

- `grace_expires_at <= now`
- `consecutive_negative_cycles >= sustained_cycles`   (default 14)
- Anchor would retain `>= min_patterns_floor` patterns after removal (default 3)
- Pattern is not pinned in `pattern_metadata.json` (P3 from stage 1 applies)
- Anchor is not shielded by stage 1 P1 rule (low-activation shield re-uses stage 1 logic; see §10)

If `removal_enabled=false`, the decay pass computes eligibility and records it in the lifecycle log but takes no action.

### §4.3 — Default parameters (B, primary policy)

```
retrieval_credit        = 1.0
selector_pick_credit    = 2.0
selector_decline_penalty= 0.25
tau_days                = 45
grace_days              = 21
removal_threshold       = 0.0
sustained_cycles        = 14
min_patterns_floor      = 3
```

These are deliberately conservative. First deployment should prune little; the shadow disagreement rate tells us which knob to move.

## §5 — Shadow policy

Parallel columns (`shadow_ledger`, `shadow_consecutive`) track a more-aggressive (A) policy. Same event stream, different weights:

```
shadow.selector_decline_penalty = 0.5   (double B)
shadow.sustained_cycles         = 7     (half B)
```

Other weights (retrieval_credit, selector_pick_credit) are identical to B — the shadow isolates the two knobs we're least sure about.

Shadow decisions are never acted upon. They are surfaced via:
- `psa advertisement diff` — anchors and patterns where B would keep but A would remove (or vice versa).
- Lifecycle log per-run counts: `N_would_remove_under_A` vs `N_actually_removed_under_B`.

Primary calibration metric during the first deployment window: **shadow agreement rate** (fraction of removal decisions where A and B agree). High agreement = conservative-safe to tighten. Disagreement concentrated in a specific source/frame = signal for targeted tuning.

## §6 — Inline write path

New module `psa/advertisement/ledger.py`. Called from `pipeline.py` after selector returns and after the trace record is finalised.

```python
def record_query_signals(
    tenant_id: str,
    atlas,
    query: str,
    retrieved_anchor_ids: list[int],
    selected_anchor_ids: set[int],
    bm25_topk_anchor_ids: set[int],   # anchors on BM25 shortlist before RRF
    config: AdvertisementDecayConfig,
) -> None:
    if not config.tracking_enabled:
        return

    for anchor_id in retrieved_anchor_ids:
        # BM25 contribution floor
        if anchor_id not in bm25_topk_anchor_ids:
            continue

        anchor = atlas.get_anchor(anchor_id)
        argmax, eps_tied = attribute_bm25_argmax(
            query, anchor.generated_query_patterns, epsilon=config.epsilon
        )
        if argmax is None:
            continue

        credited = [argmax] + eps_tied
        per_pattern_weight = 1.0 / len(credited)

        base_credit = config.retrieval_credit
        if anchor_id in selected_anchor_ids:
            base_credit += config.selector_pick_credit
        else:
            base_credit -= config.selector_decline_penalty

        shadow_credit = config.retrieval_credit
        if anchor_id in selected_anchor_ids:
            shadow_credit += config.selector_pick_credit
        else:
            shadow_credit -= config.shadow.selector_decline_penalty

        for pattern in credited:
            upsert_ledger(
                tenant_id,
                pattern_id=pattern_id_for(anchor_id, pattern),
                anchor_id=anchor_id,
                pattern_text=pattern,
                ledger_delta=base_credit * per_pattern_weight,
                shadow_delta=shadow_credit * per_pattern_weight,
                grace_days=config.grace_days,   # only used on INSERT
            )
```

**Ordering constraint:** `record_query_signals` runs after `selected_anchor_ids` is known and before the trace record is written. In `pipeline.py`, that is between the current selector call and the `trace.write_trace(...)` call.

**Performance:** one UPSERT per credited pattern per query. Typical query with 24 retrieved anchors × ~1 argmax each = ≤24 UPSERTs. On a laptop-local SQLite this is ~1–3 ms.

**Failure policy:** the ledger write is best-effort. A write exception logs a warning and returns — it must never break a live query (matches `trace.write_trace`'s reliability contract).

## §7 — Lifecycle integration

New step in `psa/lifecycle.py` fast-path, between "mine new sessions" and "prune memories":

```
advertisement_decay_pass(tenant_id, config):
    if not config.tracking_enabled:
        return

    now = utc_now()
    active_rows = select_active_ledger_rows(tenant_id)

    # §4.1 — decay and update counters
    for row in active_rows:
        apply_decay_and_update_counters(row, tau=config.tau_days, now=now)
    bulk_update(active_rows)

    # §4.2 — evaluate removal
    remove_candidates, shadow_candidates = evaluate_removal(
        active_rows, config, atlas=load_atlas(tenant_id), metadata=load_metadata(...)
    )

    log_summary(
        n_decayed=len(active_rows),
        n_at_risk=count_at_risk(active_rows, config),
        n_would_remove_under_A=len(shadow_candidates),
        n_actually_removed_under_B=len(remove_candidates) if config.removal_enabled else 0,
    )

    if not config.removal_enabled:
        return

    apply_removals(tenant_id, remove_candidates, atlas)
```

**Evaluation order honours stage 1 protections:**

1. Drop any `pinned` patterns (P3).
2. Drop any patterns whose anchor is shielded by P1 (low-activation).
3. Of what remains, require `grace_expires_at <= now`, `consecutive_negative_cycles >= sustained_cycles`, and the min-patterns-floor.

Stage 1's P1 logic is shared by importing `psa.advertisement.decay.shielded_anchors(...)` — do not re-implement.

## §8 — Atlas rebuild integration

Fresh slate: when `AtlasManager.rebuild()` produces a new atlas version, the ledger is handled in a single pass after the new atlas is built.

```python
def reset_ledger_on_rebuild(tenant_id, old_atlas_version, new_atlas):
    # 1. Archive every active row for the old atlas version.
    archive_all_active(
        tenant_id,
        reason="atlas_rebuild_reset",
        final_snapshot=True,
    )

    # 2. Insert fresh rows for every pattern in the new atlas.
    for card in new_atlas.cards:
        for pattern in card.generated_query_patterns:
            insert_fresh_row(
                tenant_id,
                pattern_id=pattern_id_for(card.anchor_id, pattern),
                anchor_id=card.anchor_id,
                pattern_text=pattern,
                grace_expires_at=now + timedelta(days=config.grace_days),
            )
```

- **Archival is wholesale**: every active row gets `removed_at=now, removal_reason="atlas_rebuild_reset"`. Preserves postmortem data without cluttering live scans.
- **Fresh insertion** covers the new atlas's patterns, including unchanged ones. They get fresh grace because the retrieval surface has been regenerated (new pattern set, possibly new BM25 index state, possibly reassigned anchor IDs). Stage 2 starts from zero knowledge per version.
- **Orphan handling**: archived rows for patterns that didn't survive rebuild stay archived; no special handling needed beyond the 90-day purge.

**Hook location:** `AtlasManager.rebuild()` after the metadata-inheritance pass (§1.1 of stage 1 spec). Ledger reset happens AFTER metadata is copied forward so both systems see a consistent rebuild event.

## §9 — Removal mechanics

When `apply_removals` runs:

1. For each candidate pattern:
   - Remove the pattern string from `anchor.generated_query_patterns` in memory.
   - Persist the updated `AnchorCard` via the existing atlas persistence path (`anchor_cards.json` write).
   - Soft-archive the ledger row: set `removed_at=now`, `removal_reason="stage2_sustained_negative"`, `final_ledger`, `final_shadow_ledger`.
   - Log a structured removal entry: `{pattern_id, anchor_id, pattern_text, final_ledger, consecutive_negative_cycles, shadow_agreed}`.

2. **BM25 re-index timing:** lazy at next query. `AnchorRetriever` detects that the anchor's pattern set changed (by hash or mtime) and re-tokenizes on read. No lifecycle-time reindex cost.

3. **Metadata hygiene:** removed patterns are NOT deleted from `pattern_metadata.json`. The entry becomes orphaned next atlas rebuild (§1.1 inheritance drops orphans naturally). Between rebuilds, the orphan is harmless.

4. **Hard delete:** a separate step (`purge_archived_ledgers`) runs weekly, deleting rows with `removed_at < now - 90d`. Matches memory forgetting cadence. Exposed as `psa advertisement purge --older-than 90d`.

## §10 — Coexistence with `psa atlas decay`

Both removal paths operate on the same `generated_query_patterns` list. Coexistence rules:

1. **No locking.** The two paths are not concurrent — `psa atlas decay` is operator-invoked and `lifecycle.py` is a nightly scheduled job; they don't race in practice.
2. **Stage 1 removes via candidate promotion.** Operator runs `psa atlas decay` → writes `anchor_cards_candidate.json` → reviews → `psa atlas promote-refinement`. Nothing touches live cards until promotion.
3. **Stage 2 removes directly.** When `removal_enabled=true`, stage 2's removal mutates `anchor_cards.json` in place.
4. **Stage 2 respects stage 1 protections.** P1 shield and P3 pin both apply to stage 2 candidate selection (imported from `psa/advertisement/decay.py`).
5. **Stage 1 R1 reinforcement is unaffected by stage 2's ledger.** Stage 1 continues to derive last-reinforced-at from trace; it does not read the ledger.
6. **Stage 2 ignores patterns already removed.** If a pattern was dropped by stage 1 operator promotion, it's no longer in `generated_query_patterns`, so the ledger row for it becomes orphaned and gets archived on the next atlas rebuild.

**Operator override:** if a stage-2-removed pattern shouldn't have been dropped, the operator can:
- Re-add it via manual edit of `anchor_cards.json` (it becomes `source="manual"` next time metadata is inspected).
- Pin it via `pattern_metadata.json` (stage 1's P3 flag) so stage 2 never removes it again.

## §11 — Configuration

Extends `MempalaceConfig` with a new `advertisement_decay` section:

```json
{
  "advertisement_decay": {
    "tracking_enabled": false,
    "removal_enabled": false,

    "retrieval_credit": 1.0,
    "selector_pick_credit": 2.0,
    "selector_decline_penalty": 0.25,
    "tau_days": 45,
    "grace_days": 21,
    "removal_threshold": 0.0,
    "sustained_cycles": 14,
    "min_patterns_floor": 3,
    "epsilon": 0.05,
    "bm25_topk_floor": 48,

    "shadow": {
      "selector_decline_penalty": 0.5,
      "sustained_cycles": 7
    }
  }
}
```

**Defaults:** `tracking_enabled=false`, `removal_enabled=false`. First deployment is a no-op until operator flips `tracking_enabled=true`. Flip `removal_enabled=true` only after one full calibration window with shadow data.

**Env overrides** (additive to existing pattern):

```
PSA_AD_DECAY_TRACKING_ENABLED=1
PSA_AD_DECAY_REMOVAL_ENABLED=1
PSA_AD_DECAY_TAU_DAYS=45
PSA_AD_DECAY_GRACE_DAYS=21
PSA_AD_DECAY_DECLINE_PENALTY=0.25
PSA_AD_DECAY_SUSTAINED_CYCLES=14
PSA_AD_DECAY_MIN_PATTERNS_FLOOR=3
PSA_AD_DECAY_SHADOW_DECLINE_PENALTY=0.5
PSA_AD_DECAY_SHADOW_SUSTAINED_CYCLES=7
```

Env wins over config file, config file wins over defaults — same precedence as the rest of PSA.

## §12 — Observability

### §12.1 — `psa advertisement status`

```
psa advertisement status [--anchor ID] [--tenant T] [--json]
```

Prints:
- Total active patterns, by source (from metadata)
- Ledger distribution histogram (20 buckets from min to max)
- Counts: in-grace, past-grace-positive, past-grace-negative-under-threshold, at-risk (`consecutive_negative_cycles >= sustained_cycles / 2`), removal-eligible
- Shadow-only at-risk: patterns A would remove but B keeps

### §12.2 — `psa advertisement diff`

```
psa advertisement diff [--tenant T] [--json]
```

Counterfactual report:
- Patterns where B and A disagree on removal eligibility today
- For each disagreement: pattern_text, anchor_id, ledger, shadow_ledger, consecutive_negative_cycles, shadow_consecutive, decision under each policy
- Aggregate: shadow-agreement rate, disagreement concentration by anchor source

### §12.3 — Lifecycle log per nightly run

Structured log line appended to the existing lifecycle log:

```json
{
  "stage": "advertisement_decay",
  "ts": "2026-04-17T...",
  "n_active": 3847,
  "n_decayed": 3847,
  "n_in_grace": 210,
  "n_at_risk": 58,
  "n_would_remove_under_A": 31,
  "n_actually_removed_under_B": 14,
  "shadow_agreement_rate": 0.83
}
```

### §12.4 — `psa advertisement rebuild-ledger`

Canonical regenerator. Recomputes ledger rows from trace + augmentation. Takes a `--dry-run` flag. Primary use cases:

1. Drift recovery (if inline writes failed silently).
2. Retroactive weight tuning (replay trace with different config to see what the ledger would look like).

### §12.5 — `psa advertisement purge`

```
psa advertisement purge [--older-than 90d] [--dry-run]
```

Hard-deletes archived rows past retention. Runs automatically weekly from lifecycle, or on-demand from CLI.

## §13 — Success metrics

**Primary (calibration phase):** shadow agreement rate, expected ≥ 0.90 after first calibration window (3–4 weeks of realistic traffic).

**Secondary (post-enable):** R@5 delta before vs after `removal_enabled=true`. Should be flat-or-positive; a regression indicates the decline penalty is too aggressive.

**Tertiary (long-term):** survival-rate curve (fraction of atlas patterns still alive after N days). Stage 2's goal is not to remove many patterns — it's to remove the *right* ones steadily.

---

## Files

### New

| File | Responsibility |
|---|---|
| `psa/advertisement/ledger.py` | `PatternLedgerRow` dataclass; `record_query_signals`, `apply_decay_and_update_counters`, `evaluate_removal`, `apply_removals`, `archive_all_active`, `insert_fresh_row`, `upsert_ledger`, `pattern_id_for`. Owns the table schema migration. |
| `psa/advertisement/attribution.py` | `attribute_bm25_argmax(query, patterns, epsilon) -> (argmax, eps_tied)`. Pure function; no DB access. Reused by inline writer and `rebuild-ledger`. |
| `psa/advertisement/config.py` | `AdvertisementDecayConfig` dataclass — typed view over the `advertisement_decay` block in `MempalaceConfig`. No duplicate parsing; reads through the existing config system. |
| `psa/advertisement/cli.py` | `psa advertisement status`, `diff`, `rebuild-ledger`, `purge` subcommands. |
| `tests/test_ledger_schema.py` | Table creation, upsert semantics, archival shape, active-row index correctness. |
| `tests/test_ledger_attribution.py` | BM25 argmax correctness, ε-tie split, BM25 floor filtering. |
| `tests/test_ledger_write_path.py` | Inline signal recording, per-credited-set weight division, tracking_enabled=false short-circuit. |
| `tests/test_ledger_decay.py` | Exponential decay arithmetic, consecutive-negative-cycles state machine, removal eligibility ordering. |
| `tests/test_ledger_shadow.py` | Shadow ledger tracks independently; shadow disagreement surfaces in diff output. |
| `tests/test_ledger_rebuild_reset.py` | Atlas rebuild archives all active rows, inserts fresh rows for new atlas patterns. |
| `tests/test_ledger_coexistence.py` | Stage 1 P1/P3 protections apply to stage 2; patterns removed by `psa atlas decay` then promoted don't resurface in stage 2 evaluation. |
| `tests/test_cli_advertisement.py` | `status`, `diff`, `rebuild-ledger`, `purge` output shapes; dry-run behaviour. |

### Modified

| File | Change |
|---|---|
| `psa/pipeline.py` | After selector returns and before `trace.write_trace`, call `ledger.record_query_signals(...)` with retrieved/selected anchor ids + BM25 top-K shortlist. |
| `psa/anchor_retriever.py` | Expose BM25-side top-K shortlist on the `RetrievalResult` so `pipeline.py` can pass it to `record_query_signals`. One additive field. |
| `psa/lifecycle.py` | New `advertisement_decay_pass(...)` step in fast path, between "mine new sessions" and "prune memories". |
| `psa/atlas.py` | `AtlasManager.rebuild()` calls `ledger.reset_ledger_on_rebuild(...)` after the existing metadata-inheritance pass. |
| `psa/trace.py` | `new_trace_record(...)` adds optional `retrieval_attribution: []` field (not populated by default; populated by pipeline when ledger is enabled). |
| `psa/config.py` | Parse the new `advertisement_decay` config block; env overrides. |
| `psa/cli.py` | Wire `advertisement` subparser dispatching to `psa/advertisement/cli.py`. |

### Not touched

- `psa/advertisement/metadata.py`, `reinforcement.py`, `decay.py`, `writer.py` — stage 1 internals are unchanged.
- `psa/forgetting.py` — memory forgetting is unrelated.
- Stage 1's `psa atlas decay` CLI path — unchanged.

---

## Success criteria

- `pattern_ledger` table created at first tenant startup after upgrade; no migration of existing tenants fails.
- With `tracking_enabled=false`, the query hot path has zero new writes and zero new SELECTs. Ledger code is a no-op.
- With `tracking_enabled=true, removal_enabled=false`: ledger rows are populated inline; nightly lifecycle logs summary counts; no `anchor_cards.json` mutation; `psa advertisement status` / `diff` report live data.
- With `tracking_enabled=true, removal_enabled=true`: nightly pass removes eligible patterns; `anchor_cards.json` is updated; ledger rows are soft-archived; lazy BM25 reindex picks up changes on next query.
- Stage 1's `psa atlas decay` continues to run unchanged, producing the same candidate output as before. A pattern removed by stage 1 promotion does not re-appear in stage 2's active ledger view.
- P1 (low-activation shield) and P3 (pinned exemption) protect stage 2 removals, via direct reuse of stage 1's logic.
- `psa advertisement rebuild-ledger` reconstructs the ledger from trace with matching final values (within floating-point tolerance) when `retrieval_attribution` augmentation is present.
- Atlas rebuild archives all active ledger rows with `removal_reason="atlas_rebuild_reset"` and inserts fresh rows for every pattern in the new atlas's cards.
- `psa advertisement purge --older-than 90d` hard-deletes archived rows past retention, matching memory forgetting cadence.
- Full test suite green. Ruff clean.
- No breaking changes to existing CLI surfaces or config.

## Follow-ups (explicit deferrals)

- **Dense/semantic attribution.** Stage 3 candidate if BM25 proves too lexical — embed each pattern once, compute cosine(query, pattern) per retrieval, consider replacing or augmenting argmax with dense.
- **Per-source decay parameters.** `per_source` config overrides (e.g. `atlas_build` patterns decay faster than `refinement`). Stage 1 already reserves the schema key.
- **Ledger carry-over across atlas rebuilds via similarity match.** Only worth building if rebuild-volatility becomes observable through archived-row volume.
- **Activation spreading / co-activation integration.** The ledger is currently per-pattern; extending to per-pattern-pair for co-activation is a different mechanism and should be specced separately if pursued.
- **Auto-tuning of weights.** Offline optimisation against R@5 or oracle labels. Deliberately deferred — knobs are operator-tuned via `diff` output during calibration.
- **Stage 3: card evolution under gated rewrite.** Independent spec. Prerequisite is stable stage 2 decay signal telling us *which* patterns to candidate for rewrite.
