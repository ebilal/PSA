# Advertisement Forgetting

**Date:** 2026-04-17
**Branch:** `feat/advertisement-forgetting` (off `main`)

## Problem

Anchor cards accumulate `generated_query_patterns` but never lose them. Three sources write into that single list today:

1. **Atlas build** — Qwen generates 10–15 full-sentence question forms per anchor at cluster time. Many are speculative — they describe what Qwen *thinks* the anchor should answer, not what it has answered.
2. **`psa atlas refine`** — offline ngram expansion from a miss log. Grounded in real queries that failed.
3. **`psa atlas curate`** — production-signal curation (fingerprints + oracle labels). Grounded in real queries that hit.

A card text becomes the union of all three, growing monotonically with every promotion. Over time:

- **Atlas-build speculation accumulates**, even when no real traffic ever confirmed it.
- **Stale phrasings from old miss-log rounds hang around**, long after the misses they targeted were resolved.
- **Refined patterns whose source memories were forgotten stay on the card**, advertising capability the system no longer has.

Advertisement forgetting closes the loop: patterns that stop being reinforced by real traffic become candidates for removal, subject to the same candidate/promote gate that governs every other card change. The branch is deliberately small, conservative, and reversible. Nothing gets deleted; stale patterns get proposed for removal in a candidate and the operator reviews before promotion.

## Goals

1. **Conservative MVP**, candidate-side only. No live card mutation. Uses Branch 1's existing candidate/promote flow.
2. **Provenance in time** — persist `created_at` and `source` per pattern so decay can distinguish "never reinforced in 60 days, 90 days old" (stale) from "never reinforced in 60 days, added yesterday" (too new to judge).
3. **Reinforcement derived, not stored** — dynamic signal comes from the per-query trace log (Branch 4). No persistent counters that can drift from the trace.
4. **Dry-run first**. Default output is a report; operator opts into writing a candidate.
5. **Protection against unfair pruning** — anchors with too little activation volume are shielded; operator-pinned patterns are exempt.
6. **Source-aware schema hooks** so later iterations can decay `atlas_build` patterns faster than `refinement` patterns without a migration.

## Non-goals

- Automatic decay on a schedule. Operator runs `psa atlas decay` manually; lifecycle integration is a follow-up.
- Decay of `query_fingerprint` entries (already FIFO-capped at 50; separate mechanism).
- Decay of memories themselves. `psa/forgetting.py` handles memory forgetting; this branch is scoped to *advertisement* (card text).
- Automatic metadata stamping at atlas-build / curate / refine time. MVP backfills lazily on first decay run. Adding creation-time hooks is a small follow-up.
- Per-source parameter overrides beyond what the config schema reserves. MVP uses one set of params for all sources.
- Token-overlap or semantic reinforcement (R3). MVP uses substring match (R1) because it's simple, fast, and interpretable — and because this is candidate-side only, the first time the rule behaves badly we'll see it in the dry-run output before it goes live.

---

## §1 — Metadata model

Per-pattern metadata lives in a sibling file, not on the card itself. Cards stay text-only; Branch 1's invariants on anchor_cards JSON shape are preserved.

**File:** `<atlas_dir>/pattern_metadata.json`

**Shape:**

```json
{
  "<normalized_key>": {
    "anchor_id": 227,
    "pattern": "what are some quick healthy lunch ideas?",
    "source": "atlas_build",
    "created_at": "2026-03-15T12:34:56+00:00"
  }
}
```

**Reserved future fields** (schema-forward, not consumed by this branch):

- `promoted_at: str | null` — when the pattern first landed on the live `anchor_cards_refined.json`. No code path in this branch sets it; meaningful values require the creation-time stamping follow-up that teaches `psa atlas promote-refinement` to update metadata. Until then, consumers treat the field as absent-or-null. Keeping it reserved (rather than shipping it always-null) avoids the "active-looking dead field" anti-pattern.
- `pinned: bool` — P3 protection flag. See §3.

**Key format:** `"{anchor_id}::{normalized_pattern}"` where

```python
def normalize_pattern(text: str) -> str:
    return " ".join(text.strip().lower().split())
```

The raw pattern is stored alongside the key so the file is human-readable without re-normalization at read time. Keys are stable across formatting variations (trailing spaces, case, double-spaces) so metadata tracks logical identity, not textual accident.

**Persistence.** Writes are atomic: write to `pattern_metadata.json.tmp`, `os.replace()` to final. Prevents half-written state if the process is interrupted mid-write.

**Source enum:**

| Value | Meaning |
|---|---|
| `atlas_build` | Qwen-generated at atlas induction. |
| `refinement` | Added by `psa atlas refine` (miss-log-driven). |
| `production_signal` | Added by `psa atlas curate` (oracle + fingerprints). |
| `manual` | Operator-placed (explicit intent outside the curation pipelines). |
| `unknown` | Backfilled retroactively — see §5.2 backfill policy. |

Reserved for future: `pinned` isn't a source value — it's a separate boolean flag `pinned: true` on the metadata entry (P3 protection). MVP supports `pinned` in the schema but has no CLI to set it yet; that's a one-command follow-up.

### §1.1 — Atlas rebuild inheritance

`pattern_metadata.json` lives per atlas version (`atlas_v{N}/pattern_metadata.json`). Without an explicit inheritance rule, `psa atlas build` would create `atlas_v{N+1}/` with no metadata, and the next decay run would backfill every pattern as `source="unknown", created_at=<now>` — resetting the forgetting clock on every rebuild and defeating the point of persisting `created_at`.

**Inheritance rule**, matching the existing `FingerprintStore.inherit_from()` pattern:

During `AtlasManager.rebuild()`, after the new atlas has been built and anchor matching has run, perform a metadata copy-forward pass:

```python
# In psa/atlas.py, after centroid matching during rebuild:
old_meta = load_pattern_metadata(previous_atlas.anchor_dir)
new_meta: dict[str, dict] = {}
for card in new_atlas.cards:
    for pattern in card.generated_query_patterns:
        key = f"{card.anchor_id}::{normalize_pattern(pattern)}"
        if key in old_meta:
            # Matched anchor (anchor_id preserved across rebuild) + matched
            # pattern text — carry the entry forward verbatim.
            new_meta[key] = old_meta[key]
save_pattern_metadata(new_atlas.anchor_dir, new_meta)
```

What this does and doesn't handle:

- **Matched anchor + matched pattern**: metadata copied forward, `created_at` preserved. This is the common case — most patterns survive a rebuild.
- **Matched anchor, new pattern** (Qwen produced different wording at rebuild time): no entry in old metadata, no entry copied, backfill stamps it as `source="unknown"` on the next decay run. Acceptable. The pattern IS new to the new atlas version.
- **Matched anchor, removed pattern** (pattern was on v{N} but not v{N+1}): old metadata entry is orphaned (not copied forward). Orphans disappear with the old atlas directory and its file.
- **New anchor** (introduced at rebuild): no entries exist; backfill stamps all its patterns on next decay run. Correct.
- **Retired anchor** (dropped at rebuild): old entries stay in the old atlas dir's metadata but don't carry forward. Retired atlas dirs are already out of scope for decay.

**Why the inheritance walks new cards, not old:** by iterating the new atlas's patterns and querying old metadata, we naturally drop orphans (old entries with no matching new pattern) without needing to track them explicitly. The new metadata file is exactly right-sized for the new atlas.

**Normalization consistency:** `normalize_pattern` is applied on both sides. A pattern whose text drifts slightly between rebuilds (e.g., punctuation change) retains its metadata as long as normalization collapses the difference.

**Operational consequence:** after a rebuild, the first decay run's `n_patterns_backfilled_this_run` count equals (approximately) the number of patterns introduced or re-worded at rebuild time — a useful diagnostic for how stable the atlas topology is across rebuilds.

## §2 — Reinforcement signal (R1: activation + substring)

A pattern is **reinforced** by a trace record when:

1. The record's `selected_anchor_ids` contains this pattern's anchor. (The anchor actually activated.)
2. The record's `query_origin` is in the active origin set (default `{"interactive"}` — matches Branch 4's rollup convention).
3. The normalized pattern appears as a substring of the normalized query text.

When these hold, the pattern's **derived** `last_reinforced_at` is updated to the record's timestamp (if more recent than the current value).

### Why R1

- **Simple and interpretable.** Substring is fast; substring hits are auditable by eye.
- **Right-grained for ngram patterns.** Refined ngrams (3–6 words) frequently appear verbatim in later queries — they were extracted from real queries to begin with.
- **Intentionally asymmetric.** LLM-authored full-sentence patterns rarely match verbatim. Under R1 they decay faster than ngrams. This is a reasonable first bias: *speculative advertisement should need validation from use.*

### Known limitation

R1 doesn't catch paraphrase. Query "when does the token refresh happen" does not reinforce pattern "how does the token refresh flow" even though both are about token refresh. A follow-up iteration may replace R1 with token-overlap or semantic matching. This branch ships R1 behind the candidate/promote gate so operators see surprises before live state changes.

## §3 — Decay rule (D1 + P1 + P3)

### D1 — Grace + reinforcement window

A pattern is a decay candidate when:

- `age >= grace_days`, AND
- `last_reinforced_at` is `None` OR older than `decay_window_days`.

Pseudocode:

```python
age_days = (now - pattern.created_at).days
if age_days < grace_days:
    return False  # protected by grace period
if pattern.last_reinforced_at is None:
    return True  # never reinforced, past grace
reinforced_days_ago = (now - pattern.last_reinforced_at).days
return reinforced_days_ago > decay_window_days
```

### P1 — Low-activation anchor shield (two-part)

An anchor is **shielded** and none of its patterns decay when:

- Its activation count (interactive-origin trace records in the trace-scan window) is below `min_anchor_activations`, **OR**
- Its activation percentile among all anchors is below `low_activation_percentile`.

**Two-part because percentiles misbehave on small tenants.** On a tenant with 20 trace records total, "below the 25th percentile" might include half the atlas. The absolute floor (`min_anchor_activations`) prevents over-shielding; the percentile catches tenants whose distribution is bimodal (a few very-active anchors dominate). Both together give sane behavior across scales.

Rationale: an anchor that rarely activates hasn't had the *chance* to reinforce its patterns. Pruning them punishes the anchor for being niche.

### P3 — Operator-pinned patterns

If `metadata[key].pinned is True`, the pattern is exempt from decay regardless of rule. MVP schema supports the flag; no CLI to set it yet (follow-up). Operators who want to pin can edit `pattern_metadata.json` manually for MVP.

### Default parameters

Live in `~/.psa/config.json`:

```json
{
  "decay": {
    "grace_days": 30,
    "decay_window_days": 60,
    "low_activation_percentile": 25,
    "min_anchor_activations": 10,
    "per_source": {}
  }
}
```

`per_source` is **schema-reserved, not consumed in MVP.** When later iterations want `atlas_build` patterns to decay at 30-day windows and `refinement` at 90, they add entries there — no migration.

These defaults are deliberately conservative. The first decay run on a real tenant should prune few patterns, not many. If telemetry shows the system is holding onto obvious junk, tighten `decay_window_days`; if it's pruning things that shouldn't decay, loosen `grace_days` or raise `min_anchor_activations`.

## §4 — CLI surface and output files

### `psa atlas decay`

```
psa atlas decay [--tenant T] [--dry-run]
                [--grace-days N] [--decay-window-days N]
                [--low-activation-percentile P] [--min-anchor-activations N]
                [--include-origin ORIGIN]
                [--verbose] [--json]
```

**Dry-run (default: false; present via `--dry-run`).** No *candidate* files written. Report printed to stdout; `--json` emits the full structured report as a single JSON object.

**Metadata exception.** Both dry-run and real mode persist metadata backfill (§5.2) — stamping `created_at` for previously unstamped patterns is provenance-establishing and non-destructive. Dry-run writing `pattern_metadata.json` means the second dry-run sees real ages, not "everything backfilled right now." Candidates stay out of the atlas dir on dry-run; only `pattern_metadata.json` may be written.

**Real.** Writes three candidate files into the current atlas directory:

| File | Purpose |
|---|---|
| `anchor_cards_candidate.json` | Cards with stale patterns removed. Branch 1's live candidate path; promotion still required. |
| `anchor_cards_candidate.meta.json` | **Summary only.** Counts, parameters, shield/pin counts, reason histogram. Bounded size. |
| `anchor_cards_candidate.decay_report.json` | **Full detail.** Every removed pattern with provenance. Written by the decay path only. |

### Coexistence

Same one-candidate-in-flight contract as refine/curate. Running `psa atlas decay` overwrites any existing candidate. Empty-run guard (no patterns would be removed) skips all three writes and exits 0 with a warning — matches Branch 3's behavior for `curate`.

### Tabular default output

```
tenant: default   atlas v14   trace records: 8,412   origins: interactive
params: grace=30d  decay_window=60d  activation_floor: <p25 or <10 activations

Summary:
  Total patterns scanned:    3,847
  Decay candidates:            312   (8.1%)
    - atlas_build source:      218   (23% of 942 atlas_build patterns)
    - refinement source:        67   (3.2% of 2,103 refinement patterns)
    - unknown source:           27   (3.4% of 802 unknown patterns)
  Anchors shielded by P1:       42
    - patterns held:           401
  Pinned (P3) exempted:          6

Top 10 anchors with most decay candidates:
  1. recipe-meal-prep              17/24 patterns stale
  2. weekly-standups               14/19 patterns stale
```

Source grouping in the default output is non-optional — operators need to see whether atlas-build patterns are being pruned disproportionately. `--verbose` adds per-pattern reasons to stdout. `--json` emits the `.meta.json` + `.decay_report.json` combined as one envelope.

### `anchor_cards_candidate.meta.json` shape (summary)

```json
{
  "source": "decay",
  "created_at": "2026-04-17T...",
  "tenant_id": "default",
  "atlas_version": 14,
  "promoted": false,
  "promoted_at": null,
  "decay_parameters": {
    "grace_days": 30,
    "decay_window_days": 60,
    "low_activation_percentile": 25,
    "min_anchor_activations": 10
  },
  "n_patterns_scanned": 3847,
  "n_patterns_removed": 312,
  "n_patterns_by_source_removed": {
    "atlas_build": 218,
    "refinement": 67,
    "production_signal": 0,
    "manual": 0,
    "unknown": 27
  },
  "n_anchors_touched": 58,
  "n_anchors_shielded": 42,
  "n_patterns_shielded": 401,
  "n_patterns_pinned_exempt": 6,
  "n_patterns_backfilled_this_run": 0,
  "pruning_by_reason": {
    "stale_unreinforced": 312
  },
  "origins": ["interactive"]
}
```

`pruning_by_reason` is a dict (not flat count) so future reasons — `duplicate_ngram_subsumed`, `operator_manual_remove`, etc. — extend without breaking the schema.

### `anchor_cards_candidate.decay_report.json` shape (detail)

```json
{
  "tenant_id": "default",
  "atlas_version": 14,
  "created_at": "2026-04-17T...",
  "removed_patterns": [
    {
      "anchor_id": 227,
      "pattern": "what are some quick healthy lunch ideas?",
      "source": "atlas_build",
      "created_at": "2026-03-15T...",
      "last_reinforced_at": null,
      "reason": "stale_unreinforced"
    }
  ],
  "shielded_anchors": [
    {"anchor_id": 301, "activation_count": 3, "patterns_held": 8}
  ]
}
```

`reason` is a string drawn from a module-level enum; every entry is self-explanatory on inspection. MVP ships only `stale_unreinforced`.

### Promotion

`psa atlas promote-refinement` is unchanged — it already handles any candidate regardless of source. After a decay promotion, the refined cards have fewer patterns; loader picks them up via Branch 1's preference. Operator runs `psa train --coactivation --force` after the promotion; the promote output already prints that hint (from Branch 3).

## §5 — Backfill + reinforcement derivation details

### §5.1 — Metadata write paths

**Write from backfill (this branch).** When decay runs and a pattern has no metadata entry, an entry is created with `source="unknown"`, `created_at=<decay run start time>`. The `pattern_metadata.json` file is updated atomically after the derivation phase. Backfilled entries PERSIST after the run — they are the only durable effect of a dry-run pass on existing tenants.

**Write from atlas / curate / refine (NOT in this branch).** Stamping patterns at creation time is a small follow-up. Each call site stamps `created_at=now` with the correct `source`. Until those land, the backfill policy handles the gap.

### §5.2 — Backfill policy (conservative, explicit)

Backfilled entries use `created_at = <decay run start time>`. This **grants every legacy pattern a full `grace_days` reprieve** on the first decay run after migration.

This is a deliberate conservative migration choice, not ground truth. The alternative — treating unknown-source patterns as "infinitely old" — would aggressively prune every pre-migration pattern in one shot, which is the wrong direction for a safety-first first pass.

Two operational consequences, spec-documented:

- The first decay run after migration prunes almost nothing. (All legacy patterns are too-young-by-backfill.)
- Subsequent runs operate normally. By the second run, real `last_reinforced_at` data has accumulated and real decay decisions start.

The candidate meta file reports `n_patterns_backfilled_this_run` so the operator sees the migration scale.

### §5.3 — Reinforcement derivation (ephemeral, not persisted)

**Principle:** reinforcement state is ephemeral. It is computed into a per-run in-memory map. It is not written back to `pattern_metadata.json`.

Metadata on disk holds only provenance (`source`, `created_at`, `promoted_at?`, `pinned?`). The decay pass has no persistent dynamic state.

This separation is load-bearing: it prevents a future contributor from accidentally persisting a `last_reinforced_at` counter that can drift from the trace. The trace is the source of truth for dynamic signal; re-derive every run.

**Algorithm:**

```python
# Ephemeral per-run map. Never written to disk.
reinforcement: dict[str, datetime] = {}

window_start = now - timedelta(days=(decay_window_days + grace_days))
# (This bound is a conservative optimization, not a mathematically exact
# minimum. Widening it later doesn't change the rule's semantics — it
# just scans more records to reach the same decision.)

for record in iter_trace_records(tenant, origins=active_origins):
    ts = parse_iso(record["timestamp"])
    if ts < window_start:
        continue
    q_norm = normalize_pattern(record["query"])
    for anchor_id in record.get("selected_anchor_ids", []):
        for pattern in cards_by_anchor[anchor_id].generated_query_patterns:
            p_norm = normalize_pattern(pattern)
            if p_norm in q_norm:
                key = f"{anchor_id}::{p_norm}"
                if key not in reinforcement or ts > reinforcement[key]:
                    reinforcement[key] = ts
```

After the loop, `reinforcement` is consulted alongside metadata `created_at` to apply D1. `reinforcement` is discarded when the decay pass exits.

### §5.4 — Origin filtering

Reinforcement respects the active origin filter. Default `{"interactive"}`. Same semantics as `psa diag` from Branch 4: production traffic only by default. `--include-origin benchmark` would allow benchmark queries to count as reinforcement — usually wrong, but available for debugging.

## Files

### New

| File | Responsibility |
|---|---|
| `psa/forgetting/__init__.py` | Package marker. |
| `psa/forgetting/metadata.py` | `PatternMetadata` dataclass, `normalize_pattern`, `load_metadata`, `save_metadata` (atomic), `backfill_unknown`. |
| `psa/forgetting/reinforcement.py` | `compute_reinforcement(tenant_id, atlas, window_start, origins) -> dict[str, datetime]`. Ephemeral per-run map. |
| `psa/forgetting/decay.py` | `DecayReport` dataclass + `decay_report(tenant_id, params, origins) -> DecayReport`. Orchestrator. |
| `psa/forgetting/writer.py` | Writes `anchor_cards_candidate.json` + `.meta.json` + `.decay_report.json`. Respects empty-run guard. |
| `psa/cli.py` | Add `_cmd_atlas_decay` handler + subparser. |
| `tests/test_forgetting_metadata.py` | Key normalization; atomic write; backfill behavior; source enum. |
| `tests/test_forgetting_reinforcement.py` | Substring + activation; origin filtering; window bound; ephemeral (not persisted). |
| `tests/test_forgetting_decay.py` | D1 rule; P1 two-part shield; P3 pin respect; source-grouped counts; empty-run guard. |
| `tests/test_cli_atlas_decay.py` | Dry-run vs. real; JSON envelope; promote-gate integration (runs through promote-refinement); coexistence with refine/curate candidates. |
| `tests/test_atlas_metadata_inheritance.py` | `AtlasManager.rebuild()` carries metadata forward for matched patterns; orphans are dropped; new patterns are not pre-stamped. |

### Modified

| File | Change |
|---|---|
| `psa/cli.py` | New `decay` subparser + handler. Promote command's recalibration hint remains unchanged (promoted decayed cards need the same `psa train --coactivation --force` follow-up). |
| `psa/config.py` | Document `decay` config block; add accessor if the config class uses explicit keys. |
| `psa/atlas.py` | `AtlasManager.rebuild()` gains a metadata-inheritance step after centroid matching (§1.1). Mirrors how `FingerprintStore.inherit_from()` is already called during rebuild. |

### Not touched (explicitly)

- `psa/atlas.py`, `psa/anchor.py` — card shape unchanged; metadata lives in a sibling file.
- `psa/curation/` — no metadata stamping at curate time in this branch. Follow-up.
- `scripts/refine_anchor_cards.py` — same; no stamping at refine time in this branch.
- `psa/forgetting.py` (existing memory forgetting) — distinct from this branch. Advertisement forgetting ≠ memory forgetting.
- Branch 1/2/3/4 artifacts — all unchanged.

## Success criteria

- `psa atlas decay --dry-run` prints a report without writing any candidate files. `pattern_metadata.json` may be updated with backfilled entries (provenance-only, non-destructive).
- `psa atlas build` (rebuild) carries `pattern_metadata.json` forward for matched `(anchor_id, normalized_pattern)` pairs. New patterns are left unstamped (get backfilled next decay run); orphaned entries are dropped. `created_at` is preserved for patterns that survive the rebuild.
- `psa atlas decay` (real) writes exactly three files: `anchor_cards_candidate.json`, `anchor_cards_candidate.meta.json`, `anchor_cards_candidate.decay_report.json`. None of them is `anchor_cards_refined.json`.
- After `psa atlas promote-refinement`, the promoted cards have fewer patterns and the loader picks them up via Branch 1's preference.
- `pattern_metadata.json` persists backfilled entries for previously unstamped patterns; the operator can see the backfill count in the summary.
- Anchors below the activation floor (`min_anchor_activations` OR `low_activation_percentile`) are shielded; none of their patterns appear in `removed_patterns`.
- Patterns with `pinned=true` in metadata are exempt regardless of D1 outcome.
- A tenant with no trace records (fresh install, never queried) runs decay cleanly: backfill happens, grace period protects everything, `n_patterns_removed=0`, no candidate written (empty-run guard fires).
- Full test suite green. Ruff clean.
- No breaking changes to existing Branch 1/2/3/4 CLI surfaces.

## Follow-ups (explicit deferrals)

- **Metadata stamping at creation time** for atlas build, curate, and refine paths. Small — each call site adds a `metadata.add(key, source, created_at)` call. When those land, new patterns get true timestamps while old ones keep their conservative backfill. This is the natural next branch after advertisement-forgetting MVP.
- **`psa atlas pin-pattern` CLI** to set `pinned=true` without hand-editing JSON. Small follow-up.
- **Per-source parameter overrides** (`per_source` config key). Allows `atlas_build` patterns to decay faster than `refinement` patterns.
- **Semantic reinforcement (R3)** if R1's substring miss-rate becomes observable through dry-run output over time.
- **Automatic decay in lifecycle slow path**. Not until we've seen manual decay behave well for a while.
- **Decay of memories** via `psa/forgetting.py` — related but separate cognitive surface.
