# Production-Signal Curation

**Date:** 2026-04-17
**Branch:** `feat/production-signal-curation` (off `main`)

## Problem

Anchor cards today are authored at atlas-build time (Qwen distills memories into a card with `generated_query_patterns`) and optionally refined from miss logs (`psa atlas refine --miss-log ...`). Both pathways ignore the live signal that arrives whenever the system runs: `query_fingerprint` (real queries that activated an anchor above threshold) and `oracle_labels.jsonl` (LLM-judged anchor-to-query coverage).

A correct system should let anchor cards absorb that signal — "what this anchor can answer" should reflect what it has actually answered and what it has been judged to cover. But not directly: raw fingerprints piped into card text would be noisy and drift-prone, and oracle labels alone under-represent actual usage.

Branch 3 adds a curated feedback loop: raw production signals → oracle-filtered pool per anchor → pluggable extractor → candidate card update → operator-approved promotion. Nothing becomes CE-visible without the promotion step Branch 1 established.

## Goals

1. Define the production-signal **contract**: what signals, which filter, what output.
2. Ship the MVP: heuristic ngram extractor over oracle-endorsed query pools. Laptop-friendly, deterministic, zero LLM cost.
3. Commit to the **architecture** so an LLM-based extractor can land later behind the same interface with no contract change.
4. Reuse Branch 1's candidate/promote flow completely — no new storage, no new promotion command.
5. Name the recalibration step that follows promotion without automating it.

## Non-goals

- Weighting between oracle-endorsed and endorsed-fingerprint queries. Skipped for MVP; follow-up when calibration data exists.
- Fuzzy / canonicalized / semantic oracle-match filtering. MVP is exact query-text match, labeled provisional.
- `LLMExtractor` implementation beyond an interface stub. Branch 3 commits to the shape; implementation comes later.
- Merging miss-log refinement with signal curation into a single candidate. One candidate in flight at a time — operator chooses which source path per cycle.
- The memory scorer question (from the earlier sequence plan, deferred to after Branch 3 lands).
- Observability rollups (Branch 4).

---

## §1 — Signal contract

Two sources feed the curator. Neither is CE-visible; both feed curation, which writes to a candidate that goes through Branch 1's promotion gate.

- **`fingerprint_store`** — persisted at `<atlas_dir>/fingerprints.json`. Already populated by `pipeline.py:278-281` for every above-threshold selected anchor. Raw, 50-cap FIFO per anchor.
- **`training/oracle_labels.jsonl`** — `OracleLabel` records from `psa label`. Each record has `query` (the query text) and `winning_oracle_set` (anchor IDs the LLM judge picked).

The MVP reads both from disk; no changes to signal *collection*.

## §2 — Pool construction with oracle filter

Per anchor `A`, the curator builds a pool with two separately-tracked components:

- **`oracle_queries`**: for every oracle label with `A ∈ winning_oracle_set`, include the label's `query`. Normative signal — queries the LLM judge said *should* land on `A`.
- **`endorsed_fingerprint_queries`**: fingerprint queries for `A` whose text exactly matches a query appearing in some oracle label where `A` is in the winning set. Represents alignment between observed demand and intended coverage.

Both lists are returned by `build_pool()` as separate fields — the internal data model preserves provenance even when the curator unions them before extraction.

**Excluded:**
- Fingerprint queries with no oracle-label support for this anchor. Drops the drift risk on unverified live signal.
- Oracle labels where `A` is not in `winning_oracle_set`.

**Empty-pool behavior:** if neither component has entries, the anchor contributes no new candidate patterns. The anchor's card is written unchanged into the candidate JSON. No error.

### Support semantics are provisional

Exact query-text match is a **conservative bootstrap**, not the intended long-term contract. It is explicit and traceable (easy to write, easy to audit), but it misses paraphrased queries and semantically equivalent phrasings that a more realistic production would need to collapse.

Metadata records the current semantics in a dedicated field (`support_semantics: "exact_query_text"`) so future runs can advertise their own rule without confusing provenance. Reserved values for later: `"canonicalized"`, `"embedding_cosine"`.

This field should prevent "whatever the MVP rule was" from drifting into doctrine.

## §3 — Extractor interface + backends

```python
class QueryPatternExtractor(Protocol):
    def extract(self, pool: list[str], n: int) -> list[str]:
        """Given a pool of queries, return up to n candidate query patterns."""
```

The interface is fixed. Two backends, one shipped, one reserved:

- **`HeuristicExtractor`** (MVP). Runs `extract_ngrams()` over each query, dedupes across the pool, sorts longer patterns first, returns up to `n`. Identical mechanism to the existing `refine_anchor_cards.py` ngram logic, pulled into `psa/curation/ngrams.py` for reuse. Zero LLM cost; deterministic.

- **`LLMExtractor`** (architectural commitment, not implemented). Interface-compliant stub whose `extract()` raises `NotImplementedError("LLM extractor reserved for follow-up")`. Docstring specifies the Qwen call contract: *"Given N real user queries this anchor has answered and been endorsed for, produce up to n representative question patterns. Prefer coverage over paraphrase."* Future implementation wraps one Qwen call per anchor with the pool rendered into the prompt.

Extractor output feeds `generated_query_patterns`. Deduplication against the anchor's existing patterns happens in the curator, not the extractor. Cap at 20 total patterns per anchor (same as current refinement).

---

## §4 — Candidate write

Reuses Branch 1's candidate shape entirely. No new disk format, no new promotion verb.

### Files written per curation run

- `<atlas_dir>/anchor_cards_candidate.json` — full card list, each card's `generated_query_patterns` augmented with extractor output (deduped, capped at 20).
- `<atlas_dir>/anchor_cards_candidate.meta.json` — Branch 1 schema plus new fields.

### Metadata schema

Branch 1's existing fields, unchanged:

```json
{
  "source": "production_signal",
  "created_at": "2026-04-17T...",
  "tenant_id": "default",
  "atlas_version": 1,
  "promoted": false,
  "promoted_at": null,
  "n_anchors_touched": 87,
  "n_patterns_added": 312
}
```

Additive Branch 3 fields:

```json
{
  "oracle_labels_read": 27599,
  "fingerprints_read": 4921,
  "n_anchors_with_oracle_support": 142,
  "n_anchors_with_endorsed_fingerprints": 34,
  "extractor": "heuristic",
  "support_semantics": "exact_query_text"
}
```

- `source = "production_signal"` extends the existing `manual | benchmark | oracle | query_fingerprint` enum. Keeps provenance honest without over-committing to a sub-source — the real tracking is `extractor` + `support_semantics`.
- Existing Branch 1 meta consumers tolerate unknown keys (`meta = json.load(...)`; no schema validator rejects extras). Forward-compatible.

### Promotion path

`psa atlas promote-refinement` is unchanged. It already preserves `source` verbatim and flips `promoted=true` / `promoted_at=<now>`. The new `source` string flows through with no code change.

### Coexistence contract: one candidate in flight at a time

If an operator ran `psa atlas refine --miss-log ...` and did not promote, running `psa atlas curate` overwrites that candidate. Inverse is also true. Merging source types into a single candidate is deferred.

This is a **deliberate contract**, not a limitation: mixing miss-log-driven patterns with signal-derived patterns under one `source` field would obscure provenance. Sequential workflows (refine → promote → curate → promote, or the reverse) accumulate patterns into the live `anchor_cards_refined.json` one source at a time.

---

## §5 — Recalibration after promotion

Promotion changes what `to_stable_card_text()` returns for the affected anchors. The CE scorer reads this at inference time with no code change — atlas loader propagates automatically (Phase 3C behavior).

But downstream learned models were trained against the *previous* card text. Their calibration is stale. After a non-trivial promotion, the operator recalibrates:

- **Minimum recommended recalibration: `psa train --coactivation --force`.** Regenerates `coactivation_train.npz` with new CE scores against promoted cards; retrains `CoActivationModel`. Expected time: ~30–60 min on CPU for ~27k oracle labels.

- **If selector/card drift appears substantial: also retrain the base selector** (`psa train --force`). When CE score distributions shift significantly across many anchors — e.g., a run where `n_anchors_touched` is a large fraction of the atlas — the cross-encoder's learned weights may also need to move. This is an operator judgment call; Branch 3 does not automate the trigger.

### Two small deliverables on this front

1. `_cmd_atlas_refine_promote` (in `psa/cli.py`) gains one print line at the bottom of successful output:

   > `Run 'psa train --coactivation --force' to recalibrate the selector against the promoted cards.`

2. This spec document names recalibration as the operator's responsibility after any promotion, with both the minimum and escalation path.

No automation on the promote path. Recalibration is expensive; bundling it with promotion would punish operators who promote multiple refinements in sequence.

---

## §6 — CLI surface

### New verb: `psa atlas curate`

```
psa atlas curate [--tenant TENANT] [--extractor heuristic|llm]
```

**Description** (for `--help`): *"Build a candidate refinement from production signals (query fingerprints + oracle labels). Output is never inference-visible; run `psa atlas promote-refinement` to promote."*

**Arguments:**
- `--tenant TENANT` — default `default`. Same resolution as other atlas subcommands.
- `--extractor` — default `heuristic`. Values `heuristic` and `llm`. Passing `llm` in MVP exits non-zero with: *"LLM extractor not implemented yet; see `psa/curation/extractor_llm.py` for the interface contract."*

**Behavior:**
1. Resolves tenant + latest atlas via `TenantManager` + `AtlasManager`.
2. Reads `oracle_labels.jsonl` and `fingerprints.json` from disk.
3. Calls `build_pool(...)` to construct `{anchor_id: Pool}`.
4. For each anchor with a non-empty pool, runs the extractor and merges results into `generated_query_patterns` (deduped, capped).
5. Writes candidate JSON + sibling meta JSON (schema from §4).
6. Prints a summary: oracle labels read, fingerprints read, anchors touched (extractor produced ≥1 new pattern), anchors with non-empty pool but zero new patterns (all extractor output duplicated existing patterns), anchors skipped for empty pool.
7. Prints the promotion hint: *"Run 'psa atlas promote-refinement' to make this candidate inference-visible."*

**Error paths:**
- No atlas → `sys.exit(1)` with tenant-aware error (matches other atlas subcommands).
- Oracle labels file missing → soft: log a warning, run with empty oracle set, produce an empty candidate. (Edge case: the curator runs before `psa label` has been run even once. Correct behavior is to produce no new patterns rather than fail.)
- Fingerprint store missing → same soft behavior. Signal absence is not failure.

---

## Files

### New

| File | Responsibility |
|---|---|
| `psa/curation/__init__.py` | Package marker. |
| `psa/curation/pool.py` | `Pool` dataclass with `oracle_queries: list[str]` and `endorsed_fingerprint_queries: list[str]`. `build_pool(atlas, oracle_labels_path) -> dict[int, Pool]` applies the oracle-endorsed filter (exact text match, MVP). |
| `psa/curation/ngrams.py` | `extract_ngrams(text, min_n=3, max_n=6) -> list[str]` — lifted from `scripts/refine_anchor_cards.py:99`. Stopword list moved here. |
| `psa/curation/extractor_heuristic.py` | `HeuristicExtractor` implementing `QueryPatternExtractor.extract()`. Calls `extract_ngrams` per query, dedupes across pool, sorts longest-first. |
| `psa/curation/extractor_llm.py` | `LLMExtractor` interface-compliant stub. `extract()` raises `NotImplementedError`. Docstring specifies the Qwen prompt contract. |
| `psa/curation/curator.py` | `curate(tenant_id, extractor_name) -> summary_dict`. Orchestrator: loads atlas + signals, builds pool, runs extractor, writes candidate + meta. |
| `tests/test_curation_pool.py` | Tests for `build_pool`: oracle-endorsement filter, exact-match semantics, empty-pool, both-components-populated. |
| `tests/test_curation_heuristic.py` | Tests for `HeuristicExtractor`: dedup, cap, ngram shape, empty pool. |
| `tests/test_cli_atlas_curate.py` | End-to-end CLI test: fixture atlas + oracle labels + fingerprints, run curate, assert candidate + meta written with new metadata fields. |

### Modified

| File | Change |
|---|---|
| `psa/cli.py` | Add `_cmd_atlas_curate` function. Register `psa atlas curate` subparser with `--extractor`. Dispatch. Add promote-hint line to `_cmd_atlas_refine_promote` output. |

### Not touched (explicit)

- `psa/anchor.py` — no changes. `to_stable_card_text()` still ignores fingerprints directly; it sees promoted patterns via `generated_query_patterns` as designed.
- `psa/fingerprints.py` — no changes. Still raw-telemetry-only.
- `psa/pipeline.py` — no changes. Fingerprint accumulation logic stays put.
- `psa/atlas.py` — no changes. `AnchorIndex.load()` still uses the Branch 1 preference order.
- `scripts/refine_anchor_cards.py` — untouched. The ngram logic is lifted into `psa/curation/ngrams.py`, but the script itself keeps its own copy for backward compatibility (explicitly out of scope: deduplicating that copy is a future cleanup).

## Tests to run

- Full suite: `uv run pytest tests/ -q` — all pass.
- Lint: `uv run ruff check . && uv run ruff format --check .` — clean.
- Manual end-to-end (not automated; operator-validation):
  1. Build a tenant, mine sessions, build atlas, label queries, run some searches (populates fingerprints).
  2. Run `psa atlas curate`.
  3. Inspect the candidate + meta files; verify schema.
  4. `psa atlas promote-refinement`.
  5. Confirm the promote output includes the recalibration hint.

## Success criteria

- `psa atlas curate` produces a candidate with correct schema, including the provisional `support_semantics` marker and extractor name.
- A tenant with no oracle labels produces an empty candidate (no per-anchor patterns added), not a crash.
- The `LLMExtractor` stub raises `NotImplementedError` with a message pointing to the file.
- Promote output now names the recalibration command.
- Zero changes to CE-visible card text except through the promotion path.
- Branch 1 candidate/promote flow continues to work for miss-log-driven refinements without regression.

## Follow-ups (explicit deferrals)

- `LLMExtractor` implementation.
- Weighting (oracle-endorsed vs endorsed-fingerprint — include counts/weights in ngram extraction).
- Canonicalized / semantic oracle-match filter (`support_semantics` other than `exact_query_text`).
- Merging source types per candidate (miss-log + signals in one run).
- Selector-retrain escalation criteria (when to run `psa train --force` in addition to `--coactivation --force`).
- Memory scorer role reconsideration — now that the production-signal path exists for card updates, revisit whether the memory scorer needs a production-signal training mode, or whether it becomes redundant.
- Dedup the ngram implementation between `scripts/refine_anchor_cards.py` and `psa/curation/ngrams.py`.
- Observability rollups (Branch 4).
