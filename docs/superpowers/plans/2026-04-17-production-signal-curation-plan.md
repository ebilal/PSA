# Production-Signal Curation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the production-signal curation pipeline: `psa atlas curate` builds a candidate refinement from query fingerprints + oracle labels, feeds it through a pluggable extractor (heuristic for MVP; LLM reserved), and writes a Branch 1 candidate. Never CE-visible without `psa atlas promote-refinement`.

**Architecture:** New `psa/curation/` package with four small modules: `ngrams.py` (pure ngram extraction, lifted from `scripts/refine_anchor_cards.py`), `pool.py` (anchor-level oracle-endorsed filter), `extractor_heuristic.py` + `extractor_llm.py` (behind a `QueryPatternExtractor` Protocol in `__init__.py`), and `curator.py` (orchestrator with empty-run guard). CLI verb `psa atlas curate` wires the pieces together; Branch 1's candidate/promote flow is reused verbatim. Promote output gains one recalibration hint.

**Tech Stack:** Python 3.13, `typing.Protocol`, existing `AtlasManager` / `TenantManager` / `FingerprintStore`.

**Design spec:** `docs/superpowers/specs/2026-04-17-production-signal-curation-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `psa/curation/__init__.py` | Create | Package marker; defines the `QueryPatternExtractor` Protocol (3 lines). |
| `psa/curation/ngrams.py` | Create | `extract_ngrams(text, min_n=3, max_n=6)` pure function + stopword set. Lifted from `scripts/refine_anchor_cards.py:24-126`. |
| `psa/curation/pool.py` | Create | `Pool` dataclass (two list fields). `build_pool(atlas, oracle_labels_path, fingerprint_store)` returns `dict[int, Pool]`, applying anchor-level oracle endorsement. |
| `psa/curation/extractor_heuristic.py` | Create | `HeuristicExtractor` implementing the Protocol. Calls `extract_ngrams` over pool queries, dedupes, caps. |
| `psa/curation/extractor_llm.py` | Create | `LLMExtractor` interface-compliant stub; `extract()` raises `NotImplementedError` with a message pointing at this file. Docstring specifies the Qwen prompt contract. |
| `psa/curation/curator.py` | Create | `curate(tenant_id, extractor_name)` orchestrator. Resolves atlas, builds pools, runs extractor, dedupes against existing `generated_query_patterns`, computes diff stats, applies empty-run guard, writes candidate + meta. |
| `psa/cli.py` | Modify | Add `_cmd_atlas_curate`, subparser, dispatch. Append one recalibration-hint line to `_cmd_atlas_refine_promote` output. |
| `tests/test_curation_ngrams.py` | Create | Tests for `extract_ngrams`: ordering, stopword filter, empty input. |
| `tests/test_curation_pool.py` | Create | Tests for `build_pool`: anchor-level endorsement, empty-pool, lists stay separate. |
| `tests/test_curation_extractors.py` | Create | Tests for `HeuristicExtractor` (dedup, cap, ordering) and `LLMExtractor` stub (raises). |
| `tests/test_curation_curator.py` | Create | Tests for `curate`: happy path writes candidate + meta; empty-run guard skips write; provisional support_semantics value recorded. |
| `tests/test_cli_atlas_curate.py` | Create | End-to-end CLI test: fixture atlas + oracle labels + fingerprint store, run `psa atlas curate`, assert candidate + meta written with new fields. |
| `tests/test_cli_atlas_refine.py` | Modify | Extend the existing promote-command test to assert the new recalibration hint appears in stdout. |

---

### Task 1: `psa/curation/ngrams.py`

Pure ngram extractor + stopword set. No dependencies beyond stdlib. Lifted from `scripts/refine_anchor_cards.py` so the curation package doesn't import from `scripts/`.

**Files:**
- Create: `psa/curation/__init__.py` (empty for now; Task 3 adds the Protocol)
- Create: `psa/curation/ngrams.py`
- Create: `tests/test_curation_ngrams.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_curation_ngrams.py`:

```python
"""Tests for psa.curation.ngrams — pure ngram extraction."""

from __future__ import annotations

from psa.curation.ngrams import extract_ngrams, STOPWORDS


def test_extract_ngrams_returns_longer_patterns_first():
    result = extract_ngrams("how does the token refresh flow work", min_n=3, max_n=6)
    lengths = [len(g.split()) for g in result]
    assert lengths == sorted(lengths, reverse=True), result


def test_extract_ngrams_drops_all_stopword_spans():
    """A span that contains only stopwords is excluded."""
    # "the is a" is all stopwords; should not appear.
    result = extract_ngrams("the is a query about auth", min_n=3, max_n=3)
    assert "the is a" not in result


def test_extract_ngrams_keeps_spans_with_any_content_word():
    """A span with ≥1 non-stopword survives."""
    result = extract_ngrams("the is token refresh", min_n=3, max_n=3)
    assert any("token" in g for g in result)


def test_extract_ngrams_empty_input():
    assert extract_ngrams("", min_n=3, max_n=6) == []


def test_extract_ngrams_shorter_than_min_n():
    """Input with fewer words than min_n yields empty."""
    assert extract_ngrams("too short", min_n=3, max_n=6) == []


def test_stopwords_is_a_frozenset_or_set():
    """STOPWORDS is exposed and a set-like collection."""
    assert "the" in STOPWORDS
    assert "token" not in STOPWORDS
```

- [ ] **Step 2: Run tests — expect ModuleNotFoundError**

```
uv run pytest tests/test_curation_ngrams.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'psa.curation.ngrams'`.

- [ ] **Step 3: Create the empty `psa/curation/__init__.py`**

Create the file with one line:

```python
"""psa.curation — production-signal card curation (fingerprints + oracle labels → candidate refinements)."""
```

- [ ] **Step 4: Create `psa/curation/ngrams.py`**

```python
"""
ngrams.py — pure ngram extraction for card curation.

Lifted from scripts/refine_anchor_cards.py to make the logic importable
from the psa package without reaching into scripts/. The script keeps
its own copy for backward compatibility; deduplicating the two is a
future cleanup (not in scope for Branch 3).
"""

from __future__ import annotations

STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the",
        "is", "was", "were", "be", "been", "being",
        "have", "has", "had",
        "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall", "can",
        "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them",
        "my", "your", "his", "our", "their",
        "this", "that",
        "what", "when", "where", "who", "which", "how", "about",
        "and", "or", "but",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "up", "into", "through",
        "during", "before", "after", "above", "below", "between",
    }
)


def extract_ngrams(text: str, min_n: int = 3, max_n: int = 6) -> list[str]:
    """Extract ngrams of length min_n..max_n from `text`, longer first.

    A span is kept iff at least one of its words is NOT in STOPWORDS.
    The result is a list of space-separated lowercase ngrams; longer
    patterns come first so downstream dedup-then-cap prefers them.
    """
    words = text.lower().split()
    result: list[str] = []
    for n in range(max_n, min_n - 1, -1):
        for i in range(len(words) - n + 1):
            gram = words[i : i + n]
            if any(w not in STOPWORDS for w in gram):
                result.append(" ".join(gram))
    return result
```

- [ ] **Step 5: Run tests — expect PASS**

```
uv run pytest tests/test_curation_ngrams.py -v
```

Expected: all 6 PASS.

- [ ] **Step 6: Commit**

```bash
git add psa/curation/__init__.py psa/curation/ngrams.py tests/test_curation_ngrams.py
git commit -m "$(cat <<'EOF'
feat: add psa.curation.ngrams (pure ngram extractor)

Lifted extract_ngrams + stopword set from scripts/refine_anchor_cards.py
into a proper importable module. First piece of the production-signal
curation pipeline. The script keeps its own copy — dedup is deferred.

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `psa/curation/pool.py`

Per-anchor pool builder with anchor-level oracle endorsement. `Pool` keeps `oracle_queries` and `endorsed_fingerprint_queries` separate even though they're unioned at extraction time — preserves provenance for later weighting work.

**Files:**
- Create: `psa/curation/pool.py`
- Create: `tests/test_curation_pool.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_curation_pool.py`:

```python
"""Tests for psa.curation.pool — anchor-level oracle-endorsement filter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from psa.curation.pool import Pool, build_pool


def _write_oracle_labels(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _fake_atlas(anchor_ids: list[int]) -> MagicMock:
    atlas = MagicMock()
    cards = []
    for aid in anchor_ids:
        c = MagicMock()
        c.anchor_id = aid
        cards.append(c)
    atlas.cards = cards
    atlas.fingerprint_store = MagicMock()
    return atlas


def test_pool_is_a_dataclass_with_two_list_fields():
    p = Pool(oracle_queries=["a", "b"], endorsed_fingerprint_queries=["c"])
    assert p.oracle_queries == ["a", "b"]
    assert p.endorsed_fingerprint_queries == ["c"]


def test_build_pool_collects_oracle_queries_per_winning_anchor(tmp_path):
    atlas = _fake_atlas([1, 2])
    atlas.fingerprint_store.get.return_value = []
    labels_path = tmp_path / "oracle_labels.jsonl"
    _write_oracle_labels(labels_path, [
        {"query": "q1", "winning_oracle_set": [1, 2]},
        {"query": "q2", "winning_oracle_set": [2]},
    ])

    pools = build_pool(atlas, str(labels_path))
    assert pools[1].oracle_queries == ["q1"]
    assert sorted(pools[2].oracle_queries) == ["q1", "q2"]


def test_build_pool_includes_fingerprints_only_when_anchor_is_oracle_endorsed(tmp_path):
    """Anchor-level rule: fingerprints flow in only if the anchor appears in ≥1 winning_oracle_set."""
    atlas = _fake_atlas([1, 2])

    # Anchor 1 has one oracle endorsement; anchor 2 has zero.
    labels_path = tmp_path / "oracle_labels.jsonl"
    _write_oracle_labels(labels_path, [
        {"query": "oracle_q", "winning_oracle_set": [1]},
    ])

    # Fingerprints exist for BOTH anchors.
    def _get(aid):
        return {1: ["fp_for_1"], 2: ["fp_for_2"]}[aid]
    atlas.fingerprint_store.get.side_effect = _get

    pools = build_pool(atlas, str(labels_path))
    assert pools[1].endorsed_fingerprint_queries == ["fp_for_1"]
    assert pools[2].endorsed_fingerprint_queries == []
    assert pools[2].oracle_queries == []


def test_build_pool_empty_when_no_signal(tmp_path):
    """No oracle labels, no fingerprints → every anchor has an empty Pool."""
    atlas = _fake_atlas([1, 2])
    atlas.fingerprint_store.get.return_value = []

    labels_path = tmp_path / "oracle_labels.jsonl"
    _write_oracle_labels(labels_path, [])

    pools = build_pool(atlas, str(labels_path))
    for aid in [1, 2]:
        assert pools[aid].oracle_queries == []
        assert pools[aid].endorsed_fingerprint_queries == []


def test_build_pool_returns_entry_for_every_atlas_anchor(tmp_path):
    """Even anchors with no signal get a Pool key (with empty lists)."""
    atlas = _fake_atlas([1, 2, 3])
    atlas.fingerprint_store.get.return_value = []
    labels_path = tmp_path / "oracle_labels.jsonl"
    _write_oracle_labels(labels_path, [{"query": "q", "winning_oracle_set": [1]}])

    pools = build_pool(atlas, str(labels_path))
    assert set(pools.keys()) == {1, 2, 3}


def test_build_pool_missing_labels_file_returns_empty_pools(tmp_path):
    """Soft: missing oracle_labels.jsonl → every anchor's Pool is empty."""
    atlas = _fake_atlas([1])
    atlas.fingerprint_store.get.return_value = ["fp_q"]

    pools = build_pool(atlas, str(tmp_path / "does_not_exist.jsonl"))
    assert pools[1].oracle_queries == []
    # Fingerprints require oracle endorsement; without any oracle labels at
    # all, the anchor is NOT endorsed, so fingerprints don't flow in.
    assert pools[1].endorsed_fingerprint_queries == []
```

- [ ] **Step 2: Run tests — expect ModuleNotFoundError**

```
uv run pytest tests/test_curation_pool.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'psa.curation.pool'`.

- [ ] **Step 3: Create `psa/curation/pool.py`**

```python
"""
pool.py — per-anchor query pool construction with anchor-level oracle endorsement.

Two signal sources feed a Pool:
- oracle_queries: queries from oracle_labels.jsonl where this anchor is in
  winning_oracle_set. Normative signal.
- endorsed_fingerprint_queries: fingerprints for this anchor, included iff the
  anchor has appeared in ≥1 winning_oracle_set (anchor-level endorsement).
  Observed signal, minimally vetted.

The two lists stay separate even though curator.py unions them at extraction
time — preserves provenance for later weighting / diagnostics.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("psa.curation.pool")


@dataclass
class Pool:
    """Per-anchor curated query pool.

    Both fields remain separate in the internal data model; curator unions
    them before extraction. Future weighting / diagnostics work will need
    the split.
    """

    oracle_queries: list[str] = field(default_factory=list)
    endorsed_fingerprint_queries: list[str] = field(default_factory=list)


def build_pool(atlas: Any, oracle_labels_path: str) -> dict[int, Pool]:
    """Build a Pool per anchor in the atlas.

    Parameters
    ----------
    atlas:
        An Atlas instance with .cards (iterable of objects with .anchor_id) and
        .fingerprint_store (a FingerprintStore providing .get(anchor_id)).
    oracle_labels_path:
        Path to oracle_labels.jsonl. Missing file is soft — returns empty pools.

    Returns
    -------
    dict mapping anchor_id → Pool for every anchor in the atlas. Anchors with
    no signal get a Pool with two empty lists (not absent from the dict).
    """
    pools: dict[int, Pool] = {c.anchor_id: Pool() for c in atlas.cards}

    # Pass 1: oracle labels. Also determine which anchors have ≥1 endorsement.
    endorsed_anchors: set[int] = set()
    if os.path.exists(oracle_labels_path):
        with open(oracle_labels_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    label = json.loads(line)
                except json.JSONDecodeError:
                    continue
                query = label.get("query")
                winning = label.get("winning_oracle_set") or []
                if not query:
                    continue
                for aid in winning:
                    if aid in pools:
                        pools[aid].oracle_queries.append(query)
                        endorsed_anchors.add(aid)
    else:
        logger.warning("Oracle labels file not found: %s", oracle_labels_path)

    # Pass 2: fingerprints — only for anchors with ≥1 oracle endorsement.
    fp_store = atlas.fingerprint_store
    for aid in pools:
        if aid not in endorsed_anchors:
            continue
        fps = fp_store.get(aid) if fp_store is not None else []
        pools[aid].endorsed_fingerprint_queries.extend(fps)

    return pools
```

- [ ] **Step 4: Run tests — expect PASS**

```
uv run pytest tests/test_curation_pool.py -v
```

Expected: all 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add psa/curation/pool.py tests/test_curation_pool.py
git commit -m "$(cat <<'EOF'
feat: add psa.curation.pool with anchor-level oracle endorsement

Pool dataclass keeps oracle_queries and endorsed_fingerprint_queries
separate (curator unions them at extraction time but provenance is
preserved for later weighting work).

build_pool reads oracle_labels.jsonl and the atlas's FingerprintStore.
An anchor gets fingerprints in its pool only if it appears in ≥1
winning_oracle_set — conservative-but-permissive MVP rule, labeled
"anchor_level_oracle_endorsement" in candidate metadata (see spec).
Missing oracle_labels.jsonl is soft; curator's empty-run guard handles
the downstream consequence.

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `QueryPatternExtractor` Protocol + `HeuristicExtractor`

The Protocol lives in `psa/curation/__init__.py` so consumers can write `from psa.curation import QueryPatternExtractor`. Heuristic extractor is the MVP backend.

**Files:**
- Modify: `psa/curation/__init__.py` (add the Protocol)
- Create: `psa/curation/extractor_heuristic.py`
- Create: `tests/test_curation_extractors.py` (covers both heuristic here and the stub in Task 4)

- [ ] **Step 1: Write the failing tests for the heuristic extractor**

Create `tests/test_curation_extractors.py`:

```python
"""Tests for psa.curation extractors — heuristic (MVP) and LLM (stub)."""

from __future__ import annotations

import pytest

from psa.curation.extractor_heuristic import HeuristicExtractor


def test_heuristic_extractor_produces_ngrams_from_pool():
    extractor = HeuristicExtractor()
    pool = [
        "how does the token refresh flow work",
        "what is the expiry for an access token",
    ]
    out = extractor.extract(pool, n=10)
    assert isinstance(out, list)
    assert len(out) <= 10
    assert all(isinstance(p, str) for p in out)
    # Should include an obviously content-bearing ngram.
    assert any("token" in p for p in out)


def test_heuristic_extractor_respects_cap():
    extractor = HeuristicExtractor()
    # Long query with many possible ngrams.
    pool = ["one two three four five six seven eight nine ten"]
    out = extractor.extract(pool, n=3)
    assert len(out) == 3


def test_heuristic_extractor_dedupes_across_pool():
    extractor = HeuristicExtractor()
    pool = [
        "refresh access token flow",
        "refresh access token flow",   # exact duplicate query
    ]
    out = extractor.extract(pool, n=50)
    # No repeated ngrams across the two identical queries.
    assert len(out) == len(set(out))


def test_heuristic_extractor_empty_pool():
    extractor = HeuristicExtractor()
    assert extractor.extract([], n=10) == []


def test_heuristic_extractor_prefers_longer_patterns():
    extractor = HeuristicExtractor()
    pool = ["how does the authentication token refresh flow work under load"]
    out = extractor.extract(pool, n=3)
    # The first (kept) pattern should be among the longest available.
    max_len = max(len(p.split()) for p in out)
    assert len(out[0].split()) == max_len
```

- [ ] **Step 2: Run tests — expect ModuleNotFoundError**

```
uv run pytest tests/test_curation_extractors.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'psa.curation.extractor_heuristic'`.

- [ ] **Step 3: Add the Protocol to `psa/curation/__init__.py`**

Replace the contents of `psa/curation/__init__.py` with:

```python
"""psa.curation — production-signal card curation (fingerprints + oracle labels → candidate refinements)."""

from __future__ import annotations

from typing import Protocol


class QueryPatternExtractor(Protocol):
    """Extractor interface: pool of queries → candidate query patterns.

    Concrete backends live in `extractor_heuristic.py` (MVP, ngrams)
    and `extractor_llm.py` (reserved; stub raises NotImplementedError).
    """

    def extract(self, pool: list[str], n: int) -> list[str]:
        """Return up to `n` candidate query patterns derived from the pool."""
        ...
```

- [ ] **Step 4: Create `psa/curation/extractor_heuristic.py`**

```python
"""
extractor_heuristic.py — HeuristicExtractor: ngram-based query pattern extraction.

MVP backend for psa.curation. Zero LLM cost; deterministic. Output feeds
into generated_query_patterns via the curator after dedup against existing
patterns.
"""

from __future__ import annotations

from .ngrams import extract_ngrams


class HeuristicExtractor:
    """Ngram-based extractor.

    For each query in the pool, extracts 3–6 word ngrams (stopword-filtered,
    longer-first), unions across the pool preserving first-seen order, and
    returns up to `n` distinct patterns.
    """

    def extract(self, pool: list[str], n: int) -> list[str]:
        if n <= 0 or not pool:
            return []
        seen: set[str] = set()
        ordered: list[str] = []
        for query in pool:
            for gram in extract_ngrams(query):
                if gram not in seen:
                    seen.add(gram)
                    ordered.append(gram)
                    if len(ordered) >= n:
                        return ordered
        return ordered
```

- [ ] **Step 5: Run tests — expect PASS**

```
uv run pytest tests/test_curation_extractors.py -v
```

Expected: all 5 PASS.

- [ ] **Step 6: Commit**

```bash
git add psa/curation/__init__.py psa/curation/extractor_heuristic.py tests/test_curation_extractors.py
git commit -m "$(cat <<'EOF'
feat: add QueryPatternExtractor Protocol + HeuristicExtractor

Protocol lives in psa/curation/__init__.py for easy import. Heuristic
backend is the MVP: extract_ngrams per query, dedupe across pool
preserving order, cap at n. Longer patterns first by construction.

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `LLMExtractor` stub

Interface-compliant stub. Documents the architectural commitment; does not implement.

**Files:**
- Create: `psa/curation/extractor_llm.py`
- Modify: `tests/test_curation_extractors.py` (add one test)

- [ ] **Step 1: Append the failing test**

Append to `tests/test_curation_extractors.py`:

```python
def test_llm_extractor_stub_raises_notimplemented():
    from psa.curation.extractor_llm import LLMExtractor

    extractor = LLMExtractor()
    with pytest.raises(NotImplementedError, match="extractor_llm.py"):
        extractor.extract(["some query"], n=5)
```

- [ ] **Step 2: Run the new test — expect ModuleNotFoundError**

```
uv run pytest tests/test_curation_extractors.py::test_llm_extractor_stub_raises_notimplemented -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'psa.curation.extractor_llm'`.

- [ ] **Step 3: Create `psa/curation/extractor_llm.py`**

```python
"""
extractor_llm.py — LLMExtractor: Qwen-based query pattern distillation (stub).

Architectural commitment, not implemented. The interface is settled; the body
is reserved for follow-up work once the MVP heuristic backend has been
validated on real tenants.

Intended behavior (for the future implementation):
    One Qwen call per anchor. The pool (oracle queries + endorsed fingerprints)
    is rendered into the prompt as a numbered list. Qwen returns up to `n`
    representative question patterns that this anchor can answer.

Prompt contract (tentative — subject to refinement when implemented):

    "You are curating an anchor description in a semantic memory index.
    Below are N real user queries that this anchor has answered and been
    endorsed for. Produce up to {n} representative question patterns that
    characterize what this anchor can answer. Prefer coverage over
    paraphrase. Return one pattern per line, no numbering, no commentary."

Why a stub:
    Branch 3 ships the pluggable architecture (Protocol in __init__.py,
    heuristic backend). Committing to the LLM interface now means a future
    branch can drop in the implementation without redesigning the contract.
"""

from __future__ import annotations


class LLMExtractor:
    """Qwen-based query pattern distillation — not yet implemented.

    Conforms to the QueryPatternExtractor Protocol by method signature; the
    body raises NotImplementedError with a pointer to this file.
    """

    def extract(self, pool: list[str], n: int) -> list[str]:
        raise NotImplementedError(
            "LLM extractor reserved for follow-up; see psa/curation/extractor_llm.py "
            "for the interface contract."
        )
```

- [ ] **Step 4: Run tests — expect PASS**

```
uv run pytest tests/test_curation_extractors.py -v
```

Expected: all 6 PASS (the 5 heuristic tests plus the new stub test).

- [ ] **Step 5: Commit**

```bash
git add psa/curation/extractor_llm.py tests/test_curation_extractors.py
git commit -m "$(cat <<'EOF'
feat: add LLMExtractor stub (Protocol-compliant NotImplementedError)

Commits to the pluggable-extractor architecture without implementing
the LLM backend. Docstring captures the tentative Qwen prompt contract
for whichever future branch picks this up. Test locks the stub shape
(raises with a pointer to the file).

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `curator.py` orchestrator with empty-run guard

Top-level function `curate(tenant_id, extractor_name)`. Loads tenant/atlas, builds pool, runs extractor, computes diff stats, applies empty-run guard, writes candidate + meta.

**Files:**
- Create: `psa/curation/curator.py`
- Create: `tests/test_curation_curator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_curation_curator.py`:

```python
"""Tests for psa.curation.curator — orchestrator with empty-run guard."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _write_atlas_dir(
    atlas_dir: Path,
    anchor_ids: list[int],
    patterns: dict[int, list[str]],
) -> None:
    """Write a minimal valid atlas_vN directory that AnchorIndex.load can read."""
    atlas_dir.mkdir(parents=True, exist_ok=True)
    cards = [
        {
            "anchor_id": aid,
            "name": f"anchor-{aid}",
            "meaning": "test",
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
            "generated_query_patterns": patterns.get(aid, []),
            "query_fingerprint": [],
        }
        for aid in anchor_ids
    ]
    (atlas_dir / "anchor_cards.json").write_text(json.dumps(cards))
    np.save(atlas_dir / "centroids.npy", np.zeros((len(anchor_ids), 768), dtype=np.float32))
    (atlas_dir / "atlas_meta.json").write_text(
        json.dumps(
            {
                "version": 1,
                "tenant_id": "test",
                "stats": {
                    "n_learned": len(anchor_ids),
                    "n_novelty": 0,
                    "n_total": len(anchor_ids),
                    "coverage": 1.0,
                    "novelty_rate": 0.0,
                    "utilization_skew": 0.0,
                },
            }
        )
    )


def _write_oracle_labels(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_fingerprints(atlas_dir: Path, data: dict[int, list[str]]) -> None:
    with open(atlas_dir / "fingerprints.json", "w") as f:
        json.dump({str(k): v for k, v in data.items()}, f)


def test_curate_happy_path_writes_candidate_and_meta(tmp_path, monkeypatch):
    from psa.curation.curator import curate

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas_dir(atlas_dir, [1, 2], patterns={1: [], 2: []})

    labels_path = tenant_dir / "training" / "oracle_labels.jsonl"
    _write_oracle_labels(labels_path, [
        {"query": "how does the token refresh flow work", "winning_oracle_set": [1]},
        {"query": "what is the user session expiry", "winning_oracle_set": [2]},
    ])
    _write_fingerprints(atlas_dir, {1: ["how does the token refresh flow work"], 2: []})

    summary = curate(tenant_id="default", extractor_name="heuristic")

    cand_path = atlas_dir / "anchor_cards_candidate.json"
    meta_path = atlas_dir / "anchor_cards_candidate.meta.json"
    assert cand_path.exists()
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text())
    assert meta["source"] == "production_signal"
    assert meta["extractor"] == "heuristic"
    assert meta["support_semantics"] == "anchor_level_oracle_endorsement"
    assert meta["promoted"] is False
    assert meta["oracle_labels_read"] == 2
    assert meta["n_anchors_with_oracle_support"] == 2
    assert meta["n_anchors_touched"] >= 1
    assert meta["n_patterns_added"] >= 1

    # Summary mirrors what was written.
    assert summary["n_anchors_touched"] == meta["n_anchors_touched"]
    assert summary["n_patterns_added"] == meta["n_patterns_added"]


def test_curate_skips_write_when_no_oracle_labels(tmp_path, monkeypatch, capsys):
    """Empty-run guard: no labels → no candidate written, no in-flight clobber."""
    from psa.curation.curator import curate

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas_dir(atlas_dir, [1], patterns={1: []})

    # Deliberately: no oracle_labels.jsonl and no fingerprints.json.

    # Simulate an in-flight miss-log candidate already sitting in the atlas dir.
    (atlas_dir / "anchor_cards_candidate.json").write_text('[{"in_flight": true}]')
    (atlas_dir / "anchor_cards_candidate.meta.json").write_text('{"source": "manual"}')

    summary = curate(tenant_id="default", extractor_name="heuristic")

    # In-flight candidate is untouched.
    assert json.loads((atlas_dir / "anchor_cards_candidate.json").read_text()) == [
        {"in_flight": True}
    ]
    assert json.loads((atlas_dir / "anchor_cards_candidate.meta.json").read_text()) == {
        "source": "manual"
    }

    assert summary["wrote_candidate"] is False
    assert summary["n_anchors_touched"] == 0
    assert summary["n_patterns_added"] == 0


def test_curate_skips_write_when_all_patterns_duplicate(tmp_path, monkeypatch):
    """Empty-run guard fires when oracle labels exist but ngrams all duplicate existing patterns."""
    from psa.curation.curator import curate

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"

    # Pre-seed generated_query_patterns with an ngram that WILL be extracted
    # from the oracle query below, so extractor output fully duplicates.
    seeded = ["refresh flow work"]  # substring of the oracle query; will be an extracted ngram
    # Add ALL possible extracted ngrams too, to guarantee 0 new patterns.
    from psa.curation.ngrams import extract_ngrams
    query_text = "how does the token refresh flow work"
    seeded = extract_ngrams(query_text)

    _write_atlas_dir(atlas_dir, [1], patterns={1: seeded})

    labels_path = tenant_dir / "training" / "oracle_labels.jsonl"
    _write_oracle_labels(labels_path, [
        {"query": query_text, "winning_oracle_set": [1]},
    ])

    summary = curate(tenant_id="default", extractor_name="heuristic")

    # No candidate file created; all extractor output duplicated existing patterns.
    assert not (atlas_dir / "anchor_cards_candidate.json").exists()
    assert not (atlas_dir / "anchor_cards_candidate.meta.json").exists()
    assert summary["wrote_candidate"] is False


def test_curate_rejects_unknown_extractor(tmp_path, monkeypatch):
    from psa.curation.curator import curate

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas_dir(atlas_dir, [1], patterns={1: []})

    import pytest
    with pytest.raises(ValueError, match="extractor"):
        curate(tenant_id="default", extractor_name="bogus")


def test_curate_llm_extractor_raises_notimplemented(tmp_path, monkeypatch):
    from psa.curation.curator import curate

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas_dir(atlas_dir, [1], patterns={1: []})

    labels_path = tenant_dir / "training" / "oracle_labels.jsonl"
    _write_oracle_labels(labels_path, [
        {"query": "a query", "winning_oracle_set": [1]},
    ])

    import pytest
    with pytest.raises(NotImplementedError):
        curate(tenant_id="default", extractor_name="llm")
```

- [ ] **Step 2: Run tests — expect ModuleNotFoundError**

```
uv run pytest tests/test_curation_curator.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'psa.curation.curator'`.

- [ ] **Step 3: Create `psa/curation/curator.py`**

```python
"""
curator.py — production-signal curation orchestrator.

curate(tenant_id, extractor_name) loads the current atlas and its
FingerprintStore, builds per-anchor Pools, runs the chosen extractor,
dedupes extractor output against existing generated_query_patterns, and
writes a Branch 1 candidate + sibling .meta.json — but only if the run
produced real changes (empty-run guard).

Never writes to anchor_cards_refined.json. Promotion stays under
`psa atlas promote-refinement`.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from pathlib import Path
from typing import Any

from .extractor_heuristic import HeuristicExtractor
from .extractor_llm import LLMExtractor
from .pool import build_pool

logger = logging.getLogger("psa.curation.curator")

MAX_PATTERNS_PER_ANCHOR = 20
SUPPORT_SEMANTICS = "anchor_level_oracle_endorsement"


def _make_extractor(name: str):
    if name == "heuristic":
        return HeuristicExtractor()
    if name == "llm":
        return LLMExtractor()
    raise ValueError(
        f"Unknown extractor: {name!r}. Must be 'heuristic' or 'llm'."
    )


def curate(tenant_id: str = "default", extractor_name: str = "heuristic") -> dict[str, Any]:
    """Run a curation pass for `tenant_id` using the named extractor.

    Returns a summary dict with keys:
        - wrote_candidate: bool
        - n_anchors_touched: int
        - n_patterns_added: int
        - oracle_labels_read: int
        - fingerprints_read: int
        - n_anchors_with_oracle_support: int
        - n_anchors_with_endorsed_fingerprints: int
        - extractor: str
        - support_semantics: str
        - atlas_version: int
        - candidate_path: str | None
        - reason: str | None  (populated when wrote_candidate is False)
    """
    from ..atlas import AtlasManager
    from ..tenant import TenantManager

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
    atlas = mgr.get_atlas()
    if atlas is None:
        raise FileNotFoundError(
            f"No atlas for tenant '{tenant_id}'. Run 'psa atlas build' first."
        )

    extractor = _make_extractor(extractor_name)

    labels_path = os.path.join(tenant.root_dir, "training", "oracle_labels.jsonl")
    pools = build_pool(atlas, labels_path)

    # Count signals (observational metrics for metadata).
    oracle_labels_read = sum(1 for p in pools.values() for _ in p.oracle_queries)
    fingerprints_read = sum(1 for p in pools.values() for _ in p.endorsed_fingerprint_queries)
    n_anchors_with_oracle_support = sum(1 for p in pools.values() if p.oracle_queries)
    n_anchors_with_endorsed_fingerprints = sum(
        1 for p in pools.values() if p.endorsed_fingerprint_queries
    )

    # Build refined card list (whether or not we ultimately write it).
    cards_by_id = {c.anchor_id: c for c in atlas.cards}
    refined_cards: list[dict] = []
    n_anchors_touched = 0
    n_patterns_added = 0

    for card in atlas.cards:
        pool = pools.get(card.anchor_id)
        existing = list(card.generated_query_patterns or [])
        new_patterns: list[str] = []
        if pool is not None:
            combined_pool = pool.oracle_queries + pool.endorsed_fingerprint_queries
            available_slots = max(0, MAX_PATTERNS_PER_ANCHOR - len(existing))
            if available_slots > 0 and combined_pool:
                candidate_grams = extractor.extract(combined_pool, n=available_slots * 4)
                existing_set = set(existing)
                for gram in candidate_grams:
                    if gram not in existing_set:
                        new_patterns.append(gram)
                        existing_set.add(gram)
                        if len(new_patterns) >= available_slots:
                            break

        if new_patterns:
            n_anchors_touched += 1
            n_patterns_added += len(new_patterns)

        merged_patterns = existing + new_patterns
        refined_cards.append(_card_to_dict(card, merged_patterns))

    # Empty-run guard: do not clobber an in-flight candidate.
    atlas_dir = Path(atlas.anchor_dir)
    candidate_path = atlas_dir / "anchor_cards_candidate.json"
    meta_path = atlas_dir / "anchor_cards_candidate.meta.json"

    summary: dict[str, Any] = {
        "wrote_candidate": False,
        "n_anchors_touched": n_anchors_touched,
        "n_patterns_added": n_patterns_added,
        "oracle_labels_read": oracle_labels_read,
        "fingerprints_read": fingerprints_read,
        "n_anchors_with_oracle_support": n_anchors_with_oracle_support,
        "n_anchors_with_endorsed_fingerprints": n_anchors_with_endorsed_fingerprints,
        "extractor": extractor_name,
        "support_semantics": SUPPORT_SEMANTICS,
        "atlas_version": atlas.version,
        "candidate_path": None,
        "reason": None,
    }

    if n_anchors_touched == 0 or n_patterns_added == 0:
        if n_anchors_with_oracle_support == 0:
            summary["reason"] = "no oracle-endorsed anchors (run 'psa label' first)"
        else:
            summary["reason"] = "extractor produced no new patterns (all duplicated existing)"
        logger.warning("Curation skipped write: %s", summary["reason"])
        return summary

    with open(candidate_path, "w") as f:
        json.dump(refined_cards, f, indent=2)

    meta = {
        "source": "production_signal",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "tenant_id": tenant_id,
        "atlas_version": atlas.version,
        "promoted": False,
        "promoted_at": None,
        "n_anchors_touched": n_anchors_touched,
        "n_patterns_added": n_patterns_added,
        "oracle_labels_read": oracle_labels_read,
        "fingerprints_read": fingerprints_read,
        "n_anchors_with_oracle_support": n_anchors_with_oracle_support,
        "n_anchors_with_endorsed_fingerprints": n_anchors_with_endorsed_fingerprints,
        "extractor": extractor_name,
        "support_semantics": SUPPORT_SEMANTICS,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    summary["wrote_candidate"] = True
    summary["candidate_path"] = str(candidate_path)
    return summary


def _card_to_dict(card: Any, merged_patterns: list[str]) -> dict:
    """Serialize an AnchorCard to the JSON dict shape used on disk.

    Uses the card's own to_dict() when available, otherwise constructs the
    expected shape field by field. The result's generated_query_patterns is
    overwritten with `merged_patterns`.
    """
    if hasattr(card, "to_dict"):
        d = card.to_dict()
    else:
        # Fallback: reconstruct from known fields (keeps the curator testable
        # with MagicMock cards that don't implement to_dict).
        d = {
            "anchor_id": card.anchor_id,
            "name": card.name,
            "meaning": card.meaning,
            "memory_types": list(getattr(card, "memory_types", [])),
            "include_terms": list(getattr(card, "include_terms", [])),
            "exclude_terms": list(getattr(card, "exclude_terms", [])),
            "prototype_examples": list(getattr(card, "prototype_examples", [])),
            "near_but_different": list(getattr(card, "near_but_different", [])),
            "centroid": list(getattr(card, "centroid", [])),
            "memory_count": getattr(card, "memory_count", 0),
            "is_novelty": getattr(card, "is_novelty", False),
            "status": getattr(card, "status", "active"),
            "metadata": dict(getattr(card, "metadata", {})),
            "generated_query_patterns": list(getattr(card, "generated_query_patterns", [])),
            "query_fingerprint": list(getattr(card, "query_fingerprint", [])),
        }
    d["generated_query_patterns"] = merged_patterns
    return d
```

Note: `AnchorCard.to_dict()` may not exist in the current codebase. If so, the `_card_to_dict` fallback path is what the curator actually uses. Verify by checking `psa/anchor.py`. If no `to_dict` is present, the tests pass via the fallback branch — that's intentional.

- [ ] **Step 4: Run tests — expect PASS**

```
uv run pytest tests/test_curation_curator.py -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add psa/curation/curator.py tests/test_curation_curator.py
git commit -m "$(cat <<'EOF'
feat: add psa.curation.curator orchestrator

curate(tenant_id, extractor_name) is the production-signal curation
entry point. Resolves atlas, builds Pool per anchor, runs the chosen
extractor, dedupes against existing generated_query_patterns, and
writes a Branch 1 candidate + meta — guarded so empty/no-op runs
never clobber an in-flight candidate.

Metadata records extractor="heuristic|llm" and the provisional
support_semantics="anchor_level_oracle_endorsement" marker.

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: CLI wiring — `psa atlas curate` + promote-hint

Add the new subcommand, dispatch, and the one-line recalibration hint on the existing promote command.

**Files:**
- Modify: `psa/cli.py`
- Modify: `tests/test_cli_atlas_refine.py`
- Create: `tests/test_cli_atlas_curate.py`

- [ ] **Step 1: Write the failing e2e test**

Create `tests/test_cli_atlas_curate.py`:

```python
"""End-to-end CLI test for `psa atlas curate`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def _write_atlas(atlas_dir: Path, anchor_ids: list[int], patterns: dict[int, list[str]]) -> None:
    atlas_dir.mkdir(parents=True, exist_ok=True)
    cards = [
        {
            "anchor_id": aid,
            "name": f"anchor-{aid}",
            "meaning": "test",
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
            "generated_query_patterns": patterns.get(aid, []),
            "query_fingerprint": [],
        }
        for aid in anchor_ids
    ]
    (atlas_dir / "anchor_cards.json").write_text(json.dumps(cards))
    np.save(atlas_dir / "centroids.npy", np.zeros((len(anchor_ids), 768), dtype=np.float32))
    (atlas_dir / "atlas_meta.json").write_text(
        json.dumps({
            "version": 1,
            "tenant_id": "test",
            "stats": {
                "n_learned": len(anchor_ids), "n_novelty": 0, "n_total": len(anchor_ids),
                "coverage": 1.0, "novelty_rate": 0.0, "utilization_skew": 0.0,
            },
        })
    )


def test_cli_atlas_curate_writes_candidate(tmp_path, monkeypatch, capsys):
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [1], patterns={1: []})

    labels = tenant_dir / "training" / "oracle_labels.jsonl"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text(
        json.dumps({"query": "how does the auth token refresh flow work", "winning_oracle_set": [1]}) + "\n"
    )
    with open(atlas_dir / "fingerprints.json", "w") as f:
        json.dump({"1": ["how does the auth token refresh flow work"]}, f)

    with patch("sys.argv", ["psa", "atlas", "curate"]):
        main()

    cand = atlas_dir / "anchor_cards_candidate.json"
    meta = atlas_dir / "anchor_cards_candidate.meta.json"
    assert cand.exists()
    meta_obj = json.loads(meta.read_text())
    assert meta_obj["source"] == "production_signal"
    assert meta_obj["extractor"] == "heuristic"
    assert meta_obj["support_semantics"] == "anchor_level_oracle_endorsement"


def test_cli_atlas_curate_llm_flag_errors_in_mvp(tmp_path, monkeypatch, capsys):
    """Passing --extractor llm in MVP exits non-zero with a pointer to the stub file."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [1], patterns={1: []})

    labels = tenant_dir / "training" / "oracle_labels.jsonl"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text(json.dumps({"query": "q", "winning_oracle_set": [1]}) + "\n")

    with patch("sys.argv", ["psa", "atlas", "curate", "--extractor", "llm"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code != 0
    out = capsys.readouterr().out
    assert "extractor_llm.py" in out


def test_cli_atlas_curate_empty_run_skips_write(tmp_path, monkeypatch, capsys):
    """No oracle labels → no candidate written; existing in-flight candidate preserved."""
    from psa.cli import main

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    tenant_dir = home / ".psa" / "tenants" / "default"
    atlas_dir = tenant_dir / "atlas_v1"
    _write_atlas(atlas_dir, [1], patterns={1: []})

    # In-flight miss-log candidate sitting there.
    (atlas_dir / "anchor_cards_candidate.json").write_text('[{"sentinel": true}]')
    (atlas_dir / "anchor_cards_candidate.meta.json").write_text('{"source": "manual"}')

    with patch("sys.argv", ["psa", "atlas", "curate"]):
        main()

    # Untouched.
    assert json.loads((atlas_dir / "anchor_cards_candidate.json").read_text()) == [
        {"sentinel": True}
    ]
    out = capsys.readouterr().out
    assert "skip" in out.lower() or "no oracle" in out.lower()
```

- [ ] **Step 2: Add the failing promote-hint assertion to the existing promote test**

In `tests/test_cli_atlas_refine.py`, find `test_promote_refinement_creates_refined_and_meta`. Immediately after the existing assertions inside that test, append:

```python
    out = capsys.readouterr().out
    assert "psa train --coactivation --force" in out, (
        "promote output must name the recalibration command"
    )
```

- [ ] **Step 3: Run tests — expect failures**

```
uv run pytest tests/test_cli_atlas_curate.py tests/test_cli_atlas_refine.py::test_promote_refinement_creates_refined_and_meta -v
```

Expected:
- `test_cli_atlas_curate_writes_candidate` FAILS with "invalid choice: 'curate'".
- `test_cli_atlas_curate_llm_flag_errors_in_mvp` FAILS same way.
- `test_cli_atlas_curate_empty_run_skips_write` FAILS same way.
- `test_promote_refinement_creates_refined_and_meta` FAILS because the recalibration hint is missing.

- [ ] **Step 4: Register the subparser in `psa/cli.py`**

Find the block that registers `promote-refinement` under `atlas_sub`. Immediately after it, add:

```python
    p_atlas_curate = atlas_sub.add_parser(
        "curate",
        help=(
            "Build a candidate refinement from production signals "
            "(query fingerprints + oracle labels). Output is never "
            "inference-visible; run `psa atlas promote-refinement` to promote."
        ),
    )
    p_atlas_curate.add_argument(
        "--extractor",
        default="heuristic",
        choices=["heuristic", "llm"],
        help=(
            "Extractor backend. 'heuristic' (default) uses ngram extraction. "
            "'llm' is a reserved architectural slot; passing it in MVP exits "
            "non-zero with a pointer to psa/curation/extractor_llm.py."
        ),
    )
```

- [ ] **Step 5: Add dispatch in `cmd_atlas()`**

Find `cmd_atlas()`. Update the dispatch block to include `curate`:

```python
def cmd_atlas(args):
    """Handle 'psa atlas <subcommand>'."""
    action = getattr(args, "atlas_action", None)
    if not action:
        print("Usage: psa atlas {build,status,health,refine,promote-refinement,curate}")
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
    elif action == "curate":
        _cmd_atlas_curate(args)
```

- [ ] **Step 6: Implement `_cmd_atlas_curate()`**

Add this function at `psa/cli.py` immediately after `_cmd_atlas_refine_promote`:

```python
def _cmd_atlas_curate(args):
    """Run production-signal curation for the current atlas."""
    from .curation.curator import curate

    tenant_id = getattr(args, "tenant", "default")
    extractor_name = getattr(args, "extractor", "heuristic")

    try:
        summary = curate(tenant_id=tenant_id, extractor_name=extractor_name)
    except NotImplementedError as e:
        # LLM extractor stub — message already names extractor_llm.py
        print(f"  Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        sys.exit(1)

    print(
        f"  Oracle labels read: {summary['oracle_labels_read']}, "
        f"fingerprints read: {summary['fingerprints_read']}"
    )
    print(
        f"  Anchors with oracle support: {summary['n_anchors_with_oracle_support']}, "
        f"with endorsed fingerprints: {summary['n_anchors_with_endorsed_fingerprints']}"
    )

    if not summary["wrote_candidate"]:
        print(f"  Skipped candidate write: {summary['reason']}")
        return

    print(
        f"  Anchors touched: {summary['n_anchors_touched']}, "
        f"patterns added: {summary['n_patterns_added']}"
    )
    print(f"  Candidate written to {summary['candidate_path']}")
    print(f"  Extractor: {summary['extractor']}, "
          f"support_semantics: {summary['support_semantics']}")
    print("  Run 'psa atlas promote-refinement' to make this candidate inference-visible.")
```

- [ ] **Step 7: Add the recalibration hint to `_cmd_atlas_refine_promote`**

In `_cmd_atlas_refine_promote`, immediately before the function returns (after the existing `print(f"  Source: ...")` line), add:

```python
    print("  Run 'psa train --coactivation --force' to recalibrate the selector against the promoted cards.")
```

- [ ] **Step 8: Run the tests — expect PASS**

```
uv run pytest tests/test_cli_atlas_curate.py tests/test_cli_atlas_refine.py -v
```

Expected: all PASS (3 new curate tests + all 9 refine/promote tests from Branch 1 still green).

- [ ] **Step 9: Commit**

```bash
git add psa/cli.py tests/test_cli_atlas_curate.py tests/test_cli_atlas_refine.py
git commit -m "$(cat <<'EOF'
feat: add `psa atlas curate` CLI + recalibration hint on promote

New subcommand wires the production-signal curation pipeline: resolve
tenant/atlas, run curate() with the chosen extractor, print a summary
that names what was read vs. what was written, respect the empty-run
guard that protects in-flight candidates.

`--extractor llm` in MVP exits non-zero pointing at the stub file;
`--extractor heuristic` is the default.

Also: `psa atlas promote-refinement` now names the recalibration
command in its success output so operators know the next step.

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Full test suite + lint

- [ ] **Step 1: Run the full test suite**

```
uv run pytest tests/ -q
```

Expected: all PASS. The flaky network-dependent `test_convo_miner::test_convo_mining` is acceptable if it's the only failure (pre-existing, confirmed passes in isolation).

- [ ] **Step 2: Lint + format check**

```
uv run ruff check .
uv run ruff format --check .
```

Expected: both clean. If `ruff format --check` reports drift, run `uv run ruff format .` and commit the result.

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
| §1 Signal contract (`fingerprints.json` + `oracle_labels.jsonl` read-only, no CE exposure) | Task 2 (build_pool reads both); inherent (never touches `to_stable_card_text`) |
| §2 Pool + anchor-level endorsement filter | Task 2 |
| §2 `oracle_queries` and `endorsed_fingerprint_queries` stay separate in model | Task 2 (Pool dataclass); test_build_pool_includes_fingerprints_only_when_anchor_is_oracle_endorsed locks the field separation |
| §2 Support-semantics value `"anchor_level_oracle_endorsement"` | Task 5 (constant SUPPORT_SEMANTICS); test_curate_happy_path asserts the value |
| §3 `QueryPatternExtractor` Protocol | Task 3 |
| §3 `HeuristicExtractor` (ngram MVP) | Task 3 |
| §3 `LLMExtractor` stub raising NotImplementedError | Task 4 |
| §3 Extractor output dedup against existing patterns; cap 20 | Task 5 (curator handles it) |
| §4 Candidate written at `anchor_cards_candidate.json` + sibling meta | Task 5 |
| §4 Meta has Branch 1 fields + new (oracle_labels_read, fingerprints_read, n_anchors_with_*, extractor, support_semantics) | Task 5 |
| §4 `source = "production_signal"` | Task 5 |
| §4 Empty-run guard | Task 5 (curator), Task 6 (CLI respects summary.wrote_candidate) |
| §4 Coexistence contract (one candidate at a time; overwrite semantics) | Inherent in Task 5 — `curate` writes only when guard passes, and when it writes it overwrites |
| §5 Promote output gains recalibration hint | Task 6 Step 7; tested in Task 6 Step 2 |
| §5 Recalibration is operator-invoked, not automated | Inherent — no code automation added |
| §6 `psa atlas curate` with `--extractor {heuristic,llm}` default heuristic | Task 6 Steps 4–5 |
| §6 Description: "build a candidate refinement from production signals" | Task 6 Step 4 (subparser help text) |
| §6 LLM in MVP exits non-zero with pointer | Task 6 Step 6 (NotImplementedError path) |
| Tests: empty-run, happy path, LLM stub, unknown extractor | Tasks 3–6 |

No gaps.

**Placeholder scan:** No TBDs, no "implement later" patterns. Every code step has a complete block.

**Type consistency check:**
- `Pool` signature in Task 2 matches its use in Task 5 (`pool.oracle_queries`, `pool.endorsed_fingerprint_queries`).
- `QueryPatternExtractor.extract(pool: list[str], n: int) -> list[str]` in Task 3 matches `HeuristicExtractor.extract` and `LLMExtractor.extract` and the curator's call site in Task 5.
- `curate(tenant_id, extractor_name)` summary dict shape is fixed in Task 5 and consumed in Task 6 (CLI).
- `SUPPORT_SEMANTICS = "anchor_level_oracle_endorsement"` in Task 5 matches the test expectation in Task 6 (`meta_obj["support_semantics"] == "anchor_level_oracle_endorsement"`).
- `_make_extractor` raises `ValueError` for unknown extractor names (Task 5); argparse's `choices=["heuristic", "llm"]` (Task 6) rejects bogus names at parse time, so the `ValueError` path is defensive for direct programmatic use — not exposed via the CLI.
