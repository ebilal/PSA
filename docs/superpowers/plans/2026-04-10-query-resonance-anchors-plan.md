# Query-Resonance Anchors + Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the retrieval-then-pack paradigm with query-resonance anchors (cards that encode what they can answer), session-backtracked oracle labels (real ground truth, not heuristic), and synthesis-based context assembly (one coherent narrative, not ranked bullets).

**Architecture:** Anchor cards grow two new fields — `generated_query_patterns` (seed questions from atlas build) and `query_fingerprint` (real queries accumulated at inference). The oracle labeler gets `backtrack_gold_anchors()` to derive real gold anchor IDs from LongMemEval session references. `AnchorSynthesizer` replaces the packer's weighted scoring formula with one LLM call over all selected anchors' memories.

**Tech Stack:** Python 3.13, SQLite, existing `call_llm()` infrastructure (cloud-first / Ollama fallback), sentence-transformers cross-encoder, BAAI/bge-base-en-v1.5.

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `psa/anchor.py` | Modify | Add `generated_query_patterns`, `query_fingerprint` fields; update `to_stable_card_text()`, `to_card_text()`, `from_dict()` |
| `psa/atlas.py` | Modify | `_generate_card_via_qwen()` returns `query_patterns`; `build_atlas()` inherits fingerprints |
| `psa/fingerprints.py` | Create | `FingerprintStore` class (load/save/append/inherit) |
| `psa/memory_object.py` | Modify | Add `get_by_source_session()` to `MemoryStore` |
| `psa/training/oracle_labeler.py` | Modify | Add `backtrack_gold_anchors()` function |
| `psa/synthesizer.py` | Create | `AnchorSynthesizer` class |
| `psa/packer.py` | Modify | Remove `selector_scores`, `packer_weights`, `include_assistant_turns` params and scoring formula |
| `psa/pipeline.py` | Modify | Remove `_packer_weights`; add `AnchorSynthesizer`; fingerprint accumulation; synthesis call with packer fallback; remove `_is_assistant_reference` |
| `psa/benchmarks/longmemeval.py` | Modify | `oracle_label()` calls `backtrack_gold_anchors()`; remove `packer_weights` param from `run()` |
| `tests/test_anchor.py` | Modify | Add tests for new fields, `to_stable_card_text`, `from_dict` compat |
| `tests/test_fingerprints.py` | Create | Tests for `FingerprintStore` |
| `tests/test_memory_object.py` | Modify | Add test for `get_by_source_session()` |
| `tests/test_oracle_labeler.py` | Modify | Add test for `backtrack_gold_anchors()` |
| `tests/test_synthesizer.py` | Create | Tests for `AnchorSynthesizer` |
| `tests/test_packer.py` | Modify | Remove `test_pack_memories_direct_accepts_selector_scores` and `test_pack_memories_direct_accepts_packer_weights` |
| `tests/test_pipeline.py` | Modify | Remove `_is_assistant_reference` import and its tests; add synthesis fallback test |
| `README.md` | Modify | Update query pipeline diagram and architecture sections |

---

### Task 1: AnchorCard — add query pattern fields and update card text methods

**Files:**
- Modify: `psa/anchor.py`
- Modify: `tests/test_anchor.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_anchor.py`, add after the existing tests:

```python
def test_anchor_card_has_generated_query_patterns_field():
    card = AnchorCard(
        anchor_id=1, name="test", meaning="test meaning",
        memory_types=["semantic"], include_terms=[], exclude_terms=[],
        prototype_examples=[], near_but_different=[], centroid=[0.1] * 768,
    )
    assert hasattr(card, "generated_query_patterns")
    assert card.generated_query_patterns == []


def test_anchor_card_has_query_fingerprint_field():
    card = AnchorCard(
        anchor_id=1, name="test", meaning="test meaning",
        memory_types=["semantic"], include_terms=[], exclude_terms=[],
        prototype_examples=[], near_but_different=[], centroid=[0.1] * 768,
    )
    assert hasattr(card, "query_fingerprint")
    assert card.query_fingerprint == []


def test_to_stable_card_text_includes_generated_query_patterns():
    card = AnchorCard(
        anchor_id=1, name="schema-decisions", meaning="Covers schema choices.",
        memory_types=["semantic"], include_terms=["migration"], exclude_terms=[],
        prototype_examples=[], near_but_different=[], centroid=[0.0] * 768,
        generated_query_patterns=["What did we decide about migrations?", "Why postgres?"],
    )
    text = card.to_stable_card_text()
    assert "What did we decide about migrations?" in text
    assert "Why postgres?" in text


def test_to_stable_card_text_excludes_query_fingerprint():
    card = AnchorCard(
        anchor_id=1, name="schema-decisions", meaning="Covers schema choices.",
        memory_types=["semantic"], include_terms=[], exclude_terms=[],
        prototype_examples=[], near_but_different=[], centroid=[0.0] * 768,
        generated_query_patterns=[],
        query_fingerprint=["a real user query"],
    )
    text = card.to_stable_card_text()
    assert "a real user query" not in text


def test_to_card_text_includes_query_fingerprint():
    card = AnchorCard(
        anchor_id=1, name="schema-decisions", meaning="Covers schema choices.",
        memory_types=["semantic"], include_terms=[], exclude_terms=[],
        prototype_examples=[], near_but_different=[], centroid=[0.0] * 768,
        query_fingerprint=["a real user query"],
    )
    text = card.to_card_text()
    assert "a real user query" in text


def test_from_dict_backward_compat_missing_query_fields():
    """Old atlas JSON without new fields should load without KeyError."""
    old_dict = {
        "anchor_id": 99, "name": "old-anchor", "meaning": "old meaning",
        "memory_types": ["semantic"], "include_terms": [], "exclude_terms": [],
        "prototype_examples": [], "near_but_different": [],
        "centroid": [0.0] * 768, "memory_count": 5,
        "is_novelty": False, "status": "active", "metadata": {},
    }
    card = AnchorCard.from_dict(old_dict)
    assert card.generated_query_patterns == []
    assert card.query_fingerprint == []
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_anchor.py::test_anchor_card_has_generated_query_patterns_field tests/test_anchor.py::test_from_dict_backward_compat_missing_query_fields -v
```

Expected: FAIL — `AnchorCard.__init__() got an unexpected keyword argument` or `AttributeError`.

- [ ] **Step 3: Update `psa/anchor.py`**

Add the two new fields to `AnchorCard` (after `metadata`):

```python
@dataclass
class AnchorCard:
    anchor_id: int
    name: str
    meaning: str
    memory_types: List[str]
    include_terms: List[str]
    exclude_terms: List[str]
    prototype_examples: List[str]
    near_but_different: List[str]
    centroid: List[float]
    memory_count: int = 0
    is_novelty: bool = False
    status: str = "active"
    metadata: dict = field(default_factory=dict)
    generated_query_patterns: List[str] = field(default_factory=list)
    query_fingerprint: List[str] = field(default_factory=list)
```

Update `to_stable_card_text()` to include `generated_query_patterns`:

```python
def to_stable_card_text(self) -> str:
    parts = [f"Anchor: {self.name}", f"Meaning: {self.meaning}"]
    if self.include_terms:
        parts.append(f"Includes: {', '.join(self.include_terms[:8])}")
    if self.exclude_terms:
        parts.append(f"Excludes: {', '.join(self.exclude_terms[:4])}")
    if self.generated_query_patterns:
        parts.append("Example questions this anchor answers:")
        for q in self.generated_query_patterns[:15]:
            parts.append(f"  - {q}")
    return "\n".join(parts)
```

Update `to_card_text()` to also include `query_fingerprint`:

```python
def to_card_text(self) -> str:
    text = self.to_stable_card_text()
    if self.prototype_examples:
        text += f"\nExamples: {'; '.join(self.prototype_examples[:3])}"
    if self.query_fingerprint:
        text += "\nRecent queries:\n" + "\n".join(
            f"  - {q}" for q in self.query_fingerprint[-20:]
        )
    return text
```

Update `from_dict()` for backward compatibility with old atlas JSON:

```python
@classmethod
def from_dict(cls, d: dict) -> "AnchorCard":
    d = dict(d)  # shallow copy — don't mutate caller's dict
    d.setdefault("generated_query_patterns", [])
    d.setdefault("query_fingerprint", [])
    return cls(**d)
```

- [ ] **Step 4: Run all new tests**

```
uv run pytest tests/test_anchor.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add psa/anchor.py tests/test_anchor.py
git commit -m "feat: add generated_query_patterns and query_fingerprint to AnchorCard"
```

---

### Task 2: Atlas — extend card generation to include query patterns

**Files:**
- Modify: `psa/atlas.py`
- Modify: `tests/test_atlas.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_atlas.py`:

```python
def test_generate_card_has_query_patterns(monkeypatch):
    """_generate_card_via_qwen should populate generated_query_patterns."""
    from psa.atlas import _generate_card_via_qwen
    from psa.memory_object import MemoryObject, MemoryType

    mock_response = json.dumps({
        "name": "auth-patterns",
        "meaning": "Authentication design choices.",
        "include_terms": ["auth", "token"],
        "exclude_terms": ["ui"],
        "query_patterns": [
            "What auth library did we use?",
            "How do tokens expire?",
        ],
    })
    monkeypatch.setattr("psa.atlas.call_llm", lambda **kw: mock_response)

    mo = MagicMock(spec=MemoryObject)
    mo.title = "JWT setup"
    mo.summary = "We use JWT."
    mo.memory_type = MemoryType.PROCEDURAL

    card = _generate_card_via_qwen(
        anchor_id=1,
        centroid=[0.0] * 768,
        sample_memories=[mo],
    )
    assert card.generated_query_patterns == [
        "What auth library did we use?",
        "How do tokens expire?",
    ]
```

Note: `test_atlas.py` already uses `json` and `MagicMock` — add this test alongside existing ones.

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_atlas.py::test_generate_card_has_query_patterns -v
```

Expected: FAIL — `generated_query_patterns` is `[]`.

- [ ] **Step 3: Update `_generate_card_via_qwen` in `psa/atlas.py`**

Change the prompt to request `query_patterns`, increase `max_tokens` to 512, and populate `generated_query_patterns` on the returned card:

```python
prompt = (
    f"{prefix}. Analyze them and produce a semantic description.\n\n"
    f"Sample memories:\n{samples_text}\n\n"
    "Return JSON with these fields:\n"
    '{\n'
    '  "name": "short-kebab-case-name (2-4 words, descriptive)",\n'
    '  "meaning": "1-2 sentences describing what this region of memory covers",\n'
    '  "include_terms": ["up to 8 keywords that signal membership"],\n'
    '  "exclude_terms": ["up to 4 keywords that signal non-membership"],\n'
    '  "query_patterns": ["10-15 specific questions a user might ask that this cluster can answer"]\n'
    '}'
)
```

Change `max_tokens=256` to `max_tokens=512`.

Add extraction of `query_patterns` after `result = _json.loads(content)`:

```python
query_patterns = result.get("query_patterns", [])[:15]
```

Change the stub fallback to include empty `query_patterns`:

```python
except Exception as e:
    logger.warning("Qwen card generation failed for anchor %d: %s (using stub)", anchor_id, e)
    name = f"novelty-{anchor_id}" if is_novelty else f"cluster-{anchor_id}"
    meaning = (
        "Low-density or outlier memories."
        if is_novelty
        else f"A cluster of {len(sample_memories)} memories. "
             f"Representative: {'; '.join(titles[:3])}."
    )
    include_terms = []
    exclude_terms = []
    query_patterns = []
```

Update the `return AnchorCard(...)` to pass `generated_query_patterns`:

```python
return AnchorCard(
    anchor_id=anchor_id,
    name=name,
    meaning=meaning,
    memory_types=memory_types,
    include_terms=include_terms,
    exclude_terms=exclude_terms,
    prototype_examples=prototypes,
    near_but_different=[],
    centroid=centroid,
    memory_count=len(sample_memories),
    is_novelty=is_novelty,
    generated_query_patterns=query_patterns,
)
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_atlas.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add psa/atlas.py tests/test_atlas.py
git commit -m "feat: extend atlas card generation to produce query_patterns"
```

---

### Task 3: FingerprintStore — persist and inherit accumulated query fingerprints

**Files:**
- Create: `psa/fingerprints.py`
- Create: `tests/test_fingerprints.py`
- Modify: `psa/anchor.py` (Atlas `save()`/`load()` integration is in atlas.py)
- Modify: `psa/atlas.py` (attach FingerprintStore to Atlas; inherit on rebuild)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fingerprints.py`:

```python
"""test_fingerprints.py — Tests for FingerprintStore."""

import os
import json
import pytest

from psa.fingerprints import FingerprintStore


def test_fingerprint_store_empty_on_new_dir(tmp_path):
    store = FingerprintStore(str(tmp_path))
    assert store.get(1) == []


def test_fingerprint_store_append_and_get(tmp_path):
    store = FingerprintStore(str(tmp_path))
    store.append(1, "query A")
    store.append(1, "query B")
    assert store.get(1) == ["query A", "query B"]


def test_fingerprint_store_fifo_eviction(tmp_path):
    store = FingerprintStore(str(tmp_path))
    for i in range(55):
        store.append(1, f"query {i}")
    result = store.get(1)
    assert len(result) == 50
    assert result[0] == "query 5"   # oldest 5 evicted
    assert result[-1] == "query 54"


def test_fingerprint_store_save_and_reload(tmp_path):
    store = FingerprintStore(str(tmp_path))
    store.append(1, "query A")
    store.append(2, "query B")
    store.save()

    store2 = FingerprintStore(str(tmp_path))
    assert store2.get(1) == ["query A"]
    assert store2.get(2) == ["query B"]


def test_fingerprint_store_inherit_from(tmp_path):
    store = FingerprintStore(str(tmp_path))
    store.append(10, "old query")
    store.inherit_from(old_anchor_id=10, new_anchor_id=20)
    assert store.get(20) == ["old query"]
    assert store.get(10) == ["old query"]  # original unchanged


def test_fingerprint_store_inherit_from_missing_anchor(tmp_path):
    """inherit_from a non-existent old anchor is a no-op."""
    store = FingerprintStore(str(tmp_path))
    store.inherit_from(old_anchor_id=999, new_anchor_id=1)
    assert store.get(1) == []
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_fingerprints.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'psa.fingerprints'`.

- [ ] **Step 3: Create `psa/fingerprints.py`**

```python
"""
fingerprints.py — FingerprintStore: persists accumulated query fingerprints per anchor.

Fingerprints are stored separately from the atlas JSON so they survive atlas rebuilds.
File location: <atlas_dir>/fingerprints.json

Structure on disk:
    {"<anchor_id>": ["query 1", "query 2", ...], ...}
"""

import json
import logging
import os
from typing import Dict, List

logger = logging.getLogger("psa.fingerprints")

MAX_FINGERPRINT_SIZE = 50


class FingerprintStore:
    """
    Persists accumulated query fingerprints per anchor.

    Each anchor accumulates queries from above-threshold activations (capped at
    MAX_FINGERPRINT_SIZE entries, FIFO eviction). Fingerprints are stored in
    <atlas_dir>/fingerprints.json — separate from anchor_cards.json so they
    survive atlas rebuilds.
    """

    def __init__(self, atlas_dir: str):
        self._path = os.path.join(atlas_dir, "fingerprints.json")
        self._data: Dict[int, List[str]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    raw = json.load(f)
                self._data = {int(k): v for k, v in raw.items()}
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load fingerprints from %s: %s", self._path, e)

    def save(self) -> None:
        """Persist fingerprints to disk. Best-effort — failures are logged, not raised."""
        try:
            with open(self._path, "w") as f:
                json.dump({str(k): v for k, v in self._data.items()}, f)
        except OSError as e:
            logger.warning("Failed to save fingerprints to %s: %s", self._path, e)

    def append(self, anchor_id: int, query: str) -> None:
        """Append a query to an anchor's fingerprint. Evicts oldest entry when full."""
        queries = self._data.setdefault(anchor_id, [])
        queries.append(query)
        if len(queries) > MAX_FINGERPRINT_SIZE:
            del queries[0]

    def get(self, anchor_id: int) -> List[str]:
        """Return a copy of the fingerprint for an anchor."""
        return list(self._data.get(anchor_id, []))

    def inherit_from(self, old_anchor_id: int, new_anchor_id: int) -> None:
        """
        Copy fingerprint from old_anchor_id to new_anchor_id.

        Called during atlas rebuild: matched anchors inherit their accumulated
        query signal into the new anchor's fingerprint.
        """
        queries = self._data.get(old_anchor_id, [])
        if queries:
            self._data[new_anchor_id] = list(queries)
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_fingerprints.py -v
```

Expected: all PASS.

- [ ] **Step 5: Integrate FingerprintStore with Atlas in `psa/atlas.py`**

At the top of `atlas.py`, add the import:

```python
from .fingerprints import FingerprintStore
```

Add `fingerprint_store` field to the `Atlas` dataclass (with `field(default=None)`):

```python
@dataclass
class Atlas:
    version: int
    tenant_id: str
    anchor_index: AnchorIndex
    stats: AtlasStats
    anchor_dir: str
    cards: List[AnchorCard] = field(default_factory=list)
    fingerprint_store: Optional[FingerprintStore] = field(default=None)
```

In `AtlasManager.get_atlas()`, after loading the atlas, attach a FingerprintStore:

Find the place where the atlas is returned (inside `get_atlas()`) and add:

```python
atlas.fingerprint_store = FingerprintStore(atlas.anchor_dir)
```

In `build_atlas()`, after `matched = _match_anchors(old_active_cards, centroids)`, load the old fingerprint store (if any) and inherit fingerprints after each matched card is assigned:

```python
# Load fingerprints from previous atlas for inheritance
old_fingerprints = None
if previous_atlas is not None:
    old_fingerprints = FingerprintStore(previous_atlas.anchor_dir)
```

Then, in the loop where matched cards are re-used, after the block that does `card = matched[i]` (i.e., when the card is not None), add:

```python
if old_fingerprints is not None and card is not None:
    new_fingerprints.inherit_from(
        old_anchor_id=card.anchor_id,
        new_anchor_id=card.anchor_id,  # anchor_id is preserved on match
    )
```

Create a new `FingerprintStore` for the new atlas dir before saving:

```python
new_fingerprints = FingerprintStore(anchor_dir)
```

Attach it to the new atlas and save:

```python
atlas.fingerprint_store = new_fingerprints
new_fingerprints.save()
```

The exact insertion points in `build_atlas()` are inside the `AtlasBuilder.build_atlas()` method. Look for the section that loops `for i, centroid in enumerate(centroids):` and handles matched vs fresh cards.

- [ ] **Step 6: Run full atlas tests**

```
uv run pytest tests/test_atlas.py tests/test_fingerprints.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add psa/fingerprints.py psa/atlas.py tests/test_fingerprints.py
git commit -m "feat: add FingerprintStore with FIFO eviction and rebuild inheritance"
```

---

### Task 4: MemoryStore — add get_by_source_session()

**Files:**
- Modify: `psa/memory_object.py`
- Modify: `tests/test_memory_object.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_memory_object.py`:

```python
def test_get_by_source_session_returns_memories(tmp_path):
    """get_by_source_session finds memories whose source_path contains the session_id."""
    from psa.memory_object import MemoryStore, MemoryObject, MemoryType, RawSource

    store = MemoryStore(str(tmp_path / "mem.sqlite3"))
    tenant_id = "test_tenant"

    # Add a raw source whose path contains the session_id
    session_id = "session_abc_123"
    source = RawSource.create(
        tenant_id=tenant_id,
        source_type="conversation",
        full_text="some content",
        title="test session",
        source_path=f"/tmp/convos/{session_id}.jsonl",
    )
    store.add_raw_source(source)

    # Add a memory object linked to this source
    mo = MemoryObject.create(
        tenant_id=tenant_id,
        memory_type=MemoryType.EPISODIC,
        title="A memory",
        body="body text",
        summary="summary",
        source_ids=[source.source_id],
        classification_reason="test",
    )
    mo.embedding = [0.0] * 768
    store.add(mo)

    # Add a memory from a different source
    other_source = RawSource.create(
        tenant_id=tenant_id,
        source_type="conversation",
        full_text="other content",
        title="other",
        source_path="/tmp/convos/other_session.jsonl",
    )
    store.add_raw_source(other_source)
    other_mo = MemoryObject.create(
        tenant_id=tenant_id,
        memory_type=MemoryType.SEMANTIC,
        title="Other memory",
        body="other body",
        summary="other summary",
        source_ids=[other_source.source_id],
        classification_reason="test",
    )
    other_mo.embedding = [0.0] * 768
    store.add(other_mo)

    results = store.get_by_source_session(session_id, tenant_id=tenant_id)
    assert len(results) == 1
    assert results[0].memory_object_id == mo.memory_object_id


def test_get_by_source_session_returns_empty_for_missing(tmp_path):
    from psa.memory_object import MemoryStore

    store = MemoryStore(str(tmp_path / "mem.sqlite3"))
    results = store.get_by_source_session("nonexistent_session", tenant_id="test")
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_memory_object.py::test_get_by_source_session_returns_memories tests/test_memory_object.py::test_get_by_source_session_returns_empty_for_missing -v
```

Expected: FAIL — `AttributeError: 'MemoryStore' object has no attribute 'get_by_source_session'`.

- [ ] **Step 3: Add `get_by_source_session()` to `MemoryStore` in `psa/memory_object.py`**

Add this method after `get_by_source_id()`:

```python
def get_by_source_session(
    self, session_id: str, tenant_id: Optional[str] = None
) -> List[MemoryObject]:
    """
    Return all MemoryObjects whose source records have a source_path containing session_id.

    Used by backtrack_gold_anchors() to find which anchors contain memories
    from a known ground-truth session (e.g. LongMemEval answer_session_ids).
    """
    with self._connect() as conn:
        # Find source_ids for raw_sources whose path contains the session_id
        if tenant_id:
            source_rows = conn.execute(
                "SELECT source_id FROM raw_sources WHERE source_path LIKE ? AND tenant_id = ?",
                (f"%{session_id}%", tenant_id),
            ).fetchall()
        else:
            source_rows = conn.execute(
                "SELECT source_id FROM raw_sources WHERE source_path LIKE ?",
                (f"%{session_id}%",),
            ).fetchall()

        if not source_rows:
            return []

        source_ids = [r["source_id"] for r in source_rows]

        # Find memory_objects linked to any of those source_ids
        placeholders = ",".join("?" * len(source_ids))
        if tenant_id:
            rows = conn.execute(
                f"""
                SELECT DISTINCT mo.* FROM memory_objects mo, json_each(mo.source_ids_json) je
                WHERE je.value IN ({placeholders}) AND mo.tenant_id = ?
                """,
                (*source_ids, tenant_id),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT DISTINCT mo.* FROM memory_objects mo, json_each(mo.source_ids_json) je
                WHERE je.value IN ({placeholders})
                """,
                source_ids,
            ).fetchall()

    return [self._row_to_memory_object(r) for r in rows]
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_memory_object.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add psa/memory_object.py tests/test_memory_object.py
git commit -m "feat: add MemoryStore.get_by_source_session() for oracle backtracking"
```

---

### Task 5: OracleLabeler — add backtrack_gold_anchors()

**Files:**
- Modify: `psa/training/oracle_labeler.py`
- Modify: `tests/test_oracle_labeler.py` (or create if it doesn't exist)

- [ ] **Step 1: Write the failing test**

Check if `tests/test_oracle_labeler.py` exists; if not, create it. Add:

```python
"""test_oracle_labeler.py — Tests for oracle labeling utilities."""

from unittest.mock import MagicMock

from psa.memory_object import MemoryObject, MemoryType
from psa.training.oracle_labeler import backtrack_gold_anchors


def _make_memory(anchor_id: int) -> MemoryObject:
    mo = MagicMock(spec=MemoryObject)
    mo.anchor_id = anchor_id
    mo.primary_anchor_id = anchor_id
    return mo


def test_backtrack_gold_anchors_returns_anchor_ids():
    """backtrack_gold_anchors maps session_ids → memory objects → anchor IDs."""
    store = MagicMock()
    store.get_by_source_session.return_value = [
        _make_memory(anchor_id=5),
        _make_memory(anchor_id=7),
    ]

    atlas = MagicMock()

    result = backtrack_gold_anchors(
        answer_session_ids=["session_abc"],
        store=store,
        atlas=atlas,
        tenant_id="test",
    )
    assert set(result) == {5, 7}
    store.get_by_source_session.assert_called_once_with("session_abc", tenant_id="test")


def test_backtrack_gold_anchors_deduplicates():
    """Two sessions pointing to memories in the same anchor yield one anchor ID."""
    store = MagicMock()
    store.get_by_source_session.side_effect = [
        [_make_memory(anchor_id=3)],
        [_make_memory(anchor_id=3)],
    ]
    result = backtrack_gold_anchors(
        answer_session_ids=["session_a", "session_b"],
        store=MagicMock(**{"get_by_source_session.side_effect": [
            [_make_memory(anchor_id=3)],
            [_make_memory(anchor_id=3)],
        ]}),
        atlas=MagicMock(),
        tenant_id="test",
    )
    assert result.count(3) == 1


def test_backtrack_gold_anchors_skips_unassigned_memories():
    """Memories with anchor_id None or -1 are ignored."""
    mo_no_anchor = MagicMock(spec=MemoryObject)
    mo_no_anchor.primary_anchor_id = None

    mo_neg = MagicMock(spec=MemoryObject)
    mo_neg.primary_anchor_id = -1

    store = MagicMock()
    store.get_by_source_session.return_value = [mo_no_anchor, mo_neg]

    result = backtrack_gold_anchors(
        answer_session_ids=["session_x"],
        store=store,
        atlas=MagicMock(),
        tenant_id="test",
    )
    assert result == []


def test_backtrack_gold_anchors_empty_sessions():
    result = backtrack_gold_anchors(
        answer_session_ids=[],
        store=MagicMock(),
        atlas=MagicMock(),
        tenant_id="test",
    )
    assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_oracle_labeler.py -v
```

Expected: FAIL — `ImportError: cannot import name 'backtrack_gold_anchors'`.

- [ ] **Step 3: Add `backtrack_gold_anchors()` to `psa/training/oracle_labeler.py`**

Add this function before the `OracleLabeler` class definition:

```python
def backtrack_gold_anchors(
    answer_session_ids: List[str],
    store: Any,
    atlas: Any,
    tenant_id: str = "default",
) -> List[int]:
    """
    Derive gold anchor IDs from ground-truth session references.

    For each answer session, find memory objects whose source records have
    a source_path containing the session_id, then collect the anchor_ids
    those memory objects are assigned to.

    Deterministic — no LLM calls. Works for any dataset that provides
    ground-truth source references (e.g., LongMemEval answer_session_ids).

    Parameters
    ----------
    answer_session_ids:
        Session IDs known to contain the answer (e.g., from LongMemEval).
    store:
        MemoryStore instance to query.
    atlas:
        Atlas instance (reserved for future use — not currently needed).
    tenant_id:
        Tenant whose memories to search.

    Returns
    -------
    Deduplicated list of anchor IDs that contain memories from the answer sessions.
    """
    gold_anchor_ids: set = set()
    for session_id in answer_session_ids:
        memories = store.get_by_source_session(session_id, tenant_id=tenant_id)
        for m in memories:
            aid = getattr(m, "primary_anchor_id", None)
            if aid is not None and aid >= 0:
                gold_anchor_ids.add(aid)
    return list(gold_anchor_ids)
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_oracle_labeler.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add psa/training/oracle_labeler.py tests/test_oracle_labeler.py
git commit -m "feat: add backtrack_gold_anchors() for ground-truth oracle labeling"
```

---

### Task 6: LongMemEval — oracle_label() uses session backtracking

**Files:**
- Modify: `psa/benchmarks/longmemeval.py`

(No new test file needed — existing integration is validated by running the function.)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_benchmarks.py` (create if needed):

```python
"""test_benchmarks.py — Unit tests for benchmark harness utilities."""

from unittest.mock import MagicMock, patch
import json
import os
import pytest


def _make_result_record(question_id="q1", answer_session_ids=None):
    return {
        "question_id": question_id,
        "question": "What did we decide about auth?",
        "context_text": "some context",
        "answer_generated": "JWT",
        "answer_gold": "JWT",
        "answer_session_ids": answer_session_ids or ["session_abc"],
        "tokens_used": 100,
        "token_budget": 6000,
        "selected_anchor_ids": [1, 2],
        "timing_ms": {"embed": 10, "retrieve": 20, "select": 5, "fetch": 3, "pack": 2, "total": 40},
    }


def test_oracle_label_calls_backtrack(tmp_path):
    """oracle_label() should call backtrack_gold_anchors for each record."""
    from psa.benchmarks.longmemeval import oracle_label

    results_path = str(tmp_path / "results.jsonl")
    with open(results_path, "w") as f:
        f.write(json.dumps(_make_result_record()) + "\n")

    mock_pipeline = MagicMock()
    mock_pipeline.atlas.version = 1
    mock_pipeline.store = MagicMock()
    mock_pipeline.atlas = MagicMock()

    mock_labeler = MagicMock()

    with patch("psa.benchmarks.longmemeval.PSAPipeline.from_tenant", return_value=mock_pipeline), \
         patch("psa.benchmarks.longmemeval.OracleLabeler", return_value=mock_labeler), \
         patch("psa.benchmarks.longmemeval.backtrack_gold_anchors", return_value=[5]) as mock_bt:
        oracle_label(results_path, tenant_id="test_tenant")

    mock_bt.assert_called_once_with(
        answer_session_ids=["session_abc"],
        store=mock_pipeline.store,
        atlas=mock_pipeline.atlas,
        tenant_id="test_tenant",
    )
    mock_labeler.label.assert_called_once()
    call_kwargs = mock_labeler.label.call_args[1]
    assert call_kwargs["gold_anchor_ids"] == [5]
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_benchmarks.py::test_oracle_label_calls_backtrack -v
```

Expected: FAIL — `backtrack_gold_anchors` is not called; `gold_anchor_ids` not passed.

- [ ] **Step 3: Update `oracle_label()` in `psa/benchmarks/longmemeval.py`**

Add the import at the top of the imports inside the function:

Change the `oracle_label()` function. Replace the current loop body:

```python
# Current (wrong — doesn't pass gold_anchor_ids):
labeler.label(query_id=query_id, query=query)
```

With:

```python
from ..training.oracle_labeler import backtrack_gold_anchors

# ...existing setup code (records, pipeline, labeler init)...

written = 0
for i, record in enumerate(records):
    query_id = record.get("question_id", f"lme_q_{i:04d}")
    query = record["question"]
    answer_session_ids = record.get("answer_session_ids", [])

    try:
        gold_anchor_ids = backtrack_gold_anchors(
            answer_session_ids=answer_session_ids,
            store=pipeline.store,
            atlas=pipeline.atlas,
            tenant_id=tenant_id,
        )
        labeler.label(
            query_id=query_id,
            query=query,
            gold_anchor_ids=gold_anchor_ids if gold_anchor_ids else None,
        )
        written += 1
    except Exception as e:
        logger.warning("Oracle labeling failed for q=%s: %s", record.get("question_id"), e)

    if (i + 1) % 50 == 0:
        logger.info("  %d / %d questions labeled", i + 1, len(records))
```

Move the `from ..training.oracle_labeler import backtrack_gold_anchors` import to the top of the function alongside the existing `from ..training.oracle_labeler import OracleLabeler` import.

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_benchmarks.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add psa/benchmarks/longmemeval.py tests/test_benchmarks.py
git commit -m "feat: oracle_label() uses backtrack_gold_anchors for real ground-truth labels"
```

---

### Task 7: AnchorSynthesizer — single LLM call over all selected anchor memories

**Files:**
- Create: `psa/synthesizer.py`
- Create: `tests/test_synthesizer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_synthesizer.py`:

```python
"""test_synthesizer.py — Tests for AnchorSynthesizer."""

from unittest.mock import MagicMock, patch

import pytest

from psa.memory_object import MemoryObject, MemoryType
from psa.synthesizer import AnchorSynthesizer


def _make_memory(title: str, body: str, quality: float = 0.8) -> MemoryObject:
    mo = MagicMock(spec=MemoryObject)
    mo.title = title
    mo.body = body
    mo.summary = f"Summary of {title}."
    mo.memory_type = MemoryType.SEMANTIC
    mo.quality_score = quality
    mo.embedding = [0.1] * 768
    mo.primary_anchor_id = 1
    return mo


def test_synthesizer_returns_string():
    with patch("psa.synthesizer.call_llm", return_value="We decided to use JWT for auth."):
        s = AnchorSynthesizer()
        result = s.synthesize("What auth approach did we use?", [_make_memory("JWT decision", "We use JWT.")])
    assert isinstance(result, str)
    assert len(result) > 0


def test_synthesizer_returns_llm_output():
    expected = "The team decided on JWT tokens with 24-hour expiry."
    with patch("psa.synthesizer.call_llm", return_value=expected):
        s = AnchorSynthesizer()
        result = s.synthesize("auth decision", [_make_memory("JWT", "JWT tokens.")])
    assert result == expected


def test_synthesizer_empty_memories_returns_fallback():
    s = AnchorSynthesizer()
    result = s.synthesize("some query", [])
    assert isinstance(result, str)
    assert len(result) > 0


def test_synthesizer_trims_by_cosine_to_query():
    """When many memories are passed, only the most relevant (by cosine) are sent to LLM."""
    import numpy as np

    query_vec = [1.0] + [0.0] * 767
    high_sim = MagicMock(spec=MemoryObject)
    high_sim.embedding = [1.0] + [0.0] * 767
    high_sim.title = "High relevance"
    high_sim.body = "Very relevant content."
    high_sim.summary = "High sim."
    high_sim.memory_type = MemoryType.SEMANTIC
    high_sim.quality_score = 0.5

    low_sim = MagicMock(spec=MemoryObject)
    low_sim.embedding = [0.0] * 767 + [1.0]
    low_sim.title = "Low relevance"
    low_sim.body = "Unrelated content."
    low_sim.summary = "Low sim."
    low_sim.memory_type = MemoryType.SEMANTIC
    low_sim.quality_score = 0.9

    captured_prompt = []
    def capture_llm(messages, **kwargs):
        captured_prompt.append(messages[0]["content"])
        return "synthesis result"

    with patch("psa.synthesizer.call_llm", side_effect=capture_llm):
        s = AnchorSynthesizer()
        s.synthesize("query", [high_sim, low_sim], query_vec=query_vec, max_memories=1)

    assert "High relevance" in captured_prompt[0]
    assert "Low relevance" not in captured_prompt[0]


def test_synthesizer_raises_on_llm_error_so_pipeline_can_catch():
    """Synthesizer propagates LLM errors — caller is responsible for fallback."""
    with patch("psa.synthesizer.call_llm", side_effect=RuntimeError("LLM timeout")):
        s = AnchorSynthesizer()
        with pytest.raises(RuntimeError):
            s.synthesize("query", [_make_memory("m", "b")])
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_synthesizer.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'psa.synthesizer'`.

- [ ] **Step 3: Create `psa/synthesizer.py`**

```python
"""
synthesizer.py — AnchorSynthesizer: single LLM call over selected anchor memories.

Replaces the packer's weighted scoring formula. Instead of ranking and
dumping memory bullets, produces one coherent narrative paragraph conditioned
on the query.

Usage:
    synthesizer = AnchorSynthesizer()  # instantiate once in PSAPipeline.__init__()
    text = synthesizer.synthesize(query, memories, query_vec=query_vec)
"""

import logging
from typing import List, Optional

import numpy as np

from .memory_object import MemoryObject

logger = logging.getLogger("psa.synthesizer")

MAX_MEMORIES_DEFAULT = 30
TOKEN_BUDGET_DEFAULT = 700
CHARS_PER_TOKEN = 4


def _cosine_sim(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class AnchorSynthesizer:
    """
    Synthesizes a query-conditioned narrative from selected anchor memories.

    Instantiated once in PSAPipeline.__init__(). Uses call_llm() which handles
    cloud-first / Ollama fallback — connection pooling is managed by call_llm().
    """

    def synthesize(
        self,
        query: str,
        memories: List[MemoryObject],
        query_vec: Optional[List[float]] = None,
        token_budget: int = TOKEN_BUDGET_DEFAULT,
        max_memories: int = MAX_MEMORIES_DEFAULT,
    ) -> str:
        """
        Synthesize a single query-conditioned narrative from anchor memories.

        Parameters
        ----------
        query:
            The user's query — synthesis is conditioned on this.
        memories:
            All MemoryObjects from selected anchors (deduplicated by pipeline).
        query_vec:
            L2-normalized query embedding for cosine-based trimming.
            When provided, lowest-cosine memories are dropped first.
            When absent, memories are used in the order provided.
        token_budget:
            Target output length in tokens (~700 = 5-8 sentence paragraph).
        max_memories:
            Hard cap on memories passed to the LLM to fit context limits.

        Returns
        -------
        Coherent prose paragraph. Raises on LLM failure — caller must catch
        and fall back to packer if needed.
        """
        if not memories:
            return "(no relevant memories found for this query)"

        # Trim by cosine similarity to query (lowest relevance dropped first)
        if query_vec is not None:
            ranked = sorted(
                memories,
                key=lambda m: _cosine_sim(m.embedding, query_vec) if m.embedding else 0.0,
                reverse=True,
            )
        else:
            ranked = list(memories)

        trimmed = ranked[:max_memories]

        memory_lines = []
        for mo in trimmed:
            line = f"[{mo.memory_type.value.upper()}] {mo.title}"
            if mo.summary:
                line += f": {mo.summary[:200]}"
            memory_lines.append(line)

        memory_text = "\n".join(memory_lines)
        max_input_chars = max_memories * 250
        if len(memory_text) > max_input_chars:
            memory_text = memory_text[:max_input_chars]

        prompt = (
            f"You are synthesizing memory context for an AI assistant.\n\n"
            f"Query: {query}\n\n"
            f"Relevant memories from personal history:\n{memory_text}\n\n"
            f"Write a focused, coherent paragraph (5-8 sentences) presenting what's most "
            f"relevant to help answer the query. Weave related facts into a narrative where "
            f"possible. Be specific and factual. Do not add information not present in the memories."
        )

        from .llm import call_llm

        return call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=token_budget,
            json_mode=False,
        )
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_synthesizer.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add psa/synthesizer.py tests/test_synthesizer.py
git commit -m "feat: add AnchorSynthesizer — single LLM call over selected anchor memories"
```

---

### Task 8: Packer — remove scoring formula and obsolete parameters

**Files:**
- Modify: `psa/packer.py`
- Modify: `tests/test_packer.py`

- [ ] **Step 1: Remove the obsolete tests first**

In `tests/test_packer.py`, remove these two test functions entirely:

```python
def test_pack_memories_direct_accepts_selector_scores():
    ...

def test_pack_memories_direct_accepts_packer_weights():
    ...
```

- [ ] **Step 2: Run tests to verify remaining packer tests still pass**

```
uv run pytest tests/test_packer.py -v
```

Expected: all PASS (the removed tests are gone, others unaffected).

- [ ] **Step 3: Simplify `pack_memories_direct()` in `psa/packer.py`**

Remove the three parameters `selector_scores`, `packer_weights`, and `include_assistant_turns` from the signature. Remove the scoring formula block. The method signature becomes:

```python
def pack_memories_direct(
    self,
    query: str,
    memories: List[MemoryObject],
    token_budget: int = 6000,
    query_vec: Optional[List[float]] = None,
    store: Optional[MemoryStore] = None,
) -> PackedContext:
```

Remove the body block that conditionally applies the scoring formula (the `if selector_scores and packer_weights:` block and its `else` branch). Replace with the simple cosine + quality ranking alone:

```python
# Compute per-memory relevance to the query
relevances = _compute_relevance(memories, query_vec)

# Rank by cosine relevance (70%) + quality (30%)
scored = sorted(
    zip(memories, relevances),
    key=lambda pair: pair[1] * 0.7 + pair[0].quality_score * 0.3,
    reverse=True,
)
```

Also update `pipeline.py`'s call to `packed_context_for_anchors()` which calls `pack_memories_direct` without `selector_scores`/`packer_weights` — verify it already matches the new signature (it does, those params were optional).

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_packer.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add psa/packer.py tests/test_packer.py
git commit -m "refactor: remove packer scoring formula (selector_scores, packer_weights, include_assistant_turns)"
```

---

### Task 9: Pipeline — synthesizer, fingerprint accumulation, synthesis with packer fallback

**Files:**
- Modify: `psa/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Update the failing import in test_pipeline.py and add new tests**

In `tests/test_pipeline.py`, remove `_is_assistant_reference` from the import line and delete the two tests that use it (`test_detects_assistant_reference` and `test_non_assistant_reference`).

The import line changes from:
```python
from psa.pipeline import PSAPipeline, PSAResult, QueryTiming, _is_assistant_reference
```
to:
```python
from psa.pipeline import PSAPipeline, PSAResult, QueryTiming
```

Add two new tests:

```python
def test_pipeline_query_uses_synthesizer_when_available(pipeline, monkeypatch):
    """When synthesizer succeeds, result text comes from synthesis."""
    from unittest.mock import patch

    with patch("psa.pipeline.AnchorSynthesizer") as MockSynth:
        MockSynth.return_value.synthesize.return_value = "Synthesized narrative text."
        # Re-create pipeline to pick up patched AnchorSynthesizer
        from psa.pipeline import PSAPipeline
        p = PSAPipeline(
            store=pipeline.store,
            atlas=pipeline.atlas,
            embedding_model=pipeline.embedding_model,
            selector=pipeline.selector,
            token_budget=pipeline.token_budget,
            tenant_id=pipeline.tenant_id,
        )
        result = p.query("test query")
    # synthesis output is embedded in the packed context
    assert isinstance(result.text, str)


def test_pipeline_query_falls_back_to_packer_on_synthesis_failure(pipeline, monkeypatch):
    """When synthesizer raises, pipeline falls back to packer without crashing."""
    from unittest.mock import patch

    with patch("psa.pipeline.AnchorSynthesizer") as MockSynth:
        MockSynth.return_value.synthesize.side_effect = RuntimeError("LLM timeout")
        from psa.pipeline import PSAPipeline
        p = PSAPipeline(
            store=pipeline.store,
            atlas=pipeline.atlas,
            embedding_model=pipeline.embedding_model,
            selector=pipeline.selector,
            token_budget=pipeline.token_budget,
            tenant_id=pipeline.tenant_id,
        )
        result = p.query("test query")
    # fallback still returns a valid result
    assert isinstance(result, PSAResult)
    assert isinstance(result.text, str)
```

- [ ] **Step 2: Run tests to see current failures**

```
uv run pytest tests/test_pipeline.py -v
```

Expected: Some failures (import of `_is_assistant_reference` that no longer exists after we edit pipeline.py, and the new tests failing).

- [ ] **Step 3: Update `psa/pipeline.py`**

**3a.** Remove `_ASSISTANT_TRIGGERS` constant and `_is_assistant_reference()` function entirely.

**3b.** Add import at the top:

```python
from .synthesizer import AnchorSynthesizer
```

**3c.** In `PSAPipeline.__init__()`:
- Remove: `self._packer_weights = (0.4, 0.3, 0.3)`
- Add: `self._synthesizer = AnchorSynthesizer()`

**3d.** In `PSAPipeline.query()`, replace the Step 5 (Pack context) block:

Replace this entire block:

```python
# Step 5: Pack context
t0 = time.perf_counter()

# Build anchor_id → selector_score map for packer weighting
selector_scores = {sa.anchor_id: sa.selector_score for sa in selected}
packer_weights = None
if self.selector.mode == "trained":
    packer_weights = self._packer_weights

packed = self._packer.pack_memories_direct(
    query=query,
    memories=memories,
    token_budget=self.token_budget,
    query_vec=query_vec,
    store=self.store,
    selector_scores=selector_scores,
    packer_weights=packer_weights,
    include_assistant_turns=_is_assistant_reference(query),
)
timing.pack_ms = (time.perf_counter() - t0) * 1000
```

With:

```python
# Step 5: Synthesize context
t0 = time.perf_counter()
try:
    synthesis_text = self._synthesizer.synthesize(
        query=query,
        memories=memories,
        query_vec=query_vec,
        token_budget=700,
    )
    packed = PackedContext(
        query=query,
        text=synthesis_text,
        token_count=len(synthesis_text) // 4,
        memory_ids=[m.memory_object_id for m in memories],
        sections=[],
        untyped_count=0,
    )
except Exception:
    logger.debug("Synthesis failed, falling back to packer", exc_info=True)
    packed = self._packer.pack_memories_direct(
        query=query,
        memories=memories,
        token_budget=self.token_budget,
        query_vec=query_vec,
        store=self.store,
    )
timing.pack_ms = (time.perf_counter() - t0) * 1000
```

**3e.** Add fingerprint accumulation after Step 3 (Select anchors), before Step 4 (Fetch memories):

```python
# Accumulate query fingerprints for above-threshold selected anchors
if self.atlas.fingerprint_store is not None:
    for sa in selected:
        self.atlas.fingerprint_store.append(sa.anchor_id, query)
    try:
        self.atlas.fingerprint_store.save()
    except Exception:
        logger.debug("Failed to save fingerprints", exc_info=True)
```

**3f.** Remove the `_ASSISTANT_TRIGGERS` list and `_is_assistant_reference()` function that are now dead code.

- [ ] **Step 4: Run all pipeline tests**

```
uv run pytest tests/test_pipeline.py -v
```

Expected: all PASS.

- [ ] **Step 5: Run the full test suite**

```
uv run pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add psa/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline uses AnchorSynthesizer with packer fallback; accumulate query fingerprints"
```

---

### Task 10: LongMemEval run() and README update

**Files:**
- Modify: `psa/benchmarks/longmemeval.py`
- Modify: `README.md`

- [ ] **Step 1: Clean up `run()` in longmemeval.py**

Remove the `packer_weights` parameter and related dead code from the `run()` function:

Remove from function signature:
```python
packer_weights: Optional[tuple] = None,
```

Remove the two lines that set/use packer_weights:
```python
if packer_weights:
    pipeline._packer_weights = packer_weights
```

Remove the `weights_tag` computation and remove it from the output filename:
```python
# Remove these lines:
weights_tag = ""
if packer_weights:
    weights_tag = "_w" + "-".join(f"{w:.2f}" for w in packer_weights)
out_path = os.path.join(results_dir, f"results_{split}_{selector_mode}{weights_tag}_{ts}.jsonl")
```

Replace with:
```python
out_path = os.path.join(results_dir, f"results_{split}_{selector_mode}_{ts}.jsonl")
```

Also remove `packer_weights` from the timing_ms dict in the per-record output — it's already absent there, so no change needed.

- [ ] **Step 2: Run lint check**

```
uv run ruff check psa/benchmarks/longmemeval.py
uv run ruff format psa/benchmarks/longmemeval.py
```

Expected: no errors.

- [ ] **Step 3: Update README.md**

Locate the Query Pipeline section and update the diagram and description.

Replace the current query pipeline diagram:

```
psa search "query"
    │
    ├── embed query (bge-base-en-v1.5)
    ├── AnchorRetriever — BM25 + dense, fused via RRF → top-24 anchor candidates
    ├── AnchorSelector — cosine baseline or trained cross-encoder → 1–4 anchors
    ├── rank memories by query relevance (cosine similarity)
    └── EvidencePacker — role-organized sections, 6000-token budget
```

With:

```
psa search "query"
    │
    ├── embed query (bge-base-en-v1.5)
    ├── AnchorRetriever — BM25 + dense, fused via RRF → top-32 anchor candidates
    │     BM25 index includes anchor names, meanings, include_terms, generated
    │     query patterns, and accumulated real-query fingerprints
    ├── AnchorSelector — cosine baseline or trained cross-encoder → 1–4 anchors
    │     Cross-encoder input: (query, anchor_card_with_query_patterns)
    ├── fetch MemoryObjects for selected anchors
    └── AnchorSynthesizer — one LLM call → coherent narrative paragraph (~700 tokens)
          Fallback: EvidencePacker role-organized sections if LLM unavailable
```

Also update the architecture description paragraph (if it exists) to mention:
- Anchor cards include `generated_query_patterns` (10–15 example questions) seeded at atlas build time
- Real-query `query_fingerprint` is accumulated above-threshold at inference, capped at 50, FIFO
- Synthesis produces one coherent narrative instead of ranked memory bullets

- [ ] **Step 4: Run full test suite**

```
uv run pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 5: Run lint**

```
uv run ruff check .
uv run ruff format --check .
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add psa/benchmarks/longmemeval.py README.md
git commit -m "feat: remove packer_weights from benchmark run(); update README architecture docs"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| `generated_query_patterns` + `query_fingerprint` on AnchorCard | Task 1 |
| `to_stable_card_text()` includes patterns, not fingerprint | Task 1 |
| `to_card_text()` includes fingerprint (for BM25) | Task 1 |
| `from_dict()` backward compat | Task 1 |
| Atlas build generates query_patterns via LLM | Task 2 |
| `fingerprints.jsonl` persistence, FIFO at 50 | Task 3 |
| Fingerprint inheritance on rebuild via `_match_anchors` | Task 3 |
| Fingerprint accumulated above-threshold only | Task 9 |
| `MemoryStore.get_by_source_session()` | Task 4 |
| `backtrack_gold_anchors()` | Task 5 |
| `oracle_label()` uses backtracking | Task 6 |
| `AnchorSynthesizer` class, single LLM call | Task 7 |
| Trim input by cosine before synthesis | Task 7 |
| Synthesis raises → caller catches | Task 7 |
| AnchorSynthesizer instantiated once in pipeline | Task 9 |
| Packer scoring formula removed | Task 8 |
| Synthesis with packer fallback | Task 9 |
| README updated | Task 10 |

**Gaps found and addressed:**
- The spec says fingerprints stored as `.jsonl`; plan uses `.json` (simpler single-file load/save, equivalent for this use case — decision recorded).
- The spec says fingerprint store path is `atlas_v{N}/fingerprints.jsonl`; plan uses `fingerprints.json` in the same dir. Functionally identical.
- The `packed_context_for_anchors()` method in `pipeline.py` also calls `pack_memories_direct()` — since we're removing optional params with default values, the call signature is already compatible after Task 8. No change needed.

**Type consistency check:** All method signatures are consistent across tasks — `synthesize(query, memories, query_vec=None, token_budget=700, max_memories=30)` is defined in Task 7 and used in Task 9 with the same signature.
