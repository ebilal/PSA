# Selector & Pipeline Quality Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve PSA trained selector and pipeline quality so all users get better search results out of the box — measured by LongMemEval R@5, Exact F1, and LLM-as-judge.

**Architecture:** Fix training data generation (balanced oracle labels), make the selector robust (recall-biased threshold, zero-anchor fallback), widen the retrieval funnel (32 candidates, max 6 anchors), enrich packed context (source-backed evidence, selector-weighted ranking), and add R@5 metric for competitive comparison.

**Tech Stack:** Python 3.13, sentence-transformers 5.x (CrossEncoder), FAISS, SQLite, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-04-10-selector-quality-improvements-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `psa/selector.py` | Modify | Zero-anchor fallback, increase DEFAULT_MAX_K |
| `psa/training/train_selector.py` | Modify | F-beta threshold, max cap, None-label filter |
| `psa/training/data_generator.py` | Modify | Min positive ratio, validation split, skew warning |
| `psa/packer.py` | Modify | Selector-weighted scoring, source enrichment |
| `psa/pipeline.py` | Modify | Pass selector scores to packer, wider shortlist default, assistant-reference flag |
| `psa/retriever.py` | Modify | Default top_k 24→32 |
| `psa/anchor.py` | Modify | Default top_k 24→32 |
| `psa/benchmarks/longmemeval.py` | Modify | R@5 metric, oracle-label subcommand, bug fixes |
| `psa/cli.py` | Modify | oracle-label CLI, packer-weights flag |
| `psa/training/oracle_labeler.py` | Modify | Label successful queries too |
| `psa/lifecycle.py` | Modify | Include successful queries in oracle labeling |
| `tests/test_selector.py` | Modify | Zero-anchor fallback tests |
| `tests/test_training.py` | Modify | Balanced data tests, threshold tests |
| `tests/test_packer.py` | Modify | Selector-weighted scoring, source enrichment tests |
| `tests/test_pipeline.py` | Modify | Score passthrough test |
| `tests/test_longmemeval.py` | Modify | R@5 metric test |

---

### Task 1: Selector Zero-Anchor Fallback

**Files:**
- Modify: `psa/selector.py:32-37,134-151`
- Modify: `psa/selector.py:196-236`
- Test: `tests/test_selector.py`

- [ ] **Step 1: Write failing tests for zero-anchor fallback and new constants**

Add to `tests/test_selector.py`:

```python
from psa.selector import (
    AnchorSelector,
    SelectedAnchor,
    _cosine_select,
    _trained_select,
    DEFAULT_MAX_K,
    TRAINED_THRESHOLD_DEFAULT,
    TRAINED_THRESHOLD_MAX,
)


def test_default_max_k_is_6():
    assert DEFAULT_MAX_K == 6


def test_trained_threshold_max_is_025():
    assert TRAINED_THRESHOLD_MAX == 0.25


def test_trained_select_fallback_on_zero_anchors():
    """When all candidates score below threshold, return top-1 by score."""
    candidates = [_make_candidate(1, 0.8), _make_candidate(2, 0.6)]
    mock_ce = MagicMock()
    # Both scores below threshold of 0.25
    mock_ce.predict.return_value = [0.1, 0.05]

    selected = _trained_select(
        query="test query",
        candidates=candidates,
        cross_encoder=mock_ce,
        max_k=4,
        threshold=0.25,
    )
    # Should return top-1 as fallback, not empty
    assert len(selected) == 1
    assert selected[0].selector_score == 0.1


def test_trained_select_respects_threshold_when_above():
    """Normal case: candidates above threshold are selected."""
    candidates = [_make_candidate(1, 0.8), _make_candidate(2, 0.6), _make_candidate(3, 0.4)]
    mock_ce = MagicMock()
    mock_ce.predict.return_value = [0.8, 0.3, 0.1]

    selected = _trained_select(
        query="test query",
        candidates=candidates,
        cross_encoder=mock_ce,
        max_k=6,
        threshold=0.25,
    )
    assert len(selected) == 2
    assert selected[0].selector_score == 0.8
    assert selected[1].selector_score == 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_selector.py::test_default_max_k_is_6 tests/test_selector.py::test_trained_threshold_max_is_025 tests/test_selector.py::test_trained_select_fallback_on_zero_anchors tests/test_selector.py::test_trained_select_respects_threshold_when_above -v`

Expected: FAIL (constants wrong, `_trained_select` not importable or returns empty list)

- [ ] **Step 3: Update constants and add fallback logic**

In `psa/selector.py`, change:

```python
DEFAULT_MAX_K = 6        # maximum anchors to return (was 4)
DEFAULT_THRESHOLD = 0.0

TRAINED_MAX_SEQ = 320
TRAINED_THRESHOLD_DEFAULT = 0.3
TRAINED_THRESHOLD_MAX = 0.25    # NEW: safety cap — tau never exceeds this
```

In `_trained_select()` (around line 138-151), replace the selection loop:

```python
    selected = []
    for cand, score in scored[:max_k]:
        if float(score) < threshold:
            break
        selected.append(
            SelectedAnchor(
                anchor_id=cand.anchor_id,
                selector_score=float(score),
                mode="trained",
                candidate=cand,
            )
        )

    # Fallback: never return 0 anchors — take top-1 by score
    if not selected and scored:
        cand, score = scored[0]
        selected.append(
            SelectedAnchor(
                anchor_id=cand.anchor_id,
                selector_score=float(score),
                mode="trained",
                candidate=cand,
            )
        )

    return selected
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_selector.py -v`

Expected: ALL PASS

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check psa/selector.py tests/test_selector.py --fix
uv run ruff format psa/selector.py tests/test_selector.py
git add psa/selector.py tests/test_selector.py
git commit -m "feat(selector): zero-anchor fallback, increase max_k to 6, add threshold cap"
```

---

### Task 2: Retriever Shortlist 24→32

**Files:**
- Modify: `psa/anchor.py:134`
- Modify: `psa/retriever.py:173`
- Modify: `psa/pipeline.py:130`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: Write failing test for new default**

Add to `tests/test_retriever.py`:

```python
from psa.retriever import AnchorRetriever
import inspect


def test_retriever_default_top_k_is_32():
    sig = inspect.signature(AnchorRetriever.retrieve)
    assert sig.parameters["top_k"].default == 32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retriever.py::test_retriever_default_top_k_is_32 -v`

Expected: FAIL (default is 24)

- [ ] **Step 3: Update defaults in three files**

In `psa/anchor.py:134`, change:
```python
    def search(self, query_vec: List[float], top_k: int = 32) -> List[Tuple[int, float]]:
```

In `psa/retriever.py:173`, change:
```python
        top_k: int = 32,
```

In `psa/pipeline.py:130`, change:
```python
        top_k_candidates: int = 32,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retriever.py tests/test_pipeline.py -v`

Expected: ALL PASS

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check psa/anchor.py psa/retriever.py psa/pipeline.py tests/test_retriever.py --fix
uv run ruff format psa/anchor.py psa/retriever.py psa/pipeline.py tests/test_retriever.py
git add psa/anchor.py psa/retriever.py psa/pipeline.py tests/test_retriever.py
git commit -m "feat(retriever): widen shortlist from 24 to 32 candidates"
```

---

### Task 3: Threshold Calibration — F-beta with Cap

**Files:**
- Modify: `psa/training/train_selector.py:377-403`
- Test: `tests/test_training.py`

- [ ] **Step 1: Write failing tests for new threshold logic**

Add to `tests/test_training.py`:

```python
from psa.training.train_selector import SelectorTrainer, THRESHOLD_MAX_CAP


def test_threshold_max_cap_constant():
    assert THRESHOLD_MAX_CAP == 0.25


def test_compute_threshold_fbeta_prefers_recall():
    """F-beta with beta=2 should prefer recall over precision."""
    trainer = SelectorTrainer(output_dir="/tmp/test_selector")

    # Create a fake validation file: many positives at low scores
    import tempfile, json, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Positives with scores that would be 0.15-0.30
        for i in range(50):
            f.write(json.dumps({"query": f"q{i}", "anchor_card": "card", "label": 1,
                                "anchor_id": i, "query_family": "single_anchor",
                                "example_type": "positive", "source_query_id": f"q{i}"}) + "\n")
        # Negatives
        for i in range(50):
            f.write(json.dumps({"query": f"nq{i}", "anchor_card": "card", "label": 0,
                                "anchor_id": i + 100, "query_family": "single_anchor",
                                "example_type": "easy_negative", "source_query_id": f"nq{i}"}) + "\n")
        val_path = f.name

    # Mock model that gives positives score ~0.2, negatives score ~0.05
    mock_model = MagicMock()
    mock_model.predict.return_value = (
        [0.2] * 50 + [0.05] * 50
    )

    tau = trainer._compute_threshold(mock_model, val_path)
    os.unlink(val_path)

    # F-beta (beta=2) should find a low threshold to capture the 0.2 positives
    assert tau <= 0.25, f"Threshold {tau} too high — should be recall-biased"
    assert tau >= 0.05, f"Threshold {tau} too low"


def test_compute_threshold_never_exceeds_cap():
    """Even with clean data, threshold must not exceed THRESHOLD_MAX_CAP."""
    trainer = SelectorTrainer(output_dir="/tmp/test_selector")

    import tempfile, json, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(50):
            f.write(json.dumps({"query": f"q{i}", "anchor_card": "card", "label": 1,
                                "anchor_id": i, "query_family": "single_anchor",
                                "example_type": "positive", "source_query_id": f"q{i}"}) + "\n")
        for i in range(50):
            f.write(json.dumps({"query": f"nq{i}", "anchor_card": "card", "label": 0,
                                "anchor_id": i + 100, "query_family": "single_anchor",
                                "example_type": "easy_negative", "source_query_id": f"nq{i}"}) + "\n")
        val_path = f.name

    # Perfect separation: positives=0.9, negatives=0.1 — Youden's J would pick ~0.5
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9] * 50 + [0.1] * 50

    tau = trainer._compute_threshold(mock_model, val_path)
    os.unlink(val_path)

    assert tau <= 0.25, f"Threshold {tau} exceeds cap of 0.25"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training.py::test_threshold_max_cap_constant tests/test_training.py::test_compute_threshold_fbeta_prefers_recall tests/test_training.py::test_compute_threshold_never_exceeds_cap -v`

Expected: FAIL (THRESHOLD_MAX_CAP not defined, Youden's J logic returns different values)

- [ ] **Step 3: Implement F-beta threshold with cap**

In `psa/training/train_selector.py`, add constant near the top (after existing constants):

```python
THRESHOLD_MAX_CAP = 0.25  # safety cap — tau never exceeds this
```

Replace `_compute_threshold()` method (lines 377-403):

```python
    def _compute_threshold(self, model, val_data_path: str) -> float:
        """
        Find threshold tau via F-beta (beta=2, recall-weighted).

        Beta=2 penalizes false negatives (rejected good anchors) 2x more
        than false positives (included mediocre anchors). This matches the
        real cost: an extra anchor wastes some token budget, a missing anchor
        loses the answer.

        Threshold is capped at THRESHOLD_MAX_CAP to prevent over-filtering.
        """
        examples = _load_training_data(val_data_path)
        if not examples:
            return 0.15
        pairs = [(ex["query"], ex["anchor_card"]) for ex in examples if ex.get("label") is not None]
        labels = [ex["label"] for ex in examples if ex.get("label") is not None]
        if not pairs:
            return 0.15
        scores = model.predict(pairs)

        beta = 2.0
        beta_sq = beta * beta
        best_tau = 0.15
        best_fbeta = -1.0

        for tau in [i / 20 for i in range(1, 20)]:
            tp = sum(1 for s, lbl in zip(scores, labels) if s >= tau and lbl == 1)
            fp = sum(1 for s, lbl in zip(scores, labels) if s >= tau and lbl == 0)
            fn = sum(1 for s, lbl in zip(scores, labels) if s < tau and lbl == 1)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            denom = (beta_sq * precision) + recall
            fbeta = ((1 + beta_sq) * precision * recall / denom) if denom > 0 else 0.0
            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_tau = tau

        return min(best_tau, THRESHOLD_MAX_CAP)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training.py -v`

Expected: ALL PASS

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check psa/training/train_selector.py tests/test_training.py --fix
uv run ruff format psa/training/train_selector.py tests/test_training.py
git add psa/training/train_selector.py tests/test_training.py
git commit -m "feat(training): F-beta threshold calibration with 0.25 cap"
```

---

### Task 4: Training Data Balance — Min Positive Ratio and Validation Split

**Files:**
- Modify: `psa/training/data_generator.py:117-180`
- Test: `tests/test_training.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_training.py`:

```python
from psa.training.data_generator import MIN_POSITIVE_RATIO


def test_min_positive_ratio_constant():
    assert MIN_POSITIVE_RATIO == 0.25


def test_generate_enforces_min_positive_ratio(tmp_path):
    """Even with few oracle positives, output should have >= 25% positives."""
    oracle_path = tmp_path / "oracle_labels.jsonl"
    # 5 labels with winners, 95 without — severely skewed
    import json
    labels = []
    for i in range(5):
        labels.append({
            "query_id": f"q_{i}", "query": f"query {i}",
            "winning_oracle_set": [i], "candidate_anchor_ids": list(range(10)),
            "all_sets": [[i]], "winning_oracle_score": 0.8,
        })
    for i in range(95):
        labels.append({
            "query_id": f"nq_{i}", "query": f"neg query {i}",
            "winning_oracle_set": [], "candidate_anchor_ids": list(range(10)),
            "all_sets": [], "winning_oracle_score": 0.1,
        })
    oracle_path.write_text("\n".join(json.dumps(l) for l in labels))

    anchor_cards = {i: f"Anchor card {i}" for i in range(10)}
    gen = DataGenerator(str(oracle_path), anchor_cards, seed=42)
    output_path = str(tmp_path / "training.jsonl")
    n = gen.generate(output_path, n_examples=200)

    with open(output_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    positives = [e for e in examples if e["label"] == 1]
    ratio = len(positives) / len(examples) if examples else 0
    assert ratio >= 0.20, f"Positive ratio {ratio:.2f} below minimum"


def test_generate_warns_on_low_positives(tmp_path, caplog):
    """Should log warning when positive ratio is low before enforcement."""
    import logging
    oracle_path = tmp_path / "oracle_labels.jsonl"
    import json
    # All failures — no positives possible
    labels = []
    for i in range(50):
        labels.append({
            "query_id": f"q_{i}", "query": f"query {i}",
            "winning_oracle_set": [], "candidate_anchor_ids": [0, 1, 2],
            "all_sets": [], "winning_oracle_score": 0.1,
        })
    oracle_path.write_text("\n".join(json.dumps(l) for l in labels))

    anchor_cards = {i: f"Anchor card {i}" for i in range(5)}
    gen = DataGenerator(str(oracle_path), anchor_cards, seed=42)
    output_path = str(tmp_path / "training.jsonl")

    with caplog.at_level(logging.WARNING, logger="psa.training.data_generator"):
        gen.generate(output_path, n_examples=100)

    assert any("positive" in r.message.lower() for r in caplog.records), \
        "Expected warning about low positive ratio"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training.py::test_min_positive_ratio_constant tests/test_training.py::test_generate_enforces_min_positive_ratio tests/test_training.py::test_generate_warns_on_low_positives -v`

Expected: FAIL

- [ ] **Step 3: Implement balance enforcement**

In `psa/training/data_generator.py`, add constant after the MIX constants (around line 47):

```python
MIN_POSITIVE_RATIO = 0.25  # minimum fraction of positive examples in output
```

In the `generate()` method (around line 137-157), after assembling all examples and before shuffle, add balance enforcement:

```python
        examples: List[TrainingExample] = []

        # 1. Synthetic positives from oracle labels
        examples.extend(
            self._generate_positives(n_synthetic // 2)
        )
        examples.extend(
            self._generate_easy_negatives(n_synthetic // 2)
        )

        # 2. Hard negatives
        examples.extend(
            self._generate_hard_negatives(n_hard_neg)
        )

        # 3. Adversarial rewrites
        examples.extend(
            self._generate_adversarial(n_adversarial)
        )

        # 4. Enforce minimum positive ratio
        positives = [e for e in examples if e.label == 1]
        negatives = [e for e in examples if e.label == 0]
        pos_ratio = len(positives) / len(examples) if examples else 0

        if pos_ratio < 0.10:
            logger.warning(
                "Very low positive ratio (%.1f%%) — oracle labels may lack "
                "winning_oracle_set entries. Training quality will be poor.",
                pos_ratio * 100,
            )

        if positives and pos_ratio < MIN_POSITIVE_RATIO:
            # Oversample positives to reach minimum ratio
            target_pos = int(len(negatives) * MIN_POSITIVE_RATIO / (1 - MIN_POSITIVE_RATIO))
            extra_needed = target_pos - len(positives)
            if extra_needed > 0:
                oversampled = [self.rng.choice(positives) for _ in range(extra_needed)]
                examples = positives + oversampled + negatives
                logger.info(
                    "Oversampled %d positives to reach %.0f%% positive ratio.",
                    extra_needed, MIN_POSITIVE_RATIO * 100,
                )

        self.rng.shuffle(examples)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training.py -v`

Expected: ALL PASS

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check psa/training/data_generator.py tests/test_training.py --fix
uv run ruff format psa/training/data_generator.py tests/test_training.py
git add psa/training/data_generator.py tests/test_training.py
git commit -m "feat(training): enforce min 25% positive ratio, warn on skew"
```

---

### Task 5: Pass Selector Scores to Packer

**Files:**
- Modify: `psa/pipeline.py:219-236`
- Modify: `psa/packer.py:315-348`
- Test: `tests/test_packer.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test for packer accepting selector scores**

Add to `tests/test_packer.py`:

```python
def test_pack_memories_direct_accepts_selector_scores():
    """pack_memories_direct should accept optional selector_scores dict."""
    import inspect
    from psa.packer import EvidencePacker
    sig = inspect.signature(EvidencePacker.pack_memories_direct)
    assert "selector_scores" in sig.parameters


def test_pack_memories_direct_accepts_packer_weights():
    """pack_memories_direct should accept optional packer_weights tuple."""
    import inspect
    from psa.packer import EvidencePacker
    sig = inspect.signature(EvidencePacker.pack_memories_direct)
    assert "packer_weights" in sig.parameters
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_packer.py::test_pack_memories_direct_accepts_selector_scores tests/test_packer.py::test_pack_memories_direct_accepts_packer_weights -v`

Expected: FAIL (parameters don't exist)

- [ ] **Step 3: Add selector_scores and packer_weights to packer**

In `psa/packer.py`, update `pack_memories_direct()` signature (around line 315):

```python
    def pack_memories_direct(
        self,
        query: str,
        memories: List[MemoryObject],
        token_budget: int = 6000,
        query_vec: Optional[List[float]] = None,
        store: Optional[MemoryStore] = None,
        selector_scores: Optional[Dict[int, float]] = None,
        packer_weights: Optional[Tuple[float, float, float]] = None,
    ) -> PackedContext:
```

Add `Dict` and `Tuple` to imports at top of file if not already present:
```python
from typing import Any, Dict, List, Optional, Tuple
```

Update the scoring formula (around line 344-348):

```python
        # Combine relevance with quality and optional selector scores for ranking
        if selector_scores and packer_weights:
            w_sel, w_cos, w_qual = packer_weights
            scored = sorted(
                zip(memories, relevances),
                key=lambda pair: (
                    w_sel * selector_scores.get(pair[0].primary_anchor_id, 0.0)
                    + w_cos * pair[1]
                    + w_qual * pair[0].quality_score
                ),
                reverse=True,
            )
        else:
            scored = sorted(
                zip(memories, relevances),
                key=lambda pair: (pair[1] * 0.7 + pair[0].quality_score * 0.3),
                reverse=True,
            )
```

- [ ] **Step 4: Update pipeline to pass scores through**

In `psa/pipeline.py`, update Step 5 (around line 228-236). Build a selector_scores dict from selected anchors and pass it:

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
        )
        timing.pack_ms = (time.perf_counter() - t0) * 1000
```

Add `_packer_weights` to `PSAPipeline.__init__()` — find the constructor and add the attribute. The default is `(0.4, 0.3, 0.3)` (balanced — final value determined by weight sweep):

```python
        self._packer_weights = (0.4, 0.3, 0.3)  # (selector, cosine, quality)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_packer.py tests/test_pipeline.py -v`

Expected: ALL PASS

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check psa/packer.py psa/pipeline.py tests/test_packer.py --fix
uv run ruff format psa/packer.py psa/pipeline.py tests/test_packer.py
git add psa/packer.py psa/pipeline.py tests/test_packer.py
git commit -m "feat(packer): selector-weighted memory ranking with configurable weights"
```

---

### Task 6: Source-Backed Context Enrichment

**Files:**
- Modify: `psa/packer.py:138-153,360-384`
- Test: `tests/test_packer.py`

- [ ] **Step 1: Write failing test for source enrichment**

Add to `tests/test_packer.py`:

```python
from unittest.mock import MagicMock, patch
from psa.memory_object import MemoryObject, MemoryType, EvidenceSpan, RawSource


def test_pack_enriches_top_memories_with_source_text():
    """Top-ranked memories should include evidence span text from raw source."""
    from psa.packer import EvidencePacker, _fetch_evidence_text

    source = RawSource(
        source_id="src1",
        tenant_id="test",
        source_type="conversation",
        source_path="chat.jsonl",
        title="chat",
        full_text="Alice said she prefers PostgreSQL because of the JSONB support and extensibility.",
        created_at="2026-01-01T00:00:00Z",
    )
    span = EvidenceSpan(source_id="src1", start_offset=20, end_offset=80)

    store = MagicMock()
    store.get_source.return_value = source

    text = _fetch_evidence_text(store, [span], max_chars=500)
    assert "PostgreSQL" in text
    assert "JSONB" in text


def test_fetch_evidence_text_keyword_fallback():
    """When no evidence spans, fall back to keyword window extraction."""
    from psa.packer import _fetch_evidence_text

    source = RawSource(
        source_id="src1",
        tenant_id="test",
        source_type="conversation",
        source_path="chat.jsonl",
        title="chat",
        full_text="Some preamble. " * 20 + "The PostgreSQL migration was decided. " + "Some epilogue. " * 20,
        created_at="2026-01-01T00:00:00Z",
    )
    store = MagicMock()
    store.get_source.return_value = source

    # No spans — use keyword search with body text as hint
    text = _fetch_evidence_text(store, [], max_chars=500, body_hint="PostgreSQL migration")
    assert "PostgreSQL" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_packer.py::test_pack_enriches_top_memories_with_source_text tests/test_packer.py::test_fetch_evidence_text_keyword_fallback -v`

Expected: FAIL (`_fetch_evidence_text` not defined)

- [ ] **Step 3: Implement `_fetch_evidence_text` helper**

Add to `psa/packer.py` (after `_fetch_source_path`, around line 153):

```python
def _fetch_evidence_text(
    store: Optional[MemoryStore],
    evidence_spans: List,
    max_chars: int = 500,
    body_hint: str = "",
    source_ids: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Retrieve relevant portions of raw source text.

    Priority:
    1. Evidence spans (exact offsets into raw source)
    2. Keyword window (find body_hint terms in raw source, extract surrounding context)

    Returns extracted text or None if no source available.
    """
    if not store:
        return None

    # Try evidence spans first
    if evidence_spans:
        chunks = []
        chars_remaining = max_chars
        for span in evidence_spans:
            if chars_remaining <= 0:
                break
            try:
                source = store.get_source(span.source_id)
                if source and source.full_text:
                    start = max(0, span.start_offset)
                    end = min(len(source.full_text), span.end_offset)
                    chunk = source.full_text[start:end].strip()
                    if chunk:
                        chunks.append(chunk)
                        chars_remaining -= len(chunk)
            except Exception:
                continue
        if chunks:
            return " ... ".join(chunks)[:max_chars]

    # Fallback: keyword window from raw source
    if body_hint and source_ids:
        import re
        hint_words = [w for w in re.findall(r"\b\w{4,}\b", body_hint.lower()) if len(w) > 3]
        if not hint_words:
            return None
        for sid in source_ids[:1]:  # check first source only
            try:
                source = store.get_source(sid)
                if not source or not source.full_text:
                    continue
                text_lower = source.full_text.lower()
                # Find best keyword match position
                best_pos = -1
                for word in hint_words:
                    pos = text_lower.find(word)
                    if pos >= 0:
                        best_pos = pos
                        break
                if best_pos >= 0:
                    window = max_chars // 2
                    start = max(0, best_pos - window)
                    end = min(len(source.full_text), best_pos + window)
                    return source.full_text[start:end].strip()
            except Exception:
                continue

    return None
```

- [ ] **Step 4: Wire enrichment into `pack_memories_direct`**

In the top-N loop (around line 360-384), after the `source_ctx` block, add evidence enrichment:

```python
            # Top-ranked memories get full body text and source provenance
            is_top = rank < TOP_N_WITH_SOURCE
            source_ctx = None
            evidence_text = None
            if is_top and store:
                source_file = _fetch_source_path(store, mo.source_ids)
                if source_file:
                    source_ctx = f"[from {source_file}]"
                # Enrich with raw source evidence
                evidence_text = _fetch_evidence_text(
                    store,
                    mo.evidence_spans,
                    max_chars=500,
                    body_hint=mo.body,
                    source_ids=mo.source_ids,
                )

            item = _format_memory_item(
                mo,
                similarity=relevance if relevance > 0 else None,
                source_context=source_ctx,
                max_body_chars=800 if is_top else 300,
                evidence_text=evidence_text,
            )
```

Update `_format_memory_item()` to accept and render `evidence_text`. Find the function and add the parameter:

```python
def _format_memory_item(
    mo: MemoryObject,
    similarity: Optional[float] = None,
    source_context: Optional[str] = None,
    max_body_chars: int = 400,
    evidence_text: Optional[str] = None,
) -> str:
```

At the end of `_format_memory_item`, before returning, append evidence if present:

```python
    if evidence_text:
        parts.append(f"  Source context: {evidence_text}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_packer.py -v`

Expected: ALL PASS

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check psa/packer.py tests/test_packer.py --fix
uv run ruff format psa/packer.py tests/test_packer.py
git add psa/packer.py tests/test_packer.py
git commit -m "feat(packer): source-backed context enrichment via evidence spans"
```

---

### Task 7: Assistant-Turn Awareness

**Files:**
- Modify: `psa/pipeline.py`
- Modify: `psa/packer.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test for assistant-reference detection**

Add to `tests/test_pipeline.py`:

```python
from psa.pipeline import _is_assistant_reference


def test_detects_assistant_reference():
    assert _is_assistant_reference("What did you suggest about auth?")
    assert _is_assistant_reference("You told me to use GraphQL")
    assert _is_assistant_reference("Remind me what you recommended")


def test_non_assistant_reference():
    assert not _is_assistant_reference("What is the database schema?")
    assert not _is_assistant_reference("How does auth work?")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py::test_detects_assistant_reference tests/test_pipeline.py::test_non_assistant_reference -v`

Expected: FAIL (`_is_assistant_reference` not defined)

- [ ] **Step 3: Implement assistant-reference detector**

Add to `psa/pipeline.py` (module-level, near the top):

```python
_ASSISTANT_TRIGGERS = [
    "you suggested", "you told me", "you mentioned", "you said",
    "you recommended", "remind me what you", "you provided",
    "you listed", "you gave me", "you described", "what did you",
    "you came up with", "you helped me", "you explained",
    "can you remind me", "you identified",
]


def _is_assistant_reference(query: str) -> bool:
    """Detect queries asking about what the AI previously said."""
    q = query.lower()
    return any(t in q for t in _ASSISTANT_TRIGGERS)
```

In `PSAPipeline.query()`, before Step 5 (packing), pass the flag:

```python
        # Step 5: Pack context
        t0 = time.perf_counter()

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
```

Add `include_assistant_turns` parameter to `pack_memories_direct()` in `psa/packer.py`:

```python
    def pack_memories_direct(
        self,
        query: str,
        memories: List[MemoryObject],
        token_budget: int = 6000,
        query_vec: Optional[List[float]] = None,
        store: Optional[MemoryStore] = None,
        selector_scores: Optional[Dict[int, float]] = None,
        packer_weights: Optional[Tuple[float, float, float]] = None,
        include_assistant_turns: bool = False,
    ) -> PackedContext:
```

In the evidence enrichment (from Task 6), when `include_assistant_turns` is True, the existing `_fetch_evidence_text` already works on `full_text` which contains both user and assistant turns. No additional change needed — the flag serves as documentation and can be used for future refinements.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`

Expected: ALL PASS

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check psa/pipeline.py psa/packer.py tests/test_pipeline.py --fix
uv run ruff format psa/pipeline.py psa/packer.py tests/test_pipeline.py
git add psa/pipeline.py psa/packer.py tests/test_pipeline.py
git commit -m "feat(pipeline): assistant-turn awareness for 'what did you suggest' queries"
```

---

### Task 8: R@5 Metric in Benchmark

**Files:**
- Modify: `psa/benchmarks/longmemeval.py`
- Modify: `psa/cli.py`
- Test: `tests/test_longmemeval.py`

- [ ] **Step 1: Write failing test for R@5 scoring**

Add to `tests/test_longmemeval.py`:

```python
def test_recall_at_k():
    from psa.benchmarks.longmemeval import _recall_at_k

    # One of the answer sessions was retrieved
    retrieved_source_paths = ["answer_abc_1.jsonl", "answer_xyz_2.jsonl", "other.jsonl"]
    answer_session_ids = ["answer_abc_1", "answer_abc_2"]

    assert _recall_at_k(retrieved_source_paths, answer_session_ids, k=5) == 1.0


def test_recall_at_k_miss():
    from psa.benchmarks.longmemeval import _recall_at_k

    retrieved_source_paths = ["other1.jsonl", "other2.jsonl"]
    answer_session_ids = ["answer_abc_1"]

    assert _recall_at_k(retrieved_source_paths, answer_session_ids, k=5) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_longmemeval.py::test_recall_at_k tests/test_longmemeval.py::test_recall_at_k_miss -v`

Expected: FAIL (`_recall_at_k` not defined)

- [ ] **Step 3: Implement R@5 in benchmark**

Add to `psa/benchmarks/longmemeval.py`:

```python
def _recall_at_k(
    retrieved_source_paths: List[str],
    answer_session_ids: List[str],
    k: int = 5,
) -> float:
    """
    Recall@k: did any answer session appear in the top-k retrieved sources?

    retrieved_source_paths are filenames like "answer_abc_1.jsonl".
    answer_session_ids are like "answer_abc_1".
    Match by checking if any answer_session_id is a prefix of any retrieved path.
    """
    if not answer_session_ids:
        return 0.0
    top_k = retrieved_source_paths[:k]
    for ans_id in answer_session_ids:
        for path in top_k:
            # Strip extension for comparison
            path_stem = path.rsplit(".", 1)[0] if "." in path else path
            if path_stem == ans_id or ans_id in path:
                return 1.0
    return 0.0
```

Update the `run()` function to include `answer_session_ids` in the results JSONL. In the record dict (around line 200-215), add:

```python
            record = {
                "question_id": question_id,
                "question": question,
                "context_text": context_text,
                "answer_generated": generated,
                "answer_gold": gold_answer,
                "answer_session_ids": example.get("answer_session_ids", []),
                # ... rest of existing fields ...
            }
```

Update `score()` to compute and report R@5. After the exact_f1 computation:

```python
    # Compute R@5 if answer_session_ids are available
    recall_scores = []
    for r in records:
        ans_ids = r.get("answer_session_ids", [])
        # Extract source paths from context (memories trace back to source files)
        # For now, use a simple heuristic: check if any answer session ID appears
        # in the context text
        if ans_ids:
            hit = any(
                any(aid in (r.get("context_text", "")) for aid in ans_ids)
            )
            recall_scores.append(1.0 if hit else 0.0)
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else None
```

Add R@5 to the result dict:

```python
    if avg_recall is not None:
        result["recall_at_5"] = round(avg_recall, 4)
```

- [ ] **Step 4: Update CLI score output**

In `psa/cli.py`, in `_cmd_longmemeval` score section (around line 835-842), add:

```python
        if "recall_at_5" in stats:
            print(f"  R@5:           {stats['recall_at_5']:.3f}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_longmemeval.py -v`

Expected: ALL PASS

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check psa/benchmarks/longmemeval.py psa/cli.py tests/test_longmemeval.py --fix
uv run ruff format psa/benchmarks/longmemeval.py psa/cli.py tests/test_longmemeval.py
git add psa/benchmarks/longmemeval.py psa/cli.py tests/test_longmemeval.py
git commit -m "feat(benchmark): add R@5 metric for MemPalace comparison"
```

---

### Task 9: Oracle-Label Benchmark Subcommand

**Files:**
- Modify: `psa/benchmarks/longmemeval.py`
- Modify: `psa/cli.py`

- [ ] **Step 1: Add oracle_label function to benchmark module**

Add to `psa/benchmarks/longmemeval.py`:

```python
def oracle_label(
    results_path: str,
    tenant_id: str = BENCH_TENANT,
) -> int:
    """
    Run the real OracleLabeler on benchmark results to produce proper
    training labels with identified winning anchor sets.

    Returns the number of oracle labels written.
    """
    from ..training.oracle_labeler import OracleLabeler
    from ..pipeline import PSAPipeline
    from ..tenant import TenantManager

    records = _load_results(results_path)
    if not records:
        raise ValueError(f"No records found in {results_path}")

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)

    try:
        pipeline = PSAPipeline.from_tenant(tenant_id=tenant_id)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No atlas for tenant '{tenant_id}'. Run 'psa benchmark longmemeval ingest' first."
        )

    labeler = OracleLabeler(pipeline=pipeline)
    oracle_path = _oracle_labels_path(tenant_id)
    os.makedirs(os.path.dirname(oracle_path), exist_ok=True)

    written = 0
    with open(oracle_path, "a", encoding="utf-8") as f:
        for i, record in enumerate(records):
            query = record["question"]
            try:
                label = labeler.label(query)
                f.write(json.dumps(label.to_dict()) + "\n")
                written += 1
            except Exception as e:
                logger.warning("Oracle labeling failed for q=%s: %s", record.get("question_id"), e)

            if (i + 1) % 50 == 0:
                logger.info("  %d / %d questions labeled", i + 1, len(records))

    logger.info("Wrote %d oracle labels to %s", written, oracle_path)
    return written
```

- [ ] **Step 2: Wire into CLI**

In `psa/cli.py`, in the `_cmd_longmemeval` function, add an `oracle-label` branch after the `score` branch:

```python
    elif lme_action == "oracle-label":
        from .benchmarks.longmemeval import oracle_label

        results_file = getattr(args, "results", None)
        if not results_file:
            import glob
            results_dir = os.path.expanduser("~/.psa/benchmarks/longmemeval")
            files = sorted(glob.glob(os.path.join(results_dir, "results_*.jsonl")))
            if not files:
                print("No results files found. Run 'psa benchmark longmemeval run' first.")
                sys.exit(1)
            results_file = files[-1]

        print(f"Running oracle labeling on: {results_file}")
        print("This uses LLM calls and may take 30-60 minutes...")
        n = oracle_label(results_file, tenant_id=tenant_id)
        print(f"\nOracle labels written: {n}")
        print("Run 'psa train' to train the selector on these labels.")
```

In the longmemeval subparser setup (around line 1321-1342), add:

```python
    p_lme_oracle = lme_sub.add_parser("oracle-label", help="Run full oracle labeling on results")
    p_lme_oracle.add_argument("--results", default=None, help="Results JSONL (auto-detects latest)")
```

- [ ] **Step 3: Lint and commit**

```bash
uv run ruff check psa/benchmarks/longmemeval.py psa/cli.py --fix
uv run ruff format psa/benchmarks/longmemeval.py psa/cli.py
git add psa/benchmarks/longmemeval.py psa/cli.py
git commit -m "feat(benchmark): add oracle-label subcommand for proper training data"
```

---

### Task 10: Benchmark Bug Fixes

**Files:**
- Modify: `psa/benchmarks/longmemeval.py` (already done in working tree)
- Modify: `psa/training/train_selector.py` (already done in working tree)
- Modify: `psa/cli.py` (already done in working tree)

- [ ] **Step 1: Verify all existing fixes are in place**

Run: `uv run pytest tests/ -v --tb=short 2>&1 | tail -30`

Expected: All existing tests pass.

- [ ] **Step 2: Run ruff on all modified files**

```bash
uv run ruff check psa/benchmarks/longmemeval.py psa/training/train_selector.py psa/cli.py --fix
uv run ruff format psa/benchmarks/longmemeval.py psa/training/train_selector.py psa/cli.py
```

- [ ] **Step 3: Commit bug fixes**

```bash
git add psa/benchmarks/longmemeval.py psa/training/train_selector.py psa/cli.py
git commit -m "fix: benchmark pipeline bug fixes (HF download, int normalization, None labels, selector CLI)"
```

---

### Task 11: Packer Weights CLI Flag

**Files:**
- Modify: `psa/benchmarks/longmemeval.py`
- Modify: `psa/cli.py`

- [ ] **Step 1: Add `--packer-weights` to benchmark run subcommand**

In `psa/cli.py`, in the longmemeval run parser (around line 1329-1339), add:

```python
    p_lme_run.add_argument(
        "--packer-weights", default=None,
        help="Packer weight tuple: selector,cosine,quality (e.g., '0.5,0.3,0.2')",
    )
```

In `_cmd_longmemeval`, update the `run` branch to parse and pass weights:

```python
    elif lme_action == "run":
        split = getattr(args, "split", "val")
        limit = getattr(args, "limit", None)
        selector_mode = getattr(args, "selector", "cosine")
        selector_model_path = getattr(args, "selector_model", None)
        packer_weights_str = getattr(args, "packer_weights", None)
        packer_weights = None
        if packer_weights_str:
            parts = packer_weights_str.split(",")
            packer_weights = tuple(float(p) for p in parts)
        print(
            f"Running LongMemEval ({split} split, {'all' if not limit else limit} questions, "
            f"selector={selector_mode}"
            f"{f', weights={packer_weights}' if packer_weights else ''})..."
        )
        out_path = run(
            split=split,
            limit=limit,
            tenant_id=tenant_id,
            selector_mode=selector_mode,
            selector_model_path=selector_model_path,
            packer_weights=packer_weights,
        )
```

- [ ] **Step 2: Add `packer_weights` parameter to `run()` in longmemeval.py**

Update the `run()` function signature:

```python
def run(
    split: str = "val",
    limit: Optional[int] = None,
    tenant_id: str = BENCH_TENANT,
    results_dir: str = RESULTS_DIR_DEFAULT,
    token_budget: int = 6000,
    selector_mode: str = "cosine",
    selector_model_path: Optional[str] = None,
    packer_weights: Optional[tuple] = None,
) -> str:
```

After creating the pipeline, set packer weights if provided:

```python
    try:
        pipeline = PSAPipeline.from_tenant(
            tenant_id=tenant_id,
            token_budget=token_budget,
            selector_mode=selector_mode,
            selector_model_path=selector_model_path,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No atlas for tenant '{tenant_id}'. Run 'psa benchmark longmemeval ingest' first."
        )

    if packer_weights:
        pipeline._packer_weights = packer_weights
```

- [ ] **Step 3: Lint and commit**

```bash
uv run ruff check psa/benchmarks/longmemeval.py psa/cli.py --fix
uv run ruff format psa/benchmarks/longmemeval.py psa/cli.py
git add psa/benchmarks/longmemeval.py psa/cli.py
git commit -m "feat(benchmark): add --packer-weights flag for weight sweep benchmarking"
```

---

### Task 12: Lifecycle Oracle Labeling — Include Successes

**Files:**
- Modify: `psa/lifecycle.py`
- Modify: `psa/training/oracle_labeler.py`

- [ ] **Step 1: Find and update query selection in lifecycle**

In `psa/lifecycle.py`, find where queries are collected for oracle labeling. The lifecycle should sample successful queries (not just poor-performing ones) to ensure balanced labels.

Read the current lifecycle oracle labeling flow and add a sampling step that includes queries with high quality_score alongside the poor performers. The ratio should be roughly 1:2 (successes:failures).

```python
        # Sample successful queries for balanced oracle labels
        if successful_queries and len(successful_queries) > 0:
            n_success_sample = max(len(poor_queries) // 2, 10)
            sampled_success = random.sample(
                successful_queries,
                min(n_success_sample, len(successful_queries)),
            )
            queries_to_label = poor_queries + sampled_success
        else:
            queries_to_label = poor_queries
```

- [ ] **Step 2: Lint and commit**

```bash
uv run ruff check psa/lifecycle.py psa/training/oracle_labeler.py --fix
uv run ruff format psa/lifecycle.py psa/training/oracle_labeler.py
git add psa/lifecycle.py psa/training/oracle_labeler.py
git commit -m "feat(lifecycle): include successful queries in oracle labeling for balanced training"
```

---

### Task 13: Full Test Suite Verification

- [ ] **Step 1: Run the complete test suite**

```bash
uv run pytest tests/ -v --tb=short 2>&1
```

Expected: All tests pass. Fix any failures introduced by the changes.

- [ ] **Step 2: Run lint on entire codebase**

```bash
uv run ruff check . --fix
uv run ruff format .
uv run ruff format --check .
```

Expected: No lint errors.

- [ ] **Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address test and lint issues from selector improvements"
```

---

## Execution Order

Tasks 1-4 are independent core improvements (selector, retriever, threshold, data balance).
Task 5 depends on nothing but is best done after Task 1 (selector constants).
Task 6-7 depend on Task 5 (packer signature changes).
Task 8-9 are benchmark improvements, independent of Tasks 1-7.
Task 10 should be done first or last (bug fixes already in working tree).
Task 11 depends on Task 5 (packer weights).
Task 12 is independent (lifecycle).
Task 13 is always last.

**Recommended parallel groups:**
- Group A: Tasks 1, 2, 3, 4 (independent core changes)
- Group B: Tasks 5, 6, 7 (packer + pipeline, sequential)
- Group C: Tasks 8, 9 (benchmark, sequential)
- Group D: Task 10, 11, 12 (bug fixes + CLI + lifecycle)
- Final: Task 13
