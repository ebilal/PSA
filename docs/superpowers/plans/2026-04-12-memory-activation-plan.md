# Memory Activation System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the retriever-selector lookup chain with a single-pass memory activation system that scores all 256 atlas anchors, learns inter-anchor co-activation patterns, and adaptively selects how many anchors each query needs. Target R@5 >= 0.95.

**Architecture:** Two new components replace the existing pipeline. FullAtlasScorer batches all 256 (query, card_text) pairs through the cross-encoder in one forward pass (~50-100ms on MPS). CoActivationSelector refines those scores with a 2-layer transformer that learns inter-anchor relationships via centroid-based attention, then applies a query-adaptive threshold to select a variable number of anchors.

**Tech Stack:** PyTorch, sentence-transformers (CrossEncoder), existing PSA infrastructure (Atlas, MemoryStore, EmbeddingModel).

**Design spec:** `docs/superpowers/specs/2026-04-12-memory-activation-design.md`

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `psa/full_atlas_scorer.py` | `FullAtlasScorer` class -- batched cross-encoder scoring of all 256 anchors |
| `psa/coactivation.py` | `CoActivationModel` (PyTorch nn.Module) + `CoActivationSelector` (inference wrapper) |
| `psa/training/train_coactivation.py` | `CoActivationTrainer` -- training loop, data loading, loss, saving |
| `psa/training/coactivation_data.py` | Generate 256-anchor training data from oracle labels |
| `tests/test_full_atlas_scorer.py` | Unit tests for FullAtlasScorer |
| `tests/test_coactivation.py` | Unit tests for CoActivationModel + CoActivationSelector |
| `tests/test_train_coactivation.py` | Unit tests for co-activation training pipeline |

### Modified files

| File | Change |
|------|--------|
| `psa/pipeline.py` | New query path using FullAtlasScorer + CoActivationSelector with graceful fallback |
| `psa/lifecycle.py` | Add co-activation training step after selector training in slow path |
| `psa/cli.py` | `psa train --coactivation` flag; `psa benchmark longmemeval run --selector coactivation` |

---

### Task 1: FullAtlasScorer -- Tests

**Files:**
- Create: `tests/test_full_atlas_scorer.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for psa.full_atlas_scorer -- batched cross-encoder over all atlas anchors."""

import numpy as np
import pytest
from unittest.mock import MagicMock


def _make_mock_atlas(n_anchors=8):
    """Build a mock Atlas with n_anchors cards, each with a 768-dim centroid."""
    atlas = MagicMock()
    cards = []
    for i in range(n_anchors):
        card = MagicMock()
        card.anchor_id = i
        card.to_stable_card_text.return_value = f"Anchor {i}: test cluster about topic {i}"
        centroid = np.random.default_rng(i).standard_normal(768).astype(np.float32)
        centroid /= np.linalg.norm(centroid)
        card.centroid = centroid.tolist()
        cards.append(card)
    atlas.cards = cards
    return atlas


def _make_mock_cross_encoder(n_anchors=8):
    """Mock cross-encoder that returns descending scores."""
    ce = MagicMock()
    ce.predict.return_value = np.linspace(1.0, 0.0, n_anchors)
    return ce


class TestFullAtlasScorer:
    def test_score_all_returns_all_anchors(self):
        """score_all() returns one AnchorScore per atlas card."""
        from psa.full_atlas_scorer import FullAtlasScorer, AnchorScore

        atlas = _make_mock_atlas(8)
        ce = _make_mock_cross_encoder(8)

        scorer = FullAtlasScorer(cross_encoder=ce, atlas=atlas)
        scores = scorer.score_all("What auth pattern did we use?")

        assert len(scores) == 8
        assert all(isinstance(s, AnchorScore) for s in scores)
        for s in scores:
            assert hasattr(s, "anchor_id")
            assert hasattr(s, "ce_score")
            assert hasattr(s, "centroid")
            assert len(s.centroid) == 768

    def test_score_all_batches_pairs(self):
        """Verifies that predict is called once with all pairs, not N times."""
        from psa.full_atlas_scorer import FullAtlasScorer

        atlas = _make_mock_atlas(256)
        ce = _make_mock_cross_encoder(256)

        scorer = FullAtlasScorer(cross_encoder=ce, atlas=atlas)
        scorer.score_all("test query")

        ce.predict.assert_called_once()
        pairs = ce.predict.call_args[0][0]
        assert len(pairs) == 256
        assert pairs[0] == ("test query", "Anchor 0: test cluster about topic 0")

    def test_score_all_sorted_by_score_desc(self):
        """Results are sorted by ce_score descending."""
        from psa.full_atlas_scorer import FullAtlasScorer

        atlas = _make_mock_atlas(8)
        ce = _make_mock_cross_encoder(8)

        scorer = FullAtlasScorer(cross_encoder=ce, atlas=atlas)
        scores = scorer.score_all("test query")

        for i in range(len(scores) - 1):
            assert scores[i].ce_score >= scores[i + 1].ce_score

    def test_cosine_fallback_when_no_cross_encoder(self):
        """Without a cross-encoder, scores are cosine similarity to query_vec."""
        from psa.full_atlas_scorer import FullAtlasScorer

        atlas = _make_mock_atlas(8)
        query_vec = np.array(atlas.cards[0].centroid, dtype=np.float32)

        scorer = FullAtlasScorer(cross_encoder=None, atlas=atlas)
        scores = scorer.score_all("ignored", query_vec=query_vec)

        assert len(scores) == 8
        assert scores[0].anchor_id == 0  # closest to itself
        assert scores[0].ce_score > scores[-1].ce_score

    def test_from_model_path_loads_cross_encoder(self, monkeypatch):
        """from_model_path() loads a CrossEncoder from disk."""
        from psa import full_atlas_scorer as mod

        mock_ce = MagicMock()
        monkeypatch.setattr(mod, "_load_cross_encoder", lambda path: mock_ce)

        atlas = _make_mock_atlas(4)
        scorer = mod.FullAtlasScorer.from_model_path("/fake/path", atlas)
        assert scorer._cross_encoder is mock_ce
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_full_atlas_scorer.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'psa.full_atlas_scorer'`

- [ ] **Step 3: Commit**

```bash
git add tests/test_full_atlas_scorer.py
git commit -m "test: add failing tests for FullAtlasScorer"
```

---

### Task 2: FullAtlasScorer -- Implementation

**Files:**
- Create: `psa/full_atlas_scorer.py`

- [ ] **Step 1: Implement FullAtlasScorer**

```python
"""
full_atlas_scorer.py -- Score all atlas anchors in a single batched forward pass.

Replaces the AnchorRetriever + AnchorSelector for the full-atlas query path.
Instead of BM25+dense retrieval (256->24) followed by cross-encoder selection
(24->k), this scores all 256 anchors directly.

Usage::

    scorer = FullAtlasScorer.from_model_path(model_path, atlas)
    scores = scorer.score_all("What auth pattern did we use?")
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger("psa.full_atlas_scorer")

TRAINED_MAX_SEQ = 320


def _load_cross_encoder(model_path: str):
    """Load a fine-tuned cross-encoder from disk."""
    try:
        from sentence_transformers.cross_encoder import CrossEncoder

        return CrossEncoder(model_path, max_length=TRAINED_MAX_SEQ)
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for FullAtlasScorer. "
            "Install with: pip install 'psa[training]'"
        )


@dataclass
class AnchorScore:
    """Score for a single anchor from the FullAtlasScorer."""

    anchor_id: int
    ce_score: float
    centroid: np.ndarray


class FullAtlasScorer:
    """
    Batch cross-encoder scoring over all atlas anchors.

    Scores every (query, anchor_card_text) pair in one forward pass,
    eliminating the retriever bottleneck that loses 26% of gold anchors
    at top-24.
    """

    def __init__(self, cross_encoder, atlas):
        self._cross_encoder = cross_encoder
        self._atlas = atlas
        self._card_texts = [c.to_stable_card_text() for c in atlas.cards]
        self._centroids = np.array(
            [c.centroid for c in atlas.cards], dtype=np.float32
        )
        self._anchor_ids = [c.anchor_id for c in atlas.cards]

    def score_all(
        self,
        query: str,
        query_vec: Optional[np.ndarray] = None,
    ) -> List[AnchorScore]:
        """
        Score all anchors in the atlas for a query.

        Returns List of AnchorScore sorted by ce_score descending.
        """
        n = len(self._anchor_ids)

        if self._cross_encoder is not None:
            pairs = [(query, self._card_texts[i]) for i in range(n)]
            try:
                raw_scores = self._cross_encoder.predict(pairs)
            except Exception as e:
                logger.warning("Cross-encoder batch failed (%s); cosine fallback", e)
                return self._cosine_fallback(query_vec)
        elif query_vec is not None:
            return self._cosine_fallback(query_vec)
        else:
            raise ValueError(
                "FullAtlasScorer has no cross-encoder and no query_vec -- "
                "cannot score anchors."
            )

        scores = [
            AnchorScore(
                anchor_id=self._anchor_ids[i],
                ce_score=float(raw_scores[i]),
                centroid=self._centroids[i],
            )
            for i in range(n)
        ]
        scores.sort(key=lambda s: s.ce_score, reverse=True)
        return scores

    def _cosine_fallback(self, query_vec: np.ndarray) -> List[AnchorScore]:
        """Score all anchors by cosine similarity to query_vec."""
        qv = np.array(query_vec, dtype=np.float32).reshape(1, -1)
        sims = (self._centroids @ qv.T).flatten()
        scores = [
            AnchorScore(
                anchor_id=self._anchor_ids[i],
                ce_score=float(sims[i]),
                centroid=self._centroids[i],
            )
            for i in range(len(self._anchor_ids))
        ]
        scores.sort(key=lambda s: s.ce_score, reverse=True)
        return scores

    @classmethod
    def from_model_path(cls, model_path: str, atlas) -> "FullAtlasScorer":
        """Load a FullAtlasScorer from a saved cross-encoder model."""
        ce = _load_cross_encoder(model_path)
        return cls(cross_encoder=ce, atlas=atlas)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_full_atlas_scorer.py -v`
Expected: All 5 tests PASS

- [ ] **Step 3: Commit**

```bash
git add psa/full_atlas_scorer.py tests/test_full_atlas_scorer.py
git commit -m "feat: add FullAtlasScorer -- batched cross-encoder over all 256 anchors"
```

---

### Task 3: CoActivationModel -- Tests

**Files:**
- Create: `tests/test_coactivation.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for psa.coactivation -- co-activation transformer + adaptive threshold."""

import numpy as np
import pytest
import torch


class TestCoActivationModel:
    def test_forward_output_shape(self):
        """Model outputs (batch, n_anchors) refined scores + (batch,) threshold."""
        from psa.coactivation import CoActivationModel

        model = CoActivationModel(n_anchors=256, centroid_dim=768)
        model.eval()

        batch = 4
        anchor_scores = torch.rand(batch, 256)
        centroids = torch.randn(batch, 256, 768)
        query_vecs = torch.randn(batch, 768)

        refined_scores, thresholds = model(anchor_scores, centroids, query_vecs)

        assert refined_scores.shape == (batch, 256)
        assert thresholds.shape == (batch,)

    def test_forward_scores_in_zero_one(self):
        """Refined scores should be in [0, 1] (sigmoid output)."""
        from psa.coactivation import CoActivationModel

        model = CoActivationModel(n_anchors=256, centroid_dim=768)
        model.eval()

        anchor_scores = torch.rand(1, 256)
        centroids = torch.randn(1, 256, 768)
        query_vecs = torch.randn(1, 768)

        with torch.no_grad():
            refined, _ = model(anchor_scores, centroids, query_vecs)

        assert (refined >= 0).all()
        assert (refined <= 1).all()

    def test_threshold_positive(self):
        """Adaptive threshold should be in (0, 1)."""
        from psa.coactivation import CoActivationModel

        model = CoActivationModel(n_anchors=256, centroid_dim=768)
        model.eval()

        with torch.no_grad():
            _, thresholds = model(
                torch.rand(2, 256),
                torch.randn(2, 256, 768),
                torch.randn(2, 768),
            )

        assert (thresholds > 0).all()
        assert (thresholds < 1).all()

    def test_permutation_equivariant(self):
        """Permuting anchor order should permute outputs identically."""
        from psa.coactivation import CoActivationModel

        torch.manual_seed(42)
        model = CoActivationModel(n_anchors=8, centroid_dim=768)
        model.eval()

        anchor_scores = torch.rand(1, 8)
        centroids = torch.randn(1, 8, 768)
        query_vecs = torch.randn(1, 768)

        with torch.no_grad():
            out_orig, tau_orig = model(anchor_scores, centroids, query_vecs)

        perm = torch.tensor([3, 1, 7, 0, 5, 2, 6, 4])
        with torch.no_grad():
            out_perm, tau_perm = model(
                anchor_scores[:, perm],
                centroids[:, perm, :],
                query_vecs,
            )

        torch.testing.assert_close(out_orig[:, perm], out_perm, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(tau_orig, tau_perm, atol=1e-5, rtol=1e-5)

    def test_small_parameter_count(self):
        """Model should have roughly 2M parameters, not more."""
        from psa.coactivation import CoActivationModel

        model = CoActivationModel(n_anchors=256, centroid_dim=768)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 5_000_000, f"Too many params: {n_params}"


class TestCoActivationSelector:
    def test_select_returns_selected_anchors(self):
        """select() returns a list of SelectedAnchor with variable k."""
        from psa.coactivation import CoActivationSelector, CoActivationModel
        from psa.full_atlas_scorer import AnchorScore
        from unittest.mock import MagicMock

        model = MagicMock(spec=CoActivationModel)
        scores_out = torch.zeros(1, 8)
        scores_out[0, :3] = 0.8
        thresholds_out = torch.tensor([0.5])
        model.return_value = (scores_out, thresholds_out)
        model.eval = MagicMock(return_value=model)

        selector = CoActivationSelector(model=model, device="cpu")

        anchor_scores = [
            AnchorScore(anchor_id=i, ce_score=0.5, centroid=np.zeros(768))
            for i in range(8)
        ]

        selected = selector.select(np.zeros(768), anchor_scores)

        assert len(selected) == 3
        assert all(s.anchor_id in (0, 1, 2) for s in selected)
        assert all(s.mode == "coactivation" for s in selected)

    def test_select_minimum_one_anchor(self):
        """Always returns at least 1 anchor even if all scores below threshold."""
        from psa.coactivation import CoActivationSelector, CoActivationModel
        from psa.full_atlas_scorer import AnchorScore
        from unittest.mock import MagicMock

        model = MagicMock(spec=CoActivationModel)
        scores_out = torch.full((1, 4), 0.1)
        thresholds_out = torch.tensor([0.5])
        model.return_value = (scores_out, thresholds_out)
        model.eval = MagicMock(return_value=model)

        selector = CoActivationSelector(model=model, device="cpu")
        anchor_scores = [
            AnchorScore(anchor_id=i, ce_score=0.3, centroid=np.zeros(768))
            for i in range(4)
        ]

        selected = selector.select(np.zeros(768), anchor_scores)
        assert len(selected) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_coactivation.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'psa.coactivation'`

- [ ] **Step 3: Commit**

```bash
git add tests/test_coactivation.py
git commit -m "test: add failing tests for CoActivationModel and CoActivationSelector"
```

---

### Task 4: CoActivationModel -- Implementation

**Files:**
- Create: `psa/coactivation.py`

- [ ] **Step 1: Implement CoActivationModel and CoActivationSelector**

```python
"""
coactivation.py -- Inter-anchor co-activation transformer with adaptive threshold.

Takes 256 cross-encoder scores + atlas centroids + query embedding and learns
which anchors activate together. Outputs refined scores and a per-query
adaptive threshold that determines how many anchors to select.

The model is permutation-equivariant: it uses atlas centroids (not learned
per-anchor embeddings) so it survives atlas rebuilds without architecture
changes.

Usage::

    selector = CoActivationSelector.from_model_path(path, device="mps")
    selected = selector.select(query_vec, anchor_scores)
"""

import logging
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .full_atlas_scorer import AnchorScore
from .selector import SelectedAnchor

logger = logging.getLogger("psa.coactivation")


class CoActivationModel(nn.Module):
    """
    Transformer that refines per-anchor cross-encoder scores by attending
    across all anchors in the atlas.

    Input per anchor token:  [ce_score (1), centroid (768)] -> projected to d_model
    Query conditioning:      query_vec (768) -> projected, concatenated with each token
    Architecture:            2-layer transformer encoder with self-attention
    Output:                  refined score per anchor (sigmoid) + adaptive threshold (sigmoid)
    """

    def __init__(
        self,
        n_anchors: int = 256,
        centroid_dim: int = 768,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_anchors = n_anchors
        self.d_model = d_model

        self.anchor_proj = nn.Linear(1 + centroid_dim, d_model // 2)
        self.query_proj = nn.Linear(centroid_dim, d_model // 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.score_head = nn.Linear(d_model, 1)

        self.threshold_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.threshold_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        anchor_scores: torch.Tensor,
        centroids: torch.Tensor,
        query_vecs: torch.Tensor,
    ):
        """
        Parameters
        ----------
        anchor_scores: (B, N) cross-encoder scores
        centroids: (B, N, 768) atlas centroids
        query_vecs: (B, 768) query embeddings

        Returns
        -------
        refined_scores: (B, N) in [0, 1]
        thresholds: (B,) in (0, 1)
        """
        B, N = anchor_scores.shape

        score_feat = anchor_scores.unsqueeze(-1)
        anchor_input = torch.cat([score_feat, centroids], dim=-1)
        anchor_tokens = self.anchor_proj(anchor_input)

        query_tokens = self.query_proj(query_vecs)
        query_tokens = query_tokens.unsqueeze(1).expand(B, N, -1)

        tokens = torch.cat([anchor_tokens, query_tokens], dim=-1)

        refined = self.transformer(tokens)

        refined_scores = torch.sigmoid(self.score_head(refined).squeeze(-1))

        tq = self.threshold_query.expand(B, -1, -1)
        attn_weights = torch.bmm(tq, refined.transpose(1, 2))
        attn_weights = torch.softmax(attn_weights / (self.d_model ** 0.5), dim=-1)
        pooled = torch.bmm(attn_weights, refined).squeeze(1)
        thresholds = torch.sigmoid(self.threshold_head(pooled).squeeze(-1))

        return refined_scores, thresholds


class CoActivationSelector:
    """
    Inference wrapper around CoActivationModel.

    Takes AnchorScore list from FullAtlasScorer + query_vec, runs the
    co-activation model, applies adaptive threshold, returns SelectedAnchors.
    """

    def __init__(self, model: CoActivationModel, device: str = "cpu"):
        self._model = model
        self._device = device
        if isinstance(model, nn.Module):
            self._model.to(device)
            self._model.eval()

    def select(
        self,
        query_vec: np.ndarray,
        anchor_scores: List[AnchorScore],
    ) -> List[SelectedAnchor]:
        """
        Run co-activation selection on scored anchors.

        Returns List of SelectedAnchor (variable length, min 1).
        """
        n = len(anchor_scores)

        ce_scores = torch.tensor(
            [s.ce_score for s in anchor_scores], dtype=torch.float32
        ).unsqueeze(0)
        centroids = torch.tensor(
            np.stack([s.centroid for s in anchor_scores]),
            dtype=torch.float32,
        ).unsqueeze(0)
        qv = torch.tensor(
            query_vec, dtype=torch.float32
        ).unsqueeze(0)

        ce_scores = ce_scores.to(self._device)
        centroids = centroids.to(self._device)
        qv = qv.to(self._device)

        with torch.no_grad():
            refined, threshold = self._model(ce_scores, centroids, qv)

        refined_np = refined[0].cpu().numpy()
        tau = float(threshold[0].cpu())

        selected = []
        for i in range(n):
            if refined_np[i] >= tau:
                selected.append(
                    SelectedAnchor(
                        anchor_id=anchor_scores[i].anchor_id,
                        selector_score=float(refined_np[i]),
                        mode="coactivation",
                        candidate=None,
                    )
                )

        if not selected:
            best_idx = int(np.argmax(refined_np))
            selected = [
                SelectedAnchor(
                    anchor_id=anchor_scores[best_idx].anchor_id,
                    selector_score=float(refined_np[best_idx]),
                    mode="coactivation",
                    candidate=None,
                )
            ]

        selected.sort(key=lambda s: s.selector_score, reverse=True)

        logger.debug(
            "CoActivation selected %d anchors (tau=%.3f, top_score=%.3f)",
            len(selected), tau, selected[0].selector_score if selected else 0,
        )
        return selected

    @classmethod
    def from_model_path(
        cls, model_path: str, device: str = "cpu"
    ) -> "CoActivationSelector":
        """Load a trained CoActivationModel from disk."""
        import json

        meta_path = os.path.join(model_path, "coactivation_version.json")
        with open(meta_path) as f:
            meta = json.load(f)

        model = CoActivationModel(
            n_anchors=meta.get("n_anchors", 256),
            centroid_dim=meta.get("centroid_dim", 768),
        )
        weights_path = os.path.join(model_path, "coactivation_model.pt")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        return cls(model=model, device=device)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_coactivation.py -v`
Expected: All 7 tests PASS

- [ ] **Step 3: Commit**

```bash
git add psa/coactivation.py tests/test_coactivation.py
git commit -m "feat: add CoActivationModel and CoActivationSelector"
```

---

### Task 5: Pipeline Integration

**Files:**
- Modify: `psa/pipeline.py`

- [ ] **Step 1: Add imports at top of pipeline.py**

After the existing imports (`from .synthesizer import AnchorSynthesizer`), add:

```python
from .full_atlas_scorer import AnchorScore, FullAtlasScorer
from .coactivation import CoActivationSelector
```

- [ ] **Step 2: Modify PSAPipeline.__init__ to accept new components**

Replace the current `__init__` with:

```python
    def __init__(
        self,
        store: MemoryStore,
        atlas: Atlas,
        embedding_model: EmbeddingModel,
        selector: Optional[AnchorSelector] = None,
        full_atlas_scorer: Optional[FullAtlasScorer] = None,
        coactivation_selector: Optional[CoActivationSelector] = None,
        token_budget: int = 6000,
        tenant_id: str = "default",
        psa_mode: str = "side-by-side",
    ):
        self.store = store
        self.atlas = atlas
        self.embedding_model = embedding_model
        self.selector = selector or AnchorSelector.cosine()
        self.full_atlas_scorer = full_atlas_scorer
        self.coactivation_selector = coactivation_selector
        self.token_budget = token_budget
        self.tenant_id = tenant_id
        self.psa_mode = psa_mode

        self._retriever = AnchorRetriever(atlas)
        self._packer = EvidencePacker(memory_store=store)
        self._synthesizer = AnchorSynthesizer()
```

- [ ] **Step 3: Add selection_mode to PSAResult**

Add `selection_mode: str = "legacy"` field to the PSAResult dataclass. Add `"selection_mode": self.selection_mode` to the `to_dict()` method.

- [ ] **Step 4: Replace retrieve+select in query() with 3-level branching**

In `query()`, after the embed step (Step 1), replace Steps 2 and 3 (retrieve + select) with:

```python
        # Step 2+3: Score and select anchors
        # Three-level graceful degradation:
        #   1. CoActivation (full_atlas_scorer + coactivation_selector)
        #   2. FullAtlas (full_atlas_scorer + top-k)
        #   3. Legacy (retriever + selector)
        selection_mode = "legacy"
        candidates = []

        if self.full_atlas_scorer is not None:
            t0 = time.perf_counter()
            anchor_scores = self.full_atlas_scorer.score_all(
                query, query_vec=query_vec,
            )
            timing.retrieve_ms = (time.perf_counter() - t0) * 1000
            logger.debug(
                "FullAtlas scored %d anchors in %.1fms",
                len(anchor_scores), timing.retrieve_ms,
            )

            t0 = time.perf_counter()
            if self.coactivation_selector is not None:
                selected = self.coactivation_selector.select(
                    query_vec=query_vec,
                    anchor_scores=anchor_scores,
                )
                selection_mode = "coactivation"
            else:
                top_k = self.selector.max_k
                selected = [
                    SelectedAnchor(
                        anchor_id=s.anchor_id,
                        selector_score=s.ce_score,
                        mode="full_atlas",
                        candidate=None,
                    )
                    for s in anchor_scores[:top_k]
                ]
                selection_mode = "full_atlas"
            timing.select_ms = (time.perf_counter() - t0) * 1000
        else:
            t0 = time.perf_counter()
            candidates = self._retriever.retrieve(
                query=query,
                embedding_model=self.embedding_model,
                top_k=top_k_candidates,
                query_vec=query_vec,
            )
            timing.retrieve_ms = (time.perf_counter() - t0) * 1000

            if not candidates:
                packed = PackedContext(
                    query=query,
                    text="(no anchor candidates found -- atlas may be empty or unbuilt)",
                    token_count=0,
                    memory_ids=[],
                    sections=[],
                    untyped_count=0,
                )
                return PSAResult(
                    query=query,
                    packed_context=packed,
                    selected_anchors=[],
                    candidates=[],
                    timing=timing,
                    tenant_id=self.tenant_id,
                    psa_mode=self.psa_mode,
                    selection_mode="legacy",
                )

            t0 = time.perf_counter()
            selected = self.selector.select(
                query=query,
                candidates=candidates,
                query_vec=query_vec,
            )
            timing.select_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            "Selected %d anchors in %.1fms (%s)",
            len(selected), timing.select_ms, selection_mode,
        )
```

The rest of `query()` (fingerprint, empty check, fetch, synthesize, return) stays unchanged. Just add `selection_mode=selection_mode` to the PSAResult at the end.

- [ ] **Step 5: Add co-activation loading to from_tenant()**

In `from_tenant()`, just before the `return cls(...)` at the end, add:

```python
        full_atlas_scorer = None
        coactivation_selector = None

        if selector_model_path and os.path.exists(selector_model_path):
            try:
                full_atlas_scorer = FullAtlasScorer.from_model_path(
                    selector_model_path, atlas
                )
            except Exception:
                logger.debug("FullAtlasScorer loading failed", exc_info=True)

        coact_path = os.path.join(tenant.root_dir, "models", "coactivation_latest")
        coact_meta = os.path.join(coact_path, "coactivation_version.json")
        if os.path.exists(coact_meta):
            try:
                import torch as _torch

                _dev = "mps" if _torch.backends.mps.is_available() else "cpu"
                coactivation_selector = CoActivationSelector.from_model_path(
                    coact_path, device=_dev
                )
            except Exception:
                logger.debug("CoActivationSelector loading failed", exc_info=True)
```

And update the `return cls(...)` to pass `full_atlas_scorer=full_atlas_scorer, coactivation_selector=coactivation_selector`.

- [ ] **Step 6: Run existing tests**

Run: `uv run pytest tests/ -v -k "pipeline" --timeout=60`
Expected: All existing pipeline tests PASS (graceful fallback to legacy path)

- [ ] **Step 7: Commit**

```bash
git add psa/pipeline.py
git commit -m "feat: integrate FullAtlasScorer + CoActivationSelector into pipeline with 3-level fallback"
```

---

### Task 6: Co-Activation Training Data Generator

**Files:**
- Create: `psa/training/coactivation_data.py`

- [ ] **Step 1: Implement the training data generator**

```python
"""
coactivation_data.py -- Generate training data for the co-activation model.

For each oracle-labeled query, runs FullAtlasScorer to get all 256 anchor
scores, pairs them with gold anchor IDs, and saves as a compressed npz file.
"""

import json
import logging
import os
from typing import List

import numpy as np

logger = logging.getLogger("psa.training.coactivation_data")


def generate_coactivation_data(
    oracle_labels_path: str,
    output_path: str,
    full_atlas_scorer,
    embedding_model,
    atlas,
) -> int:
    """
    Generate co-activation training data from oracle labels.

    For each oracle label:
    1. Embed the query
    2. Score all 256 anchors via FullAtlasScorer
    3. Collect (query_vec, scores, gold_mask, gold_k)

    Saves a single coactivation_train.npz with all examples.

    Returns number of examples written.
    """
    os.makedirs(output_path, exist_ok=True)

    labels = []
    with open(oracle_labels_path) as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(json.loads(line))

    anchor_ids = [c.anchor_id for c in atlas.cards]
    n_anchors = len(anchor_ids)

    all_query_vecs = []
    all_ce_scores = []
    all_gold_masks = []
    all_gold_ks = []

    for i, label in enumerate(labels):
        query = label["query"]
        gold_set = set(label.get("winning_oracle_set", []))
        if not gold_set:
            continue

        query_vec = embedding_model.embed(query)
        anchor_scores = full_atlas_scorer.score_all(query, query_vec=query_vec)

        score_by_id = {s.anchor_id: s.ce_score for s in anchor_scores}
        ce_scores = np.array(
            [score_by_id.get(aid, 0.0) for aid in anchor_ids],
            dtype=np.float32,
        )

        gold_mask = np.array(
            [1.0 if aid in gold_set else 0.0 for aid in anchor_ids],
            dtype=np.float32,
        )

        all_query_vecs.append(np.array(query_vec, dtype=np.float32))
        all_ce_scores.append(ce_scores)
        all_gold_masks.append(gold_mask)
        all_gold_ks.append(int(gold_mask.sum()))

        if (i + 1) % 50 == 0:
            logger.info("  %d / %d examples generated", i + 1, len(labels))

    written = len(all_query_vecs)
    if written > 0:
        centroids = np.array(
            [c.centroid for c in atlas.cards], dtype=np.float32
        )
        np.savez_compressed(
            os.path.join(output_path, "coactivation_train.npz"),
            query_vecs=np.stack(all_query_vecs),
            ce_scores=np.stack(all_ce_scores),
            gold_masks=np.stack(all_gold_masks),
            gold_ks=np.array(all_gold_ks, dtype=np.int32),
            centroids=centroids,
            anchor_ids=np.array(anchor_ids, dtype=np.int32),
        )

    logger.info("Wrote %d co-activation training examples to %s", written, output_path)
    return written
```

- [ ] **Step 2: Run lint**

Run: `uv run ruff check psa/training/coactivation_data.py`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add psa/training/coactivation_data.py
git commit -m "feat: add co-activation training data generator"
```

---

### Task 7: Co-Activation Training Loop

**Files:**
- Create: `psa/training/train_coactivation.py`
- Create: `tests/test_train_coactivation.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for psa.training.train_coactivation."""

import numpy as np
import os
import pytest
import tempfile


class TestCoActivationTrainer:
    def test_train_produces_model(self):
        """Training on synthetic data produces a saved model."""
        from psa.training.train_coactivation import CoActivationTrainer

        n_examples, n_anchors, dim = 32, 16, 768
        rng = np.random.default_rng(42)

        gold_masks = np.zeros((n_examples, n_anchors), dtype=np.float32)
        for i in range(n_examples):
            k = rng.integers(1, 4)
            gold_masks[i, :k] = 1.0

        data = {
            "query_vecs": rng.standard_normal((n_examples, dim)).astype(np.float32),
            "ce_scores": rng.random((n_examples, n_anchors)).astype(np.float32),
            "gold_masks": gold_masks,
            "gold_ks": gold_masks.sum(axis=1).astype(np.int32),
            "centroids": rng.standard_normal((n_anchors, dim)).astype(np.float32),
            "anchor_ids": np.arange(n_anchors, dtype=np.int32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez_compressed(os.path.join(tmpdir, "coactivation_train.npz"), **data)

            output_dir = os.path.join(tmpdir, "model")
            trainer = CoActivationTrainer(output_dir=output_dir)
            trainer.train(data_dir=tmpdir, n_anchors=n_anchors, epochs=2, batch_size=8)

            assert os.path.exists(os.path.join(output_dir, "coactivation_model.pt"))
            assert os.path.exists(os.path.join(output_dir, "coactivation_version.json"))

    def test_train_loss_decreases(self):
        """Training loss should decrease over epochs."""
        from psa.training.train_coactivation import CoActivationTrainer

        n_examples, n_anchors, dim = 64, 16, 768
        rng = np.random.default_rng(42)

        gold_masks = np.zeros((n_examples, n_anchors), dtype=np.float32)
        for i in range(n_examples):
            k = rng.integers(1, 4)
            gold_masks[i, :k] = 1.0

        data = {
            "query_vecs": rng.standard_normal((n_examples, dim)).astype(np.float32),
            "ce_scores": rng.random((n_examples, n_anchors)).astype(np.float32),
            "gold_masks": gold_masks,
            "gold_ks": gold_masks.sum(axis=1).astype(np.int32),
            "centroids": rng.standard_normal((n_anchors, dim)).astype(np.float32),
            "anchor_ids": np.arange(n_anchors, dtype=np.int32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez_compressed(os.path.join(tmpdir, "coactivation_train.npz"), **data)

            output_dir = os.path.join(tmpdir, "model")
            trainer = CoActivationTrainer(output_dir=output_dir)
            losses = trainer.train(
                data_dir=tmpdir, n_anchors=n_anchors, epochs=5, batch_size=16,
                return_losses=True,
            )

            assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
```

- [ ] **Step 2: Implement the training loop**

```python
"""
train_coactivation.py -- Training loop for the co-activation model.

Trains CoActivationModel on expanded oracle labels (256-anchor scores +
gold masks). Frozen cross-encoder provides per-anchor scores; this module
trains only the co-activation transformer on top.

Usage::

    trainer = CoActivationTrainer(output_dir="~/.psa/models/coactivation_v1")
    trainer.train(data_dir="~/.psa/tenants/default/training/coactivation")
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..coactivation import CoActivationModel

logger = logging.getLogger("psa.training.train_coactivation")

DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 10
DEFAULT_BATCH = 16
THRESHOLD_LOSS_WEIGHT = 0.3


class CoActivationTrainer:
    """Train a CoActivationModel from expanded oracle label data."""

    def __init__(
        self,
        output_dir: str,
        learning_rate: float = DEFAULT_LR,
    ):
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        os.makedirs(output_dir, exist_ok=True)

    def train(
        self,
        data_dir: str,
        n_anchors: int = 256,
        centroid_dim: int = 768,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH,
        val_split: float = 0.15,
        return_losses: bool = False,
    ) -> Optional[List[float]]:
        """
        Train the co-activation model.

        Returns list of per-epoch losses if return_losses=True, else None.
        """
        data_path = os.path.join(data_dir, "coactivation_train.npz")
        npz = np.load(data_path)
        query_vecs = torch.tensor(npz["query_vecs"], dtype=torch.float32)
        ce_scores = torch.tensor(npz["ce_scores"], dtype=torch.float32)
        gold_masks = torch.tensor(npz["gold_masks"], dtype=torch.float32)
        gold_ks = torch.tensor(npz["gold_ks"], dtype=torch.float32)
        centroids = torch.tensor(npz["centroids"], dtype=torch.float32)

        N = query_vecs.shape[0]
        actual_n_anchors = ce_scores.shape[1]

        centroids_batch = centroids.unsqueeze(0).expand(N, -1, -1)

        n_val = max(1, int(N * val_split))
        n_train = N - n_val
        indices = torch.randperm(N, generator=torch.Generator().manual_seed(42))
        train_idx = indices[:n_train]

        train_ds = TensorDataset(
            ce_scores[train_idx],
            centroids_batch[train_idx],
            query_vecs[train_idx],
            gold_masks[train_idx],
            gold_ks[train_idx],
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"

        model = CoActivationModel(
            n_anchors=actual_n_anchors,
            centroid_dim=centroid_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        epoch_losses = []

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0

            for batch_ce, batch_cent, batch_qv, batch_gold, batch_gk in train_loader:
                batch_ce = batch_ce.to(device)
                batch_cent = batch_cent.to(device)
                batch_qv = batch_qv.to(device)
                batch_gold = batch_gold.to(device)
                batch_gk = batch_gk.to(device)

                refined_scores, thresholds = model(batch_ce, batch_cent, batch_qv)

                score_loss = bce_loss(refined_scores, batch_gold)
                gold_k_norm = batch_gk / actual_n_anchors
                threshold_loss = mse_loss(thresholds, gold_k_norm)

                loss = score_loss + THRESHOLD_LOSS_WEIGHT * threshold_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)
            logger.info("Epoch %d/%d: loss=%.4f", epoch + 1, epochs, avg_loss)

        # Save model
        model_cpu = model.cpu()
        torch.save(
            model_cpu.state_dict(),
            os.path.join(self.output_dir, "coactivation_model.pt"),
        )

        meta = {
            "n_anchors": actual_n_anchors,
            "centroid_dim": centroid_dim,
            "training_examples": n_train,
            "epochs": epochs,
            "final_loss": epoch_losses[-1] if epoch_losses else None,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(self.output_dir, "coactivation_version.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("CoActivation model saved to %s", self.output_dir)

        if return_losses:
            return epoch_losses
        return None
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_train_coactivation.py -v`
Expected: Both tests PASS

- [ ] **Step 4: Commit**

```bash
git add psa/training/train_coactivation.py tests/test_train_coactivation.py
git commit -m "feat: add co-activation training loop"
```

---

### Task 8: CLI Integration

**Files:**
- Modify: `psa/cli.py`

- [ ] **Step 1: Add --coactivation flag to psa train argparse**

After `p_train.add_argument("--force", ...)`, add:

```python
    p_train.add_argument(
        "--coactivation",
        action="store_true",
        help="Also train co-activation model after selector (requires trained selector)",
    )
```

- [ ] **Step 2: Add co-activation training to cmd_train()**

In `cmd_train()`, after `print("  Activated trained selector.")`, before the except block, add:

```python
        # Co-activation training
        if getattr(args, "coactivation", False):
            print("Training co-activation model...")
            from .embeddings import EmbeddingModel
            from .full_atlas_scorer import FullAtlasScorer
            from .training.coactivation_data import generate_coactivation_data
            from .training.train_coactivation import CoActivationTrainer

            fas = FullAtlasScorer.from_model_path(sv.model_path, atlas)
            emb = EmbeddingModel()
            coact_data_dir = os.path.join(tenant.root_dir, "training", "coactivation")
            n_coact = generate_coactivation_data(
                oracle_labels_path=labels_path,
                output_path=coact_data_dir,
                full_atlas_scorer=fas,
                embedding_model=emb,
                atlas=atlas,
            )
            print(f"  Generated {n_coact} co-activation training examples.")

            coact_output = os.path.join(tenant.root_dir, "models", "coactivation_latest")
            coact_trainer = CoActivationTrainer(output_dir=coact_output)
            coact_trainer.train(
                data_dir=coact_data_dir,
                n_anchors=len(atlas.cards),
            )
            print(f"  Co-activation model saved to {coact_output}")
```

- [ ] **Step 3: Add --selector coactivation to benchmark**

Change the `--selector` choices in the `p_lme_run` parser:

```python
    p_lme_run.add_argument(
        "--selector",
        default="cosine",
        choices=["cosine", "trained", "coactivation"],
        help="Selector mode: cosine, trained, or coactivation",
    )
```

In the benchmark run handler, after pipeline construction, add handling for coactivation mode:

```python
        if selector_mode == "coactivation":
            from .full_atlas_scorer import FullAtlasScorer
            from .coactivation import CoActivationSelector

            model_dir = os.path.join(
                os.path.expanduser(f"~/.psa/tenants/{tenant_id}/models/selector_latest"),
                "selector_v1",
            )
            if os.path.exists(model_dir):
                pipeline.full_atlas_scorer = FullAtlasScorer.from_model_path(
                    model_dir, pipeline.atlas
                )
            coact_path = os.path.join(
                os.path.expanduser(f"~/.psa/tenants/{tenant_id}/models"),
                "coactivation_latest",
            )
            coact_meta = os.path.join(coact_path, "coactivation_version.json")
            if os.path.exists(coact_meta):
                import torch
                _dev = "mps" if torch.backends.mps.is_available() else "cpu"
                pipeline.coactivation_selector = CoActivationSelector.from_model_path(
                    coact_path, device=_dev,
                )
```

- [ ] **Step 4: Run lint and existing tests**

Run: `uv run ruff check psa/cli.py && uv run pytest tests/ -v -k "cli" --timeout=60`
Expected: No lint errors, existing tests PASS

- [ ] **Step 5: Commit**

```bash
git add psa/cli.py
git commit -m "feat: add --coactivation to psa train and --selector coactivation to benchmark"
```

---

### Task 9: Lifecycle Integration

**Files:**
- Modify: `psa/lifecycle.py`

- [ ] **Step 1: Add co-activation training after selector retraining**

In the slow-path section, after `summary["selector_retrained"] = retrained` and `print("  Selector retrained ...")`, add:

```python
            if retrained:
                try:
                    from .full_atlas_scorer import FullAtlasScorer
                    from .training.coactivation_data import generate_coactivation_data
                    from .training.train_coactivation import CoActivationTrainer
                    from .embeddings import EmbeddingModel

                    labels_path = os.path.join(
                        tenant.root_dir, "training", "oracle_labels.jsonl"
                    )
                    selector_path = state.get("selector_model_path")
                    if selector_path and os.path.exists(selector_path):
                        fas = FullAtlasScorer.from_model_path(selector_path, atlas)
                        emb = EmbeddingModel()
                        coact_data_dir = os.path.join(
                            tenant.root_dir, "training", "coactivation"
                        )
                        generate_coactivation_data(
                            oracle_labels_path=labels_path,
                            output_path=coact_data_dir,
                            full_atlas_scorer=fas,
                            embedding_model=emb,
                            atlas=atlas,
                        )
                        coact_output = os.path.join(
                            tenant.root_dir, "models", "coactivation_latest"
                        )
                        CoActivationTrainer(output_dir=coact_output).train(
                            data_dir=coact_data_dir,
                            n_anchors=len(atlas.cards),
                        )
                        summary["coactivation_trained"] = True
                        print("        Co-activation model trained.")
                except Exception as e:
                    logger.warning("Co-activation training failed: %s", e)
                    summary["coactivation_trained"] = False
```

- [ ] **Step 2: Run lifecycle tests**

Run: `uv run pytest tests/ -v -k "lifecycle" --timeout=60`
Expected: Existing tests PASS

- [ ] **Step 3: Commit**

```bash
git add psa/lifecycle.py
git commit -m "feat: add co-activation training to lifecycle slow path"
```

---

### Task 10: End-to-End Benchmark

- [ ] **Step 1: Train co-activation on longmemeval**

```bash
uv run psa train --tenant longmemeval_bench --force --coactivation
```

Expected: Selector trains, then co-activation generates 500 examples and trains in 2-5 minutes.

- [ ] **Step 2: Run benchmark**

```bash
uv run psa benchmark longmemeval run --selector coactivation
```

- [ ] **Step 3: Score and compare**

```bash
uv run psa benchmark longmemeval score \
  --results $(ls -t ~/.psa/benchmarks/longmemeval/results_*.jsonl | head -1) \
  --method both
```

Target comparison:

| Config | R@5 | F1 | LLM-judge |
|--------|-----|-------|-----------|
| cosine_k6 (baseline) | 0.856 | 0.202 | 0.330 |
| trained_rerank_k6 | 0.894 | 0.172 | 0.352 |
| **coactivation (target)** | **>= 0.95** | **>= 0.20** | **>= 0.35** |

- [ ] **Step 4: Commit benchmark results**

```bash
git add -u
git commit -m "benchmark: first co-activation results on longmemeval"
```

---

### Task 11: Lint, Format, Full Test Suite

- [ ] **Step 1: Run ruff**

```bash
uv run ruff check .
uv run ruff format .
```

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest tests/ -v --timeout=120
```

All tests must pass.

- [ ] **Step 3: Final commit**

```bash
git add -u
git commit -m "chore: lint and format pass"
```

---

## Self-Review

**Spec coverage:**
- FullAtlasScorer: Tasks 1-2
- CoActivationModel + Selector: Tasks 3-4
- Pipeline integration with 3-level fallback: Task 5
- Training data generation: Task 6
- Training loop: Task 7
- CLI integration: Task 8
- Lifecycle integration: Task 9
- End-to-end benchmark + success criteria: Task 10
- All covered.

**Placeholder scan:** No TBDs, TODOs, or vague steps. Every code step has complete code.

**Type consistency:**
- `AnchorScore` defined in Task 2, used in Tasks 4, 5, 6 -- consistent
- `SelectedAnchor` from `psa/selector.py`, used with `candidate=None` in Tasks 4, 5 -- consistent
- `CoActivationModel` defined in Task 4, used in Tasks 7, 8, 9 -- consistent
- `selection_mode` field added to `PSAResult` in Task 5, used in benchmark -- consistent
