"""Tests for psa.coactivation — CoActivationModel and CoActivationSelector."""

from unittest.mock import MagicMock

import numpy as np
import torch

from psa.coactivation import CoActivationModel, CoActivationSelector
from psa.full_atlas_scorer import AnchorScore
from psa.selector import SelectedAnchor


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_anchor_scores(n: int, dim: int = 768) -> list:
    """Create a list of n AnchorScore instances with random centroids."""
    rng = np.random.default_rng(42)
    scores = []
    for i in range(n):
        centroid = rng.random(dim).astype(np.float32)
        centroid /= np.linalg.norm(centroid)
        scores.append(
            AnchorScore(
                anchor_id=i,
                ce_score=float(rng.random()),
                centroid=centroid,
            )
        )
    return scores


def _make_query_vec(dim: int = 768) -> np.ndarray:
    rng = np.random.default_rng(99)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ── TestCoActivationModel ─────────────────────────────────────────────────────


class TestCoActivationModel:
    def test_forward_output_shape(self):
        """(batch=4, 256 anchors): refined_scores shape (4, 256), thresholds shape (4,)."""
        batch, n_anchors, dim = 4, 256, 768
        model = CoActivationModel(n_anchors=n_anchors, centroid_dim=dim)
        model.eval()

        ce_scores = torch.rand(batch, n_anchors)
        centroids = torch.randn(batch, n_anchors, dim)
        query_vec = torch.randn(batch, dim)

        with torch.no_grad():
            refined_scores, thresholds = model(ce_scores, centroids, query_vec)

        assert refined_scores.shape == (batch, n_anchors)
        assert thresholds.shape == (batch,)

    def test_forward_scores_in_zero_one(self):
        """Sigmoid ensures refined_scores are in [0, 1]."""
        batch, n_anchors, dim = 2, 16, 768
        model = CoActivationModel(n_anchors=n_anchors, centroid_dim=dim)
        model.eval()

        ce_scores = torch.rand(batch, n_anchors)
        centroids = torch.randn(batch, n_anchors, dim)
        query_vec = torch.randn(batch, dim)

        with torch.no_grad():
            refined_scores, _ = model(ce_scores, centroids, query_vec)

        assert refined_scores.min().item() >= 0.0
        assert refined_scores.max().item() <= 1.0

    def test_threshold_positive(self):
        """Threshold values are in (0, 1)."""
        batch, n_anchors, dim = 3, 16, 768
        model = CoActivationModel(n_anchors=n_anchors, centroid_dim=dim)
        model.eval()

        ce_scores = torch.rand(batch, n_anchors)
        centroids = torch.randn(batch, n_anchors, dim)
        query_vec = torch.randn(batch, dim)

        with torch.no_grad():
            _, thresholds = model(ce_scores, centroids, query_vec)

        assert thresholds.min().item() > 0.0
        assert thresholds.max().item() < 1.0

    def test_permutation_equivariant(self):
        """Permuting anchor order permutes refined_scores identically."""
        batch, n_anchors, dim = 2, 8, 768
        model = CoActivationModel(n_anchors=n_anchors, centroid_dim=dim)
        model.eval()

        rng_gen = torch.Generator()
        rng_gen.manual_seed(7)
        ce_scores = torch.rand(batch, n_anchors, generator=rng_gen)
        centroids = torch.randn(batch, n_anchors, dim, generator=rng_gen)
        query_vec = torch.randn(batch, dim, generator=rng_gen)

        # Compute original output
        with torch.no_grad():
            scores_orig, _ = model(ce_scores, centroids, query_vec)

        # Permute anchor dimension
        perm_rng = torch.Generator()
        perm_rng.manual_seed(13)
        perm = torch.randperm(n_anchors, generator=perm_rng)
        ce_scores_perm = ce_scores[:, perm]
        centroids_perm = centroids[:, perm, :]

        with torch.no_grad():
            scores_perm, _ = model(ce_scores_perm, centroids_perm, query_vec)

        # scores_perm should equal scores_orig permuted the same way
        expected = scores_orig[:, perm]
        assert torch.allclose(scores_perm, expected, atol=1e-5), (
            f"Max diff: {(scores_perm - expected).abs().max().item()}"
        )

    def test_small_parameter_count(self):
        """Model has under 5M parameters for n_anchors=256."""
        model = CoActivationModel(n_anchors=256, centroid_dim=768)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params < 5_000_000, (
            f"Model has {total_params:,} parameters, expected < 5,000,000"
        )


# ── TestCoActivationSelector ──────────────────────────────────────────────────


class TestCoActivationSelector:
    def test_select_returns_selected_anchors(self):
        """High scores for anchors 0,1,2 and low for rest → selects exactly 3."""
        n_anchors = 10
        dim = 768

        def fake_forward(ce_scores, centroids, query_vec):
            batch = ce_scores.shape[0]
            n = ce_scores.shape[1]
            scores = torch.full((batch, n), 0.0)
            scores[:, :3] = 0.8
            threshold = torch.full((batch,), 0.5)
            return scores, threshold

        mock_model = MagicMock(spec=CoActivationModel)
        mock_model.side_effect = fake_forward

        selector = CoActivationSelector(model=mock_model, device="cpu")

        query_vec = _make_query_vec(dim)
        anchor_scores = _make_anchor_scores(n_anchors, dim)

        result = selector.select(query_vec=query_vec, anchor_scores=anchor_scores)

        assert len(result) == 3
        for sa in result:
            assert isinstance(sa, SelectedAnchor)
            assert sa.mode == "coactivation"
            assert sa.anchor_id in {0, 1, 2}
            assert sa.candidate is None

    def test_select_minimum_one_anchor(self):
        """All scores below threshold → still returns exactly 1 anchor."""
        n_anchors = 5
        dim = 768

        def fake_forward(ce_scores, centroids, query_vec):
            batch = ce_scores.shape[0]
            n = ce_scores.shape[1]
            scores = torch.full((batch, n), 0.1)
            threshold = torch.full((batch,), 0.9)
            return scores, threshold

        mock_model = MagicMock(spec=CoActivationModel)
        mock_model.side_effect = fake_forward

        selector = CoActivationSelector(model=mock_model, device="cpu")

        query_vec = _make_query_vec(dim)
        anchor_scores = _make_anchor_scores(n_anchors, dim)

        result = selector.select(query_vec=query_vec, anchor_scores=anchor_scores)

        assert len(result) == 1
        assert isinstance(result[0], SelectedAnchor)
        assert result[0].mode == "coactivation"
        assert result[0].candidate is None
