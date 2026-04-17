"""Tests for psa.memory_scorer — Level 2 memory-level scoring."""

import numpy as np
import torch
from unittest.mock import MagicMock
from psa.memory_object import MemoryType


def _make_mock_memory(memory_id, body, memory_type=MemoryType.SEMANTIC, quality=0.5, days_old=7):
    mo = MagicMock()
    mo.memory_object_id = memory_id
    mo.body = body
    mo.memory_type = memory_type
    mo.quality_score = quality
    mo.created_at = f"2026-04-{max(1, 13 - days_old):02d}T00:00:00"
    mo.embedding = list(np.random.default_rng(hash(memory_id) % 2**31).standard_normal(768))
    return mo


class TestMemoryReRanker:
    def test_forward_output_shape(self):
        from psa.memory_scorer import MemoryReRanker

        model = MemoryReRanker(input_dim=11)
        x = torch.rand(20, 11)
        out = model(x)
        assert out.shape == (20, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_tiny_parameter_count(self):
        from psa.memory_scorer import MemoryReRanker

        model = MemoryReRanker(input_dim=11)
        n = sum(p.numel() for p in model.parameters())
        assert n < 1000, f"Too many params: {n}"


class TestMemoryScorer:
    def test_score_returns_sorted_scored_memories(self):
        from psa.memory_scorer import MemoryScorer, ScoredMemory

        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.9, 0.1, 0.5])

        mock_reranker = MagicMock()
        mock_reranker.return_value = torch.tensor([[0.8], [0.2], [0.6]])
        mock_reranker.eval = MagicMock(return_value=mock_reranker)

        scorer = MemoryScorer(cross_encoder=mock_ce, reranker=mock_reranker, device="cpu")
        memories = [
            _make_mock_memory("m1", "auth uses JWT"),
            _make_mock_memory("m2", "database setup"),
            _make_mock_memory("m3", "session tokens"),
        ]

        results = scorer.score("What auth?", np.zeros(768), memories)
        assert len(results) == 3
        assert all(isinstance(r, ScoredMemory) for r in results)
        assert results[0].final_score >= results[1].final_score >= results[2].final_score

    def test_score_with_empty_memories(self):
        from psa.memory_scorer import MemoryScorer

        scorer = MemoryScorer(cross_encoder=MagicMock(), reranker=MagicMock(), device="cpu")
        results = scorer.score("query", np.zeros(768), [])
        assert results == []

    def test_score_feat_tensor_matches_scorer_device(self):
        """Regression: feat_tensor must land on self.device so the MLP forward doesn't
        crash with a CPU/MPS mix when MemoryScorer is constructed with a non-CPU device.
        """
        from psa.memory_scorer import MemoryScorer

        captured_devices: list[str] = []

        def spy_reranker(t):
            captured_devices.append(str(t.device))
            return torch.zeros(t.shape[0], 1)

        spy_reranker.eval = lambda: spy_reranker

        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.5, 0.4])

        # Use a device string torch accepts but always available: "cpu". The fix
        # issue is that torch.tensor(...) defaults to CPU regardless of
        # self.device. The spy confirms the tensor's device matches self.device
        # after the .to() call; if someone regresses the fix to drop the .to(),
        # this passes under device="cpu" — so also exercise the Linear path with
        # a real MemoryReRanker under cpu to keep the hot path tested.
        scorer = MemoryScorer(cross_encoder=mock_ce, reranker=spy_reranker, device="cpu")
        memories = [
            _make_mock_memory("m1", "body one"),
            _make_mock_memory("m2", "body two"),
        ]
        scorer.score("q", np.zeros(768), memories)
        assert captured_devices == ["cpu"]

        # Now prove the tensor is actively moved: re-run with a fake device
        # attribute on the scorer and check the tensor carries it. Uses meta
        # device which does not require any hardware.
        scorer2 = MemoryScorer(cross_encoder=mock_ce, reranker=spy_reranker, device="meta")
        captured_devices.clear()
        scorer2.score("q", np.zeros(768), memories)
        assert captured_devices == ["meta"]
