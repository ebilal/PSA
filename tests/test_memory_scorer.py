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
