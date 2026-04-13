"""
memory_scorer.py — Level 2 memory-level scoring for PSA.

Two-stage re-ranking of individual MemoryObjects within selected anchors:

  Stage A: cross-encoder.predict on (query, memory.body) pairs
  Stage B: tiny MLP re-ranker on 11-dim feature vector

Feature vector (11 dims):
  [0]    ce_score           — Stage A cross-encoder score
  [1-6]  memory_type        — one-hot over MEMORY_TYPE_ORDER
  [7]    quality_score      — MemoryObject.quality_score
  [8]    body_token_count   — len(body)/4 / token_budget, capped at 1.0
  [9]    recency            — 1.0 / (1.0 + log1p(days_since_created))
  [10]   cosine_to_query    — dot product of L2-normed embeddings
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from math import log1p
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

from .memory_object import MemoryType

logger = logging.getLogger("psa.memory_scorer")

# ── Constants ─────────────────────────────────────────────────────────────────

MEMORY_TYPE_ORDER = [
    "EPISODIC",
    "SEMANTIC",
    "PROCEDURAL",
    "FAILURE",
    "TOOL_USE",
    "WORKING_DERIVATIVE",
]

_TYPE_INDEX = {name: i for i, name in enumerate(MEMORY_TYPE_ORDER)}

_FALLBACK_DAYS = 30.0


# ── ScoredMemory dataclass ────────────────────────────────────────────────────


@dataclass
class ScoredMemory:
    """A MemoryObject annotated with its final re-ranker score."""

    memory_object_id: str
    final_score: float
    memory: object  # MemoryObject


# ── MemoryReRanker MLP ────────────────────────────────────────────────────────


class MemoryReRanker(nn.Module):
    """
    Tiny MLP re-ranker: Linear(input_dim, 32) -> ReLU -> Linear(32, 1) -> Sigmoid.

    For input_dim=11: 11*32 + 32 + 32*1 + 1 = 417 parameters.
    """

    def __init__(self, input_dim: int = 11) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, input_dim) → (B, 1)
        return self.net(x)


# ── Feature helpers ───────────────────────────────────────────────────────────


def _days_since(created_at: str) -> float:
    """Return days between created_at ISO string and now (UTC). Falls back to 30."""
    try:
        dt = datetime.fromisoformat(created_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        return max(0.0, delta.total_seconds() / 86400.0)
    except Exception:
        return _FALLBACK_DAYS


def _recency(days: float) -> float:
    return 1.0 / (1.0 + log1p(days))


def _cosine_to_query(memory_embedding, query_vec: np.ndarray) -> float:
    """Dot product of L2-normalised embeddings."""
    mem_arr = np.asarray(memory_embedding, dtype=np.float32)
    q_arr = np.asarray(query_vec, dtype=np.float32)

    mem_norm = np.linalg.norm(mem_arr)
    q_norm = np.linalg.norm(q_arr)
    if mem_norm < 1e-9 or q_norm < 1e-9:
        return 0.0
    return float(np.dot(mem_arr / mem_norm, q_arr / q_norm))


def _type_onehot(memory_type: MemoryType) -> List[float]:
    """One-hot vector of length 6 over MEMORY_TYPE_ORDER."""
    idx = _TYPE_INDEX.get(memory_type.name, -1)
    vec = [0.0] * len(MEMORY_TYPE_ORDER)
    if idx >= 0:
        vec[idx] = 1.0
    return vec


def _build_feature_vector(
    ce_score: float,
    memory,
    query_vec: np.ndarray,
    token_budget: int,
) -> List[float]:
    """Build the 11-dim feature vector for a single memory."""
    onehot = _type_onehot(memory.memory_type)
    body_tok = min(1.0, (len(memory.body) / 4.0) / token_budget)
    days = _days_since(memory.created_at)
    rec = _recency(days)
    cos = _cosine_to_query(memory.embedding, query_vec)

    return [ce_score] + onehot + [memory.quality_score, body_tok, rec, cos]


# ── MemoryScorer ──────────────────────────────────────────────────────────────


class MemoryScorer:
    """
    Two-stage memory re-ranker.

    Parameters
    ----------
    cross_encoder:
        Any object with a ``predict(pairs) -> np.ndarray`` method
        (e.g. sentence_transformers CrossEncoder).
    reranker:
        A callable ``(tensor) -> tensor`` — typically a ``MemoryReRanker``
        instance.  Called in evaluation mode (model.eval()).
    device:
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(self, cross_encoder, reranker, device: str = "cpu") -> None:
        self.cross_encoder = cross_encoder
        self.reranker = reranker
        self.device = device

    # ── public API ──────────────────────────────────────────────────────────

    def score(
        self,
        query: str,
        query_vec: np.ndarray,
        memories: list,
        batch_size: int = 64,
        token_budget: int = 6000,
    ) -> List[ScoredMemory]:
        """
        Score *memories* against *query* and return them sorted by score desc.

        Parameters
        ----------
        query:       Raw query string.
        query_vec:   768-dim query embedding (L2-normalised).
        memories:    List of MemoryObject instances.
        batch_size:  Cross-encoder prediction batch size.
        token_budget: Used for body-length normalisation feature.
        """
        if not memories:
            return []

        # ── Stage A: cross-encoder ──────────────────────────────────────────
        pairs = [(query, m.body) for m in memories]
        ce_scores = self._predict_batched(pairs, batch_size)

        # ── Stage B: build feature matrix and run MLP ───────────────────────
        feature_rows = [
            _build_feature_vector(ce_scores[i], memories[i], query_vec, token_budget)
            for i in range(len(memories))
        ]
        feat_tensor = torch.tensor(feature_rows, dtype=torch.float32)

        # Switch reranker to inference mode (nn.Module.eval(), not Python eval)
        self.reranker.eval()
        with torch.no_grad():
            mlp_scores = self.reranker(feat_tensor)  # (N, 1)

        final_scores = mlp_scores.squeeze(1).cpu().numpy()  # (N,)

        # ── Assemble and sort ────────────────────────────────────────────────
        results = [
            ScoredMemory(
                memory_object_id=memories[i].memory_object_id,
                final_score=float(final_scores[i]),
                memory=memories[i],
            )
            for i in range(len(memories))
        ]
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results

    # ── internals ───────────────────────────────────────────────────────────

    def _predict_batched(self, pairs: list, batch_size: int) -> np.ndarray:
        """Run cross-encoder.predict in batches; return flat float32 array."""
        results = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            scores = self.cross_encoder.predict(batch)
            results.append(np.asarray(scores, dtype=np.float32))
        return np.concatenate(results) if results else np.array([], dtype=np.float32)

    # ── class methods ────────────────────────────────────────────────────────

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        cross_encoder,
        device: str = "cpu",
    ) -> "MemoryScorer":
        """
        Load a MemoryReRanker from *model_path*.

        Expects:
          - ``memory_scorer_model.pt``    — saved state dict
          - ``memory_scorer_version.json`` — metadata (optional, for logging)
        """
        path = Path(model_path)
        version_file = path / "memory_scorer_version.json"
        if version_file.exists():
            with version_file.open() as fh:
                meta = json.load(fh)
            logger.info("Loading MemoryReRanker version %s", meta.get("version", "unknown"))

        state = torch.load(path / "memory_scorer_model.pt", map_location=device)
        # Infer input_dim from first layer weight shape
        input_dim = state["net.0.weight"].shape[1]
        reranker = MemoryReRanker(input_dim=input_dim)
        reranker.load_state_dict(state)
        reranker.to(device)
        # Set model to inference mode
        reranker.training = False
        return cls(cross_encoder=cross_encoder, reranker=reranker, device=device)
