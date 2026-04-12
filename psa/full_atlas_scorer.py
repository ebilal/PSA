"""
full_atlas_scorer.py — Score ALL atlas anchors in a single batched pass.

FullAtlasScorer replaces the top-24 retriever shortlist with a full-atlas
cross-encoder scoring pass, eliminating the retriever bottleneck (26% of
gold anchors lost at top-24 cutoff).

Two modes:
  - cross-encoder: builds (query, card_text) pairs and calls predict() once
  - cosine fallback: uses dot-product of query_vec against all centroids
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger("psa.full_atlas_scorer")

# Max input tokens for cross-encoder (matches selector.py)
_TRAINED_MAX_SEQ = 320


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_cross_encoder(model_path: str):
    """Load a cross-encoder from disk."""
    try:
        from sentence_transformers.cross_encoder import CrossEncoder

        return CrossEncoder(model_path, max_length=_TRAINED_MAX_SEQ)
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for the trained scorer. "
            "Install it with: pip install 'psa[training]'"
        )


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class AnchorScore:
    """A scored anchor from the full-atlas pass."""

    anchor_id: int
    ce_score: float
    centroid: np.ndarray  # 768-dim L2-normalized centroid


# ── FullAtlasScorer ───────────────────────────────────────────────────────────


class FullAtlasScorer:
    """
    Score ALL atlas anchors in a single batched cross-encoder forward pass.

    Usage::

        scorer = FullAtlasScorer(cross_encoder=ce, atlas=atlas)
        anchor_scores = scorer.score_all(query)

    Or with cosine fallback (no cross-encoder)::

        scorer = FullAtlasScorer(cross_encoder=None, atlas=atlas)
        anchor_scores = scorer.score_all(query, query_vec=embedding)
    """

    def __init__(self, cross_encoder, atlas):
        self._cross_encoder = cross_encoder
        self.atlas = atlas

    def score_all(self, query: str, query_vec: Optional[np.ndarray] = None) -> List[AnchorScore]:
        """
        Score all atlas anchors and return sorted results.

        Parameters
        ----------
        query:
            The user query string. Used with cross-encoder mode.
        query_vec:
            L2-normalized query embedding (768-dim). Required for cosine
            fallback (when cross_encoder is None).

        Returns
        -------
        List of AnchorScore sorted by ce_score descending (all anchors included).
        """
        if self._cross_encoder is not None:
            return self._cross_encoder_score(query)
        else:
            return self._cosine_fallback(query_vec)

    def _cross_encoder_score(self, query: str) -> List[AnchorScore]:
        """Score all anchors with the cross-encoder in a single predict call."""
        cards = self.atlas.cards
        pairs = [(query, card.to_stable_card_text()) for card in cards]
        raw_scores = self._cross_encoder.predict(pairs)

        results = [
            AnchorScore(
                anchor_id=card.anchor_id,
                ce_score=float(score),
                centroid=np.asarray(card.centroid, dtype=np.float32),
            )
            for card, score in zip(cards, raw_scores)
        ]
        results.sort(key=lambda r: r.ce_score, reverse=True)
        return results

    def _cosine_fallback(self, query_vec: Optional[np.ndarray]) -> List[AnchorScore]:
        """Score all anchors via cosine similarity to query_vec."""
        if query_vec is None:
            raise ValueError(
                "query_vec is required when cross_encoder is None. "
                "Pass a 768-dim L2-normalized embedding."
            )
        cards = self.atlas.cards
        qvec = np.asarray(query_vec, dtype=np.float32)
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec = qvec / norm

        results = []
        for card in cards:
            centroid = np.asarray(card.centroid, dtype=np.float32)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid_normed = centroid / c_norm
            else:
                centroid_normed = centroid
            score = float(np.dot(qvec, centroid_normed))
            results.append(
                AnchorScore(
                    anchor_id=card.anchor_id,
                    ce_score=score,
                    centroid=np.asarray(card.centroid, dtype=np.float32),
                )
            )

        results.sort(key=lambda r: r.ce_score, reverse=True)
        return results

    @classmethod
    def from_model_path(cls, model_path: str, atlas) -> "FullAtlasScorer":
        """
        Factory: load a cross-encoder from disk and construct a FullAtlasScorer.

        Parameters
        ----------
        model_path:
            Path to a saved CrossEncoder model directory.
        atlas:
            Atlas instance whose cards will be scored.
        """
        ce = _load_cross_encoder(model_path)
        return cls(cross_encoder=ce, atlas=atlas)
