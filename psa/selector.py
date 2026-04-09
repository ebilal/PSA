"""
selector.py — Anchor selector for PSA.

The selector re-ranks retriever candidates and picks the 1-4 anchors
most likely to contain context useful for answering the query.

Two modes:
  "cosine"  — cosine reranking (default until training gates are met)
  "trained" — fine-tuned cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)

Training gates (from the plan):
  - >= 300 oracle-labeled queries
  - >= 200 held-out real queries
  - shortlist recall@24 >= 0.95
  - >= 50 examples per active query family

Until all gates are met the selector stays in cosine mode and logs
the gap to readiness.
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .retriever import AnchorCandidate

logger = logging.getLogger("psa.selector")

# ── Selector config ───────────────────────────────────────────────────────────

DEFAULT_MAX_K = 4        # maximum anchors to return
DEFAULT_THRESHOLD = 0.0  # minimum score to include (cosine mode: always include top-1)

# Trained mode settings
TRAINED_MAX_SEQ = 320    # max input tokens for cross-encoder
TRAINED_THRESHOLD_DEFAULT = 0.3  # score threshold in trained mode


# ── Dataclass ─────────────────────────────────────────────────────────────────


@dataclass
class SelectedAnchor:
    """An anchor chosen by the selector, with scoring metadata."""

    anchor_id: int
    selector_score: float
    mode: str   # "cosine" | "trained"
    candidate: AnchorCandidate


# ── Cosine selector (baseline) ────────────────────────────────────────────────


def _cosine_select(
    query_vec: List[float],
    candidates: List[AnchorCandidate],
    max_k: int,
    threshold: float,
) -> List[SelectedAnchor]:
    """
    Cosine re-ranking: score candidates by their dense_score (already cosine
    similarity from the retriever's AnchorIndex search).

    Returns empty list if no candidate meets the threshold — the query
    has no relevant memories.
    """
    if not candidates:
        return []

    # Sort by dense_score desc (retriever already does this, but be explicit)
    ranked = sorted(candidates, key=lambda c: c.dense_score, reverse=True)

    selected = []
    for cand in ranked[:max_k]:
        score = cand.dense_score
        if score < threshold:
            break
        selected.append(
            SelectedAnchor(
                anchor_id=cand.anchor_id,
                selector_score=score,
                mode="cosine",
                candidate=cand,
            )
        )

    return selected


# ── Trained cross-encoder selector ────────────────────────────────────────────


def _load_cross_encoder(model_path: str):
    """Load a fine-tuned cross-encoder from disk."""
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
        return CrossEncoder(model_path, max_length=TRAINED_MAX_SEQ)
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for the trained selector. "
            "Install it with: pip install 'psa[training]'"
        )


def _trained_select(
    query: str,
    candidates: List[AnchorCandidate],
    cross_encoder,
    max_k: int,
    threshold: float,
) -> List[SelectedAnchor]:
    """
    Score all candidates with a fine-tuned cross-encoder and threshold.

    Input pairs: (query, anchor_card_text) → scalar score.
    """
    if not candidates:
        return []

    pairs = [(query, c.card.to_stable_card_text()) for c in candidates]
    try:
        scores = cross_encoder.predict(pairs)
    except Exception as e:
        logger.warning("Cross-encoder prediction failed (%s); falling back to cosine", e)
        return _cosine_select(
            query_vec=[],        # unused by _cosine_select (sorts by dense_score)
            candidates=candidates,
            max_k=max_k,
            threshold=0.0,      # no threshold in fallback — always return top-k
        )

    scored = sorted(
        zip(candidates, scores), key=lambda x: x[1], reverse=True
    )

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

    return selected


# ── AnchorSelector ────────────────────────────────────────────────────────────


class AnchorSelector:
    """
    Selects 1-4 anchors from a retriever shortlist.

    Mode selection:
    - "cosine": always available, uses dense_score from retriever
    - "trained": requires a saved cross-encoder at model_path

    The selector does NOT enforce training gates — that is the caller's
    responsibility (AtlasManager or PSAPipeline checks gates before
    switching to trained mode).
    """

    def __init__(
        self,
        mode: str = "cosine",
        model_path: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD,
        max_k: int = DEFAULT_MAX_K,
    ):
        if mode not in ("cosine", "trained"):
            raise ValueError(f"Unknown selector mode {mode!r}. Choose 'cosine' or 'trained'.")
        self.mode = mode
        self.model_path = model_path
        self.threshold = threshold
        self.max_k = max_k
        self._cross_encoder = None

        if mode == "trained":
            if not model_path or not os.path.exists(model_path):
                logger.warning(
                    "Trained selector requested but model_path '%s' not found; "
                    "falling back to cosine mode.",
                    model_path,
                )
                self.mode = "cosine"
            else:
                self._cross_encoder = _load_cross_encoder(model_path)

    def select(
        self,
        query: str,
        candidates: List[AnchorCandidate],
        query_vec: Optional[List[float]] = None,
    ) -> List[SelectedAnchor]:
        """
        Select anchors from retriever candidates.

        Parameters
        ----------
        query:
            The user query string (used in trained mode).
        candidates:
            Shortlisted AnchorCandidates from AnchorRetriever.
        query_vec:
            Pre-computed query embedding (used in cosine mode; optional).

        Returns
        -------
        List of SelectedAnchor sorted by selector_score descending.
        Min 1, max self.max_k anchors.
        """
        if not candidates:
            return []

        if self.mode == "trained" and self._cross_encoder is not None:
            return _trained_select(
                query=query,
                candidates=candidates,
                cross_encoder=self._cross_encoder,
                max_k=self.max_k,
                threshold=self.threshold,
            )
        else:
            return _cosine_select(
                query_vec=query_vec or [],
                candidates=candidates,
                max_k=self.max_k,
                threshold=self.threshold,
            )

    @classmethod
    def cosine(cls, max_k: int = DEFAULT_MAX_K) -> "AnchorSelector":
        """Convenience constructor for cosine mode."""
        return cls(mode="cosine", max_k=max_k)

    @classmethod
    def trained(cls, model_path: str, threshold: float = TRAINED_THRESHOLD_DEFAULT, max_k: int = DEFAULT_MAX_K) -> "AnchorSelector":
        """Convenience constructor for trained mode."""
        return cls(mode="trained", model_path=model_path, threshold=threshold, max_k=max_k)


# ── Training gate check ───────────────────────────────────────────────────────


@dataclass
class TrainingGateStatus:
    """Reports whether selector training gates are met for a tenant."""

    oracle_count: int
    held_out_count: int
    shortlist_recall_24: Optional[float]
    query_family_counts: dict
    gates_met: bool
    blocking_reasons: List[str]


def check_training_gates(
    oracle_count: int,
    held_out_count: int,
    shortlist_recall_24: Optional[float],
    query_family_counts: Optional[dict] = None,
    oracle_min: int = 300,
    held_out_min: int = 200,
    recall_min: float = 0.95,
    family_min: int = 50,
) -> TrainingGateStatus:
    """
    Evaluate whether selector training gates are met.

    Returns a TrainingGateStatus with gates_met=True only when all gates pass.
    """
    reasons = []

    if oracle_count < oracle_min:
        reasons.append(
            f"oracle_count {oracle_count} < {oracle_min} required"
        )

    if held_out_count < held_out_min:
        reasons.append(
            f"held_out_count {held_out_count} < {held_out_min} required"
        )

    if shortlist_recall_24 is not None and shortlist_recall_24 < recall_min:
        reasons.append(
            f"shortlist recall@24 {shortlist_recall_24:.3f} < {recall_min:.3f} required"
        )
    elif shortlist_recall_24 is None:
        reasons.append("shortlist recall@24 not yet measured")

    active_families = []
    if query_family_counts:
        for family, count in query_family_counts.items():
            if count < family_min:
                reasons.append(
                    f"query family '{family}' has {count} examples (< {family_min} required)"
                )
            else:
                active_families.append(family)

    if reasons:
        logger.info(
            "Selector training gates NOT met for this tenant. Blocking reasons: %s",
            "; ".join(reasons),
        )
    else:
        logger.info("Selector training gates MET. Tenant is ready for selector training.")

    return TrainingGateStatus(
        oracle_count=oracle_count,
        held_out_count=held_out_count,
        shortlist_recall_24=shortlist_recall_24,
        query_family_counts=query_family_counts or {},
        gates_met=len(reasons) == 0,
        blocking_reasons=reasons,
    )
