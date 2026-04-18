"""
retriever.py — Hybrid anchor retrieval (BM25 + dense) for PSA.

AnchorRetriever produces a ranked list of AnchorCandidate objects by fusing:
  1. Dense retrieval — embed query, dot-product over anchor centroids (via AnchorIndex)
  2. BM25 (sparse) — TF-IDF bag-of-words over anchor card text

Fusion: reciprocal rank fusion (RRF) with k=60.

Target: 5-15ms per query.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from .anchor import AnchorCard
from .atlas import Atlas

logger = logging.getLogger("psa.retriever")

# ── Stopwords ─────────────────────────────────────────────────────────────────

_STOP_WORDS = frozenset(
    """a an the and or but of in on at to for with by from is are was were
    be been being have has had do does did will would can could shall should
    may might must that this these those it its i you he she we they me him
    her us them my your his our their what when where who which how
    """.split()
)

# ── BM25 parameters ───────────────────────────────────────────────────────────

BM25_K1 = 1.5
BM25_B = 0.75

# ── RRF parameter ─────────────────────────────────────────────────────────────

RRF_K = 60


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class AnchorCandidate:
    """A candidate anchor returned by the retriever."""

    anchor_id: int
    card: AnchorCard
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    dense_rank: int = 0
    bm25_rank: int = 0


@dataclass
class RetrievalResult:
    """Retrieval output wrapper that also carries the BM25-side shortlist.

    Used by pipeline-level callers (e.g., stage 2 ledger attribution) that
    need to know which anchors were in the BM25 top-K before RRF fusion.
    Separate from `AnchorCandidate` so the existing retrieve() API stays
    backward-compatible.
    """

    anchor_ids: List[int]
    scores: List[float]
    bm25_topk_anchor_ids: List[int] = field(default_factory=list)


# ── Tokenizer ─────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stop words, min length 2."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in _STOP_WORDS]


# ── BM25 index ────────────────────────────────────────────────────────────────


class BM25Index:
    """
    Simple BM25 index over a corpus of anchor card texts.

    Corpus is small (256 documents), so we build a full TF-IDF matrix eagerly.
    """

    def __init__(self, cards: List[AnchorCard]):
        self._cards = cards
        self._n = len(cards)
        self._doc_tokens: List[List[str]] = [_tokenize(c.to_card_text()) for c in cards]
        self._doc_lengths = [len(t) for t in self._doc_tokens]
        self._avg_dl = sum(self._doc_lengths) / max(self._n, 1)

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        df: Counter = Counter()
        for tokens in self._doc_tokens:
            df.update(set(tokens))
        self._idf: dict = {
            t: math.log((self._n - cnt + 0.5) / (cnt + 0.5) + 1.0) for t, cnt in df.items()
        }

    def score(self, query: str) -> List[float]:
        """Return BM25 score for each card in corpus order."""
        qtokens = _tokenize(query)
        scores = []
        for doc_idx, tokens in enumerate(self._doc_tokens):
            tf_counts: Counter = Counter(tokens)
            dl = self._doc_lengths[doc_idx]
            dl_norm = 1 - BM25_B + BM25_B * (dl / max(self._avg_dl, 1))
            s = 0.0
            for qt in qtokens:
                if qt not in self._idf:
                    continue
                tf = tf_counts.get(qt, 0)
                tf_score = (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * dl_norm)
                s += self._idf[qt] * tf_score
            scores.append(s)
        return scores


# ── RRF fusion ────────────────────────────────────────────────────────────────


def _reciprocal_rank_fusion(
    dense_ranked: List[int],
    bm25_ranked: List[int],
    n_cards: int,
    k: int = RRF_K,
) -> List[float]:
    """
    Compute RRF scores for each position.

    dense_ranked / bm25_ranked: ordered lists of anchor indices (0-based into
    the cards list, not anchor_ids), best first.
    n_cards: total number of cards (scores array is sized to this).

    Returns a list of RRF scores indexed by card position.
    """
    scores = [0.0] * n_cards

    for rank, idx in enumerate(dense_ranked, start=1):
        if 0 <= idx < n_cards:
            scores[idx] += 1.0 / (k + rank)

    for rank, idx in enumerate(bm25_ranked, start=1):
        if 0 <= idx < n_cards:
            scores[idx] += 1.0 / (k + rank)

    return scores


# ── AnchorRetriever ───────────────────────────────────────────────────────────


class AnchorRetriever:
    """
    Hybrid anchor retrieval: BM25 + dense, fused via RRF.

    Usage::

        retriever = AnchorRetriever(atlas)
        candidates = retriever.retrieve(query, embedding_model, top_k=24)
    """

    def __init__(self, atlas: Atlas):
        self.atlas = atlas
        self._bm25: Optional[BM25Index] = None

    def _get_bm25(self) -> BM25Index:
        """Lazily build BM25 index from anchor cards."""
        if self._bm25 is None:
            self._bm25 = BM25Index(self.atlas.cards)
        return self._bm25

    def retrieve(
        self,
        query: str,
        embedding_model,
        top_k: int = 32,
        query_vec: Optional[List[float]] = None,
    ) -> List[AnchorCandidate]:
        """
        Retrieve top-k anchor candidates for a query.

        Parameters
        ----------
        query:
            The user query string.
        embedding_model:
            psa.embeddings.EmbeddingModel instance.
        top_k:
            Number of candidates to return (plan default: 24).
        query_vec:
            Pre-computed query embedding. If provided, skips re-embedding.

        Returns
        -------
        List of AnchorCandidate sorted by RRF score descending.
        """
        cards = self.atlas.cards
        if not cards:
            return []

        k = min(top_k, len(cards))

        # ── Dense retrieval ──────────────────────────────────────────────────
        if query_vec is None:
            query_vec = embedding_model.embed(query)
        dense_hits = self.atlas.anchor_index.search(query_vec, top_k=k)
        # dense_hits: [(anchor_id, score), ...] sorted by score desc

        # Build mapping anchor_id → card list index
        id_to_idx = {c.anchor_id: i for i, c in enumerate(cards)}

        # Dense ranked order (by card index)
        dense_ranked = [id_to_idx[aid] for (aid, _) in dense_hits if aid in id_to_idx]

        # ── BM25 retrieval ───────────────────────────────────────────────────
        bm25_scores = self._get_bm25().score(query)
        bm25_ranked = sorted(range(len(cards)), key=lambda i: bm25_scores[i], reverse=True)[:k]

        # ── RRF fusion ───────────────────────────────────────────────────────
        rrf_scores = _reciprocal_rank_fusion(dense_ranked, bm25_ranked, n_cards=len(cards))

        # Build AnchorCandidate objects
        dense_score_map = {id_to_idx[aid]: score for (aid, score) in dense_hits if aid in id_to_idx}
        dense_rank_map = {idx: rank for rank, idx in enumerate(dense_ranked, start=1)}
        bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_ranked, start=1)}

        candidates = []
        for card_idx, card in enumerate(cards):
            rrf = rrf_scores[card_idx]
            if rrf == 0.0:
                continue
            candidates.append(
                AnchorCandidate(
                    anchor_id=card.anchor_id,
                    card=card,
                    dense_score=dense_score_map.get(card_idx, 0.0),
                    bm25_score=bm25_scores[card_idx] if card_idx < len(bm25_scores) else 0.0,
                    rrf_score=rrf,
                    dense_rank=dense_rank_map.get(card_idx, 999),
                    bm25_rank=bm25_rank_map.get(card_idx, 999),
                )
            )

        candidates.sort(key=lambda c: c.rrf_score, reverse=True)
        return candidates[:k]

    def retrieve_with_bm25_topk(
        self,
        query: str,
        embedding_model,
        top_k: int = 32,
        bm25_topk_floor: int = 48,
        query_vec=None,
    ) -> "RetrievalResult":
        """Retrieve candidates AND expose the BM25-side top-K shortlist.

        The main candidate list is produced by the normal retrieve() path
        (RRF over dense + BM25). In addition, this method returns the
        top-`bm25_topk_floor` anchor ids by BM25 score alone. Stage 2
        advertisement attribution uses the shortlist to gate template
        credit: a retrieved anchor only earns credit for one of its
        templates if it actually made the BM25 shortlist (i.e., lexical
        contribution was non-negligible).
        """
        candidates = self.retrieve(
            query=query,
            embedding_model=embedding_model,
            top_k=top_k,
            query_vec=query_vec,
        )
        # BM25 top-K by raw score, independent of RRF.
        bm25_scores = self._get_bm25().score(query)
        cards = self.atlas.cards
        indexed = [
            (cards[i].anchor_id, s)
            for i, s in enumerate(bm25_scores)
            if s > 0.0
        ]
        indexed.sort(key=lambda p: p[1], reverse=True)
        bm25_topk_anchor_ids = [aid for aid, _ in indexed[:bm25_topk_floor]]

        return RetrievalResult(
            anchor_ids=[c.anchor_id for c in candidates],
            scores=[c.rrf_score for c in candidates],
            bm25_topk_anchor_ids=bm25_topk_anchor_ids,
        )

    def invalidate_bm25_cache(self):
        """Call after atlas cards change (e.g., after card text refinement)."""
        self._bm25 = None
