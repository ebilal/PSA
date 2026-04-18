"""
attribution.py — BM25 argmax attribution over an anchor's patterns.

Pure function; no DB access. Reused by the inline ledger writer and by
rebuild-ledger. Given a query and an anchor's generated_query_patterns,
returns (argmax_pattern, epsilon_tied_patterns). Patterns whose BM25
score is within `epsilon` of argmax share credit with argmax downstream.

Returns (None, []) when no pattern has a positive BM25 score — stage 2
does not credit a template when lexical contribution is zero.
"""

from __future__ import annotations

import re
from typing import List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    avg_doc_len: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Single-document BM25 against a background of N=1 (no IDF damping).

    This is a deliberately simple in-document BM25 — we are attributing
    credit within one anchor's small pattern list, not ranking across a
    corpus. Term-frequency saturation + length normalization give the
    expected "best lexical match" shape without needing a global IDF.
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    dl = len(doc_tokens)
    score = 0.0
    tf: dict[str, int] = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1
    for q in query_tokens:
        if q not in tf:
            continue
        f = tf[q]
        norm = 1 - b + b * (dl / max(avg_doc_len, 1e-9))
        score += (f * (k1 + 1)) / (f + k1 * norm)
    return score


def attribute_bm25_argmax(
    query: str, patterns: List[str], epsilon: float
) -> Tuple[str | None, List[str]]:
    """Return (argmax_pattern, epsilon_tied_patterns) over BM25 scores.

    Returns (None, []) when patterns is empty or all scores are zero.
    Epsilon ties are patterns whose score is within `epsilon` of argmax,
    excluding argmax itself.
    """
    if not patterns:
        return None, []
    q_tokens = _tokenize(query)
    if not q_tokens:
        return None, []

    tokenized = [_tokenize(p) for p in patterns]
    avg_len = sum(len(t) for t in tokenized) / len(tokenized)

    scores = [_bm25_score(q_tokens, dt, avg_len) for dt in tokenized]
    max_score = max(scores)
    if max_score <= 0.0:
        return None, []

    argmax_idx = scores.index(max_score)
    argmax_pattern = patterns[argmax_idx]
    tied = [
        patterns[i]
        for i, s in enumerate(scores)
        if i != argmax_idx and (max_score - s) <= epsilon and s > 0.0
    ]
    return argmax_pattern, tied
