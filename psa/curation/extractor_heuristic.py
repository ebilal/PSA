"""
extractor_heuristic.py — HeuristicExtractor: ngram-based query pattern extraction.

MVP backend for psa.curation. Zero LLM cost; deterministic. Output feeds
into generated_query_patterns via the curator after dedup against existing
patterns.
"""

from __future__ import annotations

from .ngrams import extract_ngrams


class HeuristicExtractor:
    """Ngram-based extractor.

    For each query in the pool, extracts 3–6 word ngrams (stopword-filtered,
    longer-first), unions across the pool preserving first-seen order, and
    returns up to `n` distinct patterns.
    """

    def extract(self, pool: list[str], n: int) -> list[str]:
        if n <= 0 or not pool:
            return []
        seen: set[str] = set()
        ordered: list[str] = []
        for query in pool:
            for gram in extract_ngrams(query):
                if gram not in seen:
                    seen.add(gram)
                    ordered.append(gram)
                    if len(ordered) >= n:
                        return ordered
        return ordered
