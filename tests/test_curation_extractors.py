"""Tests for psa.curation extractors — heuristic (MVP) and LLM (stub)."""

from __future__ import annotations

from psa.curation.extractor_heuristic import HeuristicExtractor


def test_heuristic_extractor_produces_ngrams_from_pool():
    extractor = HeuristicExtractor()
    pool = [
        "how does the token refresh flow work",
        "what is the expiry for an access token",
    ]
    out = extractor.extract(pool, n=10)
    assert isinstance(out, list)
    assert len(out) <= 10
    assert all(isinstance(p, str) for p in out)
    # Should include an obviously content-bearing ngram.
    assert any("token" in p for p in out)


def test_heuristic_extractor_respects_cap():
    extractor = HeuristicExtractor()
    # Long query with many possible ngrams.
    pool = ["one two three four five six seven eight nine ten"]
    out = extractor.extract(pool, n=3)
    assert len(out) == 3


def test_heuristic_extractor_dedupes_across_pool():
    extractor = HeuristicExtractor()
    pool = [
        "refresh access token flow",
        "refresh access token flow",   # exact duplicate query
    ]
    out = extractor.extract(pool, n=50)
    # No repeated ngrams across the two identical queries.
    assert len(out) == len(set(out))


def test_heuristic_extractor_empty_pool():
    extractor = HeuristicExtractor()
    assert extractor.extract([], n=10) == []


def test_heuristic_extractor_prefers_longer_patterns():
    extractor = HeuristicExtractor()
    pool = ["how does the authentication token refresh flow work under load"]
    out = extractor.extract(pool, n=3)
    # The first (kept) pattern should be among the longest available.
    max_len = max(len(p.split()) for p in out)
    assert len(out[0].split()) == max_len
