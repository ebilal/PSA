"""Tests for psa.advertisement.attribution — BM25 argmax + ε-split."""

from __future__ import annotations


def test_argmax_single_best(tmp_path):
    from psa.advertisement.attribution import attribute_bm25_argmax

    patterns = ["cat vs dog behavior", "unrelated pattern text"]
    query = "cat vs dog behavior analysis"
    argmax, tied = attribute_bm25_argmax(query, patterns, epsilon=0.05)
    assert argmax == "cat vs dog behavior"
    assert tied == []


def test_argmax_with_epsilon_tie():
    from psa.advertisement.attribution import attribute_bm25_argmax

    # Two near-duplicate patterns — BM25 scores should be within ε
    patterns = ["token refresh flow", "token refresh flow step", "fully unrelated"]
    query = "token refresh flow"
    argmax, tied = attribute_bm25_argmax(query, patterns, epsilon=0.5)
    assert argmax in {"token refresh flow", "token refresh flow step"}
    # Both near-duplicates should appear in argmax ∪ tied
    covered = {argmax, *tied}
    assert covered == {"token refresh flow", "token refresh flow step"}


def test_argmax_empty_patterns():
    from psa.advertisement.attribution import attribute_bm25_argmax

    argmax, tied = attribute_bm25_argmax("query", [], epsilon=0.05)
    assert argmax is None
    assert tied == []


def test_argmax_no_lexical_overlap():
    from psa.advertisement.attribution import attribute_bm25_argmax

    # Query and all patterns share zero terms
    patterns = ["pattern one", "pattern two"]
    query = "xyzxyz qqq"
    argmax, tied = attribute_bm25_argmax(query, patterns, epsilon=0.05)
    # BM25 returns 0 for every pattern. All are tied at zero — we don't credit
    # any template when nothing matches.
    assert argmax is None
    assert tied == []
