"""Tests for psa.curation.ngrams — pure ngram extraction."""

from __future__ import annotations

from psa.curation.ngrams import extract_ngrams, STOPWORDS


def test_extract_ngrams_returns_longer_patterns_first():
    result = extract_ngrams("how does the token refresh flow work", min_n=3, max_n=6)
    lengths = [len(g.split()) for g in result]
    assert lengths == sorted(lengths, reverse=True), result


def test_extract_ngrams_drops_all_stopword_spans():
    """A span that contains only stopwords is excluded."""
    # "the is a" is all stopwords; should not appear.
    result = extract_ngrams("the is a query about auth", min_n=3, max_n=3)
    assert "the is a" not in result


def test_extract_ngrams_keeps_spans_with_any_content_word():
    """A span with ≥1 non-stopword survives."""
    result = extract_ngrams("the is token refresh", min_n=3, max_n=3)
    assert any("token" in g for g in result)


def test_extract_ngrams_empty_input():
    assert extract_ngrams("", min_n=3, max_n=6) == []


def test_extract_ngrams_shorter_than_min_n():
    """Input with fewer words than min_n yields empty."""
    assert extract_ngrams("too short", min_n=3, max_n=6) == []


def test_stopwords_is_a_frozenset_or_set():
    """STOPWORDS is exposed and a set-like collection."""
    assert "the" in STOPWORDS
    assert "token" not in STOPWORDS
