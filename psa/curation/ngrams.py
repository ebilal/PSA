"""
ngrams.py — pure ngram extraction for card curation.

Lifted from scripts/refine_anchor_cards.py to make the logic importable
from the psa package without reaching into scripts/. The script keeps
its own copy for backward compatibility; deduplicating the two is a
future cleanup (not in scope for Branch 3).
"""

from __future__ import annotations

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "our",
        "their",
        "this",
        "that",
        "what",
        "when",
        "where",
        "who",
        "which",
        "how",
        "about",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "up",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
    }
)


def extract_ngrams(text: str, min_n: int = 3, max_n: int = 6) -> list[str]:
    """Extract ngrams of length min_n..max_n from `text`, longer first.

    A span is kept iff at least one of its words is NOT in STOPWORDS.
    The result is a list of space-separated lowercase ngrams; longer
    patterns come first so downstream dedup-then-cap prefers them.
    """
    words = text.lower().split()
    result: list[str] = []
    for n in range(max_n, min_n - 1, -1):
        for i in range(len(words) - n + 1):
            gram = words[i : i + n]
            if any(w not in STOPWORDS for w in gram):
                result.append(" ".join(gram))
    return result
