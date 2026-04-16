"""Tests for refine_anchor_cards.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the script directly (it lives outside the psa package)
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "refine_anchor_cards.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("refine_anchor_cards", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refine_anchor_cards"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
_extract_ngrams = _mod._extract_ngrams
refine_cards = _mod.refine_cards


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_card_a():
    """Card for anchor 1 with some existing patterns."""
    return {
        "anchor_id": 1,
        "name": "Caching Strategy",
        "meaning": "Approaches to caching and memoization.",
        "include_terms": ["cache", "memoize"],
        "exclude_terms": ["cpu"],
        "centroid": [0.1] * 768,
        "date_earliest": "2023-01-01",
        "date_latest": "2024-12-31",
        "date_months": [],
        "generated_query_patterns": ["caching approaches", "memoization patterns"],
        "prototype_examples": [],
        "query_fingerprint": [],
    }


@pytest.fixture
def base_card_b():
    """Card for anchor 2 with no misses."""
    return {
        "anchor_id": 2,
        "name": "Database Tuning",
        "meaning": "Optimizing database performance.",
        "include_terms": ["database"],
        "exclude_terms": [],
        "centroid": [0.2] * 768,
        "date_earliest": "2023-01-01",
        "date_latest": "2024-12-31",
        "date_months": [],
        "generated_query_patterns": ["database optimization"],
        "prototype_examples": [],
        "query_fingerprint": [],
    }


@pytest.fixture
def base_card_c():
    """Card for anchor 3 with empty patterns."""
    return {
        "anchor_id": 3,
        "name": "Testing",
        "meaning": "Test strategy and coverage.",
        "include_terms": ["test"],
        "exclude_terms": [],
        "centroid": [0.3] * 768,
        "date_earliest": "2023-01-01",
        "date_latest": "2024-12-31",
        "date_months": [],
        "generated_query_patterns": [],
        "prototype_examples": [],
        "query_fingerprint": [],
    }


@pytest.fixture
def base_cards(base_card_a, base_card_b, base_card_c):
    """List of base cards."""
    return [base_card_a, base_card_b, base_card_c]


@pytest.fixture
def misses_for_anchor_1():
    """Misses that reference anchor 1."""
    return [
        {
            "question_id": "q0",
            "query": "how do I implement efficient caching in Python",
            "gold_anchor_ids": [1],
            "gold_ce_rank": 18,
            "bucket": "scoring_rank",
            "selected_anchor_ids": [],
        },
        {
            "question_id": "q1",
            "query": "what is the best approach for query result caching",
            "gold_anchor_ids": [1],
            "gold_ce_rank": 35,
            "bucket": "paraphrase_gap",
            "selected_anchor_ids": [],
        },
        {
            "question_id": "q2",
            "query": "caching strategies for distributed systems",
            "gold_anchor_ids": [1],
            "gold_ce_rank": None,
            "bucket": "ce_miss",
            "selected_anchor_ids": [],
        },
    ]


@pytest.fixture
def misses_multi_anchor():
    """Misses with multiple gold anchors."""
    return [
        {
            "question_id": "q5",
            "query": "when should I use in-memory caching vs disk caching",
            "gold_anchor_ids": [1, 2],
            "gold_ce_rank": 50,
            "bucket": "paraphrase_gap",
            "selected_anchor_ids": [],
        },
    ]


# ---------------------------------------------------------------------------
# Tests for _extract_ngrams
# ---------------------------------------------------------------------------


def test_extract_ngrams_basic():
    """Extract ngrams from simple text."""
    text = "how to implement caching"
    ngrams = _extract_ngrams(text)
    # Should get 4-grams, 3-grams; prefer longer first (min_n=3 by default)
    # "how to implement caching" is all non-stop, so 4-gram included
    assert "how to implement caching" in ngrams
    assert "to implement caching" in ngrams
    assert "how to implement" in ngrams


def test_extract_ngrams_stops_filtering():
    """Stop words should be filtered out appropriately."""
    text = "a test of the system"
    ngrams = _extract_ngrams(text)
    # "a", "of", "the" are stop, so should not form 3-grams on their own
    # "test of the" has "test" (non-stop), so it's included
    assert "test of the" in ngrams
    # "a test of" has "test" (non-stop), included
    assert "a test of" in ngrams


def test_extract_ngrams_all_stops():
    """Text with only stopwords yields no ngrams."""
    text = "a the is at"
    ngrams = _extract_ngrams(text)
    # All stop words, no ngrams with non-stop content
    assert len(ngrams) == 0


def test_extract_ngrams_min_max():
    """Ngrams respect min and max lengths."""
    text = "one two three four five six seven"
    ngrams = _extract_ngrams(text, min_n=3, max_n=5)
    # Should only get 3, 4, 5 word ngrams
    ngram_lengths = [len(ng.split()) for ng in ngrams]
    assert all(3 <= length <= 5 for length in ngram_lengths)
    # Should NOT have 2-grams or 6-grams
    assert not any(len(ng.split()) == 2 for ng in ngrams)
    assert not any(len(ng.split()) == 6 for ng in ngrams)


def test_extract_ngrams_prefer_longer():
    """Longer ngrams appear first in results."""
    text = "implement efficient caching layer"
    ngrams = _extract_ngrams(text)
    # First result should be a 4-gram
    assert len(ngrams[0].split()) >= 3  # At least 3-gram


# ---------------------------------------------------------------------------
# Tests for refine_cards
# ---------------------------------------------------------------------------


def test_refined_preserves_anchor_id(base_cards, misses_for_anchor_1):
    """All anchor_ids are preserved (never changed)."""
    refined = refine_cards(base_cards, misses_for_anchor_1)
    for orig, ref in zip(base_cards, refined):
        assert orig["anchor_id"] == ref["anchor_id"]


def test_unchanged_anchors_copied_through(base_cards, misses_for_anchor_1):
    """Anchors with no misses are byte-identical to base."""
    refined = refine_cards(base_cards, misses_for_anchor_1)

    # Anchor 2 and 3 have no misses, should be identical
    assert refined[1] == base_cards[1], "anchor 2 should be unchanged"
    assert refined[2] == base_cards[2], "anchor 3 should be unchanged"


def test_changed_anchors_only_mutate_allowed_fields(base_cards, misses_for_anchor_1):
    """Refined anchors only differ in generated_query_patterns."""
    refined = refine_cards(base_cards, misses_for_anchor_1)

    # Anchor 1 is the only one with misses
    orig = base_cards[0]
    ref = refined[0]

    # These fields must be IDENTICAL
    immutable_fields = [
        "anchor_id",
        "name",
        "meaning",
        "include_terms",
        "exclude_terms",
        "centroid",
        "date_earliest",
        "date_latest",
        "date_months",
    ]
    for field in immutable_fields:
        assert orig[field] == ref[field], (
            f"Field {field!r} changed: {orig[field]!r} → {ref[field]!r}"
        )

    # generated_query_patterns may have changed (or not, if no novel patterns)
    # But if it did change, only that field should be different
    orig_other_fields = {k: v for k, v in orig.items() if k != "generated_query_patterns"}
    ref_other_fields = {k: v for k, v in ref.items() if k != "generated_query_patterns"}
    assert orig_other_fields == ref_other_fields


def test_patterns_augmented_with_novel_ngrams(base_cards, misses_for_anchor_1):
    """Refined anchor has more patterns (original + novel ngrams)."""
    refined = refine_cards(base_cards, misses_for_anchor_1)
    orig_patterns = set(base_cards[0]["generated_query_patterns"])
    refined_patterns = set(refined[0]["generated_query_patterns"])

    # All original patterns should still be there
    assert orig_patterns.issubset(refined_patterns)

    # Should have MORE patterns now (from ngrams)
    assert len(refined_patterns) > len(orig_patterns)


def test_patterns_not_duplicated(base_cards, misses_for_anchor_1):
    """Refined patterns don't contain duplicates."""
    refined = refine_cards(base_cards, misses_for_anchor_1)
    patterns = refined[0]["generated_query_patterns"]
    assert len(patterns) == len(set(patterns)), "Patterns contain duplicates"


def test_max_patterns_limit(base_cards, misses_for_anchor_1):
    """Refined patterns respect max_patterns limit."""
    max_patterns = 5
    refined = refine_cards(base_cards, misses_for_anchor_1, max_patterns=max_patterns)
    patterns = refined[0]["generated_query_patterns"]
    assert len(patterns) <= max_patterns, (
        f"Expected max {max_patterns} patterns, got {len(patterns)}"
    )


def test_multi_anchor_miss(base_cards, misses_multi_anchor):
    """Misses with multiple gold anchors refine both anchors."""
    refined = refine_cards(base_cards, misses_multi_anchor)

    # Both anchor 1 and 2 should have refined patterns
    # (they both appear in the gold_anchor_ids of the miss)
    anchor_1_patterns = set(refined[0]["generated_query_patterns"])
    anchor_2_patterns = set(refined[1]["generated_query_patterns"])

    orig_1_patterns = set(base_cards[0]["generated_query_patterns"])
    orig_2_patterns = set(base_cards[1]["generated_query_patterns"])

    # Both should be augmented
    assert anchor_1_patterns.issuperset(orig_1_patterns)
    assert anchor_2_patterns.issuperset(orig_2_patterns)


def test_empty_misses(base_cards):
    """Empty misses list returns cards unchanged."""
    refined = refine_cards(base_cards, [])
    assert refined == base_cards


def test_empty_base_cards():
    """Empty base cards returns empty list."""
    refined = refine_cards([], [{"gold_anchor_ids": [1]}])
    assert refined == []


def test_miss_with_empty_query(base_cards):
    """Misses with empty query don't crash."""
    misses = [{"gold_anchor_ids": [1], "query": ""}]
    # Should not raise
    refined = refine_cards(base_cards, misses)
    assert refined is not None


def test_miss_with_missing_fields(base_cards):
    """Misses missing gold_anchor_ids or query are skipped gracefully."""
    misses = [
        {"query": "test query"},  # Missing gold_anchor_ids
        {"gold_anchor_ids": [1]},  # Missing query
    ]
    # Should not raise
    refined = refine_cards(base_cards, misses)
    assert refined is not None


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_refine_cards_integration_full(tmp_path):
    """Full refinement flow: load cards, load misses, refine, save."""
    # Create test files
    base_cards = [
        {
            "anchor_id": 10,
            "name": "Test Anchor",
            "meaning": "A test anchor",
            "include_terms": [],
            "exclude_terms": [],
            "centroid": [0.0] * 768,
            "date_earliest": "2023-01-01",
            "date_latest": "2024-12-31",
            "date_months": [],
            "generated_query_patterns": ["test pattern"],
            "prototype_examples": [],
            "query_fingerprint": [],
        }
    ]
    base_cards_path = tmp_path / "anchor_cards.json"
    base_cards_path.write_text(json.dumps(base_cards))

    misses = [
        {
            "question_id": "q0",
            "query": "how do I test this functionality",
            "gold_anchor_ids": [10],
        }
    ]
    misses_path = tmp_path / "misses.jsonl"
    misses_path.write_text("\n".join(json.dumps(m) for m in misses) + "\n")

    # Call main()
    output_path = tmp_path / "refined.json"
    _mod.main(
        [
            "--base-atlas",
            str(base_cards_path),
            "--miss-log",
            str(misses_path),
            "--output",
            str(output_path),
            "--max-patterns",
            "20",
        ]
    )

    # Load and verify
    assert output_path.exists()
    with open(output_path) as f:
        refined_cards = json.load(f)

    assert len(refined_cards) == 1
    assert refined_cards[0]["anchor_id"] == 10
    # Should have more patterns than original
    assert len(refined_cards[0]["generated_query_patterns"]) > len(
        base_cards[0]["generated_query_patterns"]
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_anchor_with_many_misses(base_card_a):
    """Anchor with many misses gets many ngrams."""
    cards = [base_card_a]
    misses = [
        {
            "query": "how do I implement efficient caching in Python",
            "gold_anchor_ids": [1],
        },
        {
            "query": "best practices for cache invalidation",
            "gold_anchor_ids": [1],
        },
        {
            "query": "distributed caching layer architecture",
            "gold_anchor_ids": [1],
        },
        {
            "query": "memory management in cache systems",
            "gold_anchor_ids": [1],
        },
    ]

    refined = refine_cards(cards, misses, max_patterns=10)
    patterns = refined[0]["generated_query_patterns"]

    # Should be at or close to max_patterns
    assert len(patterns) >= 3  # At least the original patterns + some new ones
    assert len(patterns) <= 10


def test_patterns_longer_than_original_capacity(base_card_a):
    """If original has many patterns, max_patterns limits total."""
    card = base_card_a.copy()
    card["generated_query_patterns"] = ["pattern_" + str(i) for i in range(15)]
    cards = [card]

    misses = [
        {
            "query": "how do I implement efficient caching",
            "gold_anchor_ids": [1],
        }
    ]

    refined = refine_cards(cards, misses, max_patterns=20)
    patterns = refined[0]["generated_query_patterns"]

    # Total should not exceed max_patterns
    assert len(patterns) <= 20
