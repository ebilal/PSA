"""
Offline anchor-card refinement via heuristic ngram expansion.

This script takes a base atlas card file and a miss log, and produces
refined anchor cards with improved generated_query_patterns for anchors
that are being missed.

Usage:
    cd /Users/erhanbilal/Work/Projects/memnexus/.worktrees/phase2a
    uv run python scripts/refine_anchor_cards.py \
        --base-atlas PATH/TO/anchor_cards.json \
        --miss-log PATH/TO/misses.jsonl \
        --output PATH/TO/anchor_cards_refined.json \
        --max-patterns 20
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Stopwords for ngram filtering
_STOP = {
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


def _extract_ngrams(text: str, min_n: int = 3, max_n: int = 6) -> list[str]:
    """Extract ngrams from text, filtering out stopwords.

    Parameters
    ----------
    text : str
        Input text to extract ngrams from.
    min_n : int
        Minimum ngram length (words).
    max_n : int
        Maximum ngram length (words).

    Returns
    -------
    list[str]
        List of ngrams as space-separated strings, preferring longer ngrams first.
    """
    words = text.lower().split()
    result = []

    # Extract ngrams in descending order of length (prefer longer)
    for n in range(max_n, min_n - 1, -1):
        for i in range(len(words) - n + 1):
            gram = words[i : i + n]
            # Keep if at least one non-stopword
            if any(w not in _STOP for w in gram):
                result.append(" ".join(gram))

    return result


def refine_cards(base_cards: list[dict], misses: list[dict], max_patterns: int = 20) -> list[dict]:
    """Refine anchor cards by adding ngrams from missed queries.

    For each anchor that appears in the miss log, extract 3-6 word ngrams
    from missed queries and augment the anchor's generated_query_patterns
    with novel patterns. Does NOT change anchor_id, centroid, name, meaning,
    include_terms, exclude_terms, or date fields.

    Parameters
    ----------
    base_cards : list[dict]
        Original anchor cards from atlas (list of card dicts).
    misses : list[dict]
        Missed queries from benchmark (list of miss dicts with gold_anchor_ids).
    max_patterns : int
        Maximum generated_query_patterns per anchor after refinement.

    Returns
    -------
    list[dict]
        Refined anchor cards with updated generated_query_patterns.
    """
    # Build anchor_id -> missed queries map
    missed_queries_by_anchor: dict[int, list[str]] = defaultdict(list)
    for miss in misses:
        gold_ids = miss.get("gold_anchor_ids", [])
        query = miss.get("query", "")
        if query:
            for anchor_id in gold_ids:
                missed_queries_by_anchor[anchor_id].append(query)

    # Refine each card
    refined_cards = []
    for card in base_cards:
        anchor_id = card.get("anchor_id")
        new_card = card.copy()  # Shallow copy is sufficient since we only mutate lists

        # If this anchor has misses, extract and add ngrams
        if anchor_id in missed_queries_by_anchor:
            existing_list = card.get("generated_query_patterns", [])
            existing_set = set(existing_list)
            candidates = []
            seen = set()

            # Extract ngrams from all missed queries
            for query in missed_queries_by_anchor[anchor_id]:
                ngrams = _extract_ngrams(query)
                for ngram in ngrams:
                    if ngram not in existing_set and ngram not in seen:
                        candidates.append(ngram)
                        seen.add(ngram)

            # Sort longer patterns first, then alpha, to prefer longer patterns
            candidates.sort(key=lambda x: (-len(x.split()), x))
            available_slots = max(0, max_patterns - len(existing_list))
            new_patterns = candidates[:available_slots]

            # Update card with extended patterns, preserving original list order
            new_card["generated_query_patterns"] = existing_list + new_patterns

        refined_cards.append(new_card)

    return refined_cards


def _load_cards(path: str | Path) -> list[dict]:
    """Load anchor cards from JSON file."""
    with open(path) as f:
        return json.load(f)


def _load_misses(path: str | Path) -> list[dict]:
    """Load misses from JSONL file."""
    misses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                misses.append(json.loads(line))
    return misses


def _save_cards(cards: list[dict], path: str | Path) -> None:
    """Save refined cards to JSON file."""
    with open(path, "w") as f:
        json.dump(cards, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Refine anchor cards using missed query ngrams")
    parser.add_argument(
        "--base-atlas",
        type=str,
        required=True,
        help="Path to base anchor_cards.json",
    )
    parser.add_argument(
        "--miss-log",
        type=str,
        required=True,
        help="Path to misses.jsonl from benchmark run",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for output anchor_cards_refined.json",
    )
    parser.add_argument(
        "--max-patterns",
        type=int,
        default=20,
        help="Max generated_query_patterns per anchor after refinement (default: 20)",
    )

    args = parser.parse_args(argv)

    # Load data
    print(f"Loading base cards from {args.base_atlas}...")
    base_cards = _load_cards(args.base_atlas)

    print(f"Loading misses from {args.miss_log}...")
    misses = _load_misses(args.miss_log)

    # Refine
    print(
        f"Refining {len(base_cards)} cards with {len(misses)} misses "
        f"(max_patterns={args.max_patterns})..."
    )
    refined_cards = refine_cards(base_cards, misses, args.max_patterns)

    # Save
    print(f"Saving refined cards to {args.output}...")
    _save_cards(refined_cards, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
