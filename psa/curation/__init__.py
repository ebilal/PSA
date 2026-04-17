"""psa.curation — production-signal card curation (fingerprints + oracle labels → candidate refinements)."""

from __future__ import annotations

from typing import Protocol


class QueryPatternExtractor(Protocol):
    """Extractor interface: pool of queries → candidate query patterns.

    Concrete backends live in `extractor_heuristic.py` (MVP, ngrams)
    and `extractor_llm.py` (reserved; stub raises NotImplementedError).
    """

    def extract(self, pool: list[str], n: int) -> list[str]:
        """Return up to `n` candidate query patterns derived from the pool."""
        ...
