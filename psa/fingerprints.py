"""
fingerprints.py — FingerprintStore: persists accumulated query fingerprints per anchor.

Fingerprints are stored separately from the atlas JSON so they survive atlas rebuilds.
File location: <atlas_dir>/fingerprints.json

Structure on disk:
    {"<anchor_id>": ["query 1", "query 2", ...], ...}
"""

import json
import logging
import os
from typing import Dict, List

logger = logging.getLogger("psa.fingerprints")

MAX_FINGERPRINT_SIZE = 50


class FingerprintStore:
    """
    Persists accumulated query fingerprints per anchor.

    Each anchor accumulates queries from above-threshold activations (capped at
    MAX_FINGERPRINT_SIZE entries, FIFO eviction). Fingerprints are stored in
    <atlas_dir>/fingerprints.json — separate from anchor_cards.json so they
    survive atlas rebuilds.
    """

    def __init__(self, atlas_dir: str):
        self._path = os.path.join(atlas_dir, "fingerprints.json")
        self._data: Dict[int, List[str]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    raw = json.load(f)
                self._data = {int(k): v for k, v in raw.items()}
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load fingerprints from %s: %s", self._path, e)

    def save(self) -> None:
        """Persist fingerprints to disk. Best-effort — failures are logged, not raised."""
        try:
            with open(self._path, "w") as f:
                json.dump({str(k): v for k, v in self._data.items()}, f)
        except OSError as e:
            logger.warning("Failed to save fingerprints to %s: %s", self._path, e)

    def append(self, anchor_id: int, query: str) -> None:
        """Append a query to an anchor's fingerprint. Evicts oldest entry when full."""
        queries = self._data.setdefault(anchor_id, [])
        queries.append(query)
        if len(queries) > MAX_FINGERPRINT_SIZE:
            del queries[0]

    def get(self, anchor_id: int) -> List[str]:
        """Return a copy of the fingerprint for an anchor."""
        return list(self._data.get(anchor_id, []))

    def inherit_from(self, old_anchor_id: int, new_anchor_id: int) -> None:
        """
        Copy fingerprint from old_anchor_id to new_anchor_id.

        Called during atlas rebuild: matched anchors inherit their accumulated
        query signal into the new anchor's fingerprint.
        """
        queries = self._data.get(old_anchor_id, [])
        if queries:
            self._data[new_anchor_id] = list(queries)
