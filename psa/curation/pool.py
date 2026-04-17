"""
pool.py — per-anchor query pool construction with anchor-level oracle endorsement.

Two signal sources feed a Pool:
- oracle_queries: queries from oracle_labels.jsonl where this anchor is in
  winning_oracle_set. Normative signal.
- endorsed_fingerprint_queries: fingerprints for this anchor, included iff the
  anchor has appeared in ≥1 winning_oracle_set (anchor-level endorsement).
  Observed signal, minimally vetted.

The two lists stay separate even though curator.py unions them at extraction
time — preserves provenance for later weighting / diagnostics.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("psa.curation.pool")


@dataclass
class Pool:
    """Per-anchor curated query pool.

    Both fields remain separate in the internal data model; curator unions
    them before extraction. Future weighting / diagnostics work will need
    the split.
    """

    oracle_queries: list[str] = field(default_factory=list)
    endorsed_fingerprint_queries: list[str] = field(default_factory=list)


def build_pool(atlas: Any, oracle_labels_path: str) -> dict[int, Pool]:
    """Build a Pool per anchor in the atlas.

    Parameters
    ----------
    atlas:
        An Atlas instance with .cards (iterable of objects with .anchor_id) and
        .fingerprint_store (a FingerprintStore providing .get(anchor_id)).
    oracle_labels_path:
        Path to oracle_labels.jsonl. Missing file is soft — returns empty pools.

    Returns
    -------
    dict mapping anchor_id → Pool for every anchor in the atlas. Anchors with
    no signal get a Pool with two empty lists (not absent from the dict).
    """
    pools: dict[int, Pool] = {c.anchor_id: Pool() for c in atlas.cards}

    # Pass 1: oracle labels. Also determine which anchors have ≥1 endorsement.
    endorsed_anchors: set[int] = set()
    if os.path.exists(oracle_labels_path):
        with open(oracle_labels_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    label = json.loads(line)
                except json.JSONDecodeError:
                    continue
                query = label.get("query")
                winning = label.get("winning_oracle_set") or []
                if not query:
                    continue
                for aid in winning:
                    if aid in pools:
                        pools[aid].oracle_queries.append(query)
                        endorsed_anchors.add(aid)
    else:
        logger.warning("Oracle labels file not found: %s", oracle_labels_path)

    # Pass 2: fingerprints — only for anchors with ≥1 oracle endorsement.
    fp_store = atlas.fingerprint_store
    for aid in pools:
        if aid not in endorsed_anchors:
            continue
        fps = fp_store.get(aid) if fp_store is not None else []
        pools[aid].endorsed_fingerprint_queries.extend(fps)

    return pools
