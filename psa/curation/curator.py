"""
curator.py — production-signal curation orchestrator.

curate(tenant_id, extractor_name) loads the current atlas and its
FingerprintStore, builds per-anchor Pools, runs the chosen extractor,
dedupes extractor output against existing generated_query_patterns, and
writes a Branch 1 candidate + sibling .meta.json — but only if the run
produced real changes (empty-run guard).

Never writes to anchor_cards_refined.json. Promotion stays under
`psa atlas promote-refinement`.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from pathlib import Path
from typing import Any

from .extractor_heuristic import HeuristicExtractor
from .extractor_llm import LLMExtractor
from .pool import build_pool

logger = logging.getLogger("psa.curation.curator")

MAX_PATTERNS_PER_ANCHOR = 20
SUPPORT_SEMANTICS = "anchor_level_oracle_endorsement"


def _make_extractor(name: str):
    if name == "heuristic":
        return HeuristicExtractor()
    if name == "llm":
        return LLMExtractor()
    raise ValueError(f"Unknown extractor: {name!r}. Must be 'heuristic' or 'llm'.")


def curate(tenant_id: str = "default", extractor_name: str = "heuristic") -> dict[str, Any]:
    """Run a curation pass for `tenant_id` using the named extractor.

    Returns a summary dict with keys:
        - wrote_candidate: bool
        - n_anchors_touched: int
        - n_patterns_added: int
        - oracle_labels_read: int
        - fingerprints_read: int
        - n_anchors_with_oracle_support: int
        - n_anchors_with_endorsed_fingerprints: int
        - extractor: str
        - support_semantics: str
        - atlas_version: int
        - candidate_path: str | None
        - reason: str | None  (populated when wrote_candidate is False)
    """
    from ..atlas import AtlasManager
    from ..tenant import TenantManager

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
    atlas = mgr.get_atlas()
    if atlas is None:
        raise FileNotFoundError(f"No atlas for tenant '{tenant_id}'. Run 'psa atlas build' first.")

    extractor = _make_extractor(extractor_name)

    labels_path = os.path.join(tenant.root_dir, "training", "oracle_labels.jsonl")
    pools = build_pool(atlas, labels_path)

    # Count signals (observational metrics for metadata).
    oracle_labels_read = sum(1 for p in pools.values() for _ in p.oracle_queries)
    fingerprints_read = sum(1 for p in pools.values() for _ in p.endorsed_fingerprint_queries)
    n_anchors_with_oracle_support = sum(1 for p in pools.values() if p.oracle_queries)
    n_anchors_with_endorsed_fingerprints = sum(
        1 for p in pools.values() if p.endorsed_fingerprint_queries
    )

    # Build refined card list (whether or not we ultimately write it).
    refined_cards: list[dict] = []
    n_anchors_touched = 0
    n_patterns_added = 0

    for card in atlas.cards:
        pool = pools.get(card.anchor_id)
        existing = list(card.generated_query_patterns or [])
        new_patterns: list[str] = []
        if pool is not None:
            combined_pool = pool.oracle_queries + pool.endorsed_fingerprint_queries
            available_slots = max(0, MAX_PATTERNS_PER_ANCHOR - len(existing))
            if available_slots > 0 and combined_pool:
                candidate_grams = extractor.extract(combined_pool, n=available_slots * 4)
                existing_set = set(existing)
                for gram in candidate_grams:
                    if gram not in existing_set:
                        new_patterns.append(gram)
                        existing_set.add(gram)
                        if len(new_patterns) >= available_slots:
                            break

        if new_patterns:
            n_anchors_touched += 1
            n_patterns_added += len(new_patterns)

        merged_patterns = existing + new_patterns
        refined_cards.append(_card_to_dict(card, merged_patterns))

    # Empty-run guard: do not clobber an in-flight candidate.
    atlas_dir = Path(atlas.anchor_dir)
    candidate_path = atlas_dir / "anchor_cards_candidate.json"
    meta_path = atlas_dir / "anchor_cards_candidate.meta.json"

    summary: dict[str, Any] = {
        "wrote_candidate": False,
        "n_anchors_touched": n_anchors_touched,
        "n_patterns_added": n_patterns_added,
        "oracle_labels_read": oracle_labels_read,
        "fingerprints_read": fingerprints_read,
        "n_anchors_with_oracle_support": n_anchors_with_oracle_support,
        "n_anchors_with_endorsed_fingerprints": n_anchors_with_endorsed_fingerprints,
        "extractor": extractor_name,
        "support_semantics": SUPPORT_SEMANTICS,
        "atlas_version": atlas.version,
        "candidate_path": None,
        "reason": None,
    }

    if n_anchors_touched == 0 or n_patterns_added == 0:
        if n_anchors_with_oracle_support == 0:
            summary["reason"] = "no oracle-endorsed anchors (run 'psa label' first)"
        else:
            summary["reason"] = "extractor produced no new patterns (all duplicated existing)"
        logger.warning("Curation skipped write: %s", summary["reason"])
        return summary

    with open(candidate_path, "w") as f:
        json.dump(refined_cards, f, indent=2)

    meta = {
        "source": "production_signal",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "tenant_id": tenant_id,
        "atlas_version": atlas.version,
        "promoted": False,
        "promoted_at": None,
        "n_anchors_touched": n_anchors_touched,
        "n_patterns_added": n_patterns_added,
        "oracle_labels_read": oracle_labels_read,
        "fingerprints_read": fingerprints_read,
        "n_anchors_with_oracle_support": n_anchors_with_oracle_support,
        "n_anchors_with_endorsed_fingerprints": n_anchors_with_endorsed_fingerprints,
        "extractor": extractor_name,
        "support_semantics": SUPPORT_SEMANTICS,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    summary["wrote_candidate"] = True
    summary["candidate_path"] = str(candidate_path)
    return summary


def _card_to_dict(card: Any, merged_patterns: list[str]) -> dict:
    """Serialize an AnchorCard to the JSON dict shape used on disk.

    Uses the card's own to_dict() when available, otherwise constructs the
    expected shape field by field. The result's generated_query_patterns is
    overwritten with `merged_patterns`.
    """
    if hasattr(card, "to_dict"):
        d = card.to_dict()
    else:
        # Fallback: reconstruct from known fields (keeps the curator testable
        # with MagicMock cards that don't implement to_dict).
        d = {
            "anchor_id": card.anchor_id,
            "name": card.name,
            "meaning": card.meaning,
            "memory_types": list(getattr(card, "memory_types", [])),
            "include_terms": list(getattr(card, "include_terms", [])),
            "exclude_terms": list(getattr(card, "exclude_terms", [])),
            "prototype_examples": list(getattr(card, "prototype_examples", [])),
            "near_but_different": list(getattr(card, "near_but_different", [])),
            "centroid": list(getattr(card, "centroid", [])),
            "memory_count": getattr(card, "memory_count", 0),
            "is_novelty": getattr(card, "is_novelty", False),
            "status": getattr(card, "status", "active"),
            "metadata": dict(getattr(card, "metadata", {})),
            "generated_query_patterns": list(getattr(card, "generated_query_patterns", [])),
            "query_fingerprint": list(getattr(card, "query_fingerprint", [])),
        }
    d["generated_query_patterns"] = merged_patterns
    return d
