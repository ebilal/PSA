"""
writer.py — persist a DecayReport as three files.

Output files (spec §4):
    anchor_cards_candidate.json           — refined cards (Branch 1 candidate slot)
    anchor_cards_candidate.meta.json      — summary (bounded size)
    anchor_cards_candidate.decay_report.json — full removed-patterns detail

Empty-run guard: when n_patterns_removed == 0, nothing is written
(matches Branch 3 curate behavior — protects any in-flight candidate).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any

logger = logging.getLogger("psa.advertisement.writer")


def write_decay_candidate(atlas_dir: str, report: Any) -> bool:
    """Write candidate + meta + detail. Returns True if files were written.

    Empty-run: when no patterns are removed, skips all writes (preserves
    any unrelated in-flight candidate from refine/curate).
    """
    if report.n_patterns_removed == 0:
        logger.info("Decay produced no removals — skipping candidate write.")
        return False

    os.makedirs(atlas_dir, exist_ok=True)

    cards_path = os.path.join(atlas_dir, "anchor_cards_candidate.json")
    meta_path = os.path.join(atlas_dir, "anchor_cards_candidate.meta.json")
    detail_path = os.path.join(atlas_dir, "anchor_cards_candidate.decay_report.json")

    # 1. Candidate cards.
    with open(cards_path, "w", encoding="utf-8") as f:
        json.dump(report.new_cards, f, indent=2)

    # 2. Summary meta (bounded — no per-pattern array).
    meta = {
        "source": "decay",
        "created_at": report.created_at,
        "tenant_id": report.tenant_id,
        "atlas_version": report.atlas_version,
        "promoted": False,
        "promoted_at": None,
        "decay_parameters": asdict(report.params),
        "origins": sorted(report.origins),
        "n_patterns_scanned": report.n_patterns_scanned,
        "n_patterns_removed": report.n_patterns_removed,
        "n_patterns_by_source_removed": report.n_patterns_by_source_removed,
        "n_anchors_touched": report.n_anchors_touched,
        "n_anchors_shielded": report.n_anchors_shielded,
        "n_patterns_shielded": report.n_patterns_shielded,
        "n_patterns_pinned_exempt": report.n_patterns_pinned_exempt,
        "n_patterns_backfilled_this_run": report.n_patterns_backfilled_this_run,
        "pruning_by_reason": report.pruning_by_reason,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 3. Detail report (full per-pattern provenance).
    detail = {
        "tenant_id": report.tenant_id,
        "atlas_version": report.atlas_version,
        "created_at": report.created_at,
        "removed_patterns": [asdict(r) for r in report.removed_patterns],
        "shielded_anchors": [asdict(s) for s in report.shielded_anchors],
    }
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2)

    return True
