"""
constraint_scorer.py — Rule-backed constraint scorer for PSA.

Adjusts Level 2 memory scores based on constraint satisfaction:
  - entity_overlap        (weight 0.20)
  - speaker_role_match    (weight 0.15)
  - actor_entity_match    (weight 0.15)
  - temporal_consistency  (weight 0.20)
  - stance_relevance      (weight 0.15)
  - type_match            (weight 0.15)

All features are tri-state: match=1.0, mismatch=0.0, unknown/unconstrained=0.5.
Final score: 0.70 * level2_score + 0.30 * constraint_boost
"""

from __future__ import annotations

from typing import List, Optional

from .memory_scorer import ScoredMemory
from .query_frame import QueryFrame

# ── Weights ───────────────────────────────────────────────────────────────────

_WEIGHTS = {
    "entity_overlap": 0.20,
    "speaker_role_match": 0.15,
    "actor_entity_match": 0.15,
    "temporal_consistency": 0.20,
    "stance_relevance": 0.15,
    "type_match": 0.15,
}

# Blend coefficients
_LEVEL2_WEIGHT = 0.70
_CONSTRAINT_WEIGHT = 0.30

# ── Stance alignment mappings ─────────────────────────────────────────────────

# answer_target -> set of memory stances that align
_STANCE_ALIGNMENT: dict[str, set[str]] = {
    "temporal_change": {"switched", "stopped"},
    "failure": {"failed"},
    "procedure": {"fixed", "prefers"},
    "preference": {"prefers"},
}

# answer_target -> expected memory type name
_TYPE_EXPECTATION: dict[str, str] = {
    "failure": "FAILURE",
    "procedure": "PROCEDURAL",
    "fact": "SEMANTIC",
    "preference": "SEMANTIC",
}


# ── Feature computations ──────────────────────────────────────────────────────


def _entity_overlap(frame_entities: List[str], memory_entities: List[str]) -> float:
    """Jaccard similarity, lowercased. Empty on either side -> 0.5 (unknown)."""
    if not frame_entities or not memory_entities:
        return 0.5
    f_set = {e.lower() for e in frame_entities}
    m_set = {e.lower() for e in memory_entities}
    intersection = len(f_set & m_set)
    union = len(f_set | m_set)
    return float(intersection / union) if union > 0 else 0.5


def _speaker_role_match(
    frame_role: Optional[str], memory_role: Optional[str]
) -> float:
    """Match frame.speaker_role_constraint == memory.speaker_role. Either None -> 0.5."""
    if frame_role is None or memory_role is None:
        return 0.5
    return 1.0 if frame_role == memory_role else 0.0


def _actor_entity_match(
    frame_entity: Optional[str], memory_actor_entities: List[str]
) -> float:
    """Check if frame.entity_constraint appears (lowercased) in memory.actor_entities."""
    if not frame_entity or not memory_actor_entities:
        return 0.5
    needle = frame_entity.lower()
    return 1.0 if needle in {e.lower() for e in memory_actor_entities} else 0.0


def _temporal_consistency(
    frame_time: Optional[str], memory_mentioned_at: Optional[str]
) -> float:
    """Substring match of frame.time_constraint in memory.mentioned_at.
    Either None -> 0.5. No match -> 0.3 (not definitively wrong)."""
    if frame_time is None or memory_mentioned_at is None:
        return 0.5
    return 1.0 if frame_time.lower() in memory_mentioned_at.lower() else 0.3


def _stance_relevance(answer_target: str, memory_stance: Optional[str]) -> float:
    """Alignment based on answer_target + memory stance. No stance -> 0.5."""
    if not memory_stance:
        return 0.5
    expected_stances = _STANCE_ALIGNMENT.get(answer_target)
    if expected_stances is None:
        return 0.5
    return 1.0 if memory_stance.lower() in expected_stances else 0.0


def _type_match(answer_target: str, memory_type_name: str) -> float:
    """Check expected memory type. No expectation -> 0.5. Mismatch -> 0.3."""
    expected = _TYPE_EXPECTATION.get(answer_target)
    if expected is None:
        return 0.5
    return 1.0 if memory_type_name == expected else 0.3


# ── ConstraintScorer ──────────────────────────────────────────────────────────


class ConstraintScorer:
    """
    Adjusts Level 2 ScoredMemory scores using rule-based constraint satisfaction.

    Usage::

        scorer = ConstraintScorer()
        adjusted = scorer.adjust_scores(scored_memories, query_frame)
    """

    def adjust_scores(
        self,
        scored_memories: List[ScoredMemory],
        query_frame: Optional[QueryFrame],
    ) -> List[ScoredMemory]:
        """
        Apply constraint-based adjustment to scored memories.

        When query_frame is None, returns memories in original order unchanged.
        Otherwise re-sorts by adjusted score descending.
        """
        if query_frame is None:
            return scored_memories

        adjusted: List[ScoredMemory] = []
        for sm in scored_memories:
            mem = sm.memory
            mem_entities = getattr(mem, "entities", []) or []
            mem_role = getattr(mem, "speaker_role", None)
            mem_actor_entities = getattr(mem, "actor_entities", []) or []
            mem_mentioned_at = getattr(mem, "mentioned_at", None)
            mem_stance = getattr(mem, "stance", None)
            mem_type_name = mem.memory_type.name if hasattr(mem.memory_type, "name") else str(mem.memory_type)

            features = {
                "entity_overlap": _entity_overlap(query_frame.entities, mem_entities),
                "speaker_role_match": _speaker_role_match(
                    query_frame.speaker_role_constraint, mem_role
                ),
                "actor_entity_match": _actor_entity_match(
                    query_frame.entity_constraint, mem_actor_entities
                ),
                "temporal_consistency": _temporal_consistency(
                    query_frame.time_constraint, mem_mentioned_at
                ),
                "stance_relevance": _stance_relevance(
                    query_frame.answer_target, mem_stance
                ),
                "type_match": _type_match(query_frame.answer_target, mem_type_name),
            }

            constraint_boost = sum(
                _WEIGHTS[k] * v for k, v in features.items()
            )

            new_score = _LEVEL2_WEIGHT * sm.final_score + _CONSTRAINT_WEIGHT * constraint_boost

            adjusted.append(
                ScoredMemory(
                    memory_object_id=sm.memory_object_id,
                    final_score=new_score,
                    memory=sm.memory,
                )
            )

        adjusted.sort(key=lambda r: r.final_score, reverse=True)
        return adjusted
