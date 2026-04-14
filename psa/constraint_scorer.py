"""
constraint_scorer.py — Rule-backed constraint scorer for PSA.

Adjusts Level 2 memory scores based on constraint satisfaction:
  - entity_overlap        (weight 0.17)
  - speaker_role_match    (weight 0.12)
  - actor_entity_match    (weight 0.12)
  - temporal_consistency  (weight 0.17)
  - stance_relevance      (weight 0.10)
  - type_match            (weight 0.12)
  - exact_phrase_support  (weight 0.10)
  - contradiction_penalty (weight 0.10)

All features are tri-state: match=1.0, mismatch=0.0, unknown/unconstrained=0.5.
Final score: 0.70 * level2_score + 0.30 * constraint_boost
"""

from __future__ import annotations

from typing import List, Optional

from .memory_scorer import ScoredMemory
from .query_frame import QueryFrame

# ── Weights ───────────────────────────────────────────────────────────────────

W_ENTITY = 0.17
W_SPEAKER_ROLE = 0.12
W_ACTOR_ENTITY = 0.12
W_TEMPORAL = 0.17
W_STANCE = 0.10
W_TYPE_MATCH = 0.12
W_EXACT_PHRASE = 0.10
W_CONTRADICTION = 0.10

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


def _speaker_role_match(frame_role: Optional[str], memory_role: Optional[str]) -> float:
    """Match frame.speaker_role_constraint == memory.speaker_role. Either None -> 0.5."""
    if frame_role is None or memory_role is None:
        return 0.5
    return 1.0 if frame_role == memory_role else 0.0


def _actor_entity_match(frame_entity: Optional[str], memory_actor_entities: List[str]) -> float:
    """Check if frame.entity_constraint appears (lowercased) in memory.actor_entities."""
    if not frame_entity or not memory_actor_entities:
        return 0.5
    needle = frame_entity.lower()
    return 1.0 if needle in {e.lower() for e in memory_actor_entities} else 0.0


def _temporal_consistency(frame_time: Optional[str], memory_mentioned_at: Optional[str]) -> float:
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


def _exact_phrase_support(quoted_terms: List[str], memory_body: str) -> float:
    """1.0 if any quoted term appears verbatim in memory body, 0.0 if not, 0.5 if no quoted terms."""
    if not quoted_terms:
        return 0.5
    body_lower = (memory_body or "").lower()
    for term in quoted_terms:
        if term.lower() in body_lower:
            return 1.0
    return 0.0


def _contradiction_penalty(answer_target: str, memory_stance: Optional[str]) -> float:
    """Penalize memories whose stance contradicts the query intent.
    E.g., query asks about current approach but memory says 'deprecated'."""
    if not memory_stance:
        return 0.5  # neutral
    # Current/fact queries penalized by deprecated/stopped memories
    if answer_target in ("fact", "preference") and memory_stance in ("deprecated", "stopped"):
        return 0.2
    # Failure queries penalized by "fixed" memories (the fix, not the failure)
    if answer_target == "failure" and memory_stance == "fixed":
        return 0.3
    return 0.5  # neutral


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
            mem_body = getattr(mem, "body", "") or ""
            mem_type_name = (
                mem.memory_type.name if hasattr(mem.memory_type, "name") else str(mem.memory_type)
            )

            entity = _entity_overlap(query_frame.entities, mem_entities)
            speaker = _speaker_role_match(query_frame.speaker_role_constraint, mem_role)
            actor = _actor_entity_match(query_frame.entity_constraint, mem_actor_entities)
            temporal = _temporal_consistency(query_frame.time_constraint, mem_mentioned_at)
            stance = _stance_relevance(query_frame.answer_target, mem_stance)
            type_match = _type_match(query_frame.answer_target, mem_type_name)
            exact = _exact_phrase_support(query_frame.quoted_terms, mem_body)
            contra = _contradiction_penalty(query_frame.answer_target, mem_stance)

            constraint_boost = (
                W_ENTITY * entity
                + W_SPEAKER_ROLE * speaker
                + W_ACTOR_ENTITY * actor
                + W_TEMPORAL * temporal
                + W_STANCE * stance
                + W_TYPE_MATCH * type_match
                + W_EXACT_PHRASE * exact
                + W_CONTRADICTION * contra
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
