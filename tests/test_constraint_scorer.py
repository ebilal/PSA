"""
test_constraint_scorer.py — Tests for ConstraintScorer.
"""

from unittest.mock import MagicMock

from psa.constraint_scorer import ConstraintScorer
from psa.memory_scorer import ScoredMemory
from psa.query_frame import QueryFrame


def _make_scored_memory(
    memory_id,
    score,
    entities=None,
    speaker_role=None,
    stance=None,
    mentioned_at=None,
    memory_type="SEMANTIC",
    actor_entities=None,
):
    """Create a MagicMock memory with the right attributes, wrapped in ScoredMemory."""
    memory = MagicMock()
    memory.memory_object_id = memory_id
    memory.entities = entities or []
    memory.speaker_role = speaker_role
    memory.stance = stance
    memory.mentioned_at = mentioned_at
    memory.memory_type = MagicMock()
    memory.memory_type.name = memory_type
    memory.actor_entities = actor_entities or []
    return ScoredMemory(
        memory_object_id=memory_id,
        final_score=score,
        memory=memory,
    )


def test_entity_overlap_boosts_score():
    """Memories with entity overlap should score higher than those without."""
    frame = QueryFrame(entities=["GraphQL"])
    m1 = _make_scored_memory("m1", 0.6, entities=["GraphQL", "REST"])
    m2 = _make_scored_memory("m2", 0.6, entities=["Postgres"])

    scorer = ConstraintScorer()
    results = scorer.adjust_scores([m1, m2], frame)

    result_map = {r.memory_object_id: r.final_score for r in results}
    assert result_map["m1"] > result_map["m2"], (
        f"Expected m1 ({result_map['m1']:.4f}) > m2 ({result_map['m2']:.4f})"
    )


def test_type_match_boosts_score():
    """Memories matching the expected type (answer_target) should score higher."""
    frame = QueryFrame(answer_target="failure")
    m1 = _make_scored_memory("m1", 0.6, memory_type="FAILURE")
    m2 = _make_scored_memory("m2", 0.6, memory_type="SEMANTIC")

    scorer = ConstraintScorer()
    results = scorer.adjust_scores([m1, m2], frame)

    result_map = {r.memory_object_id: r.final_score for r in results}
    assert result_map["m1"] > result_map["m2"], (
        f"Expected m1 ({result_map['m1']:.4f}) > m2 ({result_map['m2']:.4f})"
    )


def test_no_frame_returns_unchanged_order():
    """When no QueryFrame is provided, memories are returned in original order."""
    m1 = _make_scored_memory("m1", 0.9)
    m2 = _make_scored_memory("m2", 0.7)
    m3 = _make_scored_memory("m3", 0.5)

    scorer = ConstraintScorer()
    results = scorer.adjust_scores([m1, m2, m3], None)

    assert [r.memory_object_id for r in results] == ["m1", "m2", "m3"]


def test_missing_facets_get_neutral_score():
    """Memories with empty entities should not be heavily penalized (neutral 0.5 weight)."""
    frame = QueryFrame(entities=["GraphQL"])
    m = _make_scored_memory("m1", 0.5, entities=[])

    scorer = ConstraintScorer()
    results = scorer.adjust_scores([m], frame)

    assert results[0].final_score >= 0.4, f"Expected score >= 0.4, got {results[0].final_score:.4f}"


def test_exact_phrase_boosts():
    from psa.constraint_scorer import ConstraintScorer

    frame = QueryFrame(quoted_terms=["JWT refresh tokens"])
    m1 = _make_scored_memory("m1", 0.5)
    m1.memory.body = "We use JWT refresh tokens stored in HttpOnly cookies"
    m2 = _make_scored_memory("m2", 0.5)
    m2.memory.body = "Database uses connection pooling"
    scorer = ConstraintScorer()
    adjusted = scorer.adjust_scores([m1, m2], frame)
    a1 = next(s for s in adjusted if s.memory_object_id == "m1")
    a2 = next(s for s in adjusted if s.memory_object_id == "m2")
    assert a1.final_score > a2.final_score


def test_contradiction_penalty():
    from psa.constraint_scorer import ConstraintScorer

    frame = QueryFrame(answer_target="fact")
    m1 = _make_scored_memory("m1", 0.5, stance="deprecated")
    m2 = _make_scored_memory("m2", 0.5, stance=None)
    scorer = ConstraintScorer()
    adjusted = scorer.adjust_scores([m1, m2], frame)
    a1 = next(s for s in adjusted if s.memory_object_id == "m1")
    a2 = next(s for s in adjusted if s.memory_object_id == "m2")
    assert a2.final_score > a1.final_score  # deprecated penalized for fact query
