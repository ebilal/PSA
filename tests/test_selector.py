"""Tests for psa.selector — AnchorSelector (cosine mode) and check_training_gates."""

from unittest.mock import MagicMock, patch

import pytest

from psa.anchor import AnchorCard
from psa.retriever import AnchorCandidate
from psa.selector import (
    AnchorSelector,
    SelectedAnchor,
    TrainingGateStatus,
    check_training_gates,
    _cosine_select,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_card(anchor_id: int, name: str = "test") -> AnchorCard:
    return AnchorCard(
        anchor_id=anchor_id,
        name=name,
        meaning=f"Anchor {name}",
        memory_types=["SEMANTIC"],
        include_terms=[],
        exclude_terms=[],
        prototype_examples=[],
        near_but_different=[],
        centroid=[0.0] * 4,
        memory_count=1,
        is_novelty=False,
    )


def _make_candidate(anchor_id: int, dense_score: float) -> AnchorCandidate:
    card = _make_card(anchor_id, f"card_{anchor_id}")
    return AnchorCandidate(
        anchor_id=anchor_id,
        card=card,
        dense_score=dense_score,
        bm25_score=0.5,
        rrf_score=dense_score,
        dense_rank=1,
        bm25_rank=1,
    )


# ── _cosine_select ────────────────────────────────────────────────────────────


def test_cosine_select_returns_top_k():
    candidates = [_make_candidate(i, float(10 - i)) for i in range(6)]
    selected = _cosine_select(query_vec=[], candidates=candidates, max_k=3, threshold=0.0)
    assert len(selected) == 3


def test_cosine_select_sorted_descending():
    candidates = [_make_candidate(i, float(i)) for i in range(4)]
    selected = _cosine_select(query_vec=[], candidates=candidates, max_k=4, threshold=0.0)
    scores = [s.selector_score for s in selected]
    assert scores == sorted(scores, reverse=True)


def test_cosine_select_always_includes_top_1():
    # Even with high threshold, first candidate is always included
    candidates = [_make_candidate(0, 0.1), _make_candidate(1, 0.05)]
    selected = _cosine_select(query_vec=[], candidates=candidates, max_k=4, threshold=0.9)
    assert len(selected) == 1
    assert selected[0].anchor_id == 0


def test_cosine_select_threshold_filters():
    candidates = [
        _make_candidate(0, 0.9),
        _make_candidate(1, 0.8),
        _make_candidate(2, 0.1),  # below threshold
    ]
    selected = _cosine_select(query_vec=[], candidates=candidates, max_k=4, threshold=0.5)
    anchor_ids = [s.anchor_id for s in selected]
    assert 0 in anchor_ids
    assert 1 in anchor_ids
    assert 2 not in anchor_ids


def test_cosine_select_mode_label():
    candidates = [_make_candidate(0, 1.0)]
    selected = _cosine_select(query_vec=[], candidates=candidates, max_k=1, threshold=0.0)
    assert selected[0].mode == "cosine"


def test_cosine_select_empty():
    selected = _cosine_select(query_vec=[], candidates=[], max_k=4, threshold=0.0)
    assert selected == []


# ── AnchorSelector (cosine mode) ──────────────────────────────────────────────


def test_selector_cosine_classmethod():
    sel = AnchorSelector.cosine(max_k=3)
    assert sel.mode == "cosine"
    assert sel.max_k == 3


def test_selector_selects_candidates():
    sel = AnchorSelector.cosine()
    candidates = [_make_candidate(i, float(5 - i)) for i in range(5)]
    selected = sel.select(query="auth", candidates=candidates)
    assert len(selected) >= 1
    assert len(selected) <= 4


def test_selector_returns_selected_anchor_type():
    sel = AnchorSelector.cosine()
    candidates = [_make_candidate(0, 0.9)]
    selected = sel.select(query="query", candidates=candidates)
    assert all(isinstance(s, SelectedAnchor) for s in selected)


def test_selector_invalid_mode():
    with pytest.raises(ValueError, match="Unknown selector mode"):
        AnchorSelector(mode="invalid")


def test_selector_trained_mode_missing_path(tmp_path):
    # No model_path → falls back to cosine
    sel = AnchorSelector(mode="trained", model_path=str(tmp_path / "nonexistent"))
    assert sel.mode == "cosine"


def test_selector_empty_candidates():
    sel = AnchorSelector.cosine()
    selected = sel.select(query="anything", candidates=[])
    assert selected == []


# ── check_training_gates ──────────────────────────────────────────────────────


def test_gates_all_met():
    status = check_training_gates(
        oracle_count=300,
        held_out_count=200,
        shortlist_recall_24=0.95,
        query_family_counts={
            "single_anchor": 60,
            "contrastive": 55,
            "compositional": 52,
            "bridge": 50,
            "experience": 70,
        },
    )
    assert status.gates_met is True
    assert status.blocking_reasons == []


def test_gates_oracle_count_too_low():
    status = check_training_gates(
        oracle_count=100,
        held_out_count=200,
        shortlist_recall_24=0.95,
        query_family_counts={"single_anchor": 60},
    )
    assert status.gates_met is False
    assert any("oracle_count" in r for r in status.blocking_reasons)


def test_gates_held_out_too_low():
    status = check_training_gates(
        oracle_count=300,
        held_out_count=50,
        shortlist_recall_24=0.95,
    )
    assert status.gates_met is False
    assert any("held_out_count" in r for r in status.blocking_reasons)


def test_gates_recall_too_low():
    status = check_training_gates(
        oracle_count=300,
        held_out_count=200,
        shortlist_recall_24=0.80,
    )
    assert status.gates_met is False
    assert any("recall@24" in r for r in status.blocking_reasons)


def test_gates_recall_not_measured():
    status = check_training_gates(
        oracle_count=300,
        held_out_count=200,
        shortlist_recall_24=None,
    )
    assert status.gates_met is False
    assert any("not yet measured" in r for r in status.blocking_reasons)


def test_gates_family_count_too_low():
    status = check_training_gates(
        oracle_count=300,
        held_out_count=200,
        shortlist_recall_24=0.95,
        query_family_counts={"single_anchor": 10},  # below 50
    )
    assert status.gates_met is False
    assert any("single_anchor" in r for r in status.blocking_reasons)


def test_gates_status_fields():
    status = check_training_gates(
        oracle_count=500,
        held_out_count=300,
        shortlist_recall_24=0.97,
        query_family_counts={"single_anchor": 100},
    )
    assert status.oracle_count == 500
    assert status.held_out_count == 300
    assert status.shortlist_recall_24 == 0.97
    assert "single_anchor" in status.query_family_counts
