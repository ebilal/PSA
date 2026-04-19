"""test_oracle_labeler.py — Tests for oracle labeling utilities."""

from pathlib import Path
from unittest.mock import MagicMock

from psa.anchor import AnchorCard
from psa.memory_object import MemoryObject
from psa.pipeline import PSAResult, PackedContext, QueryTiming
from psa.selector import SelectedAnchor
from psa.training.oracle_labeler import OracleLabeler, backtrack_gold_anchors


def _make_memory(anchor_id):
    mo = MagicMock(spec=MemoryObject)
    mo.primary_anchor_id = anchor_id
    return mo


def test_backtrack_gold_anchors_returns_anchor_ids():
    """backtrack_gold_anchors maps session_ids → memory objects → anchor IDs."""
    store = MagicMock()
    store.get_by_source_session.return_value = [
        _make_memory(anchor_id=5),
        _make_memory(anchor_id=7),
    ]

    result = backtrack_gold_anchors(
        answer_session_ids=["session_abc"],
        store=store,
        atlas=MagicMock(),
        tenant_id="test",
    )
    assert set(result) == {5, 7}
    store.get_by_source_session.assert_called_once_with("session_abc", tenant_id="test")


def test_backtrack_gold_anchors_deduplicates():
    """Two sessions pointing to memories in the same anchor yield one anchor ID."""
    store = MagicMock()
    store.get_by_source_session.side_effect = [
        [_make_memory(anchor_id=3)],
        [_make_memory(anchor_id=3)],
    ]
    result = backtrack_gold_anchors(
        answer_session_ids=["session_a", "session_b"],
        store=store,
        atlas=MagicMock(),
        tenant_id="test",
    )
    assert result.count(3) == 1


def test_backtrack_gold_anchors_skips_unassigned_memories():
    """Memories with primary_anchor_id None or -1 are ignored."""
    mo_no_anchor = MagicMock(spec=MemoryObject)
    mo_no_anchor.primary_anchor_id = None

    mo_neg = MagicMock(spec=MemoryObject)
    mo_neg.primary_anchor_id = -1

    store = MagicMock()
    store.get_by_source_session.return_value = [mo_no_anchor, mo_neg]

    result = backtrack_gold_anchors(
        answer_session_ids=["session_x"],
        store=store,
        atlas=MagicMock(),
        tenant_id="test",
    )
    assert result == []


def test_backtrack_gold_anchors_empty_sessions():
    result = backtrack_gold_anchors(
        answer_session_ids=[],
        store=MagicMock(),
        atlas=MagicMock(),
        tenant_id="test",
    )
    assert result == []


def test_label_uses_selected_anchors_when_candidates_empty(tmp_path, monkeypatch):
    pipeline = MagicMock()
    pipeline.atlas.version = 19
    card = AnchorCard(
        anchor_id=11,
        name="Procedures",
        meaning="How to carry out repeatable tasks.",
        memory_types=["procedural"],
        include_terms=["workflow"],
        exclude_terms=[],
        prototype_examples=[],
        near_but_different=[],
        centroid=[0.1, 0.2],
    )
    pipeline.atlas._card_map = {11: card}
    pipeline.packed_context_for_anchors.return_value = MagicMock(text="ctx", token_count=42)
    pipeline.query.return_value = PSAResult(
        query="How do I deploy?",
        packed_context=PackedContext(
            query="How do I deploy?",
            text="",
            token_count=0,
            memory_ids=[],
            sections=[],
            untyped_count=0,
        ),
        selected_anchors=[
            SelectedAnchor(anchor_id=11, selector_score=0.9, mode="full_atlas", candidate=None)
        ],
        candidates=[],
        timing=QueryTiming(),
        tenant_id="default",
        psa_mode="primary",
        selection_mode="full_atlas",
    )

    monkeypatch.setattr(
        "psa.training.oracle_labeler._call_qwen_proxy_batch",
        lambda **kwargs: [
            {
                "support_coverage": 0.8,
                "procedural_utility": 0.7,
                "noise_penalty": 0.1,
                "token_cost": 0.2,
            }
        ],
    )
    monkeypatch.setattr("psa.training.oracle_labeler._qwen_task_success_batch", lambda *_: [0.9])

    out = Path(tmp_path) / "labels.jsonl"
    labeler = OracleLabeler(pipeline=pipeline, output_path=str(out))
    label = labeler.label(query_id="q1", query="How do I deploy?")

    assert label.candidate_anchor_ids == [11]
    assert label.winning_oracle_set == [11]
    assert out.exists()
