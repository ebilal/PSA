"""test_oracle_labeler.py — Tests for oracle labeling utilities."""

from unittest.mock import MagicMock

from psa.memory_object import MemoryObject
from psa.training.oracle_labeler import backtrack_gold_anchors


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
