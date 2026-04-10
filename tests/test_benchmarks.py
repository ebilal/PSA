"""test_benchmarks.py — Unit tests for benchmark harness utilities."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


def _make_result_record(question_id="q1", answer_session_ids=None):
    return {
        "question_id": question_id,
        "question": "What did we decide about auth?",
        "context_text": "some context",
        "answer_generated": "JWT",
        "answer_gold": "JWT",
        "answer_session_ids": answer_session_ids or ["session_abc"],
        "tokens_used": 100,
        "token_budget": 6000,
        "selected_anchor_ids": [1, 2],
        "timing_ms": {"embed": 10, "retrieve": 20, "select": 5, "fetch": 3, "pack": 2, "total": 40},
    }


def test_oracle_label_calls_backtrack(tmp_path):
    """oracle_label() should call backtrack_gold_anchors for each record."""
    from psa.benchmarks.longmemeval import oracle_label

    results_path = str(tmp_path / "results.jsonl")
    with open(results_path, "w") as f:
        f.write(json.dumps(_make_result_record()) + "\n")

    mock_pipeline = MagicMock()
    mock_pipeline.atlas.version = 1
    mock_labeler = MagicMock()

    with patch("psa.benchmarks.longmemeval.PSAPipeline.from_tenant", return_value=mock_pipeline), \
         patch("psa.benchmarks.longmemeval.OracleLabeler", return_value=mock_labeler), \
         patch("psa.benchmarks.longmemeval.backtrack_gold_anchors", return_value=[5]) as mock_bt:
        # Need to patch the oracle_labels_path to tmp_path to avoid writing to home dir
        with patch("psa.benchmarks.longmemeval._oracle_labels_path", return_value=str(tmp_path / "labels.jsonl")):
            oracle_label(results_path, tenant_id="test_tenant")

    mock_bt.assert_called_once_with(
        answer_session_ids=["session_abc"],
        store=mock_pipeline.store,
        atlas=mock_pipeline.atlas,
        tenant_id="test_tenant",
    )
    mock_labeler.label.assert_called_once()
    call_kwargs = mock_labeler.label.call_args[1]
    assert call_kwargs["gold_anchor_ids"] == [5]
