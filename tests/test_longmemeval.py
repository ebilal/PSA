"""
Unit tests for psa/benchmarks/longmemeval.py
"""

import hashlib
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from psa.benchmarks.longmemeval import (
    _f1_score,
    _llm_judge,
    _make_oracle_label,
    _normalize_text,
    _write_session_jsonl,
    score,
)


# ── _normalize_text ────────────────────────────────────────────────────────────


def test_normalize_text_lowercases():
    result = _normalize_text("Hello World")
    assert result == ["hello", "world"]


def test_normalize_text_strips_punctuation():
    result = _normalize_text("Hello, World! It's great.")
    assert "hello" in result
    assert "world" in result
    # punctuation removed; apostrophe/period should be gone
    for token in result:
        assert "," not in token
        assert "!" not in token
        assert "." not in token


def test_normalize_text_empty_string():
    result = _normalize_text("")
    assert result == []


def test_normalize_text_only_punctuation():
    result = _normalize_text("!!! ???")
    # All non-word chars replaced by spaces → empty tokens stripped
    assert all(t.strip() != "" for t in result)


def test_normalize_text_mixed_case_and_punctuation():
    result = _normalize_text("Python3 is GREAT!")
    assert "python3" in result
    assert "is" in result
    assert "great" in result


# ── _f1_score ──────────────────────────────────────────────────────────────────


def test_f1_score_perfect_match():
    score_val = _f1_score("the cat sat on the mat", "the cat sat on the mat")
    assert score_val == pytest.approx(1.0)


def test_f1_score_no_overlap():
    score_val = _f1_score("apple banana cherry", "dog elephant fox")
    assert score_val == pytest.approx(0.0)


def test_f1_score_partial_overlap():
    # gold: {apple, banana, cherry}, pred: {apple, banana, grape}
    # common: {apple, banana} = 2
    # precision = 2/3, recall = 2/3
    # F1 = 2*(2/3)*(2/3) / (2/3 + 2/3) = (8/9) / (4/3) = 2/3
    score_val = _f1_score("apple banana cherry", "apple banana grape")
    assert 0.0 < score_val < 1.0
    assert score_val == pytest.approx(2 / 3, rel=1e-5)


def test_f1_score_empty_gold():
    score_val = _f1_score("", "some answer here")
    assert score_val == pytest.approx(0.0)


def test_f1_score_empty_pred():
    score_val = _f1_score("some gold answer", "")
    assert score_val == pytest.approx(0.0)


def test_f1_score_both_empty():
    score_val = _f1_score("", "")
    assert score_val == pytest.approx(0.0)


def test_f1_score_subset():
    # gold={cat,dog,elephant}(3), pred={cat,dog}(2), common=2
    # precision=2/2=1.0, recall=2/3 → F1=2*1*(2/3)/(1+2/3)=0.8
    score_val = _f1_score("cat dog elephant", "cat dog")
    assert score_val == pytest.approx(0.8, rel=1e-5)


# ── _llm_judge ─────────────────────────────────────────────────────────────────


def test_llm_judge_pass_response():
    mock_llm = MagicMock(return_value="PASS")
    result = _llm_judge("What is 2+2?", "4", "4", mock_llm)
    assert result == pytest.approx(1.0)
    mock_llm.assert_called_once()


def test_llm_judge_fail_response():
    mock_llm = MagicMock(return_value="FAIL")
    result = _llm_judge("What is 2+2?", "4", "5", mock_llm)
    assert result == pytest.approx(0.0)


def test_llm_judge_unexpected_response():
    mock_llm = MagicMock(return_value="MAYBE")
    result = _llm_judge("What is 2+2?", "4", "perhaps 4", mock_llm)
    assert result is None


def test_llm_judge_exception_returns_none():
    mock_llm = MagicMock(side_effect=RuntimeError("LLM unavailable"))
    result = _llm_judge("What is 2+2?", "4", "4", mock_llm)
    assert result is None


def test_llm_judge_pass_case_insensitive():
    # response has lowercase "pass" which becomes "PASS" after .upper()
    mock_llm = MagicMock(return_value="  pass  ")
    result = _llm_judge("Q", "A", "A", mock_llm)
    assert result == pytest.approx(1.0)


def test_llm_judge_pass_embedded_in_response():
    # "PASS" contained within longer response still returns 1.0
    mock_llm = MagicMock(return_value="PASS - the answer is correct")
    result = _llm_judge("Q", "A", "A", mock_llm)
    assert result == pytest.approx(1.0)


def test_llm_judge_sends_correct_messages():
    mock_llm = MagicMock(return_value="PASS")
    _llm_judge("My question?", "Gold answer", "Generated answer", mock_llm)
    args, kwargs = mock_llm.call_args
    messages = args[0]
    assert isinstance(messages, list)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert "My question?" in content
    assert "Gold answer" in content
    assert "Generated answer" in content
    assert kwargs.get("max_tokens") == 10
    assert kwargs.get("json_mode") is False


# ── _make_oracle_label ─────────────────────────────────────────────────────────


def _sample_record():
    return {
        "question_id": "q_0001",
        "question": "Where did Alice go on vacation?",
        "answer_generated": "She went to Paris.",
        "answer_gold": "Alice went to London.",
        "selected_anchor_ids": [1, 2],
        "context_text": "Some context here.",
        "tokens_used": 100,
        "token_budget": 6000,
        "timing_ms": {},
    }


def test_make_oracle_label_fields():
    record = _sample_record()
    label = _make_oracle_label(record, f1=0.1)

    assert label["query"] == record["question"]
    assert label["atlas_version"] == -1
    assert label["runtime_model_id"] == "longmemeval"
    assert label["candidate_anchor_ids"] == [1, 2]
    assert label["winning_oracle_score"] == pytest.approx(0.1)
    assert label["is_high_complexity"] is False
    assert label["metadata"]["source"] == "longmemeval"
    assert label["metadata"]["question_id"] == "q_0001"
    assert label["metadata"]["exact_f1"] == pytest.approx(0.1)
    assert "labeled_at" in label


def test_make_oracle_label_query_id_is_stable():
    record = _sample_record()
    label1 = _make_oracle_label(record, f1=0.1)
    label2 = _make_oracle_label(record, f1=0.2)
    # query_id should be the same for the same question regardless of f1
    assert label1["query_id"] == label2["query_id"]


def test_make_oracle_label_query_id_format():
    record = _sample_record()
    label = _make_oracle_label(record, f1=0.1)
    q_hash = hashlib.md5(
        record["question"].encode(), usedforsecurity=False
    ).hexdigest()[:8]
    assert label["query_id"] == f"lme_{q_hash}"


def test_make_oracle_label_different_questions_different_ids():
    record_a = _sample_record()
    record_b = _sample_record()
    record_b["question"] = "What did Bob eat for breakfast?"
    label_a = _make_oracle_label(record_a, f1=0.0)
    label_b = _make_oracle_label(record_b, f1=0.0)
    assert label_a["query_id"] != label_b["query_id"]


def test_make_oracle_label_missing_selected_anchors():
    record = _sample_record()
    del record["selected_anchor_ids"]
    label = _make_oracle_label(record, f1=0.0)
    assert label["candidate_anchor_ids"] == []


# ── _write_session_jsonl ───────────────────────────────────────────────────────


def test_write_session_jsonl_creates_file(tmp_path):
    path = str(tmp_path / "session.jsonl")
    messages = [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hi!"},
    ]
    _write_session_jsonl(path, "sess_001", messages)
    assert os.path.exists(path)


def test_write_session_jsonl_correct_records(tmp_path):
    path = str(tmp_path / "session.jsonl")
    messages = [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hi!"},
    ]
    _write_session_jsonl(path, "sess_001", messages)

    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    rec1 = json.loads(lines[1])

    assert rec0["type"] == "message"
    assert rec0["role"] == "user"
    assert rec0["content"] == "Hello there"
    assert rec0["session_id"] == "sess_001"

    assert rec1["role"] == "assistant"
    assert rec1["content"] == "Hi!"
    assert rec1["session_id"] == "sess_001"


def test_write_session_jsonl_empty_messages(tmp_path):
    path = str(tmp_path / "empty_session.jsonl")
    _write_session_jsonl(path, "sess_empty", [])
    with open(path) as f:
        content = f.read()
    assert content == ""


def test_write_session_jsonl_missing_fields(tmp_path):
    """Messages with missing role/content should use defaults."""
    path = str(tmp_path / "session.jsonl")
    messages = [{}]
    _write_session_jsonl(path, "sess_x", messages)
    with open(path) as f:
        rec = json.loads(f.read().strip())
    assert rec["role"] == "user"
    assert rec["content"] == ""


# ── score() ───────────────────────────────────────────────────────────────────


def _write_results_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_result_record(question_id="q_001", question="Q?", gold="gold answer", generated="gold answer", anchors=None):
    return {
        "question_id": question_id,
        "question": question,
        "answer_gold": gold,
        "answer_generated": generated,
        "selected_anchor_ids": anchors or [],
        "context_text": "",
        "tokens_used": 10,
        "token_budget": 6000,
        "timing_ms": {},
    }


def test_score_exact_f1_perfect_match(tmp_path):
    results_file = str(tmp_path / "results.jsonl")
    oracle_file = str(tmp_path / "oracle_labels.jsonl")
    record = _make_result_record(gold="the answer is yes", generated="the answer is yes")
    _write_results_jsonl(results_file, [record])

    with patch("psa.benchmarks.longmemeval._oracle_labels_path", return_value=oracle_file):
        result = score(results_file, method="exact", tenant_id="test_tenant")

    assert result["exact_f1"] == pytest.approx(1.0)
    assert result["n_questions"] == 1
    # Perfect match → f1=1.0 >= 0.3, so NO oracle label should be written
    assert result["oracle_labels_written"] == 0


def test_score_writes_oracle_labels_for_failures(tmp_path):
    results_file = str(tmp_path / "results.jsonl")
    oracle_file = str(tmp_path / "oracle_labels.jsonl")

    records = [
        _make_result_record(question_id="q_001", gold="correct answer", generated="correct answer"),  # perfect match
        _make_result_record(question_id="q_002", gold="correct answer", generated="totally wrong"),   # failure
        _make_result_record(question_id="q_003", gold="correct answer", generated="totally different"),  # failure
    ]
    _write_results_jsonl(results_file, records)

    with patch("psa.benchmarks.longmemeval._oracle_labels_path", return_value=oracle_file):
        result = score(results_file, method="exact", tenant_id="test_tenant")

    assert result["oracle_labels_written"] == 2
    assert os.path.exists(oracle_file)

    with open(oracle_file) as f:
        labels = [json.loads(l) for l in f if l.strip()]
    assert len(labels) == 2


def test_score_no_oracle_labels_for_perfect_match(tmp_path):
    results_file = str(tmp_path / "results.jsonl")
    oracle_file = str(tmp_path / "oracle_labels.jsonl")

    records = [
        _make_result_record(gold="answer one", generated="answer one"),
        _make_result_record(gold="answer two", generated="answer two"),
    ]
    _write_results_jsonl(results_file, records)

    with patch("psa.benchmarks.longmemeval._oracle_labels_path", return_value=oracle_file):
        result = score(results_file, method="exact", tenant_id="test_tenant")

    assert result["oracle_labels_written"] == 0
    # File may or may not exist; if it does, it should be empty
    if os.path.exists(oracle_file):
        with open(oracle_file) as f:
            content = f.read().strip()
        assert content == ""


def test_score_exact_f1_computation(tmp_path):
    results_file = str(tmp_path / "results.jsonl")
    oracle_file = str(tmp_path / "oracle_labels.jsonl")

    # Two records: one perfect (f1=1.0), one no overlap (f1=0.0)
    records = [
        _make_result_record(gold="apple banana", generated="apple banana"),  # f1=1.0
        _make_result_record(gold="apple banana", generated="cherry dog"),    # f1=0.0
    ]
    _write_results_jsonl(results_file, records)

    with patch("psa.benchmarks.longmemeval._oracle_labels_path", return_value=oracle_file):
        result = score(results_file, method="exact", tenant_id="test_tenant")

    assert result["exact_f1"] == pytest.approx(0.5)
    assert result["n_questions"] == 2


def test_score_llm_method_uses_call_llm(tmp_path):
    results_file = str(tmp_path / "results.jsonl")
    oracle_file = str(tmp_path / "oracle_labels.jsonl")

    record = _make_result_record(gold="some answer", generated="some answer")
    _write_results_jsonl(results_file, [record])

    # call_llm is imported inside score() via `from ..llm import call_llm`,
    # so we must patch it at the source module.
    mock_llm = MagicMock(return_value="PASS")
    with patch("psa.benchmarks.longmemeval._oracle_labels_path", return_value=oracle_file):
        with patch("psa.llm.call_llm", mock_llm):
            result = score(results_file, method="llm", tenant_id="test_tenant")

    assert "llm_score" in result
    assert result["llm_score"] == pytest.approx(1.0)
    mock_llm.assert_called_once()


def test_score_both_method_returns_both_scores(tmp_path):
    results_file = str(tmp_path / "results.jsonl")
    oracle_file = str(tmp_path / "oracle_labels.jsonl")

    record = _make_result_record(gold="correct", generated="correct")
    _write_results_jsonl(results_file, [record])

    mock_llm = MagicMock(return_value="PASS")
    with patch("psa.benchmarks.longmemeval._oracle_labels_path", return_value=oracle_file):
        with patch("psa.llm.call_llm", mock_llm):
            result = score(results_file, method="both", tenant_id="test_tenant")

    assert "exact_f1" in result
    assert "llm_score" in result


def test_score_empty_file_raises(tmp_path):
    results_file = str(tmp_path / "empty.jsonl")
    with open(results_file, "w") as f:
        f.write("")

    with pytest.raises(ValueError, match="No records found"):
        score(results_file, method="exact", tenant_id="test_tenant")


def test_score_oracle_labels_path_patched(tmp_path):
    """Ensure patching _oracle_labels_path prevents writing to real ~/.psa."""
    results_file = str(tmp_path / "results.jsonl")
    oracle_file = str(tmp_path / "oracle_labels.jsonl")

    record = _make_result_record(gold="a", generated="z")  # failure
    _write_results_jsonl(results_file, [record])

    with patch("psa.benchmarks.longmemeval._oracle_labels_path", return_value=oracle_file) as mock_path_fn:
        result = score(results_file, method="exact", tenant_id="test_tenant")

    mock_path_fn.assert_called_once_with("test_tenant")
    assert result["oracle_labels_path"] == oracle_file
