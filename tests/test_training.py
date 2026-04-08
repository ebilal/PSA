"""
Tests for psa.training — unit tests only, no actual model training.

Covers:
  - oracle_labeler: score formulas, OracleLabel serialization
  - data_generator: query family inference, data mix ratios, adversarial rewrites
  - train_selector: SelectorVersion serialization, training gate check
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from psa.training.oracle_labeler import (
    CandidateSetScore,
    OracleLabel,
    W_NOISE,
    W_PROCEDURAL,
    W_SUPPORT,
    W_TASK_SUCCESS,
    W_TOKEN_COST,
    oracle_score,
    score_noise_penalty,
    score_support_coverage,
    score_token_cost,
)
from psa.training.data_generator import (
    DataGenerator,
    MIX_ADVERSARIAL,
    MIX_HARD_NEG,
    MIX_SYNTHETIC,
    TrainingExample,
    _ADVERSARIAL_TRANSFORMS,
)
from psa.training.train_selector import (
    SelectorVersion,
    _load_training_data,
    _split_by_example_type,
)


# ── oracle_score formulas ─────────────────────────────────────────────────────


def test_oracle_score_all_perfect():
    score = oracle_score(
        support_coverage=1.0,
        task_success=1.0,
        procedural_utility=1.0,
        noise_penalty=0.0,
        token_cost=0.0,
    )
    expected = W_SUPPORT + W_TASK_SUCCESS + W_PROCEDURAL
    assert abs(score - expected) < 1e-9


def test_oracle_score_all_zero():
    score = oracle_score(
        support_coverage=0.0,
        task_success=0.0,
        procedural_utility=0.0,
        noise_penalty=0.0,
        token_cost=0.0,
    )
    assert score == 0.0


def test_oracle_score_weights():
    # W_NOISE and W_TOKEN_COST are negative
    assert W_NOISE < 0
    assert W_TOKEN_COST < 0
    assert W_SUPPORT > 0
    assert W_TASK_SUCCESS > 0
    assert W_PROCEDURAL > 0


def test_oracle_score_penalizes_noise():
    score_no_noise = oracle_score(1.0, 1.0, 1.0, 0.0, 0.0)
    score_with_noise = oracle_score(1.0, 1.0, 1.0, 0.5, 0.0)
    assert score_no_noise > score_with_noise


def test_oracle_score_penalizes_token_cost():
    score_cheap = oracle_score(1.0, 1.0, 1.0, 0.0, 0.0)
    score_expensive = oracle_score(1.0, 1.0, 1.0, 0.0, 0.5)
    assert score_cheap > score_expensive


# ── score_support_coverage ────────────────────────────────────────────────────


def test_support_coverage_full():
    score = score_support_coverage([1, 2, 3], [1, 2, 3])
    assert score == 1.0


def test_support_coverage_partial():
    score = score_support_coverage([1, 2], [1, 2, 3])
    assert abs(score - 2 / 3) < 1e-9


def test_support_coverage_no_gold():
    assert score_support_coverage([1, 2], []) == 0.0


def test_support_coverage_no_overlap():
    assert score_support_coverage([4, 5], [1, 2, 3]) == 0.0


# ── score_noise_penalty ────────────────────────────────────────────────────────


def test_noise_penalty_zero():
    # all selected are oracle
    assert score_noise_penalty([1, 2], [1, 2, 3]) == 0.0


def test_noise_penalty_full():
    # none selected are oracle
    score = score_noise_penalty([4, 5], [1, 2])
    assert score == 1.0


def test_noise_penalty_partial():
    score = score_noise_penalty([1, 4], [1, 2, 3])
    assert abs(score - 0.5) < 1e-9


def test_noise_penalty_empty():
    assert score_noise_penalty([], [1, 2]) == 0.0


# ── score_token_cost ──────────────────────────────────────────────────────────


def test_token_cost_at_budget():
    assert score_token_cost(6000, budget=6000) == 1.0


def test_token_cost_below_budget():
    score = score_token_cost(3000, budget=6000)
    assert score == 0.5


def test_token_cost_exceeds_budget():
    score = score_token_cost(12000, budget=6000)
    assert score == 1.0  # capped at 1.0


def test_token_cost_zero():
    assert score_token_cost(0, budget=6000) == 0.0


# ── OracleLabel serialization ─────────────────────────────────────────────────


def _make_oracle_label():
    cs = CandidateSetScore(
        anchor_ids=[1, 2],
        support_coverage=0.9,
        procedural_utility=0.5,
        noise_penalty=0.1,
        token_cost=0.4,
        task_success=0.8,
        oracle_score=0.7,
        packed_tokens=1200,
    )
    return OracleLabel(
        query_id="q1",
        query="How does auth work?",
        atlas_version=1,
        runtime_model_id="claude-haiku-4-5-20251001",
        candidate_anchor_ids=[1, 2, 3],
        all_sets=[cs],
        winning_oracle_set=[1, 2],
        winning_oracle_score=0.7,
        labeled_at="2026-01-01T00:00:00+00:00",
        is_high_complexity=False,
    )


def test_oracle_label_round_trip():
    label = _make_oracle_label()
    d = label.to_dict()
    restored = OracleLabel.from_dict(d)
    assert restored.query_id == label.query_id
    assert restored.winning_oracle_set == label.winning_oracle_set
    assert len(restored.all_sets) == 1
    assert restored.all_sets[0].oracle_score == pytest.approx(0.7)


def test_oracle_label_to_dict_keys():
    label = _make_oracle_label()
    d = label.to_dict()
    assert "query_id" in d
    assert "winning_oracle_set" in d
    assert "all_sets" in d
    assert "labeled_at" in d


# ── DataGenerator ─────────────────────────────────────────────────────────────


def _make_labels_file(tmp_path, n=10):
    """Write n oracle labels to a JSONL file."""
    path = tmp_path / "labels.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            label = {
                "query_id": f"q{i}",
                "query": f"How does feature {i} work?",
                "winning_oracle_set": [i, i + 1],
                "candidate_anchor_ids": [i, i + 1, i + 2, i + 3],
                "is_high_complexity": False,
            }
            f.write(json.dumps(label) + "\n")
    return str(path)


def _make_anchor_cards(n=20):
    """Simple dict of anchor_id → card text."""
    return {i: f"Anchor card {i}: text about topic {i}" for i in range(n)}


def test_data_generator_loads_labels(tmp_path):
    path = _make_labels_file(tmp_path, n=5)
    cards = _make_anchor_cards()
    gen = DataGenerator(oracle_labels_path=path, anchor_cards=cards)
    assert len(gen._labels) == 5


def test_data_generator_missing_file():
    gen = DataGenerator(
        oracle_labels_path="/nonexistent/path.jsonl",
        anchor_cards=_make_anchor_cards(),
    )
    assert gen._labels == []


def test_data_generator_generate_returns_count(tmp_path):
    path = _make_labels_file(tmp_path, n=10)
    cards = _make_anchor_cards()
    gen = DataGenerator(oracle_labels_path=path, anchor_cards=cards)
    out = tmp_path / "training.jsonl"
    count = gen.generate(str(out), n_examples=50)
    assert count == 50
    assert out.exists()


def test_data_generator_generate_no_labels(tmp_path):
    gen = DataGenerator(
        oracle_labels_path="/nonexistent/path.jsonl",
        anchor_cards=_make_anchor_cards(),
    )
    out = tmp_path / "training.jsonl"
    count = gen.generate(str(out), n_examples=50)
    assert count == 0


def test_data_generator_output_format(tmp_path):
    path = _make_labels_file(tmp_path, n=10)
    cards = _make_anchor_cards()
    gen = DataGenerator(oracle_labels_path=path, anchor_cards=cards)
    out = tmp_path / "training.jsonl"
    gen.generate(str(out), n_examples=20)

    with open(out) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 20

    for ex in lines:
        assert "query" in ex
        assert "anchor_card" in ex
        assert "label" in ex
        assert ex["label"] in (0, 1)
        assert "example_type" in ex
        assert "query_family" in ex


def test_data_generator_mix_ratios(tmp_path):
    """Verify approximate mix: 60% synthetic, 20% hard-neg, 20% adversarial."""
    path = _make_labels_file(tmp_path, n=15)
    cards = _make_anchor_cards()
    gen = DataGenerator(oracle_labels_path=path, anchor_cards=cards, seed=42)
    out = tmp_path / "training.jsonl"
    gen.generate(str(out), n_examples=100)

    with open(out) as f:
        examples = [json.loads(l) for l in f if l.strip()]

    by_type = {}
    for ex in examples:
        t = ex["example_type"]
        by_type[t] = by_type.get(t, 0) + 1

    n_synthetic = by_type.get("positive", 0) + by_type.get("easy_negative", 0)
    n_hard_neg = by_type.get("hard_negative", 0)
    n_adversarial = by_type.get("adversarial", 0)

    n_adversarial_expected = 100 - int(100 * MIX_SYNTHETIC) - int(100 * MIX_HARD_NEG)
    assert abs(n_synthetic - int(100 * MIX_SYNTHETIC)) <= 5
    assert abs(n_hard_neg - int(100 * MIX_HARD_NEG)) <= 5
    assert abs(n_adversarial - n_adversarial_expected) <= 5


def test_data_generator_adversarial_rewrites(tmp_path):
    """Adversarial examples should be rewrites of positives (same anchor_id)."""
    path = _make_labels_file(tmp_path, n=10)
    cards = _make_anchor_cards()
    gen = DataGenerator(oracle_labels_path=path, anchor_cards=cards, seed=1)
    out = tmp_path / "training.jsonl"
    gen.generate(str(out), n_examples=50)

    with open(out) as f:
        examples = [json.loads(l) for l in f if l.strip()]

    adversarials = [ex for ex in examples if ex["example_type"] == "adversarial"]
    assert len(adversarials) > 0
    # All adversarial examples are label=1 (rewrites of positives)
    assert all(ex["label"] == 1 for ex in adversarials)


def test_adversarial_transforms_callable():
    """All adversarial transforms should be callable and return strings."""
    query = "How does authentication work?"
    for transform in _ADVERSARIAL_TRANSFORMS:
        result = transform(query)
        assert isinstance(result, str)
        assert len(result) > 0


def test_data_generator_family_summary(tmp_path):
    path = _make_labels_file(tmp_path, n=20)
    cards = _make_anchor_cards()
    gen = DataGenerator(oracle_labels_path=path, anchor_cards=cards)
    summary = gen.query_family_summary()
    assert isinstance(summary, dict)
    assert "single_anchor" in summary
    assert "bridge" in summary


# ── _load_training_data / _split_by_example_type ──────────────────────────────


def test_load_training_data(tmp_path):
    path = tmp_path / "train.jsonl"
    examples = [
        {"query": "q1", "anchor_card": "a1", "label": 1, "example_type": "positive"},
        {"query": "q2", "anchor_card": "a2", "label": 0, "example_type": "hard_negative"},
        "not json",  # malformed line — should be skipped
    ]
    with open(path, "w") as f:
        for ex in examples:
            if isinstance(ex, dict):
                f.write(json.dumps(ex) + "\n")
            else:
                f.write(ex + "\n")

    loaded = _load_training_data(str(path))
    assert len(loaded) == 2
    assert loaded[0]["query"] == "q1"


def test_split_by_example_type():
    examples = [
        {"example_type": "positive", "label": 1},
        {"example_type": "positive", "label": 1},
        {"example_type": "hard_negative", "label": 0},
    ]
    by_type = _split_by_example_type(examples)
    assert len(by_type["positive"]) == 2
    assert len(by_type["hard_negative"]) == 1


def test_split_by_example_type_default():
    # Missing 'example_type' defaults to 'positive'
    examples = [{"label": 1}]
    by_type = _split_by_example_type(examples)
    assert "positive" in by_type


# ── SelectorVersion serialization ─────────────────────────────────────────────


def test_selector_version_round_trip():
    sv = SelectorVersion(
        version=1,
        atlas_version=2,
        embedding_model="BAAI/bge-base-en-v1.5",
        runtime_model_id="claude-haiku-4-5-20251001",
        base_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        training_examples=12000,
        val_task_success=0.87,
        threshold_tau=0.35,
        trained_at="2026-01-01T00:00:00+00:00",
        model_path="/tmp/selector_v1",
        query_family_mix={"single_anchor": 3000, "bridge": 2000},
    )
    d = sv.to_dict()
    restored = SelectorVersion.from_dict(d)

    assert restored.version == 1
    assert restored.atlas_version == 2
    assert restored.threshold_tau == pytest.approx(0.35)
    assert restored.query_family_mix["single_anchor"] == 3000


def test_selector_version_to_dict_keys():
    sv = SelectorVersion(
        version=1, atlas_version=1, embedding_model="m", runtime_model_id="r",
        base_model="b", training_examples=100, val_task_success=0.8,
        threshold_tau=0.3, trained_at="now", model_path="/tmp",
        query_family_mix={},
    )
    d = sv.to_dict()
    for key in ("version", "atlas_version", "base_model", "training_examples",
                "val_task_success", "threshold_tau", "trained_at", "model_path",
                "query_family_mix"):
        assert key in d
