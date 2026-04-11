"""Tests for psa.training.data_split — query-grouped train/val splitter."""

import json


def _make_example(query_id, query, label):
    return {
        "query": query,
        "anchor_card": "some card text",
        "label": label,
        "anchor_id": 1,
        "query_family": "bridge",
        "example_type": "positive" if label == 1 else "hard_negative",
        "source_query_id": query_id,
    }


def _write_examples(path, examples):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def _read_examples(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_split_creates_two_files(tmp_path):
    from psa.training.data_split import split_train_val

    examples_path = str(tmp_path / "examples.jsonl")
    examples = []
    for i in range(4):
        for j in range(3):
            examples.append(_make_example(f"q{i}", f"query {i}", j % 2))
    _write_examples(examples_path, examples)

    train_path = str(tmp_path / "train.jsonl")
    val_path = str(tmp_path / "val.jsonl")
    stats = split_train_val(examples_path, train_path, val_path, val_fraction=0.25, seed=42)

    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    assert stats["n_train_examples"] + stats["n_val_examples"] == 12
    assert stats["n_train_queries"] + stats["n_val_queries"] == 4


def test_split_is_query_grouped(tmp_path):
    from psa.training.data_split import split_train_val

    examples_path = str(tmp_path / "examples.jsonl")
    examples = []
    for i in range(10):
        for j in range(5):
            examples.append(_make_example(f"q{i}", f"query {i}", j % 2))
    _write_examples(examples_path, examples)

    train_path = str(tmp_path / "train.jsonl")
    val_path = str(tmp_path / "val.jsonl")
    split_train_val(examples_path, train_path, val_path, val_fraction=0.15, seed=42)

    train = _read_examples(train_path)
    val = _read_examples(val_path)
    train_qids = {ex["source_query_id"] for ex in train}
    val_qids = {ex["source_query_id"] for ex in val}
    assert train_qids & val_qids == set()


def test_split_deterministic(tmp_path):
    from psa.training.data_split import split_train_val

    examples_path = str(tmp_path / "examples.jsonl")
    examples = [_make_example(f"q{i}", f"query {i}", i % 2) for i in range(20)]
    _write_examples(examples_path, examples)

    split_train_val(examples_path, str(tmp_path / "t1.jsonl"), str(tmp_path / "v1.jsonl"), seed=42)
    split_train_val(examples_path, str(tmp_path / "t2.jsonl"), str(tmp_path / "v2.jsonl"), seed=42)
    assert _read_examples(str(tmp_path / "v1.jsonl")) == _read_examples(str(tmp_path / "v2.jsonl"))


def test_split_fallback_when_no_query_id(tmp_path):
    from psa.training.data_split import split_train_val

    examples_path = str(tmp_path / "examples.jsonl")
    examples = []
    for i in range(6):
        ex = _make_example(None, f"query {i}", i % 2)
        del ex["source_query_id"]
        examples.append(ex)
    _write_examples(examples_path, examples)

    train_path = str(tmp_path / "train.jsonl")
    val_path = str(tmp_path / "val.jsonl")
    stats = split_train_val(examples_path, train_path, val_path, val_fraction=0.3, seed=42)
    assert stats["n_train_examples"] + stats["n_val_examples"] == 6


def test_split_min_val_queries_safety(tmp_path):
    from psa.training.data_split import split_train_val

    examples_path = str(tmp_path / "examples.jsonl")
    examples = [_make_example(f"q{i}", f"query {i}", i % 2) for i in range(3)]
    _write_examples(examples_path, examples)

    train_path = str(tmp_path / "train.jsonl")
    val_path = str(tmp_path / "val.jsonl")
    stats = split_train_val(
        examples_path, train_path, val_path, val_fraction=0.15, seed=42, min_val_queries=1
    )
    assert stats["n_val_queries"] >= 1


def test_split_reports_positive_rates(tmp_path):
    from psa.training.data_split import split_train_val

    examples_path = str(tmp_path / "examples.jsonl")
    examples = []
    for i in range(10):
        examples.append(_make_example(f"q{i}", f"query {i}", 1))
        examples.append(_make_example(f"q{i}", f"query {i}", 0))
    _write_examples(examples_path, examples)

    train_path = str(tmp_path / "train.jsonl")
    val_path = str(tmp_path / "val.jsonl")
    stats = split_train_val(examples_path, train_path, val_path, val_fraction=0.2, seed=42)

    assert "train_positive_rate" in stats
    assert "val_positive_rate" in stats
    assert 0.0 <= stats["train_positive_rate"] <= 1.0
    assert 0.0 <= stats["val_positive_rate"] <= 1.0
