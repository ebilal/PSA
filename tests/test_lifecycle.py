"""Tests for psa.lifecycle — state persistence through retrain path."""

import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from psa.lifecycle import LifecyclePipeline


@dataclass
class FakeSelectorVersion:
    version: int = 1
    model_path: str = "/fake/model/path"


def test_retrain_selector_mutates_state_dict():
    """_retrain_selector should mutate the state dict directly, not write separately."""
    lp = LifecyclePipeline(base_dir="/tmp/test_lifecycle")
    state = {"selector_mode": "cosine", "selector_version": 0}

    tenant = MagicMock()
    tenant.root_dir = "/tmp/test_lifecycle/tenants/test"
    store = MagicMock()
    atlas = MagicMock()
    atlas.version = 1
    atlas.cards = []

    fake_sv = FakeSelectorVersion(version=1, model_path="/fake/model/path")

    # Create a labels file with 300+ lines
    labels_dir = os.path.join(tenant.root_dir, "training")
    os.makedirs(labels_dir, exist_ok=True)
    labels_path = os.path.join(labels_dir, "oracle_labels.jsonl")
    with open(labels_path, "w") as f:
        for i in range(350):
            f.write(f'{{"query_id": "{i}"}}\n')

    with (
        patch("psa.selector.check_training_gates") as mock_gates,
        patch("psa.training.data_generator.DataGenerator") as mock_gen_cls,
        patch("psa.training.train_selector.SelectorTrainer") as mock_trainer_cls,
        patch("psa.training.data_split.split_train_val") as mock_split,
    ):
        mock_gates.return_value = MagicMock(gates_met=True, blocking_reasons=[])
        mock_gen_cls.return_value.generate.return_value = 500
        mock_split.return_value = {
            "n_train_queries": 80,
            "n_val_queries": 20,
            "n_train_examples": 400,
            "n_val_examples": 100,
            "train_positive_rate": 0.5,
            "val_positive_rate": 0.5,
        }
        mock_trainer_cls.return_value.train.return_value = fake_sv

        result = lp._retrain_selector(tenant, store, atlas, state)

    assert result is True
    assert state["selector_mode"] == "trained"
    assert state["selector_model_path"] == "/fake/model/path"
    assert state["selector_version"] == 1


def test_retrain_selector_gates_not_met_leaves_state():
    """When training gates are not met, state dict should not be modified."""
    lp = LifecyclePipeline(base_dir="/tmp/test_lifecycle2")
    state = {"selector_mode": "cosine"}

    tenant = MagicMock()
    tenant.root_dir = "/tmp/test_lifecycle2/tenants/test"
    store = MagicMock()
    atlas = MagicMock()
    atlas.version = 1

    # Create labels file with too few lines
    labels_dir = os.path.join(tenant.root_dir, "training")
    os.makedirs(labels_dir, exist_ok=True)
    labels_path = os.path.join(labels_dir, "oracle_labels.jsonl")
    with open(labels_path, "w") as f:
        for i in range(10):
            f.write(f'{{"query_id": "{i}"}}\n')

    result = lp._retrain_selector(tenant, store, atlas, state)
    assert result is False
    assert state["selector_mode"] == "cosine"
    assert "selector_model_path" not in state


def test_lifecycle_source_does_not_reference_benchmark_path():
    """Regression: lifecycle.py must not auto-train from benchmark artifacts.

    Benchmark-derived memory scorer training used to run in the slow path.
    That's a research/production boundary violation; the block was removed
    in Branch 2. This test locks the removal.
    """
    import inspect

    import psa.lifecycle as _mod

    src = inspect.getsource(_mod)
    assert "benchmarks/longmemeval" not in src, (
        "lifecycle.py must not reference the benchmark results directory"
    )
    assert "MemoryScorerTrainer" not in src, (
        "lifecycle.py must not import or call MemoryScorerTrainer"
    )
    assert 'mode="benchmark"' not in src and "mode='benchmark'" not in src, (
        "lifecycle.py must not invoke any trainer with mode=benchmark"
    )
