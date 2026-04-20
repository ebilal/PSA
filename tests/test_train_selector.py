"""Tests for psa.training.train_selector helpers."""

from __future__ import annotations

import sys
import types

import pytest


def test_select_training_device_defaults_to_cpu(monkeypatch):
    """Default device is CPU because MPS crashes CrossEncoder.fit() at scale."""
    from psa.training.train_selector import _select_training_device

    monkeypatch.delenv("PSA_SELECTOR_TRAIN_DEVICE", raising=False)
    assert _select_training_device() == "cpu"


def test_select_training_device_respects_env_override(monkeypatch):
    """Operators can opt into MPS (or cuda) via env var."""
    from psa.training.train_selector import _select_training_device

    monkeypatch.setenv("PSA_SELECTOR_TRAIN_DEVICE", "mps")
    assert _select_training_device() == "mps"

    monkeypatch.setenv("PSA_SELECTOR_TRAIN_DEVICE", "cuda")
    assert _select_training_device() == "cuda"


def test_select_training_device_ignores_empty_env(monkeypatch):
    """Empty env string is not an override — fall through to the default."""
    from psa.training.train_selector import _select_training_device

    monkeypatch.setenv("PSA_SELECTOR_TRAIN_DEVICE", "")
    assert _select_training_device() == "cpu"


def test_selector_trainer_accepts_nondefault_hparams(monkeypatch):
    """Trainer should keep constructor overrides for metadata persistence."""
    from psa.training.train_selector import SelectorTrainer

    trainer = SelectorTrainer(
        output_dir="/tmp/selector",
        atlas_version=7,
        learning_rate=1e-5,
        batch_size=48,
        epochs=5,
        warmup_ratio=0.2,
    )

    assert trainer.learning_rate == 1e-5
    assert trainer.batch_size == 48
    assert trainer.epochs == 5
    assert trainer.warmup_ratio == 0.2


def test_selector_trainer_train_persists_nondefault_hparams(monkeypatch, tmp_path):
    """train() must forward instance hparams into the written selector metadata."""
    import json

    from psa.training.train_selector import SelectorTrainer

    class DummyCrossEncoder:
        def __init__(self, *args, **kwargs):
            self.saved_path = None

        def save(self, path):
            self.saved_path = path

    class DummyDataset:
        pass

    phase_calls = []

    def fake_train_phase(self, model, train_examples, epochs=1):
        phase_calls.append(epochs)
        return 0.0

    torch_module = types.ModuleType("torch")
    torch_module.__path__ = []
    torch_utils_module = types.ModuleType("torch.utils")
    torch_utils_module.__path__ = []
    torch_utils_data_module = types.ModuleType("torch.utils.data")
    torch_utils_data_module.Dataset = DummyDataset
    torch_utils_module.data = torch_utils_data_module
    torch_module.utils = torch_utils_module

    sentence_transformers_module = types.ModuleType("sentence_transformers")
    sentence_transformers_module.__path__ = []
    cross_encoder_module = types.ModuleType("sentence_transformers.cross_encoder")
    cross_encoder_module.CrossEncoder = DummyCrossEncoder
    sentence_transformers_module.cross_encoder = cross_encoder_module

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils_module)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data_module)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers_module)
    monkeypatch.setitem(sys.modules, "sentence_transformers.cross_encoder", cross_encoder_module)
    monkeypatch.setitem(sys.modules, "datasets", types.ModuleType("datasets"))
    monkeypatch.setitem(sys.modules, "accelerate", types.ModuleType("accelerate"))

    monkeypatch.setattr(SelectorTrainer, "_train_phase", fake_train_phase)

    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        json.dumps(
            {
                "query": "q",
                "anchor_card": "a",
                "label": 1,
                "example_type": "positive",
                "query_family": "bridge",
            }
        )
        + "\n"
        + json.dumps(
            {
                "query": "q2",
                "anchor_card": "b",
                "label": 0,
                "example_type": "hard_negative",
                "query_family": "bridge",
            }
        )
        + "\n"
        + json.dumps(
            {
                "query": "q3",
                "anchor_card": "c",
                "label": 0,
                "example_type": "adversarial",
                "query_family": "bridge",
            }
        )
        + "\n"
    )

    trainer = SelectorTrainer(
        output_dir=str(tmp_path / "out"),
        atlas_version=7,
        learning_rate=1e-5,
        batch_size=48,
        epochs=5,
        warmup_ratio=0.2,
    )
    trainer.train(train_data_path=str(train_path))

    meta_path = tmp_path / "out" / "selector_v1" / "selector_version.json"
    with open(meta_path) as f:
        meta = json.load(f)

    assert phase_calls == [5, 5, 5]
    assert meta["learning_rate"] == 1e-5
    assert meta["batch_size"] == 48
    assert meta["epochs"] == 5
    assert meta["warmup_ratio"] == 0.2


def test_selector_trainer_check_requirements_requires_training_extras(monkeypatch):
    """Missing datasets/accelerate should fail fast during requirements checks."""
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers.cross_encoder")

    from psa.training.train_selector import SelectorTrainer

    monkeypatch.setitem(sys.modules, "datasets", None)
    monkeypatch.setitem(sys.modules, "accelerate", None)

    with pytest.raises(ImportError, match="datasets|accelerate"):
        SelectorTrainer._check_requirements()
