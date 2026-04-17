"""Tests for psa.training.train_selector helpers."""

from __future__ import annotations


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
