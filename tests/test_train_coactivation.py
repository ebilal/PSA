"""
tests/test_train_coactivation.py — Tests for CoActivationTrainer.

Two tests:
  1. test_train_produces_model — 32 examples, 16 anchors, 2 epochs → files exist
  2. test_train_loss_decreases — 64 examples, 5 epochs → losses[-1] < losses[0]
"""

import os
import json
import sys

import numpy as np
import pytest


def _make_synthetic_npz(tmp_path, n_examples, n_anchors, centroid_dim=768, seed=0):
    """Generate synthetic coactivation training data and save as npz."""
    rng = np.random.default_rng(seed)

    query_vecs = rng.standard_normal((n_examples, centroid_dim)).astype(np.float32)
    # L2-normalize
    norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
    query_vecs = query_vecs / np.maximum(norms, 1e-8)

    ce_scores = rng.uniform(0.0, 1.0, size=(n_examples, n_anchors)).astype(np.float32)

    gold_masks = np.zeros((n_examples, n_anchors), dtype=np.float32)
    for i in range(n_examples):
        k = rng.integers(1, 4)  # 1–3 gold anchors
        gold_idx = rng.choice(n_anchors, size=k, replace=False)
        gold_masks[i, gold_idx] = 1.0

    gold_ks = gold_masks.sum(axis=1).astype(np.int32)

    centroids = rng.standard_normal((n_anchors, centroid_dim)).astype(np.float32)
    norms_c = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms_c, 1e-8)

    anchor_ids = np.arange(n_anchors, dtype=np.int32)

    npz_path = tmp_path / "coactivation_train.npz"
    np.savez(
        str(npz_path),
        query_vecs=query_vecs,
        ce_scores=ce_scores,
        gold_masks=gold_masks,
        gold_ks=gold_ks,
        centroids=centroids,
        anchor_ids=anchor_ids,
    )
    return str(tmp_path)


def test_train_produces_model(tmp_path):
    """Training with 32 examples and 2 epochs should produce model files."""
    pytest.importorskip("torch")

    from psa.training.train_coactivation import CoActivationTrainer

    data_dir = _make_synthetic_npz(tmp_path, n_examples=32, n_anchors=16)
    output_dir = str(tmp_path / "output")

    trainer = CoActivationTrainer(output_dir=output_dir)
    trainer.train(
        data_dir=data_dir,
        n_anchors=16,
        centroid_dim=768,
        epochs=2,
        batch_size=8,
    )

    assert os.path.exists(os.path.join(output_dir, "coactivation_model.pt"))
    assert os.path.exists(os.path.join(output_dir, "coactivation_version.json"))


def test_train_loss_decreases(tmp_path, monkeypatch):
    """Training loss over 5 epochs should decrease on CPU deterministically."""
    pytest.importorskip("torch")

    import torch

    from psa.training.train_coactivation import CoActivationTrainer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(0)
    data_dir = _make_synthetic_npz(tmp_path, n_examples=64, n_anchors=16, seed=1)
    output_dir = str(tmp_path / "output2")

    trainer = CoActivationTrainer(output_dir=output_dir, learning_rate=1e-3)
    losses = trainer.train(
        data_dir=data_dir,
        n_anchors=16,
        centroid_dim=768,
        epochs=5,
        batch_size=16,
        return_losses=True,
    )

    assert losses is not None, "Expected losses list when return_losses=True"
    assert len(losses) == 5, f"Expected 5 loss values, got {len(losses)}"
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


def test_train_handles_zero_examples(tmp_path):
    """Training should fail with a controlled error when there is no data."""
    pytest.importorskip("torch")

    from psa.training.train_coactivation import CoActivationTrainer

    data_dir = _make_synthetic_npz(tmp_path, n_examples=0, n_anchors=8, seed=5)
    output_dir = str(tmp_path / "output_zero")

    trainer = CoActivationTrainer(output_dir=output_dir)
    with pytest.raises(ValueError, match="zero examples"):
        trainer.train(
            data_dir=data_dir,
            n_anchors=8,
            centroid_dim=768,
            epochs=1,
            batch_size=1,
        )


def test_train_handles_single_example(tmp_path):
    """Training should not build an empty train loader for a single example."""
    pytest.importorskip("torch")

    import json

    from psa.training.train_coactivation import CoActivationTrainer

    data_dir = _make_synthetic_npz(tmp_path, n_examples=1, n_anchors=8, seed=4)
    output_dir = str(tmp_path / "output_single")

    trainer = CoActivationTrainer(output_dir=output_dir)
    trainer.train(
        data_dir=data_dir,
        n_anchors=8,
        centroid_dim=768,
        epochs=1,
        batch_size=1,
    )

    with open(os.path.join(output_dir, "coactivation_version.json")) as f:
        meta = json.load(f)

    assert os.path.exists(os.path.join(output_dir, "coactivation_model.pt"))
    assert os.path.exists(os.path.join(output_dir, "coactivation_version.json"))
    assert meta["validation_loss"] is None


def test_train_reports_progress_callback(tmp_path):
    """Training should emit per-epoch and validation progress through the callback."""
    pytest.importorskip("torch")

    from psa.training.train_coactivation import CoActivationTrainer

    data_dir = _make_synthetic_npz(tmp_path, n_examples=32, n_anchors=16, seed=2)
    output_dir = str(tmp_path / "output3")
    messages = []

    trainer = CoActivationTrainer(output_dir=output_dir, learning_rate=1e-3)
    trainer.train(
        data_dir=data_dir,
        n_anchors=16,
        centroid_dim=768,
        epochs=3,
        batch_size=8,
        progress_callback=messages.append,
    )

    epoch_messages = [m for m in messages if m.startswith("Epoch ")]
    assert len(epoch_messages) == 3
    assert any(m.startswith("Training on device:") for m in messages)
    assert any(m.startswith("Validation loss:") for m in messages)


def test_train_accepts_weight_decay(tmp_path):
    """Trainer should accept weight_decay and persist it in metadata."""
    pytest.importorskip("torch")

    import torch

    from psa.training.train_coactivation import CoActivationTrainer

    data_dir = _make_synthetic_npz(tmp_path, n_examples=16, n_anchors=8, seed=3)
    output_dir = str(tmp_path / "output4")

    optimizer_calls = {}
    original_adamw = torch.optim.AdamW

    class _RecordingAdamW:
        def __init__(self, params, lr, weight_decay):
            optimizer_calls["lr"] = lr
            optimizer_calls["weight_decay"] = weight_decay
            self._optimizer = original_adamw(
                params,
                lr=lr,
                weight_decay=weight_decay,
            )

        def zero_grad(self):
            self._optimizer.zero_grad()

        def step(self):
            self._optimizer.step()

    torch.optim.AdamW = _RecordingAdamW
    trainer = CoActivationTrainer(
        output_dir=output_dir,
        learning_rate=1e-3,
        weight_decay=0.05,
    )
    try:
        trainer.train(
            data_dir=data_dir,
            n_anchors=8,
            centroid_dim=768,
            epochs=1,
            batch_size=8,
        )
    finally:
        torch.optim.AdamW = original_adamw

    with open(os.path.join(output_dir, "coactivation_version.json")) as f:
        meta = json.load(f)

    assert optimizer_calls["weight_decay"] == 0.05
    assert optimizer_calls["lr"] == 1e-3
    assert meta["weight_decay"] == 0.05


def test_run_training_subprocess_invokes_module_and_checks_artifacts(tmp_path, monkeypatch):
    """Inference-contaminated callers should be able to isolate training in a fresh process."""
    from psa.training.train_coactivation import run_training_subprocess

    calls = {}
    data_dir = str(tmp_path / "data")
    output_dir = str(tmp_path / "output")
    os.makedirs(data_dir)
    os.makedirs(output_dir)

    def fake_run(cmd, check):
        calls["cmd"] = cmd
        calls["check"] = check
        with open(os.path.join(output_dir, "coactivation_model.pt"), "wb") as handle:
            handle.write(b"ok")
        with open(os.path.join(output_dir, "coactivation_version.json"), "w") as handle:
            json.dump({"validation_loss": 0.1}, handle)

    monkeypatch.setattr("subprocess.run", fake_run)

    run_training_subprocess(
        data_dir=data_dir,
        output_dir=output_dir,
        n_anchors=256,
        centroid_dim=768,
        epochs=8,
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=0.01,
    )

    assert calls["check"] is True
    assert calls["cmd"][:4] == [
        sys.executable,
        "-m",
        "psa.training.train_coactivation",
        "--data-dir",
    ]
    assert "--output-dir" in calls["cmd"]
    assert "--epochs" in calls["cmd"]
    assert "--weight-decay" in calls["cmd"]
