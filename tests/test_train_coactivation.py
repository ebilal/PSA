"""
tests/test_train_coactivation.py — Tests for CoActivationTrainer.

Two tests:
  1. test_train_produces_model — 32 examples, 16 anchors, 2 epochs → files exist
  2. test_train_loss_decreases — 64 examples, 5 epochs → losses[-1] < losses[0]
"""

import os

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


def test_train_loss_decreases(tmp_path):
    """Training loss over 5 epochs should decrease (last < first)."""
    pytest.importorskip("torch")

    from psa.training.train_coactivation import CoActivationTrainer

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
