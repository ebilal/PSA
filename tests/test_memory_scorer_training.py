"""
tests/test_memory_scorer_training.py — Unit tests for MemoryScorerTrainer.
"""

import json
import random
import tempfile
from pathlib import Path


def _make_synthetic_data(n: int, path: Path) -> None:
    """Write n synthetic JSONL examples to path."""
    rng = random.Random(0)
    type_orders = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
    with path.open("w") as fh:
        for i in range(n):
            record = {
                "ce_score": rng.uniform(0.0, 1.0),
                "type_vec": type_orders[i % len(type_orders)],
                "quality_score": rng.uniform(0.5, 1.0),
                "body_norm": rng.uniform(0.0, 1.0),
                "recency": rng.uniform(0.0, 1.0),
                "cosine": rng.uniform(0.0, 1.0),
                "label": rng.randint(0, 1),
                "query_id": f"q_{i}",
            }
            fh.write(json.dumps(record) + "\n")


def test_train_memory_scorer():
    """MemoryScorerTrainer trains on synthetic data, saves model files, loss decreases."""
    from psa.training.train_memory_scorer import MemoryScorerTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        data_path = tmpdir_path / "memory_scorer_data.jsonl"
        output_dir = tmpdir_path / "model_output"

        _make_synthetic_data(100, data_path)

        trainer = MemoryScorerTrainer(output_dir=str(output_dir), learning_rate=1e-3)
        losses = trainer.train(
            data_path=str(data_path),
            epochs=5,
            batch_size=32,
            val_split=0.15,
            return_losses=True,
        )

        # Model files exist
        assert (output_dir / "memory_scorer_model.pt").exists(), "model .pt file not found"
        assert (output_dir / "memory_scorer_version.json").exists(), "version json not found"

        # Version metadata is valid
        with (output_dir / "memory_scorer_version.json").open() as fh:
            meta = json.load(fh)
        assert meta["input_dim"] == 11
        assert meta["hidden_dim"] == 32
        assert meta["training_examples"] == 100
        assert meta["epochs"] == 5
        assert "final_loss" in meta
        assert "val_loss" in meta
        assert "trained_at" in meta

        # Losses were returned and loss decreased over 5 epochs
        assert losses is not None
        assert len(losses) == 5
        # Loss should decrease overall (first epoch higher than last)
        assert losses[0] > losses[-1], (
            f"Expected loss to decrease, got first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )
