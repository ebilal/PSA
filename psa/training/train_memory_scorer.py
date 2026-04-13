"""
train_memory_scorer.py — Training loop for the Level 2 memory-level re-ranker MLP.

Trains MemoryReRanker (from psa.memory_scorer) on JSONL data produced by
memory_scorer_data.generate_memory_scorer_data().

JSONL record format:
  {"ce_score": 0.61, "type_vec": [0,0,1,0,0,0], "quality_score": 0.85,
   "body_norm": 0.2, "recency": 0.7, "cosine": 0.72, "label": 1, "query_id": "q_0"}

Feature vector order (11 dims):
  [0]   ce_score
  [1-6] type_vec (6-dim one-hot)
  [7]   quality_score
  [8]   body_norm
  [9]   recency
  [10]  cosine
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from psa.memory_scorer import MemoryReRanker

logger = logging.getLogger("psa.training.train_memory_scorer")


def _load_jsonl(data_path: str):
    """Load all non-empty valid JSONL records from data_path."""
    records = []
    with open(data_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _record_to_features(record: dict) -> List[float]:
    """Flatten a JSONL record into an 11-dim feature list."""
    type_vec = record["type_vec"]  # list of 6 floats
    return (
        [float(record["ce_score"])]
        + [float(v) for v in type_vec]
        + [
            float(record["quality_score"]),
            float(record["body_norm"]),
            float(record["recency"]),
            float(record["cosine"]),
        ]
    )


class MemoryScorerTrainer:
    """
    Train a MemoryReRanker MLP for Level 2 memory re-ranking.

    Parameters
    ----------
    output_dir:
        Directory where model files will be saved.
    learning_rate:
        Adam learning rate (default 1e-3).
    """

    def __init__(self, output_dir: str, learning_rate: float = 1e-3) -> None:
        self.output_dir = output_dir
        self.learning_rate = learning_rate

    def train(
        self,
        data_path: str,
        epochs: int = 20,
        batch_size: int = 64,
        val_split: float = 0.15,
        return_losses: bool = False,
    ) -> Optional[List[float]]:
        """
        Train the MemoryReRanker on JSONL data.

        Parameters
        ----------
        data_path:
            Path to JSONL training data file.
        epochs:
            Number of training epochs (default 20).
        batch_size:
            Mini-batch size (default 64).
        val_split:
            Fraction of data held out for validation (default 0.15).
        return_losses:
            If True, return list of per-epoch training losses.

        Returns
        -------
        List of per-epoch training losses if return_losses=True, else None.
        """
        records = _load_jsonl(data_path)
        if not records:
            raise ValueError(f"No valid records found in {data_path}")

        # Build feature matrix and label vector
        X_list = [_record_to_features(r) for r in records]
        y_list = [float(r["label"]) for r in records]

        X_all = torch.tensor(X_list, dtype=torch.float32)
        y_all = torch.tensor(y_list, dtype=torch.float32)

        input_dim = X_all.shape[1]
        n_total = len(records)

        # Train/val split (deterministic seed=42)
        torch.manual_seed(42)
        perm = torch.randperm(n_total)
        n_val = max(1, int(n_total * val_split))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        logger.info(
            "Training MemoryReRanker: input_dim=%d, train=%d, val=%d",
            input_dim,
            len(train_idx),
            len(val_idx),
        )

        model = MemoryReRanker(input_dim=input_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        epoch_losses: List[float] = []

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle training data each epoch
            perm_train = torch.randperm(len(X_train))
            X_shuf = X_train[perm_train]
            y_shuf = y_train[perm_train]

            for start in range(0, len(X_shuf), batch_size):
                X_batch = X_shuf[start : start + batch_size]
                y_batch = y_shuf[start : start + batch_size]

                optimizer.zero_grad()
                preds = model(X_batch).squeeze(1)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            epoch_losses.append(avg_train_loss)
            logger.debug("Epoch %d/%d -- train_loss=%.4f", epoch, epochs, avg_train_loss)

        # Validation loss
        model.train(False)
        with torch.no_grad():
            val_preds = model(X_val).squeeze(1)
            val_loss = float(criterion(val_preds, y_val).item())

        final_loss = epoch_losses[-1] if epoch_losses else 0.0
        logger.info(
            "Training complete: final_train_loss=%.4f, val_loss=%.4f", final_loss, val_loss
        )

        # Save model
        os.makedirs(self.output_dir, exist_ok=True)
        model_path = Path(self.output_dir) / "memory_scorer_model.pt"
        version_path = Path(self.output_dir) / "memory_scorer_version.json"

        torch.save(model.state_dict(), str(model_path))

        meta = {
            "input_dim": input_dim,
            "hidden_dim": 32,
            "training_examples": n_total,
            "epochs": epochs,
            "final_loss": round(final_loss, 6),
            "val_loss": round(val_loss, 6),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        with version_path.open("w") as fh:
            json.dump(meta, fh, indent=2)

        logger.info("Model saved to %s", self.output_dir)

        if return_losses:
            return epoch_losses
        return None
