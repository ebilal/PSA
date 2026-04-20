"""
train_coactivation.py -- Train the PSA CoActivationModel.

Loads coactivation_train.npz (produced by coactivation_data.py),
splits into train/val, runs AdamW training, and saves:
  - coactivation_model.pt       -- model state_dict
  - coactivation_version.json   -- training metadata

Loss:
    BCE(refined_scores, gold_masks) + 0.3 * MSE(thresholds, gold_ks / n_anchors)

Requirements: torch (install via psa[training])
"""

import json
import logging
import os
import subprocess
import sys
import argparse
from datetime import datetime, timezone
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger("psa.training.train_coactivation")

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EPOCHS = 8
BATCH_SIZE = 16


def run_training_subprocess(
    *,
    data_dir: str,
    output_dir: str,
    n_anchors: int = 256,
    centroid_dim: int = 768,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> None:
    """Run co-activation training in a fresh Python process.

    This isolates the trainer from sentence-transformers inference state in the
    parent process, which otherwise segfaults on some Apple Silicon setups when
    training follows full-atlas scoring in the same interpreter.
    """
    cmd = [
        sys.executable,
        "-m",
        "psa.training.train_coactivation",
        "--data-dir",
        data_dir,
        "--output-dir",
        output_dir,
        "--n-anchors",
        str(n_anchors),
        "--centroid-dim",
        str(centroid_dim),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(weight_decay),
    ]
    subprocess.run(cmd, check=True)

    model_path = os.path.join(output_dir, "coactivation_model.pt")
    version_path = os.path.join(output_dir, "coactivation_version.json")
    if not (os.path.exists(model_path) and os.path.exists(version_path)):
        raise RuntimeError(
            f"Co-activation subprocess completed without writing expected artifacts in {output_dir}"
        )


class CoActivationTrainer:
    """
    Train the CoActivationModel on coactivation_train.npz data.

    Parameters
    ----------
    output_dir:
        Directory where coactivation_model.pt and
        coactivation_version.json will be written.
    learning_rate:
        AdamW learning rate. Default: 1e-4.
    """

    def __init__(
        self,
        output_dir: str,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
    ):
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def train(
        self,
        data_dir: str,
        n_anchors: int = 256,
        centroid_dim: int = 768,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        val_split: float = 0.15,
        return_losses: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Optional[List[float]]:
        """
        Train the CoActivationModel.

        Parameters
        ----------
        data_dir:
            Directory containing coactivation_train.npz.
        n_anchors:
            Number of anchors (overridden by actual data if different).
        centroid_dim:
            Centroid dimension (default 768).
        epochs:
            Number of training epochs. Default: 8.
        batch_size:
            Mini-batch size.
        val_split:
            Fraction of data used for validation (default 0.15).
        return_losses:
            If True, return list of per-epoch training losses.

        Returns
        -------
        Optional[List[float]]
            List of per-epoch mean training losses if return_losses=True,
            else None.
        """
        def _report(message: str) -> None:
            logger.info(message)
            if progress_callback is not None:
                progress_callback(message)

        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError(
                "PyTorch is required for CoActivation training. "
                "Install it with: pip install 'psa[training]'"
            )

        from psa.coactivation import CoActivationModel

        # Load data
        npz_path = os.path.join(data_dir, "coactivation_train.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Missing coactivation training data: {npz_path}. "
                "Expected coactivation_train.npz in data_dir."
            )
        data = np.load(npz_path)

        query_vecs = data["query_vecs"]  # (N, 768)
        ce_scores = data["ce_scores"]  # (N, n_anchors)
        gold_masks = data["gold_masks"]  # (N, n_anchors)
        gold_ks = data["gold_ks"]  # (N,)
        centroids = data["centroids"]  # (n_anchors, 768)
        anchor_features_np = data.get("anchor_features")  # (n_anchors, feat_dim) or None

        N = query_vecs.shape[0]
        if N == 0:
            raise ValueError(
                f"coactivation_train.npz in {data_dir} contains zero examples; "
                "training requires at least one example."
            )
        actual_n_anchors = ce_scores.shape[1]
        actual_centroid_dim = centroids.shape[1]
        actual_anchor_feature_dim = (
            int(anchor_features_np.shape[1]) if anchor_features_np is not None else 0
        )

        logger.info(
            "Loaded %d examples, %d anchors, centroid_dim=%d, anchor_feature_dim=%d",
            N,
            actual_n_anchors,
            actual_centroid_dim,
            actual_anchor_feature_dim,
        )

        # Train / val split (seed=42)
        rng = np.random.default_rng(42)
        idx = rng.permutation(N)
        if N <= 1:
            n_val = 0
        else:
            n_val = min(max(1, int(N * val_split)), N - 1)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        # Shared centroid tensor — expanded per-batch in the training loop
        # to avoid materializing (N, 256, 768) in memory.
        centroids_t = torch.from_numpy(centroids).float()  # (n_anchors, 768)

        # Shared anchor features tensor (may be None for old data without features)
        anchor_features_t: Optional[torch.Tensor] = (
            torch.from_numpy(anchor_features_np).float() if anchor_features_np is not None else None
        )  # (n_anchors, feat_dim) or None

        def _make_tensors(indices):
            qv = torch.from_numpy(query_vecs[indices]).float()
            ce = torch.from_numpy(ce_scores[indices]).float()
            gm = torch.from_numpy(gold_masks[indices]).float()
            gk = torch.from_numpy(gold_ks[indices]).float()
            return qv, ce, gm, gk

        train_qv, train_ce, train_gm, train_gk = _make_tensors(train_idx)
        val_qv, val_ce, val_gm, val_gk = _make_tensors(val_idx)

        train_ds = TensorDataset(train_qv, train_ce, train_gm, train_gk)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Device selection: cuda > cpu.
        # MPS is excluded — TransformerEncoder causes SIGSEGV on Apple Silicon
        # for the (batch, 256, 256) tensor shapes used here. CPU is fast enough
        # for a ~2M param model (~2 min for 10 epochs on 1000 examples).
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        _report(f"Training on device: {device}")

        # Model + optimiser
        model = CoActivationModel(
            n_anchors=actual_n_anchors,
            centroid_dim=actual_centroid_dim,
            anchor_feature_dim=actual_anchor_feature_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        # Training loop
        epoch_losses: List[float] = []

        for epoch in range(1, epochs + 1):
            model.train()
            batch_losses: List[float] = []

            for batch in train_loader:
                b_qv, b_ce, b_gm, b_gk = [t.to(device) for t in batch]
                # Expand shared centroids to batch size
                b_c = centroids_t.unsqueeze(0).expand(b_ce.shape[0], -1, -1).to(device)
                # Expand anchor features to batch size (or None if not available)
                b_af = (
                    anchor_features_t.unsqueeze(0).expand(b_ce.shape[0], -1, -1).to(device)
                    if anchor_features_t is not None
                    else None
                )

                optimizer.zero_grad()

                refined_scores, thresholds = model(b_ce, b_c, b_qv, b_af)

                # gold_ks / n_anchors as soft threshold target
                threshold_targets = b_gk / actual_n_anchors

                loss = bce_loss(refined_scores, b_gm) + 0.3 * mse_loss(
                    thresholds, threshold_targets
                )
                loss.backward()
                optimizer.step()

                batch_losses.append(float(loss.item()))

            epoch_mean = float(np.mean(batch_losses))
            epoch_losses.append(epoch_mean)
            _report(f"Epoch {epoch}/{epochs} loss={epoch_mean:.4f}")

        # Validation pass
        model.train(False)
        with torch.no_grad():
            if val_ce.shape[0] > 0:
                val_c_batch = centroids_t.unsqueeze(0).expand(val_ce.shape[0], -1, -1)
                val_af_batch = (
                    anchor_features_t.unsqueeze(0).expand(val_ce.shape[0], -1, -1)
                    if anchor_features_t is not None
                    else None
                )
                r, t = model(
                    val_ce.to(device),
                    val_c_batch.to(device),
                    val_qv.to(device),
                    val_af_batch.to(device) if val_af_batch is not None else None,
                )
                tt = val_gk.to(device).float() / actual_n_anchors
                val_loss = float((bce_loss(r, val_gm.to(device)) + 0.3 * mse_loss(t, tt)).item())
            else:
                val_loss = None
        _report(
            "Validation loss: "
            + (f"{val_loss:.4f}" if val_loss is not None else "unavailable (no validation set)")
        )

        # Save artefacts
        os.makedirs(self.output_dir, exist_ok=True)
        model_path = os.path.join(self.output_dir, "coactivation_model.pt")
        torch.save(model.state_dict(), model_path)

        meta = {
            "n_anchors": actual_n_anchors,
            "centroid_dim": actual_centroid_dim,
            "anchor_feature_dim": actual_anchor_feature_dim,
            "query_frame_dim": 0,  # future training with frame features will set 11
            "training_examples": int(N),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "validation_loss": val_loss,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        version_path = os.path.join(self.output_dir, "coactivation_version.json")
        with open(version_path, "w") as f:
            json.dump(meta, f, indent=2)

        _report(f"Co-activation model saved to {self.output_dir}")

        return epoch_losses if return_losses else None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the PSA co-activation model")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-anchors", type=int, default=256)
    parser.add_argument("--centroid-dim", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    trainer = CoActivationTrainer(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    trainer.train(
        data_dir=args.data_dir,
        n_anchors=args.n_anchors,
        centroid_dim=args.centroid_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        progress_callback=print,
    )


if __name__ == "__main__":
    main()
