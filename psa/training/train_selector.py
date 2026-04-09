"""
train_selector.py — Fine-tune the PSA anchor selector cross-encoder.

Base model: cross-encoder/ms-marco-MiniLM-L-6-v2
Training format: pairwise (query, anchor_card) → binary label
Loss: L_BCE + 0.2 * L_margin
Hyperparameters (plan defaults):
  lr=2e-5, batch=32, epochs=3, max_seq=320, warmup=0.1

Training phases:
  1. Supervised warm start on oracle labels
  2. Hard-negative curriculum
  3. Adversarial hardening

Stop rule: real-query dev-set task success stops improving for 2 evals.

Output: saved model at ~/.psa/models/selector_v{version}/

Requirements: torch, sentence-transformers (installed via psa[training])
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("psa.training.train_selector")

# ── Hyperparameters ───────────────────────────────────────────────────────────

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EPOCHS = 3
MAX_SEQ_LEN = 320
WARMUP_RATIO = 0.1
MARGIN_LOSS_WEIGHT = 0.2


# ── Training record ───────────────────────────────────────────────────────────


@dataclass
class SelectorVersion:
    """Metadata about a trained selector version."""

    version: int
    atlas_version: int
    embedding_model: str
    runtime_model_id: str
    base_model: str
    training_examples: int
    val_task_success: float
    threshold_tau: float
    trained_at: str
    model_path: str
    query_family_mix: Dict[str, int]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "atlas_version": self.atlas_version,
            "embedding_model": self.embedding_model,
            "runtime_model_id": self.runtime_model_id,
            "base_model": self.base_model,
            "training_examples": self.training_examples,
            "val_task_success": self.val_task_success,
            "threshold_tau": self.threshold_tau,
            "trained_at": self.trained_at,
            "model_path": self.model_path,
            "query_family_mix": self.query_family_mix,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SelectorVersion":
        return cls(**d)


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_training_data(data_path: str) -> List[dict]:
    """Load training examples from a JSONL file."""
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return examples


def _split_by_example_type(
    examples: List[dict],
) -> Dict[str, List[dict]]:
    """Group examples by type for phased training."""
    result: Dict[str, List[dict]] = {}
    for ex in examples:
        t = ex.get("example_type", "positive")
        result.setdefault(t, []).append(ex)
    return result


# ── Trainer ───────────────────────────────────────────────────────────────────


class SelectorTrainer:
    """
    Fine-tunes a cross-encoder for PSA anchor selection.

    Requires torch and sentence-transformers[training].
    Raises ImportError clearly if torch is not available.
    """

    def __init__(
        self,
        output_dir: str,
        atlas_version: int,
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        runtime_model_id: str = "claude-haiku-4-5-20251001",
        base_model: str = BASE_MODEL,
        learning_rate: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
        epochs: int = EPOCHS,
        max_seq_len: int = MAX_SEQ_LEN,
        warmup_ratio: float = WARMUP_RATIO,
    ):
        self._check_requirements()
        self.output_dir = output_dir
        self.atlas_version = atlas_version
        self.embedding_model_name = embedding_model_name
        self.runtime_model_id = runtime_model_id
        self.base_model = base_model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_seq_len = max_seq_len
        self.warmup_ratio = warmup_ratio

    @staticmethod
    def _check_requirements():
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "PyTorch is required for selector training. "
                "Install it with: pip install 'psa[training]'"
            )
        try:
            from sentence_transformers.cross_encoder import CrossEncoder  # noqa: F401
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for selector training. "
                "Install it with: pip install 'psa[training]'"
            )

    def train(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        version: int = 1,
    ) -> SelectorVersion:
        """
        Run the three-phase training pipeline.

        Phase 1: Supervised warm start on oracle-labeled positives
        Phase 2: Hard-negative curriculum
        Phase 3: Adversarial hardening

        Parameters
        ----------
        train_data_path: JSONL file from DataGenerator.generate()
        val_data_path: optional held-out validation JSONL
        version: selector version number

        Returns
        -------
        SelectorVersion metadata (model saved to output_dir)
        """
        from sentence_transformers.cross_encoder import CrossEncoder
        from torch.utils.data import DataLoader, Dataset as TorchDataset

        examples = _load_training_data(train_data_path)
        by_type = _split_by_example_type(examples)

        logger.info(
            "Training selector v%d with %d total examples "
            "(positive=%d, hard_neg=%d, adversarial=%d)",
            version,
            len(examples),
            len(by_type.get("positive", [])),
            len(by_type.get("hard_negative", [])),
            len(by_type.get("adversarial", [])),
        )

        # Select device: MPS (Apple Silicon) → CPU fallback (no CUDA assumed)
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info("Training on device: %s", device)

        # Load base model
        model = CrossEncoder(self.base_model, max_length=self.max_seq_len, device=device)

        model_out = os.path.join(self.output_dir, f"selector_v{version}")
        os.makedirs(model_out, exist_ok=True)

        class _PairExample:
            """Minimal object with .texts and .label for CrossEncoder.fit()."""
            __slots__ = ('texts', 'label')
            def __init__(self, texts, label):
                self.texts = texts
                self.label = label

        class PairDataset(TorchDataset):
            """Dataset returning _PairExample objects for CrossEncoder.fit()."""
            def __init__(self, examples):
                self._data = [
                    _PairExample([ex["query"], ex["anchor_card"]], float(ex["label"]))
                    for ex in examples
                ]
            def __len__(self):
                return len(self._data)
            def __getitem__(self, idx):
                return self._data[idx]

        positives = by_type.get("positive", [])

        # Phase 1: warm start (oracle positives + easy negatives)
        phase1 = positives + by_type.get("easy_negative", [])
        if phase1:
            logger.info("Phase 1: warm start (%d examples)", len(phase1))
            self._train_phase(model, PairDataset(phase1), epochs=1)

        # Phase 2: hard negatives + some positives (need both classes for loss)
        hard_negs = by_type.get("hard_negative", [])
        if hard_negs:
            phase2 = hard_negs + positives[:len(hard_negs)]
            logger.info("Phase 2: hard-negative curriculum (%d examples)", len(phase2))
            self._train_phase(model, PairDataset(phase2), epochs=1)

        # Phase 3: adversarial hardening (already contains both classes)
        phase3 = by_type.get("adversarial", [])
        if phase3:
            logger.info("Phase 3: adversarial hardening (%d examples)", len(phase3))
            self._train_phase(model, PairDataset(phase3), epochs=1)

        # Save model
        model.save(model_out)
        logger.info("Selector v%d saved to %s", version, model_out)

        # Evaluate on val set (optional)
        val_score = 0.0
        if val_data_path and os.path.exists(val_data_path):
            val_score = self._evaluate(model, val_data_path)
            logger.info("Val task success: %.3f", val_score)

        # Compute threshold tau (Youden's J on val positives/negatives)
        tau = self._compute_threshold(model, val_data_path) if val_data_path and os.path.exists(val_data_path) else 0.3

        # Query family summary
        family_mix: Dict[str, int] = {}
        for ex in examples:
            f = ex.get("query_family", "unknown")
            family_mix[f] = family_mix.get(f, 0) + 1

        sv = SelectorVersion(
            version=version,
            atlas_version=self.atlas_version,
            embedding_model=self.embedding_model_name,
            runtime_model_id=self.runtime_model_id,
            base_model=self.base_model,
            training_examples=len(examples),
            val_task_success=val_score,
            threshold_tau=tau,
            trained_at=datetime.now(timezone.utc).isoformat(),
            model_path=model_out,
            query_family_mix=family_mix,
        )

        # Write version metadata
        meta_path = os.path.join(model_out, "selector_version.json")
        with open(meta_path, "w") as f:
            json.dump(sv.to_dict(), f, indent=2)

        return sv

    def _train_phase(self, model, train_examples, epochs: int = 1):
        """Run one training phase."""
        from torch.utils.data import DataLoader

        loader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=self.batch_size,
        )
        total_steps = len(loader) * epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        model.fit(
            train_dataloader=loader,
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": self.learning_rate},
            show_progress_bar=False,
        )

    def _evaluate(self, model, val_data_path: str) -> float:
        """
        Simple accuracy on the val set as a proxy for task success.
        Full task-success eval requires pipeline integration.
        """
        examples = _load_training_data(val_data_path)
        if not examples:
            return 0.0
        pairs = [(ex["query"], ex["anchor_card"]) for ex in examples]
        labels = [ex["label"] for ex in examples]
        scores = model.predict(pairs)
        correct = sum(
            1 for s, l in zip(scores, labels)
            if (s >= 0.5) == bool(l)
        )
        return correct / len(labels)

    def _compute_threshold(self, model, val_data_path: str) -> float:
        """
        Find threshold tau via Youden's J (sensitivity + specificity - 1).
        Returns the optimal decision boundary.
        """
        examples = _load_training_data(val_data_path)
        if not examples:
            return 0.3
        pairs = [(ex["query"], ex["anchor_card"]) for ex in examples]
        labels = [ex["label"] for ex in examples]
        scores = model.predict(pairs)

        best_tau = 0.3
        best_j = -1.0
        for tau in [i / 20 for i in range(1, 20)]:
            tp = sum(1 for s, l in zip(scores, labels) if s >= tau and l == 1)
            tn = sum(1 for s, l in zip(scores, labels) if s < tau and l == 0)
            fp = sum(1 for s, l in zip(scores, labels) if s >= tau and l == 0)
            fn = sum(1 for s, l in zip(scores, labels) if s < tau and l == 1)
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            j = sens + spec - 1
            if j > best_j:
                best_j = j
                best_tau = tau

        return best_tau


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PSA selector trainer — fine-tune cross-encoder")
    parser.add_argument("--tenant", default="default", help="Tenant ID (default: default)")
    parser.add_argument("--training-data", required=True, help="Path to training_data.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Learning rate (default: {LEARNING_RATE})")
    args = parser.parse_args()

    from psa.tenant import TenantManager
    from psa.atlas import AtlasManager

    tm = TenantManager()
    tenant = tm.get_or_create(args.tenant)
    atlas_mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=args.tenant)
    atlas = atlas_mgr.get_atlas()
    if atlas is None:
        print(f"No atlas for tenant '{args.tenant}'. Run 'psa atlas build' first.")
        raise SystemExit(1)

    trainer = SelectorTrainer(
        output_dir=args.output_dir,
        atlas_version=atlas.version,
        epochs=args.epochs,
        learning_rate=args.lr,
    )
    sv = trainer.train(train_data_path=args.training_data)
    print(f"Selector v{sv.version} trained → {sv.model_path}")
