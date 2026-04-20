from __future__ import annotations

import json
import os
import time
import gc
from datetime import datetime, timezone
from typing import Any

import numpy as np


SELECTOR_TIE_MARGIN = 0.005
COACTIVATION_TIE_MARGIN = 0.002
SPEEDUP_THRESHOLD = 0.20

SELECTOR_KEY_FIELDS = ("learning_rate", "batch_size", "epochs", "warmup_ratio")
COACTIVATION_KEY_FIELDS = ("learning_rate", "weight_decay", "batch_size", "epochs")


DEFAULT_SELECTOR_CANDIDATES = [
    {"learning_rate": 2e-5, "batch_size": 32, "epochs": 3, "warmup_ratio": 0.10},
    {"learning_rate": 1e-5, "batch_size": 32, "epochs": 3, "warmup_ratio": 0.10},
    {"learning_rate": 3e-5, "batch_size": 32, "epochs": 3, "warmup_ratio": 0.10},
    {"learning_rate": 2e-5, "batch_size": 32, "epochs": 3, "warmup_ratio": 0.05},
    {"learning_rate": 2e-5, "batch_size": 32, "epochs": 3, "warmup_ratio": 0.15},
    {"learning_rate": 2e-5, "batch_size": 16, "epochs": 4, "warmup_ratio": 0.10},
]

DEFAULT_COACTIVATION_CANDIDATES = [
    {"learning_rate": 1e-4, "weight_decay": 0.01, "batch_size": 16, "epochs": 10},
    {"learning_rate": 5e-5, "weight_decay": 0.01, "batch_size": 16, "epochs": 10},
    {"learning_rate": 2e-4, "weight_decay": 0.01, "batch_size": 16, "epochs": 10},
    {"learning_rate": 1e-4, "weight_decay": 0.0, "batch_size": 16, "epochs": 10},
    {"learning_rate": 1e-4, "weight_decay": 0.05, "batch_size": 16, "epochs": 10},
    {"learning_rate": 1e-4, "weight_decay": 0.01, "batch_size": 32, "epochs": 10},
    {"learning_rate": 1e-4, "weight_decay": 0.01, "batch_size": 16, "epochs": 8},
    {"learning_rate": 1e-4, "weight_decay": 0.01, "batch_size": 16, "epochs": 12},
]


def _stable_candidate_key(candidate: dict[str, Any], fields: tuple[str, ...]) -> str:
    hyperparameter_fields = {key: candidate.get(key) for key in fields}
    return json.dumps(hyperparameter_fields, sort_keys=True)


def _validate_unique_candidate_keys(
    candidates: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
    label: str,
) -> None:
    seen_keys: set[str] = set()
    for candidate in candidates:
        candidate_key = _stable_candidate_key(candidate, fields)
        if candidate_key in seen_keys:
            raise ValueError(f"duplicate {label} candidate key: {candidate_key}")
        seen_keys.add(candidate_key)


def choose_best_selector_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidates must not be empty")
    _validate_unique_candidate_keys(
        candidates,
        fields=SELECTOR_KEY_FIELDS,
        label="selector",
    )

    metric_best = max(candidates, key=lambda candidate: candidate["val_score"])
    speedup_cutoff = metric_best["runtime_sec"] * (1.0 - SPEEDUP_THRESHOLD)
    tied_candidates = [
        candidate
        for candidate in candidates
        if (metric_best["val_score"] - candidate["val_score"]) <= SELECTOR_TIE_MARGIN
        and candidate["runtime_sec"] <= speedup_cutoff
    ]
    if not tied_candidates:
        return metric_best
    return min(
        tied_candidates,
        key=lambda candidate: (
            candidate["runtime_sec"],
            -candidate["val_score"],
            _stable_candidate_key(candidate, SELECTOR_KEY_FIELDS),
        ),
    )


def choose_best_coactivation_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidates must not be empty")
    _validate_unique_candidate_keys(
        candidates,
        fields=COACTIVATION_KEY_FIELDS,
        label="coactivation",
    )

    metric_best = min(candidates, key=lambda candidate: candidate["val_loss"])
    speedup_cutoff = metric_best["runtime_sec"] * (1.0 - SPEEDUP_THRESHOLD)
    tied_candidates = [
        candidate
        for candidate in candidates
        if (candidate["val_loss"] - metric_best["val_loss"]) <= COACTIVATION_TIE_MARGIN
        and candidate["runtime_sec"] <= speedup_cutoff
    ]
    if not tied_candidates:
        return metric_best
    return min(
        tied_candidates,
        key=lambda candidate: (
            candidate["runtime_sec"],
            candidate["val_loss"],
            _stable_candidate_key(candidate, COACTIVATION_KEY_FIELDS),
        ),
    )


def write_sweep_results(path: str, rows: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(rows, handle, indent=2)


def write_sweep_summary(path: str, payload: dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def load_sweep_results(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return json.load(handle)


def _count_nonempty_lines(path: str) -> int:
    with open(path) as handle:
        return sum(1 for line in handle if line.strip())


def _candidate_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:02d}"


def _release_torch_caches() -> None:
    """Best-effort cleanup between runs in a long-lived sweep process."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _rows_by_candidate(path: str) -> dict[str, dict[str, Any]]:
    rows = load_sweep_results(path)
    return {row["candidate_id"]: row for row in rows if "candidate_id" in row}


def _upsert_candidate_result(path: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = load_sweep_results(path)
    updated = False
    for index, existing in enumerate(rows):
        if existing.get("candidate_id") == row.get("candidate_id"):
            rows[index] = row
            updated = True
            break
    if not updated:
        rows.append(row)
    rows.sort(key=lambda candidate: candidate.get("candidate_id", ""))
    write_sweep_results(path, rows)
    return rows


def run_selector_candidate(
    trainer_cls,
    *,
    trainer_kwargs: dict[str, Any],
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    trainer = trainer_cls(output_dir=output_dir, **trainer_kwargs)
    version = trainer.train(train_data_path=train_data_path, val_data_path=val_data_path)
    runtime_sec = time.perf_counter() - started
    del trainer
    _release_torch_caches()
    return {
        "kind": "selector",
        **trainer_kwargs,
        "runtime_sec": runtime_sec,
        "val_score": version.val_task_success,
        "model_path": version.model_path,
        "threshold_tau": version.threshold_tau,
        "training_examples": version.training_examples,
    }


def run_coactivation_candidate(
    trainer_cls,
    *,
    trainer_kwargs: dict[str, Any],
    data_dir: str,
    n_anchors: int,
    output_dir: str,
    centroid_dim: int = 768,
) -> dict[str, Any]:
    init_kwargs = {
        "learning_rate": trainer_kwargs["learning_rate"],
        "weight_decay": trainer_kwargs["weight_decay"],
    }
    started = time.perf_counter()
    trainer = trainer_cls(output_dir=output_dir, **init_kwargs)
    trainer.train(
        data_dir=data_dir,
        n_anchors=n_anchors,
        centroid_dim=centroid_dim,
        epochs=trainer_kwargs["epochs"],
        batch_size=trainer_kwargs["batch_size"],
    )
    runtime_sec = time.perf_counter() - started
    with open(os.path.join(output_dir, "coactivation_version.json")) as handle:
        meta = json.load(handle)
    del trainer
    _release_torch_caches()
    return {
        "kind": "coactivation",
        **trainer_kwargs,
        "runtime_sec": runtime_sec,
        "val_loss": meta["validation_loss"],
        "model_path": output_dir,
        "training_examples": meta["training_examples"],
    }


def run_default_sweep(tenant_id: str = "default") -> dict[str, Any]:
    from psa.atlas import AtlasManager
    from psa.embeddings import EmbeddingModel
    from psa.full_atlas_scorer import FullAtlasScorer
    from psa.tenant import TenantManager
    from psa.training.coactivation_data import generate_coactivation_data
    from psa.training.data_generator import DataGenerator
    from psa.training.data_split import split_train_val
    from psa.training.train_coactivation import CoActivationTrainer
    from psa.training.train_selector import SelectorTrainer

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    atlas_mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
    atlas = atlas_mgr.get_atlas()
    if atlas is None:
        raise FileNotFoundError(f"No atlas for tenant '{tenant_id}'. Run 'psa atlas build' first.")

    labels_path = os.path.join(tenant.root_dir, "training", "oracle_labels.jsonl")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"No oracle labels found for tenant '{tenant_id}' at {labels_path}. Run 'psa label' first."
        )
    label_count = _count_nonempty_lines(labels_path)
    if label_count == 0:
        raise ValueError(f"Oracle label file is empty: {labels_path}")

    training_dir = os.path.join(tenant.root_dir, "training")
    examples_path = os.path.join(training_dir, "training_data.jsonl")
    train_path = os.path.join(training_dir, "train_data.jsonl")
    val_path = os.path.join(training_dir, "val_data.jsonl")

    anchor_cards = {card.anchor_id: card.to_stable_card_text() for card in atlas.cards}
    generator = DataGenerator(oracle_labels_path=labels_path, anchor_cards=anchor_cards)
    n_generated = generator.generate(
        output_path=examples_path,
        n_examples=max(1000, label_count * 20),
    )
    if n_generated == 0:
        raise ValueError(f"No selector training examples were generated from {labels_path}")
    split_stats = split_train_val(examples_path, train_path, val_path)

    sweep_root = os.path.join(tenant.root_dir, "training", "sweeps", "2026-04-19-defaults")
    selector_root = os.path.join(sweep_root, "selector")
    coactivation_root = os.path.join(sweep_root, "coactivation")
    coactivation_data_dir = os.path.join(sweep_root, "coactivation_data")
    selector_results_path = os.path.join(sweep_root, "selector_results.json")
    coactivation_results_path = os.path.join(sweep_root, "coactivation_results.json")
    os.makedirs(selector_root, exist_ok=True)
    os.makedirs(coactivation_root, exist_ok=True)

    print(f"Running selector sweep for tenant '{tenant_id}' on atlas v{atlas.version}...", flush=True)
    selector_rows_by_candidate = _rows_by_candidate(selector_results_path)
    for index, candidate in enumerate(DEFAULT_SELECTOR_CANDIDATES, start=1):
        candidate_id = _candidate_id("selector", index)
        if candidate_id in selector_rows_by_candidate:
            row = selector_rows_by_candidate[candidate_id]
            print(
                "  "
                f"{candidate_id}: reusing val_score={row['val_score']:.4f}, "
                f"runtime={row['runtime_sec']:.2f}s",
                flush=True,
            )
        else:
            row = run_selector_candidate(
                SelectorTrainer,
                trainer_kwargs={"atlas_version": atlas.version, **candidate},
                train_data_path=train_path,
                val_data_path=val_path,
                output_dir=os.path.join(selector_root, candidate_id),
            )
            row["candidate_id"] = candidate_id
            _upsert_candidate_result(selector_results_path, row)
            selector_rows_by_candidate[candidate_id] = row
            print(
                "  "
                f"{candidate_id}: val_score={row['val_score']:.4f}, "
                f"runtime={row['runtime_sec']:.2f}s, "
                f"lr={row['learning_rate']}, batch={row['batch_size']}, "
                f"epochs={row['epochs']}, warmup={row['warmup_ratio']}",
                flush=True,
            )

    selector_rows = [
        selector_rows_by_candidate[_candidate_id("selector", index)]
        for index in range(1, len(DEFAULT_SELECTOR_CANDIDATES) + 1)
    ]
    selector_winner = choose_best_selector_candidate(selector_rows)
    print(
        "Selector winner: "
        f"{selector_winner['candidate_id']} "
        f"(val_score={selector_winner['val_score']:.4f}, runtime={selector_winner['runtime_sec']:.2f}s)",
        flush=True,
    )

    full_atlas_scorer = FullAtlasScorer.from_model_path(selector_winner["model_path"], atlas)
    embedding_model = EmbeddingModel()
    n_coactivation_examples = generate_coactivation_data(
        oracle_labels_path=labels_path,
        output_path=coactivation_data_dir,
        full_atlas_scorer=full_atlas_scorer,
        embedding_model=embedding_model,
        atlas=atlas,
    )
    if n_coactivation_examples == 0:
        raise ValueError(f"No co-activation training examples were generated from {labels_path}")

    coactivation_npz_path = os.path.join(coactivation_data_dir, "coactivation_train.npz")
    with np.load(coactivation_npz_path) as coactivation_npz:
        centroid_dim = int(coactivation_npz["centroids"].shape[1])
        coactivation_anchor_count = int(coactivation_npz["ce_scores"].shape[1])
        anchor_feature_dim = (
            int(coactivation_npz["anchor_features"].shape[1])
            if "anchor_features" in coactivation_npz.files
            else 0
        )

    print(
        "Running co-activation sweep "
        f"({n_coactivation_examples} examples, {coactivation_anchor_count} anchors)...",
        flush=True,
    )
    coactivation_rows_by_candidate = _rows_by_candidate(coactivation_results_path)
    for index, candidate in enumerate(DEFAULT_COACTIVATION_CANDIDATES, start=1):
        candidate_id = _candidate_id("coactivation", index)
        if candidate_id in coactivation_rows_by_candidate:
            row = coactivation_rows_by_candidate[candidate_id]
            print(
                "  "
                f"{candidate_id}: reusing val_loss={row['val_loss']:.4f}, "
                f"runtime={row['runtime_sec']:.2f}s",
                flush=True,
            )
        else:
            row = run_coactivation_candidate(
                CoActivationTrainer,
                trainer_kwargs=dict(candidate),
                data_dir=coactivation_data_dir,
                n_anchors=coactivation_anchor_count,
                output_dir=os.path.join(coactivation_root, candidate_id),
                centroid_dim=centroid_dim,
            )
            row["candidate_id"] = candidate_id
            _upsert_candidate_result(coactivation_results_path, row)
            coactivation_rows_by_candidate[candidate_id] = row
            print(
                "  "
                f"{candidate_id}: val_loss={row['val_loss']:.4f}, "
                f"runtime={row['runtime_sec']:.2f}s, "
                f"lr={row['learning_rate']}, weight_decay={row['weight_decay']}, "
                f"batch={row['batch_size']}, epochs={row['epochs']}",
                flush=True,
            )

    coactivation_rows = [
        coactivation_rows_by_candidate[_candidate_id("coactivation", index)]
        for index in range(1, len(DEFAULT_COACTIVATION_CANDIDATES) + 1)
    ]
    coactivation_winner = choose_best_coactivation_candidate(coactivation_rows)
    print(
        "Co-activation winner: "
        f"{coactivation_winner['candidate_id']} "
        f"(val_loss={coactivation_winner['val_loss']:.4f}, runtime={coactivation_winner['runtime_sec']:.2f}s)",
        flush=True,
    )

    summary = {
        "tenant_id": tenant_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sweep_root": sweep_root,
        "atlas_version": atlas.version,
        "selector_dataset": {
            "oracle_labels": label_count,
            "generated_examples": n_generated,
            **split_stats,
        },
        "coactivation_dataset": {
            "examples": n_coactivation_examples,
            "n_anchors": coactivation_anchor_count,
            "centroid_dim": centroid_dim,
            "anchor_feature_dim": anchor_feature_dim,
        },
        "selector_results_path": selector_results_path,
        "coactivation_results_path": coactivation_results_path,
        "selector_winner": selector_winner,
        "coactivation_winner": coactivation_winner,
    }
    write_sweep_summary(os.path.join(sweep_root, "summary.json"), summary)

    print("Sweep artifacts written to:", sweep_root, flush=True)
    print(
        "Winning defaults:",
        json.dumps(
            {
                "selector": {
                    key: selector_winner[key]
                    for key in SELECTOR_KEY_FIELDS
                },
                "coactivation": {
                    key: coactivation_winner[key]
                    for key in COACTIVATION_KEY_FIELDS
                },
            },
            indent=2,
        ),
        sep="\n",
        flush=True,
    )
    return summary
