from __future__ import annotations

import json
import os
import time
from typing import Any


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
    return {
        "kind": "selector",
        **trainer_kwargs,
        "runtime_sec": runtime_sec,
        "val_score": version.val_task_success,
        "model_path": version.model_path,
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
    started = time.perf_counter()
    trainer = trainer_cls(output_dir=output_dir, **trainer_kwargs)
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
    return {
        "kind": "coactivation",
        **trainer_kwargs,
        "runtime_sec": runtime_sec,
        "val_loss": meta["validation_loss"],
        "model_path": output_dir,
    }
