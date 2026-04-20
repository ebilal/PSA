"""Tests for the bounded training sweep harness."""

import pytest


def test_choose_best_selector_candidate_prefers_faster_tied_run():
    from psa.training.hparam_sweep import choose_best_selector_candidate

    candidates = [
        {
            "name": "baseline",
            "learning_rate": 2e-5,
            "batch_size": 32,
            "epochs": 3,
            "warmup_ratio": 0.10,
            "val_score": 0.820,
            "runtime_sec": 100.0,
        },
        {
            "name": "faster",
            "learning_rate": 1e-5,
            "batch_size": 32,
            "epochs": 3,
            "warmup_ratio": 0.10,
            "val_score": 0.818,
            "runtime_sec": 70.0,
        },
        {
            "name": "slower",
            "learning_rate": 3e-5,
            "batch_size": 32,
            "epochs": 3,
            "warmup_ratio": 0.10,
            "val_score": 0.821,
            "runtime_sec": 140.0,
        },
    ]

    best = choose_best_selector_candidate(candidates)
    assert best["name"] == "faster"


def test_choose_best_selector_candidate_is_order_independent():
    from psa.training.hparam_sweep import choose_best_selector_candidate

    candidates = [
        {
            "name": "metric_best",
            "learning_rate": 2e-5,
            "batch_size": 32,
            "epochs": 3,
            "warmup_ratio": 0.10,
            "val_score": 0.821,
            "runtime_sec": 100.0,
        },
        {
            "name": "faster_tied",
            "learning_rate": 1e-5,
            "batch_size": 32,
            "epochs": 3,
            "warmup_ratio": 0.10,
            "val_score": 0.818,
            "runtime_sec": 70.0,
        },
        {
            "name": "too_slow",
            "learning_rate": 3e-5,
            "batch_size": 32,
            "epochs": 3,
            "warmup_ratio": 0.10,
            "val_score": 0.820,
            "runtime_sec": 130.0,
        },
    ]

    forward_best = choose_best_selector_candidate(candidates)
    reverse_best = choose_best_selector_candidate(list(reversed(candidates)))

    assert forward_best["name"] == "faster_tied"
    assert reverse_best["name"] == "faster_tied"


def test_choose_best_selector_candidate_breaks_equal_runtime_ties_deterministically():
    from psa.training.hparam_sweep import choose_best_selector_candidate

    candidates = [
        {
            "name": "metric_best",
            "learning_rate": 2e-5,
            "batch_size": 32,
            "epochs": 3,
            "warmup_ratio": 0.10,
            "val_score": 0.821,
            "runtime_sec": 100.0,
        },
        {
            "name": "eligible_lower_score",
            "learning_rate": 1e-5,
            "batch_size": 32,
            "epochs": 3,
            "warmup_ratio": 0.10,
            "val_score": 0.818,
            "runtime_sec": 70.0,
        },
        {
            "name": "eligible_higher_score",
            "learning_rate": 3e-5,
            "batch_size": 32,
            "epochs": 3,
            "warmup_ratio": 0.10,
            "val_score": 0.819,
            "runtime_sec": 70.0,
        },
    ]

    forward_best = choose_best_selector_candidate(candidates)
    reverse_best = choose_best_selector_candidate(list(reversed(candidates)))

    assert forward_best["name"] == "eligible_higher_score"
    assert reverse_best["name"] == "eligible_higher_score"


def test_choose_best_selector_candidate_ignores_incidental_result_fields_in_stable_tie_break():
    from psa.training.hparam_sweep import choose_best_selector_candidate

    candidates = [
        {
            "name": "metric_best",
            "learning_rate": 2e-5,
            "batch_size": 32,
            "val_score": 0.821,
            "runtime_sec": 100.0,
            "model_path": "/tmp/metric-best",
            "kind": "selector",
            "debug_note": "ignore-me",
        },
        {
            "name": "z_candidate",
            "learning_rate": 1e-5,
            "batch_size": 16,
            "epochs": 4,
            "warmup_ratio": 0.10,
            "val_score": 0.819,
            "runtime_sec": 70.0,
            "model_path": "/tmp/a-path",
            "kind": "selector",
            "debug_note": "aaa",
        },
        {
            "name": "a_candidate",
            "learning_rate": 2e-5,
            "batch_size": 16,
            "epochs": 4,
            "warmup_ratio": 0.10,
            "val_score": 0.819,
            "runtime_sec": 70.0,
            "model_path": "/tmp/z-path",
            "kind": "selector",
            "debug_note": "zzz",
        },
    ]

    forward_best = choose_best_selector_candidate(candidates)
    reverse_best = choose_best_selector_candidate(list(reversed(candidates)))

    assert forward_best["name"] == "z_candidate"
    assert reverse_best["name"] == "z_candidate"


def test_choose_best_selector_candidate_rejects_duplicate_hyperparameter_keys():
    from psa.training.hparam_sweep import choose_best_selector_candidate

    candidates = [
        {
            "name": "first",
            "learning_rate": 2e-5,
            "batch_size": 16,
            "epochs": 4,
            "warmup_ratio": 0.10,
            "val_score": 0.819,
            "runtime_sec": 70.0,
            "model_path": "/tmp/first",
        },
        {
            "name": "second",
            "learning_rate": 2e-5,
            "batch_size": 16,
            "epochs": 4,
            "warmup_ratio": 0.10,
            "val_score": 0.819,
            "runtime_sec": 70.0,
            "model_path": "/tmp/second",
        },
    ]

    with pytest.raises(ValueError, match="duplicate selector candidate key"):
        choose_best_selector_candidate(candidates)


def test_choose_best_coactivation_candidate_keeps_metric_best_when_only_faster_run_is_outside_tie_margin():
    from psa.training.hparam_sweep import choose_best_coactivation_candidate

    candidates = [
        {
            "name": "baseline",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "batch_size": 16,
            "epochs": 10,
            "val_loss": 0.1200,
            "runtime_sec": 100.0,
        },
        {
            "name": "faster",
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "batch_size": 16,
            "epochs": 10,
            "val_loss": 0.1215,
            "runtime_sec": 70.0,
        },
        {
            "name": "best",
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "batch_size": 16,
            "epochs": 10,
            "val_loss": 0.1190,
            "runtime_sec": 120.0,
        },
    ]

    best = choose_best_coactivation_candidate(candidates)
    assert best["name"] == "best"


def test_choose_best_coactivation_candidate_is_order_independent():
    from psa.training.hparam_sweep import choose_best_coactivation_candidate

    candidates = [
        {
            "name": "metric_best",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "batch_size": 16,
            "epochs": 10,
            "val_loss": 0.1200,
            "runtime_sec": 100.0,
        },
        {
            "name": "faster_tied",
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "batch_size": 16,
            "epochs": 10,
            "val_loss": 0.1215,
            "runtime_sec": 70.0,
        },
        {
            "name": "too_slow",
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "batch_size": 16,
            "epochs": 10,
            "val_loss": 0.1205,
            "runtime_sec": 90.0,
        },
    ]

    forward_best = choose_best_coactivation_candidate(candidates)
    reverse_best = choose_best_coactivation_candidate(list(reversed(candidates)))

    assert forward_best["name"] == "faster_tied"
    assert reverse_best["name"] == "faster_tied"


def test_choose_best_coactivation_candidate_breaks_equal_runtime_ties_deterministically():
    from psa.training.hparam_sweep import choose_best_coactivation_candidate

    candidates = [
        {
            "name": "metric_best",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "batch_size": 16,
            "epochs": 10,
            "val_loss": 0.1200,
            "runtime_sec": 100.0,
        },
        {
            "name": "eligible_higher_loss",
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "batch_size": 16,
            "epochs": 10,
            "val_loss": 0.1215,
            "runtime_sec": 70.0,
        },
        {
            "name": "eligible_lower_loss",
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "batch_size": 16,
            "epochs": 10,
            "val_loss": 0.1205,
            "runtime_sec": 70.0,
        },
    ]

    forward_best = choose_best_coactivation_candidate(candidates)
    reverse_best = choose_best_coactivation_candidate(list(reversed(candidates)))

    assert forward_best["name"] == "eligible_lower_loss"
    assert reverse_best["name"] == "eligible_lower_loss"


def test_choose_best_coactivation_candidate_ignores_incidental_result_fields_in_stable_tie_break():
    from psa.training.hparam_sweep import choose_best_coactivation_candidate

    candidates = [
        {
            "name": "metric_best",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "batch_size": 16,
            "epochs": 10,
            "val_loss": 0.1200,
            "runtime_sec": 100.0,
            "model_path": "/tmp/metric-best",
            "kind": "coactivation",
            "debug_note": "ignore-me",
        },
        {
            "name": "z_candidate",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "batch_size": 32,
            "epochs": 10,
            "val_loss": 0.1210,
            "runtime_sec": 70.0,
            "model_path": "/tmp/a-path",
            "kind": "coactivation",
            "debug_note": "aaa",
        },
        {
            "name": "a_candidate",
            "learning_rate": 5e-5,
            "weight_decay": 0.05,
            "batch_size": 16,
            "epochs": 10,
            "val_loss": 0.1210,
            "runtime_sec": 70.0,
            "model_path": "/tmp/z-path",
            "kind": "coactivation",
            "debug_note": "zzz",
        },
    ]

    forward_best = choose_best_coactivation_candidate(candidates)
    reverse_best = choose_best_coactivation_candidate(list(reversed(candidates)))

    assert forward_best["name"] == "a_candidate"
    assert reverse_best["name"] == "a_candidate"


def test_choose_best_coactivation_candidate_rejects_duplicate_hyperparameter_keys():
    from psa.training.hparam_sweep import choose_best_coactivation_candidate

    candidates = [
        {
            "name": "first",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "batch_size": 16,
            "epochs": 10,
            "val_loss": 0.1210,
            "runtime_sec": 70.0,
            "model_path": "/tmp/first",
        },
        {
            "name": "second",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "batch_size": 16,
            "epochs": 10,
            "val_loss": 0.1210,
            "runtime_sec": 70.0,
            "model_path": "/tmp/second",
        },
    ]

    with pytest.raises(ValueError, match="duplicate coactivation candidate key"):
        choose_best_coactivation_candidate(candidates)


def test_write_results_json(tmp_path):
    from psa.training.hparam_sweep import write_sweep_results

    out = tmp_path / "results.json"
    rows = [{"name": "baseline", "val_score": 0.8, "runtime_sec": 10.0}]

    write_sweep_results(str(out), rows)

    assert out.exists()
    assert '"name": "baseline"' in out.read_text()
