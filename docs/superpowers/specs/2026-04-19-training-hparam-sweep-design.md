# Training Hyperparameter Sweep Defaults — Design Spec

## Problem

PSA currently uses fixed training defaults for both major learned components:

- the selector cross-encoder fine-tune in [psa/training/train_selector.py](/Users/erhanbilal/Work/Projects/memnexus/psa/training/train_selector.py)
- the co-activation model trainer in [psa/training/train_coactivation.py](/Users/erhanbilal/Work/Projects/memnexus/psa/training/train_coactivation.py)

Those defaults were chosen heuristically, not from measured results on the current tenant data. The user wants default values driven by actual experiments, with priority on validation performance, but still bounded by a realistic laptop runtime.

## Goal

Run a bounded local hyperparameter sweep for both trainers on real tenant data, choose the best validation-performing configuration that stays within a practical laptop training budget, and make those values the new code defaults.

## Non-goals

- building a general-purpose HPO framework
- distributed or cloud sweeps
- optimizing for fastest training
- changing model architectures
- changing training targets or loss definitions except where a hyperparameter needs to be surfaced explicitly

## Constraints

### Data source

All experiments use the current tenant's real training inputs:

- selector: generated train/validation JSONL from the existing oracle-label pipeline
- co-activation: generated `coactivation_train.npz` from the existing oracle labels and atlas

### Optimization target

Primary objective: best validation performance.

Secondary constraint: the selected defaults must fit within a realistic laptop budget. The sweep may include slower candidates, but the final chosen defaults should not make normal local retraining unreasonable.

### Runtime budget

Use a bounded search, not an open-ended combinatorial sweep.

Practical guardrails:

- selector candidate set: at most 6 runs
- co-activation candidate set: at most 8 runs
- no single default candidate should be accepted if its runtime is grossly worse than the current baseline for only marginal validation gain

The exact acceptance rule should be explicit in code or sweep output:

- pick the best validation run
- for selector, if another run is within `0.005` validation score of the best run and at least `20%` faster, prefer the faster one
- for co-activation, if another run is within `0.002` validation loss of the best run and at least `20%` faster, prefer the faster one

## Current behavior

### Selector

The selector trainer fine-tunes a downloaded pretrained model:

- base model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- behavior: PSA downloads that pretrained model, then fine-tunes it on PSA training pairs

Current default hyperparameters:

- `learning_rate = 2e-5`
- `batch_size = 32`
- `epochs = 3`
- `warmup_ratio = 0.1`

Validation signal today:

- `SelectorTrainer._evaluate()` computes validation accuracy on held-out pair labels
- threshold calibration is computed separately from validation examples

### Co-activation

The co-activation trainer trains PSA’s own transformer from scratch.

Current default hyperparameters:

- `learning_rate = 1e-4`
- `batch_size = 16`
- `epochs = 10`
- implicit AdamW defaults for `weight_decay`

Validation signal today:

- final validation loss computed after training

## Design

### 1. Surface the tunable hyperparameters explicitly

#### Selector trainer

Expose and persist the values needed for a bounded sweep:

- `learning_rate`
- `batch_size`
- `warmup_ratio`
- `epochs`

If the training library already fixes `weight_decay` internally and PSA cannot pass it cleanly without invasive changes, leave selector `weight_decay` out of scope for this pass. The goal is measured improvement with low implementation risk.

#### Co-activation trainer

Add explicit constructor or train-time parameters for:

- `learning_rate`
- `weight_decay`
- `batch_size`
- `epochs`

`weight_decay` must no longer be hidden inside `torch.optim.AdamW(...)` defaults if it is part of the sweep.

### 2. Add a small local sweep harness

Create a repo-local sweep utility that can:

- build or reuse the current tenant training inputs
- iterate over a bounded list of candidate hyperparameter combinations
- run one training job per candidate
- collect:
  - hyperparameters
  - validation metric
  - wall-clock runtime
  - output path
- emit a compact machine-readable results file

Recommended location:

- `psa/training/hparam_sweep.py`

This is a targeted evaluation harness, not a reusable framework abstraction layer.

### 3. Use trainer-native validation metrics

#### Selector ranking rule

Selector candidates are ranked by:

1. highest validation score from `SelectorTrainer`
2. if scores are effectively tied, lower runtime wins

For this pass, use the trainer’s existing validation score instead of inventing a new metric layer.

#### Co-activation ranking rule

Co-activation candidates are ranked by:

1. lowest validation loss
2. if losses are effectively tied, lower runtime wins

### 4. Candidate search spaces

Keep the sweep centered around current defaults.

#### Selector search

Recommended bounded grid:

- learning rate: `1e-5`, `2e-5`, `3e-5`
- warmup ratio: `0.05`, `0.1`, `0.15`
- batch size: `16`, `32`
- epochs: `3`, `4`

Do not run the full Cartesian product. Instead, run a staged or hand-pruned sweep of at most 6 candidates around current defaults.

Example pattern:

1. baseline
2. lower lr
3. higher lr
4. lower warmup
5. higher warmup
6. best-so-far plus either batch-size or epoch variant

#### Co-activation search

Recommended bounded grid:

- learning rate: `5e-5`, `1e-4`, `2e-4`
- weight decay: `0.0`, `0.01`, `0.05`
- batch size: `16`, `32`
- epochs: `8`, `10`, `12`

Again, do not run the full Cartesian product. Use a bounded set of at most 8 candidates centered around the current default.

### 5. Make the winning values the code defaults

After the sweep finishes:

- update selector defaults in `train_selector.py`
- update co-activation defaults in `train_coactivation.py`
- update any CLI help text or README wording that names default values

The committed defaults should reflect the chosen winner, not just the sweep script’s runtime arguments.

### 6. Persist the experiment results

Write the sweep summary to a dated document under `docs/` so the default changes are traceable.

Recommended contents:

- tenant id
- dataset sizes
- candidate hyperparameters
- validation metric for each run
- runtime for each run
- selected winner
- reason the winner was chosen

## Implementation sketch

### Selector

Minimal work:

- add or confirm constructor plumbing for the tunable selector parameters
- make sweep harness instantiate `SelectorTrainer(...)` with candidate values
- store each candidate under an isolated output directory

### Co-activation

Minimal work:

- add explicit `weight_decay` parameter to `CoActivationTrainer`
- pass it to `torch.optim.AdamW(...)`
- make sweep harness instantiate the trainer with candidate values

### Results format

Use a simple JSONL or JSON summary file per sweep run. Avoid adding a database or complex registry.

## Validation

Before changing defaults:

- confirm the sweep runner reproduces the current baseline values cleanly
- confirm candidate runs produce distinct recorded metrics

After changing defaults:

- run focused tests for both trainers
- run one real selector training smoke check with the chosen defaults
- run one real co-activation training smoke check with the chosen defaults

## Risks

### Overfitting to one tenant

This sweep is intentionally tenant-specific because the user asked for defaults based on real local performance. The resulting defaults are still global code defaults, so document that they are chosen from the current tenant’s data rather than from a broad benchmark suite.

### Validation metric mismatch

Selector validation accuracy is an imperfect proxy for downstream retrieval quality. That is acceptable for this pass because the request is to tune current training defaults, not redesign the selector objective.

### Sweep cost

Full selector retraining is expensive on CPU. The bounded candidate count is mandatory, not optional.

## Success criteria

- both trainers have an explicit, reproducible sweep path
- co-activation `weight_decay` is tunable
- the chosen defaults are backed by recorded experimental results
- the repo defaults are updated to the winning hyperparameters
- tests and one real smoke check pass after the change
