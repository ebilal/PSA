# 2026-04-19 Hyperparameter Sweep Results

This sweep used real tenant data from `default`.

- Selector evaluation data: `~/.psa/tenants/default/training/train_data.jsonl` and `val_data.jsonl`
- Selector metric: validation task-success proxy (`val_task_success`)
- Co-activation evaluation data: existing `~/.psa/tenants/default/training/coactivation/coactivation_train.npz`
- Co-activation metric: final validation loss
- Tie-break rule: use the bounded-sweep rule from the design spec

## Selector

Successful full runs:

| Candidate | LR | Batch | Epochs | Warmup | Val Score | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| `selector_01` | `2e-5` | 32 | 3 | 0.10 | 0.7588 | 1327.65 |
| `selector_02` | `1e-5` | 32 | 3 | 0.10 | 0.7303 | 1395.73 |
| `selector_03` | `3e-5` | 32 | 3 | 0.10 | 0.7688 | 1408.08 |

Unstable candidates on this macOS Apple Silicon laptop:

| Candidate | LR | Batch | Epochs | Warmup | Result |
|---|---:|---:|---:|---:|---|
| `selector_04` | `2e-5` | 32 | 3 | 0.05 | process exited during Phase 1 |
| `selector_05` | `2e-5` | 32 | 3 | 0.15 | process exited during Phase 1 |
| `selector_06` | `2e-5` | 16 | 4 | 0.10 | process exited during Phase 1 |

Chosen default: `learning_rate=3e-5`, `batch_size=32`, `epochs=3`, `warmup_ratio=0.1`

Reason: `selector_03` had the best validation score among the stable full runs, and no faster stable run landed within the selector tie margin.

## Co-activation

| Candidate | LR | Weight Decay | Batch | Epochs | Val Loss | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| `coactivation_01` | `1e-4` | 0.01 | 16 | 10 | 0.010235 | 36.09 |
| `coactivation_02` | `5e-5` | 0.01 | 16 | 10 | 0.010054 | 35.35 |
| `coactivation_03` | `2e-4` | 0.01 | 16 | 10 | 0.009277 | 35.40 |
| `coactivation_04` | `1e-4` | 0.00 | 16 | 10 | 0.009170 | 35.32 |
| `coactivation_05` | `1e-4` | 0.05 | 16 | 10 | 0.009234 | 35.31 |
| `coactivation_06` | `1e-4` | 0.01 | 32 | 10 | 0.011053 | 36.37 |
| `coactivation_07` | `1e-4` | 0.01 | 16 | 8 | 0.009351 | 28.30 |
| `coactivation_08` | `1e-4` | 0.01 | 16 | 12 | 0.008867 | 42.24 |

Chosen default: `learning_rate=1e-4`, `weight_decay=0.01`, `batch_size=16`, `epochs=8`

Reason: `coactivation_08` had the lowest raw validation loss, but `coactivation_07` was within the `0.002` loss margin and more than 20% faster, so the tie-break rule selected `coactivation_07`.
