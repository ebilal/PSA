# Two-Level Memory Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance the memory activation system with enriched Level 1 anchor features (type distribution, metadata) and a new Level 2 memory-level scorer that replaces the packer's heuristic ranking. Fix MPS device handling to achieve ~340ms latency. Targets: R@5 >= 0.93, F1 >= 0.22, LLM-judge >= 0.40.

**Architecture:** Level 1 co-activation model gets 8 new per-anchor features (type distribution, count, quality). Level 2 adds a MemoryScorer: cross-encoder scores (query, memory_body) pairs, then a tiny MLP re-ranker (11-dim input, ~400 params) produces final relevance scores. The packer fills the token budget in Level 2's order with no heuristic re-ranking.

**Tech Stack:** PyTorch, sentence-transformers, existing PSA infrastructure. All models load on MPS (Apple Silicon).

**Design spec:** `docs/superpowers/specs/2026-04-13-two-level-memory-activation-design.md`

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `psa/memory_scorer.py` | `MemoryReRanker` (MLP nn.Module) + `MemoryScorer` (inference wrapper) + `ScoredMemory` dataclass |
| `psa/training/memory_scorer_data.py` | Generate memory-level relevance labels via LLM or text overlap |
| `psa/training/train_memory_scorer.py` | Training loop for the re-ranker MLP |
| `tests/test_memory_scorer.py` | Unit tests for MemoryScorer |
| `tests/test_memory_scorer_training.py` | Unit tests for memory scorer data + training |

### Modified files

| File | Change |
|------|--------|
| `psa/anchor.py` | Add `type_distribution`, `avg_quality`, `memory_count_norm` fields to AnchorCard |
| `psa/atlas.py` | Compute per-anchor features at build time |
| `psa/coactivation.py` | Extend `anchor_proj` input dim (+8 features) |
| `psa/training/coactivation_data.py` | Include anchor features in npz |
| `psa/training/train_coactivation.py` | Load and pass anchor features during training |
| `psa/full_atlas_scorer.py` | Change default device from "cpu" to "mps" |
| `psa/pipeline.py` | Add MemoryScorer to query flow; fix MPS device; pass `pre_ranked` to packer |
| `psa/packer.py` | Add `pre_ranked` parameter to `pack_memories_direct()` |
| `psa/cli.py` | Fix MPS device; add `--memory-scorer` flag |
| `psa/lifecycle.py` | Fix MPS device; add memory scorer training to slow path |

---

### Task 1: Fix MPS Device Handling

The SIGSEGV was caused by mixed devices (embedding on MPS, cross-encoder on CPU). Fix: load everything on MPS consistently.

**Files:**
- Modify: `psa/full_atlas_scorer.py`
- Modify: `psa/pipeline.py`
- Modify: `psa/cli.py`
- Modify: `psa/lifecycle.py`
- Modify: `psa/training/coactivation_data.py`
- Modify: `tests/test_full_atlas_scorer.py`

- [ ] **Step 1: Fix full_atlas_scorer.py default device**

In `_load_cross_encoder`, change default from `"cpu"` to `"mps"`:

```python
def _load_cross_encoder(model_path: str, device: str = "mps"):
    """Load a cross-encoder from disk.

    Parameters
    ----------
    model_path: path to the saved model
    device: "mps", "cpu", or "cuda". Default "mps" for Apple Silicon.
        IMPORTANT: must match the device used by EmbeddingModel to avoid
        SIGSEGV from mixed MPS/CPU state.
    """
```

Change `from_model_path` default:

```python
@classmethod
def from_model_path(cls, model_path: str, atlas, device: str = "mps") -> "FullAtlasScorer":
```

- [ ] **Step 2: Fix pipeline.py device detection**

In `from_tenant()`, change the device detection for both FullAtlasScorer and CoActivationSelector to use MPS when available:

```python
        import torch as _torch
        _device = "mps" if _torch.backends.mps.is_available() else "cpu"
```

Use `_device` for both `FullAtlasScorer.from_model_path(..., device=_device)` and `CoActivationSelector.from_model_path(..., device=_device)`.

- [ ] **Step 3: Fix cli.py -- remove torch.set_default_device workaround**

In `cmd_train()` co-activation block, remove `_torch.set_default_device("cpu")`. Remove `device="cpu"` from `FullAtlasScorer.from_model_path` call (uses new MPS default).

- [ ] **Step 4: Fix coactivation_data.py -- remove _ensure_cpu_default**

Delete the `_ensure_cpu_default()` function and its call in `generate_coactivation_data()`.

- [ ] **Step 5: Fix test assertion**

In `tests/test_full_atlas_scorer.py`, update:

```python
    mock_load.assert_called_once_with("/fake/model/path", device="mps")
```

- [ ] **Step 6: Run tests and commit**

Run: `uv run pytest tests/test_full_atlas_scorer.py tests/test_coactivation.py -v`
Expected: All pass

```bash
git add psa/full_atlas_scorer.py psa/pipeline.py psa/cli.py psa/lifecycle.py psa/training/coactivation_data.py tests/test_full_atlas_scorer.py
git commit -m "fix: use MPS consistently for all models, remove CPU workaround"
```

---

### Task 2: Add Anchor Features to AnchorCard + Atlas

**Files:**
- Modify: `psa/anchor.py`
- Modify: `psa/atlas.py`

- [ ] **Step 1: Add fields to AnchorCard**

In `psa/anchor.py`, add after `query_fingerprint`:

```python
    type_distribution: List[float] = field(default_factory=lambda: [0.0] * 6)
    avg_quality: float = 0.0
    memory_count_norm: float = 0.0
```

- [ ] **Step 2: Compute features in atlas build**

In `psa/atlas.py`, in the card generation loop, after `card.memory_count = len(cluster_mems)`:

```python
            if cluster_mems:
                from .memory_object import MemoryType
                type_order = [
                    MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL,
                    MemoryType.FAILURE, MemoryType.TOOL_USE, MemoryType.WORKING_DERIVATIVE,
                ]
                type_counts = [
                    sum(1 for m in cluster_mems if m.memory_type == t) for t in type_order
                ]
                total = max(sum(type_counts), 1)
                card.type_distribution = [c / total for c in type_counts]
                card.avg_quality = sum(m.quality_score for m in cluster_mems) / len(cluster_mems)
```

After all cards built, normalize memory_count:

```python
        max_count = max((c.memory_count for c in cards), default=1)
        for c in cards:
            c.memory_count_norm = c.memory_count / max(max_count, 1)
```

- [ ] **Step 3: Run tests and commit**

Run: `uv run pytest tests/test_anchor.py tests/test_atlas.py -v`

```bash
git add psa/anchor.py psa/atlas.py
git commit -m "feat: add type_distribution, avg_quality, memory_count_norm to AnchorCard"
```

---

### Task 3: Update Co-Activation Model for Enriched Features

**Files:**
- Modify: `psa/coactivation.py`
- Modify: `tests/test_coactivation.py`

- [ ] **Step 1: Add anchor_feature_dim to CoActivationModel.__init__**

```python
    def __init__(
        self,
        n_anchors: int = 256,
        centroid_dim: int = 768,
        anchor_feature_dim: int = 8,
        d_model: int = 256,
        ...
    ):
        ...
        self.anchor_feature_dim = anchor_feature_dim
        self.anchor_proj = nn.Linear(1 + centroid_dim + anchor_feature_dim, half)
```

- [ ] **Step 2: Update forward() with optional anchor_features parameter**

```python
    def forward(self, ce_scores, centroids, query_vec, anchor_features=None):
        B, N, D = centroids.shape
        ce = ce_scores.unsqueeze(-1)

        if anchor_features is not None:
            anchor_input = torch.cat([ce, centroids, anchor_features], dim=-1)
        else:
            anchor_input = torch.cat([
                ce, centroids,
                torch.zeros(B, N, self.anchor_feature_dim, device=ce.device),
            ], dim=-1)

        anchor_tokens = self.anchor_proj(anchor_input)
```

- [ ] **Step 3: Update CoActivationSelector.select() to accept anchor_features**

Add `anchor_features: Optional[np.ndarray] = None` parameter. Build tensor and pass to model.

- [ ] **Step 4: Update from_model_path to read anchor_feature_dim from metadata**

- [ ] **Step 5: Add tests for new feature path + backward compat**

- [ ] **Step 6: Run tests and commit**

```bash
git add psa/coactivation.py tests/test_coactivation.py
git commit -m "feat: extend co-activation model with per-anchor type/quality features"
```

---

### Task 4: Update Co-Activation Training Data + Trainer

**Files:**
- Modify: `psa/training/coactivation_data.py`
- Modify: `psa/training/train_coactivation.py`

- [ ] **Step 1: Add anchor features to npz in coactivation_data.py**

Build features array from atlas cards and add to npz save:

```python
    anchor_features = np.array(
        [card.type_distribution + [card.avg_quality, card.memory_count_norm] for card in cards],
        dtype=np.float32,
    )
```

Add `anchor_features=anchor_features` to `np.savez` call.

- [ ] **Step 2: Update train_coactivation.py to load and pass features**

Load `anchor_features` from npz (with None fallback for old data). Expand per batch. Pass to model forward call.

Save `anchor_feature_dim` in version JSON.

- [ ] **Step 3: Run tests and commit**

```bash
git add psa/training/coactivation_data.py psa/training/train_coactivation.py
git commit -m "feat: include anchor features in co-activation training"
```

---

### Task 5: MemoryScorer -- Tests

**Files:**
- Create: `tests/test_memory_scorer.py`

- [ ] **Step 1: Write failing tests**

4 tests:
1. `test_forward_output_shape` -- MemoryReRanker(input_dim=11) on (20, 11) input returns (20, 1) in [0,1]
2. `test_tiny_parameter_count` -- under 1000 params
3. `test_score_returns_sorted_scored_memories` -- mock CE and reranker, verify 3 memories sorted by score
4. `test_score_with_empty_memories` -- returns empty list

Use `MagicMock` for cross_encoder and reranker. Mock MemoryObjects with body, memory_type, quality_score, created_at, embedding.

- [ ] **Step 2: Verify fail, commit**

```bash
git add tests/test_memory_scorer.py
git commit -m "test: add failing tests for MemoryScorer"
```

---

### Task 6: MemoryScorer -- Implementation

**Files:**
- Create: `psa/memory_scorer.py`

- [ ] **Step 1: Implement ScoredMemory, MemoryReRanker, MemoryScorer**

`MemoryReRanker(nn.Module)`: Linear(11,32) -> ReLU -> Linear(32,1) -> Sigmoid

`MemoryScorer`:
- `score(query, query_vec, memories)`: Stage A cross-encoder on (query, body) pairs batched at 64. Stage B: build 11-dim feature vector [ce_score, type_onehot(6), quality, body_norm, recency, cosine]. Run through reranker. Return sorted ScoredMemory list.
- `from_model_path(model_path, cross_encoder, device)`: load weights + metadata

- [ ] **Step 2: Run tests, commit**

```bash
git add psa/memory_scorer.py tests/test_memory_scorer.py
git commit -m "feat: add MemoryScorer -- Level 2 memory-level re-ranking"
```

---

### Task 7: Memory Scorer Training Data + Training Loop

**Files:**
- Create: `psa/training/memory_scorer_data.py`
- Create: `psa/training/train_memory_scorer.py`
- Create: `tests/test_memory_scorer_training.py`

- [ ] **Step 1: Implement memory_scorer_data.py**

`generate_memory_scorer_data(results_path, output_path, pipeline, mode="benchmark")`:
- mode="benchmark": token F1 overlap between memory.body and gold_answer, threshold 0.3
- mode="llm": call_llm for relevance judgment (cloud API with Ollama fallback)
- Outputs JSONL with {ce_score, type_vec, quality_score, body_norm, recency, cosine, label}

- [ ] **Step 2: Implement train_memory_scorer.py**

`MemoryScorerTrainer.train(data_path, epochs=20, batch_size=64)`:
- Load JSONL, build tensors, train/val split, BCE loss, save model + metadata

- [ ] **Step 3: Write test, run, commit**

```bash
git add psa/training/memory_scorer_data.py psa/training/train_memory_scorer.py tests/test_memory_scorer_training.py
git commit -m "feat: add memory scorer training pipeline"
```

---

### Task 8: Pipeline Integration

**Files:**
- Modify: `psa/pipeline.py`
- Modify: `psa/packer.py`

- [ ] **Step 1: Add pre_ranked to packer**

`pack_memories_direct(..., pre_ranked=False)`: when True, skip 0.7*cosine+0.3*quality sort, pack in received order.

- [ ] **Step 2: Add MemoryScorer to pipeline query()**

After `_fetch_memories()`, before synthesize/pack: if `self.memory_scorer` exists, call it, re-order memories, set `_pre_ranked=True`.

- [ ] **Step 3: Load MemoryScorer in from_tenant()**

Load from `{tenant}/models/memory_scorer_latest/` if present. Reuse cross-encoder from FullAtlasScorer.

- [ ] **Step 4: Run tests, commit**

```bash
git add psa/pipeline.py psa/packer.py
git commit -m "feat: integrate MemoryScorer into pipeline with pre_ranked packer"
```

---

### Task 9: CLI + Lifecycle Integration

**Files:**
- Modify: `psa/cli.py`
- Modify: `psa/lifecycle.py`

- [ ] **Step 1: Add --memory-scorer to psa train argparse + cmd_train**

- [ ] **Step 2: Add memory scorer training to lifecycle slow path**

- [ ] **Step 3: Run tests, commit**

```bash
git add psa/cli.py psa/lifecycle.py
git commit -m "feat: add --memory-scorer to psa train and lifecycle"
```

---

### Task 10: End-to-End Benchmark

- [ ] **Step 1: Rebuild atlas** (computes anchor features)
- [ ] **Step 2: Train selector + co-activation + memory scorer**
- [ ] **Step 3: Run benchmark with coactivation selector**
- [ ] **Step 4: Score and compare against targets**
- [ ] **Step 5: Commit results**

---

### Task 11: Lint, Format, Full Test Suite

- [ ] **Step 1: ruff check + format**
- [ ] **Step 2: Full pytest**
- [ ] **Step 3: Commit**

---

## Self-Review

**Spec coverage:**
- Level 1 features: Tasks 2-4
- Level 2 MemoryScorer: Tasks 5-6
- Level 2 training (benchmark + LLM modes): Task 7
- Pipeline + packer integration: Task 8
- CLI + lifecycle: Task 9
- MPS fix: Task 1
- Benchmark: Task 10
- All covered.

**Placeholder scan:** No TBDs. All code steps have complete code blocks.

**Type consistency:**
- `ScoredMemory` defined Task 6, used Task 8 -- consistent
- `MemoryReRanker` defined Task 6, used Tasks 7, 8 -- consistent
- `anchor_feature_dim=8` in Task 3, 8-dim features in Task 2 -- consistent
- `pre_ranked` in Task 8 matches packer and pipeline -- consistent
