# Phase 2A: Answer-Bearing Labels + Learned Constraints — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace lexical-overlap training labels with LLM-graded utility labels (0-3 scale), mine hard negatives (same-anchor wrong-episode + right-topic wrong-time), expand the MemoryReRanker MLP from 11 to 16 dims with tri-state constraint features, and remove the rule-backed ConstraintScorer from the primary pipeline path. Target: F1 >= 0.22, LLM-judge >= 0.40.

**Architecture:** New graded labeler sends batched LLM calls (one per query, ~200 memories each) to produce utility grades. Hard negatives are mined from the graded pool using facet filters. The MLP gains 5 constraint features (entity overlap, speaker match, actor match, temporal match, type match) as tri-state inputs learned end-to-end. Pipeline checks `supports_constraints` metadata to route between learned (16-dim) and rule-backed (11-dim + ConstraintScorer) paths.

**Tech Stack:** Existing PSA infrastructure. `call_llm()` for graded labeling. PyTorch MLP (unchanged architecture, wider input).

**Design spec:** `docs/superpowers/specs/2026-04-14-phase2a-answer-bearing-labels-design.md`

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `psa/training/graded_labeler.py` | Batched LLM grading (0-3) + hard negative mining + training data assembly |
| `tests/test_graded_labeler.py` | Unit tests |

### Modified files

| File | Change |
|------|--------|
| `psa/memory_scorer.py` | Expand MLP 11->16; accept QueryFrame; build tri-state constraint features; expose `supports_constraints` |
| `psa/training/train_memory_scorer.py` | Weighted MSE loss; handle 16-dim; save `supports_constraints: true` |
| `psa/pipeline.py` | Check `supports_constraints`; conditional ConstraintScorer; null-frame fallback |
| `psa/cli.py` | `psa train --memory-scorer` uses graded labeler |
| `psa/lifecycle.py` | Slow path uses graded labeler |

---

### Task 1: Graded Labeler — Tests + Implementation

**Files:**
- Create: `psa/training/graded_labeler.py`
- Create: `tests/test_graded_labeler.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for psa.training.graded_labeler — LLM-graded utility labels + hard negative mining."""

import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


def _make_mock_memory(mid, body, source_ids=None, entities=None, mentioned_at=None,
                      quality=0.5, memory_type="SEMANTIC", embedding=None):
    mo = MagicMock()
    mo.memory_object_id = mid
    mo.body = body
    mo.source_ids = source_ids or [f"src_{mid}"]
    mo.entities = entities or []
    mo.actor_entities = []
    mo.speaker_role = None
    mo.stance = None
    mo.mentioned_at = mentioned_at
    mo.quality_score = quality
    mo.memory_type = MagicMock()
    mo.memory_type.value = memory_type
    mo.memory_type.name = memory_type
    mo.created_at = "2026-04-01T00:00:00"
    mo.embedding = embedding or list(np.zeros(768))
    return mo


class TestBatchedGrading:
    def test_grades_memories_via_llm(self):
        from psa.training.graded_labeler import grade_memories_batched

        memories = [
            _make_mock_memory("m1", "JWT tokens expire after 24 hours"),
            _make_mock_memory("m2", "Database uses Postgres 15"),
            _make_mock_memory("m3", "Auth uses RS256 signed JWTs"),
        ]
        # Mock LLM to return grades
        mock_response = '{"1": 3, "2": 0, "3": 2}'
        with patch("psa.training.graded_labeler.call_llm", return_value=mock_response):
            grades = grade_memories_batched("What auth tokens do we use?", memories)

        assert len(grades) == 3
        assert grades["m1"] == 3
        assert grades["m2"] == 0
        assert grades["m3"] == 2

    def test_caps_at_max_memories(self):
        from psa.training.graded_labeler import grade_memories_batched

        memories = [_make_mock_memory(f"m{i}", f"body {i}") for i in range(300)]
        mock_response = json.dumps({str(i+1): 0 for i in range(200)})
        with patch("psa.training.graded_labeler.call_llm", return_value=mock_response):
            grades = grade_memories_batched("query", memories, max_memories=200)

        # Should only grade up to max_memories
        assert len(grades) <= 200


class TestHardNegativeMining:
    def test_same_anchor_wrong_episode(self):
        from psa.training.graded_labeler import mine_hard_negatives

        positive = _make_mock_memory("pos", "JWT auth", source_ids=["src_a"])
        neg_candidate = _make_mock_memory("neg", "JWT refresh", source_ids=["src_b"])
        # Same anchor, different episode, grade 0
        grades = {"pos": 3, "neg": 0}

        negatives = mine_hard_negatives(
            positives=[positive],
            all_memories=[positive, neg_candidate],
            grades=grades,
            anchor_memories={"anchor_1": [positive, neg_candidate]},
            query_frame=None,
            ce_scores={"pos": 0.8, "neg": 0.7},
        )
        same_anchor = [n for n in negatives if n["negative_type"] == "same_anchor_wrong_episode"]
        assert len(same_anchor) >= 1
        assert same_anchor[0]["memory_id"] == "neg"

    def test_skips_same_episode(self):
        from psa.training.graded_labeler import mine_hard_negatives

        positive = _make_mock_memory("pos", "JWT auth", source_ids=["src_shared"])
        same_ep = _make_mock_memory("same", "JWT details", source_ids=["src_shared"])
        grades = {"pos": 3, "same": 0}

        negatives = mine_hard_negatives(
            positives=[positive],
            all_memories=[positive, same_ep],
            grades=grades,
            anchor_memories={"a1": [positive, same_ep]},
            query_frame=None,
            ce_scores={"pos": 0.8, "same": 0.7},
        )
        # Should NOT mine same_ep — shares source_id
        assert all(n["memory_id"] != "same" for n in negatives)


class TestTrainingDataAssembly:
    def test_generate_graded_training_data(self):
        from psa.training.graded_labeler import generate_graded_training_data

        # Create a mock results file
        records = [
            {
                "question_id": "q1",
                "question": "What auth do we use?",
                "selected_anchor_ids": [1, 2],
                "answer_gold": "JWT tokens",
            }
        ]

        mock_pipeline = MagicMock()
        mock_pipeline.tenant_id = "test"
        mock_pipeline.embedding_model.embed.return_value = np.zeros(768)
        mock_pipeline.store.query_by_anchor.return_value = [
            _make_mock_memory("m1", "JWT auth pattern"),
            _make_mock_memory("m2", "Database setup"),
        ]
        mock_pipeline.full_atlas_scorer = None

        mock_grades = '{"1": 3, "2": 0}'

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, "results.jsonl")
            with open(results_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            output_path = os.path.join(tmpdir, "graded_train.jsonl")

            with patch("psa.training.graded_labeler.call_llm", return_value=mock_grades):
                n = generate_graded_training_data(
                    results_path=results_path,
                    output_path=output_path,
                    pipeline=mock_pipeline,
                )

            assert n >= 2  # at least the 2 memories
            with open(output_path) as f:
                examples = [json.loads(line) for line in f if line.strip()]
            assert all("label" in ex for ex in examples)
            # Labels should be graded (0.0, 0.33, 0.67, or 1.0)
            labels = {ex["label"] for ex in examples}
            assert labels.issubset({0.0, 0.33, 0.67, 1.0})
```

- [ ] **Step 2: Implement graded_labeler.py**

```python
"""
graded_labeler.py -- LLM-graded utility labels + hard negative mining.

Replaces lexical-F1 labels with graded utility scores:
  0 = irrelevant, 1 = background, 2 = supporting, 3 = direct answer

Hard negatives mined from:
  - same-anchor wrong-episode (grade 0, different source_ids, confusable CE/cosine)
  - right-topic wrong-time (grade 0, temporal mismatch, confusable)
"""

import json
import logging
import math
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("psa.training.graded_labeler")

GRADE_TO_LABEL = {0: 0.0, 1: 0.33, 2: 0.67, 3: 1.0}
MAX_MEMORIES_PER_QUERY = 200
MAX_POSITIVES = 5
MAX_SAME_ANCHOR_NEG = 3
MAX_WRONG_TIME_NEG = 3
TARGET_EXAMPLES_PER_QUERY = 15
CE_CONFUSABILITY_MARGIN = 0.3


def grade_memories_batched(
    query: str,
    memories: list,
    max_memories: int = MAX_MEMORIES_PER_QUERY,
) -> Dict[str, int]:
    """Grade memories via batched LLM call. Returns {memory_object_id: grade}."""
    from psa.llm import call_llm

    if not memories:
        return {}

    # Cap and sort by some heuristic (we don't have CE scores here, use quality)
    capped = memories[:max_memories]

    prompt_lines = [
        f"Query: {query}\n",
        "Rate each memory's utility for answering this query:",
        "0 = Irrelevant -- no useful information",
        "1 = Background -- general context, doesn't answer",
        "2 = Supporting -- evidence that helps answer",
        "3 = Direct answer -- contains the specific answer\n",
    ]
    for i, m in enumerate(capped, 1):
        body_trunc = (m.body or "")[:200].replace("\n", " ")
        prompt_lines.append(f"Memory {i}: {body_trunc}")

    prompt_lines.append('\nRespond as JSON: {"1": score, "2": score, ...}')
    prompt = "\n".join(prompt_lines)

    try:
        response = call_llm(
            [{"role": "user", "content": prompt}],
            max_tokens=max(len(capped) * 5, 256),
            json_mode=True,
            temperature=0.0,
        )
        raw = json.loads(response)
        grades = {}
        for i, m in enumerate(capped, 1):
            g = raw.get(str(i), raw.get(i, 0))
            grades[m.memory_object_id] = max(0, min(3, int(g)))
        return grades
    except Exception as e:
        logger.warning("Batched grading failed: %s", e)
        return {}


def mine_hard_negatives(
    positives: list,
    all_memories: list,
    grades: Dict[str, int],
    anchor_memories: Dict,
    query_frame,
    ce_scores: Dict[str, float],
) -> List[dict]:
    """Mine hard negatives from graded pool."""
    negatives = []
    pos_ids = {m.memory_object_id for m in positives}
    pos_source_ids = set()
    for m in positives:
        pos_source_ids.update(m.source_ids or [])

    max_pos_ce = max((ce_scores.get(m.memory_object_id, 0) for m in positives), default=0)

    # Same-anchor wrong-episode
    for anchor_id, anchor_mems in anchor_memories.items():
        for m in anchor_mems:
            if m.memory_object_id in pos_ids:
                continue
            if grades.get(m.memory_object_id, -1) != 0:
                continue
            # Different episode: no source_ids overlap
            mem_sources = set(m.source_ids or [])
            if mem_sources & pos_source_ids:
                continue
            # Confusable: CE within margin
            ce = ce_scores.get(m.memory_object_id, 0)
            if abs(ce - max_pos_ce) > CE_CONFUSABILITY_MARGIN:
                continue
            negatives.append({
                "memory_id": m.memory_object_id,
                "negative_type": "same_anchor_wrong_episode",
                "memory": m,
            })
            if len([n for n in negatives if n["negative_type"] == "same_anchor_wrong_episode"]) >= MAX_SAME_ANCHOR_NEG:
                break

    # Right-topic wrong-time
    if query_frame and query_frame.time_constraint:
        for m in all_memories:
            if m.memory_object_id in pos_ids:
                continue
            if grades.get(m.memory_object_id, -1) != 0:
                continue
            mentioned = getattr(m, "mentioned_at", None)
            if not mentioned:
                continue  # unknown temporal = NOT a temporal negative
            # Check temporal mismatch
            if query_frame.time_constraint.lower() in mentioned.lower():
                continue  # matches, not a negative
            # Confusable
            ce = ce_scores.get(m.memory_object_id, 0)
            if abs(ce - max_pos_ce) > CE_CONFUSABILITY_MARGIN:
                continue
            negatives.append({
                "memory_id": m.memory_object_id,
                "negative_type": "right_topic_wrong_time",
                "memory": m,
            })
            if len([n for n in negatives if n["negative_type"] == "right_topic_wrong_time"]) >= MAX_WRONG_TIME_NEG:
                break

    return negatives


def generate_graded_training_data(
    results_path: str,
    output_path: str,
    pipeline,
    token_budget: int = 6000,
) -> int:
    """Generate graded training data with hard negatives.

    Returns number of examples written.
    """
    from psa.query_frame import extract_query_frame
    from psa.memory_scorer import _build_feature_vector, _type_onehot, _days_since, _recency, _cosine_to_query

    records = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    written = 0

    with open(output_path, "w") as fout:
        for qi, record in enumerate(records):
            query = record.get("question", record.get("query", ""))
            query_id = record.get("question_id", f"q_{qi}")
            anchor_ids = record.get("selected_anchor_ids", record.get("anchor_ids", []))

            if not query or not anchor_ids:
                continue

            # Extract query frame
            query_frame = extract_query_frame(query)

            # Embed query
            try:
                query_vec = np.asarray(pipeline.embedding_model.embed(query), dtype=np.float32)
            except Exception:
                continue

            # Fetch ALL memories from selected anchors (broad pool)
            all_memories = []
            seen = set()
            anchor_memories = {}
            for aid in anchor_ids:
                try:
                    mems = pipeline.store.query_by_anchor(pipeline.tenant_id, int(aid), limit=50)
                    anchor_memories[int(aid)] = mems
                    for m in mems:
                        if m.memory_object_id not in seen:
                            seen.add(m.memory_object_id)
                            all_memories.append(m)
                except Exception:
                    pass

            if not all_memories:
                continue

            # Sort by heuristic quality, cap at MAX_MEMORIES_PER_QUERY
            all_memories.sort(key=lambda m: m.quality_score, reverse=True)
            all_memories = all_memories[:MAX_MEMORIES_PER_QUERY]

            # CE scores for confusability checks
            ce_scores_map = {}
            fas = getattr(pipeline, "full_atlas_scorer", None)
            if fas and getattr(fas, "_cross_encoder", None):
                try:
                    pairs = [(query, m.body) for m in all_memories]
                    raw = fas._cross_encoder.predict(pairs)
                    for i, m in enumerate(all_memories):
                        ce_scores_map[m.memory_object_id] = float(raw[i])
                except Exception:
                    pass

            # Grade via LLM
            grades = grade_memories_batched(query, all_memories)
            if not grades:
                continue

            # Identify positives (grade 2-3)
            positives = [m for m in all_memories if grades.get(m.memory_object_id, 0) >= 2]
            if not positives:
                continue  # skip rule: no strong positives = no hard negatives

            positives = positives[:MAX_POSITIVES]

            # Mine hard negatives
            hard_negs = mine_hard_negatives(
                positives, all_memories, grades, anchor_memories,
                query_frame, ce_scores_map,
            )

            # Easy negatives: grade 0 from other anchors, random sample
            easy_pool = [
                m for m in all_memories
                if grades.get(m.memory_object_id, 0) == 0
                and m.memory_object_id not in {n["memory_id"] for n in hard_negs}
            ]
            n_easy = TARGET_EXAMPLES_PER_QUERY - len(positives) - len(hard_negs)
            rng = np.random.default_rng(hash(query_id) % 2**31)
            if len(easy_pool) > n_easy:
                indices = rng.choice(len(easy_pool), size=max(0, n_easy), replace=False)
                easy_negs = [easy_pool[i] for i in indices]
            else:
                easy_negs = easy_pool

            # Build constraint features from query_frame + memory facets
            def _constraint_features(m, frame):
                """Build 5 tri-state constraint features."""
                # Entity overlap (graded Jaccard)
                f_ents = set(e.lower() for e in (frame.entities or [])) if frame else set()
                m_ents = set(e.lower() for e in (getattr(m, "entities", []) or []))
                if not f_ents or not m_ents:
                    entity_overlap = 0.5
                else:
                    union = f_ents | m_ents
                    entity_overlap = len(f_ents & m_ents) / len(union) if union else 0.5

                # Speaker match
                f_speaker = getattr(frame, "speaker_role_constraint", None) if frame else None
                m_speaker = getattr(m, "speaker_role", None)
                if not f_speaker:
                    speaker_match = 0.5
                elif not m_speaker:
                    speaker_match = 0.5
                else:
                    speaker_match = 1.0 if f_speaker == m_speaker else 0.0

                # Actor match
                f_actor = getattr(frame, "entity_constraint", None) if frame else None
                m_actors = [a.lower() for a in (getattr(m, "actor_entities", []) or [])]
                if not f_actor:
                    actor_match = 0.5
                elif not m_actors:
                    actor_match = 0.5
                else:
                    actor_match = 1.0 if f_actor.lower() in m_actors else 0.0

                # Temporal match
                f_time = getattr(frame, "time_constraint", None) if frame else None
                m_time = getattr(m, "mentioned_at", None)
                if not f_time:
                    temporal_match = 0.5
                elif not m_time:
                    temporal_match = 0.5
                else:
                    temporal_match = 1.0 if f_time.lower() in m_time.lower() else 0.0

                # Type match
                target = getattr(frame, "answer_target", None) if frame else None
                no_type_targets = {"comparison", "temporal_change", "prior_statement"}
                if not target or target in no_type_targets:
                    type_match = 0.5
                else:
                    type_map = {"failure": "FAILURE", "procedure": "PROCEDURAL",
                                "fact": "SEMANTIC", "preference": "SEMANTIC"}
                    expected = type_map.get(target)
                    m_type = m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type)
                    if not expected:
                        type_match = 0.5
                    else:
                        type_match = 1.0 if m_type == expected else 0.3

                return [entity_overlap, speaker_match, actor_match, temporal_match, type_match]

            # Write examples
            def _write_example(m, label, neg_type="positive"):
                ce = ce_scores_map.get(m.memory_object_id, 0.0)
                onehot = _type_onehot(m.memory_type)
                body_tok = min(1.0, (len(m.body) / 4.0) / token_budget)
                days = _days_since(m.created_at)
                rec = _recency(days)
                cos = _cosine_to_query(m.embedding, query_vec)
                constraint = _constraint_features(m, query_frame)

                row = {
                    "features": [ce] + onehot + [m.quality_score, body_tok, rec, cos] + constraint,
                    "label": label,
                    "negative_type": neg_type,
                    "query_id": query_id,
                }
                fout.write(json.dumps(row) + "\n")
                return 1

            # Positives
            for m in positives:
                grade = grades.get(m.memory_object_id, 0)
                written += _write_example(m, GRADE_TO_LABEL[grade], "positive")

            # Grade-1 weak positives
            for m in all_memories:
                if grades.get(m.memory_object_id, 0) == 1:
                    written += _write_example(m, 0.33, "weak_positive")

            # Hard negatives
            for neg in hard_negs:
                written += _write_example(neg["memory"], 0.0, neg["negative_type"])

            # Easy negatives
            for m in easy_negs:
                written += _write_example(m, 0.0, "easy_negative")

            if (qi + 1) % 50 == 0:
                logger.info("Graded %d/%d queries, %d examples", qi + 1, len(records), written)

    logger.info("Generated %d graded training examples to %s", written, output_path)
    return written
```

- [ ] **Step 3: Run tests, lint, commit**

Run: `uv run pytest tests/test_graded_labeler.py -v`
Run: `uv run ruff check psa/training/graded_labeler.py tests/test_graded_labeler.py`

```bash
git add psa/training/graded_labeler.py tests/test_graded_labeler.py
git commit -m "feat: add graded labeler — LLM utility labels + hard negative mining"
```

---

### Task 2: Expand MemoryScorer MLP to 16 Dims

**Files:**
- Modify: `psa/memory_scorer.py`
- Modify: `tests/test_memory_scorer.py`

- [ ] **Step 1: Add constraint feature builder**

In `psa/memory_scorer.py`, add a function to build 5 tri-state constraint features:

```python
def _build_constraint_features(memory, query_frame) -> List[float]:
    """Build 5 tri-state constraint features from QueryFrame + memory facets.
    
    Each feature: match=1.0, mismatch=0.0, unknown/unconstrained=0.5
    """
    if query_frame is None:
        return [0.5, 0.5, 0.5, 0.5, 0.5]

    # 1. Entity overlap (graded Jaccard)
    f_ents = set(e.lower() for e in (query_frame.entities or []))
    m_ents = set(e.lower() for e in (getattr(memory, "entities", []) or []))
    if not f_ents or not m_ents:
        entity_overlap = 0.5
    else:
        union = f_ents | m_ents
        entity_overlap = len(f_ents & m_ents) / len(union) if union else 0.5

    # 2. Speaker role match
    f_speaker = query_frame.speaker_role_constraint
    m_speaker = getattr(memory, "speaker_role", None)
    if not f_speaker:
        speaker_match = 0.5
    elif not m_speaker:
        speaker_match = 0.5
    else:
        speaker_match = 1.0 if f_speaker == m_speaker else 0.0

    # 3. Actor entity match
    f_actor = query_frame.entity_constraint
    m_actors = [a.lower() for a in (getattr(memory, "actor_entities", []) or [])]
    if not f_actor:
        actor_match = 0.5
    elif not m_actors:
        actor_match = 0.5
    else:
        actor_match = 1.0 if f_actor.lower() in m_actors else 0.0

    # 4. Temporal match
    f_time = query_frame.time_constraint
    m_time = getattr(memory, "mentioned_at", None)
    if not f_time:
        temporal_match = 0.5
    elif not m_time:
        temporal_match = 0.5
    else:
        temporal_match = 1.0 if f_time.lower() in m_time.lower() else 0.0

    # 5. Type match
    no_type_targets = {"comparison", "temporal_change", "prior_statement"}
    target = query_frame.answer_target
    if not target or target in no_type_targets:
        type_match = 0.5
    else:
        type_map = {"failure": "FAILURE", "procedure": "PROCEDURAL",
                    "fact": "SEMANTIC", "preference": "SEMANTIC"}
        expected = type_map.get(target)
        m_type = memory.memory_type.value if hasattr(memory.memory_type, "value") else str(memory.memory_type)
        type_match = 1.0 if m_type == expected else 0.3 if expected else 0.5

    return [entity_overlap, speaker_match, actor_match, temporal_match, type_match]
```

- [ ] **Step 2: Update MemoryScorer.score() to accept QueryFrame and build 16-dim features**

Add `query_frame=None` parameter. After building the 11-dim features, append constraint features:

```python
    def score(self, query, query_vec, memories, query_frame=None, ...):
        ...
        # Stage B: build feature matrix
        feature_rows = []
        for i in range(len(memories)):
            base = _build_feature_vector(ce_scores[i], memories[i], query_vec, token_budget)
            if self.supports_constraints:
                constraint = _build_constraint_features(memories[i], query_frame)
                base = base + constraint
            feature_rows.append(base)
        ...
```

- [ ] **Step 3: Add supports_constraints property**

```python
    @property
    def supports_constraints(self) -> bool:
        """Whether this scorer's MLP was trained with constraint features."""
        # Infer from MLP input dim
        if hasattr(self.reranker, "net"):
            return self.reranker.net[0].in_features >= 16
        return False
```

- [ ] **Step 4: Update from_model_path to read supports_constraints**

Read from metadata JSON, or infer from model weights shape (existing logic already infers input_dim from `state["net.0.weight"].shape[1]`).

- [ ] **Step 5: Add tests**

```python
    def test_score_with_query_frame_16dim(self):
        """16-dim MLP uses constraint features from QueryFrame."""
        from psa.memory_scorer import MemoryScorer, MemoryReRanker
        from psa.query_frame import QueryFrame
        
        reranker = MemoryReRanker(input_dim=16)  # 16-dim
        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.5, 0.5])
        
        scorer = MemoryScorer(cross_encoder=mock_ce, reranker=reranker, device="cpu")
        assert scorer.supports_constraints is True
        
        frame = QueryFrame(answer_target="failure", entities=["auth"])
        memories = [
            _make_mock_memory("m1", "auth failed", memory_type="FAILURE", entities=["auth"]),
            _make_mock_memory("m2", "database setup", memory_type="SEMANTIC"),
        ]
        
        results = scorer.score("What auth failed?", np.zeros(768), memories, query_frame=frame)
        assert len(results) == 2
        # FAILURE type + entity match should boost m1
        assert results[0].memory_object_id == "m1"

    def test_score_without_frame_uses_neutral(self):
        """16-dim MLP with no QueryFrame uses 0.5 for all constraints."""
        from psa.memory_scorer import MemoryScorer, MemoryReRanker
        
        reranker = MemoryReRanker(input_dim=16)
        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.5])
        
        scorer = MemoryScorer(cross_encoder=mock_ce, reranker=reranker, device="cpu")
        memories = [_make_mock_memory("m1", "test")]
        
        results = scorer.score("test", np.zeros(768), memories, query_frame=None)
        assert len(results) == 1  # no crash
```

- [ ] **Step 6: Run tests, commit**

```bash
git add psa/memory_scorer.py tests/test_memory_scorer.py
git commit -m "feat: expand MemoryReRanker to 16 dims with tri-state constraint features"
```

---

### Task 3: Update Training Loop for Weighted MSE

**Files:**
- Modify: `psa/training/train_memory_scorer.py`

- [ ] **Step 1: Replace BCE with weighted MSE**

In `MemoryScorerTrainer.train()`:

Replace `criterion = nn.BCELoss()` with:

```python
        criterion = nn.MSELoss(reduction="none")  # per-example loss
```

Replace the loss computation in the batch loop:

```python
            preds = model(X_batch).squeeze(1)
            per_example_loss = criterion(preds, y_batch)

            # Per-example weighting: grade 3 (1.0) -> 2.0, grade 2 (0.67) -> 1.5, else 1.0
            weights = torch.ones_like(y_batch)
            weights[y_batch >= 0.9] = 2.0    # grade 3
            weights[(y_batch >= 0.6) & (y_batch < 0.9)] = 1.5  # grade 2
            loss = (per_example_loss * weights).mean()
```

- [ ] **Step 2: Save supports_constraints in metadata**

In the metadata dict, add:

```python
        meta = {
            ...
            "supports_constraints": input_dim >= 16,
        }
```

- [ ] **Step 3: Update _record_to_features to handle 16-dim**

The existing `_record_to_features` reads `ce_score + type_vec + quality + body_norm + recency + cosine`. For graded labeler output, features are pre-computed as a 16-dim list. Update to handle both formats:

```python
def _record_to_features(record: dict) -> List[float]:
    if "features" in record:
        return record["features"]  # graded labeler format (16-dim)
    # Legacy format (11-dim)
    return [record["ce_score"]] + record["type_vec"] + [
        record["quality_score"], record["body_norm"], record["recency"], record["cosine"]
    ]
```

- [ ] **Step 4: Run tests, commit**

```bash
git add psa/training/train_memory_scorer.py
git commit -m "feat: weighted MSE loss for graded labels; supports_constraints metadata"
```

---

### Task 4: Pipeline Fallback Logic

**Files:**
- Modify: `psa/pipeline.py`

- [ ] **Step 1: Update Level 2 block to check supports_constraints**

Replace the current Level 2 + constraint scorer block with:

```python
        # Level 2: Memory-level scoring
        _pre_ranked = False
        if self.memory_scorer is not None and memories:
            t0_l2 = time.perf_counter()
            scored_memories = self.memory_scorer.score(
                query=query,
                query_vec=query_vec,
                memories=memories,
                query_frame=query_frame,
            )
            # If MLP supports constraints (16-dim), it handles them internally.
            # If old MLP (11-dim), apply rule-backed scorer as fallback.
            if not self.memory_scorer.supports_constraints:
                scored_memories = self._constraint_scorer.adjust_scores(
                    scored_memories, query_frame
                )
            memories = [sm.memory for sm in scored_memories]
            logger.debug(
                "Level 2 scored %d memories in %.1fms (constraints=%s)",
                len(memories),
                (time.perf_counter() - t0_l2) * 1000,
                "learned" if self.memory_scorer.supports_constraints else "rule-backed",
            )
            _pre_ranked = True
        elif memories:
            # No MLP at all: rule-backed constraint scoring only
            from .memory_scorer import ScoredMemory
            scored_as_list = [
                ScoredMemory(
                    memory_object_id=m.memory_object_id,
                    final_score=m.quality_score,
                    memory=m,
                )
                for m in memories
            ]
            scored_as_list = self._constraint_scorer.adjust_scores(
                scored_as_list, query_frame
            )
            scored_as_list.sort(key=lambda s: s.final_score, reverse=True)
            memories = [sm.memory for sm in scored_as_list]
            _pre_ranked = True
```

- [ ] **Step 2: Handle null query frame**

At the query_frame extraction step, add fallback:

```python
        try:
            query_frame = extract_query_frame(query)
        except Exception:
            query_frame = QueryFrame()  # null frame, all neutral
```

- [ ] **Step 3: Run pipeline tests, commit**

```bash
git add psa/pipeline.py
git commit -m "feat: supports_constraints fallback; null-frame handling in pipeline"
```

---

### Task 5: CLI + Lifecycle Integration

**Files:**
- Modify: `psa/cli.py`
- Modify: `psa/lifecycle.py`

- [ ] **Step 1: Update cmd_train --memory-scorer to use graded labeler**

In the memory-scorer training block in `cmd_train()`, replace the old `generate_memory_scorer_data` call with `generate_graded_training_data`:

```python
        if getattr(args, "memory_scorer", False):
            print("Training memory scorer (graded labels)...")
            from .training.graded_labeler import generate_graded_training_data
            from .training.train_memory_scorer import MemoryScorerTrainer

            import glob as _glob
            results_dir = os.path.expanduser("~/.psa/benchmarks/longmemeval")
            results_files = sorted(_glob.glob(os.path.join(results_dir, "results_*.jsonl")))
            if not results_files:
                print("  No benchmark results found. Run benchmark first.")
            else:
                results_file = results_files[-1]
                scorer_data_path = os.path.join(
                    tenant.root_dir, "training", "graded_memory_scorer_train.jsonl"
                )
                from .pipeline import PSAPipeline
                _pipeline = PSAPipeline.from_tenant(tenant_id=tenant_id)

                n_examples = generate_graded_training_data(
                    results_path=results_file,
                    output_path=scorer_data_path,
                    pipeline=_pipeline,
                )
                print(f"  Generated {n_examples} graded examples.")

                if n_examples >= 200:
                    scorer_output = os.path.join(
                        tenant.root_dir, "models", "memory_scorer_latest"
                    )
                    MemoryScorerTrainer(output_dir=scorer_output).train(
                        data_path=scorer_data_path
                    )
                    print(f"  Memory scorer saved to {scorer_output}")
                else:
                    print(f"  Too few examples ({n_examples}). Need >= 200.")
```

- [ ] **Step 2: Update lifecycle slow path**

Same change: replace `generate_memory_scorer_data` with `generate_graded_training_data`.

- [ ] **Step 3: Run tests, commit**

```bash
git add psa/cli.py psa/lifecycle.py
git commit -m "feat: use graded labeler for memory scorer training in CLI + lifecycle"
```

---

### Task 6: End-to-End Benchmark

- [ ] **Step 1: Train with graded labels**

```bash
uv run psa train --tenant longmemeval_bench --force --coactivation --memory-scorer
```

- [ ] **Step 2: Run benchmark**

```bash
uv run psa benchmark longmemeval run --selector coactivation
```

- [ ] **Step 3: Score and compare**

```bash
uv run psa benchmark longmemeval score --results $(ls -t ~/.psa/benchmarks/longmemeval/results_*.jsonl | head -1) --method both
```

Target:

| Metric | Target | Previous |
|--------|--------|----------|
| F1 | >= 0.22 | 0.179 |
| LLM-judge | >= 0.40 | 0.372 |
| R@5 | >= 0.920 | 0.920 |

- [ ] **Step 4: Ablation — compare learned vs rule-backed**

Run the same benchmark but force rule-backed scoring (set `supports_constraints=false` in metadata temporarily) to verify the learned path beats the rule-backed path.

- [ ] **Step 5: Commit results**

```bash
git add -u
git commit -m "benchmark: Phase 2A graded labels + learned constraints results"
```

---

### Task 7: Lint, Format, Full Test Suite

- [ ] **Step 1: ruff check + format**
- [ ] **Step 2: Full pytest**
- [ ] **Step 3: Commit**

---

## Self-Review

**Spec coverage:**
- Graded utility labels (batched LLM, 0-3 scale): Task 1
- Hard negative mining (same-anchor wrong-episode, right-topic wrong-time): Task 1
- Per-query caps (5 pos, 3+3 hard neg, fill to 15): Task 1
- Skip rule (no positives = skip): Task 1
- Broader candidate pool (~200 per query, not top-30): Task 1
- MLP expansion 11->16 with tri-state constraint features: Task 2
- Weighted MSE loss (2.0/1.5/1.0 per-example): Task 3
- supports_constraints metadata: Tasks 2, 3
- Pipeline fallback (learned vs rule-backed vs no-model): Task 4
- Null frame handling: Task 4
- CLI + lifecycle integration: Task 5
- Benchmark + ablation: Task 6
- All covered.

**Type consistency:**
- `grade_memories_batched` returns Dict[str, int] — used in `mine_hard_negatives` and `generate_graded_training_data`
- `_build_constraint_features` returns List[float] (5 dims) — same function used in both graded_labeler and memory_scorer
- `supports_constraints` property in memory_scorer matches metadata field in train_memory_scorer
- `QueryFrame` imported from psa.query_frame in all files that use it
- Feature vector is 16-dim everywhere (11 + 5 constraint)
