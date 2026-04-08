"""
oracle_labeler.py — Two-stage oracle labeling for selector training.

For each training query:
  1. Run hybrid retrieval → top 24 anchor candidates
  2. Cheap stage (Qwen2.5-7B-Instruct): score ALL candidate sets via proxy metrics
     - SupportCoverage, ProceduralUtility, NoisePenalty, TokenCost
  3. Expensive stage (runtime model): TaskSuccess for top-3 surviving sets only
  4. Compute OracleScore and select oracle anchor set
  5. Persist labeled record

OracleScore = 0.45*SupportCoverage + 0.20*TaskSuccess
            + 0.15*ProceduralUtility - 0.10*NoisePenalty - 0.10*TokenCost

Labeling records are stored as JSONL at:
  ~/.psa/tenants/{tenant_id}/training/oracle_labels.jsonl
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("psa.training.oracle_labeler")

# ── Oracle score weights ───────────────────────────────────────────────────────

W_SUPPORT = 0.45
W_TASK_SUCCESS = 0.20
W_PROCEDURAL = 0.15
W_NOISE = -0.10
W_TOKEN_COST = -0.10

# ── Candidate set sizes to evaluate ───────────────────────────────────────────

# (max_size, max_combinations_to_evaluate)
CANDIDATE_SET_SPECS = [
    (1, 8),    # top-8 singles
    (2, 10),   # top-10 pairs
    (3, 5),    # top-5 triples
    (4, 2),    # top-2 quadruples (high-complexity queries only)
]

EXPENSIVE_TOP_N = 3  # number of sets to pass to the expensive TaskSuccess stage


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class CandidateSetScore:
    anchor_ids: List[int]
    support_coverage: float
    procedural_utility: float
    noise_penalty: float
    token_cost: float
    task_success: Optional[float]   # None until expensive stage
    oracle_score: float
    packed_tokens: int


@dataclass
class OracleLabel:
    """A complete oracle labeling record for one training query."""

    query_id: str
    query: str
    atlas_version: int
    runtime_model_id: str
    candidate_anchor_ids: List[int]
    all_sets: List[CandidateSetScore]
    winning_oracle_set: List[int]
    winning_oracle_score: float
    labeled_at: str
    is_high_complexity: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OracleLabel":
        sets = [CandidateSetScore(**s) for s in d.pop("all_sets", [])]
        return cls(all_sets=sets, **d)


# ── Proxy scoring (cheap stage) ────────────────────────────────────────────────


def score_support_coverage(
    candidate_anchor_ids: List[int],
    gold_anchor_ids: List[int],
    gold_chunk_ids: Optional[List[str]] = None,
) -> float:
    """
    Fraction of gold anchors covered by the candidate set.

    Primary: overlap with gold anchor IDs.
    Fallback (when gold anchors unknown): 0.0 — caller should provide gold evidence.
    """
    if not gold_anchor_ids:
        return 0.0
    covered = len(set(candidate_anchor_ids) & set(gold_anchor_ids))
    return covered / len(gold_anchor_ids)


def score_procedural_utility(
    candidate_memories_by_anchor: Dict[int, List[Any]],
) -> float:
    """
    Proxy: does the candidate set include procedural/failure/tool_use memories?

    Score = fraction of selected anchors with at least one
    procedural/failure/tool_use memory.
    """
    from psa.memory_object import MemoryType

    high_utility_types = {MemoryType.PROCEDURAL, MemoryType.FAILURE, MemoryType.TOOL_USE}
    if not candidate_memories_by_anchor:
        return 0.0

    useful_anchors = 0
    for memories in candidate_memories_by_anchor.values():
        if any(m.memory_type in high_utility_types for m in memories):
            useful_anchors += 1

    return useful_anchors / len(candidate_memories_by_anchor)


def score_noise_penalty(
    candidate_anchor_ids: List[int],
    oracle_anchor_ids: List[int],
) -> float:
    """
    Fraction of selected anchors that are NOT in the oracle set.

    High noise = many off-target anchors were selected.
    """
    if not candidate_anchor_ids:
        return 0.0
    noise = len(set(candidate_anchor_ids) - set(oracle_anchor_ids))
    return noise / len(candidate_anchor_ids)


def score_token_cost(packed_tokens: int, budget: int = 6000) -> float:
    """Normalized token cost: packed_tokens / budget."""
    return min(packed_tokens / max(budget, 1), 1.0)


def oracle_score(
    support_coverage: float,
    task_success: float,
    procedural_utility: float,
    noise_penalty: float,
    token_cost: float,
) -> float:
    """
    Compute the composite OracleScore.

    OracleScore = 0.45*SupportCoverage + 0.20*TaskSuccess
                + 0.15*ProceduralUtility - 0.10*NoisePenalty - 0.10*TokenCost
    """
    return (
        W_SUPPORT * support_coverage
        + W_TASK_SUCCESS * task_success
        + W_PROCEDURAL * procedural_utility
        + W_NOISE * noise_penalty
        + W_TOKEN_COST * token_cost
    )


# ── Qwen cheap-stage caller ────────────────────────────────────────────────────


def _call_qwen_proxy(
    query: str,
    candidate_set: List[int],
    anchor_cards_text: Dict[int, str],
    endpoint: str,
    model: str,
    timeout: int = 30,
) -> Dict[str, float]:
    """
    Call Qwen2.5-7B-Instruct to score a candidate anchor set with proxy metrics.

    Returns dict with keys: support_coverage, procedural_utility, noise_penalty, token_cost
    """
    import urllib.request

    cards_text = "\n\n".join(
        f"[anchor_{aid}]\n{anchor_cards_text.get(aid, '(no card)')}"
        for aid in candidate_set
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Candidate anchor set: {candidate_set}\n\n"
        f"Anchor cards:\n{cards_text}\n\n"
        "Score this anchor set (0.0 to 1.0 each):\n"
        '{"support_coverage": ..., "procedural_utility": ..., '
        '"noise_penalty": ..., "token_cost": ...}'
    )

    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 128,
            "response_format": {"type": "json_object"},
        }
    ).encode()

    try:
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        scores = json.loads(content)
        return {
            "support_coverage": float(scores.get("support_coverage", 0)),
            "procedural_utility": float(scores.get("procedural_utility", 0)),
            "noise_penalty": float(scores.get("noise_penalty", 0)),
            "token_cost": float(scores.get("token_cost", 0)),
        }
    except Exception as e:
        logger.warning("Qwen proxy scoring failed: %s", e)
        return {
            "support_coverage": 0.0,
            "procedural_utility": 0.0,
            "noise_penalty": 0.0,
            "token_cost": 0.0,
        }


# ── OracleLabeler ──────────────────────────────────────────────────────────────


class OracleLabeler:
    """
    Two-stage oracle labeler for selector training data.

    Requires:
    - A running PSA pipeline (retriever + atlas)
    - Access to gold anchor labels for training queries
    - A local Qwen endpoint (cheap stage)
    - An Anthropic API key (expensive stage, optional if task_success_fn provided)
    """

    def __init__(
        self,
        pipeline,                         # PSAPipeline instance
        output_path: str,                 # JSONL output file path
        qwen_endpoint: str = "http://localhost:11434/v1/chat/completions",
        qwen_model: str = "qwen2.5:7b",
        runtime_model_id: str = "claude-haiku-4-5-20251001",
        task_success_fn=None,             # callable(query, packed_context) → float
    ):
        self.pipeline = pipeline
        self.output_path = output_path
        self.qwen_endpoint = qwen_endpoint
        self.qwen_model = qwen_model
        self.runtime_model_id = runtime_model_id
        self.task_success_fn = task_success_fn
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def label(
        self,
        query_id: str,
        query: str,
        gold_anchor_ids: Optional[List[int]] = None,
        is_high_complexity: bool = False,
        top_k_candidates: int = 24,
    ) -> OracleLabel:
        """
        Run the two-stage oracle labeling for one query.

        Parameters
        ----------
        query_id: unique identifier for this query
        query: the query text
        gold_anchor_ids: ground-truth anchors (if available)
        is_high_complexity: if True, evaluate quadruple candidate sets too
        top_k_candidates: retriever shortlist size

        Returns
        -------
        OracleLabel with the winning oracle anchor set.
        """
        # Step 1: Retrieve candidates
        result = self.pipeline.query(query, top_k_candidates=top_k_candidates)
        candidates = result.candidates
        candidate_ids = [c.anchor_id for c in candidates]

        # Step 2: Generate candidate sets to evaluate
        specs = CANDIDATE_SET_SPECS.copy()
        if not is_high_complexity:
            specs = [s for s in specs if s[0] < 4]

        anchor_cards_text = {
            c.anchor_id: c.card.to_card_text()
            for c in candidates
        }

        # Step 3: Cheap stage — score all candidate sets
        all_sets: List[CandidateSetScore] = []

        for (max_size, max_combos) in specs:
            candidate_pool = candidate_ids[:max(8, max_size * 4)]
            combos = list(combinations(candidate_pool, max_size))[:max_combos]
            for combo in combos:
                combo_list = list(combo)
                # Cheap proxy scoring
                proxy = _call_qwen_proxy(
                    query=query,
                    candidate_set=combo_list,
                    anchor_cards_text=anchor_cards_text,
                    endpoint=self.qwen_endpoint,
                    model=self.qwen_model,
                )
                # Compute preliminary oracle score (TaskSuccess=0 until expensive stage)
                score = oracle_score(
                    support_coverage=proxy["support_coverage"],
                    task_success=0.0,
                    procedural_utility=proxy["procedural_utility"],
                    noise_penalty=proxy["noise_penalty"],
                    token_cost=proxy["token_cost"],
                )
                all_sets.append(
                    CandidateSetScore(
                        anchor_ids=combo_list,
                        support_coverage=proxy["support_coverage"],
                        procedural_utility=proxy["procedural_utility"],
                        noise_penalty=proxy["noise_penalty"],
                        token_cost=proxy["token_cost"],
                        task_success=None,
                        oracle_score=score,
                        packed_tokens=0,
                    )
                )

        # Step 4: Expensive stage — TaskSuccess for top-N surviving sets
        all_sets.sort(key=lambda s: s.oracle_score, reverse=True)
        top_sets = all_sets[:EXPENSIVE_TOP_N]

        if self.task_success_fn is not None:
            for cs in top_sets:
                # Pack context for this anchor set and measure task success
                try:
                    packed = self.pipeline.packed_context_for_anchors(
                        query=query, anchor_ids=cs.anchor_ids
                    )
                    cs.task_success = self.task_success_fn(query, packed.text)
                    cs.packed_tokens = packed.token_count
                except Exception as e:
                    logger.warning("TaskSuccess scoring failed: %s", e)
                    cs.task_success = 0.0

                # Recompute oracle score with task_success
                cs.oracle_score = oracle_score(
                    support_coverage=cs.support_coverage,
                    task_success=cs.task_success or 0.0,
                    procedural_utility=cs.procedural_utility,
                    noise_penalty=cs.noise_penalty,
                    token_cost=cs.token_cost,
                )

        # Step 5: Select winner
        best = max(all_sets, key=lambda s: s.oracle_score)

        label = OracleLabel(
            query_id=query_id,
            query=query,
            atlas_version=self.pipeline.atlas.version,
            runtime_model_id=self.runtime_model_id,
            candidate_anchor_ids=candidate_ids,
            all_sets=all_sets,
            winning_oracle_set=best.anchor_ids,
            winning_oracle_score=best.oracle_score,
            labeled_at=datetime.now(timezone.utc).isoformat(),
            is_high_complexity=is_high_complexity,
        )

        # Step 6: Persist
        self._append_label(label)
        return label

    def _append_label(self, label: OracleLabel):
        with open(self.output_path, "a") as f:
            f.write(json.dumps(label.to_dict()) + "\n")

    def load_labels(self) -> List[OracleLabel]:
        """Load all persisted oracle labels from the JSONL file."""
        labels = []
        if not os.path.exists(self.output_path):
            return labels
        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        labels.append(OracleLabel.from_dict(json.loads(line)))
                    except Exception as e:
                        logger.warning("Failed to parse oracle label: %s", e)
        return labels


if __name__ == "__main__":
    import argparse
    import os
    import random

    parser = argparse.ArgumentParser(description="PSA oracle labeler — generate training labels")
    parser.add_argument("--tenant", default="default", help="Tenant ID (default: default)")
    parser.add_argument("--n-queries", type=int, default=500, help="Number of queries to label (default: 500)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: ~/.psa/tenants/<tenant>/training/oracle_labels.jsonl)",
    )
    args = parser.parse_args()

    from psa.tenant import TenantManager
    from psa.pipeline import PSAPipeline

    tm = TenantManager()
    tenant = tm.get_or_create(args.tenant)
    output = args.output or os.path.join(tenant.root_dir, "training", "oracle_labels.jsonl")

    try:
        pipeline = PSAPipeline.from_tenant(tenant_id=args.tenant, psa_mode="primary")
    except FileNotFoundError:
        print(f"No atlas for tenant '{args.tenant}'. Run 'psa atlas build' first.")
        raise SystemExit(1)

    # Use memory object titles + summaries as self-supervised queries
    from psa.memory_object import MemoryStore, MemoryType
    store = MemoryStore(db_path=tenant.memory_db_path)
    all_memories = []
    for mtype in MemoryType:
        all_memories.extend(
            store.query_by_type(tenant_id=args.tenant, memory_type=mtype, limit=1000)
        )

    if not all_memories:
        # Fall back to anchor card texts as query source
        queries = [
            (f"anchor_{c.anchor_id}", c.name + ". " + c.meaning)
            for c in pipeline.atlas.cards
        ]
    else:
        queries = [(m.memory_object_id, m.title + ". " + m.summary) for m in all_memories]

    random.shuffle(queries)
    queries = queries[:args.n_queries]

    labeler = OracleLabeler(
        pipeline=pipeline,
        output_path=output,
    )

    count = 0
    for query_id, query_text in queries:
        try:
            labeler.label(query_id=query_id, query=query_text)
            count += 1
            if count % 10 == 0:
                print(f"  Labeled {count}/{len(queries)}...", flush=True)
        except Exception as e:
            logger.warning("Failed to label query %s: %s", query_id, e)

    print(f"Labeled {count} queries → {output}")
