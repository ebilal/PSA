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
    (1, 8),  # top-8 singles
    (2, 10),  # top-10 pairs
    (3, 5),  # top-5 triples
    (4, 2),  # top-2 quadruples (high-complexity queries only)
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
    task_success: Optional[float]  # None until expensive stage
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
        d = dict(d)  # shallow copy to avoid mutating caller's dict
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


# ── LLM cheap-stage caller (batched) ─────────────────────────────────────────

_ZERO_SCORES: Dict[str, float] = {
    "support_coverage": 0.0,
    "procedural_utility": 0.0,
    "noise_penalty": 0.0,
    "token_cost": 0.0,
}


def _call_qwen_proxy_batch(
    query: str,
    candidate_sets: List[List[int]],
    anchor_cards_text: Dict[int, str],
    endpoint: str = "",
    model: str = "",
    timeout: int = 120,
) -> List[Dict[str, float]]:
    """
    Score ALL candidate sets for one query in a single LLM call.

    Uses the unified LLM caller (cloud first, local fallback).

    Returns a list of score dicts (same order as candidate_sets).
    Falls back to zero scores for any set that can't be parsed.
    """
    from psa.llm import call_llm

    n = len(candidate_sets)

    def _post(messages: List[dict]) -> str:
        return call_llm(
            messages=messages, temperature=0.0, max_tokens=128 * n + 128, timeout=timeout
        )

    def _parse_scores(raw_sets: list, indices: List[int]) -> Dict[int, Dict[str, float]]:
        """Return {original_index: scores} for items that parse successfully."""
        parsed = {}
        for local_i, orig_i in enumerate(indices):
            if local_i >= len(raw_sets):
                break
            s = raw_sets[local_i]
            try:
                parsed[orig_i] = {
                    "support_coverage": float(s["support_coverage"]),
                    "procedural_utility": float(s["procedural_utility"]),
                    "noise_penalty": float(s["noise_penalty"]),
                    "token_cost": float(s["token_cost"]),
                }
            except (KeyError, TypeError, ValueError):
                pass  # will be retried
        return parsed

    def _make_payload(sets_subset: List[List[int]]) -> bytes:
        k = len(sets_subset)
        sets_lines = "\n".join(f"  set_{i}: {cs}" for i, cs in enumerate(sets_subset))
        needed: set = set()
        for cs in sets_subset:
            needed.update(cs)
        cards = "\n\n".join(
            f"[anchor_{aid}]\n{anchor_cards_text.get(aid, '(no card)')}" for aid in sorted(needed)
        )
        p = (
            f"Query: {query}\n\n"
            f"Anchor cards available:\n{cards}\n\n"
            f"Score each of the {k} candidate anchor sets below.\n"
            f"For each set return four float scores in [0.0, 1.0]:\n"
            f"  support_coverage  — fraction of query evidence covered\n"
            f"  procedural_utility — presence of procedural/failure/tool memories\n"
            f"  noise_penalty     — fraction of anchors that are off-target\n"
            f"  token_cost        — normalized packed-token cost\n\n"
            f"Candidate sets:\n{sets_lines}\n\n"
            f'Return JSON: {{"sets": [{{"support_coverage": 0.0, "procedural_utility": 0.0, '
            f'"noise_penalty": 0.0, "token_cost": 0.0}}, ...]}}\n'
            f'Return exactly {k} objects in the "sets" array, one per set, in order.'
        )
        return [{"role": "user", "content": p}]

    all_indices = list(range(n))
    results: Dict[int, Dict[str, float]] = {}

    # First attempt: all sets in one call
    try:
        content = _post(_make_payload(candidate_sets))
        raw_sets = json.loads(content).get("sets", [])
        results.update(_parse_scores(raw_sets, all_indices))
    except Exception as e:
        logger.warning("LLM batch proxy scoring failed on first attempt: %s", e)

    # Retry any indices that are still missing or malformed
    missing = [i for i in all_indices if i not in results]
    if missing:
        logger.info("Retrying %d missing/malformed sets", len(missing))
        retry_sets = [candidate_sets[i] for i in missing]
        try:
            content = _post(_make_payload(retry_sets))
            raw_sets = json.loads(content).get("sets", [])
            results.update(_parse_scores(raw_sets, missing))
        except Exception as e:
            logger.warning("LLM retry scoring failed: %s", e)

    # Fill any still-missing with zeros (give up after one retry)
    still_missing = [i for i in all_indices if i not in results]
    if still_missing:
        logger.warning(
            "%d sets could not be scored after retry; using zero scores", len(still_missing)
        )
        for i in still_missing:
            results[i] = dict(_ZERO_SCORES)

    return [results[i] for i in all_indices]


# ── LLM task-success judge ───────────────────────────────────────────────────


def _qwen_task_success(
    query: str,
    packed_context: str,
    endpoint: str = "http://localhost:11434/v1/chat/completions",
    model: str = "qwen2.5:7b",
    timeout: int = 120,
) -> float:
    """
    Use LLM as a judge: given a query and packed context, how well does
    the context support answering the query?

    Returns a score in [0.0, 1.0].
    """
    from psa.llm import call_llm

    prompt = (
        "You are evaluating whether a retrieved context is useful for answering a query.\n\n"
        f"QUERY: {query}\n\n"
        f"RETRIEVED CONTEXT:\n{packed_context[:3000]}\n\n"
        "Score how well this context helps answer the query.\n"
        "Consider:\n"
        "- Does the context contain information directly relevant to the query?\n"
        "- Could someone answer the query using ONLY this context?\n"
        "- Is there noise (irrelevant information) that would confuse the answer?\n\n"
        'Return JSON: {"score": 0.0 to 1.0, "reason": "one sentence"}\n'
        "0.0 = context is completely irrelevant\n"
        "0.5 = context is partially relevant but incomplete\n"
        "1.0 = context fully answers the query"
    )

    try:
        content = call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
            timeout=timeout,
        )
        result = json.loads(content)
        return max(0.0, min(1.0, float(result.get("score", 0.0))))
    except Exception as e:
        logger.warning("Task-success judge failed: %s", e)
        return 0.0


def backtrack_gold_anchors(
    answer_session_ids: List[str],
    store: Any,
    atlas: Any,
    tenant_id: str = "default",
) -> List[int]:
    """
    Derive gold anchor IDs from ground-truth session references.

    For each answer session, find memory objects whose source records have
    a source_path containing the session_id, then collect the anchor_ids
    those memory objects are assigned to.

    Deterministic — no LLM calls. Works for any dataset that provides
    ground-truth source references (e.g., LongMemEval answer_session_ids).

    Parameters
    ----------
    answer_session_ids:
        Session IDs known to contain the answer (e.g., from LongMemEval).
    store:
        MemoryStore instance to query.
    atlas:
        Atlas instance (reserved for future use — not currently needed).
    tenant_id:
        Tenant whose memories to search.

    Returns
    -------
    Deduplicated list of anchor IDs that contain memories from the answer sessions.
    """
    gold_anchor_ids: set = set()
    for session_id in answer_session_ids:
        memories = store.get_by_source_session(session_id, tenant_id=tenant_id)
        for m in memories:
            aid = getattr(m, "primary_anchor_id", None)
            if aid is not None and aid >= 0:
                gold_anchor_ids.add(aid)
    return list(gold_anchor_ids)


# ── OracleLabeler ──────────────────────────────────────────────────────────────


class OracleLabeler:
    """
    Two-stage oracle labeler for selector training data.

    Stage 1 (cheap): LLM scores ALL candidate sets with proxy metrics.
    Stage 2 (expensive): LLM judges task success for top-3 sets —
    does the packed context from this anchor set actually help answer the query?
    """

    def __init__(
        self,
        pipeline,  # PSAPipeline instance
        output_path: str,  # JSONL output file path
        qwen_endpoint: str = "http://localhost:11434/v1/chat/completions",
        qwen_model: str = "qwen2.5:7b",
        runtime_model_id: str = "qwen2.5:7b",
        task_success_fn=None,  # callable(query, packed_context) → float
        use_task_success: bool = True,  # enable expensive stage by default
        use_llm: bool = True,  # False → fast path (gold-anchor overlap only, no LLM)
    ):
        self.pipeline = pipeline
        self.output_path = output_path
        self.qwen_endpoint = qwen_endpoint
        self.qwen_model = qwen_model
        self.runtime_model_id = runtime_model_id
        self.use_llm = use_llm
        self.use_task_success = use_task_success and use_llm
        if task_success_fn is not None:
            self.task_success_fn = task_success_fn
        elif self.use_task_success:
            self.task_success_fn = lambda q, ctx: _qwen_task_success(
                q, ctx, endpoint=qwen_endpoint, model=qwen_model
            )
        else:
            self.task_success_fn = None
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

        # Step 3: Score candidate sets.
        # Fast path: when gold anchor IDs are known AND use_llm=False, skip all
        # LLM calls and rank purely by deterministic SupportCoverage (gold
        # overlap). This makes labeling instantaneous for benchmarks that ship
        # ground-truth anchors (e.g. LongMemEval). Set use_llm=True (modes
        # "local" or "api") to run the full two-stage pipeline instead.
        all_combos: List[List[int]] = []
        for max_size, max_combos in specs:
            candidate_pool = candidate_ids[: max(8, max_size * 4)]
            combos = list(combinations(candidate_pool, max_size))[:max_combos]
            all_combos.extend(list(c) for c in combos)

        all_sets: List[CandidateSetScore] = []

        if gold_anchor_ids and not self.use_llm:
            # Gold-anchor fast path: no LLM calls needed.
            for combo_list in all_combos:
                support_cov = score_support_coverage(combo_list, gold_anchor_ids)
                score = oracle_score(
                    support_coverage=support_cov,
                    task_success=0.0,
                    procedural_utility=0.5,
                    noise_penalty=0.0,
                    token_cost=0.0,
                )
                all_sets.append(
                    CandidateSetScore(
                        anchor_ids=combo_list,
                        support_coverage=support_cov,
                        procedural_utility=0.5,
                        noise_penalty=0.0,
                        token_cost=0.0,
                        task_success=None,
                        oracle_score=score,
                        packed_tokens=0,
                    )
                )
        else:
            # Full LLM path: use proxy scoring for all sets.
            # Triggered when use_llm=True (modes "local" / "api") or when no
            # gold anchors are available.
            # Batching reduces ~23 HTTP round-trips/query to 1, cutting labeling
            # time from ~16h to ~1h for 500 queries on an M4 Mac.
            anchor_cards_text = {c.anchor_id: c.card.to_card_text() for c in candidates}
            proxy_scores = _call_qwen_proxy_batch(
                query=query,
                candidate_sets=all_combos,
                anchor_cards_text=anchor_cards_text,
                endpoint=self.qwen_endpoint,
                model=self.qwen_model,
            )
            for combo_list, proxy in zip(all_combos, proxy_scores):
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

        # Step 4: Expensive stage — TaskSuccess for top-N surviving sets.
        # Skipped on the gold-anchor fast path (SupportCoverage already optimal).
        all_sets.sort(key=lambda s: s.oracle_score, reverse=True)
        top_sets = all_sets[:EXPENSIVE_TOP_N]

        if self.task_success_fn is not None and not gold_anchor_ids:
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


def _load_queries_from_sessions(
    sessions_dir: str,
    max_queries: int = 5000,
) -> List[Tuple[str, str]]:
    """
    Extract real user messages from Claude Code session JSONL files.

    These are the actual questions you typed during sessions — much better
    training signal than memory titles because they match the phrasing you'll
    use at retrieval time.

    Filters out: slash commands, tool caveats, XML-tagged system messages,
    very short messages (< 30 chars), and pure file-path messages.

    max_queries: stop scanning once this many unique queries are collected.
    """
    import glob as _glob
    import re as _re

    _skip_patterns = _re.compile(
        r"^(/|<|https?://|```|\.|\.\.)"  # slash commands, XML, URLs, code, paths
    )
    queries: List[Tuple[str, str]] = []
    seen: set = set()

    for fpath in _glob.glob(os.path.join(sessions_dir, "**", "*.jsonl"), recursive=True):
        if len(queries) >= max_queries:
            break
        try:
            with open(fpath) as f:
                for line in f:
                    if len(queries) >= max_queries:
                        break
                    d = json.loads(line)
                    if d.get("type") != "user":
                        continue
                    content = d["message"].get("content", "")
                    if not isinstance(content, str):
                        continue
                    content = content.strip()
                    # Must be a plain text message of reasonable length
                    if len(content) < 30 or len(content) > 800:
                        continue
                    if _skip_patterns.match(content):
                        continue
                    # Deduplicate on full content to avoid false collisions
                    if content in seen:
                        continue
                    seen.add(content)
                    msg_id = d.get("uuid", f"{os.path.basename(fpath)}_{len(queries)}")
                    queries.append((msg_id, content))
        except Exception:
            pass

    logger.info("Loaded %d real user queries from %s", len(queries), sessions_dir)
    return queries


if __name__ == "__main__":
    import argparse
    import os
    import random

    parser = argparse.ArgumentParser(description="PSA oracle labeler — generate training labels")
    parser.add_argument("--tenant", default="default", help="Tenant ID (default: default)")
    parser.add_argument(
        "--n-queries",
        type=int,
        default=300,
        help="Number of queries to label (default: 300, minimum gate)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: ~/.psa/tenants/<tenant>/training/oracle_labels.jsonl)",
    )
    parser.add_argument(
        "--sessions-dir",
        default=None,
        help="Path to Claude Code sessions dir (e.g. ~/.claude/projects). "
        "When provided, real user messages from session JSONL files are used "
        "as queries instead of memory titles. Better training signal.",
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

    if args.sessions_dir:
        queries = _load_queries_from_sessions(args.sessions_dir)
        if not queries:
            print(
                f"No usable user messages found in {args.sessions_dir}. "
                "Falling back to memory titles."
            )
            args.sessions_dir = None

    if not args.sessions_dir:
        # Self-supervised fallback: derive queries from memory titles + summaries
        from psa.memory_object import MemoryStore, MemoryType

        store = MemoryStore(db_path=tenant.memory_db_path)
        all_memories = []
        for mtype in MemoryType:
            all_memories.extend(
                store.query_by_type(tenant_id=args.tenant, memory_type=mtype, limit=1000)
            )

        if not all_memories:
            queries = [
                (f"anchor_{c.anchor_id}", c.name + ". " + c.meaning) for c in pipeline.atlas.cards
            ]
        else:
            queries = [(m.memory_object_id, m.title + ". " + m.summary) for m in all_memories]

    random.shuffle(queries)
    queries = queries[: args.n_queries]

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
