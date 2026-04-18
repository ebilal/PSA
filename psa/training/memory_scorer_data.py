"""
memory_scorer_data.py — Training data generator for Level 2 memory-level re-ranker MLP.

Two labeling modes:
  - benchmark: token-level F1 between memory.body and answer_gold (F1 > 0.3 → label 1)
  - llm: ask LLM "Is this memory relevant? Respond 1 or 0."

Output JSONL format (one line per query/memory pair):
  {"ce_score": 0.61, "type_vec": [0,0,1,0,0,0], "quality_score": 0.85,
   "body_norm": 0.2, "recency": 0.7, "cosine": 0.72, "label": 1, "query_id": "q_0"}
"""

import json
import logging

import numpy as np

from psa.memory_scorer import _type_onehot, _days_since, _recency, _cosine_to_query

logger = logging.getLogger("psa.training.memory_scorer_data")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _token_f1(text_a, text_b) -> float:
    """Compute token-level F1 between two texts (tokenize by whitespace split)."""
    tokens_a = set(str(text_a).lower().split())
    tokens_b = set(str(text_b).lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    precision = len(intersection) / len(tokens_a)
    recall = len(intersection) / len(tokens_b)
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return 2.0 * precision * recall / denom


def _llm_relevance_batch(query: str, memory_bodies: list[str]) -> list[int]:
    """Judge relevance for N memories in ONE LLM call.

    Same batching pattern as `_call_qwen_proxy_batch` in oracle_labeler:
    one structured prompt returning a JSON array of 0/1 labels mapped back
    by index. For a memory-scorer training run over hundreds of memories
    per query this converts O(N_memories) API calls into O(1) per query.

    Returns a list[int] of 0/1 per input memory. Failed parses → 0
    (conservative: treat as not-relevant rather than fabricate positives).
    """
    from psa.llm import call_llm

    n = len(memory_bodies)
    if n == 0:
        return []

    mem_block = "\n\n".join(
        f"=== MEMORY {i + 1} ===\n{body[:1500]}" for i, body in enumerate(memory_bodies)
    )
    prompt = (
        f"Query: {query}\n\n"
        f"{mem_block}\n\n"
        f"For each of the {n} memories above, decide whether it is relevant to the query.\n"
        f'Return JSON: {{"labels": [0_or_1, 0_or_1, ..., 0_or_1]}}\n'
        f"Return exactly {n} values in the labels array, in order."
    )
    try:
        content = call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16 * n + 64,
        ).strip()
        if content.startswith("```"):
            content = content.strip("`").split("\n", 1)[-1]
        result = json.loads(content)
        raw = result.get("labels", [])
        out: list[int] = []
        for i in range(n):
            try:
                v = int(raw[i])
                out.append(1 if v else 0)
            except (IndexError, TypeError, ValueError):
                out.append(0)
        return out
    except Exception as e:
        logger.warning("LLM relevance batch failed: %s; defaulting to 0s", e)
        return [0] * n


def _llm_relevance(query: str, memory_body: str) -> int:
    """Call LLM to determine if memory is relevant to the query. Returns 0 or 1.

    Single-memory wrapper retained for backward compatibility and tests.
    New loops should call `_llm_relevance_batch` to avoid O(N) round-trips.
    """
    from psa.llm import call_llm

    prompt = (
        f"Query: {query}\n\n"
        f"Memory: {memory_body}\n\n"
        "Is this memory relevant to the query? Respond with only 1 (yes) or 0 (no)."
    )
    try:
        response = call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8,
            json_mode=False,
        ).strip()
        return 1 if response.startswith("1") else 0
    except Exception:
        logger.warning("LLM relevance call failed; defaulting to 0")
        return 0


# ── Main generator ─────────────────────────────────────────────────────────────


def generate_memory_scorer_data(
    results_path: str,
    output_path: str,
    pipeline,
    mode: str = "benchmark",
    token_budget: int = 6000,
) -> int:
    """
    Generate training data for the MemoryReRanker MLP.

    Parameters
    ----------
    results_path:
        Path to a JSONL file where each record has:
          - "query": query string
          - "query_id": optional query identifier
          - "anchor_ids": list of selected anchor IDs (ints)
          - "answer_gold": reference answer (used in benchmark mode)
    output_path:
        Path to write the output JSONL training data.
    pipeline:
        A PSAPipeline (or compatible) instance. Must have:
          - pipeline.store: MemoryStore
          - pipeline.embedding_model: callable (query -> np.ndarray)
          - pipeline.full_atlas_scorer: FullAtlasScorer (optional, for ce_score)
    mode:
        "benchmark" (F1-based labels) or "llm" (LLM-based labels).
    token_budget:
        Token budget for body_norm feature normalization.

    Returns
    -------
    Number of (query, memory) pairs written.
    """
    if mode not in ("benchmark", "llm"):
        raise ValueError(f"Unknown mode: {mode!r}. Must be 'benchmark' or 'llm'.")

    store = pipeline.store
    embedder = pipeline.embedding_model
    cross_encoder = None
    fas = getattr(pipeline, "full_atlas_scorer", None)
    if fas is not None:
        cross_encoder = getattr(fas, "_cross_encoder", None)

    total_written = 0
    queries_processed = 0

    with open(results_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            query = record.get("question", record.get("query", ""))
            query_id = record.get("question_id", record.get("query_id", f"q_{queries_processed}"))
            anchor_ids = record.get("selected_anchor_ids", record.get("anchor_ids", []))
            answer_gold = record.get("answer_gold", "")

            if not query or not anchor_ids:
                queries_processed += 1
                continue

            # Embed the query
            try:
                query_vec = np.asarray(embedder.embed(query), dtype=np.float32)
            except Exception:
                logger.warning("Failed to embed query %s; skipping", query_id)
                queries_processed += 1
                continue

            # Fetch memories for selected anchors
            memories = []
            for anchor_id in anchor_ids:
                try:
                    anchor_memories = store.query_by_anchor(
                        pipeline.tenant_id, int(anchor_id), limit=50
                    )
                    memories.extend(anchor_memories)
                except Exception:
                    pass

            if not memories:
                queries_processed += 1
                continue

            # Batch CE scores if cross-encoder available
            ce_scores_map = {}
            if cross_encoder is not None:
                try:
                    pairs = [(query, m.body) for m in memories]
                    raw_scores = cross_encoder.predict(pairs)
                    for i, m in enumerate(memories):
                        ce_scores_map[m.memory_object_id] = float(raw_scores[i])
                except Exception:
                    logger.warning("Cross-encoder batch predict failed for query %s", query_id)

            # When mode is LLM, batch all memories into one judge call per query.
            # Benchmark mode uses deterministic F1 overlap — no LLM calls needed.
            labels_by_memory_id: dict[str, int] = {}
            if mode != "benchmark" and memories:
                batch_labels = _llm_relevance_batch(query, [m.body for m in memories])
                for m, lab in zip(memories, batch_labels):
                    labels_by_memory_id[m.memory_object_id] = lab

            for memory in memories:
                ce_score = ce_scores_map.get(memory.memory_object_id, 0.0)

                # Label
                if mode == "benchmark":
                    f1 = _token_f1(memory.body, answer_gold)
                    label = 1 if f1 > 0.3 else 0
                else:
                    label = labels_by_memory_id.get(memory.memory_object_id, 0)

                # Features
                type_vec = _type_onehot(memory.memory_type)
                quality_score = float(getattr(memory, "quality_score", 0.5))
                body_norm = min(1.0, (len(memory.body) / 4.0) / token_budget)
                days = _days_since(getattr(memory, "created_at", ""))
                recency = _recency(days)

                mem_emb = getattr(memory, "embedding", None)
                if mem_emb is not None:
                    cosine = _cosine_to_query(mem_emb, query_vec)
                else:
                    cosine = 0.0

                row = {
                    "ce_score": round(ce_score, 6),
                    "type_vec": type_vec,
                    "quality_score": round(quality_score, 6),
                    "body_norm": round(body_norm, 6),
                    "recency": round(recency, 6),
                    "cosine": round(float(cosine), 6),
                    "label": label,
                    "query_id": query_id,
                }
                fout.write(json.dumps(row) + "\n")
                total_written += 1

            queries_processed += 1
            if queries_processed % 50 == 0:
                logger.info(
                    "generate_memory_scorer_data: processed %d queries, %d pairs so far",
                    queries_processed,
                    total_written,
                )

    logger.info(
        "generate_memory_scorer_data: done — %d queries, %d pairs written to %s",
        queries_processed,
        total_written,
        output_path,
    )
    return total_written
