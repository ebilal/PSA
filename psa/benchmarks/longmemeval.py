"""
longmemeval.py — LongMemEval benchmark harness for PSA.

Three sub-commands:

  psa benchmark longmemeval ingest   — download dataset, mine sessions, build atlas
  psa benchmark longmemeval run      — query PSA for each question, generate answers
  psa benchmark longmemeval score    — score answers, write oracle labels

Results: ~/.psa/benchmarks/longmemeval/
Oracle labels: existing oracle_labels.jsonl format, picked up by 'psa train'.

Benchmarks run against an isolated tenant (default: 'longmemeval_bench').
"""

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("psa.benchmarks.longmemeval")

BENCH_TENANT = "longmemeval_bench"
HF_DATASET = "xiaowu0162/longmemeval"
RESULTS_DIR_DEFAULT = os.path.expanduser("~/.psa/benchmarks/longmemeval")


# ── Ingest ─────────────────────────────────────────────────────────────────────


def ingest(tenant_id: str = BENCH_TENANT, results_dir: str = RESULTS_DIR_DEFAULT) -> None:
    """Download LongMemEval and ingest all sessions into PSA."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "The 'huggingface_hub' library is required for LongMemEval ingestion.\n"
            "Install with: pip install huggingface_hub"
        )

    from ..convo_miner import mine_convos
    from ..config import MempalaceConfig
    from ..tenant import TenantManager

    logger.info("Loading LongMemEval dataset from HuggingFace...")
    data_path = hf_hub_download(HF_DATASET, "longmemeval_oracle", repo_type="dataset")
    with open(data_path, encoding="utf-8") as f:
        examples = json.load(f)

    sessions: Dict[str, List[Dict]] = {}
    for example in examples:
        for session_id, messages in zip(
            example.get("haystack_session_ids", []),
            example.get("haystack_sessions", []),
        ):
            if session_id not in sessions:
                sessions[session_id] = messages

    logger.info("Found %d unique sessions to ingest.", len(sessions))
    os.makedirs(results_dir, exist_ok=True)

    tm = TenantManager()
    tm.get_or_create(tenant_id)
    cfg = MempalaceConfig()

    # Set PSA_TENANT_ID so that _mine_convos_psa writes to the benchmark tenant,
    # not the user's configured default tenant.
    prev_tenant = os.environ.get("PSA_TENANT_ID")
    os.environ["PSA_TENANT_ID"] = tenant_id
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for session_id, messages in sessions.items():
                session_path = os.path.join(tmpdir, f"{session_id}.jsonl")
                _write_session_jsonl(session_path, session_id, messages)

            mine_convos(
                convo_dir=tmpdir,
                palace_path=cfg.palace_path,
            )
    finally:
        if prev_tenant is None:
            os.environ.pop("PSA_TENANT_ID", None)
        else:
            os.environ["PSA_TENANT_ID"] = prev_tenant

    logger.info("Ingestion complete. Building atlas...")
    _build_atlas(tenant_id)
    logger.info("Done. Tenant '%s' is ready for benchmarking.", tenant_id)


def _write_session_jsonl(path: str, session_id: str, messages: List[Dict]) -> None:
    """Write one session as Claude Code JSONL format for convo_miner."""
    with open(path, "w", encoding="utf-8") as f:
        for msg in messages:
            record = {
                "type": "message",
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
                "session_id": session_id,
            }
            f.write(json.dumps(record) + "\n")


def _build_atlas(tenant_id: str) -> None:
    from ..atlas import AtlasManager
    from ..memory_object import MemoryStore
    from ..tenant import TenantManager

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    store = MemoryStore(tenant.memory_db_path)
    mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
    atlas = mgr.get_or_build(store)
    logger.info("Atlas ready: version %s, %d anchors.", atlas.version, len(atlas.cards))


# ── Run ────────────────────────────────────────────────────────────────────────


def run(
    split: str = "val",
    limit: Optional[int] = None,
    tenant_id: str = BENCH_TENANT,
    results_dir: str = RESULTS_DIR_DEFAULT,
    token_budget: int = 6000,
    selector_mode: str = "cosine",
    selector_model_path: Optional[str] = None,
    packer_weights: Optional[tuple] = None,
) -> str:
    """
    Run each LongMemEval question through PSA and generate answers.

    Returns the path to the results JSONL file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Install huggingface_hub: pip install huggingface_hub")

    from ..pipeline import PSAPipeline
    from ..llm import call_llm

    logger.info("Loading LongMemEval dataset (split=%s)...", split)
    data_path = hf_hub_download(HF_DATASET, "longmemeval_oracle", repo_type="dataset")
    with open(data_path, encoding="utf-8") as f:
        all_examples = json.load(f)
    examples = [e for e in all_examples if e.get("split", "val") == split]
    if not examples:
        # Dataset may not have a split field — use all examples
        logger.warning(
            "No examples found for split=%r; using all %d examples.",
            split,
            len(all_examples),
        )
        examples = all_examples
    if limit:
        examples = examples[:limit]

    logger.info("Running %d questions (tenant=%s)...", len(examples), tenant_id)

    try:
        pipeline = PSAPipeline.from_tenant(
            tenant_id=tenant_id,
            token_budget=token_budget,
            selector_mode=selector_mode,
            selector_model_path=selector_model_path,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No atlas for tenant '{tenant_id}'. Run 'psa benchmark longmemeval ingest' first."
        )

    if packer_weights:
        pipeline._packer_weights = packer_weights

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(results_dir, f"results_{split}_{selector_mode}_{ts}.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(examples):
            question_id = example.get("question_id", f"q_{i:04d}")
            question = example.get("question", "")
            gold_answer = example.get("answer", "")

            result = pipeline.query(question)
            context_text = result.text

            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Answer the following question based only on the provided context.\n\n"
                        f"Context:\n{context_text}\n\n"
                        f"Question: {question}\n\nAnswer:"
                    ),
                }
            ]
            generated = call_llm(messages, max_tokens=256, json_mode=False)

            record = {
                "question_id": question_id,
                "question": question,
                "context_text": context_text,
                "answer_generated": generated,
                "answer_gold": gold_answer,
                "answer_session_ids": example.get("answer_session_ids", []),
                "tokens_used": result.token_count,
                "token_budget": token_budget,
                "selected_anchor_ids": [a.anchor_id for a in result.selected_anchors],
                "timing_ms": {
                    "embed": round(result.timing.embed_ms, 1),
                    "retrieve": round(result.timing.retrieve_ms, 1),
                    "select": round(result.timing.select_ms, 1),
                    "fetch": round(result.timing.fetch_ms, 1),
                    "pack": round(result.timing.pack_ms, 1),
                    "total": round(result.timing.total_ms, 1),
                },
            }
            f.write(json.dumps(record) + "\n")

            if (i + 1) % 50 == 0:
                logger.info("  %d / %d questions complete", i + 1, len(examples))

    logger.info("Results written to %s", out_path)
    return out_path


# ── Score ──────────────────────────────────────────────────────────────────────


def score(
    results_path: str,
    method: str = "both",
    tenant_id: str = BENCH_TENANT,
) -> Dict[str, Any]:
    """
    Score benchmark results and write oracle labels for failures.

    Parameters
    ----------
    results_path:
        Path to results JSONL from run().
    method:
        "exact" — F1 token overlap; "llm" — LLM-as-judge; "both" — both.
    tenant_id:
        PSA tenant for oracle label output path.
    """
    records = _load_results(results_path)
    if not records:
        raise ValueError(f"No records found in {results_path}")

    exact_f1_scores = [_f1_score(r["answer_gold"], r["answer_generated"]) for r in records]
    avg_exact_f1 = sum(exact_f1_scores) / len(exact_f1_scores)

    avg_llm = None
    if method in ("llm", "both"):
        from ..llm import call_llm

        llm_scores = [
            _llm_judge(r["question"], r["answer_gold"], r["answer_generated"], call_llm)
            for r in records
        ]
        valid = [s for s in llm_scores if s is not None]
        avg_llm = sum(valid) / len(valid) if valid else 0.0

    FAILURE_THRESHOLD = 0.3
    SUCCESS_THRESHOLD = 0.5
    oracle_labels_written = 0
    oracle_path = _oracle_labels_path(tenant_id)
    os.makedirs(os.path.dirname(oracle_path), exist_ok=True)

    with open(oracle_path, "a", encoding="utf-8") as f:
        for record, f1 in zip(records, exact_f1_scores):
            if f1 < FAILURE_THRESHOLD:
                label = _make_oracle_label(record, f1)
                f.write(json.dumps(label) + "\n")
                oracle_labels_written += 1
            elif f1 >= SUCCESS_THRESHOLD:
                # Write success labels so training has positive examples
                label = _make_oracle_label(record, f1, success=True)
                f.write(json.dumps(label) + "\n")
                oracle_labels_written += 1

    # Compute R@5 if answer_session_ids are available
    recall_scores = []
    for r in records:
        ans_ids = r.get("answer_session_ids", [])
        if ans_ids:
            hit = any(ans_id in r.get("context_text", "") for ans_id in ans_ids)
            recall_scores.append(1.0 if hit else 0.0)
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else None

    result: Dict[str, Any] = {
        "exact_f1": round(avg_exact_f1, 4),
        "n_questions": len(records),
        "oracle_labels_written": oracle_labels_written,
        "oracle_labels_path": oracle_path,
    }
    if avg_llm is not None:
        result["llm_score"] = round(avg_llm, 4)
    if avg_recall is not None:
        result["recall_at_5"] = round(avg_recall, 4)
    return result


def oracle_label(
    results_path: str,
    tenant_id: str = BENCH_TENANT,
) -> int:
    """
    Run the real OracleLabeler on benchmark results to produce proper
    training labels with identified winning anchor sets.

    Returns the number of oracle labels written.
    """
    from ..training.oracle_labeler import OracleLabeler
    from ..pipeline import PSAPipeline

    records = _load_results(results_path)
    if not records:
        raise ValueError(f"No records found in {results_path}")

    try:
        pipeline = PSAPipeline.from_tenant(tenant_id=tenant_id)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No atlas for tenant '{tenant_id}'. Run 'psa benchmark longmemeval ingest' first."
        )

    oracle_path = _oracle_labels_path(tenant_id)
    os.makedirs(os.path.dirname(oracle_path), exist_ok=True)

    labeler = OracleLabeler(pipeline=pipeline, output_path=oracle_path)

    written = 0
    for i, record in enumerate(records):
        query_id = record.get("question_id", f"lme_q_{i:04d}")
        query = record["question"]
        try:
            labeler.label(query_id=query_id, query=query)
            written += 1
        except Exception as e:
            logger.warning("Oracle labeling failed for q=%s: %s", record.get("question_id"), e)

        if (i + 1) % 50 == 0:
            logger.info("  %d / %d questions labeled", i + 1, len(records))

    logger.info("Wrote %d oracle labels to %s", written, oracle_path)
    return written


def _load_results(path: str) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _normalize_text(text) -> List[str]:
    import re

    return re.sub(r"[^\w\s]", " ", str(text).lower()).split()


def _f1_score(gold: str, pred: str) -> float:
    gold_tokens = set(_normalize_text(gold))
    pred_tokens = set(_normalize_text(pred))
    if not gold_tokens or not pred_tokens:
        return 0.0
    common = gold_tokens & pred_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _llm_judge(question: str, gold: str, generated: str, call_llm) -> Optional[float]:
    messages = [
        {
            "role": "user",
            "content": (
                f"Judge whether the answer correctly addresses the question.\n"
                f"Question: {question}\nGold answer: {gold}\nGenerated answer: {generated}\n\n"
                f"Reply with exactly one word: PASS or FAIL."
            ),
        }
    ]
    try:
        response = call_llm(messages, max_tokens=10, json_mode=False).strip().upper()
        if "PASS" in response:
            return 1.0
        if "FAIL" in response:
            return 0.0
        return None
    except Exception:
        return None


def _recall_at_k(
    retrieved_source_paths: List[str],
    answer_session_ids: List[str],
    k: int = 5,
) -> float:
    """
    Recall@k: did any answer session appear in the top-k retrieved sources?

    retrieved_source_paths are filenames like "answer_abc_1.jsonl".
    answer_session_ids are like "answer_abc_1".
    Match by checking if any answer_session_id is a prefix of any retrieved path.
    """
    if not answer_session_ids:
        return 0.0
    top_k = retrieved_source_paths[:k]
    for ans_id in answer_session_ids:
        for path in top_k:
            path_stem = path.rsplit(".", 1)[0] if "." in path else path
            if path_stem == ans_id or ans_id in path:
                return 1.0
    return 0.0


def _oracle_labels_path(tenant_id: str) -> str:
    return os.path.expanduser(f"~/.psa/tenants/{tenant_id}/training/oracle_labels.jsonl")


def _make_oracle_label(record: Dict, f1: float, success: bool = False) -> Dict:
    """Create an oracle label for a question result."""
    now = datetime.now(timezone.utc).isoformat()
    q_hash = hashlib.md5(record["question"].encode(), usedforsecurity=False).hexdigest()[:8]
    anchor_ids = record.get("selected_anchor_ids", [])
    return {
        "query_id": f"lme_{q_hash}",
        "query": record["question"],
        "atlas_version": -1,
        "runtime_model_id": "longmemeval",
        "candidate_anchor_ids": anchor_ids,
        "all_sets": [anchor_ids] if success else [],
        "winning_oracle_set": anchor_ids if success else [],
        "winning_oracle_score": f1,
        "labeled_at": now,
        "is_high_complexity": False,
        "metadata": {
            "source": "longmemeval",
            "question_id": record.get("question_id", ""),
            "exact_f1": f1,
        },
    }
