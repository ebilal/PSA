"""
pipeline.py — Unified PSA query pipeline.

PSAPipeline orchestrates the full retrieval flow:
  retriever → selector → memory fetch → packer

In psa_mode="side-by-side" it runs alongside the existing raw ChromaDB search
and returns both results so callers can compare. In psa_mode="primary" it is
the sole search path.

Usage::

    pipeline = PSAPipeline.from_config()
    result = pipeline.query("What auth pattern did we use?")
    print(result.text)
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .atlas import Atlas, AtlasManager
from .coactivation import CoActivationSelector
from .constraint_scorer import ConstraintScorer
from .embeddings import EmbeddingModel
from .full_atlas_scorer import FullAtlasScorer
from .memory_object import MemoryObject, MemoryStore
from .memory_scorer import MemoryScorer
from .packer import EvidencePacker, PackedContext
from .query_frame import extract_query_frame
from .retriever import AnchorCandidate, AnchorRetriever
from .selector import AnchorSelector, SelectedAnchor
from .synthesizer import AnchorSynthesizer
from .tenant import TenantManager

logger = logging.getLogger("psa.pipeline")


# ── Timing ────────────────────────────────────────────────────────────────────


@dataclass
class QueryTiming:
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    select_ms: float = 0.0
    fetch_ms: float = 0.0
    pack_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.retrieve_ms + self.select_ms + self.fetch_ms + self.pack_ms


# ── PSAResult ─────────────────────────────────────────────────────────────────


@dataclass
class PSAResult:
    """Complete result from a PSA pipeline query."""

    query: str
    packed_context: PackedContext
    selected_anchors: List[SelectedAnchor]
    candidates: List[AnchorCandidate]
    timing: QueryTiming
    tenant_id: str
    psa_mode: str
    selection_mode: str = "legacy"

    @property
    def text(self) -> str:
        return self.packed_context.text

    @property
    def token_count(self) -> int:
        return self.packed_context.token_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "text": self.text,
            "token_count": self.token_count,
            "selected_anchor_ids": [a.anchor_id for a in self.selected_anchors],
            "memory_ids": self.packed_context.memory_ids,
            "timing_ms": {
                "embed": round(self.timing.embed_ms, 1),
                "retrieve": round(self.timing.retrieve_ms, 1),
                "select": round(self.timing.select_ms, 1),
                "fetch": round(self.timing.fetch_ms, 1),
                "pack": round(self.timing.pack_ms, 1),
                "total": round(self.timing.total_ms, 1),
            },
            "psa_mode": self.psa_mode,
            "tenant_id": self.tenant_id,
            "selection_mode": self.selection_mode,
        }


# ── compose_and_record ────────────────────────────────────────────────────────


def compose_and_record(
    *,
    tenant_id: str,
    trace_record: dict,
    attribution,
    selected_anchor_ids,
    config,
    record_signals_fn=None,
) -> None:
    """Write trace first; only if it succeeded, record ledger signals.

    The stage 2 ledger is a materialised view of events the trace records.
    To guarantee ``psa advertisement rebuild-ledger`` is canonical, trace must
    be source of truth: if the trace write fails or is disabled, skip the
    ledger write so the two cannot drift.

    ``record_signals_fn`` is injectable for unit tests. Production callers
    omit it and the function constructs the default ledger writer.
    """
    from .trace import write_trace

    trace_record["retrieval_attribution"] = [
        {
            "anchor_id": a.anchor_id,
            "bm25_argmax_pattern": a.argmax_pattern,
            "bm25_epsilon_tied": list(a.eps_tied_patterns),
            "bm25_floor_passed": a.bm25_floor_passed,
        }
        for a in attribution
    ]
    trace_written = write_trace(trace_record, tenant_id=tenant_id)
    if not trace_written:
        return
    if not getattr(config, "tracking_enabled", False):
        return
    if record_signals_fn is None:
        import sqlite3

        def _default(**kw):
            db_path = os.path.expanduser(f"~/.psa/tenants/{tenant_id}/memory.sqlite3")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            with sqlite3.connect(db_path) as db:
                from .advertisement.ledger import create_schema, record_query_signals

                create_schema(db)
                record_query_signals(db=db, **kw)

        record_signals_fn = _default
    record_signals_fn(
        attribution=attribution,
        selected_anchor_ids=selected_anchor_ids,
        config=config,
    )


# ── PSAPipeline ───────────────────────────────────────────────────────────────


class PSAPipeline:
    """
    Unified PSA query pipeline.

    Retriever (BM25+dense) → Selector (cosine or trained) →
    Memory fetch → Deterministic packer.
    """

    def __init__(
        self,
        store: MemoryStore,
        atlas: Atlas,
        embedding_model: EmbeddingModel,
        selector: Optional[AnchorSelector] = None,
        token_budget: int = 6000,
        tenant_id: str = "default",
        psa_mode: str = "side-by-side",
        full_atlas_scorer: Optional[FullAtlasScorer] = None,
        coactivation_selector: Optional[CoActivationSelector] = None,
        memory_scorer: Optional[MemoryScorer] = None,
    ):
        self.store = store
        self.atlas = atlas
        self.embedding_model = embedding_model
        self.selector = selector or AnchorSelector.cosine()
        self.token_budget = token_budget
        self.tenant_id = tenant_id
        self.psa_mode = psa_mode
        self.full_atlas_scorer = full_atlas_scorer
        self.coactivation_selector = coactivation_selector
        self.memory_scorer = memory_scorer

        self._retriever = AnchorRetriever(atlas)
        self._packer = EvidencePacker(memory_store=store)
        self._synthesizer = AnchorSynthesizer()
        self._constraint_scorer = ConstraintScorer()

        # Stage 2 advertisement-decay config. Defaults are off — tracking is
        # lazily evaluated from the user's MempalaceConfig when the pipeline
        # first runs a query. Kept as a property rather than an __init__ arg
        # so existing callers stay unchanged.
        self._ad_config = None

    @property
    def ad_config(self):
        """Load AdvertisementDecayConfig on first access."""
        if self._ad_config is None:
            from .advertisement.config import AdvertisementDecayConfig
            from .config import MempalaceConfig

            self._ad_config = AdvertisementDecayConfig.from_mempalace(MempalaceConfig())
        return self._ad_config

    def query(
        self,
        query: str,
        top_k_candidates: int = 32,
        *,
        query_origin: str = "interactive",
    ) -> PSAResult:
        """
        Run the full PSA query pipeline.

        Parameters
        ----------
        query:
            User query string.
        top_k_candidates:
            Retriever shortlist size (default 24 per plan).
        query_origin:
            Tag for the trace record (``"interactive"`` by default).
            Pass ``"labeling"``, ``"benchmark"``, or ``"inspect"`` from
            non-interactive callers so diag rollups can filter cleanly.

        Returns
        -------
        PSAResult with packed context and timing.
        """
        import datetime as _dt
        import hashlib as _h

        from .trace import new_trace_record

        t_start = time.perf_counter()
        _run_id = (
            _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "_"
            + _h.md5(query.encode(), usedforsecurity=False).hexdigest()[:6]
        )
        _trace = new_trace_record(
            run_id=_run_id,
            timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            tenant_id=self.tenant_id,
            atlas_version=getattr(self.atlas, "version", 0),
            query=query,
            query_origin=query_origin,
        )
        _trace["token_budget"] = self.token_budget
        # Maps memory_object_id → the selected anchor_id via which it was first fetched.
        _memory_to_source_anchor: dict[str, int] = {}

        timing = QueryTiming()
        result: Optional[PSAResult] = None
        _bm25_topk_anchor_ids: Optional[List[int]] = None

        try:
            # Step 1: Embed query
            t0 = time.perf_counter()
            query_vec = self.embedding_model.embed(query)
            timing.embed_ms = (time.perf_counter() - t0) * 1000

            # Step 1.5: Extract query frame
            query_frame = extract_query_frame(query)
            logger.debug(
                "QueryFrame: target=%s mode=%s entities=%s confidence=%.2f",
                query_frame.answer_target,
                query_frame.retrieval_mode,
                query_frame.entities,
                query_frame.confidence,
            )

            # Steps 2 + 3: Retrieve and select anchors (3-level degradation)
            if self.full_atlas_scorer is not None:
                # Level 1: Full-atlas scoring path
                t0 = time.perf_counter()
                anchor_scores = self.full_atlas_scorer.score_all(query, query_vec=query_vec)
                timing.retrieve_ms = (time.perf_counter() - t0) * 1000

                logger.debug(
                    "Full-atlas scored %d anchors in %.1fms",
                    len(anchor_scores),
                    timing.retrieve_ms,
                )

                candidates = []  # no retriever shortlist in full-atlas mode

                if not anchor_scores:
                    packed = PackedContext(
                        query=query,
                        text="(no anchor candidates found — atlas may be empty or unbuilt)",
                        token_count=0,
                        memory_ids=[],
                        sections=[],
                        untyped_count=0,
                    )
                    result = PSAResult(
                        query=query,
                        packed_context=packed,
                        selected_anchors=[],
                        candidates=[],
                        timing=timing,
                        tenant_id=self.tenant_id,
                        psa_mode=self.psa_mode,
                        selection_mode="full_atlas",
                    )
                    _trace["selection_mode"] = "full_atlas"
                    _trace["result_kind"] = "empty_selection"
                    _trace["empty_selection"] = True
                    # top_anchor_scores stays []
                else:
                    t0 = time.perf_counter()
                    if self.coactivation_selector is not None:
                        # Level 1a: co-activation model selection
                        selected = self.coactivation_selector.select(query_vec, anchor_scores)
                        selection_mode = "coactivation"
                        # Log the REFINED scores the selector actually decided over,
                        # re-ranked by refined score desc. Fall back to raw CE if the
                        # side-channel attribute isn't populated (shouldn't happen when
                        # select() was called, but defensive).
                        refined_pairs = getattr(
                            self.coactivation_selector, "last_refined_scores", None
                        )
                        selected_ids = {sa.anchor_id for sa in selected}
                        if refined_pairs:
                            refined_sorted = sorted(refined_pairs, key=lambda p: p[1], reverse=True)
                            _trace["top_anchor_scores"] = [
                                {
                                    "anchor_id": aid,
                                    "score": round(float(score), 4),
                                    "score_source": "coactivation_refined",
                                    "rank": rank + 1,
                                    "selected": aid in selected_ids,
                                }
                                for rank, (aid, score) in enumerate(refined_sorted[:24])
                            ]
                        else:
                            # Defensive fallback: label honestly as pre-coactivation CE
                            # scores so rollups can detect this and exclude them from
                            # "near miss under live selector" interpretations.
                            _trace["top_anchor_scores"] = [
                                {
                                    "anchor_id": s.anchor_id,
                                    "score": round(float(s.ce_score), 4),
                                    "score_source": "full_atlas",
                                    "rank": rank + 1,
                                    "selected": s.anchor_id in selected_ids,
                                }
                                for rank, s in enumerate(anchor_scores[:24])
                            ]
                    else:
                        # Level 1b: top-k from full-atlas scores, wrapped as SelectedAnchor
                        top_scores = anchor_scores[: self.selector.max_k]
                        selected = [
                            SelectedAnchor(
                                anchor_id=s.anchor_id,
                                selector_score=s.ce_score,
                                mode="full_atlas",
                                candidate=None,
                            )
                            for s in top_scores
                        ]
                        selection_mode = "full_atlas"
                        selected_ids = {sa.anchor_id for sa in selected}
                        _trace["top_anchor_scores"] = [
                            {
                                "anchor_id": s.anchor_id,
                                "score": round(float(s.ce_score), 4),
                                "score_source": "full_atlas",
                                "rank": rank + 1,
                                "selected": s.anchor_id in selected_ids,
                            }
                            for rank, s in enumerate(anchor_scores[:24])
                        ]
                    timing.select_ms = (time.perf_counter() - t0) * 1000
            else:
                # Level 2: Legacy retriever + selector path
                t0 = time.perf_counter()
                # When stage 2 tracking is enabled, additionally capture the
                # BM25-side top-K shortlist so record_query_signals can gate
                # template credit on lexical contribution.
                try:
                    _tracking = self.ad_config.tracking_enabled
                except Exception:
                    _tracking = False
                if _tracking:
                    _retrieval = self._retriever.retrieve_with_bm25_topk(
                        query=query,
                        embedding_model=self.embedding_model,
                        top_k=top_k_candidates,
                        bm25_topk_floor=self.ad_config.bm25_topk_floor,
                        query_vec=query_vec,
                    )
                    _bm25_topk_anchor_ids = _retrieval.bm25_topk_anchor_ids
                    candidates = self._retriever.retrieve(
                        query=query,
                        embedding_model=self.embedding_model,
                        top_k=top_k_candidates,
                        query_vec=query_vec,
                    )
                else:
                    _bm25_topk_anchor_ids = None
                    candidates = self._retriever.retrieve(
                        query=query,
                        embedding_model=self.embedding_model,
                        top_k=top_k_candidates,
                        query_vec=query_vec,
                    )
                timing.retrieve_ms = (time.perf_counter() - t0) * 1000

                logger.debug(
                    "Retrieved %d candidates in %.1fms", len(candidates), timing.retrieve_ms
                )

                if not candidates:
                    packed = PackedContext(
                        query=query,
                        text="(no anchor candidates found — atlas may be empty or unbuilt)",
                        token_count=0,
                        memory_ids=[],
                        sections=[],
                        untyped_count=0,
                    )
                    result = PSAResult(
                        query=query,
                        packed_context=packed,
                        selected_anchors=[],
                        candidates=[],
                        timing=timing,
                        tenant_id=self.tenant_id,
                        psa_mode=self.psa_mode,
                        selection_mode="legacy",
                    )
                    _trace["selection_mode"] = "legacy"
                    _trace["result_kind"] = "empty_selection"
                    _trace["empty_selection"] = True
                else:
                    t0 = time.perf_counter()
                    selected = self.selector.select(
                        query=query,
                        candidates=candidates,
                        query_vec=query_vec,
                    )
                    timing.select_ms = (time.perf_counter() - t0) * 1000
                    selection_mode = "legacy"
                    selected_ids = {sa.anchor_id for sa in selected}
                    _trace["top_anchor_scores"] = [
                        {
                            "anchor_id": c.anchor_id,
                            "score": round(float(c.rrf_score), 4),
                            "score_source": "retriever",
                            "rank": rank + 1,
                            "selected": c.anchor_id in selected_ids,
                        }
                        for rank, c in enumerate(candidates[:24])
                    ]

            # The two early-return paths above set result; skip selection-dependent work.
            if result is None:
                logger.debug("Selected %d anchors in %.1fms", len(selected), timing.select_ms)

                # Accumulate query fingerprints for above-threshold selected anchors
                if self.atlas.fingerprint_store is not None and selected:
                    for sa in selected:
                        self.atlas.fingerprint_store.append(sa.anchor_id, query)
                    self.atlas.fingerprint_store.save()

                # No anchors met the threshold — nothing relevant in memory
                if not selected:
                    packed = PackedContext(
                        query=query,
                        text="(no relevant memories found for this query)",
                        token_count=0,
                        memory_ids=[],
                        sections=[],
                        untyped_count=0,
                    )
                    result = PSAResult(
                        query=query,
                        packed_context=packed,
                        selected_anchors=[],
                        candidates=candidates,
                        timing=timing,
                        tenant_id=self.tenant_id,
                        psa_mode=self.psa_mode,
                        selection_mode=selection_mode,
                    )
                    _trace["selection_mode"] = selection_mode
                    _trace["result_kind"] = "empty_selection"
                    _trace["empty_selection"] = True
                    _trace["selected_anchor_ids"] = []
                else:
                    # Step 4: Fetch memories for selected anchors (inline for source tracking)
                    t0 = time.perf_counter()
                    seen_ids: set = set()
                    memories: List[MemoryObject] = []
                    for sa in selected:
                        anchor_memories = self.store.query_by_anchor(
                            tenant_id=self.tenant_id,
                            anchor_id=sa.anchor_id,
                            limit=50,
                        )
                        for mo in anchor_memories:
                            if mo.memory_object_id in _memory_to_source_anchor:
                                continue  # first selected anchor wins on multi-assign
                            _memory_to_source_anchor[mo.memory_object_id] = sa.anchor_id
                            if mo.memory_object_id not in seen_ids:
                                seen_ids.add(mo.memory_object_id)
                                memories.append(mo)

                    # Record usage telemetry: these memories were fetched for selected anchors
                    if memories:
                        try:
                            self.store.record_selected([m.memory_object_id for m in memories])
                        except Exception:
                            logger.debug("Failed to record selected telemetry", exc_info=True)

                    # Re-sort globally by quality_score desc
                    memories.sort(key=lambda m: m.quality_score, reverse=True)
                    timing.fetch_ms = (time.perf_counter() - t0) * 1000

                    logger.debug("Fetched %d memories in %.1fms", len(memories), timing.fetch_ms)

                    # Level 2: Memory-level scoring (if available)
                    _pre_ranked = False
                    if self.memory_scorer is not None and memories:
                        t0_l2 = time.perf_counter()
                        scored_memories = self.memory_scorer.score(
                            query=query,
                            query_vec=query_vec,
                            memories=memories,
                        )
                        scored_memories = self._constraint_scorer.adjust_scores(
                            scored_memories, query_frame
                        )
                        memories = [sm.memory for sm in scored_memories]
                        logger.debug(
                            "Level 2 scored %d memories in %.1fms",
                            len(memories),
                            (time.perf_counter() - t0_l2) * 1000,
                        )
                        _pre_ranked = True
                    elif memories:
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
                        memories = [sm.memory for sm in scored_as_list]
                        _pre_ranked = True

                    # Step 5: Synthesize context
                    t0 = time.perf_counter()
                    _synthesis_succeeded = False
                    try:
                        synthesis_text = self._synthesizer.synthesize(
                            query=query,
                            memories=memories,
                            query_vec=query_vec,
                            token_budget=self.token_budget,
                        )
                        packed = PackedContext(
                            query=query,
                            text=synthesis_text,
                            token_count=len(synthesis_text) // 4,
                            memory_ids=[m.memory_object_id for m in memories],
                            sections=[],
                            untyped_count=0,
                        )
                        _synthesis_succeeded = True
                    except Exception:
                        logger.debug("Synthesis failed, falling back to packer", exc_info=True)
                        packed = self._packer.pack_memories_direct(
                            query=query,
                            memories=memories,
                            token_budget=self.token_budget,
                            query_vec=query_vec,
                            store=self.store,
                            pre_ranked=_pre_ranked,
                        )
                    timing.pack_ms = (time.perf_counter() - t0) * 1000

                    # Record usage telemetry: these memories made it into packed context
                    if packed.memory_ids:
                        try:
                            self.store.record_packed(packed.memory_ids)
                        except Exception:
                            logger.debug("Failed to record packed telemetry", exc_info=True)

                    logger.debug("Packed %d tokens in %.1fms", packed.token_count, timing.pack_ms)
                    logger.info(
                        "PSA query complete: %.1fms total (%d memories, %d tokens)",
                        timing.total_ms,
                        len(memories),
                        packed.token_count,
                    )

                    result = PSAResult(
                        query=query,
                        packed_context=packed,
                        selected_anchors=selected,
                        candidates=candidates,
                        timing=timing,
                        tenant_id=self.tenant_id,
                        psa_mode=self.psa_mode,
                        selection_mode=selection_mode,
                    )
                    _trace["selection_mode"] = selection_mode
                    _trace["result_kind"] = (
                        "synthesized" if _synthesis_succeeded else "packer_fallback"
                    )
                    _trace["selected_anchor_ids"] = [sa.anchor_id for sa in selected]
                    _trace["packed_memories"] = [
                        {
                            "memory_id": mid,
                            "source_anchor_id": _memory_to_source_anchor.get(mid, -1),
                        }
                        for mid in packed.memory_ids
                    ]

        finally:
            # Tail: populate remaining trace fields, write once, return.
            if result is None:
                # An exception bubbled out of the pipeline body. Synthesize a
                # stub result so callers never see None, and tag the trace as
                # pipeline_error — semantically distinct from empty_selection
                # so the miss rollup doesn't count crashes as misses.
                packed_fallback = PackedContext(
                    query=query,
                    text="(pipeline error)",
                    token_count=0,
                    memory_ids=[],
                    sections=[],
                    untyped_count=0,
                )
                result = PSAResult(
                    query=query,
                    packed_context=packed_fallback,
                    selected_anchors=[],
                    candidates=[],
                    timing=timing,
                    tenant_id=self.tenant_id,
                    psa_mode=self.psa_mode,
                    selection_mode="legacy",
                )
                _trace["result_kind"] = "pipeline_error"
                # empty_selection stays False — this is a crash, not an empty selection.

            _trace["tokens_used"] = result.packed_context.token_count
            _trace["timing_ms"] = {
                "embed": round(timing.embed_ms, 1),
                "retrieve": round(timing.retrieve_ms, 1),
                "select": round(timing.select_ms, 1),
                "fetch": round(timing.fetch_ms, 1),
                "pack": round(timing.pack_ms, 1),
                "total": round((time.perf_counter() - t_start) * 1000, 1),
            }
            # Populate packed_memories only if not already set (success path sets it above).
            if not _trace["packed_memories"]:
                _trace["packed_memories"] = [
                    {
                        "memory_id": mid,
                        "source_anchor_id": _memory_to_source_anchor.get(mid, -1),
                    }
                    for mid in result.packed_context.memory_ids
                ]
            # Stage 2: trace-first, ledger-second. When tracking is disabled,
            # compose_and_record writes trace and skips the ledger entirely.
            _retrieved_anchor_ids = [c.anchor_id for c in candidates] if "candidates" in locals() else []
            _selected_ids = set(_trace.get("selected_anchor_ids", []))
            if _bm25_topk_anchor_ids is not None:
                from .advertisement.ledger import compute_attribution

                _attribution = compute_attribution(
                    query=query,
                    retrieved_anchor_ids=_retrieved_anchor_ids,
                    atlas=self.atlas,
                    bm25_topk_anchor_ids=_bm25_topk_anchor_ids,
                    epsilon=self.ad_config.epsilon,
                )
                _cfg = self.ad_config
            else:
                _attribution = []
                _cfg = type("DisabledConfig", (), {"tracking_enabled": False})()

            compose_and_record(
                tenant_id=self.tenant_id,
                trace_record=_trace,
                attribution=_attribution,
                selected_anchor_ids=_selected_ids,
                config=_cfg,
            )

        return result

    def packed_context_for_anchors(
        self,
        query: str,
        anchor_ids: List[int],
    ) -> PackedContext:
        """
        Pack context for a specific set of anchor IDs.

        Used by the oracle labeler to evaluate candidate anchor sets —
        pack the memories from these anchors and return the result.
        """
        seen_ids: set = set()
        memories: List[MemoryObject] = []
        for aid in anchor_ids:
            for mo in self.store.query_by_anchor(self.tenant_id, aid, limit=50):
                if mo.memory_object_id not in seen_ids:
                    seen_ids.add(mo.memory_object_id)
                    memories.append(mo)

        query_vec = self.embedding_model.embed(query)
        return self._packer.pack_memories_direct(
            query=query,
            memories=memories,
            token_budget=self.token_budget,
            query_vec=query_vec,
            store=self.store,
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Simplified search API compatible with searcher.search_memories() return format.

        Returns dict with "query", "results" (list of hits), and "psa_context".
        """
        result = self.query(query)

        hits = []
        for mo in result.packed_context.memory_ids:
            fetched = self.store.get(mo)
            if fetched:
                hits.append(
                    {
                        "text": fetched.body,
                        "title": fetched.title,
                        "memory_type": fetched.memory_type.value,
                        "similarity": 1.0,  # relevance already baked into selection
                        "source_file": fetched.source_ids[0] if fetched.source_ids else "?",
                        "wing": "psa",
                        "room": fetched.memory_type.value,
                    }
                )
            if len(hits) >= n_results:
                break

        return {
            "query": query,
            "results": hits,
            "psa_context": result.to_dict(),
        }

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_tenant(
        cls,
        tenant_id: str = "default",
        token_budget: int = 6000,
        selector_mode: str = "cosine",
        selector_model_path: Optional[str] = None,
        selector_max_k: int = 6,
        selector_min_k: Optional[int] = None,
        selector_rerank_only: bool = False,
        psa_mode: str = "side-by-side",
        base_dir: Optional[str] = None,
    ) -> "PSAPipeline":
        """
        Build a PSAPipeline for a tenant using default paths.

        Raises AtlasCorpusTooSmall / FileNotFoundError if no atlas is available.
        """
        tm = TenantManager(base_dir=base_dir)
        tenant = tm.get_or_create(tenant_id)
        store = MemoryStore(db_path=tenant.memory_db_path)
        embedding_model = EmbeddingModel()

        atlas_mgr = AtlasManager(
            tenant_dir=tenant.root_dir,
            tenant_id=tenant_id,
        )
        atlas = atlas_mgr.get_atlas()
        if atlas is None:
            raise FileNotFoundError(
                f"No atlas found for tenant '{tenant_id}'. Run 'psa atlas build' to build one."
            )

        # Read lifecycle state for selector mode override
        lifecycle_path = os.path.join(tenant.root_dir, "lifecycle_state.json")
        if os.path.exists(lifecycle_path):
            try:
                with open(lifecycle_path) as _lf:
                    lstate = json.load(_lf)
                selector_mode = lstate.get("selector_mode", selector_mode)
                selector_model_path = lstate.get("selector_model_path", selector_model_path)
            except (json.JSONDecodeError, OSError):
                pass

        # Load calibrated threshold from trained model metadata
        selector_threshold = None
        if selector_model_path:
            sv_path = os.path.join(selector_model_path, "selector_version.json")
            try:
                with open(sv_path) as _sf:
                    sv_meta = json.load(_sf)
                selector_threshold = sv_meta.get("threshold_tau")
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                pass

        # Trained selector default: rerank-only (CE ranks, always returns max_k).
        # Ablation evidence: rerank_only achieves R@5=0.896 vs threshold's 0.766.
        if selector_mode == "trained" and not selector_rerank_only and selector_min_k is None:
            selector_rerank_only = True

        selector_kwargs = dict(
            mode=selector_mode,
            model_path=selector_model_path,
            max_k=selector_max_k,
            min_k=selector_min_k,
            rerank_only=selector_rerank_only,
        )
        if selector_threshold is not None:
            selector_kwargs["threshold"] = selector_threshold
        selector = AnchorSelector(**selector_kwargs)

        # Load FullAtlasScorer from selector model path (if available)
        full_atlas_scorer = None
        if selector_model_path:
            try:
                full_atlas_scorer = FullAtlasScorer.from_model_path(
                    model_path=selector_model_path,
                    atlas=atlas,
                )
            except Exception:
                logger.debug(
                    "FullAtlasScorer not loaded from %s", selector_model_path, exc_info=True
                )

        # Load CoActivationSelector (if model exists)
        coactivation_selector = None
        coactivation_model_dir = os.path.join(tenant.root_dir, "models", "coactivation_latest")
        coactivation_version_path = os.path.join(
            coactivation_model_dir, "coactivation_version.json"
        )
        _device = "cpu"
        if os.path.exists(coactivation_version_path):
            try:
                import torch as _torch

                _device = "mps" if _torch.backends.mps.is_available() else "cpu"
                coactivation_selector = CoActivationSelector.from_model_path(
                    model_path=coactivation_model_dir,
                    device=_device,
                )
            except Exception:
                logger.debug(
                    "CoActivationSelector not loaded from %s",
                    coactivation_model_dir,
                    exc_info=True,
                )

        # Load MemoryScorer (if model exists)
        memory_scorer = None
        scorer_model_dir = os.path.join(tenant.root_dir, "models", "memory_scorer_latest")
        scorer_meta = os.path.join(scorer_model_dir, "memory_scorer_version.json")
        if os.path.exists(scorer_meta) and full_atlas_scorer is not None:
            try:
                memory_scorer = MemoryScorer.from_model_path(
                    model_path=scorer_model_dir,
                    cross_encoder=full_atlas_scorer._cross_encoder,
                    device=_device,
                )
            except Exception:
                logger.debug("MemoryScorer not loaded", exc_info=True)

        return cls(
            store=store,
            atlas=atlas,
            embedding_model=embedding_model,
            selector=selector,
            token_budget=token_budget,
            tenant_id=tenant_id,
            psa_mode=psa_mode,
            full_atlas_scorer=full_atlas_scorer,
            coactivation_selector=coactivation_selector,
            memory_scorer=memory_scorer,
        )

    @classmethod
    def from_config(cls) -> "PSAPipeline":
        """Build a PSAPipeline from the user's PSA config."""
        from .config import MempalaceConfig

        cfg = MempalaceConfig()
        return cls.from_tenant(
            tenant_id=cfg.tenant_id,
            token_budget=cfg.token_budget,
            psa_mode=cfg.psa_mode,
        )
