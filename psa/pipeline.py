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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .atlas import Atlas, AtlasManager
from .embeddings import EmbeddingModel
from .memory_object import MemoryObject, MemoryStore, MemoryType
from .packer import EvidencePacker, PackedContext
from .retriever import AnchorCandidate, AnchorRetriever
from .selector import AnchorSelector, SelectedAnchor
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
        }


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
    ):
        self.store = store
        self.atlas = atlas
        self.embedding_model = embedding_model
        self.selector = selector or AnchorSelector.cosine()
        self.token_budget = token_budget
        self.tenant_id = tenant_id
        self.psa_mode = psa_mode

        self._retriever = AnchorRetriever(atlas)
        self._packer = EvidencePacker(memory_store=store)

    def query(
        self,
        query: str,
        top_k_candidates: int = 24,
    ) -> PSAResult:
        """
        Run the full PSA query pipeline.

        Parameters
        ----------
        query:
            User query string.
        top_k_candidates:
            Retriever shortlist size (default 24 per plan).

        Returns
        -------
        PSAResult with packed context and timing.
        """
        timing = QueryTiming()

        # Step 1: Embed query
        t0 = time.perf_counter()
        query_vec = self.embedding_model.embed(query)
        timing.embed_ms = (time.perf_counter() - t0) * 1000

        # Step 2: Retrieve anchor candidates
        t0 = time.perf_counter()
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
            return PSAResult(
                query=query,
                packed_context=packed,
                selected_anchors=[],
                candidates=[],
                timing=timing,
                tenant_id=self.tenant_id,
                psa_mode=self.psa_mode,
            )

        # Step 3: Select anchors
        t0 = time.perf_counter()
        selected = self.selector.select(
            query=query,
            candidates=candidates,
            query_vec=query_vec,
        )
        timing.select_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            "Selected %d anchors in %.1fms", len(selected), timing.select_ms
        )

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
            return PSAResult(
                query=query,
                packed_context=packed,
                selected_anchors=[],
                candidates=candidates,
                timing=timing,
                tenant_id=self.tenant_id,
                psa_mode=self.psa_mode,
            )

        # Step 4: Fetch memories for selected anchors
        t0 = time.perf_counter()
        memories = self._fetch_memories(selected)
        timing.fetch_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            "Fetched %d memories in %.1fms", len(memories), timing.fetch_ms
        )

        # Step 5: Pack context
        t0 = time.perf_counter()
        packed = self._packer.pack_memories_direct(
            query=query,
            memories=memories,
            token_budget=self.token_budget,
            query_vec=query_vec,
            store=self.store,
        )
        timing.pack_ms = (time.perf_counter() - t0) * 1000

        # Record usage telemetry: these memories made it into packed context
        if packed.memory_ids:
            try:
                self.store.record_packed(packed.memory_ids)
            except Exception:
                logger.debug("Failed to record packed telemetry", exc_info=True)

        logger.debug(
            "Packed %d tokens in %.1fms", packed.token_count, timing.pack_ms
        )
        logger.info(
            "PSA query complete: %.1fms total (%d memories, %d tokens)",
            timing.total_ms,
            len(memories),
            packed.token_count,
        )

        return PSAResult(
            query=query,
            packed_context=packed,
            selected_anchors=selected,
            candidates=candidates,
            timing=timing,
            tenant_id=self.tenant_id,
            psa_mode=self.psa_mode,
        )

    def _fetch_memories(self, selected: List[SelectedAnchor]) -> List[MemoryObject]:
        """
        Fetch non-duplicate memories for selected anchors.

        Memories are sorted within each anchor by quality_score desc,
        then deduplicated across anchors by memory_object_id.
        """
        seen_ids: set = set()
        memories: List[MemoryObject] = []

        for sel in selected:
            anchor_memories = self.store.query_by_anchor(
                tenant_id=self.tenant_id,
                anchor_id=sel.anchor_id,
                limit=50,
            )
            for mo in anchor_memories:
                if mo.memory_object_id in seen_ids:
                    continue
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
        return memories

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
                f"No atlas found for tenant '{tenant_id}'. "
                "Run 'psa atlas build' to build one."
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

        selector_kwargs = dict(mode=selector_mode, model_path=selector_model_path)
        if selector_threshold is not None:
            selector_kwargs["threshold"] = selector_threshold
        selector = AnchorSelector(**selector_kwargs)

        return cls(
            store=store,
            atlas=atlas,
            embedding_model=embedding_model,
            selector=selector,
            token_budget=token_budget,
            tenant_id=tenant_id,
            psa_mode=psa_mode,
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
