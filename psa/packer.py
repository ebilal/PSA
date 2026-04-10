"""
packer.py — Transitional evidence packer for PSA Phase 1.

Takes raw ChromaDB retrieval results (from the existing search path) and
organizes them into a role-structured context block under a token budget.

When typed MemoryObjects exist for retrieved chunks (linked via source_id),
they are used to annotate the role section. Untyped results fall into
"RAW CONTEXT" — nothing is lost during the transition.

This is the Phase 1 transitional packer. The Phase 3 runtime packer operates
on selected anchors → memory objects → supporting chunks instead of directly
on raw retrieval results.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .memory_object import MemoryObject, MemoryStore, MemoryType

logger = logging.getLogger("psa.packer")

# ── Token budget ──────────────────────────────────────────────────────────────

CHARS_PER_TOKEN = 4  # rough approximation


def _token_count(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


# ── Role sections (ordered by priority in the packed context) ─────────────────

ROLE_ORDER = [
    MemoryType.FAILURE,
    MemoryType.PROCEDURAL,
    MemoryType.TOOL_USE,
    MemoryType.EPISODIC,
    MemoryType.SEMANTIC,
    MemoryType.WORKING_DERIVATIVE,
]

ROLE_HEADERS = {
    MemoryType.FAILURE: "FAILURE WARNINGS",
    MemoryType.PROCEDURAL: "PROCEDURAL GUIDANCE",
    MemoryType.TOOL_USE: "TOOL-USE NOTES",
    MemoryType.EPISODIC: "RELEVANT PRIOR EPISODES",
    MemoryType.SEMANTIC: "FACTS & CONCEPTS",
    MemoryType.WORKING_DERIVATIVE: "WORKING NOTES",
}


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class PackedSection:
    """A single role section in the packed context."""

    role: Optional[MemoryType]  # None → RAW CONTEXT
    header: str
    items: List[str] = field(default_factory=list)
    memory_ids: List[str] = field(default_factory=list)

    def render(self) -> str:
        if not self.items:
            return ""
        return self.header + "\n" + "\n".join(f"- {item}" for item in self.items)

    @property
    def token_count(self) -> int:
        return _token_count(self.render())


@dataclass
class PackedContext:
    """The final packed context returned to callers."""

    query: str
    text: str
    token_count: int
    memory_ids: List[str]
    sections: List[PackedSection]
    untyped_count: int  # number of raw (untyped) results included


# ── Deduplication helpers ─────────────────────────────────────────────────────


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:12]


# ── Hit formatting helpers ────────────────────────────────────────────────────


def _format_memory_item(
    mo: MemoryObject,
    similarity: Optional[float] = None,
    source_context: Optional[str] = None,
    max_body_chars: int = 300,
    evidence_text: Optional[str] = None,
) -> str:
    """Render a MemoryObject as a bullet point with optional source context."""
    parts = [mo.title]
    if mo.body and mo.body != mo.title:
        body = mo.body if len(mo.body) <= max_body_chars else mo.body[: max_body_chars - 3] + "..."
        parts.append(body)
    if mo.memory_type == MemoryType.EPISODIC and mo.success_label is not None:
        parts.append(f"outcome: {'success' if mo.success_label else 'failure'}")
    if similarity is not None:
        parts.append(f"(relevance: {similarity:.2f})")
    text = " — ".join(parts)
    if source_context:
        text += f"\n  Source: {source_context}"
    if evidence_text:
        parts.append(f"  Source context: {evidence_text}")
    return text


def _compute_relevance(
    memories: List[MemoryObject],
    query_vec: Optional[List[float]],
) -> List[float]:
    """Compute cosine similarity between each memory and the query embedding."""
    if query_vec is None:
        return [0.0] * len(memories)
    try:
        from .embeddings import EmbeddingModel

        return [
            EmbeddingModel.cosine_similarity(query_vec, mo.embedding) if mo.embedding else 0.0
            for mo in memories
        ]
    except Exception:
        return [0.0] * len(memories)


def _fetch_source_path(
    store: Optional[MemoryStore],
    source_ids: List[str],
) -> Optional[str]:
    """Follow source_ids to get the source file path for provenance."""
    if not store or not source_ids:
        return None
    try:
        source = store.get_source(source_ids[0])
        if source and source.source_path:
            # Show just the filename, not the full path
            import os

            return os.path.basename(source.source_path)
    except Exception:
        pass
    return None


def _fetch_evidence_text(
    store: Optional[MemoryStore],
    evidence_spans: List,
    max_chars: int = 500,
    body_hint: str = "",
    source_ids: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Retrieve relevant portions of raw source text.

    Priority:
    1. Evidence spans (exact offsets into raw source)
    2. Keyword window (find body_hint terms in raw source, extract surrounding context)

    Returns extracted text or None if no source available.
    """
    if not store:
        return None

    # Try evidence spans first
    if evidence_spans:
        chunks = []
        chars_remaining = max_chars
        for span in evidence_spans:
            if chars_remaining <= 0:
                break
            try:
                source = store.get_source(span.source_id)
                if source and source.full_text:
                    start = max(0, span.start_offset)
                    end = min(len(source.full_text), span.end_offset)
                    chunk = source.full_text[start:end].strip()
                    if chunk:
                        chunks.append(chunk)
                        chars_remaining -= len(chunk)
            except Exception:
                continue
        if chunks:
            return " ... ".join(chunks)[:max_chars]

    # Fallback: keyword window from raw source
    if body_hint and source_ids:
        import re

        hint_words = [w for w in re.findall(r"\b\w{4,}\b", body_hint.lower()) if len(w) > 3]
        if not hint_words:
            return None
        for sid in source_ids[:1]:
            try:
                source = store.get_source(sid)
                if not source or not source.full_text:
                    continue
                text_lower = source.full_text.lower()
                best_pos = -1
                for word in hint_words:
                    pos = text_lower.find(word)
                    if pos >= 0:
                        best_pos = pos
                        break
                if best_pos >= 0:
                    window = max_chars // 2
                    start = max(0, best_pos - window)
                    end = min(len(source.full_text), best_pos + window)
                    return source.full_text[start:end].strip()
            except Exception:
                continue

    return None


def _format_raw_item(hit: Dict[str, Any]) -> str:
    """Render a raw ChromaDB hit as a bullet point."""
    text = hit.get("text", "").strip()
    source = hit.get("source_file", "?")
    wing = hit.get("wing", "")
    room = hit.get("room", "")
    location = f"{wing}/{room}" if wing and room else source
    if text:
        preview = text if len(text) <= 400 else text[:397] + "..."
        return f"[{location}] {preview}"
    return f"[{location}] (empty)"


# ── EvidencePacker ────────────────────────────────────────────────────────────


class EvidencePacker:
    """
    Organizes raw retrieval results into a role-structured packed context.

    During Phase 1 the input is raw ChromaDB hits. If MemoryObjects exist for
    those hits (linked by source_id), they annotate the role sections.
    Untyped hits go into RAW CONTEXT — nothing is dropped.
    """

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        embedding_model=None,
    ):
        self.memory_store = memory_store
        self._embedding_model = embedding_model

    def pack(
        self,
        query: str,
        retrieved_results: Dict[str, Any],
        token_budget: int = 6000,
    ) -> PackedContext:
        """
        Pack retrieved results into a role-organized context block.

        Parameters
        ----------
        query:
            The user query (used for relevance scoring display).
        retrieved_results:
            Dict from searcher.search_memories(): {"query": ..., "results": [...]}
        token_budget:
            Maximum tokens for the packed output.

        Returns
        -------
        PackedContext with the assembled text, section breakdown, and metadata.
        """
        hits = retrieved_results.get("results", [])
        if not hits:
            return PackedContext(
                query=query,
                text="(no results)",
                token_count=0,
                memory_ids=[],
                sections=[],
                untyped_count=0,
            )

        # Step 1: Match hits to MemoryObjects (if store is available)
        typed_sections: Dict[MemoryType, PackedSection] = {
            role: PackedSection(role=role, header=ROLE_HEADERS[role]) for role in ROLE_ORDER
        }
        raw_section = PackedSection(role=None, header="RAW CONTEXT")

        seen_hashes: set = set()
        all_memory_ids: List[str] = []
        untyped_count = 0

        # Score each hit: prefer higher similarity, then typed memories
        for hit in hits:
            text = hit.get("text", "").strip()
            if not text:
                continue

            h = _text_hash(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            similarity = hit.get("similarity")

            # Try to match to a MemoryObject
            mo = self._find_memory_for_hit(hit)
            if mo is not None:
                item = _format_memory_item(mo, similarity)
                typed_sections[mo.memory_type].items.append(item)
                typed_sections[mo.memory_type].memory_ids.append(mo.memory_object_id)
                all_memory_ids.append(mo.memory_object_id)
            else:
                raw_section.items.append(_format_raw_item(hit))
                untyped_count += 1

        # Step 2: Assemble sections within budget
        header_text = f"MEMORY CONTEXT — {query}\n{'=' * 60}\n\n"
        budget_remaining = token_budget - _token_count(header_text)

        assembled_sections: List[PackedSection] = []

        for role in ROLE_ORDER:
            sec = typed_sections[role]
            if not sec.items:
                continue
            sec_text = sec.render()
            cost = _token_count(sec_text) + 2  # +2 for separating newlines
            if cost <= budget_remaining:
                assembled_sections.append(sec)
                budget_remaining -= cost

        # Raw context last
        if raw_section.items:
            # Trim raw context to fit remaining budget
            raw_section.items = _trim_to_budget(raw_section.items, budget_remaining - 20)
            if raw_section.items:
                assembled_sections.append(raw_section)

        # Step 3: Render final text
        if not assembled_sections:
            final_text = header_text + "(all results exceeded token budget)"
        else:
            parts = [header_text]
            for sec in assembled_sections:
                rendered = sec.render()
                if rendered:
                    parts.append(rendered)
                    parts.append("")  # blank line between sections
            final_text = "\n".join(parts)

        return PackedContext(
            query=query,
            text=final_text,
            token_count=_token_count(final_text),
            memory_ids=all_memory_ids,
            sections=assembled_sections,
            untyped_count=untyped_count,
        )

    def _find_memory_for_hit(self, hit: Dict[str, Any]) -> Optional[MemoryObject]:
        """
        Try to find a MemoryObject linked to a raw ChromaDB hit.

        Matching strategy: look up memories whose source_ids include the
        hit's source_file (best-effort; Phase 1 link is imprecise).
        """
        if self.memory_store is None:
            return None

        # Phase 1: no precise source_id link between ChromaDB drawers and
        # MemoryObjects yet. Return None and let everything fall into RAW CONTEXT.
        # Phase 2+ will wire this properly via source_id metadata.
        return None

    def pack_memories_direct(
        self,
        query: str,
        memories: List[MemoryObject],
        token_budget: int = 6000,
        query_vec: Optional[List[float]] = None,
        store: Optional[MemoryStore] = None,
        selector_scores: Optional[Dict[int, float]] = None,
        packer_weights: Optional[Tuple[float, float, float]] = None,
    ) -> PackedContext:
        """
        Pack MemoryObjects directly (used when memories are retrieved via PSA path).

        When query_vec is provided, memories are ranked by query relevance
        (cosine similarity) instead of static quality_score. When store is
        provided, source records are fetched to enrich top-ranked items.
        """
        if not memories:
            return PackedContext(
                query=query,
                text="(no PSA memories found)",
                token_count=0,
                memory_ids=[],
                sections=[],
                untyped_count=0,
            )

        # Compute per-memory relevance to the query
        relevances = _compute_relevance(memories, query_vec)

        # Combine relevance with quality and optional selector scores for ranking
        if selector_scores and packer_weights:
            w_sel, w_cos, w_qual = packer_weights
            scored = sorted(
                zip(memories, relevances),
                key=lambda pair: (
                    w_sel * selector_scores.get(pair[0].primary_anchor_id, 0.0)
                    + w_cos * pair[1]
                    + w_qual * pair[0].quality_score
                ),
                reverse=True,
            )
        else:
            scored = sorted(
                zip(memories, relevances),
                key=lambda pair: pair[1] * 0.7 + pair[0].quality_score * 0.3,
                reverse=True,
            )

        # Top N memories get source context and full body text
        TOP_N_WITH_SOURCE = 10

        typed_sections: Dict[MemoryType, PackedSection] = {
            role: PackedSection(role=role, header=ROLE_HEADERS[role]) for role in ROLE_ORDER
        }

        seen_ids: set = set()
        all_memory_ids: List[str] = []

        for rank, (mo, relevance) in enumerate(scored):
            if mo.memory_object_id in seen_ids:
                continue
            seen_ids.add(mo.memory_object_id)

            # Top-ranked memories get full body text and source provenance
            is_top = rank < TOP_N_WITH_SOURCE
            source_ctx = None
            evidence_text = None
            if is_top and store:
                source_file = _fetch_source_path(store, mo.source_ids)
                if source_file:
                    source_ctx = f"[from {source_file}]"
                evidence_text = _fetch_evidence_text(
                    store,
                    mo.evidence_spans,
                    max_chars=500,
                    body_hint=mo.body,
                    source_ids=mo.source_ids,
                )

            item = _format_memory_item(
                mo,
                similarity=relevance if relevance > 0 else None,
                source_context=source_ctx,
                max_body_chars=800 if is_top else 300,
                evidence_text=evidence_text,
            )
            role = mo.memory_type
            if role not in typed_sections:
                role = MemoryType.SEMANTIC
            typed_sections[role].items.append(item)
            typed_sections[role].memory_ids.append(mo.memory_object_id)
            all_memory_ids.append(mo.memory_object_id)

        header_text = f"PSA MEMORY CONTEXT — {query}\n{'=' * 60}\n\n"
        budget_remaining = token_budget - _token_count(header_text)

        assembled_sections: List[PackedSection] = []
        for role in ROLE_ORDER:
            sec = typed_sections[role]
            if not sec.items:
                continue
            cost = _token_count(sec.render()) + 2
            if cost <= budget_remaining:
                assembled_sections.append(sec)
                budget_remaining -= cost
            else:
                # Incrementally pack items that fit instead of dropping the whole section
                partial = PackedSection(role=sec.role, header=sec.header, items=[], memory_ids=[])
                header_cost = _token_count(sec.header) + 2
                if header_cost < budget_remaining:
                    partial_budget = budget_remaining - header_cost
                    for item, mid in zip(sec.items, sec.memory_ids):
                        item_cost = _token_count(f"- {item}\n")
                        if item_cost > partial_budget:
                            break
                        partial.items.append(item)
                        partial.memory_ids.append(mid)
                        partial_budget -= item_cost
                    if partial.items:
                        assembled_sections.append(partial)
                        budget_remaining -= _token_count(partial.render()) + 2

        parts = [header_text]
        for sec in assembled_sections:
            rendered = sec.render()
            if rendered:
                parts.append(rendered)
                parts.append("")
        final_text = "\n".join(parts)

        # Only include memory IDs from sections that actually made it into the budget
        packed_memory_ids = []
        for sec in assembled_sections:
            packed_memory_ids.extend(sec.memory_ids)

        return PackedContext(
            query=query,
            text=final_text,
            token_count=_token_count(final_text),
            memory_ids=packed_memory_ids,
            sections=assembled_sections,
            untyped_count=0,
        )


def _trim_to_budget(items: List[str], budget_tokens: int) -> List[str]:
    """Trim item list to fit within a token budget."""
    result = []
    remaining = budget_tokens
    for item in items:
        cost = _token_count(f"- {item}\n")
        if cost > remaining:
            break
        result.append(item)
        remaining -= cost
    return result
