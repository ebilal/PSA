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
from typing import Any, Dict, List, Optional

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


def _format_memory_item(mo: MemoryObject, similarity: Optional[float] = None) -> str:
    """Render a MemoryObject as a bullet point."""
    parts = [mo.title]
    if mo.body and mo.body != mo.title:
        # Truncate long bodies
        body = mo.body if len(mo.body) <= 300 else mo.body[:297] + "..."
        parts.append(body)
    if mo.memory_type == MemoryType.EPISODIC and mo.success_label is not None:
        parts.append(f"outcome: {'success' if mo.success_label else 'failure'}")
    if similarity is not None:
        parts.append(f"(relevance: {similarity:.2f})")
    return " — ".join(parts)


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
    ) -> PackedContext:
        """
        Pack MemoryObjects directly (used when memories are retrieved via PSA path).

        This is the Phase 3+ path. In Phase 1 this is called when
        psa_mode == "side-by-side" to show what PSA would produce.
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

        typed_sections: Dict[MemoryType, PackedSection] = {
            role: PackedSection(role=role, header=ROLE_HEADERS[role]) for role in ROLE_ORDER
        }

        seen_ids: set = set()
        all_memory_ids: List[str] = []

        # Score by memory quality
        scored = sorted(memories, key=lambda m: m.quality_score, reverse=True)

        for mo in scored:
            if mo.memory_object_id in seen_ids:
                continue
            seen_ids.add(mo.memory_object_id)
            item = _format_memory_item(mo)
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

        return PackedContext(
            query=query,
            text=final_text,
            token_count=_token_count(final_text),
            memory_ids=all_memory_ids,
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
