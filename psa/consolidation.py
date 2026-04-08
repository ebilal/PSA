"""
consolidation.py — Raw text → typed memory objects via Qwen2.5-7B-Instruct.

The ConsolidationPipeline converts raw source records into typed MemoryObjects:
  1. Persist raw source record (immutable)
  2. Structure-aware chunking (fine / mid / section)
  3. Offline consolidation via Qwen2.5-7B-Instruct
  4. Retention filtering (score >= 0.65 + value criteria)
  5. Deduplication (merge instead of write if cosine > 0.92)
  6. Embed + persist

This module does NOT delete or modify existing ChromaDB raw storage.
PSA objects are additive derived indexes on top of raw traces.

Model requirement: Qwen2.5-7B-Instruct accessible via ollama or
a local OpenAI-compatible endpoint at QWEN_ENDPOINT (default:
http://localhost:11434/v1/chat/completions).
"""

import json
import logging
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .memory_object import EvidenceSpan, MemoryObject, MemoryStore, MemoryType, RawSource

logger = logging.getLogger("psa.consolidation")

# ── Constants ────────────────────────────────────────────────────────────────

QWEN_ENDPOINT = os.environ.get(
    "QWEN_ENDPOINT", "http://localhost:11434/v1/chat/completions"
)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen2.5:7b")

RETENTION_THRESHOLD = 0.65  # minimum retention score to keep a memory object

# Token approximation: 4 chars ≈ 1 token
CHARS_PER_TOKEN = 4

# Chunk target sizes in tokens
FINE_TOKENS = (80, 180)
MID_TOKENS = (220, 450)
SECTION_TOKENS = (500, 1500)


# ── Chunk dataclass ───────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """A hierarchical text chunk with provenance."""

    chunk_id: str
    source_id: str
    level: str  # "fine" | "mid" | "section"
    text: str
    start_offset: int
    end_offset: int
    parent_chunk_id: Optional[str] = None
    parse_quality: str = "ok"  # "ok" | "low"
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return len(self.text) // CHARS_PER_TOKEN


# ── Structure-aware chunking ──────────────────────────────────────────────────


def _make_chunk_id(source_id: str, level: str, index: int) -> str:
    return f"{source_id[:8]}_{level}_{index:04d}"


def chunk_hierarchical(
    source: RawSource,
    source_type: str = "prose",
) -> List[Chunk]:
    """
    Produce fine, mid, and section chunks from a raw source.

    source_type: "code" | "conversation" | "prose"
    """
    text = source.full_text
    if not text.strip():
        return []

    sections = _split_sections(text, source_type)
    chunks: List[Chunk] = []
    chunk_idx = 0

    for sec_text, sec_start in sections:
        sec_end = sec_start + len(sec_text)
        sec_id = _make_chunk_id(source.source_id, "section", chunk_idx)
        sec_chunk = Chunk(
            chunk_id=sec_id,
            source_id=source.source_id,
            level="section",
            text=sec_text,
            start_offset=sec_start,
            end_offset=sec_end,
        )
        chunks.append(sec_chunk)
        chunk_idx += 1

        # Split section into mid chunks
        mids = _split_by_token_target(sec_text, sec_start, MID_TOKENS, source_type)
        for mid_text, mid_start in mids:
            mid_end = mid_start + len(mid_text)
            mid_id = _make_chunk_id(source.source_id, "mid", chunk_idx)
            mid_chunk = Chunk(
                chunk_id=mid_id,
                source_id=source.source_id,
                level="mid",
                text=mid_text,
                start_offset=mid_start,
                end_offset=mid_end,
                parent_chunk_id=sec_id,
            )
            chunks.append(mid_chunk)
            chunk_idx += 1

            # Split mid into fine chunks
            fines = _split_by_token_target(mid_text, mid_start, FINE_TOKENS, source_type)
            for fine_text, fine_start in fines:
                fine_end = fine_start + len(fine_text)
                fine_id = _make_chunk_id(source.source_id, "fine", chunk_idx)
                chunks.append(
                    Chunk(
                        chunk_id=fine_id,
                        source_id=source.source_id,
                        level="fine",
                        text=fine_text,
                        start_offset=fine_start,
                        end_offset=fine_end,
                        parent_chunk_id=mid_id,
                    )
                )
                chunk_idx += 1

    return chunks


def _split_sections(text: str, source_type: str) -> List[Tuple[str, int]]:
    """Split text into section-level chunks. Returns (text, start_offset) pairs."""
    if source_type == "code":
        return _split_by_boundaries(text, patterns=[r"\n(?=class |def |async def )"])
    if source_type == "conversation":
        return _split_by_boundaries(text, patterns=[r"\n(?=Human:|User:|>)"])
    # prose: split by heading or double-newline
    return _split_by_boundaries(text, patterns=[r"\n#{1,3} ", r"\n\n"])


def _split_by_boundaries(text: str, patterns: List[str]) -> List[Tuple[str, int]]:
    """Split text at regex boundaries, returning (segment, offset) pairs."""
    combined = "|".join(f"(?:{p})" for p in patterns)
    positions = [0]
    for m in re.finditer(combined, text):
        positions.append(m.start())
    positions.append(len(text))

    result = []
    for i in range(len(positions) - 1):
        seg = text[positions[i] : positions[i + 1]]
        if seg.strip():
            result.append((seg, positions[i]))
    return result if result else [(text, 0)]


def _split_by_token_target(
    text: str,
    base_offset: int,
    token_range: Tuple[int, int],
    source_type: str,
) -> List[Tuple[str, int]]:
    """Split text into chunks targeting the given token range."""
    target_chars = token_range[1] * CHARS_PER_TOKEN
    overlap_chars = 40  # small overlap to preserve context across boundaries

    if len(text) <= target_chars:
        return [(text, base_offset)]

    result = []
    start = 0
    while start < len(text):
        end = min(start + target_chars, len(text))
        # Try to break at a sentence boundary
        if end < len(text):
            break_pos = text.rfind(". ", start, end)
            if break_pos > start + target_chars // 2:
                end = break_pos + 2
        seg = text[start:end]
        if seg.strip():
            result.append((seg, base_offset + start))
        start = max(start + 1, end - overlap_chars)

    return result if result else [(text, base_offset)]


# ── Qwen consolidation ────────────────────────────────────────────────────────


_CONSOLIDATION_SYSTEM = """\
You are a memory consolidation assistant. Your job is to extract reusable,
typed memory objects from raw text. Be concise and precise.

Memory types:
- episodic: a specific event or episode that happened
- semantic: a stable fact, concept, or definition
- procedural: a reusable how-to, pattern, or strategy
- failure: a mistake, error, or cautionary lesson
- tool_use: specific usage of a tool, command, API, or system
- working_derivative: a derived synthesis from multiple sources

Output a JSON object with a "memories" array. Each memory has:
  type: one of the memory types above
  title: concise noun phrase (max 80 chars)
  body: the full memory text (1-4 sentences, verbatim where important)
  summary: one-sentence synopsis (max 120 chars)
  classification_reason: why this memory type was chosen (1 sentence)
  evidence_chunk_ids: array of chunk IDs that support this memory
  retention_score: float in [0,1] — how reusable and durable this memory is
  uncertainty: "low" | "medium" | "high" — how confident you are

Retention score guidelines:
  >= 0.80: reusable strategy, costly failure, durable fact, recurring pattern
  0.65–0.79: useful but context-dependent
  < 0.65: transient, speculative, or low-value (do not include these)

Do NOT include:
- Transient scratchpad chatter or internal reasoning with no reusable value
- Unsupported speculation
- Repetitive low-value traces
- Anything with retention_score < 0.65

If there is nothing worth persisting, return {"memories": []}.
"""

_CONSOLIDATION_USER_TEMPLATE = """\
Source type: {source_type}
Source path: {source_path}

=== SECTION CHUNKS ===
{section_chunks}

=== FINE CHUNKS (for evidence references) ===
{fine_chunks}

Extract memory objects from the above source. Use chunk IDs (like "{first_chunk_id}") \
in evidence_chunk_ids.
"""

_REPAIR_SUFFIX = (
    "\n\nThe previous response was not valid JSON. "
    "Return ONLY a JSON object with a 'memories' array. No prose, no markdown."
)


def _call_qwen(messages: List[dict], timeout: int = 60) -> str:
    """Call the Qwen2.5-7B-Instruct endpoint. Returns the raw response text."""
    try:
        import urllib.request

        payload = json.dumps(
            {
                "model": QWEN_MODEL,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2048,
                "response_format": {"type": "json_object"},
            }
        ).encode()

        req = urllib.request.Request(
            QWEN_ENDPOINT,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Qwen endpoint call failed: {e}") from e


def _parse_qwen_output(raw: str) -> List[dict]:
    """Parse Qwen JSON output into a list of raw memory dicts."""
    try:
        parsed = json.loads(raw)
        return parsed.get("memories", [])
    except json.JSONDecodeError:
        return []


def _consolidate_with_qwen(
    source: RawSource,
    chunks: List[Chunk],
    max_section_chars: int = 4000,
) -> List[dict]:
    """
    Run Qwen consolidation on a source record.

    Returns a list of raw memory dicts (not yet MemoryObjects).
    Retries once with a repair prompt if the first response is invalid JSON.
    """
    sections = [c for c in chunks if c.level == "section"]
    fines = [c for c in chunks if c.level == "fine"]

    section_text = "\n---\n".join(
        f"[{c.chunk_id}] {c.text[:max_section_chars]}" for c in sections[:10]
    )
    fine_text = "\n".join(
        f"[{c.chunk_id}] {c.text[:300]}" for c in fines[:30]
    )
    first_chunk_id = fines[0].chunk_id if fines else (sections[0].chunk_id if sections else "?")

    user_content = _CONSOLIDATION_USER_TEMPLATE.format(
        source_type=source.source_type,
        source_path=source.source_path or "(no path)",
        section_chunks=section_text or "(none)",
        fine_chunks=fine_text or "(none)",
        first_chunk_id=first_chunk_id,
    )

    messages = [
        {"role": "system", "content": _CONSOLIDATION_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    raw = _call_qwen(messages)
    memories = _parse_qwen_output(raw)

    if not memories and raw.strip():
        # Retry with repair prompt
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": _REPAIR_SUFFIX})
        raw2 = _call_qwen(messages)
        memories = _parse_qwen_output(raw2)

    return memories


# ── Retention filtering ───────────────────────────────────────────────────────


def _passes_retention(raw_memory: dict) -> bool:
    """Apply retention filter: score >= 0.65 and at least one value criterion."""
    score = float(raw_memory.get("retention_score", 0))
    if score < RETENTION_THRESHOLD:
        return False

    body = (raw_memory.get("body", "") + " " + raw_memory.get("title", "")).lower()
    mtype = raw_memory.get("type", "")

    # Always keep procedural, failure, tool_use if score threshold is met
    if mtype in ("procedural", "failure", "tool_use"):
        return True

    # For episodic/semantic: require non-trivial content
    if len(body.strip()) < 30:
        return False

    return True


def _raw_to_memory_object(
    raw: dict,
    source: RawSource,
    tenant_id: str,
) -> Optional[MemoryObject]:
    """Convert a raw memory dict from Qwen into a MemoryObject."""
    try:
        mtype_str = raw.get("type", "semantic").lower().replace("-", "_")
        mtype = MemoryType(mtype_str)
    except ValueError:
        mtype = MemoryType.SEMANTIC

    evidence_chunk_ids = raw.get("evidence_chunk_ids") or []
    if not isinstance(evidence_chunk_ids, list):
        evidence_chunk_ids = []

    return MemoryObject.create(
        tenant_id=tenant_id,
        memory_type=mtype,
        title=str(raw.get("title", "Untitled"))[:200],
        body=str(raw.get("body", "")),
        summary=str(raw.get("summary", ""))[:300],
        source_ids=[source.source_id],
        classification_reason=str(raw.get("classification_reason", "")),
        evidence_chunk_ids=evidence_chunk_ids,
        evidence_spans=[
            EvidenceSpan(
                source_id=source.source_id,
                start_offset=0,
                end_offset=len(source.full_text),
            )
        ],
        quality_score=float(raw.get("retention_score", 0)),
    )


# ── Deduplication ─────────────────────────────────────────────────────────────


def _dedup_against_store(
    mo: MemoryObject,
    store: MemoryStore,
    embedding_model,
    similarity_threshold: float = 0.92,
) -> Tuple[bool, Optional[str]]:
    """
    Check if a near-duplicate already exists in the store.

    Uses embedding similarity search to check against ALL memories of the same
    type, not just the most recent ones.

    Returns (is_duplicate, existing_id_or_None).
    """
    if mo.embedding is None:
        return False, None

    from .embeddings import EmbeddingModel

    # Check all memories of same type with embeddings
    existing = store.query_by_type(
        mo.tenant_id, mo.memory_type, limit=10_000, exclude_duplicates=True
    )
    for ex in existing:
        if ex.embedding is None:
            continue
        sim = EmbeddingModel.cosine_similarity(mo.embedding, ex.embedding)
        if sim >= similarity_threshold:
            return True, ex.memory_object_id

    return False, None


# ── Main pipeline ─────────────────────────────────────────────────────────────


class ConsolidationPipeline:
    """
    Converts raw source records into typed MemoryObjects.

    Usage::

        store = MemoryStore(db_path="~/.psa/tenants/default/memory.sqlite3")
        pipeline = ConsolidationPipeline(store, tenant_id="default")
        memories = pipeline.consolidate(
            raw_text="...",
            source_type="project_file",
            source_path="/project/auth.py",
        )
    """

    def __init__(
        self,
        store: MemoryStore,
        tenant_id: str = "default",
        embedding_model=None,
        use_llm: bool = True,
    ):
        self.store = store
        self.tenant_id = tenant_id
        self._embedding_model = embedding_model
        self.use_llm = use_llm

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from .embeddings import EmbeddingModel

            self._embedding_model = EmbeddingModel()
        return self._embedding_model

    def consolidate(
        self,
        raw_text: str,
        source_type: str,
        source_path: Optional[str] = None,
        title: str = "",
        metadata: Optional[dict] = None,
    ) -> List[MemoryObject]:
        """
        Full consolidation pipeline for a single raw source.

        Steps: ingest → chunk → consolidate → filter → dedup → embed → persist.
        Returns the list of persisted MemoryObjects (may be empty).
        """
        if not raw_text.strip():
            return []

        # Step 1: Persist raw source (immutable)
        source = RawSource.create(
            tenant_id=self.tenant_id,
            source_type=source_type,
            full_text=raw_text,
            title=title,
            source_path=source_path,
            metadata=metadata or {},
        )
        self.store.add_source(source)

        # Step 2: Hierarchical chunking
        chunk_source_type = _infer_chunk_type(source_type, source_path)
        chunks = chunk_hierarchical(source, source_type=chunk_source_type)

        if not chunks:
            logger.debug("No chunks produced for source %s", source.source_id)
            return []

        # Step 3: Offline consolidation
        if self.use_llm:
            try:
                raw_memories = _consolidate_with_qwen(source, chunks)
            except RuntimeError as e:
                logger.warning("Qwen consolidation failed for %s: %s", source.source_id, e)
                raw_memories = []
        else:
            raw_memories = []

        if not raw_memories:
            logger.debug("No memories extracted from source %s", source.source_id)
            return []

        # Step 4: Retention filtering
        retained = [r for r in raw_memories if _passes_retention(r)]
        logger.debug(
            "Retention: %d/%d memories kept from source %s",
            len(retained),
            len(raw_memories),
            source.source_id,
        )

        # Step 5-6: Convert, embed, dedup, persist
        embedding_model = self._get_embedding_model()
        persisted: List[MemoryObject] = []

        for raw in retained:
            mo = _raw_to_memory_object(raw, source, self.tenant_id)
            if mo is None:
                continue

            # Embed
            embed_text = f"{mo.memory_type.value}: {mo.title}\n{mo.summary}\n{mo.body}"
            try:
                mo.embedding = embedding_model.embed(embed_text)
            except Exception as e:
                logger.warning(
                    "Embedding failed for memory %s: %s — skipping (would be un-searchable)",
                    mo.memory_object_id, e,
                )
                continue

            # Dedup
            is_dup, existing_id = _dedup_against_store(mo, self.store, embedding_model)
            if is_dup:
                # Mark as duplicate, still persist for audit trail
                mo.is_duplicate = True
                mo.duplicate_of = existing_id
                logger.debug(
                    "Memory %s is a near-duplicate of %s", mo.memory_object_id, existing_id
                )

            self.store.add(mo)
            if not mo.is_duplicate:
                persisted.append(mo)

        return persisted

    def consolidate_batch(
        self,
        sources: List[dict],
    ) -> List[MemoryObject]:
        """
        Consolidate multiple sources.

        Each source dict: {raw_text, source_type, source_path?, title?, metadata?}
        """
        all_memories = []
        for src in sources:
            memories = self.consolidate(**src)
            all_memories.extend(memories)
        return all_memories


def _infer_chunk_type(source_type: str, source_path: Optional[str]) -> str:
    """Map source_type + file extension to chunking strategy."""
    if source_type == "conversation":
        return "conversation"
    if source_path:
        ext = os.path.splitext(source_path)[1].lower()
        if ext in (".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb"):
            return "code"
    return "prose"
