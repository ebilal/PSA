"""
synthesizer.py — AnchorSynthesizer: single LLM call over selected anchor memories.

Replaces the packer's weighted scoring formula. Instead of ranking and
dumping memory bullets, produces one coherent narrative paragraph conditioned
on the query.

Usage:
    synthesizer = AnchorSynthesizer()  # instantiate once in PSAPipeline.__init__()
    text = synthesizer.synthesize(query, memories, query_vec=query_vec)
"""

import logging
from typing import List, Optional

from .embeddings import EmbeddingModel
from .llm import call_llm
from .memory_object import MemoryObject

logger = logging.getLogger("psa.synthesizer")

MAX_MEMORIES_DEFAULT = 30
TOKEN_BUDGET_DEFAULT = 700
CHARS_PER_TOKEN = 4


class AnchorSynthesizer:
    """
    Synthesizes a query-conditioned narrative from selected anchor memories.

    Instantiated once in PSAPipeline.__init__(). Uses call_llm() which handles
    cloud-first / Ollama fallback — connection pooling is managed by call_llm().
    """

    def synthesize(
        self,
        query: str,
        memories: List[MemoryObject],
        query_vec: Optional[List[float]] = None,
        token_budget: int = TOKEN_BUDGET_DEFAULT,
        max_memories: int = MAX_MEMORIES_DEFAULT,
    ) -> str:
        """
        Synthesize a single query-conditioned narrative from anchor memories.

        Parameters
        ----------
        query:
            The user's query — synthesis is conditioned on this.
        memories:
            All MemoryObjects from selected anchors (deduplicated by pipeline).
        query_vec:
            L2-normalized query embedding for cosine-based trimming.
            When provided, lowest-cosine memories are dropped first.
            When absent, memories are used in the order provided.
        token_budget:
            Target output length in tokens (~700 = 5-8 sentence paragraph).
        max_memories:
            Hard cap on memories passed to the LLM to fit context limits.

        Returns
        -------
        Coherent prose paragraph. Raises on LLM failure — caller must catch
        and fall back to packer if needed.
        """
        if not memories:
            return "(no relevant memories found for this query)"

        # Trim by cosine similarity to query (lowest relevance dropped first)
        if query_vec is not None:
            ranked = sorted(
                memories,
                key=lambda m: (
                    EmbeddingModel.cosine_similarity(m.embedding, query_vec) if m.embedding else 0.0
                ),
                reverse=True,
            )
        else:
            ranked = list(memories)

        trimmed = ranked[:max_memories]

        memory_lines = []
        for mo in trimmed:
            line = f"[{mo.memory_type.value.upper()}] {mo.title}"
            if mo.summary:
                line += f": {mo.summary}"
            if mo.body and mo.body != mo.summary:
                line += f"\n{mo.body[:500]}"
            memory_lines.append(line)

        memory_text = "\n".join(memory_lines)
        # Cap input to avoid overrunning LLM context.
        # Reserve ~2x the output budget for prompt overhead; remaining space for memories.
        max_input_tokens = max(token_budget * 4, 2000)  # generous input budget
        max_input_chars = max_input_tokens * CHARS_PER_TOKEN
        if len(memory_text) > max_input_chars:
            # Trim whole lines rather than mid-sentence
            lines = memory_text.split("\n")
            trimmed_lines = []
            chars = 0
            for line in lines:
                if chars + len(line) + 1 > max_input_chars:
                    break
                trimmed_lines.append(line)
                chars += len(line) + 1
            memory_text = "\n".join(trimmed_lines)

        prompt = (
            f"You are synthesizing memory context for an AI assistant.\n\n"
            f"Query: {query}\n\n"
            f"Relevant memories from personal history:\n{memory_text}\n\n"
            f"Write a focused, coherent paragraph (5-8 sentences) presenting what's most "
            f"relevant to help answer the query. Weave related facts into a narrative where "
            f"possible. Be specific and factual. Do not add information not present in the memories."
        )

        return call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=min(token_budget, 800),
            json_mode=False,
        )
