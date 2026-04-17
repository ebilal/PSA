"""
extractor_llm.py — LLMExtractor: Qwen-based query pattern distillation (stub).

Architectural commitment, not implemented. The interface is settled; the body
is reserved for follow-up work once the MVP heuristic backend has been
validated on real tenants.

Intended behavior (for the future implementation):
    One Qwen call per anchor. The pool (oracle queries + endorsed fingerprints)
    is rendered into the prompt as a numbered list. Qwen returns up to `n`
    representative question patterns that this anchor can answer.

Prompt contract (tentative — subject to refinement when implemented):

    "You are curating an anchor description in a semantic memory index.
    Below are N real user queries that this anchor has answered and been
    endorsed for. Produce up to {n} representative question patterns that
    characterize what this anchor can answer. Prefer coverage over
    paraphrase. Return one pattern per line, no numbering, no commentary."

Why a stub:
    Branch 3 ships the pluggable architecture (Protocol in __init__.py,
    heuristic backend). Committing to the LLM interface now means a future
    branch can drop in the implementation without redesigning the contract.
"""

from __future__ import annotations


class LLMExtractor:
    """Qwen-based query pattern distillation — not yet implemented.

    Conforms to the QueryPatternExtractor Protocol by method signature; the
    body raises NotImplementedError with a pointer to this file.
    """

    def extract(self, pool: list[str], n: int) -> list[str]:
        raise NotImplementedError(
            "LLM extractor reserved for follow-up; see psa/curation/extractor_llm.py "
            "for the interface contract."
        )
