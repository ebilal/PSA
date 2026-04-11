"""test_synthesizer.py — Tests for AnchorSynthesizer."""

from unittest.mock import MagicMock, patch

import pytest

from psa.memory_object import MemoryObject, MemoryType
from psa.synthesizer import AnchorSynthesizer


def _make_memory(title: str, body: str, quality: float = 0.8) -> MemoryObject:
    mo = MagicMock(spec=MemoryObject)
    mo.title = title
    mo.body = body
    mo.summary = f"Summary of {title}."
    mo.memory_type = MemoryType.SEMANTIC
    mo.quality_score = quality
    mo.embedding = [0.1] * 768
    mo.primary_anchor_id = 1
    return mo


def test_synthesizer_returns_string():
    with patch("psa.synthesizer.call_llm", return_value="We decided to use JWT for auth."):
        s = AnchorSynthesizer()
        result = s.synthesize(
            "What auth approach did we use?", [_make_memory("JWT decision", "We use JWT.")]
        )
    assert isinstance(result, str)
    assert len(result) > 0


def test_synthesizer_returns_llm_output():
    expected = "The team decided on JWT tokens with 24-hour expiry."
    with patch("psa.synthesizer.call_llm", return_value=expected):
        s = AnchorSynthesizer()
        result = s.synthesize("auth decision", [_make_memory("JWT", "JWT tokens.")])
    assert result == expected


def test_synthesizer_empty_memories_returns_fallback():
    s = AnchorSynthesizer()
    result = s.synthesize("some query", [])
    assert isinstance(result, str)
    assert len(result) > 0


def test_synthesizer_trims_by_cosine_to_query():
    """When max_memories=1, only the highest-cosine memory is sent to LLM."""
    query_vec = [1.0] + [0.0] * 767

    high_sim = MagicMock(spec=MemoryObject)
    high_sim.embedding = [1.0] + [0.0] * 767
    high_sim.title = "High relevance"
    high_sim.body = "Very relevant content."
    high_sim.summary = "High sim summary."
    high_sim.memory_type = MemoryType.SEMANTIC
    high_sim.quality_score = 0.5

    low_sim = MagicMock(spec=MemoryObject)
    low_sim.embedding = [0.0] * 767 + [1.0]
    low_sim.title = "Low relevance"
    low_sim.body = "Unrelated content."
    low_sim.summary = "Low sim summary."
    low_sim.memory_type = MemoryType.SEMANTIC
    low_sim.quality_score = 0.9

    captured_prompt = []

    def capture_llm(messages, **kwargs):
        captured_prompt.append(messages[0]["content"])
        return "synthesis result"

    with patch("psa.synthesizer.call_llm", side_effect=capture_llm):
        s = AnchorSynthesizer()
        s.synthesize("query", [high_sim, low_sim], query_vec=query_vec, max_memories=1)

    assert "High relevance" in captured_prompt[0]
    assert "Low relevance" not in captured_prompt[0]


def test_synthesizer_raises_on_llm_error_so_pipeline_can_catch():
    """Synthesizer propagates LLM errors — caller is responsible for fallback."""
    with patch("psa.synthesizer.call_llm", side_effect=RuntimeError("LLM timeout")):
        s = AnchorSynthesizer()
        with pytest.raises(RuntimeError):
            s.synthesize("query", [_make_memory("m", "b")])
