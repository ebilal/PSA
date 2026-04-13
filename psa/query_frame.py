"""
query_frame.py — Query frame extraction for PSA.

Analyzes a query string and produces a structured QueryFrame describing:
- answer_target: what type of answer is expected
- entities: named entities mentioned
- time_constraint: temporal scope if any
- speaker_role_constraint: speaker filter if self-referential
- entity_constraint: named person constraint
- retrieval_mode: how to retrieve memories
- confidence: how confident we are in the frame

Uses a regex-based pattern matcher as the primary strategy, with an optional
LLM fallback when confidence is low.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("psa.query_frame")

# ── Common words to filter out of entity extraction ─────────────────────────

_STOP_WORDS = {
    "How", "What", "Why", "When", "Where", "Who", "Can", "Does", "Did",
    "The", "This", "That", "There", "These", "Those", "Our", "We", "Us",
    "They", "Them", "He", "She", "It", "Is", "Are", "Was", "Were", "Has",
    "Have", "Had", "Do", "Be", "Been", "Being", "Will", "Would", "Could",
    "Should", "May", "Might", "Shall", "Must", "Need",
}

# ── Pattern definitions ──────────────────────────────────────────────────────

# PROCEDURE patterns
_PROCEDURE_PATTERNS = [
    r"\bhow\s+do\s+I\b",
    r"\bhow\s+to\b",
    r"\bsteps\s+to\b",
    r"\bset\s+up\b",
    r"\bconfigure\b",
    r"\binstall\b",
]

# FAILURE patterns
_FAILURE_PATTERNS = [
    r"\bwhat\s+went\s+wrong\b",
    r"\bwhy\s+did\s+it\s+fail\b",
    r"\bbroke\b",
    r"\bbug\b",
    r"\berror\b",
    r"\bfailed\b",
    r"\bfailure\b",
]

# TEMPORAL patterns
_TEMPORAL_PATTERNS = [
    r"\bchanged\b",
    r"\bused\s+to\b",
    r"\bcurrently\b",
    r"\bover\s+time\b",
    r"\bwhen\s+did\b",
    r"\bhistory\b",
    r"\bevolved\b",
    r"\bpreviously\b",
    r"\bused\s+to\b",
]

# PREFERENCE patterns
_PREFERENCE_PATTERNS = [
    r"\bprefer\b",
    r"\brecommend\b",
    r"\bbest\s+way\b",
    r"\bshould\s+we\s+use\b",
    r"\bwhich\s+is\s+better\b",
]

# COMPARISON patterns
_COMPARISON_PATTERNS = [
    r"\bcompare\b",
    r"\bdifference\b",
    r"\bvs\b",
    r"\bversus\b",
    r"\btrade-off\b",
    r"\btradeoff\b",
    r"\bpros\s+and\s+cons\b",
]

# SPEAKER pattern: "what did <Name> say/tell/mention"
_SPEAKER_PATTERN = re.compile(
    r"\bwhat\s+did\s+([A-Z][a-z]+)\s+(?:say|tell|mention)\b",
    re.IGNORECASE,
)
# Also match: "<Name> said"
_NAME_SAID_PATTERN = re.compile(
    r"\b([A-Z][a-z]+)\s+said\b",
)

# SELF_SPEAKER patterns
_SELF_SPEAKER_PATTERNS = [
    r"\byou\s+told\s+me\b",
    r"\bI\s+said\b",
    r"\bI\s+told\s+you\b",
    r"\byou\s+mentioned\b",
]

# TIME constraint patterns
_TIME_PATTERNS = [
    (r"\blast\s+week\b", "last week"),
    (r"\blast\s+month\b", "last month"),
    (r"\blast\s+year\b", "last year"),
    (r"\byesterday\b", "yesterday"),
    (r"\btoday\b", "today"),
    (r"\bthis\s+week\b", "this week"),
    (r"\bthis\s+month\b", "this month"),
    (r"\bbefore\s+\w+\b", None),    # "before X" — capture the match
    (r"\bin\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", None),
    (r"\bin\s+Q[1-4]\b", None),
    (r"\brecently\b", "recently"),
    (r"\bearlier\b", "earlier"),
]

# CamelCase entity pattern (e.g. GraphQL, PostgreSQL, TanStack)
_CAMEL_CASE_PATTERN = re.compile(r"\b[A-Z][a-z]+[A-Z]\w*\b|\b[A-Z]{2,}\w*\b")

# Quoted string pattern
_QUOTED_PATTERN = re.compile(r'"([^"]+)"|\'([^\']+)\'')


# ── QueryFrame dataclass ─────────────────────────────────────────────────────


@dataclass
class QueryFrame:
    answer_target: str = "fact"
    # fact, preference, procedure, failure, temporal_change, prior_statement, comparison
    entities: List[str] = field(default_factory=list)
    time_constraint: Optional[str] = None
    speaker_role_constraint: Optional[str] = None
    entity_constraint: Optional[str] = None  # named person
    retrieval_mode: str = "single_hop"
    # single_hop, compare_over_time, multi_hop, abstention_risk
    confidence: float = 0.0


# ── Helpers ──────────────────────────────────────────────────────────────────


def _matches_any(text: str, patterns: list) -> bool:
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def _derive_retrieval_mode(answer_target: str) -> str:
    if answer_target == "temporal_change":
        return "compare_over_time"
    if answer_target == "comparison":
        return "multi_hop"
    return "single_hop"


# ── Pattern-based extraction ─────────────────────────────────────────────────


def _pattern_extract(query: str) -> QueryFrame:
    frame = QueryFrame()
    signals = 0

    # --- Speaker constraint check first (specific answer_target) ---
    speaker_match = _SPEAKER_PATTERN.search(query)
    if not speaker_match:
        speaker_match = _NAME_SAID_PATTERN.search(query)
    if speaker_match:
        frame.entity_constraint = speaker_match.group(1).capitalize()
        frame.answer_target = "prior_statement"
        signals += 2  # strong signal

    # --- Self-speaker ---
    elif _matches_any(query, _SELF_SPEAKER_PATTERNS):
        frame.speaker_role_constraint = "assistant"
        frame.answer_target = "prior_statement"
        signals += 2

    # --- Answer target classification (only if not already set by speaker) ---
    if frame.answer_target == "fact":
        # Count matching patterns within each category for stronger signal
        proc_count = sum(
            1 for p in _PROCEDURE_PATTERNS if re.search(p, query, re.IGNORECASE)
        )
        if proc_count > 0:
            frame.answer_target = "procedure"
            signals += min(proc_count, 2)  # cap at 2 to avoid inflating
        elif _matches_any(query, _FAILURE_PATTERNS):
            frame.answer_target = "failure"
            signals += 1
        elif _matches_any(query, _COMPARISON_PATTERNS):
            frame.answer_target = "comparison"
            signals += 1
        elif _matches_any(query, _PREFERENCE_PATTERNS):
            frame.answer_target = "preference"
            signals += 1

    # --- Time constraints ---
    for pat, label in _TIME_PATTERNS:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            frame.time_constraint = label if label else m.group(0)
            signals += 1
            # Time signals imply temporal_change if not already more specific
            if frame.answer_target == "fact":
                frame.answer_target = "temporal_change"
            elif frame.answer_target not in ("prior_statement", "comparison"):
                frame.answer_target = "temporal_change"
            break

    # Check temporal patterns (don't override time-based temporal_change)
    if frame.answer_target not in ("procedure", "failure", "comparison", "preference", "prior_statement"):
        if _matches_any(query, _TEMPORAL_PATTERNS):
            frame.answer_target = "temporal_change"
            signals += 1

    # --- Entity extraction ---
    # CamelCase entities
    camel_matches = _CAMEL_CASE_PATTERN.findall(query)
    for m in camel_matches:
        if m not in _STOP_WORDS and len(m) > 1:
            frame.entities.append(m)

    # Quoted strings
    for m in _QUOTED_PATTERN.finditer(query):
        val = m.group(1) or m.group(2)
        if val and val not in frame.entities:
            frame.entities.append(val)

    if frame.entities:
        signals += 1

    # --- Deduplicate entities ---
    seen = set()
    unique_entities = []
    for e in frame.entities:
        if e not in seen:
            seen.add(e)
            unique_entities.append(e)
    frame.entities = unique_entities

    # --- Retrieval mode ---
    frame.retrieval_mode = _derive_retrieval_mode(frame.answer_target)

    # --- Confidence ---
    frame.confidence = min(signals / 4.0, 1.0)

    return frame


# ── LLM fallback ─────────────────────────────────────────────────────────────

_LLM_PROMPT = """Analyze this query and return a JSON object with these fields:
- answer_target: one of "fact", "preference", "procedure", "failure", "temporal_change", "prior_statement", "comparison"
- entities: list of named entities (technologies, people, projects) mentioned
- time_constraint: temporal scope if any (null otherwise)
- speaker_role_constraint: "assistant" if self-referential, null otherwise
- entity_constraint: named person if asking about what someone said (null otherwise)
- retrieval_mode: one of "single_hop", "compare_over_time", "multi_hop", "abstention_risk"

Query: {query}

Respond with valid JSON only."""


def _llm_extract(query: str) -> Optional[QueryFrame]:
    try:
        from .llm import call_llm

        prompt = _LLM_PROMPT.format(query=query)
        response = call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
            json_mode=True,
            timeout=30,
        )
        data = json.loads(response)
        frame = QueryFrame(
            answer_target=data.get("answer_target", "fact"),
            entities=data.get("entities", []),
            time_constraint=data.get("time_constraint"),
            speaker_role_constraint=data.get("speaker_role_constraint"),
            entity_constraint=data.get("entity_constraint"),
            retrieval_mode=data.get("retrieval_mode", "single_hop"),
            confidence=1.0,
        )
        return frame
    except Exception as e:
        logger.debug("LLM fallback failed: %s", e)
        return None


# ── Public API ────────────────────────────────────────────────────────────────


def extract_query_frame(
    query: str,
    entity_registry=None,
    use_llm_fallback: bool = False,
) -> QueryFrame:
    """
    Analyze a query string and return a structured QueryFrame.

    Uses regex pattern matching as the primary strategy. When confidence < 0.6
    and use_llm_fallback=True, falls back to an LLM call.

    Args:
        query: The query string to analyze.
        entity_registry: Optional entity registry for entity validation (unused currently).
        use_llm_fallback: Whether to use LLM when pattern confidence is low.

    Returns:
        A QueryFrame with answer_target, entities, constraints, and retrieval_mode.
    """
    frame = _pattern_extract(query)

    if use_llm_fallback and frame.confidence < 0.6:
        llm_frame = _llm_extract(query)
        if llm_frame is not None:
            return llm_frame

    return frame
