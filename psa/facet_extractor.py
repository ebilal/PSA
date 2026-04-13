"""Heuristic facet extractor: extracts entities, temporal markers, speaker role,
actor entities, and stance from raw text using regex patterns. No LLM required."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExtractedFacets:
    entities: List[str]
    actor_entities: List[str]
    speaker_role: Optional[str]
    stance: Optional[str]
    mentioned_at: Optional[str]


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# CamelCase: mixed-case identifiers like GraphQL, MyService, OAuth2
# Matches words starting with uppercase that contain at least one more uppercase letter
_RE_CAMEL = re.compile(r"\b[A-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b")

# Quoted strings (double or single quotes)
_RE_QUOTED = re.compile(r'"([^"]+)"|\'([^\']+)\'')

# File paths: tokens containing '/' or '.' that look like paths
_RE_FILE_PATH = re.compile(
    r"(?<!\w)"
    r"(?:[a-zA-Z0-9_.-]+/[a-zA-Z0-9_./:-]+|[a-zA-Z0-9_-]+\.[a-zA-Z]{2,6}(?:/[a-zA-Z0-9_./:-]*)?)"
    r"(?!\w)"
)

# URLs
_RE_URL = re.compile(r"https?://\S+")

# @mentions
_RE_MENTION = re.compile(r"@[A-Za-z][A-Za-z0-9_.-]+")

# ISO date or year-month: 2026-03 or 2026-03-15
_RE_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}(?:-\d{2})?)\b")

# Month name + year: "March 2026"
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
)
_RE_MONTH_YEAR = re.compile(rf"\b({_MONTHS})\s+(\d{{4}})\b")

# Relative time markers
_RE_RELATIVE = re.compile(
    r"\b(last\s+(?:week|month|year|quarter)|yesterday|today|"
    r"this\s+(?:week|month|year|quarter)|Q[1-4]\s+\d{{4}}|Q[1-4])\b",
    re.IGNORECASE,
)

# Speaker: line starts with "> "
_RE_USER_QUOTE = re.compile(r"(?:^|\n)>\s", re.MULTILINE)

# Stance patterns
_STANCE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("switched", re.compile(r"\b(switched|migrated|moved|transitioned)\s+(to|from|away)\b", re.IGNORECASE)),
    ("stopped", re.compile(r"\b(stopped\s+using|no\s+longer\s+using|abandoned|dropped)\b", re.IGNORECASE)),
    ("deprecated", re.compile(r"\b(deprecated|obsolete|legacy|removed)\b", re.IGNORECASE)),
    ("failed", re.compile(r"\b(failed|broke|broken|crashing|crash)\b", re.IGNORECASE)),
    ("fixed", re.compile(r"\b(fixed|resolved|patched|repaired)\b", re.IGNORECASE)),
    ("prefers", re.compile(r"\b(prefer[s]?|recommend[s]?|suggest[s]?|advocate[s]?)\b", re.IGNORECASE)),
]

# Actor entities: "Name said/told/mentioned/suggested ..."
_RE_ACTOR = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+"
    r"(?:said|told|mentioned|suggested|noted|added|explained|asked|replied)\b"
)


def extract_facets(text: str, entity_registry=None) -> ExtractedFacets:
    """Extract facets from *text* using heuristic regex patterns.

    Parameters
    ----------
    text:
        Raw text to analyse.
    entity_registry:
        Optional entity registry (unused by heuristics, reserved for future use).

    Returns
    -------
    ExtractedFacets
    """
    entities: set[str] = set()

    # CamelCase tokens
    for m in _RE_CAMEL.finditer(text):
        entities.add(m.group())

    # Quoted strings
    for m in _RE_QUOTED.finditer(text):
        value = m.group(1) or m.group(2)
        if value:
            entities.add(value)

    # File paths (skip pure domain-like tokens such as "e.g." or "i.e.")
    for m in _RE_FILE_PATH.finditer(text):
        token = m.group()
        # Must contain a slash or look like a real file extension
        if "/" in token or re.search(r"\.[a-z]{2,6}$", token):
            entities.add(token)

    # URLs
    for m in _RE_URL.finditer(text):
        entities.add(m.group())

    # @mentions
    for m in _RE_MENTION.finditer(text):
        entities.add(m.group())

    # Deduplicate and sort
    entity_list = sorted(entities)

    # ------------------------------------------------------------------
    # Temporal
    # ------------------------------------------------------------------
    mentioned_at: Optional[str] = None

    iso = _RE_ISO_DATE.search(text)
    if iso:
        mentioned_at = iso.group(1)
    else:
        month_year = _RE_MONTH_YEAR.search(text)
        if month_year:
            mentioned_at = month_year.group(0)
        else:
            rel = _RE_RELATIVE.search(text)
            if rel:
                mentioned_at = rel.group(0)

    # ------------------------------------------------------------------
    # Speaker role
    # ------------------------------------------------------------------
    if _RE_USER_QUOTE.search(text):
        speaker_role: Optional[str] = "user"
    else:
        speaker_role = "assistant"

    # ------------------------------------------------------------------
    # Stance
    # ------------------------------------------------------------------
    stance: Optional[str] = None
    for label, pattern in _STANCE_PATTERNS:
        if pattern.search(text):
            stance = label
            break

    # ------------------------------------------------------------------
    # Actor entities
    # ------------------------------------------------------------------
    actor_entities: list[str] = []
    seen_actors: set[str] = set()
    for m in _RE_ACTOR.finditer(text):
        name = m.group(1)
        if name not in seen_actors:
            seen_actors.add(name)
            actor_entities.append(name)

    return ExtractedFacets(
        entities=entity_list,
        actor_entities=actor_entities,
        speaker_role=speaker_role,
        stance=stance,
        mentioned_at=mentioned_at,
    )
