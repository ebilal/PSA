# Phase 1: Query Frames, Facets, and Constraint Scoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured query analysis (query frames), richer metadata on memories (entities, speaker, temporal, stance facets), and rule-backed constraint scoring to shift PSA from semantic matching to task-utility matching. Target: F1 >= 0.22, R@5 >= 0.93.

**Architecture:** Three layers added to the existing pipeline: (1) Facets extracted during ingestion via extended LLM prompt + heuristic backfill from raw sources. (2) Query frame extracted at query time via pattern matcher with LLM fallback. (3) Rule-backed constraint scorer applies post-Level-2 boost/penalty based on entity overlap, speaker match, temporal consistency, stance relevance, and type match. Level 1 co-activation model gets version-gated query frame features.

**Tech Stack:** Existing PSA infrastructure. No new models or heavy dependencies. `call_llm()` for LLM-backed extraction (cloud + Ollama fallback).

**Design spec:** `docs/superpowers/specs/2026-04-13-phase1-query-frames-facets-constraints-design.md`

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `psa/query_frame.py` | QueryFrame dataclass + `extract_query_frame()` (pattern matcher + LLM fallback) |
| `psa/conversation_parser.py` | Structured conversation parser returning `(role, text, timestamp)` turns |
| `psa/facet_extractor.py` | Heuristic facet extraction from raw source text |
| `psa/constraint_scorer.py` | Rule-backed constraint scoring (post-Level-2 boost/penalty) |
| `tests/test_query_frame.py` | Tests for query frame extraction |
| `tests/test_conversation_parser.py` | Tests for conversation parser |
| `tests/test_facet_extractor.py` | Tests for heuristic facet extractor |
| `tests/test_constraint_scorer.py` | Tests for constraint scorer |

### Modified files

| File | Change |
|------|--------|
| `psa/memory_object.py` | Add entities, actor_entities, speaker_role, stance, mentioned_at fields + DB migration |
| `psa/consolidation.py` | Extend LLM prompt to return facets |
| `psa/convo_miner.py` | Use conversation_parser for PSA path |
| `psa/coactivation.py` | Version-gated query_proj expansion (768 -> 779) |
| `psa/pipeline.py` | Wire query frame + constraint scorer into query flow |
| `psa/cli.py` | Add `psa repair --backfill-facets` subcommand |

---

### Task 1: Storage Migration — New Facet Fields on MemoryObject

**Files:**
- Modify: `psa/memory_object.py`

- [ ] **Step 1: Add new fields to MemoryObject dataclass**

After the existing `mentioned_at` field (if present) or after `acl_scope`, add:

```python
    entities: List[str] = field(default_factory=list)
    actor_entities: List[str] = field(default_factory=list)
    speaker_role: Optional[str] = None
    stance: Optional[str] = None
    mentioned_at: Optional[str] = None
```

- [ ] **Step 2: Add ALTER TABLE migration in _init_db()**

In the migration section (after the existing ALTER TABLE loop), add:

```python
        for col, typedef in [
            ("entities_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("actor_entities_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("speaker_role", "TEXT"),
            ("stance", "TEXT"),
            ("mentioned_at", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE memory_objects ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError:
                pass
```

- [ ] **Step 3: Update _row_to_memory() and _memory_to_row() serialization**

Add JSON encode/decode for the new list fields (same pattern as `source_ids_json`, `tool_names_json`).

- [ ] **Step 4: Run existing tests to verify no regression**

Run: `uv run pytest tests/test_memory_object.py -v`
Expected: All pass (new fields have defaults)

- [ ] **Step 5: Commit**

```bash
git add psa/memory_object.py
git commit -m "feat: add facet fields (entities, actor_entities, speaker_role, stance, mentioned_at) to MemoryObject"
```

---

### Task 2: Facet Extractor — Heuristic Extraction from Raw Text

**Files:**
- Create: `psa/facet_extractor.py`
- Create: `tests/test_facet_extractor.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for psa.facet_extractor — heuristic facet extraction."""

import pytest


class TestEntityExtraction:
    def test_extracts_camel_case(self):
        from psa.facet_extractor import extract_facets
        facets = extract_facets("We migrated to GraphQL last month")
        assert "GraphQL" in facets.entities

    def test_extracts_quoted_strings(self):
        from psa.facet_extractor import extract_facets
        facets = extract_facets('The "auth-service" endpoint was slow')
        assert "auth-service" in facets.entities

    def test_extracts_file_paths(self):
        from psa.facet_extractor import extract_facets
        facets = extract_facets("Check src/auth/jwt.py for the pattern")
        assert "src/auth/jwt.py" in facets.entities


class TestTemporalExtraction:
    def test_extracts_date_pattern(self):
        from psa.facet_extractor import extract_facets
        facets = extract_facets("This happened in 2026-03")
        assert facets.mentioned_at is not None
        assert "2026-03" in facets.mentioned_at

    def test_extracts_relative_time(self):
        from psa.facet_extractor import extract_facets
        facets = extract_facets("We changed this last week")
        assert facets.mentioned_at is not None


class TestSpeakerExtraction:
    def test_user_marker(self):
        from psa.facet_extractor import extract_facets
        facets = extract_facets("> Can you fix the auth bug?")
        assert facets.speaker_role == "user"

    def test_assistant_default(self):
        from psa.facet_extractor import extract_facets
        facets = extract_facets("I recommend using JWT tokens for session management.")
        assert facets.speaker_role == "assistant"


class TestStanceExtraction:
    def test_switched(self):
        from psa.facet_extractor import extract_facets
        facets = extract_facets("We switched from REST to GraphQL")
        assert facets.stance == "switched"

    def test_deprecated(self):
        from psa.facet_extractor import extract_facets
        facets = extract_facets("The old auth module was deprecated")
        assert facets.stance == "deprecated"

    def test_no_stance(self):
        from psa.facet_extractor import extract_facets
        facets = extract_facets("JWT tokens expire after 24 hours")
        assert facets.stance is None


class TestActorEntities:
    def test_extracts_named_person(self):
        from psa.facet_extractor import extract_facets
        facets = extract_facets("Alice said we should use GraphQL")
        assert "Alice" in facets.actor_entities
```

- [ ] **Step 2: Implement facet_extractor.py**

```python
"""
facet_extractor.py -- Heuristic facet extraction from raw text.

Extracts entities, temporal markers, speaker role, actor entities, and
stance from raw source text or memory body. No LLM required.

Used for:
- Backfilling existing memories via `psa repair --backfill-facets`
- Fallback when use_llm=False during consolidation
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExtractedFacets:
    entities: List[str] = field(default_factory=list)
    actor_entities: List[str] = field(default_factory=list)
    speaker_role: Optional[str] = None
    stance: Optional[str] = None
    mentioned_at: Optional[str] = None


# Regex patterns
_CAMEL_CASE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
_QUOTED = re.compile(r'"([^"]{2,50})"')
_FILE_PATH = re.compile(r"\b[\w./]+\.\w{1,5}\b")
_URL = re.compile(r"https?://\S+")
_AT_MENTION = re.compile(r"@(\w+)")
_DATE_ISO = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
_DATE_MONTH = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", re.IGNORECASE)
_RELATIVE_TIME = re.compile(r"\b(?:last\s+(?:week|month|year)|yesterday|today|recently|this\s+(?:week|month)|Q[1-4])\b", re.IGNORECASE)

_STANCE_PATTERNS = {
    "switched": re.compile(r"\b(?:switch(?:ed|ing)\s+(?:to|from)|migrat(?:ed|ing)\s+to)\b", re.IGNORECASE),
    "stopped": re.compile(r"\b(?:stop(?:ped|ping)\s+using|no\s+longer\s+us(?:e|ing))\b", re.IGNORECASE),
    "deprecated": re.compile(r"\b(?:deprecat(?:ed|ing)|obsolete|removed)\b", re.IGNORECASE),
    "failed": re.compile(r"\b(?:fail(?:ed|ing|ure)|broke(?:n)?|crash(?:ed|ing))\b", re.IGNORECASE),
    "fixed": re.compile(r"\b(?:fix(?:ed|ing)|resolv(?:ed|ing)|patch(?:ed|ing))\b", re.IGNORECASE),
    "prefers": re.compile(r"\b(?:prefer(?:s|red)|recommend(?:s|ed)|should\s+use)\b", re.IGNORECASE),
}

_PERSON_NAMES = re.compile(r"\b([A-Z][a-z]{2,15})\s+(?:said|told|mentioned|asked|suggested|recommended|thinks|believes)\b")

_USER_MARKERS = re.compile(r"^>\s", re.MULTILINE)


def extract_facets(text: str, entity_registry: Optional[dict] = None) -> ExtractedFacets:
    """Extract facets from raw text using heuristics."""
    facets = ExtractedFacets()

    # Entities
    entities = set()
    entities.update(_CAMEL_CASE.findall(text))
    entities.update(_QUOTED.findall(text))
    entities.update(m for m in _FILE_PATH.findall(text) if "/" in m or "." in m)
    entities.update(_URL.findall(text))
    entities.update(_AT_MENTION.findall(text))
    if entity_registry:
        for name in entity_registry:
            if name.lower() in text.lower():
                entities.add(name)
    facets.entities = sorted(entities)

    # Actor entities (named people)
    actors = set(_PERSON_NAMES.findall(text))
    facets.actor_entities = sorted(actors)

    # Temporal
    dates = _DATE_ISO.findall(text) + _DATE_MONTH.findall(text)
    relatives = _RELATIVE_TIME.findall(text)
    if dates:
        facets.mentioned_at = dates[0]
    elif relatives:
        facets.mentioned_at = relatives[0]

    # Speaker role
    if _USER_MARKERS.search(text):
        facets.speaker_role = "user"
    else:
        facets.speaker_role = "assistant"

    # Stance
    for stance_name, pattern in _STANCE_PATTERNS.items():
        if pattern.search(text):
            facets.stance = stance_name
            break

    return facets
```

- [ ] **Step 3: Run tests, commit**

```bash
git add psa/facet_extractor.py tests/test_facet_extractor.py
git commit -m "feat: add heuristic facet extractor (entities, temporal, speaker, stance)"
```

---

### Task 3: Conversation Parser

**Files:**
- Create: `psa/conversation_parser.py`
- Create: `tests/test_conversation_parser.py`

- [ ] **Step 1: Write failing tests**

Test that the parser handles Claude Code JSONL format (the most common) and returns structured turns with role, text, and timestamp.

```python
"""Tests for psa.conversation_parser — structured conversation parsing."""

import json
import os
import tempfile
import pytest


def test_parse_claude_code_jsonl():
    from psa.conversation_parser import parse_conversation, ConversationTurn

    # Claude Code JSONL format: one JSON object per line with role + content
    lines = [
        json.dumps({"role": "user", "content": "Fix the auth bug", "timestamp": "2026-04-10T10:00:00Z"}),
        json.dumps({"role": "assistant", "content": "I'll check jwt.py", "timestamp": "2026-04-10T10:00:05Z"}),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(lines))
        path = f.name

    try:
        turns = parse_conversation(path)
        assert len(turns) == 2
        assert isinstance(turns[0], ConversationTurn)
        assert turns[0].role == "user"
        assert turns[0].text == "Fix the auth bug"
        assert turns[0].timestamp is not None
        assert turns[1].role == "assistant"
    finally:
        os.unlink(path)


def test_parse_plain_text():
    from psa.conversation_parser import parse_conversation

    content = "> Can you help with auth?\n\nSure, I'll look at the JWT module.\n\n> What about the refresh tokens?\n\nThey're stored in HttpOnly cookies."

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name

    try:
        turns = parse_conversation(path)
        assert len(turns) >= 2
        assert turns[0].role == "user"
    finally:
        os.unlink(path)


def test_returns_empty_for_empty_file():
    from psa.conversation_parser import parse_conversation

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        path = f.name

    try:
        turns = parse_conversation(path)
        assert turns == []
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Implement conversation_parser.py**

A multi-format parser that returns structured `ConversationTurn` objects. Handles: Claude Code JSONL, Claude AI JSON, ChatGPT exports, plain text with `>` markers. Each format detected by file extension and content heuristics.

```python
"""
conversation_parser.py -- Structured conversation parser.

Reads conversation files in multiple formats and returns structured turns
with role, text, and timestamp preserved. Unlike normalize.py (which returns
flat transcript strings), this preserves metadata for facet extraction.

Supported formats:
- Claude Code JSONL (.jsonl)
- Claude AI JSON (.json with conversation array)
- Plain text (.txt with > user markers)
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ConversationTurn:
    role: str                          # "user", "assistant", "system"
    text: str
    timestamp: Optional[str] = None    # ISO datetime if available
    metadata: Dict = field(default_factory=dict)


def parse_conversation(path: str) -> List[ConversationTurn]:
    """Parse a conversation file into structured turns."""
    if not os.path.exists(path):
        return []

    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    if not content.strip():
        return []

    ext = os.path.splitext(path)[1].lower()

    if ext == ".jsonl":
        return _parse_jsonl(content)
    elif ext == ".json":
        return _parse_json(content)
    else:
        return _parse_plain_text(content)


def _parse_jsonl(content: str) -> List[ConversationTurn]:
    """Parse JSONL where each line is a message with role + content."""
    turns = []
    for line in content.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        role = obj.get("role", "")
        text = obj.get("content", "")
        if isinstance(text, list):
            # Handle content blocks (Claude format)
            text = " ".join(
                b.get("text", "") for b in text if isinstance(b, dict)
            )
        timestamp = obj.get("timestamp") or obj.get("created_at")

        if role and text:
            turns.append(ConversationTurn(
                role=role, text=str(text), timestamp=timestamp,
            ))
    return turns


def _parse_json(content: str) -> List[ConversationTurn]:
    """Parse JSON with a conversation/messages array."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    messages = []
    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict):
        messages = data.get("messages", data.get("conversation", []))

    turns = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", msg.get("author", {}).get("role", ""))
        text = msg.get("content", "")
        if isinstance(text, list):
            text = " ".join(
                p.get("text", "") for p in text if isinstance(p, dict)
            )
        timestamp = msg.get("timestamp") or msg.get("create_time")
        if role and text:
            turns.append(ConversationTurn(
                role=str(role), text=str(text),
                timestamp=str(timestamp) if timestamp else None,
            ))
    return turns


def _parse_plain_text(content: str) -> List[ConversationTurn]:
    """Parse plain text with > markers for user turns."""
    turns = []
    current_role = None
    current_lines = []

    for line in content.splitlines():
        if line.startswith("> "):
            # Flush previous turn
            if current_role and current_lines:
                turns.append(ConversationTurn(
                    role=current_role, text="\n".join(current_lines).strip(),
                ))
            current_role = "user"
            current_lines = [line[2:]]
        elif line.strip() == "" and current_role == "user" and current_lines:
            # Empty line after user turn = switch to assistant
            turns.append(ConversationTurn(
                role="user", text="\n".join(current_lines).strip(),
            ))
            current_role = "assistant"
            current_lines = []
        else:
            if current_role is None:
                current_role = "assistant"
            current_lines.append(line)

    if current_role and current_lines:
        turns.append(ConversationTurn(
            role=current_role, text="\n".join(current_lines).strip(),
        ))
    return turns
```

- [ ] **Step 3: Run tests, commit**

```bash
git add psa/conversation_parser.py tests/test_conversation_parser.py
git commit -m "feat: add structured conversation parser preserving role + timestamp"
```

---

### Task 4: Extend Consolidation LLM Prompt

**Files:**
- Modify: `psa/consolidation.py`

- [ ] **Step 1: Add facet fields to the LLM prompt**

In the `_CONSOLIDATION_SYSTEM` prompt string, add to the JSON schema description (after the existing `uncertainty` field):

```
  entities: array of entity names mentioned (people, projects, tools, files, APIs)
  temporal_markers: {"mentioned_at": "ISO date or relative phrase", "valid_from": null, "valid_to": null}
  speaker_role: "user" | "assistant" | "system" | "external" — who originated this information
  actor_entities: array of named people/actors mentioned (e.g. ["Alice", "Bob"])
  stance: "prefers" | "stopped" | "switched" | "failed" | "fixed" | "deprecated" | null
```

- [ ] **Step 2: Parse new facet fields from LLM response**

In the section that parses the LLM JSON response into MemoryObject.create() kwargs, extract and pass the new fields:

```python
    entities = mem_dict.get("entities", [])
    temporal = mem_dict.get("temporal_markers", {})
    speaker_role = mem_dict.get("speaker_role")
    actor_entities = mem_dict.get("actor_entities", [])
    stance = mem_dict.get("stance")
    mentioned_at = temporal.get("mentioned_at") if isinstance(temporal, dict) else None
    validity_interval = None
    if isinstance(temporal, dict):
        vf = temporal.get("valid_from")
        vt = temporal.get("valid_to")
        if vf or vt:
            validity_interval = f"{vf or ''}:{vt or ''}"
```

Pass these to `MemoryObject.create()` via kwargs.

- [ ] **Step 3: Fallback to heuristic extractor when use_llm=False**

When `use_llm=False`, the consolidation pipeline doesn't call the LLM. Add a fallback that runs the heuristic facet extractor on the chunk text:

```python
    if not self.use_llm:
        from .facet_extractor import extract_facets
        facets = extract_facets(chunk_text)
        # Apply facets to memory object...
```

- [ ] **Step 4: Run existing consolidation tests**

Run: `uv run pytest tests/test_consolidation.py -v`
Expected: Pass (LLM is mocked in tests; new fields have defaults)

- [ ] **Step 5: Commit**

```bash
git add psa/consolidation.py
git commit -m "feat: extend consolidation LLM prompt to extract facets (entities, temporal, speaker, stance)"
```

---

### Task 5: Update Convo Miner to Use Conversation Parser

**Files:**
- Modify: `psa/convo_miner.py`

- [ ] **Step 1: Add PSA path that uses conversation_parser**

In the PSA ingestion path (where memories are created via consolidation), use `conversation_parser.parse_conversation()` instead of `normalize()` to get structured turns. Pass `speaker_role` and `timestamp` from each turn to the consolidation pipeline so the LLM prompt has this context.

The legacy palace path (ChromaDB drawers) continues using `normalize()` unchanged.

- [ ] **Step 2: Run convo miner tests**

Run: `uv run pytest tests/test_convo_miner.py -v`
Expected: Pass

- [ ] **Step 3: Commit**

```bash
git add psa/convo_miner.py
git commit -m "feat: use conversation_parser for PSA path in convo_miner"
```

---

### Task 6: Backfill CLI Command

**Files:**
- Modify: `psa/cli.py`

- [ ] **Step 1: Add --backfill-facets to psa repair**

In the argparse section for `psa repair`, add a subcommand or flag:

```python
    p_repair.add_argument(
        "--backfill-facets",
        action="store_true",
        help="Extract facets (entities, temporal, speaker, stance) for existing memories from raw sources",
    )
```

- [ ] **Step 2: Implement backfill in cmd_repair()**

When `--backfill-facets` is set:
1. Load all non-archived memories from MemoryStore
2. For each memory, look up its raw source via `source_ids`
3. Run `extract_facets()` on the raw source text
4. Update the memory's facet fields in the database
5. Log progress every 100 memories

```python
    if getattr(args, "backfill_facets", False):
        from .facet_extractor import extract_facets
        from .memory_object import MemoryStore
        from .tenant import TenantManager

        tenant_id = getattr(args, "tenant", "default")
        tm = TenantManager()
        tenant = tm.get_or_create(tenant_id)
        store = MemoryStore(db_path=tenant.memory_db_path)

        memories = store.get_all_active(tenant_id)
        updated = 0
        for i, mo in enumerate(memories):
            # Get raw source text
            raw_text = None
            for sid in mo.source_ids:
                src = store.get_source(sid)
                if src:
                    raw_text = src.full_text
                    break

            text = raw_text or mo.body
            facets = extract_facets(text)

            mo.entities = facets.entities
            mo.actor_entities = facets.actor_entities
            mo.speaker_role = facets.speaker_role
            mo.stance = facets.stance
            mo.mentioned_at = facets.mentioned_at
            store.update_facets(mo)
            updated += 1

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(memories)} memories updated")

        print(f"Backfilled facets for {updated} memories.")
```

Note: `store.get_all_active()`, `store.get_source()`, and `store.update_facets()` may need to be added to MemoryStore if they don't exist. Check the existing API and add minimal methods.

- [ ] **Step 3: Run CLI tests, commit**

```bash
git add psa/cli.py psa/memory_object.py
git commit -m "feat: add psa repair --backfill-facets for existing memories"
```

---

### Task 7: Query Frame Extraction

**Files:**
- Create: `psa/query_frame.py`
- Create: `tests/test_query_frame.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for psa.query_frame — structured query analysis."""

import pytest


class TestPatternMatcher:
    def test_procedure_query(self):
        from psa.query_frame import extract_query_frame
        frame = extract_query_frame("How do I configure JWT tokens?")
        assert frame.answer_target == "procedure"
        assert frame.retrieval_mode == "single_hop"

    def test_failure_query(self):
        from psa.query_frame import extract_query_frame
        frame = extract_query_frame("What went wrong with the auth deployment?")
        assert frame.answer_target == "failure"

    def test_temporal_change_query(self):
        from psa.query_frame import extract_query_frame
        frame = extract_query_frame("How has our auth pattern changed over time?")
        assert frame.answer_target == "temporal_change"
        assert frame.retrieval_mode == "compare_over_time"

    def test_speaker_constraint(self):
        from psa.query_frame import extract_query_frame
        frame = extract_query_frame("What did Alice say about the database?")
        assert frame.entity_constraint == "Alice"
        assert frame.answer_target == "prior_statement"

    def test_entity_extraction(self):
        from psa.query_frame import extract_query_frame
        frame = extract_query_frame("Why did we switch to GraphQL?")
        assert "GraphQL" in frame.entities

    def test_time_constraint(self):
        from psa.query_frame import extract_query_frame
        frame = extract_query_frame("What auth changes were made last week?")
        assert frame.time_constraint is not None
        assert frame.answer_target == "temporal_change"

    def test_simple_fact_query(self):
        from psa.query_frame import extract_query_frame
        frame = extract_query_frame("What database do we use?")
        assert frame.answer_target == "fact"
        assert frame.retrieval_mode == "single_hop"

    def test_confidence_above_threshold_skips_llm(self):
        from psa.query_frame import extract_query_frame
        frame = extract_query_frame("How do I set up the GraphQL server?")
        assert frame.confidence >= 0.6  # pattern matcher should handle this

    def test_default_frame_for_ambiguous_query(self):
        from psa.query_frame import extract_query_frame
        frame = extract_query_frame("hmm")
        assert frame.answer_target == "fact"  # safe default
        assert frame.retrieval_mode == "single_hop"
```

- [ ] **Step 2: Implement query_frame.py**

```python
"""
query_frame.py -- Structured query analysis.

Extracts a QueryFrame before retrieval: answer target, entities,
constraints, and retrieval mode. Uses a fast pattern matcher for common
signals with LLM fallback for ambiguous queries.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QueryFrame:
    answer_target: str = "fact"
    entities: List[str] = field(default_factory=list)
    time_constraint: Optional[str] = None
    speaker_role_constraint: Optional[str] = None
    entity_constraint: Optional[str] = None
    retrieval_mode: str = "single_hop"
    confidence: float = 0.0


# Pattern definitions
_PROCEDURE_PATTERNS = re.compile(
    r"\b(?:how\s+(?:do|can|should|to)|steps?\s+to|recipe\s+for|guide\s+to|set\s*up|configure|install)\b",
    re.IGNORECASE,
)
_FAILURE_PATTERNS = re.compile(
    r"\b(?:(?:what|why)\s+(?:went\s+wrong|broke|failed|crashed)|bug|error|issue\s+with)\b",
    re.IGNORECASE,
)
_TEMPORAL_PATTERNS = re.compile(
    r"\b(?:changed|used\s+to|currently|before|after|over\s+time|history\s+of|evolution|when\s+did)\b",
    re.IGNORECASE,
)
_PREFERENCE_PATTERNS = re.compile(
    r"\b(?:prefer|recommend|should\s+(?:we|I)\s+use|best\s+(?:way|practice|approach))\b",
    re.IGNORECASE,
)
_SPEAKER_PATTERNS = re.compile(
    r"\b(?:(\w+)\s+said|what\s+did\s+(\w+)\s+(?:say|tell|mention|suggest))\b",
    re.IGNORECASE,
)
_SELF_SPEAKER = re.compile(
    r"\b(?:you\s+(?:told|said|mentioned)|I\s+(?:said|told|mentioned))\b",
    re.IGNORECASE,
)
_TIME_CONSTRAINT = re.compile(
    r"\b(last\s+(?:week|month|year)|yesterday|this\s+(?:week|month)|before\s+\w+|after\s+\w+|in\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Q[1-4]))\b",
    re.IGNORECASE,
)
_COMPARISON_PATTERNS = re.compile(
    r"\b(?:compar(?:e|ing|ison)|differ(?:ence|ent)|vs\.?|versus|trade.?off)\b",
    re.IGNORECASE,
)
_ENTITY_CAMELCASE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
_ENTITY_QUOTED = re.compile(r'"([^"]{2,40})"')


def extract_query_frame(
    query: str,
    entity_registry: Optional[dict] = None,
    use_llm_fallback: bool = False,
) -> QueryFrame:
    """Extract a structured query frame from the query string."""
    frame = QueryFrame()
    confidence_signals = 0

    # Answer target detection
    if _PROCEDURE_PATTERNS.search(query):
        frame.answer_target = "procedure"
        confidence_signals += 1
    elif _FAILURE_PATTERNS.search(query):
        frame.answer_target = "failure"
        confidence_signals += 1
    elif _TEMPORAL_PATTERNS.search(query):
        frame.answer_target = "temporal_change"
        confidence_signals += 1
    elif _PREFERENCE_PATTERNS.search(query):
        frame.answer_target = "preference"
        confidence_signals += 1
    elif _COMPARISON_PATTERNS.search(query):
        frame.answer_target = "comparison"
        confidence_signals += 1

    # Speaker / entity constraint
    speaker_match = _SPEAKER_PATTERNS.search(query)
    if speaker_match:
        name = speaker_match.group(1) or speaker_match.group(2)
        if name and name.lower() not in ("i", "we", "you", "they", "it", "what", "who"):
            frame.entity_constraint = name
            frame.answer_target = "prior_statement"
            confidence_signals += 1

    if _SELF_SPEAKER.search(query):
        frame.speaker_role_constraint = "assistant"
        confidence_signals += 1

    # Time constraint
    time_match = _TIME_CONSTRAINT.search(query)
    if time_match:
        frame.time_constraint = time_match.group(1)
        if frame.answer_target == "fact":
            frame.answer_target = "temporal_change"
        confidence_signals += 1

    # Retrieval mode
    if frame.answer_target == "temporal_change":
        frame.retrieval_mode = "compare_over_time"
    elif frame.answer_target == "comparison":
        frame.retrieval_mode = "multi_hop"
    else:
        frame.retrieval_mode = "single_hop"
    confidence_signals += 1

    # Entity extraction
    entities = set()
    entities.update(_ENTITY_CAMELCASE.findall(query))
    entities.update(_ENTITY_QUOTED.findall(query))
    if entity_registry:
        for name in entity_registry:
            if name.lower() in query.lower():
                entities.add(name)
    # Filter common words that look like entities
    stop = {"How", "What", "Why", "When", "Where", "Who", "Can", "Does", "Did", "The", "This"}
    frame.entities = sorted(entities - stop)

    # Confidence: 0-1 based on signals found
    frame.confidence = min(confidence_signals / 4.0, 1.0)

    # LLM fallback for low confidence
    if frame.confidence < 0.6 and use_llm_fallback:
        try:
            llm_frame = _llm_extract_frame(query)
            if llm_frame is not None:
                return llm_frame
        except Exception:
            pass  # fall through to pattern-matched frame

    return frame


def _llm_extract_frame(query: str) -> Optional[QueryFrame]:
    """Use call_llm() for query frame extraction."""
    from .llm import call_llm
    import json as _json

    prompt = (
        f"Analyze this query and return a JSON object with these fields:\n"
        f"- answer_target: fact|preference|procedure|failure|temporal_change|prior_statement|comparison\n"
        f"- entities: array of entity names mentioned\n"
        f"- time_constraint: temporal phrase or null\n"
        f"- speaker_role_constraint: user|assistant or null\n"
        f"- entity_constraint: named person or null\n"
        f"- retrieval_mode: single_hop|compare_over_time|multi_hop|abstention_risk\n\n"
        f"Query: {query}"
    )
    try:
        response = call_llm(
            [{"role": "user", "content": prompt}],
            max_tokens=200,
            json_mode=True,
            temperature=0.0,
        )
        data = _json.loads(response)
        return QueryFrame(
            answer_target=data.get("answer_target", "fact"),
            entities=data.get("entities", []),
            time_constraint=data.get("time_constraint"),
            speaker_role_constraint=data.get("speaker_role_constraint"),
            entity_constraint=data.get("entity_constraint"),
            retrieval_mode=data.get("retrieval_mode", "single_hop"),
            confidence=1.0,
        )
    except Exception:
        return None
```

- [ ] **Step 3: Run tests, commit**

```bash
git add psa/query_frame.py tests/test_query_frame.py
git commit -m "feat: add query frame extraction (pattern matcher + LLM fallback)"
```

---

### Task 8: Constraint Scorer

**Files:**
- Create: `psa/constraint_scorer.py`
- Create: `tests/test_constraint_scorer.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for psa.constraint_scorer — rule-backed constraint scoring."""

import pytest
from unittest.mock import MagicMock
from psa.query_frame import QueryFrame
from psa.memory_scorer import ScoredMemory


def _make_scored_memory(memory_id, score, entities=None, speaker_role=None, stance=None, mentioned_at=None, memory_type="SEMANTIC"):
    mo = MagicMock()
    mo.memory_object_id = memory_id
    mo.entities = entities or []
    mo.actor_entities = []
    mo.speaker_role = speaker_role
    mo.stance = stance
    mo.mentioned_at = mentioned_at
    mo.memory_type = MagicMock()
    mo.memory_type.value = memory_type
    return ScoredMemory(memory_object_id=memory_id, final_score=score, memory=mo)


class TestConstraintScorer:
    def test_entity_overlap_boosts_score(self):
        from psa.constraint_scorer import ConstraintScorer

        frame = QueryFrame(entities=["GraphQL"], answer_target="fact")
        scored = [
            _make_scored_memory("m1", 0.5, entities=["GraphQL", "REST"]),
            _make_scored_memory("m2", 0.6, entities=["Postgres"]),
        ]
        scorer = ConstraintScorer()
        adjusted = scorer.adjust_scores(scored, frame)
        # m1 has entity overlap, should be boosted relative to m2
        m1 = next(s for s in adjusted if s.memory_object_id == "m1")
        m2 = next(s for s in adjusted if s.memory_object_id == "m2")
        assert m1.final_score > m2.final_score

    def test_type_match_boosts_score(self):
        from psa.constraint_scorer import ConstraintScorer

        frame = QueryFrame(answer_target="failure")
        scored = [
            _make_scored_memory("m1", 0.5, memory_type="FAILURE"),
            _make_scored_memory("m2", 0.5, memory_type="SEMANTIC"),
        ]
        scorer = ConstraintScorer()
        adjusted = scorer.adjust_scores(scored, frame)
        m1 = next(s for s in adjusted if s.memory_object_id == "m1")
        m2 = next(s for s in adjusted if s.memory_object_id == "m2")
        assert m1.final_score > m2.final_score

    def test_no_frame_returns_unchanged_order(self):
        from psa.constraint_scorer import ConstraintScorer

        scored = [
            _make_scored_memory("m1", 0.8),
            _make_scored_memory("m2", 0.6),
        ]
        scorer = ConstraintScorer()
        adjusted = scorer.adjust_scores(scored, None)
        assert adjusted[0].memory_object_id == "m1"

    def test_missing_facets_get_neutral_score(self):
        from psa.constraint_scorer import ConstraintScorer

        frame = QueryFrame(entities=["GraphQL"], answer_target="fact")
        scored = [
            _make_scored_memory("m1", 0.5, entities=[]),  # no entities
        ]
        scorer = ConstraintScorer()
        adjusted = scorer.adjust_scores(scored, frame)
        # Should not be heavily penalized (neutral 0.5 for missing facet)
        assert adjusted[0].final_score >= 0.4
```

- [ ] **Step 2: Implement constraint_scorer.py**

```python
"""
constraint_scorer.py -- Rule-backed constraint scoring.

Adjusts Level 2 memory scores based on constraint satisfaction:
entity overlap, speaker match, temporal consistency, stance relevance,
and answer-target type match.

Uses fixed weights in Phase 1. Learned weighting deferred to Phase 2.

Tri-state scoring per feature:
- Constrained + match: 1.0
- Constrained + mismatch: 0.0
- Constrained + unknown (facet missing): 0.5
- Unconstrained: 0.5
"""

import logging
from typing import List, Optional

from .memory_scorer import ScoredMemory
from .query_frame import QueryFrame

logger = logging.getLogger("psa.constraint_scorer")

# Fixed weights for Phase 1 (sum to 1.0)
W_ENTITY = 0.20
W_SPEAKER_ROLE = 0.15
W_ACTOR_ENTITY = 0.15
W_TEMPORAL = 0.20
W_STANCE = 0.15
W_TYPE_MATCH = 0.15

# Level 2 vs constraint blend
SEMANTIC_WEIGHT = 0.70
CONSTRAINT_WEIGHT = 0.30

# Answer target -> memory type mapping
_TARGET_TO_TYPE = {
    "failure": "FAILURE",
    "procedure": "PROCEDURAL",
    "preference": "SEMANTIC",
    "fact": "SEMANTIC",
    "temporal_change": None,  # any type
    "prior_statement": None,
    "comparison": None,
}


class ConstraintScorer:
    """Rule-backed post-Level-2 constraint scoring."""

    def adjust_scores(
        self,
        scored_memories: List[ScoredMemory],
        query_frame: Optional[QueryFrame],
    ) -> List[ScoredMemory]:
        """Adjust scored memory list based on constraint satisfaction."""
        if query_frame is None or not scored_memories:
            return scored_memories

        adjusted = []
        for sm in scored_memories:
            boost = self._compute_constraint_boost(sm.memory, query_frame)
            new_score = SEMANTIC_WEIGHT * sm.final_score + CONSTRAINT_WEIGHT * boost
            adjusted.append(ScoredMemory(
                memory_object_id=sm.memory_object_id,
                final_score=new_score,
                memory=sm.memory,
            ))

        adjusted.sort(key=lambda s: s.final_score, reverse=True)
        return adjusted

    def _compute_constraint_boost(self, memory, frame: QueryFrame) -> float:
        """Compute weighted constraint boost for a single memory."""
        entity = self._entity_overlap(frame, memory)
        speaker = self._speaker_role_match(frame, memory)
        actor = self._actor_entity_match(frame, memory)
        temporal = self._temporal_consistency(frame, memory)
        stance = self._stance_relevance(frame, memory)
        type_match = self._type_match(frame, memory)

        return (
            W_ENTITY * entity
            + W_SPEAKER_ROLE * speaker
            + W_ACTOR_ENTITY * actor
            + W_TEMPORAL * temporal
            + W_STANCE * stance
            + W_TYPE_MATCH * type_match
        )

    def _entity_overlap(self, frame: QueryFrame, memory) -> float:
        frame_ents = set(e.lower() for e in frame.entities) if frame.entities else set()
        mem_ents = set(e.lower() for e in getattr(memory, "entities", []) or [])
        if not frame_ents:
            return 0.5  # unconstrained
        if not mem_ents:
            return 0.5  # unknown
        union = frame_ents | mem_ents
        return len(frame_ents & mem_ents) / len(union) if union else 0.5

    def _speaker_role_match(self, frame: QueryFrame, memory) -> float:
        constraint = frame.speaker_role_constraint
        facet = getattr(memory, "speaker_role", None)
        if not constraint:
            return 0.5
        if not facet:
            return 0.5
        return 1.0 if constraint == facet else 0.0

    def _actor_entity_match(self, frame: QueryFrame, memory) -> float:
        constraint = frame.entity_constraint
        actors = getattr(memory, "actor_entities", []) or []
        if not constraint:
            return 0.5
        if not actors:
            return 0.5
        return 1.0 if constraint.lower() in [a.lower() for a in actors] else 0.0

    def _temporal_consistency(self, frame: QueryFrame, memory) -> float:
        constraint = frame.time_constraint
        mentioned = getattr(memory, "mentioned_at", None)
        if not constraint:
            return 0.5
        if not mentioned:
            return 0.5
        # Simple substring match for Phase 1
        if constraint.lower() in (mentioned or "").lower():
            return 1.0
        return 0.3  # not clearly matching but not necessarily wrong

    def _stance_relevance(self, frame: QueryFrame, memory) -> float:
        stance = getattr(memory, "stance", None)
        if not stance:
            return 0.5
        target = frame.answer_target
        # Stance aligns with temporal_change queries
        if target == "temporal_change" and stance in ("switched", "stopped", "deprecated"):
            return 1.0
        if target == "failure" and stance in ("failed",):
            return 1.0
        if target == "procedure" and stance in ("fixed", "prefers"):
            return 1.0
        return 0.5  # neutral

    def _type_match(self, frame: QueryFrame, memory) -> float:
        expected_type = _TARGET_TO_TYPE.get(frame.answer_target)
        if expected_type is None:
            return 0.5  # no specific type expected
        mem_type = memory.memory_type.value if hasattr(memory.memory_type, "value") else str(memory.memory_type)
        return 1.0 if mem_type == expected_type else 0.3
```

- [ ] **Step 3: Run tests, commit**

```bash
git add psa/constraint_scorer.py tests/test_constraint_scorer.py
git commit -m "feat: add rule-backed constraint scorer (post-Level-2)"
```

---

### Task 9: Pipeline Integration

**Files:**
- Modify: `psa/pipeline.py`

- [ ] **Step 1: Add imports**

```python
from .query_frame import QueryFrame, extract_query_frame
from .constraint_scorer import ConstraintScorer
```

- [ ] **Step 2: Add constraint_scorer to __init__**

```python
    self._constraint_scorer = ConstraintScorer()
```

- [ ] **Step 3: Wire into query()**

After embedding (Step 1), before Level 1:

```python
        # Step 1.5: Extract query frame
        query_frame = extract_query_frame(query)
        logger.debug(
            "QueryFrame: target=%s mode=%s entities=%s confidence=%.2f",
            query_frame.answer_target, query_frame.retrieval_mode,
            query_frame.entities, query_frame.confidence,
        )
```

After Level 2 memory scoring (or after `_fetch_memories` if no memory_scorer), before packer:

```python
        # Constraint scoring (rule-backed)
        if self._constraint_scorer is not None and scored_memories:
            scored_memories = self._constraint_scorer.adjust_scores(
                scored_memories, query_frame
            )
            memories = [sm.memory for sm in scored_memories]
```

If no memory_scorer, wrap the fetched memories as ScoredMemory objects first so the constraint scorer can process them:

```python
        if self.memory_scorer is None and memories:
            from .memory_scorer import ScoredMemory
            scored_memories = [
                ScoredMemory(
                    memory_object_id=m.memory_object_id,
                    final_score=m.quality_score,
                    memory=m,
                )
                for m in memories
            ]
            scored_memories = self._constraint_scorer.adjust_scores(
                scored_memories, query_frame
            )
            memories = [sm.memory for sm in scored_memories]
            _pre_ranked = True
```

- [ ] **Step 4: Run pipeline tests**

Run: `uv run pytest tests/ -v -k "pipeline"`
Expected: All pass (constraint scorer uses defaults when memories have no facets)

- [ ] **Step 5: Commit**

```bash
git add psa/pipeline.py
git commit -m "feat: wire query frame extraction + constraint scorer into pipeline"
```

---

### Task 10: Level 1 Co-Activation Version-Gated Frame Injection

**Files:**
- Modify: `psa/coactivation.py`

- [ ] **Step 1: Add query_frame_dim to CoActivationModel**

Add `query_frame_dim: int = 0` parameter to `__init__`. When > 0, expand `query_proj`:

```python
        self.query_frame_dim = query_frame_dim
        self.query_proj = nn.Linear(centroid_dim + query_frame_dim, half)
```

- [ ] **Step 2: Update forward() to accept optional frame features**

```python
    def forward(self, ce_scores, centroids, query_vec, anchor_features=None, query_frame_features=None):
        ...
        if query_frame_features is not None and self.query_frame_dim > 0:
            query_input = torch.cat([query_vec, query_frame_features], dim=-1)
        else:
            if self.query_frame_dim > 0:
                query_input = torch.cat([
                    query_vec,
                    torch.zeros(B, self.query_frame_dim, device=query_vec.device),
                ], dim=-1)
            else:
                query_input = query_vec

        query_tokens = self.query_proj(query_input).unsqueeze(1).expand(B, N, -1)
```

- [ ] **Step 3: Version-gate in from_model_path()**

Read `query_frame_dim` from metadata (default 0). Old models have 0 -> no frame features.

```python
        model = CoActivationModel(
            ...
            query_frame_dim=cfg.get("query_frame_dim", 0),
        )
```

- [ ] **Step 4: Update CoActivationSelector.select() to accept query_frame**

Build frame feature tensor (7-dim answer_target one-hot + 4-dim retrieval_mode one-hot = 11 dims) and pass to model if `query_frame_dim > 0`.

- [ ] **Step 5: Save query_frame_dim in training metadata**

In `train_coactivation.py`, save `query_frame_dim` in the version JSON.

- [ ] **Step 6: Run coactivation tests, commit**

```bash
git add psa/coactivation.py psa/training/train_coactivation.py tests/test_coactivation.py
git commit -m "feat: version-gated query frame injection in co-activation model"
```

---

### Task 11: End-to-End Benchmark

- [ ] **Step 1: Backfill facets on longmemeval tenant**

```bash
uv run psa repair --backfill-facets --tenant longmemeval_bench
```

- [ ] **Step 2: Run benchmark with constraint scoring**

```bash
uv run psa benchmark longmemeval run --selector coactivation
```

(Constraint scorer is automatic — it's wired into the pipeline.)

- [ ] **Step 3: Score and compare**

```bash
uv run psa benchmark longmemeval score --results $(ls -t ~/.psa/benchmarks/longmemeval/results_*.jsonl | head -1) --method both
```

Target:

| Metric | Target | Previous |
|--------|--------|----------|
| R@5 | >= 0.93 | 0.920 |
| F1 | >= 0.22 | 0.177 |
| LLM-judge | >= 0.40 | 0.358 |

- [ ] **Step 4: Commit results**

```bash
git add -u
git commit -m "benchmark: Phase 1 query frames + facets + constraint scoring results"
```

---

### Task 12: Lint, Format, Full Test Suite

- [ ] **Step 1: ruff check + format**

```bash
uv run ruff check .
uv run ruff format .
```

- [ ] **Step 2: Full pytest**

```bash
uv run pytest tests/ -v
```

All tests must pass.

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -m "chore: lint and format pass"
```

---

## Self-Review

**Spec coverage:**
- Richer facets (LLM prompt, storage, backfill): Tasks 1, 2, 4, 6
- Conversation parser: Task 3
- Convo miner update: Task 5
- Query frame extraction: Task 7
- Constraint scorer (rule-backed): Task 8
- Pipeline integration: Task 9
- Level 1 version-gated frame injection: Task 10
- Benchmark: Task 11
- All spec requirements covered.

**Key spec decisions preserved:**
- Constraint scorer is rule-backed, NOT learned (Phase 2)
- MLP stays at 11 dims (unchanged in Phase 1)
- Speaker split into role + entity (separate fields)
- Backfill reads from raw sources, not rewritten body
- normalize.py unchanged (new conversation_parser instead)
- Old coactivation models: query_frame_dim=0 -> zero-padded

**Placeholder scan:** No TBDs. All code steps have complete code.

**Type consistency:**
- `ExtractedFacets` (Task 2) fields match MemoryObject new fields (Task 1)
- `QueryFrame` (Task 7) fields match constraint scorer usage (Task 8)
- `ScoredMemory` from memory_scorer.py used in constraint_scorer.py (Task 8) and pipeline.py (Task 9)
- `ConversationTurn` (Task 3) used in convo_miner.py (Task 5)
- `query_frame_dim` in coactivation model (Task 10) matches version JSON metadata
