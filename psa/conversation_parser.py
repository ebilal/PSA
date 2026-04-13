"""
conversation_parser.py — Parse conversation files into structured turns.

Unlike normalize.py (which returns flat transcript strings), this preserves
role, text, and timestamp metadata per turn.

Supported formats:
    - Claude Code JSONL (.jsonl)
    - Claude AI JSON export (.json)
    - Plain text with > markers (.txt and all other extensions)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ConversationTurn:
    role: str  # "user", "assistant", "system"
    text: str
    timestamp: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


def parse_conversation(path: str) -> List[ConversationTurn]:
    """
    Parse a conversation file and return a list of ConversationTurn objects.

    Returns an empty list for missing or empty files.
    """
    p = Path(path)
    if not p.exists():
        return []

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if not content.strip():
        return []

    ext = p.suffix.lower()
    if ext == ".jsonl":
        return _parse_jsonl(content)
    elif ext == ".json":
        return _parse_json(content)
    else:
        return _parse_plain_text(content)


# ---------------------------------------------------------------------------
# Format parsers
# ---------------------------------------------------------------------------


def _parse_jsonl(content: str) -> List[ConversationTurn]:
    """Parse newline-delimited JSON, one object per line."""
    turns: List[ConversationTurn] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        turn = _obj_to_turn(obj)
        if turn is not None:
            turns.append(turn)
    return turns


def _parse_json(content: str) -> List[ConversationTurn]:
    """Parse a JSON file that is either a list of message objects or a dict
    with a 'messages' / 'conversation' key."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return _parse_plain_text(content)

    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict):
        messages = data.get("messages") or data.get("conversation") or []
    else:
        return []

    turns: List[ConversationTurn] = []
    for obj in messages:
        if not isinstance(obj, dict):
            continue
        turn = _obj_to_turn(obj)
        if turn is not None:
            turns.append(turn)
    return turns


def _obj_to_turn(obj: dict) -> Optional[ConversationTurn]:
    """Convert a JSON message object to a ConversationTurn, handling both
    string content and Claude Code list-of-blocks content."""
    role = obj.get("role", "")
    if not role:
        return None

    raw_content = obj.get("content", "")
    if isinstance(raw_content, str):
        text = raw_content
    elif isinstance(raw_content, list):
        # Claude Code block list: [{"type": "text", "text": "..."}, ...]
        parts = []
        for block in raw_content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            elif isinstance(block, str):
                parts.append(block)
        text = "\n".join(p for p in parts if p)
    else:
        text = str(raw_content)

    timestamp = obj.get("timestamp") or obj.get("created_at") or None

    # Carry any extra keys as metadata (excluding role, content, timestamp)
    skip = {"role", "content", "timestamp", "created_at"}
    metadata = {k: v for k, v in obj.items() if k not in skip}

    return ConversationTurn(role=role, text=text, timestamp=timestamp, metadata=metadata)


def _parse_plain_text(content: str) -> List[ConversationTurn]:
    """Parse plain text with '> ' user markers.

    Lines starting with '> ' belong to the user.  Everything else is
    the assistant.  Blank lines separate turns; consecutive lines of the
    same role are joined together.
    """
    turns: List[ConversationTurn] = []
    current_role: Optional[str] = None
    current_lines: List[str] = []

    def flush():
        nonlocal current_role, current_lines
        if current_role is not None and any(ln.strip() for ln in current_lines):
            text = "\n".join(current_lines).strip()
            if text:
                turns.append(ConversationTurn(role=current_role, text=text))
        current_role = None
        current_lines = []

    for line in content.splitlines():
        if line.startswith("> "):
            # User line
            user_text = line[2:]
            if current_role == "user":
                current_lines.append(user_text)
            else:
                flush()
                current_role = "user"
                current_lines = [user_text]
        elif line.strip() == "":
            # Blank line — potential role boundary; don't flush yet, keep
            # collecting so multi-paragraph turns are preserved, but mark
            # that we need a separator.
            current_lines.append("")
        else:
            # Assistant / continuation line
            if current_role == "assistant":
                current_lines.append(line)
            else:
                flush()
                current_role = "assistant"
                current_lines = [line]

    flush()
    return turns
