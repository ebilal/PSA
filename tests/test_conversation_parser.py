"""Tests for conversation_parser.py."""

import json
import os
import tempfile

from psa.conversation_parser import parse_conversation


def test_parse_claude_code_jsonl():
    """Parse a Claude Code JSONL file with 2 turns."""
    lines = [
        json.dumps(
            {
                "role": "user",
                "content": "Fix the auth bug",
                "timestamp": "2026-04-10T10:00:00Z",
            }
        ),
        json.dumps(
            {
                "role": "assistant",
                "content": "I'll check jwt.py",
                "timestamp": "2026-04-10T10:00:05Z",
            }
        ),
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(lines))
        path = f.name
    try:
        turns = parse_conversation(path)
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[0].text == "Fix the auth bug"
        assert turns[0].timestamp == "2026-04-10T10:00:00Z"
        assert turns[1].role == "assistant"
        assert turns[1].text == "I'll check jwt.py"
        assert turns[1].timestamp == "2026-04-10T10:00:05Z"
    finally:
        os.unlink(path)


def test_parse_plain_text():
    """Parse a plain text file with > user markers."""
    content = "> Can you help?\n\nSure, I'll look.\n\n> What about tokens?\n\nThey're in cookies."
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = f.name
    try:
        turns = parse_conversation(path)
        assert len(turns) >= 2
        assert turns[0].role == "user"
    finally:
        os.unlink(path)


def test_returns_empty_for_empty_file():
    """Empty file returns empty list."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        path = f.name
    try:
        turns = parse_conversation(path)
        assert turns == []
    finally:
        os.unlink(path)
