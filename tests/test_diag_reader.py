"""Tests for psa.diag.trace_reader — origins contract."""

from __future__ import annotations

import json
from pathlib import Path


def _write_trace(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_reader_origins_none_returns_all_records(tmp_path, monkeypatch):
    from psa.diag.trace_reader import iter_trace_records

    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    _write_trace(path, [
        {"run_id": "r1", "query_origin": "interactive"},
        {"run_id": "r2", "query_origin": "benchmark"},
        {"run_id": "r3", "query_origin": "labeling"},
    ])

    records = list(iter_trace_records("default", origins=None))
    assert [r["run_id"] for r in records] == ["r1", "r2", "r3"]


def test_reader_origins_interactive_filters(tmp_path, monkeypatch):
    from psa.diag.trace_reader import iter_trace_records

    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    _write_trace(path, [
        {"run_id": "r1", "query_origin": "interactive"},
        {"run_id": "r2", "query_origin": "benchmark"},
    ])

    records = list(iter_trace_records("default", origins={"interactive"}))
    assert [r["run_id"] for r in records] == ["r1"]


def test_reader_empty_origins_returns_nothing(tmp_path, monkeypatch):
    """Explicit filter-everything-out: origins=set() yields zero records."""
    from psa.diag.trace_reader import iter_trace_records

    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    _write_trace(path, [
        {"run_id": "r1", "query_origin": "interactive"},
    ])

    records = list(iter_trace_records("default", origins=set()))
    assert records == []


def test_reader_missing_file_yields_nothing(tmp_path, monkeypatch):
    from psa.diag.trace_reader import iter_trace_records

    monkeypatch.setenv("HOME", str(tmp_path))
    records = list(iter_trace_records("does_not_exist", origins=None))
    assert records == []


def test_reader_skips_malformed_lines(tmp_path, monkeypatch):
    from psa.diag.trace_reader import iter_trace_records

    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"run_id": "r1", "query_origin": "interactive"}\n'
        'not-json\n'
        '{"run_id": "r2", "query_origin": "interactive"}\n'
    )
    records = list(iter_trace_records("default", origins=None))
    assert [r["run_id"] for r in records] == ["r1", "r2"]
