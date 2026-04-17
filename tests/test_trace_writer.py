"""Tests for psa.trace — per-query JSONL writer."""

from __future__ import annotations

import json


def test_write_trace_appends_record(tmp_path, monkeypatch):
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PSA_TRACE", raising=False)

    record = {"run_id": "r1", "query": "hello", "tenant_id": "default"}
    write_trace(record, tenant_id="default")

    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    assert path.exists()
    line = path.read_text().strip()
    assert json.loads(line) == record


def test_write_trace_appends_multiple_records(tmp_path, monkeypatch):
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PSA_TRACE", raising=False)

    write_trace({"run_id": "r1", "tenant_id": "default"}, tenant_id="default")
    write_trace({"run_id": "r2", "tenant_id": "default"}, tenant_id="default")

    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["run_id"] == "r1"
    assert json.loads(lines[1])["run_id"] == "r2"


def test_write_trace_disabled_by_env(tmp_path, monkeypatch):
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PSA_TRACE", "0")

    write_trace({"run_id": "r1", "tenant_id": "default"}, tenant_id="default")

    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    assert not path.exists()


def test_write_trace_disabled_by_config(tmp_path, monkeypatch):
    """When MempalaceConfig.trace_queries is False, write_trace no-ops."""
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PSA_TRACE", raising=False)

    # Write a config file with trace_queries=False.
    cfg_dir = tmp_path / ".psa"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.json").write_text('{"trace_queries": false}')

    write_trace({"run_id": "r1", "tenant_id": "default"}, tenant_id="default")

    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    assert not path.exists()


def test_write_trace_enabled_by_default(tmp_path, monkeypatch):
    """No env flag, no config — write happens."""
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PSA_TRACE", raising=False)

    write_trace({"run_id": "r1", "tenant_id": "default"}, tenant_id="default")

    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    assert path.exists()


def test_write_trace_env_wins_over_config(tmp_path, monkeypatch):
    """PSA_TRACE=1 overrides config trace_queries=False."""
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PSA_TRACE", "1")

    cfg_dir = tmp_path / ".psa"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.json").write_text('{"trace_queries": false}')

    write_trace({"run_id": "r1", "tenant_id": "default"}, tenant_id="default")

    path = tmp_path / ".psa" / "tenants" / "default" / "query_trace.jsonl"
    assert path.exists()
