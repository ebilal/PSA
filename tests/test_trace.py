"""Tests for psa.trace — bool return + retrieval_attribution field."""

from __future__ import annotations


def test_write_trace_returns_true_on_success(tmp_path, monkeypatch):
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    ok = write_trace({"query": "hello"}, tenant_id="default")
    assert ok is True


def test_write_trace_returns_false_when_disabled(tmp_path, monkeypatch):
    from psa.trace import write_trace

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PSA_TRACE", "0")
    ok = write_trace({"query": "hello"}, tenant_id="default")
    assert ok is False


def test_new_trace_record_includes_retrieval_attribution_field():
    from psa.trace import new_trace_record

    r = new_trace_record(
        run_id="r",
        timestamp="2026-04-17T00:00:00+00:00",
        tenant_id="default",
        atlas_version=1,
        query="q",
    )
    assert "retrieval_attribution" in r
    assert r["retrieval_attribution"] == []
