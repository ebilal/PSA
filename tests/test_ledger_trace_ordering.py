"""Tests for trace-first-ledger-second semantics in pipeline.compose_and_record."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_ledger_not_written_when_trace_fails(tmp_path, monkeypatch):
    """If trace.write_trace returns False, record_query_signals is not called."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PSA_TRACE", "0")  # force trace to return False

    from psa.pipeline import compose_and_record

    recorded = {"count": 0}

    def fake_record(**kw):
        recorded["count"] += 1

    compose_and_record(
        tenant_id="default",
        trace_record={"run_id": "r", "timestamp": "t", "query": "q"},
        attribution=[],
        selected_anchor_ids=set(),
        config=MagicMock(tracking_enabled=True),
        record_signals_fn=fake_record,
    )
    # ledger write skipped because trace was disabled
    assert recorded["count"] == 0


def test_ledger_written_when_trace_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # trace enabled by default

    from psa.pipeline import compose_and_record

    recorded = {"count": 0}

    def fake_record(**kw):
        recorded["count"] += 1

    compose_and_record(
        tenant_id="default",
        trace_record={"run_id": "r", "timestamp": "t", "query": "q"},
        attribution=[],
        selected_anchor_ids=set(),
        config=MagicMock(tracking_enabled=True),
        record_signals_fn=fake_record,
    )
    assert recorded["count"] == 1


def test_ledger_not_written_when_tracking_disabled(tmp_path, monkeypatch):
    """Even if trace succeeds, skip ledger when tracking_enabled=False."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from psa.pipeline import compose_and_record

    recorded = {"count": 0}

    def fake_record(**kw):
        recorded["count"] += 1

    compose_and_record(
        tenant_id="default",
        trace_record={"run_id": "r", "timestamp": "t", "query": "q"},
        attribution=[],
        selected_anchor_ids=set(),
        config=MagicMock(tracking_enabled=False),
        record_signals_fn=fake_record,
    )
    assert recorded["count"] == 0


def test_compose_populates_retrieval_attribution_field(tmp_path, monkeypatch):
    """compose_and_record should add retrieval_attribution to the trace record."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from psa.pipeline import compose_and_record
    from psa.advertisement.ledger import AnchorAttribution

    trace_record = {"run_id": "r", "timestamp": "t", "query": "q"}
    attribution = [
        AnchorAttribution(
            anchor_id=5,
            argmax_pattern="pattern alpha",
            eps_tied_patterns=["pattern beta"],
            credited=["pattern alpha", "pattern beta"],
            bm25_floor_passed=True,
        ),
        AnchorAttribution(
            anchor_id=9,
            argmax_pattern=None,
            bm25_floor_passed=False,
        ),
    ]

    compose_and_record(
        tenant_id="default",
        trace_record=trace_record,
        attribution=attribution,
        selected_anchor_ids={5},
        config=MagicMock(tracking_enabled=False),
        record_signals_fn=lambda **kw: None,
    )
    assert "retrieval_attribution" in trace_record
    assert len(trace_record["retrieval_attribution"]) == 2
    first = trace_record["retrieval_attribution"][0]
    assert first["anchor_id"] == 5
    assert first["bm25_argmax_pattern"] == "pattern alpha"
    assert first["bm25_epsilon_tied"] == ["pattern beta"]
    assert first["bm25_floor_passed"] is True
    second = trace_record["retrieval_attribution"][1]
    assert second["anchor_id"] == 9
    assert second["bm25_argmax_pattern"] is None
    assert second["bm25_floor_passed"] is False


def test_compose_and_record_writes_ledger_when_attribution_present(tmp_path, monkeypatch):
    """Regression for the Level 1 bug: when attribution is non-empty AND
    tracking_enabled is True AND trace succeeds, the ledger writer must fire
    with the attribution payload (not skip it)."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from psa.pipeline import compose_and_record
    from psa.advertisement.ledger import AnchorAttribution

    captured = {}

    def fake_record(**kw):
        captured["attribution"] = kw["attribution"]
        captured["selected_anchor_ids"] = kw["selected_anchor_ids"]

    attribution = [
        AnchorAttribution(
            anchor_id=1,
            argmax_pattern="some pattern",
            credited=["some pattern"],
            bm25_floor_passed=True,
        )
    ]
    compose_and_record(
        tenant_id="default",
        trace_record={"run_id": "r", "timestamp": "t", "query": "q"},
        attribution=attribution,
        selected_anchor_ids={1},
        config=MagicMock(tracking_enabled=True),
        record_signals_fn=fake_record,
    )
    assert captured.get("attribution") == attribution
    assert captured.get("selected_anchor_ids") == {1}
