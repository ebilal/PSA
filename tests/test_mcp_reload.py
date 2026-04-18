"""MCP server honors the reload marker between queries."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_mcp_calls_reload_atlas_when_marker_advances(tmp_path, monkeypatch):
    from psa.mcp_server import maybe_reload_pipeline

    monkeypatch.setenv("HOME", str(tmp_path))
    pipeline = MagicMock()
    state = {"last_reload_mtime": 0.0}

    from psa.advertisement.reload import write_reload_marker
    write_reload_marker(tenant_id="default", changed_anchor_ids=[1])

    maybe_reload_pipeline(pipeline=pipeline, state=state, tenant_id="default")
    pipeline.reload_atlas.assert_called_once()
    assert state["last_reload_mtime"] > 0.0


def test_mcp_does_not_reload_when_marker_unchanged(tmp_path, monkeypatch):
    from psa.mcp_server import maybe_reload_pipeline
    from psa.advertisement.reload import write_reload_marker, marker_mtime

    monkeypatch.setenv("HOME", str(tmp_path))
    write_reload_marker(tenant_id="default", changed_anchor_ids=[1])
    m = marker_mtime(tenant_id="default")

    pipeline = MagicMock()
    state = {"last_reload_mtime": m}
    maybe_reload_pipeline(pipeline=pipeline, state=state, tenant_id="default")
    pipeline.reload_atlas.assert_not_called()


def test_mcp_does_not_reload_when_no_marker(tmp_path, monkeypatch):
    """Marker absent → no reload."""
    from psa.mcp_server import maybe_reload_pipeline

    monkeypatch.setenv("HOME", str(tmp_path))
    pipeline = MagicMock()
    state = {"last_reload_mtime": 0.0}
    maybe_reload_pipeline(pipeline=pipeline, state=state, tenant_id="default")
    pipeline.reload_atlas.assert_not_called()
