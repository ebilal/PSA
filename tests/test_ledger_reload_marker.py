"""Tests for reload marker file used by long-lived pipelines."""

from __future__ import annotations

import json
import time


def test_write_reload_marker_creates_file(tmp_path, monkeypatch):
    from psa.advertisement.reload import write_reload_marker

    monkeypatch.setenv("HOME", str(tmp_path))
    write_reload_marker(tenant_id="default", changed_anchor_ids=[1, 2, 3])

    marker = tmp_path / ".psa" / "tenants" / "default" / "atlas_reload_requested"
    assert marker.exists()
    body = json.loads(marker.read_text())
    assert set(body["changed_anchor_ids"]) == {1, 2, 3}
    assert "timestamp" in body


def test_marker_should_reload(tmp_path, monkeypatch):
    from psa.advertisement.reload import write_reload_marker, should_reload

    monkeypatch.setenv("HOME", str(tmp_path))
    assert should_reload(tenant_id="default", last_reload_mtime=0.0) is False
    write_reload_marker(tenant_id="default", changed_anchor_ids=[1])
    time.sleep(0.01)
    assert should_reload(tenant_id="default", last_reload_mtime=0.0) is True


def test_should_reload_respects_last_reload_mtime(tmp_path, monkeypatch):
    from psa.advertisement.reload import write_reload_marker, should_reload, marker_mtime

    monkeypatch.setenv("HOME", str(tmp_path))
    write_reload_marker(tenant_id="default", changed_anchor_ids=[1])
    m = marker_mtime(tenant_id="default")
    # Same mtime → no reload needed
    assert should_reload(tenant_id="default", last_reload_mtime=m) is False
