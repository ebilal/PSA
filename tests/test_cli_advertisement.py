"""Tests for psa advertisement CLI subcommands."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys


def _seed(tmp_path):
    from psa.advertisement.ledger import create_schema, upsert_ledger
    db_path = tmp_path / ".psa" / "tenants" / "default" / "memory.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as db:
        create_schema(db)
        upsert_ledger(db, "pid-1", 1, "alpha", 2.5, 2.0, grace_days=21)
        upsert_ledger(db, "pid-2", 1, "beta", -0.3, -0.5, grace_days=21)


def test_status_prints_distribution(tmp_path, monkeypatch):
    _seed(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    env = {**os.environ, "HOME": str(tmp_path)}
    r = subprocess.run(
        [sys.executable, "-m", "psa", "advertisement", "status", "--json"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"
    data = json.loads(r.stdout)
    assert data["n_active"] == 2
    assert "histogram" in data
