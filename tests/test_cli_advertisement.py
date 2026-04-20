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


def test_diff_prints_counterfactual(tmp_path, monkeypatch):
    """B and A policies disagree when shadow_consecutive ≥ 7 but cnc < 14."""
    import sqlite3
    from psa.advertisement.ledger import create_schema
    db_path = tmp_path / ".psa" / "tenants" / "default" / "memory.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as db:
        create_schema(db)
        db.execute(
            """
            INSERT INTO pattern_ledger
                (pattern_id, anchor_id, pattern_text,
                 ledger, consecutive_negative_cycles,
                 shadow_ledger, shadow_consecutive,
                 grace_expires_at, created_at, last_updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "pid-1", 1, "would-A-removes",
                -0.5, 8, -0.5, 10,
                "2000-01-01T00:00:00+00:00",
                "2025-01-01T00:00:00+00:00",
                "2025-01-01T00:00:00+00:00",
            ),
        )
        db.commit()
    monkeypatch.setenv("HOME", str(tmp_path))
    env = {**os.environ, "HOME": str(tmp_path)}
    r = subprocess.run(
        [sys.executable, "-m", "psa", "advertisement", "diff", "--json"],
        capture_output=True, text=True,
        env=env,
    )
    assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"
    data = json.loads(r.stdout)
    assert data["shadow_only_would_remove"] == 1


def test_rebuild_ledger_recomputes_from_trace(tmp_path, monkeypatch):
    """rebuild-ledger walks trace + recomputes ledger state."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Write a minimal trace with retrieval_attribution
    trace_dir = tmp_path / ".psa" / "tenants" / "default"
    trace_dir.mkdir(parents=True)
    record = {
        "timestamp": "2026-04-17T00:00:00+00:00",
        "query": "cat vs dog",
        "selected_anchor_ids": [1],
        "retrieval_attribution": [
            {
                "anchor_id": 1,
                "bm25_argmax_pattern": "cat vs dog behavior",
                "bm25_epsilon_tied": [],
                "bm25_floor_passed": True,
            }
        ],
    }
    (trace_dir / "query_trace.jsonl").write_text(json.dumps(record) + "\n")

    env = {**os.environ, "HOME": str(tmp_path)}
    r = subprocess.run(
        [sys.executable, "-m", "psa", "advertisement", "rebuild-ledger", "--dry-run", "--json"],
        capture_output=True, text=True,
        env=env,
    )
    assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"
    data = json.loads(r.stdout)
    assert data["records_processed"] == 1
    assert data["derived_patterns"] >= 1


def test_purge_deletes_archived_rows_past_retention(tmp_path, monkeypatch):
    import sqlite3
    from datetime import datetime, timedelta, timezone
    from psa.advertisement.ledger import create_schema

    db_path = tmp_path / ".psa" / "tenants" / "default" / "memory.sqlite3"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as db:
        create_schema(db)
        old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        recent = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        for pid, removed_at in [("old", old), ("recent", recent), ("active", None)]:
            db.execute(
                """
                INSERT INTO pattern_ledger
                    (pattern_id, anchor_id, pattern_text, ledger, shadow_ledger,
                     grace_expires_at, created_at, last_updated_at, removed_at)
                VALUES (?, ?, ?, 0.0, 0.0, '', '', '', ?)
                """,
                (pid, 1, "p", removed_at),
            )
        db.commit()

    monkeypatch.setenv("HOME", str(tmp_path))
    env = {**os.environ, "HOME": str(tmp_path)}
    r = subprocess.run(
        [sys.executable, "-m", "psa", "advertisement", "purge", "--older-than-days", "90"],
        capture_output=True, text=True,
        env=env,
    )
    assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"

    with sqlite3.connect(db_path) as db:
        remaining = {row[0] for row in db.execute("SELECT pattern_id FROM pattern_ledger")}
    assert "old" not in remaining
    assert "recent" in remaining
    assert "active" in remaining


def test_status_handles_missing_pattern_ledger_table(tmp_path, monkeypatch):
    """User has memory.sqlite3 but no pattern_ledger table → no crash."""
    monkeypatch.setenv("HOME", str(tmp_path))
    db_path = tmp_path / ".psa" / "tenants" / "default" / "memory.sqlite3"
    db_path.parent.mkdir(parents=True)
    # Create an empty SQLite file with no pattern_ledger table.
    sqlite3.connect(db_path).close()

    env = {**os.environ, "HOME": str(tmp_path)}
    r = subprocess.run(
        [sys.executable, "-m", "psa", "advertisement", "status", "--json"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"
    data = json.loads(r.stdout)
    assert data["n_active"] == 0


def test_diff_handles_missing_pattern_ledger_table(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    db_path = tmp_path / ".psa" / "tenants" / "default" / "memory.sqlite3"
    db_path.parent.mkdir(parents=True)
    sqlite3.connect(db_path).close()

    env = {**os.environ, "HOME": str(tmp_path)}
    r = subprocess.run(
        [sys.executable, "-m", "psa", "advertisement", "diff", "--json"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["disagreements"] == []


def test_purge_handles_missing_pattern_ledger_table(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    db_path = tmp_path / ".psa" / "tenants" / "default" / "memory.sqlite3"
    db_path.parent.mkdir(parents=True)
    sqlite3.connect(db_path).close()

    env = {**os.environ, "HOME": str(tmp_path)}
    r = subprocess.run(
        [sys.executable, "-m", "psa", "advertisement", "purge", "--json"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["deleted"] == 0


def test_atlas_help_does_not_expose_decay_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    env = {**os.environ, "HOME": str(tmp_path)}
    r = subprocess.run(
        [sys.executable, "-m", "psa", "atlas", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"
    assert "decay" not in r.stdout
