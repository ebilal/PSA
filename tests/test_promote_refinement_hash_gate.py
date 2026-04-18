"""Hash gate at promotion prevents stale candidates from overwriting refined."""

from __future__ import annotations

import json
import os
import subprocess
import sys


def _setup_tenant(tmp_path, monkeypatch, version=1):
    """Create ~/.psa/tenants/default/atlas_v<version>/ under tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    atlas_dir = tmp_path / ".psa" / "tenants" / "default" / f"atlas_v{version}"
    atlas_dir.mkdir(parents=True)
    # AtlasManager.latest_version() requires atlas_meta.json to recognise the dir.
    (atlas_dir / "atlas_meta.json").write_text(
        '{"version": ' + str(version) + ', "tenant_id": "default"}'
    )
    return atlas_dir


def _run_promote(tmp_path):
    env = {
        **os.environ,
        "HOME": str(tmp_path),
    }
    return subprocess.run(
        [sys.executable, "-m", "psa", "atlas", "promote-refinement"],
        env=env,
        capture_output=True,
        text=True,
    )


def test_promote_refuses_when_refined_hash_mismatches(tmp_path, monkeypatch):
    """Candidate meta records hash X, but refined now has hash Y → refused."""
    atlas_dir = _setup_tenant(tmp_path, monkeypatch)
    refined = atlas_dir / "anchor_cards_refined.json"
    refined.write_text('{"state": "A"}')  # hash X
    candidate = atlas_dir / "anchor_cards_candidate.json"
    candidate.write_text('{"state": "candidate"}')

    # Stamp the meta against the current (state=A) refined contents.
    from psa.advertisement.writer import stamp_refined_hash
    meta = {"source": "decay", "tenant_id": "default", "atlas_version": 1}
    stamp_refined_hash(meta, atlas_dir)
    (atlas_dir / "anchor_cards_candidate.meta.json").write_text(json.dumps(meta))

    # Simulate stage 2 (or another candidate) mutating refined after candidate
    # was generated. The recorded hash now does not match.
    refined.write_text('{"state": "B"}')

    result = _run_promote(tmp_path)
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "refined cards changed" in combined or "hash" in combined


def test_promote_succeeds_when_refined_hash_matches(tmp_path, monkeypatch):
    atlas_dir = _setup_tenant(tmp_path, monkeypatch)
    refined = atlas_dir / "anchor_cards_refined.json"
    refined.write_text('{"state": "A"}')
    candidate = atlas_dir / "anchor_cards_candidate.json"
    candidate.write_text('{"state": "candidate"}')

    from psa.advertisement.writer import stamp_refined_hash
    meta = {"source": "decay", "tenant_id": "default", "atlas_version": 1}
    stamp_refined_hash(meta, atlas_dir)
    (atlas_dir / "anchor_cards_candidate.meta.json").write_text(json.dumps(meta))
    # Refined is unchanged — hash still matches.

    result = _run_promote(tmp_path)
    assert result.returncode == 0, f"stdout={result.stdout} stderr={result.stderr}"


def test_promote_succeeds_on_first_run_when_refined_absent_and_stays_absent(
    tmp_path, monkeypatch
):
    """First-time promotion: no refined file at generation OR promotion."""
    atlas_dir = _setup_tenant(tmp_path, monkeypatch)
    candidate = atlas_dir / "anchor_cards_candidate.json"
    candidate.write_text('{"state": "candidate"}')

    from psa.advertisement.writer import stamp_refined_hash
    meta = {"source": "decay", "tenant_id": "default", "atlas_version": 1}
    stamp_refined_hash(meta, atlas_dir)
    # At stamp time, refined doesn't exist → refined_existed_at_generation=False
    (atlas_dir / "anchor_cards_candidate.meta.json").write_text(json.dumps(meta))

    result = _run_promote(tmp_path)
    assert result.returncode == 0, f"stdout={result.stdout} stderr={result.stderr}"


def test_promote_refuses_when_refined_absent_at_generation_then_created(
    tmp_path, monkeypatch
):
    """Meta says refined_existed=False, but a refined file exists now (stage 2
    created it in the interim)."""
    atlas_dir = _setup_tenant(tmp_path, monkeypatch)
    candidate = atlas_dir / "anchor_cards_candidate.json"
    candidate.write_text('{"state": "candidate"}')

    from psa.advertisement.writer import stamp_refined_hash
    meta = {"source": "decay", "tenant_id": "default", "atlas_version": 1}
    stamp_refined_hash(meta, atlas_dir)
    (atlas_dir / "anchor_cards_candidate.meta.json").write_text(json.dumps(meta))
    # Simulate stage 2 creating a refined file after candidate was generated.
    (atlas_dir / "anchor_cards_refined.json").write_text('{"new": true}')

    result = _run_promote(tmp_path)
    assert result.returncode != 0


def test_promote_refuses_legacy_candidate_without_hash_field(tmp_path, monkeypatch):
    """Legacy candidate meta predates the hash gate — no refined_hash field."""
    atlas_dir = _setup_tenant(tmp_path, monkeypatch)
    refined = atlas_dir / "anchor_cards_refined.json"
    refined.write_text('{"state": "A"}')
    candidate = atlas_dir / "anchor_cards_candidate.json"
    candidate.write_text('{"state": "candidate"}')
    # Legacy meta shape — no refined_hash_at_generation field.
    legacy_meta = {"source": "decay", "tenant_id": "default", "atlas_version": 1}
    (atlas_dir / "anchor_cards_candidate.meta.json").write_text(json.dumps(legacy_meta))

    result = _run_promote(tmp_path)
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "missing" in combined or "legacy" in combined or "regenerate" in combined
