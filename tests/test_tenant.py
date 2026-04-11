"""
test_tenant.py — Tests for psa.tenant.TenantManager and Tenant.

Covers:
- Path scoping (each tenant gets its own directory)
- Default tenant creation
- get / get_or_create / exists / list
- Invalid tenant_id rejection
- Subdirectory structure (atlas/, models/)
- Memory db path helper
"""

import os

import pytest

from psa.tenant import TenantManager


@pytest.fixture
def mgr(tmp_dir):
    return TenantManager(base_dir=os.path.join(tmp_dir, "tenants"))


# ── create ───────────────────────────────────────────────────────────────────


def test_create_makes_directory(mgr, tmp_dir):
    tenant = mgr.create("alice")
    assert os.path.isdir(tenant.root_dir)


def test_create_makes_subdirs(mgr):
    tenant = mgr.create("bob")
    assert os.path.isdir(tenant.atlas_dir)
    assert os.path.isdir(tenant.models_dir)


def test_create_is_idempotent(mgr):
    mgr.create("carol")
    tenant2 = mgr.create("carol")  # should not raise
    assert tenant2.tenant_id == "carol"


def test_create_sets_display_name(mgr):
    tenant = mgr.create("dave", display_name="Dave's Workspace")
    assert tenant.display_name == "Dave's Workspace"


def test_create_display_name_defaults_to_id(mgr):
    tenant = mgr.create("eve")
    assert tenant.display_name == "eve"


# ── get ──────────────────────────────────────────────────────────────────────


def test_get_existing(mgr):
    mgr.create("frank")
    tenant = mgr.get("frank")
    assert tenant is not None
    assert tenant.tenant_id == "frank"


def test_get_nonexistent_returns_none(mgr):
    assert mgr.get("nobody") is None


# ── get_or_create ────────────────────────────────────────────────────────────


def test_get_or_create_existing(mgr):
    mgr.create("grace")
    tenant = mgr.get_or_create("grace")
    assert tenant.tenant_id == "grace"


def test_get_or_create_new(mgr):
    tenant = mgr.get_or_create("henry")
    assert os.path.isdir(tenant.root_dir)


# ── get_default ──────────────────────────────────────────────────────────────


def test_get_default(mgr):
    tenant = mgr.get_default()
    assert tenant.tenant_id == TenantManager.DEFAULT_TENANT_ID
    assert os.path.isdir(tenant.root_dir)


# ── exists ───────────────────────────────────────────────────────────────────


def test_exists_after_create(mgr):
    mgr.create("iris")
    assert mgr.exists("iris") is True


def test_not_exists_before_create(mgr):
    assert mgr.exists("jack") is False


def test_exists_invalid_id_returns_false(mgr):
    assert mgr.exists("INVALID ID!") is False


# ── list ─────────────────────────────────────────────────────────────────────


def test_list_empty(mgr):
    assert mgr.list() == []


def test_list_after_creates(mgr):
    mgr.create("kate")
    mgr.create("leo")
    ids = [t.tenant_id for t in mgr.list()]
    assert "kate" in ids
    assert "leo" in ids


def test_list_sorted(mgr):
    mgr.create("zara")
    mgr.create("amir")
    mgr.create("marco")
    ids = [t.tenant_id for t in mgr.list()]
    assert ids == sorted(ids)


# ── path helpers ─────────────────────────────────────────────────────────────


def test_memory_db_path(mgr):
    tenant = mgr.create("nina")
    assert tenant.memory_db_path.endswith("memory.sqlite3")
    assert tenant.memory_db_path.startswith(tenant.root_dir)


def test_tenant_paths_scoped(mgr):
    t1 = mgr.create("omar")
    t2 = mgr.create("petra")
    assert t1.root_dir != t2.root_dir
    assert t1.memory_db_path != t2.memory_db_path


# ── validation ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_id",
    [
        "",  # empty
        "A-Capital",  # uppercase
        "has space",  # space
        "has!bang",  # special char
        "a" * 65,  # too long
    ],
)
def test_invalid_tenant_id_raises(mgr, bad_id):
    with pytest.raises(ValueError, match="Invalid tenant_id"):
        mgr.create(bad_id)


@pytest.mark.parametrize(
    "good_id",
    ["default", "user-123", "tenant_a", "abc", "a1b2c3", "a-b_c"],
)
def test_valid_tenant_ids_accepted(mgr, good_id):
    tenant = mgr.create(good_id)
    assert tenant.tenant_id == good_id
