"""
conftest.py — Shared fixtures for PSA tests.

Provides isolated palace instances so tests never touch the user's real
data or leak temp files on failure.

HOME is redirected to a temp directory at module load time — before any
psa imports — so module-level initialisations write to a throwaway
location instead of the real user profile.
"""

import os
import shutil
import tempfile

# ── Isolate HOME before any psa imports ──────────────────────────
_original_env = {}
_session_tmp = tempfile.mkdtemp(prefix="psa_session_")

# Preserve the HuggingFace model cache so embedding tests can use
# locally-cached models even after HOME is redirected.
_real_home = os.path.expanduser("~")
os.environ.setdefault("HF_HOME", os.path.join(_real_home, ".cache", "huggingface"))

for _var in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH"):
    _original_env[_var] = os.environ.get(_var)

os.environ["HOME"] = _session_tmp
os.environ["USERPROFILE"] = _session_tmp
os.environ["HOMEDRIVE"] = os.path.splitdrive(_session_tmp)[0] or "C:"
os.environ["HOMEPATH"] = os.path.splitdrive(_session_tmp)[1] or _session_tmp

# Now it is safe to import psa modules that trigger initialisation.
import chromadb  # noqa: E402
import pytest  # noqa: E402

from psa.config import MempalaceConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_mcp_cache():
    """Reset the MCP server's cached ChromaDB client/collection between tests."""

    def _clear_cache():
        try:
            from psa import mcp_server

            mcp_server._client_cache = None
            mcp_server._collection_cache = None
        except (ImportError, AttributeError):
            pass

    _clear_cache()
    yield
    _clear_cache()


@pytest.fixture(autouse=True)
def _mock_qwen_card_generation_in_tests(monkeypatch):
    """Use stub cards in tests to avoid slow Qwen HTTP calls during atlas builds."""
    from psa import atlas as _atlas_mod
    from psa.anchor import AnchorCard

    def _stub_card(anchor_id, centroid, sample_memories, is_novelty=False):
        titles = [m.title for m in sample_memories[:3]]
        return AnchorCard(
            anchor_id=anchor_id,
            name=f"novelty-{anchor_id}" if is_novelty else f"cluster-{anchor_id}",
            meaning=f"Test cluster with {len(sample_memories)} memories: {'; '.join(titles[:2])}",
            memory_types=list({m.memory_type.value for m in sample_memories[:5]}),
            include_terms=[],
            exclude_terms=[],
            prototype_examples=titles,
            near_but_different=[],
            centroid=centroid,
            memory_count=len(sample_memories),
            is_novelty=is_novelty,
        )

    monkeypatch.setattr(_atlas_mod, "_generate_card_via_qwen", _stub_card)


@pytest.fixture(scope="session", autouse=True)
def _isolate_home():
    """Ensure HOME points to a temp dir for the entire test session.

    The env vars were already set at module level (above) so that
    module-level initialisations are captured.  This fixture simply
    restores the originals on teardown and cleans up the temp dir.
    """
    yield
    for var, orig in _original_env.items():
        if orig is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = orig
    shutil.rmtree(_session_tmp, ignore_errors=True)


@pytest.fixture
def tmp_dir():
    """Create and auto-cleanup a temporary directory."""
    d = tempfile.mkdtemp(prefix="psa_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def palace_path(tmp_dir):
    """Path to an empty palace directory inside tmp_dir."""
    p = os.path.join(tmp_dir, "palace")
    os.makedirs(p)
    return p


@pytest.fixture
def config(tmp_dir, palace_path):
    """A MempalaceConfig pointing at the temp palace."""
    cfg_dir = os.path.join(tmp_dir, "config")
    os.makedirs(cfg_dir)
    import json

    with open(os.path.join(cfg_dir, "config.json"), "w") as f:
        json.dump({"palace_path": palace_path}, f)
    return MempalaceConfig(config_dir=cfg_dir)


@pytest.fixture
def collection(palace_path):
    """A ChromaDB collection pre-seeded in the temp palace."""
    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_or_create_collection("psa_drawers")
    return col


@pytest.fixture
def seeded_collection(collection):
    """Collection with a handful of representative drawers."""
    collection.add(
        ids=[
            "drawer_proj_backend_aaa",
            "drawer_proj_backend_bbb",
            "drawer_proj_frontend_ccc",
            "drawer_notes_planning_ddd",
        ],
        documents=[
            "The authentication module uses JWT tokens for session management. "
            "Tokens expire after 24 hours. Refresh tokens are stored in HttpOnly cookies.",
            "Database migrations are handled by Alembic. We use PostgreSQL 15 "
            "with connection pooling via pgbouncer.",
            "The React frontend uses TanStack Query for server state management. "
            "All API calls go through a centralized fetch wrapper.",
            "Sprint planning: migrate auth to passkeys by Q3. "
            "Evaluate ChromaDB alternatives for vector search.",
        ],
        metadatas=[
            {
                "wing": "project",
                "room": "backend",
                "source_file": "auth.py",
                "chunk_index": 0,
                "added_by": "miner",
                "filed_at": "2026-01-01T00:00:00",
            },
            {
                "wing": "project",
                "room": "backend",
                "source_file": "db.py",
                "chunk_index": 0,
                "added_by": "miner",
                "filed_at": "2026-01-02T00:00:00",
            },
            {
                "wing": "project",
                "room": "frontend",
                "source_file": "App.tsx",
                "chunk_index": 0,
                "added_by": "miner",
                "filed_at": "2026-01-03T00:00:00",
            },
            {
                "wing": "notes",
                "room": "planning",
                "source_file": "sprint.md",
                "chunk_index": 0,
                "added_by": "miner",
                "filed_at": "2026-01-04T00:00:00",
            },
        ],
    )
    return collection
