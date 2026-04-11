"""Tests for psa.migrate — migrate_palace_to_psa (non-destructive ChromaDB → PSA)."""

from unittest.mock import MagicMock, patch

import pytest

from psa.migrate import (
    MigrationStats,
    _migrate_drawer,
    _ROOM_TO_MEMORY_TYPE,
    migrate_palace_to_psa,
)
from psa.memory_object import MemoryObject, MemoryStore, MemoryType


# ── MigrationStats ────────────────────────────────────────────────────────────


def test_migration_stats_defaults():
    stats = MigrationStats()
    assert stats.total == 0
    assert stats.migrated == 0
    assert stats.skipped == 0
    assert stats.failed == 0
    assert stats.errors == []


def test_migration_stats_to_dict():
    stats = MigrationStats(total=10, migrated=8, skipped=1, failed=1, errors=["oops"])
    d = stats.to_dict()
    assert d["total"] == 10
    assert d["migrated"] == 8
    assert d["skipped"] == 1
    assert d["failed"] == 1
    assert "oops" in d["errors"]


def test_migration_stats_errors_capped():
    stats = MigrationStats(errors=[f"err{i}" for i in range(20)])
    d = stats.to_dict()
    assert len(d["errors"]) == 10


# ── _ROOM_TO_MEMORY_TYPE mapping ──────────────────────────────────────────────


def test_room_to_memory_type_episodic():
    assert _ROOM_TO_MEMORY_TYPE["episodic"] == MemoryType.EPISODIC
    assert _ROOM_TO_MEMORY_TYPE["episodes"] == MemoryType.EPISODIC


def test_room_to_memory_type_procedural():
    assert _ROOM_TO_MEMORY_TYPE["procedural"] == MemoryType.PROCEDURAL


def test_room_to_memory_type_failure():
    assert _ROOM_TO_MEMORY_TYPE["failure"] == MemoryType.FAILURE
    assert _ROOM_TO_MEMORY_TYPE["errors"] == MemoryType.FAILURE


def test_room_to_memory_type_tool_use():
    assert _ROOM_TO_MEMORY_TYPE["tool_use"] == MemoryType.TOOL_USE


def test_room_to_memory_type_working():
    assert _ROOM_TO_MEMORY_TYPE["working"] == MemoryType.WORKING_DERIVATIVE


# ── _migrate_drawer ───────────────────────────────────────────────────────────


def _make_store_mock(already_exists=False):
    store = MagicMock(spec=MemoryStore)
    store.get_by_source_id.return_value = MagicMock() if already_exists else None
    store.add.return_value = None
    return store


def _make_embedding_model_mock():
    em = MagicMock()
    em.embed.return_value = [0.1] * 768
    return em


def test_migrate_drawer_new(tmp_path):
    store = _make_store_mock(already_exists=False)
    em = _make_embedding_model_mock()
    stats = MigrationStats()

    _migrate_drawer(
        drawer_id="d1",
        document="This is a test document.",
        metadata={"room": "episodic", "wing": "project_x"},
        store=store,
        embedding_model=em,
        tenant_id="default",
        stats=stats,
    )

    assert stats.migrated == 1
    assert stats.skipped == 0
    store.add.assert_called_once()


def test_migrate_drawer_already_exists():
    store = _make_store_mock(already_exists=True)
    em = _make_embedding_model_mock()
    stats = MigrationStats()

    _migrate_drawer(
        drawer_id="d-existing",
        document="Content",
        metadata={},
        store=store,
        embedding_model=em,
        tenant_id="default",
        stats=stats,
    )

    assert stats.skipped == 1
    assert stats.migrated == 0
    store.add.assert_not_called()


def test_migrate_drawer_infers_memory_type():
    """Room name should be mapped to the correct MemoryType."""
    store = _make_store_mock(already_exists=False)
    em = _make_embedding_model_mock()
    stats = MigrationStats()

    captured_mo = []

    def capture_add(mo, embedding=None):
        captured_mo.append(mo)

    store.add.side_effect = capture_add

    _migrate_drawer(
        drawer_id="d2",
        document="Procedure for deployment",
        metadata={"room": "procedural"},
        store=store,
        embedding_model=em,
        tenant_id="default",
        stats=stats,
    )

    assert len(captured_mo) == 1
    assert captured_mo[0].memory_type == MemoryType.PROCEDURAL


def test_migrate_drawer_unknown_room_defaults_to_semantic():
    store = _make_store_mock(already_exists=False)
    em = _make_embedding_model_mock()
    stats = MigrationStats()

    captured_mo = []
    store.add.side_effect = lambda mo, embedding=None: captured_mo.append(mo)

    _migrate_drawer(
        drawer_id="d3",
        document="Some content",
        metadata={"room": "unknownroom"},
        store=store,
        embedding_model=em,
        tenant_id="default",
        stats=stats,
    )

    assert captured_mo[0].memory_type == MemoryType.SEMANTIC


def test_migrate_drawer_title_from_metadata():
    store = _make_store_mock(already_exists=False)
    em = _make_embedding_model_mock()
    stats = MigrationStats()

    captured_mo = []
    store.add.side_effect = lambda mo, embedding=None: captured_mo.append(mo)

    _migrate_drawer(
        drawer_id="d4",
        document="Body text",
        metadata={"title": "Custom Title", "room": "semantic"},
        store=store,
        embedding_model=em,
        tenant_id="default",
        stats=stats,
    )

    assert captured_mo[0].title == "Custom Title"


def test_migrate_drawer_source_id_recorded():
    store = _make_store_mock(already_exists=False)
    em = _make_embedding_model_mock()
    stats = MigrationStats()

    captured_mo = []
    store.add.side_effect = lambda mo, embedding=None: captured_mo.append(mo)

    _migrate_drawer(
        drawer_id="source-abc",
        document="Content",
        metadata={},
        store=store,
        embedding_model=em,
        tenant_id="default",
        stats=stats,
    )

    assert "source-abc" in captured_mo[0].source_ids


# ── migrate_palace_to_psa (integration-style with mocks) ─────────────────────


def test_migrate_raises_if_chromadb_missing(tmp_path):
    with patch.dict("sys.modules", {"chromadb": None}):
        with pytest.raises(ImportError, match="chromadb is required"):
            migrate_palace_to_psa(
                chroma_path=str(tmp_path / "palace"),
                tenant_id="default",
            )


def test_migrate_raises_if_palace_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        migrate_palace_to_psa(
            chroma_path=str(tmp_path / "nonexistent"),
            tenant_id="default",
        )


def test_migrate_empty_collection(tmp_path):
    """If collection has 0 drawers, return stats with total=0."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection

    palace_path = tmp_path / "palace"
    palace_path.mkdir()

    mock_tenant = MagicMock()
    mock_tenant.memory_db_path = str(tmp_path / "memory.sqlite3")

    import chromadb as _chromadb

    with (
        patch.object(_chromadb, "PersistentClient", return_value=mock_client),
        patch("psa.migrate.TenantManager") as mock_tm,
        patch("psa.migrate.MemoryStore"),
        patch("psa.migrate.EmbeddingModel"),
    ):
        mock_tm.return_value.get_or_create.return_value = mock_tenant

        stats = migrate_palace_to_psa(
            chroma_path=str(palace_path),
            tenant_id="default",
        )

    assert stats.total == 0
    assert stats.migrated == 0


def test_migrate_collection_not_found(tmp_path):
    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("not found")
    mock_client.list_collections.return_value = []

    palace_path = tmp_path / "palace"
    palace_path.mkdir()

    import chromadb as _chromadb

    with patch.object(_chromadb, "PersistentClient", return_value=mock_client):
        with pytest.raises(ValueError, match="not found"):
            migrate_palace_to_psa(
                chroma_path=str(palace_path),
                collection_name="missing_collection",
                tenant_id="default",
            )


# ── get_by_source_id (MemoryStore) ────────────────────────────────────────────


def test_memory_store_get_by_source_id(tmp_path):
    """Verify get_by_source_id round-trips via the SQLite json_each query."""
    from psa.memory_object import MemoryStore, MemoryType

    db_path = str(tmp_path / "mem.sqlite3")
    store = MemoryStore(db_path=db_path)

    mo = MemoryObject.create(
        tenant_id="t",
        memory_type=MemoryType.SEMANTIC,
        title="Test",
        body="Body",
        summary="Sum",
        source_ids=["my-source-id-001"],
        classification_reason="test",
        quality_score=0.8,
    )
    mo.embedding = []
    store.add(mo)

    found = store.get_by_source_id("my-source-id-001")
    assert found is not None
    assert found.memory_object_id == mo.memory_object_id

    not_found = store.get_by_source_id("nonexistent-source")
    assert not_found is None
