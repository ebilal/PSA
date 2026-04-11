"""
migrate.py — Non-destructive ChromaDB palace → PSA migration.

Reads all drawers from an existing ChromaDB collection and creates
corresponding MemoryObjects in the PSA MemoryStore, re-embedding
each with BAAI/bge-base-en-v1.5.

Invariants:
  - Never deletes or modifies ChromaDB data.
  - source_ids on every MemoryObject point back to the original drawer ID.
  - If a MemoryObject with the same source_id already exists, it is skipped.
  - Reports counts: total, migrated, skipped (already exists), failed.

Usage::

    from psa.migrate import migrate_palace_to_psa

    stats = migrate_palace_to_psa(
        chroma_path="~/.psa/palace",
        collection_name="mempalace",
        tenant_id="default",
    )
    print(f"Migrated {stats['migrated']} drawers.")
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .embeddings import EmbeddingModel
from .memory_object import MemoryObject, MemoryStore, MemoryType
from .tenant import TenantManager

logger = logging.getLogger("psa.migrate")

# ── Room → MemoryType mapping ─────────────────────────────────────────────────

# Heuristic map from legacy MemPalace "room" names to PSA MemoryType.
# Unknown rooms fall back to SEMANTIC.
_ROOM_TO_MEMORY_TYPE: Dict[str, MemoryType] = {
    "episodic": MemoryType.EPISODIC,
    "episodes": MemoryType.EPISODIC,
    "experience": MemoryType.EPISODIC,
    "semantic": MemoryType.SEMANTIC,
    "facts": MemoryType.SEMANTIC,
    "concepts": MemoryType.SEMANTIC,
    "procedural": MemoryType.PROCEDURAL,
    "procedures": MemoryType.PROCEDURAL,
    "how_to": MemoryType.PROCEDURAL,
    "failure": MemoryType.FAILURE,
    "failures": MemoryType.FAILURE,
    "errors": MemoryType.FAILURE,
    "tool_use": MemoryType.TOOL_USE,
    "tools": MemoryType.TOOL_USE,
    "tool": MemoryType.TOOL_USE,
    "working": MemoryType.WORKING_DERIVATIVE,
    "scratch": MemoryType.WORKING_DERIVATIVE,
    "draft": MemoryType.WORKING_DERIVATIVE,
}


# ── MigrationStats ────────────────────────────────────────────────────────────


@dataclass
class MigrationStats:
    """Summary of a migration run."""

    total: int = 0
    migrated: int = 0
    skipped: int = 0  # already exists in MemoryStore
    failed: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "migrated": self.migrated,
            "skipped": self.skipped,
            "failed": self.failed,
            "errors": self.errors[:10],  # cap error messages for readability
        }


# ── Public API ────────────────────────────────────────────────────────────────


def migrate_palace_to_psa(
    chroma_path: str,
    collection_name: str = "mempalace",
    tenant_id: str = "default",
    base_dir: Optional[str] = None,
    batch_size: int = 64,
) -> MigrationStats:
    """
    Migrate all drawers from a ChromaDB palace collection to PSA MemoryStore.

    Parameters
    ----------
    chroma_path:
        Path to the ChromaDB persistence directory (e.g. ~/.psa/palace).
    collection_name:
        ChromaDB collection name (default: "mempalace").
    tenant_id:
        PSA tenant to write into (default: "default").
    base_dir:
        PSA base directory (default: ~/.psa).
    batch_size:
        Number of drawers to fetch from ChromaDB per batch.

    Returns
    -------
    MigrationStats with total/migrated/skipped/failed counts.
    """
    stats = MigrationStats()

    # Load ChromaDB
    try:
        import chromadb
    except ImportError:
        raise ImportError(
            "chromadb is required for palace migration. Install it with: pip install chromadb"
        )

    chroma_path = os.path.expanduser(chroma_path)
    if not os.path.exists(chroma_path):
        raise FileNotFoundError(
            f"ChromaDB palace not found at '{chroma_path}'. Check the path and try again."
        )

    client = chromadb.PersistentClient(path=chroma_path)
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        raise ValueError(
            f"Collection '{collection_name}' not found in ChromaDB at '{chroma_path}'. "
            "Available: " + str([c.name for c in client.list_collections()])
        )

    # Set up PSA store
    tm = TenantManager(base_dir=base_dir)
    tenant = tm.get_or_create(tenant_id)
    store = MemoryStore(db_path=tenant.memory_db_path)
    embedding_model = EmbeddingModel()

    # Get total count
    total = collection.count()
    stats.total = total
    logger.info(
        "Migrating %d drawers from '%s' → PSA tenant '%s'", total, collection_name, tenant_id
    )

    if total == 0:
        return stats

    # Fetch in batches
    offset = 0
    while offset < total:
        try:
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.error("ChromaDB fetch failed at offset %d: %s", offset, e)
            stats.failed += min(batch_size, total - offset)
            stats.errors.append(f"Fetch error at offset {offset}: {e}")
            offset += batch_size
            continue

        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        for drawer_id, doc, meta in zip(ids, documents, metadatas):
            stats.total = total  # already set but keep consistent
            try:
                _migrate_drawer(
                    drawer_id=drawer_id,
                    document=doc or "",
                    metadata=meta or {},
                    store=store,
                    embedding_model=embedding_model,
                    tenant_id=tenant_id,
                    stats=stats,
                )
            except Exception as e:
                logger.warning("Failed to migrate drawer %s: %s", drawer_id, e)
                stats.failed += 1
                stats.errors.append(f"drawer {drawer_id}: {e}")

        offset += batch_size

    logger.info(
        "Migration complete: %d migrated, %d skipped, %d failed (total %d)",
        stats.migrated,
        stats.skipped,
        stats.failed,
        stats.total,
    )
    return stats


# ── Internal helpers ──────────────────────────────────────────────────────────


def _migrate_drawer(
    drawer_id: str,
    document: str,
    metadata: dict,
    store: MemoryStore,
    embedding_model: EmbeddingModel,
    tenant_id: str,
    stats: MigrationStats,
) -> None:
    """Migrate a single ChromaDB drawer to PSA MemoryObject."""
    # Check if already migrated (source_id match)
    existing = store.get_by_source_id(drawer_id)
    if existing is not None:
        stats.skipped += 1
        return

    # Infer MemoryType from room metadata
    room = (metadata.get("room") or metadata.get("type") or "").lower()
    memory_type = _ROOM_TO_MEMORY_TYPE.get(room, MemoryType.SEMANTIC)

    # Build title from metadata or first line of document
    title = (
        metadata.get("title")
        or metadata.get("name")
        or (document[:80].split("\n")[0].strip() if document else "Untitled")
    )

    # Embed the document
    embedding = embedding_model.embed(document) if document else []

    # Build and store MemoryObject
    mo = MemoryObject.create(
        tenant_id=tenant_id,
        memory_type=memory_type,
        title=title,
        body=document,
        summary=document[:256] if document else "",
        source_ids=[drawer_id],
        classification_reason=f"migrated from palace room={room or 'unknown'}",
        quality_score=float(metadata.get("quality_score", 0.5)),
    )

    mo.embedding = embedding
    store.add(mo)
    stats.migrated += 1


# ── Convenience: get ChromaDB path from config ────────────────────────────────


def get_palace_chroma_path() -> Optional[str]:
    """Return the ChromaDB palace path from PSA config, if configured."""
    try:
        from .config import MempalaceConfig

        cfg = MempalaceConfig()
        palace = getattr(cfg, "palace_path", None) or os.path.expanduser("~/.psa/palace")
        return palace if os.path.exists(palace) else None
    except Exception:
        return None
