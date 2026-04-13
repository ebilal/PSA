"""
memory_object.py — Core memory object schema and SQLite-backed store for PSA.

Memory objects are derived from raw traces and organized into the Persistent
Semantic Atlas. Raw source records are immutable; memory objects are typed
indexes built on top of them.
"""

import json
import os
import sqlite3
import struct
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional


class MemoryType(str, Enum):
    """The six memory types in the Persistent Semantic Atlas."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    FAILURE = "failure"
    TOOL_USE = "tool_use"
    WORKING_DERIVATIVE = "working_derivative"


@dataclass
class EvidenceSpan:
    """A reference to a specific span within a raw source record."""

    source_id: str
    start_offset: int
    end_offset: int
    chunk_id: Optional[str] = None


@dataclass
class MemoryObject:
    """
    A typed, reusable memory derived from raw experience.

    Memory objects are derived indexes — raw source records are always preserved.
    Every object retains source_ids and evidence_spans so it can be traced back
    to its origin.
    """

    memory_object_id: str
    tenant_id: str
    memory_type: MemoryType
    title: str
    body: str
    summary: str
    source_ids: List[str]  # links to raw source records
    evidence_chunk_ids: List[str]  # supporting chunk IDs for important claims
    evidence_spans: List[EvidenceSpan]  # exact raw text positions
    classification_reason: str  # why this memory type was assigned
    created_at: str
    updated_at: str
    # Embedding stored separately; embedding_ref is the row ID in the embeddings table
    embedding: Optional[List[float]] = None
    embedding_ref: Optional[str] = None
    # Atlas assignment (filled after induction)
    primary_anchor_id: Optional[int] = None
    secondary_anchor_ids: List[int] = field(default_factory=list)
    assignment_confidence: float = 0.0
    # Metadata
    task_type: Optional[str] = None
    tool_names: List[str] = field(default_factory=list)
    success_label: Optional[bool] = None
    quality_score: float = 0.0
    validity_interval: Optional[str] = None
    acl_scope: Optional[str] = None
    # Deduplication
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    # Usage telemetry (populated by query pipeline)
    select_count: int = 0
    pack_count: int = 0
    last_selected: Optional[str] = None
    last_packed: Optional[str] = None
    # Lifecycle (populated by forgetting system)
    is_archived: bool = False
    archived_at: Optional[str] = None
    # Retrieval facets (populated during consolidation or backfill)
    entities: List[str] = field(default_factory=list)
    actor_entities: List[str] = field(default_factory=list)
    speaker_role: Optional[str] = None
    stance: Optional[str] = None
    mentioned_at: Optional[str] = None

    @classmethod
    def create(
        cls,
        tenant_id: str,
        memory_type: MemoryType,
        title: str,
        body: str,
        summary: str,
        source_ids: List[str],
        classification_reason: str,
        evidence_chunk_ids: Optional[List[str]] = None,
        evidence_spans: Optional[List[EvidenceSpan]] = None,
        **kwargs,
    ) -> "MemoryObject":
        """Create a new memory object with auto-generated ID and timestamps."""
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            memory_object_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            memory_type=memory_type,
            title=title,
            body=body,
            summary=summary,
            source_ids=source_ids,
            evidence_chunk_ids=evidence_chunk_ids or [],
            evidence_spans=evidence_spans or [],
            classification_reason=classification_reason,
            created_at=now,
            updated_at=now,
            **kwargs,
        )


@dataclass
class RawSource:
    """An immutable raw source record. Never modified after creation."""

    source_id: str
    tenant_id: str
    source_type: str  # "project_file", "conversation", "tool_log", "manual"
    source_path: Optional[str]
    title: str
    full_text: str
    created_at: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        tenant_id: str,
        source_type: str,
        full_text: str,
        title: str = "",
        source_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> "RawSource":
        return cls(
            source_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            source_type=source_type,
            source_path=source_path,
            title=title or (os.path.basename(source_path) if source_path else "untitled"),
            full_text=full_text,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )


class MemoryStore:
    """
    SQLite-backed store for memory objects and raw source records.

    Raw source records are immutable. Memory objects are derived indexes.
    Embeddings are stored as binary blobs (numpy float32, L2-normalized).
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS raw_sources (
        source_id     TEXT PRIMARY KEY,
        tenant_id     TEXT NOT NULL,
        source_type   TEXT NOT NULL,
        source_path   TEXT,
        title         TEXT NOT NULL,
        full_text     TEXT NOT NULL,
        created_at    TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS memory_objects (
        memory_object_id      TEXT PRIMARY KEY,
        tenant_id             TEXT NOT NULL,
        memory_type           TEXT NOT NULL,
        title                 TEXT NOT NULL,
        body                  TEXT NOT NULL,
        summary               TEXT NOT NULL,
        source_ids_json       TEXT NOT NULL DEFAULT '[]',
        evidence_chunk_ids_json TEXT NOT NULL DEFAULT '[]',
        evidence_spans_json   TEXT NOT NULL DEFAULT '[]',
        classification_reason TEXT NOT NULL DEFAULT '',
        created_at            TEXT NOT NULL,
        updated_at            TEXT NOT NULL,
        embedding_blob        BLOB,
        embedding_dim         INTEGER,
        embedding_ref         TEXT,
        primary_anchor_id     INTEGER,
        secondary_anchor_ids_json TEXT NOT NULL DEFAULT '[]',
        assignment_confidence REAL NOT NULL DEFAULT 0.0,
        task_type             TEXT,
        tool_names_json       TEXT NOT NULL DEFAULT '[]',
        success_label         INTEGER,
        quality_score         REAL NOT NULL DEFAULT 0.0,
        validity_interval     TEXT,
        acl_scope             TEXT,
        is_duplicate          INTEGER NOT NULL DEFAULT 0,
        duplicate_of          TEXT,
        select_count          INTEGER NOT NULL DEFAULT 0,
        pack_count            INTEGER NOT NULL DEFAULT 0,
        last_selected         TEXT,
        last_packed           TEXT,
        is_archived           INTEGER NOT NULL DEFAULT 0,
        archived_at           TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_mo_tenant     ON memory_objects(tenant_id);
    CREATE INDEX IF NOT EXISTS idx_mo_type       ON memory_objects(memory_type);
    CREATE INDEX IF NOT EXISTS idx_mo_anchor     ON memory_objects(primary_anchor_id);
    CREATE INDEX IF NOT EXISTS idx_mo_created    ON memory_objects(created_at);
    CREATE INDEX IF NOT EXISTS idx_src_tenant    ON raw_sources(tenant_id);
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)
            # Migrate existing databases: add new columns if missing
            for col, typedef in [
                ("select_count", "INTEGER NOT NULL DEFAULT 0"),
                ("pack_count", "INTEGER NOT NULL DEFAULT 0"),
                ("last_selected", "TEXT"),
                ("last_packed", "TEXT"),
                ("is_archived", "INTEGER NOT NULL DEFAULT 0"),
                ("archived_at", "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE memory_objects ADD COLUMN {col} {typedef}")
                except sqlite3.OperationalError:
                    pass  # column already exists
            # Create index on is_archived after migration ensures column exists
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mo_archived ON memory_objects(is_archived)"
            )
            # Phase 1 facet columns
            for col, typedef in [
                ("entities_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("actor_entities_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("speaker_role", "TEXT"),
                ("stance", "TEXT"),
                ("mentioned_at", "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE memory_objects ADD COLUMN {col} {typedef}")
                except sqlite3.OperationalError:
                    pass  # column already exists

    # ── Raw source operations (immutable) ────────────────────────────────────

    def add_source(self, source: RawSource) -> str:
        """Persist a raw source record. Returns source_id."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO raw_sources
                  (source_id, tenant_id, source_type, source_path, title, full_text,
                   created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source.source_id,
                    source.tenant_id,
                    source.source_type,
                    source.source_path,
                    source.title,
                    source.full_text,
                    source.created_at,
                    json.dumps(source.metadata),
                ),
            )
        return source.source_id

    def get_source(self, source_id: str) -> Optional[RawSource]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM raw_sources WHERE source_id = ?", (source_id,)
            ).fetchone()
        if not row:
            return None
        return RawSource(
            source_id=row["source_id"],
            tenant_id=row["tenant_id"],
            source_type=row["source_type"],
            source_path=row["source_path"],
            title=row["title"],
            full_text=row["full_text"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata_json"]),
        )

    # ── Memory object operations ─────────────────────────────────────────────

    @staticmethod
    def _encode_embedding(embedding: Optional[List[float]]) -> Optional[bytes]:
        if embedding is None:
            return None
        return struct.pack(f"{len(embedding)}f", *embedding)

    @staticmethod
    def _decode_embedding(blob: Optional[bytes], dim: Optional[int]) -> Optional[List[float]]:
        if blob is None or dim is None:
            return None
        return list(struct.unpack(f"{dim}f", blob))

    def add(self, mo: MemoryObject) -> str:
        """Persist a memory object. Returns memory_object_id."""
        blob = self._encode_embedding(mo.embedding)
        dim = len(mo.embedding) if mo.embedding else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_objects
                  (memory_object_id, tenant_id, memory_type, title, body, summary,
                   source_ids_json, evidence_chunk_ids_json, evidence_spans_json,
                   classification_reason, created_at, updated_at,
                   embedding_blob, embedding_dim, embedding_ref,
                   primary_anchor_id, secondary_anchor_ids_json, assignment_confidence,
                   task_type, tool_names_json, success_label, quality_score,
                   validity_interval, acl_scope, is_duplicate, duplicate_of,
                   select_count, pack_count, last_selected, last_packed,
                   is_archived, archived_at,
                   entities_json, actor_entities_json, speaker_role, stance, mentioned_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mo.memory_object_id,
                    mo.tenant_id,
                    mo.memory_type.value,
                    mo.title,
                    mo.body,
                    mo.summary,
                    json.dumps(mo.source_ids),
                    json.dumps(mo.evidence_chunk_ids),
                    json.dumps([asdict(s) for s in mo.evidence_spans]),
                    mo.classification_reason,
                    mo.created_at,
                    mo.updated_at,
                    blob,
                    dim,
                    mo.embedding_ref,
                    mo.primary_anchor_id,
                    json.dumps(mo.secondary_anchor_ids),
                    mo.assignment_confidence,
                    mo.task_type,
                    json.dumps(mo.tool_names),
                    (1 if mo.success_label else 0) if mo.success_label is not None else None,
                    mo.quality_score,
                    mo.validity_interval,
                    mo.acl_scope,
                    1 if mo.is_duplicate else 0,
                    mo.duplicate_of,
                    mo.select_count,
                    mo.pack_count,
                    mo.last_selected,
                    mo.last_packed,
                    1 if mo.is_archived else 0,
                    mo.archived_at,
                    json.dumps(mo.entities),
                    json.dumps(mo.actor_entities),
                    mo.speaker_role,
                    mo.stance,
                    mo.mentioned_at,
                ),
            )
        return mo.memory_object_id

    _ADD_SQL = """
        INSERT OR REPLACE INTO memory_objects
          (memory_object_id, tenant_id, memory_type, title, body, summary,
           source_ids_json, evidence_chunk_ids_json, evidence_spans_json,
           classification_reason, created_at, updated_at,
           embedding_blob, embedding_dim, embedding_ref,
           primary_anchor_id, secondary_anchor_ids_json, assignment_confidence,
           task_type, tool_names_json, success_label, quality_score,
           validity_interval, acl_scope, is_duplicate, duplicate_of,
           select_count, pack_count, last_selected, last_packed,
           is_archived, archived_at,
           entities_json, actor_entities_json, speaker_role, stance, mentioned_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    def _add_params(self, mo: MemoryObject) -> tuple:
        blob = self._encode_embedding(mo.embedding)
        dim = len(mo.embedding) if mo.embedding else None
        return (
            mo.memory_object_id,
            mo.tenant_id,
            mo.memory_type.value,
            mo.title,
            mo.body,
            mo.summary,
            json.dumps(mo.source_ids),
            json.dumps(mo.evidence_chunk_ids),
            json.dumps([asdict(s) for s in mo.evidence_spans]),
            mo.classification_reason,
            mo.created_at,
            mo.updated_at,
            blob,
            dim,
            mo.embedding_ref,
            mo.primary_anchor_id,
            json.dumps(mo.secondary_anchor_ids),
            mo.assignment_confidence,
            mo.task_type,
            json.dumps(mo.tool_names),
            (1 if mo.success_label else 0) if mo.success_label is not None else None,
            mo.quality_score,
            mo.validity_interval,
            mo.acl_scope,
            1 if mo.is_duplicate else 0,
            mo.duplicate_of,
            mo.select_count,
            mo.pack_count,
            mo.last_selected,
            mo.last_packed,
            1 if mo.is_archived else 0,
            mo.archived_at,
            json.dumps(mo.entities),
            json.dumps(mo.actor_entities),
            mo.speaker_role,
            mo.stance,
            mo.mentioned_at,
        )

    def batch_add(self, memories: List[MemoryObject]) -> List[str]:
        """Persist multiple memory objects in a single transaction."""
        conn = self._connect()
        conn.execute("BEGIN")
        try:
            for mo in memories:
                conn.execute(self._ADD_SQL, self._add_params(mo))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return [mo.memory_object_id for mo in memories]

    def get(self, memory_object_id: str) -> Optional[MemoryObject]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_objects WHERE memory_object_id = ?",
                (memory_object_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_memory_object(row)

    def _row_to_memory_object(self, row: sqlite3.Row) -> MemoryObject:
        spans_data = json.loads(row["evidence_spans_json"])
        spans = [EvidenceSpan(**s) for s in spans_data]
        success = row["success_label"]
        d = dict(row)
        return MemoryObject(
            memory_object_id=row["memory_object_id"],
            tenant_id=row["tenant_id"],
            memory_type=MemoryType(row["memory_type"]),
            title=row["title"],
            body=row["body"],
            summary=row["summary"],
            source_ids=json.loads(row["source_ids_json"]),
            evidence_chunk_ids=json.loads(row["evidence_chunk_ids_json"]),
            evidence_spans=spans,
            classification_reason=row["classification_reason"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            embedding=self._decode_embedding(row["embedding_blob"], row["embedding_dim"]),
            embedding_ref=row["embedding_ref"],
            primary_anchor_id=row["primary_anchor_id"],
            secondary_anchor_ids=json.loads(row["secondary_anchor_ids_json"]),
            assignment_confidence=row["assignment_confidence"],
            task_type=row["task_type"],
            tool_names=json.loads(row["tool_names_json"]),
            success_label=(bool(success) if success is not None else None),
            quality_score=row["quality_score"],
            validity_interval=row["validity_interval"],
            acl_scope=row["acl_scope"],
            is_duplicate=bool(row["is_duplicate"]),
            duplicate_of=row["duplicate_of"],
            select_count=row["select_count"] or 0,
            pack_count=row["pack_count"] or 0,
            last_selected=row["last_selected"],
            last_packed=row["last_packed"],
            is_archived=bool(row["is_archived"]),
            archived_at=row["archived_at"],
            entities=json.loads(d.get("entities_json", "[]") or "[]"),
            actor_entities=json.loads(d.get("actor_entities_json", "[]") or "[]"),
            speaker_role=d.get("speaker_role"),
            stance=d.get("stance"),
            mentioned_at=d.get("mentioned_at"),
        )

    def delete(self, memory_object_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM memory_objects WHERE memory_object_id = ?",
                (memory_object_id,),
            )
        return cur.rowcount > 0

    def count(self, tenant_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM memory_objects WHERE tenant_id = ? AND is_archived = 0",
                (tenant_id,),
            ).fetchone()
        return row[0]

    def count_by_anchor(self, tenant_id: str) -> Dict[int, int]:
        """Return {anchor_id: count} for all anchors in a single query."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT primary_anchor_id, COUNT(*) as cnt
                FROM memory_objects
                WHERE tenant_id = ? AND is_duplicate = 0 AND is_archived = 0
                      AND primary_anchor_id IS NOT NULL
                GROUP BY primary_anchor_id
                """,
                (tenant_id,),
            ).fetchall()
        return {row["primary_anchor_id"]: row["cnt"] for row in rows}

    def count_by_type(self, tenant_id: str) -> Dict[str, int]:
        """Return {memory_type: count} in a single query."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_type, COUNT(*) as cnt
                FROM memory_objects
                WHERE tenant_id = ? AND is_duplicate = 0 AND is_archived = 0
                GROUP BY memory_type
                """,
                (tenant_id,),
            ).fetchall()
        return {row["memory_type"]: row["cnt"] for row in rows}

    def query_by_type(
        self,
        tenant_id: str,
        memory_type: MemoryType,
        limit: int = 100,
        exclude_duplicates: bool = True,
    ) -> List[MemoryObject]:
        where = "tenant_id = ? AND memory_type = ? AND is_archived = 0"
        params: list = [tenant_id, memory_type.value]
        if exclude_duplicates:
            where += " AND is_duplicate = 0"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memory_objects WHERE {where} ORDER BY created_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        return [self._row_to_memory_object(r) for r in rows]

    def query_by_anchor(
        self,
        tenant_id: str,
        anchor_id: int,
        limit: int = 50,
    ) -> List[MemoryObject]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_objects
                WHERE tenant_id = ? AND primary_anchor_id = ? AND is_duplicate = 0 AND is_archived = 0
                ORDER BY quality_score DESC, created_at DESC
                LIMIT ?
                """,
                (tenant_id, anchor_id, limit),
            ).fetchall()
        return [self._row_to_memory_object(r) for r in rows]

    def get_all_with_embeddings(self, tenant_id: str) -> List[MemoryObject]:
        """Return all non-duplicate memories that have embeddings (for atlas building)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_objects
                WHERE tenant_id = ? AND embedding_blob IS NOT NULL AND is_duplicate = 0 AND is_archived = 0
                ORDER BY created_at ASC
                """,
                (tenant_id,),
            ).fetchall()
        return [self._row_to_memory_object(r) for r in rows]

    def get_by_source_id(
        self, source_id: str, tenant_id: Optional[str] = None
    ) -> Optional[MemoryObject]:
        """Return the first MemoryObject linked to a given source_id, or None."""
        if tenant_id:
            query = """
                SELECT mo.* FROM memory_objects mo, json_each(mo.source_ids_json) je
                WHERE je.value = ? AND mo.tenant_id = ?
                LIMIT 1
            """
            params = (source_id, tenant_id)
        else:
            query = """
                SELECT mo.* FROM memory_objects mo, json_each(mo.source_ids_json) je
                WHERE je.value = ?
                LIMIT 1
            """
            params = (source_id,)
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return self._row_to_memory_object(row) if row else None

    def get_by_source_session(
        self, session_id: str, tenant_id: Optional[str] = None
    ) -> List[MemoryObject]:
        """
        Return all MemoryObjects whose source records have a source_path containing session_id.

        Used by backtrack_gold_anchors() to find which anchors contain memories
        from a known ground-truth session (e.g. LongMemEval answer_session_ids).
        """
        with self._connect() as conn:
            # Find source_ids for raw_sources whose path contains the session_id
            if tenant_id:
                source_rows = conn.execute(
                    "SELECT source_id FROM raw_sources WHERE source_path LIKE ? AND tenant_id = ?",
                    (f"%{session_id}%", tenant_id),
                ).fetchall()
            else:
                source_rows = conn.execute(
                    "SELECT source_id FROM raw_sources WHERE source_path LIKE ?",
                    (f"%{session_id}%",),
                ).fetchall()

            if not source_rows:
                return []

            source_ids = [r["source_id"] for r in source_rows]

            # Find memory_objects linked to any of those source_ids
            placeholders = ",".join("?" * len(source_ids))
            if tenant_id:
                rows = conn.execute(
                    f"""
                    SELECT DISTINCT mo.* FROM memory_objects mo, json_each(mo.source_ids_json) je
                    WHERE je.value IN ({placeholders}) AND mo.tenant_id = ?
                    """,
                    (*source_ids, tenant_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT DISTINCT mo.* FROM memory_objects mo, json_each(mo.source_ids_json) je
                    WHERE je.value IN ({placeholders})
                    """,
                    source_ids,
                ).fetchall()

        return [self._row_to_memory_object(r) for r in rows]

    def update_anchor_assignment(
        self,
        memory_object_id: str,
        primary_anchor_id: int,
        secondary_anchor_ids: Optional[List[int]] = None,
        confidence: float = 0.0,
    ):
        """Update anchor assignment after atlas induction."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memory_objects
                SET primary_anchor_id = ?, secondary_anchor_ids_json = ?,
                    assignment_confidence = ?, updated_at = ?
                WHERE memory_object_id = ?
                """,
                (
                    primary_anchor_id,
                    json.dumps(secondary_anchor_ids or []),
                    confidence,
                    now,
                    memory_object_id,
                ),
            )

    def batch_update_anchor_assignments(
        self,
        updates: List[dict],
    ):
        """Batch update anchor assignments in a single transaction.

        Each dict in updates must have: memory_object_id, primary_anchor_id,
        secondary_anchor_ids, confidence.
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        conn.execute("BEGIN")
        try:
            for u in updates:
                conn.execute(
                    """
                    UPDATE memory_objects
                    SET primary_anchor_id = ?, secondary_anchor_ids_json = ?,
                        assignment_confidence = ?, updated_at = ?
                    WHERE memory_object_id = ?
                    """,
                    (
                        u["primary_anchor_id"],
                        json.dumps(u.get("secondary_anchor_ids", [])),
                        u.get("confidence", 0.0),
                        now,
                        u["memory_object_id"],
                    ),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    # ── Usage telemetry ─────────────────────────────────────────────────────

    def record_selected(self, memory_object_ids: List[str]):
        """Increment select_count and update last_selected for fetched memories."""
        if not memory_object_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        placeholders = ",".join("?" for _ in memory_object_ids)
        conn.execute(
            f"""
            UPDATE memory_objects
            SET select_count = select_count + 1, last_selected = ?
            WHERE memory_object_id IN ({placeholders})
            """,
            [now] + list(memory_object_ids),
        )
        conn.commit()

    def record_packed(self, memory_object_ids: List[str]):
        """Increment pack_count and update last_packed for memories in context."""
        if not memory_object_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        placeholders = ",".join("?" for _ in memory_object_ids)
        conn.execute(
            f"""
            UPDATE memory_objects
            SET pack_count = pack_count + 1, last_packed = ?
            WHERE memory_object_id IN ({placeholders})
            """,
            [now] + list(memory_object_ids),
        )
        conn.commit()

    # ── Lifecycle / forgetting ──────────────────────────────────────────────

    def query_by_anchor_for_pruning(self, tenant_id: str, anchor_id: int) -> List[MemoryObject]:
        """Return ALL non-archived memories for an anchor, ordered worst-first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_objects
                WHERE tenant_id = ? AND primary_anchor_id = ?
                      AND is_duplicate = 0 AND is_archived = 0
                ORDER BY pack_count ASC, select_count ASC, quality_score ASC
                """,
                (tenant_id, anchor_id),
            ).fetchall()
        return [self._row_to_memory_object(r) for r in rows]

    def archive_memories(self, memory_object_ids: List[str]):
        """Archive memories (soft-delete for forgetting)."""
        if not memory_object_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        placeholders = ",".join("?" for _ in memory_object_ids)
        conn.execute(
            f"""
            UPDATE memory_objects
            SET is_archived = 1, archived_at = ?
            WHERE memory_object_id IN ({placeholders})
            """,
            [now] + list(memory_object_ids),
        )
        conn.commit()

    def delete_old_archived(self, tenant_id: str, older_than_days: int = 90) -> int:
        """Hard-delete memories archived more than older_than_days ago."""
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
        conn = self._connect()
        cur = conn.execute(
            "DELETE FROM memory_objects WHERE tenant_id = ? AND is_archived = 1 AND archived_at < ?",
            (tenant_id, cutoff),
        )
        conn.commit()
        return cur.rowcount

    def forgetting_stats(self, tenant_id: str) -> dict:
        """Return forgetting/lifecycle statistics."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN is_archived = 0 THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN is_archived = 1 THEN 1 ELSE 0 END) as archived,
                    SUM(CASE WHEN pack_count = 0 AND is_archived = 0 THEN 1 ELSE 0 END) as never_packed,
                    SUM(CASE WHEN select_count = 0 AND is_archived = 0 THEN 1 ELSE 0 END) as never_selected
                FROM memory_objects WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchone()
        return {
            "total": row["total"] or 0,
            "active": row["active"] or 0,
            "archived": row["archived"] or 0,
            "never_packed": row["never_packed"] or 0,
            "never_selected": row["never_selected"] or 0,
        }

    def update_facets(self, mo: MemoryObject) -> None:
        """Update only the facet fields for an existing memory."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE memory_objects SET
                    entities_json = ?,
                    actor_entities_json = ?,
                    speaker_role = ?,
                    stance = ?,
                    mentioned_at = ?,
                    updated_at = ?
                WHERE memory_object_id = ?""",
                (
                    json.dumps(mo.entities),
                    json.dumps(mo.actor_entities),
                    mo.speaker_role,
                    mo.stance,
                    mo.mentioned_at,
                    datetime.now(timezone.utc).isoformat(),
                    mo.memory_object_id,
                ),
            )

    def get_all_active(self, tenant_id: str) -> List[MemoryObject]:
        """Return all non-archived, non-duplicate memories for a tenant."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_objects WHERE tenant_id = ? AND is_archived = 0 AND is_duplicate = 0",
                (tenant_id,),
            ).fetchall()
        return [self._row_to_memory_object(r) for r in rows]

    def get_processed_source_paths(self, tenant_id: str) -> set:
        """Return set of source_paths already in raw_sources for dedup."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source_path FROM raw_sources WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchall()
        return {row["source_path"] for row in rows if row["source_path"]}
