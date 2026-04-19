#!/usr/bin/env python3
"""
MemPalace MCP Server — read/write palace access for Claude Code
================================================================
Install: claude mcp add psa -- python -m psa.mcp_server [--palace /path/to/palace]

Tools (read):
  psa_status          — total drawers, wing/room breakdown
  psa_list_wings      — all wings with drawer counts
  psa_list_rooms      — rooms within a wing
  psa_get_taxonomy    — full wing → room → count tree
  psa_search          — semantic search, optional wing/room filter
  psa_check_duplicate — check if content already exists before filing

Tools (write):
  psa_add_drawer      — file verbatim content into a wing/room
  psa_delete_drawer   — remove a drawer by ID

PSA Atlas Tools (additive — require psa_mode != "off"):
  psa_atlas_search    — full PSA pipeline query (retriever + selector + packer)
  psa_store_memory    — store a typed memory object in the PSA MemoryStore
  psa_atlas_status    — atlas overview (version, anchor count, memory count)
  psa_list_anchors    — list all anchors with utilization stats
  psa_atlas_health    — atlas health report (novelty rate, utilization skew)
  psa_rebuild_atlas   — trigger atlas rebuild for the tenant
"""

import argparse
import os
import sys
import json
import logging
import hashlib
from datetime import datetime

from .config import MempalaceConfig, DEFAULT_TENANT_ID
from .version import __version__
from .searcher import search_memories

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger("psa_mcp")


def _parse_args():
    parser = argparse.ArgumentParser(description="PSA MCP Server")
    parser.add_argument(
        "--palace",
        metavar="PATH",
        help="Path to the palace directory (overrides config file and env var)",
    )
    args, _ = parser.parse_known_args()
    return args


_args = _parse_args()

if _args.palace:
    os.environ["PSA_PALACE_PATH"] = os.path.abspath(_args.palace)

_config = MempalaceConfig()

_client_cache = None
_collection_cache = None


def _get_collection(create=False):
    """Return the ChromaDB collection, caching the client between calls."""
    global _client_cache, _collection_cache
    try:
        import chromadb

        if _client_cache is None:
            _client_cache = chromadb.PersistentClient(path=_config.palace_path)
        if create:
            _collection_cache = _client_cache.get_or_create_collection(_config.collection_name)
        elif _collection_cache is None:
            _collection_cache = _client_cache.get_collection(_config.collection_name)
        return _collection_cache
    except Exception:
        return None


def _no_palace():
    return {
        "error": "No palace found",
        "hint": "Run: psa init <dir> && psa mine <dir>",
    }


# ==================== READ TOOLS ====================


def tool_status():
    col = _get_collection()
    if not col:
        return _no_palace()
    count = col.count()
    wings = {}
    rooms = {}
    try:
        all_meta = col.get(include=["metadatas"], limit=10000)["metadatas"]
        for m in all_meta:
            w = m.get("wing", "unknown")
            r = m.get("room", "unknown")
            wings[w] = wings.get(w, 0) + 1
            rooms[r] = rooms.get(r, 0) + 1
    except Exception:
        pass
    return {
        "total_drawers": count,
        "wings": wings,
        "rooms": rooms,
        "palace_path": _config.palace_path,
        "protocol": PALACE_PROTOCOL,
        "aaak_dialect": AAAK_SPEC,
    }


# ── AAAK Dialect Spec ─────────────────────────────────────────────────────────
# Included in status response so the AI learns it on first wake-up call.
# Also available via psa_get_aaak_spec tool.

PALACE_PROTOCOL = """IMPORTANT — PSA Memory Protocol:
1. ON WAKE-UP: Call psa_status to load palace overview.
2. BEFORE RESPONDING about any past event or decision: call psa_atlas_search or psa_search FIRST. Never guess — verify.
3. IF UNSURE about a fact: say "let me check" and query the atlas. Wrong is worse than slow.

This protocol ensures the AI KNOWS before it speaks. Storage is not memory — but storage + this protocol = memory."""

AAAK_SPEC = """AAAK is a compressed memory dialect that MemPalace uses for efficient storage.
It is designed to be readable by both humans and LLMs without decoding.

FORMAT:
  ENTITIES: 3-letter uppercase codes. ALC=Alice, JOR=Jordan, RIL=Riley, MAX=Max, BEN=Ben.
  EMOTIONS: *action markers* before/during text. *warm*=joy, *fierce*=determined, *raw*=vulnerable, *bloom*=tenderness.
  STRUCTURE: Pipe-separated fields. FAM: family | PROJ: projects | ⚠: warnings/reminders.
  DATES: ISO format (2026-03-31). COUNTS: Nx = N mentions (e.g., 570x).
  IMPORTANCE: ★ to ★★★★★ (1-5 scale).
  HALLS: hall_facts, hall_events, hall_discoveries, hall_preferences, hall_advice.
  WINGS: wing_user, wing_agent, wing_team, wing_code, wing_myproject, wing_hardware, wing_ue5, wing_ai_research.
  ROOMS: Hyphenated slugs representing named ideas (e.g., chromadb-setup, gpu-pricing).

EXAMPLE:
  FAM: ALC→♡JOR | 2D(kids): RIL(18,sports) MAX(11,chess+swimming) | BEN(contributor)

Read AAAK naturally — expand codes mentally, treat *markers* as emotional context.
When WRITING AAAK: use entity codes, mark emotions, keep structure tight."""


def tool_list_wings():
    col = _get_collection()
    if not col:
        return _no_palace()
    wings = {}
    try:
        all_meta = col.get(include=["metadatas"], limit=10000)["metadatas"]
        for m in all_meta:
            w = m.get("wing", "unknown")
            wings[w] = wings.get(w, 0) + 1
    except Exception:
        pass
    return {"wings": wings}


def tool_list_rooms(wing: str = None):
    col = _get_collection()
    if not col:
        return _no_palace()
    rooms = {}
    try:
        kwargs = {"include": ["metadatas"], "limit": 10000}
        if wing:
            kwargs["where"] = {"wing": wing}
        all_meta = col.get(**kwargs)["metadatas"]
        for m in all_meta:
            r = m.get("room", "unknown")
            rooms[r] = rooms.get(r, 0) + 1
    except Exception:
        pass
    return {"wing": wing or "all", "rooms": rooms}


def tool_get_taxonomy():
    col = _get_collection()
    if not col:
        return _no_palace()
    taxonomy = {}
    try:
        all_meta = col.get(include=["metadatas"], limit=10000)["metadatas"]
        for m in all_meta:
            w = m.get("wing", "unknown")
            r = m.get("room", "unknown")
            if w not in taxonomy:
                taxonomy[w] = {}
            taxonomy[w][r] = taxonomy[w].get(r, 0) + 1
    except Exception:
        pass
    return {"taxonomy": taxonomy}


def tool_search(query: str, limit: int = 5, wing: str = None, room: str = None):
    return search_memories(
        query,
        palace_path=_config.palace_path,
        wing=wing,
        room=room,
        n_results=limit,
    )


def tool_check_duplicate(content: str, threshold: float = 0.9):
    col = _get_collection()
    if not col:
        return _no_palace()
    try:
        results = col.query(
            query_texts=[content],
            n_results=5,
            include=["metadatas", "documents", "distances"],
        )
        duplicates = []
        if results["ids"] and results["ids"][0]:
            for i, drawer_id in enumerate(results["ids"][0]):
                dist = results["distances"][0][i]
                # ChromaDB default is L2; cosine distance is in [0,2]; clamp to [0,1]
                similarity = round(max(0.0, 1 - dist), 3)
                if similarity >= threshold:
                    meta = results["metadatas"][0][i]
                    doc = results["documents"][0][i]
                    duplicates.append(
                        {
                            "id": drawer_id,
                            "wing": meta.get("wing", "?"),
                            "room": meta.get("room", "?"),
                            "similarity": similarity,
                            "content": doc[:200] + "..." if len(doc) > 200 else doc,
                        }
                    )
        return {
            "is_duplicate": len(duplicates) > 0,
            "matches": duplicates,
        }
    except Exception as e:
        return {"error": str(e)}


def tool_get_aaak_spec():
    """Return the AAAK dialect specification."""
    return {"aaak_spec": AAAK_SPEC}


# ==================== WRITE TOOLS ====================


def tool_add_drawer(
    wing: str, room: str, content: str, source_file: str = None, added_by: str = "mcp"
):
    """File verbatim content into a wing/room. Checks for duplicates first."""
    col = _get_collection(create=True)
    if not col:
        return _no_palace()

    # Duplicate check
    dup = tool_check_duplicate(content, threshold=0.9)
    if dup.get("is_duplicate"):
        return {
            "success": False,
            "reason": "duplicate",
            "matches": dup["matches"],
        }

    drawer_id = f"drawer_{wing}_{room}_{hashlib.md5((content[:100] + datetime.now().isoformat()).encode(), usedforsecurity=False).hexdigest()[:16]}"

    try:
        col.add(
            ids=[drawer_id],
            documents=[content],
            metadatas=[
                {
                    "wing": wing,
                    "room": room,
                    "source_file": source_file or "",
                    "chunk_index": 0,
                    "added_by": added_by,
                    "filed_at": datetime.now().isoformat(),
                }
            ],
        )
        logger.info(f"Filed drawer: {drawer_id} → {wing}/{room}")
        return {"success": True, "drawer_id": drawer_id, "wing": wing, "room": room}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_delete_drawer(drawer_id: str):
    """Delete a single drawer by ID."""
    col = _get_collection()
    if not col:
        return _no_palace()
    existing = col.get(ids=[drawer_id])
    if not existing["ids"]:
        return {"success": False, "error": f"Drawer not found: {drawer_id}"}
    try:
        col.delete(ids=[drawer_id])
        logger.info(f"Deleted drawer: {drawer_id}")
        return {"success": True, "drawer_id": drawer_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Stage 2 reload hook + persistent pipeline cache ──────────────────────────
#
# _tenant_pipelines holds one PSAPipeline per tenant_id for the server lifetime.
# Each entry also tracks last_reload_mtime so advertisement-decay atlas removals
# are picked up between queries without rebuilding the whole pipeline.

_tenant_pipelines: dict = {}  # {tenant_id: {"pipeline": PSAPipeline, "last_reload_mtime": float}}


def maybe_reload_pipeline(*, pipeline, state: dict, tenant_id: str) -> None:
    """Trigger pipeline.reload_atlas() if the advertisement reload marker has
    advanced since the last reload tracked in `state`."""
    from .advertisement.reload import marker_mtime, should_reload

    last = state.get("last_reload_mtime", 0.0)
    if should_reload(tenant_id=tenant_id, last_reload_mtime=last):
        pipeline.reload_atlas()
        state["last_reload_mtime"] = marker_mtime(tenant_id=tenant_id)


def _ensure_fresh_pipeline(pipeline, tenant_id: str) -> None:
    state = _tenant_pipelines.setdefault(tenant_id, {"pipeline": None, "last_reload_mtime": 0.0})
    maybe_reload_pipeline(pipeline=pipeline, state=state, tenant_id=tenant_id)


# ── PSA Atlas tool helpers ────────────────────────────────────────────────────


def _get_psa_pipeline(tenant_id: str = DEFAULT_TENANT_ID):
    """Return a cached PSAPipeline, building it on first call. Returns None if atlas not built."""
    entry = _tenant_pipelines.get(tenant_id)
    if entry and entry.get("pipeline"):
        pipeline = entry["pipeline"]
        _ensure_fresh_pipeline(pipeline, tenant_id)
        return pipeline
    try:
        from .pipeline import PSAPipeline

        pipeline = PSAPipeline.from_tenant(tenant_id=tenant_id)
        _tenant_pipelines[tenant_id] = {"pipeline": pipeline, "last_reload_mtime": 0.0}
        return pipeline
    except FileNotFoundError:
        _tenant_pipelines.pop(tenant_id, None)
        return None
    except Exception as e:
        logger.warning("PSAPipeline init failed: %s", e)
        return None


def _get_psa_store(tenant_id: str = DEFAULT_TENANT_ID):
    """Return (TenantManager, MemoryStore) for the given tenant."""
    from .tenant import TenantManager
    from .memory_object import MemoryStore

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    store = MemoryStore(db_path=tenant.memory_db_path)
    return tenant, store


def _get_psa_atlas(tenant_id: str = DEFAULT_TENANT_ID):
    """Return (Atlas, AtlasManager) or (None, mgr) if no atlas exists."""
    from .tenant import TenantManager
    from .atlas import AtlasManager

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
    atlas = mgr.get_atlas()
    return atlas, mgr


# ── PSA Atlas MCP tool functions ──────────────────────────────────────────────


def tool_psa_atlas_search(query: str, tenant_id: str = DEFAULT_TENANT_ID, token_budget: int = 6000):
    """Run the full PSA pipeline query (retriever + selector + packer)."""
    pipeline = _get_psa_pipeline(tenant_id)
    if pipeline is None:
        return {
            "error": f"No PSA atlas built for tenant '{tenant_id}'. Run 'psa atlas build' first."
        }
    pipeline.token_budget = token_budget
    result = pipeline.query(query)
    return result.to_dict()


def tool_psa_store_memory(
    title: str,
    body: str,
    memory_type: str = "SEMANTIC",
    tenant_id: str = DEFAULT_TENANT_ID,
    quality_score: float = 0.7,
):
    """Store a typed memory object in the PSA MemoryStore."""
    from .memory_object import MemoryObject, MemoryType
    from .embeddings import EmbeddingModel

    try:
        mtype = MemoryType[memory_type.upper()]
    except KeyError:
        valid = [t.name for t in MemoryType]
        return {"error": f"Unknown memory_type '{memory_type}'. Valid: {valid}"}

    _, store = _get_psa_store(tenant_id)
    em = EmbeddingModel()
    embedding = em.embed(f"{title}\n{body}")

    mo = MemoryObject.create(
        tenant_id=tenant_id,
        memory_type=mtype,
        title=title,
        body=body,
        summary=body[:256],
        source_ids=[],
        classification_reason="stored via MCP psa_store_memory tool",
        quality_score=quality_score,
    )
    mo.embedding = embedding
    store.add(mo)
    return {
        "memory_object_id": mo.memory_object_id,
        "memory_type": mtype.value,
        "title": title,
        "tenant_id": tenant_id,
        "stored": True,
    }


def tool_psa_atlas_status(tenant_id: str = DEFAULT_TENANT_ID):
    """Atlas overview: version, anchor count, memory count, PSA mode."""
    atlas, _ = _get_psa_atlas(tenant_id)
    _, store = _get_psa_store(tenant_id)

    if atlas is None:
        return {
            "status": "no_atlas",
            "tenant_id": tenant_id,
            "message": "No atlas built yet. Run 'psa atlas build' to build one.",
        }

    total_memories = sum(
        len(store.query_by_anchor(tenant_id=tenant_id, anchor_id=c.anchor_id, limit=100_000))
        for c in atlas.cards
    )
    return {
        "status": "ok",
        "tenant_id": tenant_id,
        "atlas_version": atlas.version,
        "total_anchors": len(atlas.cards),
        "novelty_anchors": sum(1 for c in atlas.cards if c.is_novelty),
        "learned_anchors": sum(1 for c in atlas.cards if not c.is_novelty),
        "total_memories_indexed": total_memories,
    }


def tool_psa_list_anchors(tenant_id: str = DEFAULT_TENANT_ID):
    """List all anchors with their names and memory counts."""
    atlas, _ = _get_psa_atlas(tenant_id)
    _, store = _get_psa_store(tenant_id)

    if atlas is None:
        return {"error": f"No atlas for tenant '{tenant_id}'."}

    anchors = []
    for card in atlas.cards:
        count = len(
            store.query_by_anchor(tenant_id=tenant_id, anchor_id=card.anchor_id, limit=100_000)
        )
        anchors.append(
            {
                "anchor_id": card.anchor_id,
                "name": card.name,
                "meaning": card.meaning[:120],
                "memory_count": count,
                "is_novelty": card.is_novelty,
                "memory_types": card.memory_types,
            }
        )
    anchors.sort(key=lambda a: a["memory_count"], reverse=True)
    return {"tenant_id": tenant_id, "atlas_version": atlas.version, "anchors": anchors}


def tool_psa_atlas_health(tenant_id: str = DEFAULT_TENANT_ID):
    """Atlas health report: novelty rate, utilization skew, rebuild recommendation."""
    atlas, _ = _get_psa_atlas(tenant_id)
    _, store = _get_psa_store(tenant_id)

    if atlas is None:
        return {"error": f"No atlas for tenant '{tenant_id}'."}

    from .health import AtlasHealthMonitor

    monitor = AtlasHealthMonitor()
    report = monitor.check_health(atlas, store, tenant_id=tenant_id)
    return report.to_dict()


_rebuild_lock_path = None  # unused, kept for clarity


def tool_psa_rebuild_atlas(tenant_id: str = DEFAULT_TENANT_ID):
    """Trigger atlas rebuild for the tenant (may take several minutes)."""
    import fcntl
    from .tenant import TenantManager
    from .atlas import AtlasManager
    from .memory_object import MemoryStore

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    store = MemoryStore(db_path=tenant.memory_db_path)
    mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)

    lock_path = os.path.join(tenant.root_dir, ".atlas_rebuild.lock")
    try:
        lock_fd = open(lock_path, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, IOError):
        return {"status": "error", "error": "Atlas rebuild already in progress for this tenant."}

    try:
        atlas = mgr.rebuild(store)
        return {
            "status": "rebuilt",
            "tenant_id": tenant_id,
            "atlas_version": atlas.version,
            "total_anchors": len(atlas.cards),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


# ==================== MCP PROTOCOL ====================

TOOLS = {
    "psa_status": {
        "description": "Palace overview — total drawers, wing and room counts",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_status,
    },
    "psa_list_wings": {
        "description": "List all wings with drawer counts",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_list_wings,
    },
    "psa_list_rooms": {
        "description": "List rooms within a wing (or all rooms if no wing given)",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing": {"type": "string", "description": "Wing to list rooms for (optional)"},
            },
        },
        "handler": tool_list_rooms,
    },
    "psa_get_taxonomy": {
        "description": "Full taxonomy: wing → room → drawer count",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_get_taxonomy,
    },
    "psa_search": {
        "description": "Semantic search. Returns verbatim drawer content with similarity scores.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "limit": {"type": "integer", "description": "Max results (default 5)"},
                "wing": {"type": "string", "description": "Filter by wing (optional)"},
                "room": {"type": "string", "description": "Filter by room (optional)"},
            },
            "required": ["query"],
        },
        "handler": tool_search,
    },
    "psa_check_duplicate": {
        "description": "Check if content already exists in the palace before filing",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Content to check"},
                "threshold": {
                    "type": "number",
                    "description": "Similarity threshold 0-1 (default 0.9)",
                },
            },
            "required": ["content"],
        },
        "handler": tool_check_duplicate,
    },
    "psa_add_drawer": {
        "description": "File verbatim content into the palace. Checks for duplicates first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing": {"type": "string", "description": "Wing (project name)"},
                "room": {
                    "type": "string",
                    "description": "Room (aspect: backend, decisions, meetings...)",
                },
                "content": {
                    "type": "string",
                    "description": "Verbatim content to store — exact words, never summarized",
                },
                "source_file": {"type": "string", "description": "Where this came from (optional)"},
                "added_by": {"type": "string", "description": "Who is filing this (default: mcp)"},
            },
            "required": ["wing", "room", "content"],
        },
        "handler": tool_add_drawer,
    },
    "psa_delete_drawer": {
        "description": "Delete a drawer by ID. Irreversible.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drawer_id": {"type": "string", "description": "ID of the drawer to delete"},
            },
            "required": ["drawer_id"],
        },
        "handler": tool_delete_drawer,
    },
    # ── PSA Atlas Tools ───────────────────────────────────────────────────────
    "psa_atlas_search": {
        "description": (
            "Search using the PSA pipeline: retriever → selector → packer. "
            "Returns role-organized context (FAILURE → PROCEDURAL → TOOL-USE → "
            "EPISODES → FACTS → RAW CONTEXT) within a token budget. "
            "Requires an atlas to be built ('psa atlas build')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or task to retrieve context for",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant identifier (default: 'default')",
                },
                "token_budget": {
                    "type": "integer",
                    "description": "Maximum tokens in the packed context (default: 6000)",
                },
            },
            "required": ["query"],
        },
        "handler": tool_psa_atlas_search,
    },
    "psa_store_memory": {
        "description": (
            "Store a typed memory object in the PSA MemoryStore. "
            "memory_type must be one of: EPISODIC, SEMANTIC, PROCEDURAL, "
            "FAILURE, TOOL_USE, WORKING_DERIVATIVE."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short title for the memory"},
                "body": {"type": "string", "description": "Full memory content"},
                "memory_type": {
                    "type": "string",
                    "description": "Memory type: EPISODIC | SEMANTIC | PROCEDURAL | FAILURE | TOOL_USE | WORKING_DERIVATIVE",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant identifier (default: 'default')",
                },
                "quality_score": {
                    "type": "number",
                    "description": "Quality score 0.0-1.0 (default: 0.7)",
                },
            },
            "required": ["title", "body"],
        },
        "handler": tool_psa_store_memory,
    },
    "psa_atlas_status": {
        "description": "PSA atlas overview: version, anchor count, memory count. Shows whether an atlas has been built for the tenant.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant identifier (default: 'default')",
                },
            },
        },
        "handler": tool_psa_atlas_status,
    },
    "psa_list_anchors": {
        "description": "List all PSA anchors with their names, meanings, and memory counts. Useful for understanding what memory regions have been learned.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant identifier (default: 'default')",
                },
            },
        },
        "handler": tool_psa_list_anchors,
    },
    "psa_atlas_health": {
        "description": "Atlas health report: novelty rate, utilization skew, and rebuild recommendation. High novelty rate (>8%) or skew (>3×) indicates the atlas should be rebuilt.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant identifier (default: 'default')",
                },
            },
        },
        "handler": tool_psa_atlas_health,
    },
    "psa_rebuild_atlas": {
        "description": "Rebuild the PSA atlas for a tenant. This may take several minutes for large memory stores. Run after significant new memory ingestion or when health check recommends rebuild.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant identifier (default: 'default')",
                },
            },
        },
        "handler": tool_psa_rebuild_atlas,
    },
}


def handle_request(request):
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "psa", "version": __version__},
            },
        }
    elif method == "notifications/initialized":
        return None
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {"name": n, "description": t["description"], "inputSchema": t["input_schema"]}
                    for n, t in TOOLS.items()
                ]
            },
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        if tool_name not in TOOLS:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
        # Coerce argument types based on input_schema.
        # MCP JSON transport may deliver integers as floats or strings;
        # ChromaDB and Python slicing require native int.
        schema_props = TOOLS[tool_name]["input_schema"].get("properties", {})
        for key, value in list(tool_args.items()):
            prop_schema = schema_props.get(key, {})
            declared_type = prop_schema.get("type")
            if declared_type == "integer" and not isinstance(value, int):
                tool_args[key] = int(value)
            elif declared_type == "number" and not isinstance(value, (int, float)):
                tool_args[key] = float(value)
        try:
            result = TOOLS[tool_name]["handler"](**tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
            }
        except Exception:
            logger.exception(f"Tool error in {tool_name}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": "Internal tool error"},
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def main():
    logger.info("MemPalace MCP Server starting...")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            response = handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Server error: {e}")


if __name__ == "__main__":
    main()
