#!/usr/bin/env python3
"""
MemPalace — Give your AI a memory. No API key required.

Two ways to ingest:
  Projects:      psa mine ~/projects/my_app          (code, docs, notes)
  Conversations: psa mine ~/chats/ --mode convos     (Claude, ChatGPT, Slack)

Same palace. Same search. Different ingest strategies.

Commands:
    psa init <dir>                  Detect rooms from folder structure
    psa split <dir>                 Split concatenated mega-files into per-session files
    psa mine <dir>                  Mine project files (default)
    psa mine <dir> --mode convos    Mine conversation exports
    psa search "query"              Find anything, exact words
    psa wake-up                     Show L0 + L1 wake-up context
    psa wake-up --wing my_app       Wake-up for a specific project
    psa status                      Show what's been filed

PSA Atlas Commands (require psa_mode != "off"):
    psa atlas build                 Build or rebuild the PSA atlas for the tenant
    psa atlas status                Show atlas version, anchor count, memory count
    psa atlas health                Show health report (novelty rate, skew, rebuild recommendation)
    psa benchmark                   Compare PSA pipeline vs raw ChromaDB search
    psa migrate                     Migrate ChromaDB palace to PSA MemoryStore (non-destructive)

Examples:
    psa init ~/projects/my_app
    psa mine ~/projects/my_app
    psa mine ~/chats/claude-sessions --mode convos
    psa search "why did we switch to GraphQL"
    psa search "pricing discussion" --wing my_app --room costs
    psa atlas build
    psa atlas health
    psa migrate --palace ~/.psa/palace
"""

import os
import sys
import argparse
from pathlib import Path

from .config import MempalaceConfig


def cmd_init(args):
    """Set up PSA for a directory — creates config and mines initial memories."""
    import json
    from pathlib import Path

    project_dir = str(Path(args.dir).expanduser().resolve())
    print(f"\n  Setting up PSA for: {project_dir}")

    # Initialize config directory
    cfg = MempalaceConfig()
    cfg.init()
    print(f"  Config ready at {cfg._config_dir}")

    # Optionally mine the directory now
    if not getattr(args, "skip_mine", False):
        from .miner import mine
        print(f"\n  Mining {project_dir}...")
        mine(
            project_dir=project_dir,
            palace_path=cfg.palace_path,
            agent="psa",
            limit=0,
            dry_run=False,
            respect_gitignore=True,
            include_ignored=[],
        )
    print("\n  Done. Run 'psa atlas build' to build the PSA atlas.")


def cmd_mine(args):
    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
    include_ignored = []
    for raw in args.include_ignored or []:
        include_ignored.extend(part.strip() for part in raw.split(",") if part.strip())

    if args.mode == "convos":
        from .convo_miner import mine_convos

        mine_convos(
            convo_dir=args.dir,
            palace_path=palace_path,
            wing=args.wing,
            agent=args.agent,
            limit=args.limit,
            dry_run=args.dry_run,
            extract_mode=args.extract,
        )
    else:
        from .miner import mine

        mine(
            project_dir=args.dir,
            palace_path=palace_path,
            wing_override=args.wing,
            agent=args.agent,
            limit=args.limit,
            dry_run=args.dry_run,
            respect_gitignore=not args.no_gitignore,
            include_ignored=include_ignored,
        )


def cmd_search(args):
    """Search — PSA pipeline by default, raw ChromaDB as fallback."""
    cfg = MempalaceConfig()
    query = args.query
    n_results = args.results

    # Try PSA pipeline first (psa_mode=primary)
    tenant_id = cfg.tenant_id
    try:
        from .pipeline import PSAPipeline
        pipeline = PSAPipeline.from_tenant(
            tenant_id=tenant_id,
            token_budget=cfg.token_budget,
            psa_mode=cfg.psa_mode,
        )
        result = pipeline.query(query)
        print(f"\n── PSA Search: {query!r} ──\n")
        print(result.text or "(no context found)")
        print(f"\n[{len(result.selected_anchors)} anchors selected, "
              f"{result.token_count} tokens, "
              f"{result.timing.total_ms:.0f}ms]")
        return
    except FileNotFoundError:
        pass  # no atlas yet — fall through to raw search
    except Exception:
        pass  # pipeline error — fall through to raw search

    # Fallback: raw ChromaDB search
    from .searcher import search, SearchError
    palace_path = os.path.expanduser(args.palace) if args.palace else cfg.palace_path
    try:
        search(
            query=query,
            palace_path=palace_path,
            wing=getattr(args, "wing", None),
            room=getattr(args, "room", None),
            n_results=n_results,
        )
    except SearchError:
        sys.exit(1)


def cmd_wakeup(args):
    """Show PSA atlas status as session wake-up context."""
    cfg = MempalaceConfig()
    tenant_id = cfg.tenant_id

    try:
        from .tenant import TenantManager
        from .atlas import AtlasManager
        from .memory_object import MemoryStore

        tm = TenantManager()
        tenant = tm.get_or_create(tenant_id)
        store = MemoryStore(db_path=tenant.memory_db_path)
        mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
        atlas = mgr.get_atlas()

        if atlas is None:
            print(f"No PSA atlas built. Run 'psa atlas build' first.")
            print(f"(For legacy wake-up: PSA_MODE=off psa legacy wake-up)")
            return

        total = sum(
            len(store.query_by_anchor(tenant_id=tenant_id, anchor_id=c.anchor_id, limit=100_000))
            for c in atlas.cards
        )
        learned = sum(1 for c in atlas.cards if not c.is_novelty)
        novelty = sum(1 for c in atlas.cards if c.is_novelty)
        print(f"\n── PSA Session Wake-Up (tenant: {tenant_id}) ──")
        print(f"  Atlas v{atlas.version}: {len(atlas.cards)} anchors "
              f"({learned} learned, {novelty} novelty)")
        print(f"  Memories indexed: {total}")
        print(f"\nTop anchors by memory count:")
        sorted_cards = sorted(atlas.cards, key=lambda c: c.memory_count, reverse=True)
        for card in sorted_cards[:5]:
            print(f"  [{card.anchor_id}] {card.name} — {card.meaning[:60]}")
        print(f"\nUse 'psa search <query>' to retrieve memories.")
    except Exception as e:
        print(f"PSA wake-up failed: {e}")
        print("Run 'psa atlas build' first.")


def cmd_split(args):
    """Split concatenated transcript mega-files into per-session files."""
    from .split_mega_files import main as split_main
    import sys

    # Rebuild argv for split_mega_files argparse
    argv = ["--source", args.dir]
    if args.output_dir:
        argv += ["--output-dir", args.output_dir]
    if args.dry_run:
        argv.append("--dry-run")
    if args.min_sessions != 2:
        argv += ["--min-sessions", str(args.min_sessions)]

    old_argv = sys.argv
    sys.argv = ["psa split"] + argv
    try:
        split_main()
    finally:
        sys.argv = old_argv


def cmd_status(args):
    from .miner import status

    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
    status(palace_path=palace_path)


def cmd_repair(args):
    """Rebuild palace vector index from SQLite metadata."""
    import chromadb
    import shutil

    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path

    if not os.path.isdir(palace_path):
        print(f"\n  No palace found at {palace_path}")
        return

    print(f"\n{'=' * 55}")
    print("  MemPalace Repair")
    print(f"{'=' * 55}\n")
    print(f"  Palace: {palace_path}")

    # Try to read existing drawers
    try:
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("psa_drawers")
        total = col.count()
        print(f"  Drawers found: {total}")
    except Exception as e:
        print(f"  Error reading palace: {e}")
        print("  Cannot recover — palace may need to be re-mined from source files.")
        return

    if total == 0:
        print("  Nothing to repair.")
        return

    # Extract all drawers in batches
    print("\n  Extracting drawers...")
    batch_size = 5000
    all_ids = []
    all_docs = []
    all_metas = []
    offset = 0
    while offset < total:
        batch = col.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
        all_ids.extend(batch["ids"])
        all_docs.extend(batch["documents"])
        all_metas.extend(batch["metadatas"])
        offset += batch_size
    print(f"  Extracted {len(all_ids)} drawers")

    # Backup and rebuild
    backup_path = palace_path + ".backup"
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    print(f"  Backing up to {backup_path}...")
    shutil.copytree(palace_path, backup_path)

    print("  Rebuilding collection...")
    client.delete_collection("psa_drawers")
    new_col = client.create_collection("psa_drawers")

    filed = 0
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i : i + batch_size]
        batch_docs = all_docs[i : i + batch_size]
        batch_metas = all_metas[i : i + batch_size]
        new_col.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
        filed += len(batch_ids)
        print(f"  Re-filed {filed}/{len(all_ids)} drawers...")

    print(f"\n  Repair complete. {filed} drawers rebuilt.")
    print(f"  Backup saved at {backup_path}")
    print(f"\n{'=' * 55}\n")


# ── PSA Atlas CLI commands ────────────────────────────────────────────────────


def cmd_atlas(args):
    """Handle 'psa atlas <subcommand>'."""
    action = getattr(args, "atlas_action", None)
    if not action:
        print("Usage: psa atlas {build,status,health}")
        return

    if action == "build":
        _cmd_atlas_build(args)
    elif action == "status":
        _cmd_atlas_status(args)
    elif action == "health":
        _cmd_atlas_health(args)


def _cmd_atlas_build(args):
    tenant_id = getattr(args, "tenant", "default")
    print(f"Building PSA atlas for tenant '{tenant_id}'...")

    try:
        from .tenant import TenantManager
        from .atlas import AtlasManager, AtlasCorpusTooSmall, AtlasUnstable
        from .memory_object import MemoryStore

        tm = TenantManager()
        tenant = tm.get_or_create(tenant_id)
        store = MemoryStore(db_path=tenant.memory_db_path)
        mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)

        atlas = mgr.rebuild(store)
        print(f"  Atlas v{atlas.version} built.")
        print(f"  Anchors: {len(atlas.cards)} ({sum(1 for c in atlas.cards if not c.is_novelty)} learned, "
              f"{sum(1 for c in atlas.cards if c.is_novelty)} novelty)")
    except AtlasCorpusTooSmall as e:
        print(f"  Error: {e}")
        print("  Mine more content first ('psa mine <dir>') then try again.")
        sys.exit(1)
    except AtlasUnstable as e:
        print(f"  Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  Error building atlas: {e}")
        sys.exit(1)


def _cmd_atlas_status(args):
    tenant_id = getattr(args, "tenant", "default")
    try:
        from .tenant import TenantManager
        from .atlas import AtlasManager
        from .memory_object import MemoryStore

        tm = TenantManager()
        tenant = tm.get_or_create(tenant_id)
        store = MemoryStore(db_path=tenant.memory_db_path)
        mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
        atlas = mgr.get_atlas()

        if atlas is None:
            print(f"No atlas built for tenant '{tenant_id}'. Run 'psa atlas build'.")
            return

        total = sum(
            len(store.query_by_anchor(tenant_id=tenant_id, anchor_id=c.anchor_id, limit=100_000))
            for c in atlas.cards
        )
        print(f"Atlas v{atlas.version} — tenant '{tenant_id}'")
        print(f"  Anchors: {len(atlas.cards)} ({sum(1 for c in atlas.cards if not c.is_novelty)} learned, "
              f"{sum(1 for c in atlas.cards if c.is_novelty)} novelty)")
        print(f"  Memories indexed: {total}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def _cmd_atlas_health(args):
    tenant_id = getattr(args, "tenant", "default")
    try:
        from .tenant import TenantManager
        from .atlas import AtlasManager
        from .memory_object import MemoryStore
        from .health import AtlasHealthMonitor

        tm = TenantManager()
        tenant = tm.get_or_create(tenant_id)
        store = MemoryStore(db_path=tenant.memory_db_path)
        mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
        atlas = mgr.get_atlas()

        if atlas is None:
            print(f"No atlas for tenant '{tenant_id}'. Run 'psa atlas build'.")
            return

        monitor = AtlasHealthMonitor()
        report = monitor.check_health(atlas, store, tenant_id=tenant_id)
        print(report.summary())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_benchmark(args):
    """Compare PSA pipeline vs raw ChromaDB search."""
    print("PSA benchmark mode — comparing PSA pipeline vs raw ChromaDB search.")
    print("(Full benchmarking requires a populated palace and atlas. Run 'psa mine' and 'psa atlas build' first.)")

    tenant_id = getattr(args, "tenant", "default")
    query = getattr(args, "query", None)

    if not query:
        print("Usage: psa benchmark --query 'your query here'")
        return

    try:
        from .searcher import search_memories
        from .config import MempalaceConfig
        cfg = MempalaceConfig()

        print(f"\n--- Raw ChromaDB search ---")
        raw_results = search_memories(query, n_results=5, palace_path=cfg.palace_path)
        for i, r in enumerate(raw_results.get("results", []), 1):
            print(f"  [{i}] {r.get('title', '?')} ({r.get('similarity', 0):.3f})")

        print(f"\n--- PSA pipeline search ---")
        from .pipeline import PSAPipeline
        try:
            pipeline = PSAPipeline.from_tenant(tenant_id=tenant_id, psa_mode=cfg.psa_mode)
            result = pipeline.query(query)
            print(f"  Packed context: {result.token_count} tokens")
            print(f"  Selected anchors: {[a.anchor_id for a in result.selected_anchors]}")
            print(f"  Pipeline timing: {result.timing.total_ms:.1f}ms total")
        except FileNotFoundError:
            print(f"  (No atlas for tenant '{tenant_id}' — run 'psa atlas build' first)")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_migrate(args):
    """Migrate ChromaDB palace to PSA MemoryStore (non-destructive)."""
    from .migrate import migrate_palace_to_psa

    palace_path = getattr(args, "palace", None) or os.path.expanduser("~/.psa/palace")
    tenant_id = getattr(args, "tenant", "default")
    collection = getattr(args, "collection", "mempalace")

    print(f"Migrating palace '{palace_path}' → PSA tenant '{tenant_id}'...")
    print("(This is non-destructive — original palace data will not be modified.)")

    try:
        stats = migrate_palace_to_psa(
            chroma_path=palace_path,
            collection_name=collection,
            tenant_id=tenant_id,
        )
        print(f"\n  Migration complete:")
        print(f"    Total drawers: {stats.total}")
        print(f"    Migrated:      {stats.migrated}")
        print(f"    Skipped:       {stats.skipped} (already existed)")
        print(f"    Failed:        {stats.failed}")
        if stats.errors:
            print(f"    Errors: {stats.errors[:3]}")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  Error: {e}")
        sys.exit(1)


def cmd_legacy(args):
    """Route to a legacy palace command."""
    action = getattr(args, "legacy_action", None)
    if not action:
        print("Usage: psa legacy {compress,wake-up}")
        print("Note: set PSA_MODE=off for full legacy mode.")
        return
    if action == "compress":
        cmd_compress(args)
    elif action == "wake-up":
        _legacy_wakeup(args)
    else:
        print(f"Unknown legacy action: {action}")


def _legacy_wakeup(args):
    """Legacy L0+L1 wake-up via MemoryStack (requires layers.py)."""
    try:
        from .layers import MemoryStack
        palace_path = os.path.expanduser(args.palace) if getattr(args, "palace", None) else MempalaceConfig().palace_path
        stack = MemoryStack(palace_path=palace_path)
        text = stack.wake_up(wing=getattr(args, "wing", None))
        tokens = len(text) // 4
        print(f"Wake-up text (~{tokens} tokens):")
        print("=" * 50)
        print(text)
    except ImportError:
        print("Legacy wake-up requires the MemoryStack layer (layers.py). "
              "This file has been removed in PSA primary mode.")
        print("Use 'psa wake-up' for PSA atlas wake-up context.")


def cmd_hook(args):
    """Run hook logic: reads JSON from stdin, outputs JSON to stdout."""
    from .hooks_cli import run_hook

    run_hook(hook_name=args.hook, harness=args.harness)


def cmd_instructions(args):
    """Output skill instructions to stdout."""
    from .instructions_cli import run_instructions

    run_instructions(name=args.name)


def cmd_compress(args):
    """Compress drawers in a wing using AAAK Dialect."""
    import chromadb
    try:
        from .dialect import Dialect
    except ImportError:
        print("Error: 'compress' requires the dialect module which has been removed in PSA v4.")
        print("Use 'psa mine' to re-ingest content through the consolidation pipeline.")
        return

    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path

    # Load dialect (with optional entity config)
    config_path = args.config
    if not config_path:
        for candidate in ["entities.json", os.path.join(palace_path, "entities.json")]:
            if os.path.exists(candidate):
                config_path = candidate
                break

    if config_path and os.path.exists(config_path):
        dialect = Dialect.from_config(config_path)
        print(f"  Loaded entity config: {config_path}")
    else:
        dialect = Dialect()

    # Connect to palace
    try:
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("psa_drawers")
    except Exception:
        print(f"\n  No palace found at {palace_path}")
        print("  Run: psa init <dir> then psa mine <dir>")
        sys.exit(1)

    # Query drawers in batches to avoid SQLite variable limit (~999)
    where = {"wing": args.wing} if args.wing else None
    _BATCH = 500
    docs, metas, ids = [], [], []
    offset = 0
    while True:
        try:
            kwargs = {"include": ["documents", "metadatas"], "limit": _BATCH, "offset": offset}
            if where:
                kwargs["where"] = where
            batch = col.get(**kwargs)
        except Exception as e:
            if not docs:
                print(f"\n  Error reading drawers: {e}")
                sys.exit(1)
            break
        batch_docs = batch.get("documents", [])
        if not batch_docs:
            break
        docs.extend(batch_docs)
        metas.extend(batch.get("metadatas", []))
        ids.extend(batch.get("ids", []))
        offset += len(batch_docs)
        if len(batch_docs) < _BATCH:
            break

    if not docs:
        wing_label = f" in wing '{args.wing}'" if args.wing else ""
        print(f"\n  No drawers found{wing_label}.")
        return

    print(
        f"\n  Compressing {len(docs)} drawers"
        + (f" in wing '{args.wing}'" if args.wing else "")
        + "..."
    )
    print()

    total_original = 0
    total_compressed = 0
    compressed_entries = []

    for doc, meta, doc_id in zip(docs, metas, ids):
        compressed = dialect.compress(doc, metadata=meta)
        stats = dialect.compression_stats(doc, compressed)

        total_original += stats["original_chars"]
        total_compressed += stats["compressed_chars"]

        compressed_entries.append((doc_id, compressed, meta, stats))

        if args.dry_run:
            wing_name = meta.get("wing", "?")
            room_name = meta.get("room", "?")
            source = Path(meta.get("source_file", "?")).name
            print(f"  [{wing_name}/{room_name}] {source}")
            print(
                f"    {stats['original_tokens']}t -> {stats['compressed_tokens']}t ({stats['ratio']:.1f}x)"
            )
            print(f"    {compressed}")
            print()

    # Store compressed versions (unless dry-run)
    if not args.dry_run:
        try:
            comp_col = client.get_or_create_collection("psa_compressed")
            for doc_id, compressed, meta, stats in compressed_entries:
                comp_meta = dict(meta)
                comp_meta["compression_ratio"] = round(stats["ratio"], 1)
                comp_meta["original_tokens"] = stats["original_tokens"]
                comp_col.upsert(
                    ids=[doc_id],
                    documents=[compressed],
                    metadatas=[comp_meta],
                )
            print(
                f"  Stored {len(compressed_entries)} compressed drawers in 'psa_compressed' collection."
            )
        except Exception as e:
            print(f"  Error storing compressed drawers: {e}")
            sys.exit(1)

    # Summary
    ratio = total_original / max(total_compressed, 1)
    orig_tokens = Dialect.count_tokens("x" * total_original)
    comp_tokens = Dialect.count_tokens("x" * total_compressed)
    print(f"  Total: {orig_tokens:,}t -> {comp_tokens:,}t ({ratio:.1f}x compression)")
    if args.dry_run:
        print("  (dry run -- nothing stored)")


def main():
    parser = argparse.ArgumentParser(
        description="MemPalace — Give your AI a memory. No API key required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--palace",
        default=None,
        help="Where the palace lives (default: from ~/.psa/config.json or ~/.psa/palace)",
    )

    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Detect rooms from your folder structure")
    p_init.add_argument("dir", help="Project directory to set up")
    p_init.add_argument(
        "--yes", action="store_true", help="Auto-accept all detected entities (non-interactive)"
    )

    # mine
    p_mine = sub.add_parser("mine", help="Mine files into the palace")
    p_mine.add_argument("dir", help="Directory to mine")
    p_mine.add_argument(
        "--mode",
        choices=["projects", "convos"],
        default="projects",
        help="Ingest mode: 'projects' for code/docs (default), 'convos' for chat exports",
    )
    p_mine.add_argument("--wing", default=None, help="Wing name (default: directory name)")
    p_mine.add_argument(
        "--no-gitignore",
        action="store_true",
        help="Don't respect .gitignore files when scanning project files",
    )
    p_mine.add_argument(
        "--include-ignored",
        action="append",
        default=[],
        help="Always scan these project-relative paths even if ignored; repeat or pass comma-separated paths",
    )
    p_mine.add_argument(
        "--agent",
        default="psa",
        help="Your name — recorded on every drawer (default: psa)",
    )
    p_mine.add_argument("--limit", type=int, default=0, help="Max files to process (0 = all)")
    p_mine.add_argument(
        "--dry-run", action="store_true", help="Show what would be filed without filing"
    )
    p_mine.add_argument(
        "--extract",
        choices=["exchange", "general"],
        default="exchange",
        help="Extraction strategy for convos mode: 'exchange' (default) or 'general' (5 memory types)",
    )

    # search
    p_search = sub.add_parser("search", help="Find anything, exact words")
    p_search.add_argument("query", help="What to search for")
    p_search.add_argument("--wing", default=None, help="Limit to one project")
    p_search.add_argument("--room", default=None, help="Limit to one room")
    p_search.add_argument("--results", type=int, default=5, help="Number of results")

    # wake-up
    p_wakeup = sub.add_parser("wake-up", help="Show PSA atlas session wake-up context")
    p_wakeup.add_argument("--wing", default=None, help="(legacy, ignored in PSA mode)")

    # legacy — wraps old palace commands for backward compatibility
    p_legacy = sub.add_parser(
        "legacy",
        help="Run a legacy palace command (compress, wake-up). Set PSA_MODE=off for full legacy mode.",
    )
    legacy_sub = p_legacy.add_subparsers(dest="legacy_action")
    p_l_compress = legacy_sub.add_parser("compress", help="Compress drawers using AAAK Dialect")
    p_l_compress.add_argument("--wing", default=None)
    p_l_compress.add_argument("--dry-run", action="store_true")
    p_l_compress.add_argument("--config", default=None)
    p_l_wakeup = legacy_sub.add_parser("wake-up", help="Show L0 + L1 wake-up context")
    p_l_wakeup.add_argument("--wing", default=None)

    # split
    p_split = sub.add_parser(
        "split",
        help="Split concatenated transcript mega-files into per-session files (run before mine)",
    )
    p_split.add_argument("dir", help="Directory containing transcript files")
    p_split.add_argument(
        "--output-dir",
        default=None,
        help="Write split files here (default: same directory as source files)",
    )
    p_split.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be split without writing files",
    )
    p_split.add_argument(
        "--min-sessions",
        type=int,
        default=2,
        help="Only split files containing at least N sessions (default: 2)",
    )

    # atlas
    p_atlas = sub.add_parser("atlas", help="PSA atlas commands (build, status, health)")
    p_atlas.add_argument(
        "--tenant", default="default", help="Tenant identifier (default: 'default')"
    )
    atlas_sub = p_atlas.add_subparsers(dest="atlas_action")
    atlas_sub.add_parser("build", help="Build or rebuild the PSA atlas for the tenant")
    atlas_sub.add_parser("status", help="Show atlas version and anchor count")
    atlas_sub.add_parser("health", help="Show health report (novelty rate, utilization skew)")

    # benchmark
    p_benchmark = sub.add_parser(
        "benchmark", help="Compare PSA pipeline vs raw ChromaDB search"
    )
    p_benchmark.add_argument("--query", required=True, help="Query to compare")
    p_benchmark.add_argument(
        "--tenant", default="default", help="Tenant identifier (default: 'default')"
    )

    # migrate
    p_migrate = sub.add_parser(
        "migrate", help="Migrate ChromaDB palace to PSA MemoryStore (non-destructive)"
    )
    p_migrate.add_argument(
        "--palace", default=None, help="Path to ChromaDB palace (default: ~/.psa/palace)"
    )
    p_migrate.add_argument(
        "--tenant", default="default", help="PSA tenant to write into (default: 'default')"
    )
    p_migrate.add_argument(
        "--collection", default="mempalace", help="ChromaDB collection name (default: mempalace)"
    )

    # hook
    p_hook = sub.add_parser(
        "hook",
        help="Run hook logic (reads JSON from stdin, outputs JSON to stdout)",
    )
    hook_sub = p_hook.add_subparsers(dest="hook_action")
    p_hook_run = hook_sub.add_parser("run", help="Execute a hook")
    p_hook_run.add_argument(
        "--hook",
        required=True,
        choices=["session-start", "stop", "precompact"],
        help="Hook name to run",
    )
    p_hook_run.add_argument(
        "--harness",
        required=True,
        choices=["claude-code", "codex"],
        help="Harness type (determines stdin JSON format)",
    )

    # instructions
    p_instructions = sub.add_parser(
        "instructions",
        help="Output skill instructions to stdout",
    )
    instructions_sub = p_instructions.add_subparsers(dest="instructions_name")
    for instr_name in ["init", "search", "mine", "help", "status"]:
        instructions_sub.add_parser(instr_name, help=f"Output {instr_name} instructions")

    # repair
    sub.add_parser(
        "repair",
        help="Rebuild palace vector index from stored data (fixes segfaults after corruption)",
    )

    # status
    sub.add_parser("status", help="Show what's been filed")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Handle two-level subcommands
    if args.command == "atlas":
        if not getattr(args, "atlas_action", None):
            p_atlas.print_help()
            return
        cmd_atlas(args)
        return

    if args.command == "legacy":
        if not getattr(args, "legacy_action", None):
            p_legacy.print_help()
            return
        cmd_legacy(args)
        return

    if args.command == "hook":
        if not getattr(args, "hook_action", None):
            p_hook.print_help()
            return
        cmd_hook(args)
        return

    if args.command == "instructions":
        name = getattr(args, "instructions_name", None)
        if not name:
            p_instructions.print_help()
            return
        args.name = name
        cmd_instructions(args)
        return

    dispatch = {
        "init": cmd_init,
        "mine": cmd_mine,
        "split": cmd_split,
        "search": cmd_search,
        "wake-up": cmd_wakeup,
        "repair": cmd_repair,
        "status": cmd_status,
        "benchmark": cmd_benchmark,
        "migrate": cmd_migrate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
