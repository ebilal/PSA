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
    except Exception as e:
        import logging
        logging.getLogger("psa.cli").warning("PSA pipeline failed, falling back to raw search: %s", e)

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

    if action in ("build", "rebuild"):
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


def cmd_label(args):
    """Run oracle labeling — score anchor sets for queries using Qwen."""
    tenant_id = getattr(args, "tenant", "default")
    n_queries = getattr(args, "n_queries", 50)
    sessions_dir = getattr(args, "sessions_dir", None) or os.path.expanduser("~/.claude/projects")

    from .tenant import TenantManager
    from .pipeline import PSAPipeline
    from .training.oracle_labeler import OracleLabeler, _load_queries_from_sessions

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    labels_path = os.path.join(tenant.root_dir, "training", "oracle_labels.jsonl")

    # Reset: delete all existing labels
    if getattr(args, "reset", False):
        import glob
        label_files = glob.glob(os.path.join(tenant.root_dir, "training", "oracle_labels*.jsonl"))
        for f in label_files:
            os.remove(f)
            print(f"  Deleted {os.path.basename(f)}")
        training_data = os.path.join(tenant.root_dir, "training", "training_data.jsonl")
        if os.path.exists(training_data):
            os.remove(training_data)
            print(f"  Deleted training_data.jsonl")
        print("  Labels reset. Starting fresh.\n")

    # Count existing
    existing = 0
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            existing = sum(1 for line in f if line.strip())

    try:
        pipeline = PSAPipeline.from_tenant(tenant_id=tenant_id, psa_mode="primary", selector_mode="cosine")
    except FileNotFoundError:
        print(f"No atlas for tenant '{tenant_id}'. Run 'psa atlas build' first.")
        return

    all_queries = _load_queries_from_sessions(sessions_dir)

    # Deduplicate against already-labeled
    import json
    labeled_ids = set()
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        labeled_ids.add(json.loads(line).get("query_id", ""))
                    except Exception:
                        pass
    available = [(qid, qt) for qid, qt in all_queries if qid not in labeled_ids]
    queries = available[:n_queries] if n_queries > 0 else available

    if not queries:
        print(f"No new queries to label. {existing} labels exist, {len(all_queries)} total queries found.")
        return

    from .llm import _load_config as _llm_config
    llm_cfg = _llm_config()
    llm_name = llm_cfg.get("cloud_model") if llm_cfg.get("provider") != "local" and llm_cfg.get("cloud_api_key") else llm_cfg.get("local_model", "local")
    print(f"Labeling {len(queries)} of {len(available)} available queries using {llm_name} ({existing} already labeled)...")
    labeler = OracleLabeler(pipeline=pipeline, output_path=labels_path)
    labeled = 0
    for i, (qid, query_text) in enumerate(queries, 1):
        try:
            labeler.label(query_id=qid, query=query_text)
            labeled += 1
            if labeled % 5 == 0 or labeled == len(queries):
                print(f"  Labeled {labeled}/{len(queries)}...", flush=True)
        except Exception as e:
            print(f"  Failed to label query: {e}")

    total = existing + labeled
    can_train = "ready to train" if total >= 300 else f"{300 - total} more needed to train"
    print(f"\nDone. {labeled} new labels. Total: {total} ({can_train}).")


def cmd_train(args):
    """Train the selector model from oracle labels."""
    tenant_id = getattr(args, "tenant", "default")
    force = getattr(args, "force", False)

    from .tenant import TenantManager
    from .memory_object import MemoryStore
    from .atlas import AtlasManager
    from .selector import check_training_gates
    from .training.data_generator import DataGenerator
    from .training.train_selector import SelectorTrainer

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    store = MemoryStore(db_path=tenant.memory_db_path)
    atlas_mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
    atlas = atlas_mgr.get_atlas()

    if atlas is None:
        print(f"No atlas for tenant '{tenant_id}'. Run 'psa atlas build' first.")
        return

    labels_path = os.path.join(tenant.root_dir, "training", "oracle_labels.jsonl")
    if not os.path.exists(labels_path):
        print("No oracle labels found. Run 'psa label' first.")
        return

    with open(labels_path) as f:
        label_count = sum(1 for line in f if line.strip())

    # Check gates
    if not force:
        gate_status = check_training_gates(
            oracle_count=label_count,
            held_out_count=0,
            shortlist_recall_24=1.0,
        )
        if not gate_status.gates_met:
            print(f"Training gates not met ({label_count}/300 labels):")
            for reason in gate_status.blocking_reasons:
                print(f"  - {reason}")
            print("\nUse --force to train anyway.")
            return

    print(f"Training selector from {label_count} oracle labels...")

    # Generate training data
    training_path = os.path.join(tenant.root_dir, "training", "training_data.jsonl")
    anchor_cards = {c.anchor_id: c.to_stable_card_text() for c in atlas.cards}
    gen = DataGenerator(oracle_labels_path=labels_path, anchor_cards=anchor_cards)
    n_written = gen.generate(output_path=training_path, n_examples=max(1000, label_count * 20))
    print(f"  Generated {n_written} training examples.")

    # Train
    output_dir = os.path.join(tenant.root_dir, "models", "selector_latest")
    trainer = SelectorTrainer(output_dir=output_dir, atlas_version=atlas.version)
    try:
        sv = trainer.train(train_data_path=training_path)
        print(f"  Selector trained → {sv.model_path}")

        # Activate
        from .lifecycle import LifecyclePipeline
        lp = LifecyclePipeline()
        lp._write_selector_mode(tenant.root_dir, "trained", sv.model_path)
        print(f"  Activated trained selector.")
    except Exception as e:
        print(f"  Training failed: {e}")


def cmd_lifecycle(args):
    """Handle 'psa lifecycle <subcommand>'."""
    action = getattr(args, "lifecycle_action", None)
    if not action:
        print("Usage: psa lifecycle {run,status,install,uninstall}")
        return

    tenant_id = getattr(args, "tenant", "default")

    if action == "run":
        from .lifecycle import LifecyclePipeline
        lp = LifecyclePipeline()
        force = getattr(args, "force_rebuild", False)
        label_batch = getattr(args, "label_batch", 0)
        lp.run(tenant_id=tenant_id, force_rebuild=force, label_batch_size=label_batch)

    elif action == "status":
        from .lifecycle import LifecyclePipeline
        lp = LifecyclePipeline()
        state = lp.status(tenant_id=tenant_id)
        print(f"\nLifecycle state (tenant: {tenant_id}):")
        if not state:
            print("  (no lifecycle runs yet)")
        else:
            for k, v in sorted(state.items()):
                print(f"  {k}: {v}")

    elif action == "install":
        label_batch = getattr(args, "label_batch", 0)
        _lifecycle_install(tenant_id, label_batch=label_batch)

    elif action == "uninstall":
        _lifecycle_uninstall()


def _lifecycle_install(tenant_id: str, label_batch: int = 0):
    """Install macOS launchd plist for nightly lifecycle runs."""
    import subprocess
    import sys
    cfg = MempalaceConfig()
    hour = cfg.nightly_hour
    plist_name = "com.psa.lifecycle"
    plist_dir = os.path.expanduser("~/Library/LaunchAgents")
    plist_path = os.path.join(plist_dir, f"{plist_name}.plist")

    psa_bin = os.path.join(os.path.dirname(sys.executable), "psa")
    if not os.path.exists(psa_bin):
        psa_bin = "psa"

    label_batch_args = ""
    if label_batch > 0:
        label_batch_args = f"""
        <string>--label-batch</string>
        <string>{label_batch}</string>"""

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{plist_name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{psa_bin}</string>
        <string>lifecycle</string>
        <string>run</string>
        <string>--tenant</string>
        <string>{tenant_id}</string>{label_batch_args}
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{os.path.expanduser("~/.psa/lifecycle.log")}</string>
    <key>StandardErrorPath</key>
    <string>{os.path.expanduser("~/.psa/lifecycle.log")}</string>
</dict>
</plist>
"""
    os.makedirs(plist_dir, exist_ok=True)
    with open(plist_path, "w") as f:
        f.write(plist_content)

    subprocess.run(["launchctl", "load", plist_path], check=False)
    print(f"Installed launchd plist at {plist_path}")
    print(f"Lifecycle will run daily at {hour}:00 for tenant '{tenant_id}'.")
    print(f"Logs: ~/.psa/lifecycle.log")


def _lifecycle_uninstall():
    """Remove macOS launchd plist."""
    import subprocess
    plist_name = "com.psa.lifecycle"
    plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{plist_name}.plist")
    if os.path.exists(plist_path):
        subprocess.run(["launchctl", "unload", plist_path], check=False)
        os.remove(plist_path)
        print(f"Uninstalled launchd plist: {plist_path}")
    else:
        print("No launchd plist found. Nothing to uninstall.")


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
    atlas_sub.add_parser("rebuild", help="Force rebuild the atlas (same as build)")

    # lifecycle
    p_lifecycle = sub.add_parser("lifecycle", help="PSA lifecycle management (run, status, install, uninstall)")
    p_lifecycle.add_argument(
        "--tenant", default="default", help="Tenant identifier (default: 'default')"
    )
    lifecycle_sub = p_lifecycle.add_subparsers(dest="lifecycle_action")
    p_lc_run = lifecycle_sub.add_parser("run", help="Run lifecycle pipeline manually")
    p_lc_run.add_argument("--force-rebuild", action="store_true", help="Force atlas rebuild")
    p_lc_run.add_argument("--label-batch", type=int, default=0,
                          help="Max queries to label per run (0 = all remaining up to 300, default: 0)")
    lifecycle_sub.add_parser("status", help="Show lifecycle state")
    p_lc_install = lifecycle_sub.add_parser("install", help="Install macOS launchd plist for nightly runs")
    p_lc_install.add_argument("--label-batch", type=int, default=0,
                              help="Max queries to label per nightly run (0 = all remaining, 30 = ~90 min)")
    lifecycle_sub.add_parser("uninstall", help="Remove macOS launchd plist")

    # label
    p_label = sub.add_parser("label", help="Run oracle labeling — score anchor sets for queries")
    p_label.add_argument("--tenant", default="default", help="Tenant identifier")
    p_label.add_argument("--n-queries", type=int, default=0, help="Number of queries to label (0 = all available, default: all)")
    p_label.add_argument("--sessions-dir", default=None, help="Path to Claude Code sessions (default: ~/.claude/projects)")
    p_label.add_argument("--reset", action="store_true", help="Delete all existing labels and start from scratch")

    # train
    p_train = sub.add_parser("train", help="Train the selector model from oracle labels")
    p_train.add_argument("--tenant", default="default", help="Tenant identifier")
    p_train.add_argument("--force", action="store_true", help="Train even if gates are not met")

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

    if args.command == "label":
        cmd_label(args)
        return

    if args.command == "train":
        cmd_train(args)
        return

    if args.command == "lifecycle":
        if not getattr(args, "lifecycle_action", None):
            p_lifecycle.print_help()
            return
        cmd_lifecycle(args)
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
