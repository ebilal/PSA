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
        print(
            f"\n[{len(result.selected_anchors)} anchors selected, "
            f"{result.token_count} tokens, "
            f"{result.timing.total_ms:.0f}ms]"
        )
        return
    except FileNotFoundError:
        pass  # no atlas yet — fall through to raw search
    except Exception as e:
        import logging

        logging.getLogger("psa.cli").warning(
            "PSA pipeline failed, falling back to raw search: %s", e
        )

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
            print("No PSA atlas built. Run 'psa atlas build' first.")
            print("(For legacy wake-up: PSA_MODE=off psa legacy wake-up)")
            return

        total = sum(
            len(store.query_by_anchor(tenant_id=tenant_id, anchor_id=c.anchor_id, limit=100_000))
            for c in atlas.cards
        )
        learned = sum(1 for c in atlas.cards if not c.is_novelty)
        novelty = sum(1 for c in atlas.cards if c.is_novelty)
        print(f"\n── PSA Session Wake-Up (tenant: {tenant_id}) ──")
        print(
            f"  Atlas v{atlas.version}: {len(atlas.cards)} anchors "
            f"({learned} learned, {novelty} novelty)"
        )
        print(f"  Memories indexed: {total}")
        print("\nTop anchors by memory count:")
        sorted_cards = sorted(atlas.cards, key=lambda c: c.memory_count, reverse=True)
        for card in sorted_cards[:5]:
            print(f"  [{card.anchor_id}] {card.name} — {card.meaning[:60]}")
        print("\nUse 'psa search <query>' to retrieve memories.")
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
    if getattr(args, "backfill_facets", False):
        from .facet_extractor import extract_facets
        from .memory_object import MemoryStore
        from .tenant import TenantManager

        tenant_id = getattr(args, "tenant", "default")
        tm = TenantManager()
        tenant = tm.get_or_create(tenant_id)
        store = MemoryStore(db_path=tenant.memory_db_path)

        print(f"Backfilling facets for tenant '{tenant_id}'...")
        memories = store.get_all_active(tenant_id)
        print(f"  Found {len(memories)} active memories.")

        updated = 0
        for i, mo in enumerate(memories):
            # Try raw source first (more accurate), fall back to body
            raw_text = None
            for sid in mo.source_ids or []:
                src = store.get_source(sid)
                if src and src.full_text:
                    raw_text = src.full_text
                    break

            text = raw_text or mo.body
            facets = extract_facets(text)

            mo.entities = facets.entities
            mo.actor_entities = facets.actor_entities
            mo.speaker_role = facets.speaker_role
            mo.stance = facets.stance
            mo.mentioned_at = facets.mentioned_at
            store.update_facets(mo)
            updated += 1

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(memories)} memories updated")

        print(f"  Backfilled facets for {updated} memories.")
        return

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
        print("Usage: psa atlas {build,status,health,refine,promote-refinement}")
        return

    if action in ("build", "rebuild"):
        _cmd_atlas_build(args)
    elif action == "status":
        _cmd_atlas_status(args)
    elif action == "health":
        _cmd_atlas_health(args)
    elif action == "refine":
        _cmd_atlas_refine(args)
    elif action == "promote-refinement":
        _cmd_atlas_refine_promote(args)


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
        print(
            f"  Anchors: {len(atlas.cards)} ({sum(1 for c in atlas.cards if not c.is_novelty)} learned, "
            f"{sum(1 for c in atlas.cards if c.is_novelty)} novelty)"
        )
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
        print(
            f"  Anchors: {len(atlas.cards)} ({sum(1 for c in atlas.cards if not c.is_novelty)} learned, "
            f"{sum(1 for c in atlas.cards if c.is_novelty)} novelty)"
        )
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


def _cmd_atlas_refine(args):
    """Refine anchor cards into a candidate artifact (not inference-visible)."""
    import datetime as _dt
    import importlib.util
    import json
    import os
    from pathlib import Path

    from .tenant import TenantManager
    from .atlas import AtlasManager

    tenant_id = getattr(args, "tenant", "default")
    miss_log_path = args.miss_log
    max_patterns = getattr(args, "max_patterns", 20)
    source = getattr(args, "source", "manual") or "manual"

    if not os.path.exists(miss_log_path):
        print(f"  Error: miss log not found: {miss_log_path}")
        sys.exit(1)

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
    version = mgr.latest_version()
    if version is None:
        print(f"  Error: no atlas for tenant '{tenant_id}'. Run 'psa atlas build' first.")
        sys.exit(1)

    atlas_dir = Path(tenant.root_dir) / f"atlas_v{version}"
    base_cards_path = atlas_dir / "anchor_cards.json"
    candidate_path = atlas_dir / "anchor_cards_candidate.json"
    candidate_meta_path = atlas_dir / "anchor_cards_candidate.meta.json"

    if not base_cards_path.exists():
        print(f"  Error: {base_cards_path} not found.")
        sys.exit(1)

    script_path = Path(__file__).parent.parent / "scripts" / "refine_anchor_cards.py"
    if not script_path.exists():
        print(f"  Error: refinement script not found at {script_path}.")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("refine_anchor_cards", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with open(base_cards_path) as f:
        base_cards = json.load(f)

    misses = []
    with open(miss_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                misses.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not misses:
        print(f"  Warning: miss log {miss_log_path} has 0 valid entries.")
        print("  Nothing to refine — skipping write (no candidate produced).")
        return

    refined = mod.refine_cards(base_cards, misses, max_patterns=max_patterns)

    # Diff stats for metadata
    base_by_id = {c.get("anchor_id"): c for c in base_cards}
    n_anchors_touched = 0
    n_patterns_added = 0
    for r in refined:
        b = base_by_id.get(r.get("anchor_id"), {})
        before = set(b.get("generated_query_patterns", []) or [])
        after = set(r.get("generated_query_patterns", []) or [])
        delta = after - before
        if delta:
            n_anchors_touched += 1
            n_patterns_added += len(delta)

    with open(candidate_path, "w") as f:
        json.dump(refined, f, indent=2)

    meta = {
        "source": source,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "tenant_id": tenant_id,
        "atlas_version": version,
        "promoted": False,
        "promoted_at": None,
        "miss_log_path": str(miss_log_path),
        "n_anchors_touched": n_anchors_touched,
        "n_patterns_added": n_patterns_added,
    }
    with open(candidate_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Candidate written to {candidate_path}")
    print(f"  Metadata written to {candidate_meta_path}")
    print(
        f"  Source: {source}, anchors touched: {n_anchors_touched}, "
        f"patterns added: {n_patterns_added}"
    )
    print("  Run 'psa atlas promote-refinement' to make this candidate inference-visible.")


def _cmd_atlas_refine_promote(args):
    """Promote the current candidate refinement into the live refined artifact."""
    import datetime as _dt
    import json
    import shutil
    from pathlib import Path

    from .tenant import TenantManager
    from .atlas import AtlasManager

    tenant_id = getattr(args, "tenant", "default")
    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
    mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
    version = mgr.latest_version()
    if version is None:
        print(f"  Error: no atlas for tenant '{tenant_id}'. Run 'psa atlas build' first.")
        sys.exit(1)

    atlas_dir = Path(tenant.root_dir) / f"atlas_v{version}"
    candidate_path = atlas_dir / "anchor_cards_candidate.json"
    candidate_meta_path = atlas_dir / "anchor_cards_candidate.meta.json"
    refined_path = atlas_dir / "anchor_cards_refined.json"
    refined_meta_path = atlas_dir / "anchor_cards_refined.meta.json"

    if not candidate_path.exists():
        print(f"  Error: no candidate to promote at {candidate_path}.")
        print("  Run 'psa atlas refine --miss-log PATH' first.")
        sys.exit(1)

    # Copy cards verbatim
    shutil.copyfile(candidate_path, refined_path)

    # Rewrite meta with promoted=true / promoted_at=<now>, preserving source.
    if candidate_meta_path.exists():
        with open(candidate_meta_path) as f:
            meta = json.load(f)
    else:
        # Candidate without meta — rare but handle gracefully.
        meta = {
            "source": "unknown",
            "tenant_id": tenant_id,
            "atlas_version": version,
        }

    meta["promoted"] = True
    meta["promoted_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    with open(refined_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Promoted: {candidate_path} → {refined_path}")
    print(f"  Metadata: {refined_meta_path}")
    print(f"  Source: {meta.get('source', 'unknown')}, promoted_at: {meta['promoted_at']}")


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
            print("  Deleted training_data.jsonl")
        print("  Labels reset. Starting fresh.\n")

    # Count existing
    existing = 0
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            existing = sum(1 for line in f if line.strip())

    try:
        pipeline = PSAPipeline.from_tenant(
            tenant_id=tenant_id, psa_mode="primary", selector_mode="cosine"
        )
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
        print(
            f"No new queries to label. {existing} labels exist, {len(all_queries)} total queries found."
        )
        return

    from .llm import _load_config as _llm_config

    llm_cfg = _llm_config()
    llm_name = (
        llm_cfg.get("cloud_model")
        if llm_cfg.get("provider") != "local" and llm_cfg.get("cloud_api_key")
        else llm_cfg.get("local_model", "local")
    )
    print(
        f"Labeling {len(queries)} of {len(available)} available queries using {llm_name} ({existing} already labeled)..."
    )
    labeler = OracleLabeler(pipeline=pipeline, output_path=labels_path)
    labeled = 0
    for i, (qid, query_text) in enumerate(queries, 1):
        try:
            labeler.label(query_id=qid, query=query_text)
            labeled += 1
            if labeled == 1 or labeled % 10 == 0:
                print(f"  [{labeled}/{len(queries)}] scored...", flush=True)
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
    from .atlas import AtlasManager
    from .selector import check_training_gates
    from .training.data_generator import DataGenerator
    from .training.train_selector import SelectorTrainer

    tm = TenantManager()
    tenant = tm.get_or_create(tenant_id)
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
    examples_path = os.path.join(tenant.root_dir, "training", "training_data.jsonl")
    anchor_cards = {c.anchor_id: c.to_stable_card_text() for c in atlas.cards}
    gen = DataGenerator(oracle_labels_path=labels_path, anchor_cards=anchor_cards)
    n_written = gen.generate(output_path=examples_path, n_examples=max(1000, label_count * 20))
    print(f"  Generated {n_written} training examples.")

    # Query-grouped train/val split (no leakage)
    from .training.data_split import split_train_val

    train_path = os.path.join(tenant.root_dir, "training", "train_data.jsonl")
    val_path = os.path.join(tenant.root_dir, "training", "val_data.jsonl")
    split_stats = split_train_val(examples_path, train_path, val_path)
    print(
        f"  Split: {split_stats['n_train_queries']}/{split_stats['n_val_queries']} queries, "
        f"positive rate: train={split_stats['train_positive_rate']:.1%} "
        f"val={split_stats['val_positive_rate']:.1%}"
    )

    # Train
    output_dir = os.path.join(tenant.root_dir, "models", "selector_latest")
    trainer = SelectorTrainer(output_dir=output_dir, atlas_version=atlas.version)
    try:
        sv = trainer.train(train_data_path=train_path, val_data_path=val_path)
        print(f"  Selector trained → {sv.model_path}")

        # Activate
        from .lifecycle import LifecyclePipeline

        lp = LifecyclePipeline()
        lp._write_selector_mode(tenant.root_dir, "trained", sv.model_path)
        print("  Activated trained selector.")

        # Co-activation training
        if getattr(args, "coactivation", False):
            print("Training co-activation model...")
            from .embeddings import EmbeddingModel
            from .full_atlas_scorer import FullAtlasScorer
            from .training.coactivation_data import generate_coactivation_data
            from .training.train_coactivation import CoActivationTrainer

            fas = FullAtlasScorer.from_model_path(sv.model_path, atlas)
            emb = EmbeddingModel()
            coact_data_dir = os.path.join(tenant.root_dir, "training", "coactivation")
            n_coact = generate_coactivation_data(
                oracle_labels_path=labels_path,
                output_path=coact_data_dir,
                full_atlas_scorer=fas,
                embedding_model=emb,
                atlas=atlas,
            )
            print(f"  Generated {n_coact} co-activation training examples.")

            coact_output = os.path.join(tenant.root_dir, "models", "coactivation_latest")
            coact_trainer = CoActivationTrainer(output_dir=coact_output)
            coact_trainer.train(
                data_dir=coact_data_dir,
                n_anchors=len(atlas.cards),
            )
            print(f"  Co-activation model saved to {coact_output}")

        if getattr(args, "memory_scorer", False):
            print("Training memory scorer...")
            from .training.memory_scorer_data import generate_memory_scorer_data
            from .training.train_memory_scorer import MemoryScorerTrainer

            import glob as _glob

            results_dir = os.path.expanduser("~/.psa/benchmarks/longmemeval")
            results_files = sorted(_glob.glob(os.path.join(results_dir, "results_*.jsonl")))
            if not results_files:
                print("  No benchmark results found. Run benchmark first.")
            else:
                results_file = results_files[-1]
                scorer_data_path = os.path.join(
                    tenant.root_dir, "training", "memory_scorer_train.jsonl"
                )
                from .pipeline import PSAPipeline

                _pipeline = PSAPipeline.from_tenant(tenant_id=tenant_id)
                n_examples = generate_memory_scorer_data(
                    results_path=results_file,
                    output_path=scorer_data_path,
                    pipeline=_pipeline,
                    mode="benchmark",
                )
                print(f"  Generated {n_examples} memory scorer examples.")

                if n_examples >= 200:
                    scorer_output = os.path.join(tenant.root_dir, "models", "memory_scorer_latest")
                    MemoryScorerTrainer(output_dir=scorer_output).train(data_path=scorer_data_path)
                    print(f"  Memory scorer saved to {scorer_output}")
                else:
                    print(f"  Too few examples ({n_examples}). Need >= 200.")
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

    # Remove any previously loaded instance before loading
    subprocess.run(
        ["launchctl", "bootout", f"gui/{os.getuid()}/{plist_name}"],
        check=False,
        capture_output=True,
    )
    result = subprocess.run(
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", plist_path],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Warning: launchctl bootstrap failed: {result.stderr.strip()}")
    print(f"Installed launchd plist at {plist_path}")
    print(f"Lifecycle will run daily at {hour}:00 for tenant '{tenant_id}'.")
    print("Logs: ~/.psa/lifecycle.log")


def _lifecycle_uninstall():
    """Remove macOS launchd plist."""
    import subprocess

    plist_name = "com.psa.lifecycle"
    plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{plist_name}.plist")
    if os.path.exists(plist_path):
        subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}/{plist_name}"],
            check=False,
            capture_output=True,
        )
        os.remove(plist_path)
        print(f"Uninstalled launchd plist: {plist_path}")
    else:
        print("No launchd plist found. Nothing to uninstall.")


def cmd_inspect(args):
    """Inspect what context PSA injects for a query."""
    from .inspect import inspect_query

    query = args.query
    tenant_id = getattr(args, "tenant", "default")
    token_budget = getattr(args, "token_budget", 6000)
    verbose = getattr(args, "verbose", False)

    try:
        result = inspect_query(query, tenant_id=tenant_id, token_budget=token_budget)
        if verbose:
            print(result.render_verbose())
        else:
            print(result.render_brief())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'psa atlas build' to build an atlas first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def _print_log_diff(old: dict, new: dict) -> None:
    """Print a human-readable diff between two log entries."""
    print(f"DIFF: {old['run_id']} -> {new['run_id']}")
    print(f"Query: {new['query']!r}")
    old_tok = old.get("tokens_used", 0)
    new_tok = new.get("tokens_used", 0)
    print(f"Tokens: {old_tok} -> {new_tok} ({new_tok - old_tok:+d})")
    old_sel = set(old.get("selected_anchor_ids", []))
    new_sel = set(new.get("selected_anchor_ids", []))
    added = new_sel - old_sel
    removed = old_sel - new_sel
    if added:
        print(f"Anchors added:   {sorted(added)}")
    if removed:
        print(f"Anchors removed: {sorted(removed)}")
    if not added and not removed:
        print("Anchors: unchanged")


def cmd_log(args):
    """Manage the PSA query inspection log."""
    import json
    from .inspect import load_log

    tenant_id = getattr(args, "tenant", "default")
    action = getattr(args, "log_action", None)

    if action == "list" or action is None:
        n = getattr(args, "n", 20)
        entries = load_log(tenant_id=tenant_id)
        if not entries:
            print(f"No log entries for tenant '{tenant_id}'.")
            return
        print(f"Recent queries for tenant '{tenant_id}' (newest first):")
        for e in entries[:n]:
            ts = e.get("timestamp", "?")[:19].replace("T", " ")
            tokens = e.get("tokens_used", 0)
            budget = e.get("token_budget", 0)
            run_id = e.get("run_id", "?")
            query = e.get("query", "?")
            print(f"  [{ts}] {run_id}  {tokens}/{budget} tok  {query!r}")

    elif action == "show":
        run_id = args.run_id
        entries = load_log(tenant_id=tenant_id)
        match = next((e for e in entries if e.get("run_id") == run_id), None)
        if match is None:
            print(f"No log entry with run_id '{run_id}'.")
            sys.exit(1)
        print(json.dumps(match, indent=2))

    elif action == "diff":
        entries = load_log(tenant_id=tenant_id)
        query_filter = getattr(args, "query", None)
        run_id_a = getattr(args, "run_id_a", None)
        run_id_b = getattr(args, "run_id_b", None)

        if query_filter:
            matches = [e for e in entries if e.get("query") == query_filter]
            if len(matches) < 2:
                print(
                    f"Need at least 2 log entries for query {query_filter!r}. Found {len(matches)}."
                )
                sys.exit(1)
            e_new, e_old = matches[0], matches[1]
        elif run_id_a and run_id_b:
            e_old = next((e for e in entries if e.get("run_id") == run_id_a), None)
            e_new = next((e for e in entries if e.get("run_id") == run_id_b), None)
            if not e_old or not e_new:
                print("Could not find both run IDs.")
                sys.exit(1)
        else:
            print("Provide --query or two run IDs.")
            sys.exit(1)

        _print_log_diff(e_old, e_new)


def _cmd_longmemeval(args):
    """Handle longmemeval benchmark subcommands."""
    from .benchmarks.longmemeval import ingest, run, score

    lme_action = getattr(args, "lme_action", None)
    tenant_id = getattr(args, "tenant", "longmemeval_bench")

    if lme_action == "ingest":
        print(f"Ingesting LongMemEval sessions into tenant '{tenant_id}'...")
        ingest(tenant_id=tenant_id)
        print("Done. Run 'psa benchmark longmemeval run' to benchmark.")

    elif lme_action == "run":
        split = getattr(args, "split", "val")
        limit = getattr(args, "limit", None)
        selector_mode = getattr(args, "selector", "cosine")
        selector_model_path = getattr(args, "selector_model", None)
        max_k = getattr(args, "max_k", 6)
        min_k = getattr(args, "min_k", None)
        ce_budget = getattr(args, "ce_budget", None)
        rerank_only = getattr(args, "rerank_only", False)
        print(
            f"Running LongMemEval ({split} split, {'all' if not limit else limit} questions, "
            f"selector={selector_mode}, max_k={max_k}"
            f"{f', min_k={min_k}' if min_k else ''}"
            f"{f', ce_budget={ce_budget}' if ce_budget else ''}"
            f"{', rerank_only' if rerank_only else ''})..."
        )

        coactivation_pipeline = None
        if selector_mode == "coactivation":
            from .full_atlas_scorer import FullAtlasScorer
            from .coactivation import CoActivationSelector
            from .pipeline import PSAPipeline
            import os as _os

            # Build pipeline with cosine fallback, then attach coactivation components
            try:
                coactivation_pipeline = PSAPipeline.from_tenant(
                    tenant_id=tenant_id,
                    selector_mode="cosine",
                    selector_model_path=selector_model_path,
                    selector_max_k=max_k,
                    selector_min_k=min_k,
                    selector_rerank_only=rerank_only,
                )
            except FileNotFoundError:
                pass

            if coactivation_pipeline is not None:
                # Find selector model
                model_dir = getattr(args, "selector_model", None)
                if not model_dir:
                    model_dir = _os.path.join(
                        _os.path.expanduser(f"~/.psa/tenants/{tenant_id}/models/selector_latest"),
                        "selector_v1",
                    )
                if _os.path.exists(model_dir):
                    coactivation_pipeline.full_atlas_scorer = FullAtlasScorer.from_model_path(
                        model_dir, coactivation_pipeline.atlas
                    )
                coact_path = _os.path.join(
                    _os.path.expanduser(f"~/.psa/tenants/{tenant_id}/models"),
                    "coactivation_latest",
                )
                coact_meta = _os.path.join(coact_path, "coactivation_version.json")
                if _os.path.exists(coact_meta):
                    import torch

                    _dev = "mps" if torch.backends.mps.is_available() else "cpu"
                    coactivation_pipeline.coactivation_selector = (
                        CoActivationSelector.from_model_path(
                            coact_path,
                            device=_dev,
                            min_k=min_k if min_k is not None else max_k,
                            top_ce_budget=ce_budget,
                        )
                    )

        out_path = run(
            split=split,
            limit=limit,
            tenant_id=tenant_id,
            selector_mode=selector_mode,
            selector_model_path=selector_model_path,
            max_k=max_k,
            min_k=min_k,
            rerank_only=rerank_only,
            pipeline=coactivation_pipeline,
        )
        print(f"Results: {out_path}")
        print("Run 'psa benchmark longmemeval score' next.")

    elif lme_action == "score":
        results_file = getattr(args, "results", None)
        method = getattr(args, "method", "both")

        if not results_file:
            import glob

            results_dir = os.path.expanduser("~/.psa/benchmarks/longmemeval")
            files = sorted(glob.glob(os.path.join(results_dir, "results_*.jsonl")))
            if not files:
                print("No results files found. Run 'psa benchmark longmemeval run' first.")
                sys.exit(1)
            results_file = files[-1]
            print(f"Scoring: {results_file}")

        stats = score(results_file, method=method, tenant_id=tenant_id)
        print("\nLongMemEval Score")
        print(f"  Questions:     {stats['n_questions']}")
        print(f"  Exact F1:      {stats['exact_f1']:.3f}")
        if "recall_at_5" in stats:
            print(f"  R@5:           {stats['recall_at_5']:.3f}")
        if "llm_score" in stats:
            print(f"  LLM-as-judge:  {stats['llm_score']:.3f}")
        if "anchor_count_distribution" in stats:
            dist = stats["anchor_count_distribution"]
            print(f"  Anchor distribution: {dict(sorted(dist.items()))}")
        if "gold_hit_rate" in stats:
            print(f"  Gold-hit rate: {stats['gold_hit_rate']:.3f}")
        print(f"\nOracle labels written: {stats['oracle_labels_written']}")
        print(f"  -> {stats['oracle_labels_path']}")
        print("\nRun 'psa train' to train the selector on these labels.")

    elif lme_action == "oracle-label":
        from .benchmarks.longmemeval import oracle_label

        results_file = getattr(args, "results", None)
        if not results_file:
            import glob

            results_dir = os.path.expanduser("~/.psa/benchmarks/longmemeval")
            files = sorted(glob.glob(os.path.join(results_dir, "results_*.jsonl")))
            if not files:
                print("No results files found. Run 'psa benchmark longmemeval run' first.")
                sys.exit(1)
            results_file = files[-1]

        lme_mode = getattr(args, "mode", "fast")
        _mode_desc = {
            "fast": "fast (gold-anchor overlap, no LLM — ~2 min)",
            "local": "local Ollama LLM — 30-60 min",
            "api": "cloud API LLM — 20-40 min",
        }
        print(f"Running oracle labeling on: {results_file}")
        print(f"Mode: {_mode_desc.get(lme_mode, lme_mode)}")
        n = oracle_label(results_file, tenant_id=tenant_id, mode=lme_mode)
        print(f"\nOracle labels written: {n}")
        print("Run 'psa train' to train the selector on these labels.")

    else:
        print("Usage: psa benchmark longmemeval ingest|run|score|oracle-label")
        sys.exit(1)


def cmd_benchmark(args):
    """Benchmark commands — longmemeval harness or quick PSA vs ChromaDB comparison."""
    bench_cmd = getattr(args, "bench_cmd", None)
    if bench_cmd == "longmemeval":
        _cmd_longmemeval(args)
        return

    query = getattr(args, "query", None)
    if not query:
        print("Usage: psa benchmark --query 'your query here'")
        print("       psa benchmark longmemeval ingest|run|score")
        return

    print("PSA benchmark mode — comparing PSA pipeline vs raw ChromaDB search.")
    print("(Requires a populated palace and atlas. Run 'psa mine' and 'psa atlas build' first.)")

    tenant_id = getattr(args, "tenant", "default")
    try:
        from .searcher import search_memories
        from .config import MempalaceConfig

        cfg = MempalaceConfig()

        print("\n--- Raw ChromaDB search ---")
        raw_results = search_memories(query, n_results=5, palace_path=cfg.palace_path)
        for i, r in enumerate(raw_results.get("results", []), 1):
            print(f"  [{i}] {r.get('title', '?')} ({r.get('similarity', 0):.3f})")

        print("\n--- PSA pipeline search ---")
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
        print("\n  Migration complete:")
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

        palace_path = (
            os.path.expanduser(args.palace)
            if getattr(args, "palace", None)
            else MempalaceConfig().palace_path
        )
        stack = MemoryStack(palace_path=palace_path)
        text = stack.wake_up(wing=getattr(args, "wing", None))
        tokens = len(text) // 4
        print(f"Wake-up text (~{tokens} tokens):")
        print("=" * 50)
        print(text)
    except ImportError:
        print(
            "Legacy wake-up requires the MemoryStack layer (layers.py). "
            "This file has been removed in PSA primary mode."
        )
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
    p_atlas_refine = atlas_sub.add_parser(
        "refine",
        help="Refine anchor card generated_query_patterns from a miss log",
    )
    p_atlas_refine.add_argument(
        "--miss-log",
        required=True,
        help="Path to a JSONL miss log produced by benchmark runs",
    )
    p_atlas_refine.add_argument(
        "--max-patterns",
        type=int,
        default=20,
        help="Max total generated_query_patterns per anchor after refinement (default: 20)",
    )
    p_atlas_refine.add_argument(
        "--source",
        default="manual",
        help=(
            "Provenance marker for this refinement — one of "
            "'manual', 'benchmark', 'oracle', 'query_fingerprint'. "
            "Defaults to 'manual'. Stored in the candidate's .meta.json and "
            "preserved verbatim on promotion."
        ),
    )

    atlas_sub.add_parser(
        "promote-refinement",
        help=(
            "Promote the current atlas candidate (anchor_cards_candidate.json) to "
            "the live refined artifact (anchor_cards_refined.json). This is the "
            "only write path to the live refined file."
        ),
    )

    # lifecycle
    p_lifecycle = sub.add_parser(
        "lifecycle", help="PSA lifecycle management (run, status, install, uninstall)"
    )
    p_lifecycle.add_argument(
        "--tenant", default="default", help="Tenant identifier (default: 'default')"
    )
    lifecycle_sub = p_lifecycle.add_subparsers(dest="lifecycle_action")
    p_lc_run = lifecycle_sub.add_parser("run", help="Run lifecycle pipeline manually")
    p_lc_run.add_argument("--force-rebuild", action="store_true", help="Force atlas rebuild")
    p_lc_run.add_argument(
        "--label-batch",
        type=int,
        default=0,
        help="Max queries to label per run (0 = all remaining up to 300, default: 0)",
    )
    lifecycle_sub.add_parser("status", help="Show lifecycle state")
    p_lc_install = lifecycle_sub.add_parser(
        "install", help="Install macOS launchd plist for nightly runs"
    )
    p_lc_install.add_argument(
        "--label-batch",
        type=int,
        default=0,
        help="Max queries to label per nightly run (0 = all remaining, 30 = ~90 min)",
    )
    lifecycle_sub.add_parser("uninstall", help="Remove macOS launchd plist")

    # label
    p_label = sub.add_parser("label", help="Run oracle labeling — score anchor sets for queries")
    p_label.add_argument("--tenant", default="default", help="Tenant identifier")
    p_label.add_argument(
        "--n-queries",
        type=int,
        default=0,
        help="Number of queries to label (0 = all available, default: all)",
    )
    p_label.add_argument(
        "--sessions-dir",
        default=None,
        help="Path to Claude Code sessions (default: ~/.claude/projects)",
    )
    p_label.add_argument(
        "--reset", action="store_true", help="Delete all existing labels and start from scratch"
    )

    # train
    p_train = sub.add_parser("train", help="Train the selector model from oracle labels")
    p_train.add_argument("--tenant", default="default", help="Tenant identifier")
    p_train.add_argument("--force", action="store_true", help="Train even if gates are not met")
    p_train.add_argument(
        "--coactivation",
        action="store_true",
        help="Also train co-activation model after selector (requires trained selector)",
    )
    p_train.add_argument(
        "--memory-scorer",
        action="store_true",
        help="Also train memory-level re-ranker (requires benchmark results)",
    )

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect what context PSA injects for a query")
    p_inspect.add_argument("query", help="Query string to inspect")
    p_inspect.add_argument(
        "--tenant", default="default", help="Tenant identifier (default: 'default')"
    )
    p_inspect.add_argument("--token-budget", dest="token_budget", type=int, default=6000)
    p_inspect.add_argument(
        "--verbose", action="store_true", help="Show full trace with all candidates"
    )

    # log
    p_log = sub.add_parser("log", help="Manage the PSA query inspection log")
    p_log.add_argument("--tenant", default="default", help="Tenant identifier (default: 'default')")
    log_sub = p_log.add_subparsers(dest="log_action")
    p_log_list = log_sub.add_parser("list", help="List recent logged queries")
    p_log_list.add_argument("-n", type=int, default=20)
    p_log_show = log_sub.add_parser("show", help="Show a log entry by run ID")
    p_log_show.add_argument("run_id")
    p_log_diff = log_sub.add_parser(
        "diff", help="Diff two log entries (run_id_a is baseline, run_id_b is newer)"
    )
    p_log_diff.add_argument("--query", default=None)
    p_log_diff.add_argument("run_id_a", nargs="?", default=None)
    p_log_diff.add_argument("run_id_b", nargs="?", default=None)

    # benchmark
    p_benchmark = sub.add_parser(
        "benchmark", help="Benchmark commands (longmemeval harness or --query for quick comparison)"
    )
    p_benchmark.add_argument(
        "--query", default=None, help="Quick query comparison (PSA vs ChromaDB)"
    )
    p_benchmark.add_argument(
        "--tenant", default="default", help="Tenant identifier (default: 'default')"
    )
    bench_sub = p_benchmark.add_subparsers(dest="bench_cmd")

    p_lme = bench_sub.add_parser("longmemeval", help="LongMemEval benchmark harness")
    p_lme.add_argument(
        "--tenant",
        default="longmemeval_bench",
        help="Tenant for benchmark data (default: longmemeval_bench)",
    )
    lme_sub = p_lme.add_subparsers(dest="lme_action")
    lme_sub.add_parser("ingest", help="Download dataset and ingest sessions into PSA")
    p_lme_run = lme_sub.add_parser("run", help="Run questions through PSA, generate answers")
    p_lme_run.add_argument("--split", default="val", choices=["val", "test"])
    p_lme_run.add_argument("--limit", type=int, default=None)
    p_lme_run.add_argument(
        "--selector",
        default="cosine",
        choices=["cosine", "trained", "coactivation"],
        help="Selector mode: cosine (default) or trained cross-encoder",
    )
    p_lme_run.add_argument(
        "--selector-model",
        default=None,
        help="Path to trained selector model (auto-detected if omitted)",
    )
    p_lme_run.add_argument(
        "--max-k",
        type=int,
        default=6,
        help="Maximum anchors to select (default: 6)",
    )
    p_lme_run.add_argument(
        "--min-k",
        type=int,
        default=None,
        help="Minimum anchors to select (backfill from top-scored if threshold filters too many)",
    )
    p_lme_run.add_argument(
        "--ce-budget",
        type=int,
        default=None,
        dest="ce_budget",
        help=(
            "For coactivation selector: restrict model input to top-N CE-ranked anchors "
            "(default: use all anchors). E.g. --ce-budget 48."
        ),
    )
    p_lme_run.add_argument(
        "--rerank-only",
        action="store_true",
        default=False,
        help="Ignore threshold, return top max-k reranked by cross-encoder",
    )
    p_lme_score = lme_sub.add_parser("score", help="Score answers and write oracle labels")
    p_lme_score.add_argument("--results", default=None)
    p_lme_score.add_argument("--method", default="both", choices=["exact", "llm", "both"])
    p_lme_oracle = lme_sub.add_parser("oracle-label", help="Run oracle labeling on results")
    p_lme_oracle.add_argument("--results", default=None, help="Results JSONL (auto-detects latest)")
    p_lme_oracle.add_argument(
        "--mode",
        default="fast",
        choices=["fast", "local", "api"],
        help=(
            "Labeling strategy: 'fast' = gold-anchor overlap only (no LLM, ~2 min); "
            "'local' = full two-stage via local Ollama (~30-60 min); "
            "'api' = full two-stage via cloud API (~20-40 min). "
            "Default: fast."
        ),
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
    p_repair = sub.add_parser(
        "repair",
        help="Rebuild palace vector index from stored data (fixes segfaults after corruption)",
    )
    p_repair.add_argument(
        "--backfill-facets",
        action="store_true",
        help="Extract facets (entities, temporal, speaker, stance) for existing memories from raw sources",
    )
    p_repair.add_argument(
        "--tenant",
        default="default",
        help="Tenant to backfill (default: 'default')",
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
        "inspect": cmd_inspect,
        "log": cmd_log,
        "benchmark": cmd_benchmark,
        "migrate": cmd_migrate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
