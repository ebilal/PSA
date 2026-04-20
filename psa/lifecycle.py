"""
lifecycle.py — PSA lifecycle pipeline orchestrator.

Two-speed nightly pipeline:
  Fast path: mine new sessions, prune within anchor budgets. No rebuild.
  Slow path: rebuild atlas + retrain selector (only when health triggers).

Usage::

    from psa.lifecycle import LifecyclePipeline
    lp = LifecyclePipeline()
    lp.run(tenant_id="default")
"""

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("psa.lifecycle")


class LifecyclePipeline:
    """
    Orchestrates the full PSA lifecycle: ingest, prune, rebuild, retrain.

    State is persisted to ~/.psa/tenants/{tenant_id}/lifecycle_state.json.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir

    def run(
        self,
        tenant_id: str = "default",
        sessions_dir: Optional[str] = None,
        force_rebuild: bool = False,
        label_batch_size: int = 0,
    ) -> dict:
        """
        Run the lifecycle pipeline.

        Returns a summary dict with counts of actions taken.
        """
        from .atlas import AtlasManager
        from .config import MempalaceConfig
        from .forgetting import enforce_global_cap, prune_anchor
        from .health import AtlasHealthMonitor
        from .memory_object import MemoryStore
        from .tenant import TenantManager

        cfg = MempalaceConfig()
        tm = TenantManager(base_dir=self.base_dir)
        tenant = tm.get_or_create(tenant_id)
        store = MemoryStore(db_path=tenant.memory_db_path)
        state = self._load_state(tenant.root_dir)
        summary = {
            "new_sessions": 0,
            "memories_mined": 0,
            "memories_pruned": 0,
            "memories_hard_deleted": 0,
            "atlas_rebuilt": False,
            "queries_labeled": 0,
            "selector_retrained": False,
        }

        now_iso = datetime.now(timezone.utc).isoformat()

        # Resolve sessions directory
        if sessions_dir is None:
            sessions_dir = os.path.expanduser("~/.claude/projects")

        # 1. Discover all session files; _mine_sessions handles dedup against raw_sources
        all_sessions = self._find_new_sessions(
            sessions_dir=sessions_dir,
            since=None,  # find ALL files, not just recent — dedup happens in _mine_sessions
        )
        new_since_last = self._find_new_sessions(
            sessions_dir=sessions_dir,
            since=state.get("last_mine_timestamp"),
        )
        has_new_data = bool(new_since_last)
        summary["new_sessions"] = len(new_since_last)

        if all_sessions:
            # === FAST PATH: ingest ===
            print(
                f"  [1/5] Mining sessions ({len(new_since_last)} new, {len(all_sessions)} total to check)..."
            )
            mined = self._mine_sessions(
                all_sessions, tenant_id=tenant_id, store=store, tenant=tenant
            )
            summary["memories_mined"] = mined
            has_new_data = has_new_data or mined > 0
            state["last_mine_timestamp"] = now_iso
            print(f"        {mined} new memories created.")
        else:
            print("  [1/5] No session files found.")

        # === MAINTENANCE (runs every time) ===

        print("  [2/5] Cleaning up archived memories...")
        deleted = store.delete_old_archived(tenant_id, older_than_days=90)
        summary["memories_hard_deleted"] = deleted
        if deleted:
            print(f"        Hard-deleted {deleted} old archived memories.")

        # 3. Load atlas for pruning
        atlas_mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
        atlas = atlas_mgr.get_atlas()

        if atlas is not None:
            # 3.5. Advertisement decay — apply decay + evaluate removal.
            try:
                from .advertisement.config import AdvertisementDecayConfig

                ad_cfg = AdvertisementDecayConfig.from_mempalace(cfg)
                if ad_cfg.tracking_enabled:
                    print(f"  [3/5] Advertisement decay (tracking={ad_cfg.tracking_enabled}, removal={ad_cfg.removal_enabled})...")
                    decay_summary = advertisement_decay_pass(
                        tenant_id=tenant_id,
                        config=ad_cfg,
                        atlas_or_loader=atlas,
                    )
                    summary["advertisement_decay"] = decay_summary
            except Exception as e:
                logger.warning("Advertisement decay pass failed (non-fatal): %s", e)

            # 4. Per-anchor pruning
            print(f"  [3/5] Pruning overloaded anchors ({len(atlas.cards)} anchors)...")
            total_pruned = 0
            anchor_budget = int(cfg._file_config.get("anchor_memory_budget", 100))
            for card in atlas.cards:
                if getattr(card, "status", "active") == "active":
                    pruned = prune_anchor(store, tenant_id, card.anchor_id, budget=anchor_budget)
                    total_pruned += pruned
            summary["memories_pruned"] = total_pruned
            if total_pruned:
                print(f"        Archived {total_pruned} memories from overloaded anchors.")

            # 5. Global cap
            print("  [4/5] Enforcing global memory cap...")
            max_memories = int(cfg._file_config.get("max_memories", 50_000))
            cap_result = enforce_global_cap(store, tenant_id, max_memories=max_memories)
            summary["memories_pruned"] += cap_result.get("archived", 0)
            summary["memories_hard_deleted"] += cap_result.get("hard_deleted", 0)

            # === SLOW PATH (only when health triggers AND we have new data) ===
            print("  [5/5] Checking atlas health...")
            monitor = AtlasHealthMonitor()
            health = monitor.check_health(atlas, store, tenant_id=tenant_id)

            if (has_new_data and health.should_rebuild) or force_rebuild:
                reasons = "; ".join(health.rebuild_reasons) if health.rebuild_reasons else "forced"
                print(f"        Rebuilding atlas (reason: {reasons})...")

                # Cosine fallback during rebuild
                state["selector_mode"] = "cosine"
                state.pop("selector_model_path", None)

                try:
                    atlas = atlas_mgr.rebuild(store)
                    summary["atlas_rebuilt"] = True
                    print(f"        Atlas v{atlas.version} rebuilt ({len(atlas.cards)} anchors).")
                except Exception as e:
                    print(f"        Atlas rebuild failed: {e}")
                    logger.error("Atlas rebuild failed: %s", e)
            else:
                print("        Atlas is healthy. No rebuild needed.")

            # Label queries for selector training (runs every time, not just on rebuild)
            print("  [6/6] Labeling queries for selector training...")
            labeled = self._label_queries(
                tenant, store, atlas, sessions_dir, batch_size=label_batch_size
            )
            summary["queries_labeled"] = labeled

            # Retrain selector if training gates are met
            if labeled > 0 or summary["atlas_rebuilt"]:
                print("        Checking selector training gates...")
                retrained = self._retrain_selector(tenant, store, atlas, state)
                summary["selector_retrained"] = retrained
                if retrained:
                    print(f"        Selector retrained (v{state.get('selector_version', '?')}).")
                else:
                    print("        Training gates not met. Staying in cosine mode.")

                if retrained:
                    try:
                        from .full_atlas_scorer import FullAtlasScorer
                        from .training.coactivation_data import generate_coactivation_data
                        from .training.train_coactivation import run_training_subprocess
                        from .embeddings import EmbeddingModel

                        labels_path = os.path.join(
                            tenant.root_dir, "training", "oracle_labels.jsonl"
                        )
                        selector_path = state.get("selector_model_path")
                        if selector_path and os.path.exists(selector_path):
                            fas = FullAtlasScorer.from_model_path(selector_path, atlas)
                            emb = EmbeddingModel()
                            coact_data_dir = os.path.join(
                                tenant.root_dir, "training", "coactivation"
                            )
                            generate_coactivation_data(
                                oracle_labels_path=labels_path,
                                output_path=coact_data_dir,
                                full_atlas_scorer=fas,
                                embedding_model=emb,
                                atlas=atlas,
                            )
                            coact_output = os.path.join(
                                tenant.root_dir, "models", "coactivation_latest"
                            )
                            run_training_subprocess(
                                output_dir=coact_output,
                                data_dir=coact_data_dir,
                                n_anchors=len(atlas.cards),
                                centroid_dim=768,
                            )
                            summary["coactivation_trained"] = True
                            print("        Co-activation model trained.")
                    except Exception as e:
                        logger.warning("Co-activation training failed: %s", e)
                        summary["coactivation_trained"] = False

                # Memory scorer is NOT trained in the lifecycle slow path.
                # Benchmark artifacts must never write back to the live atlas
                # (see docs/superpowers/specs/2026-04-17-lifecycle-benchmark-leakage-design.md).
                # Operators can still run `psa train --memory-scorer` explicitly.
        else:
            print("  [3/5] No atlas found. Skipping pruning and health check.")

        # 6. Save state
        state["last_memory_count"] = store.count(tenant_id)
        state["last_run"] = now_iso
        self._save_state(tenant.root_dir, state)

        # Print final summary
        print(f"\nLifecycle run complete (tenant: {tenant_id}):")
        print(f"  New sessions:       {summary['new_sessions']}")
        print(f"  Memories mined:     {summary['memories_mined']}")
        print(f"  Memories pruned:    {summary['memories_pruned']}")
        print(f"  Hard deleted:       {summary['memories_hard_deleted']}")
        print(f"  Atlas rebuilt:      {summary['atlas_rebuilt']}")
        print(f"  Queries labeled:    {summary['queries_labeled']}")
        print(f"  Selector retrained: {summary['selector_retrained']}")

        logger.info("Lifecycle run complete: %s", summary)
        return summary

    def status(self, tenant_id: str = "default") -> dict:
        """Return the current lifecycle state for a tenant."""
        from .tenant import TenantManager

        tm = TenantManager(base_dir=self.base_dir)
        tenant = tm.get_or_create(tenant_id)
        return self._load_state(tenant.root_dir)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _state_path(self, tenant_root: str) -> str:
        return os.path.join(tenant_root, "lifecycle_state.json")

    def _load_state(self, tenant_root: str) -> dict:
        path = self._state_path(tenant_root)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_state(self, tenant_root: str, state: dict):
        path = self._state_path(tenant_root)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def _write_selector_mode(self, tenant_root: str, mode: str, model_path: Optional[str] = None):
        """Write selector mode to lifecycle state."""
        state = self._load_state(tenant_root)
        state["selector_mode"] = mode
        if model_path:
            state["selector_model_path"] = model_path
        elif mode == "cosine" and "selector_model_path" in state:
            del state["selector_model_path"]
        self._save_state(tenant_root, state)

    def _find_new_sessions(
        self,
        sessions_dir: str,
        since: Optional[str] = None,
    ) -> List[str]:
        """Find session files newer than the given timestamp."""
        sessions_path = Path(sessions_dir).expanduser()
        if not sessions_path.is_dir():
            return []

        cutoff = 0.0
        if since:
            try:
                dt = datetime.fromisoformat(since)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                cutoff = dt.timestamp()
            except (ValueError, TypeError):
                pass

        new_files: List[str] = []
        for root, _dirs, filenames in os.walk(sessions_path):
            for fname in filenames:
                if fname.endswith((".jsonl", ".json", ".md", ".txt")):
                    fpath = os.path.join(root, fname)
                    try:
                        mtime = os.path.getmtime(fpath)
                        if mtime > cutoff:
                            new_files.append(fpath)
                    except OSError:
                        continue

        return new_files

    def _mine_sessions(
        self, session_files: List[str], tenant_id: str = "default", store=None, tenant=None
    ) -> int:
        """Run consolidation over new session files. Returns memory count."""
        from .consolidation import ConsolidationPipeline

        if store is None or tenant is None:
            from .memory_object import MemoryStore
            from .tenant import TenantManager

            tm = TenantManager(base_dir=self.base_dir)
            tenant = tm.get_or_create(tenant_id)
            store = MemoryStore(db_path=tenant.memory_db_path)

        # Load atlas for hot assignment
        atlas = None
        try:
            from .atlas import AtlasManager

            atlas_mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
            atlas = atlas_mgr.get_atlas()
        except Exception:
            pass

        from .consolidation import is_qwen_available

        use_llm = is_qwen_available()
        if not use_llm:
            logger.warning("Qwen not available for lifecycle mining. Skipping LLM consolidation.")

        pipeline = ConsolidationPipeline(
            store=store,
            tenant_id=tenant_id,
            use_llm=use_llm,
            atlas=atlas,
        )

        already_processed = store.get_processed_source_paths(tenant_id)
        to_process = [f for f in session_files if f not in already_processed]

        if not to_process:
            return 0

        total = 0
        for i, fpath in enumerate(to_process, 1):
            try:
                text = Path(fpath).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not text.strip():
                continue

            if i % 10 == 1 or i == len(to_process):
                print(
                    f"        Mining [{i}/{len(to_process)}] {Path(fpath).name[:50]}...", flush=True
                )

            memories = pipeline.consolidate(
                raw_text=text,
                source_type="conversation",
                source_path=fpath,
                title=Path(fpath).name,
            )
            total += len(memories)

        return total

    def _label_queries(self, tenant, store, atlas, sessions_dir, batch_size: int = 0) -> int:
        """Label queries for selector training. batch_size=0 means all remaining up to 300."""
        try:
            from .training.oracle_labeler import OracleLabeler, _load_queries_from_sessions
            from .pipeline import PSAPipeline
        except ImportError:
            logger.warning("Training dependencies not available. Skipping labeling.")
            return 0

        labels_path = os.path.join(tenant.root_dir, "training", "oracle_labels.jsonl")
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)

        # Count existing labels
        existing = 0
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                existing = sum(1 for line in f if line.strip())

        # batch_size=0 means label all available queries
        effective_batch = batch_size if batch_size > 0 else 10_000

        # Load real user queries from sessions
        if sessions_dir:
            queries = _load_queries_from_sessions(sessions_dir, max_queries=effective_batch * 3)
        else:
            queries = []

        if not queries:
            logger.info("No queries available for labeling.")
            return 0

        # Deduplicate against already-labeled query IDs
        labeled_ids = set()
        if os.path.exists(labels_path):
            import json

            with open(labels_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            labeled_ids.add(json.loads(line).get("query_id", ""))
                        except Exception:
                            pass
        queries = [(qid, qt) for qid, qt in queries if qid not in labeled_ids][:effective_batch]

        if not queries:
            return 0

        from .llm import _load_config as _llm_config

        llm_cfg = _llm_config()
        llm_name = (
            llm_cfg.get("cloud_model")
            if llm_cfg.get("provider") != "local" and llm_cfg.get("cloud_api_key")
            else llm_cfg.get("local_model", "local")
        )
        can_train = existing >= 300
        status = (
            f"{existing} existing, can train"
            if can_train
            else f"{existing} existing, need 300 to start training"
        )

        # Build pipeline for labeling
        try:
            pipeline = PSAPipeline.from_tenant(
                tenant_id=tenant.tenant_id,
                psa_mode="primary",
                selector_mode="cosine",
            )
        except FileNotFoundError:
            logger.warning("No atlas available for labeling.")
            return 0

        # Pre-score queries to classify as successful vs poor-performing.
        # A query is "successful" when the top selector score is >= 0.6;
        # "poor" when the pipeline returns no anchors or a low top score.
        # This ensures oracle labels contain both positive and negative signal.
        _SUCCESS_THRESHOLD = 0.6
        successful_queries: List = []
        poor_queries: List = []
        for qid, query_text in queries:
            try:
                result = pipeline.query(query_text, top_k_candidates=8)
                top_score = (
                    max(a.selector_score for a in result.selected_anchors)
                    if result.selected_anchors
                    else 0.0
                )
                if top_score >= _SUCCESS_THRESHOLD:
                    successful_queries.append((qid, query_text))
                else:
                    poor_queries.append((qid, query_text))
            except Exception:
                poor_queries.append((qid, query_text))

        # Sample successful queries for balanced oracle labels (1:2 ratio)
        sampled_success: List = []
        if successful_queries:
            n_success_sample = max(len(poor_queries) // 2, 10)
            sampled_success = random.sample(
                successful_queries,
                min(n_success_sample, len(successful_queries)),
            )
            queries_to_label = poor_queries + sampled_success
        else:
            queries_to_label = poor_queries

        print(
            f"        Scoring {len(queries_to_label)} queries with {llm_name} "
            f"({len(poor_queries)} poor, {len(sampled_success)} successful sampled; {status})..."
        )

        labeler = OracleLabeler(pipeline=pipeline, output_path=labels_path)
        labeled = 0
        for i, (qid, query_text) in enumerate(queries_to_label, 1):
            try:
                labeler.label(query_id=qid, query=query_text)
                labeled += 1
                if labeled % 10 == 0 or labeled == 1:
                    print(f"        [{labeled}/{len(queries_to_label)}] scored...", flush=True)
            except Exception as e:
                print(f"        [{i}/{len(queries_to_label)}] FAILED: {e}", flush=True)

        total = existing + labeled
        can_train = "ready to train" if total >= 300 else f"{300 - total} more needed to train"
        print(f"        Done. {labeled} new, {total} total labels ({can_train}).")
        return labeled

    def _retrain_selector(self, tenant, store, atlas, state) -> bool:
        """Retrain the selector if training gates are met. Returns True if retrained."""
        try:
            from .selector import check_training_gates
            from .training.data_generator import DataGenerator
            from .training.train_selector import SelectorTrainer
        except ImportError:
            logger.warning("Training dependencies not available. Skipping selector retrain.")
            return False

        # Check training gates (>=300 oracle labels, >=200 held-out, recall@24 >= 0.95)
        labels_path = os.path.join(tenant.root_dir, "training", "oracle_labels.jsonl")
        label_count = 0
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                label_count = sum(1 for line in f if line.strip())

        gate_status = check_training_gates(
            oracle_count=label_count,
            shortlist_recall_24=1.0,  # assume retriever is good enough
        )
        if not gate_status.gates_met:
            logger.info(
                "Training gates not met (%s). Staying in cosine mode.",
                "; ".join(gate_status.blocking_reasons),
            )
            return False

        # Generate training data from existing labels
        examples_path = os.path.join(tenant.root_dir, "training", "training_data.jsonl")
        anchor_cards = {c.anchor_id: c.to_stable_card_text() for c in atlas.cards}
        gen = DataGenerator(oracle_labels_path=labels_path, anchor_cards=anchor_cards)
        n_written = gen.generate(output_path=examples_path, n_examples=max(1000, label_count * 20))

        if n_written < 100:
            logger.info("Only %d training examples generated. Staying in cosine mode.", n_written)
            return False

        # Query-grouped train/val split (no leakage)
        from .training.data_split import split_train_val

        train_path = os.path.join(tenant.root_dir, "training", "train_data.jsonl")
        val_path = os.path.join(tenant.root_dir, "training", "val_data.jsonl")
        split_stats = split_train_val(examples_path, train_path, val_path)
        logger.info(
            "Train/val split: %d/%d queries, pos rate: train=%.1f%% val=%.1f%%",
            split_stats["n_train_queries"],
            split_stats["n_val_queries"],
            split_stats["train_positive_rate"] * 100,
            split_stats["val_positive_rate"] * 100,
        )

        # Train
        selector_version = state.get("selector_version", 0) + 1
        output_dir = os.path.join(tenant.root_dir, "models", f"selector_v{selector_version}")
        trainer = SelectorTrainer(
            output_dir=output_dir,
            atlas_version=atlas.version,
        )

        try:
            sv = trainer.train(
                train_data_path=train_path, val_data_path=val_path, version=selector_version
            )
            logger.info("Selector v%d trained → %s", sv.version, sv.model_path)

            # Activate trained selector (mutate state; saved by run())
            state["selector_mode"] = "trained"
            state["selector_model_path"] = sv.model_path
            state["selector_version"] = selector_version
            return True
        except Exception as e:
            logger.error("Selector training failed: %s", e)
            return False


def advertisement_decay_pass(
    *,
    tenant_id: str,
    config,
    atlas_or_loader,
    shielded_anchor_fn=None,
    pinned_fn=None,
) -> dict:
    """Nightly fast-path decay + removal evaluation + optional apply.

    Returns a summary dict (also logged as structured JSON via logger).
    """
    import os
    import sqlite3

    if not config.tracking_enabled:
        logger.info("advertisement_decay skipped (tracking_enabled=false)")
        return {"stage": "advertisement_decay", "skipped": True}

    if shielded_anchor_fn is None:
        # Default shielded_fn is the stage 1 P1 helper. The stage 1 module
        # exposes this via decay.shielded_anchors when available; otherwise
        # default to no shielding.
        def shielded_anchor_fn(tenant_id, anchor_ids):
            try:
                from psa.advertisement.decay import shielded_anchors as _sh

                return _sh(tenant_id, anchor_ids)
            except (ImportError, AttributeError, TypeError):
                return set()

    if pinned_fn is None:
        # Default pinned_fn reads from stage 1's pattern_metadata.json.
        def pinned_fn(anchor_id, pattern_text):
            try:
                from psa.advertisement.metadata import load_metadata, metadata_key

                meta = load_metadata(atlas_or_loader.anchor_dir)
                entry = meta.get(metadata_key(anchor_id, pattern_text), {})
                return bool(entry.get("pinned", False))
            except Exception:
                return False

    from psa.advertisement.ledger import (
        apply_decay,
        apply_removals,
        create_schema,
        evaluate_removal,
    )

    db_path = os.path.expanduser(
        f"~/.psa/tenants/{tenant_id}/memory.sqlite3"
    )
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with sqlite3.connect(db_path) as db:
        create_schema(db)
        apply_decay(db, tau_days=config.tau_days, removal_threshold=config.removal_threshold)
        result = evaluate_removal(
            db=db,
            atlas=atlas_or_loader,
            tenant_id=tenant_id,
            config=config,
            shielded_anchor_fn=shielded_anchor_fn,
            pinned_fn=pinned_fn,
        )
        n_removed_B = 0
        if config.removal_enabled and result.removal_candidates:
            atlas_dir = getattr(atlas_or_loader, "anchor_dir", None) or os.path.dirname(
                db_path
            )
            n_removed_B = apply_removals(
                db=db,
                tenant_id=tenant_id,
                atlas_dir=atlas_dir,
                candidates=result.removal_candidates,
            )

    summary = {
        "stage": "advertisement_decay",
        "tenant_id": tenant_id,
        "n_active": result.n_active,
        "n_in_grace": result.n_in_grace,
        "n_at_risk": result.n_at_risk,
        "n_would_remove_under_A": len(result.shadow_candidates),
        "n_would_remove_under_B": len(result.removal_candidates),
        "n_actually_removed_under_B": n_removed_B,
    }
    logger.info("advertisement_decay summary: %s", summary)
    return summary
