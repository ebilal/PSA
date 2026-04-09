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
            # Pass all files; _mine_sessions skips files already in PSA raw_sources
            logger.info("Mining: %d total session files (%d new since last run)", len(all_sessions), len(new_since_last))
            mined = self._mine_sessions(all_sessions, tenant_id=tenant_id, store=store, tenant=tenant)
            summary["memories_mined"] = mined
            has_new_data = has_new_data or mined > 0
            state["last_mine_timestamp"] = now_iso

        # === MAINTENANCE (runs every time) ===

        # 2. Delete old archived memories (hard-delete after 90 days)
        deleted = store.delete_old_archived(tenant_id, older_than_days=90)
        summary["memories_hard_deleted"] = deleted

        # 3. Load atlas for pruning
        atlas_mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=tenant_id)
        atlas = atlas_mgr.get_atlas()

        if atlas is not None:
            # 4. Per-anchor pruning
            total_pruned = 0
            anchor_budget = int(
                cfg._file_config.get("anchor_memory_budget", 100)
            )
            for card in atlas.cards:
                if getattr(card, "status", "active") == "active":
                    pruned = prune_anchor(
                        store, tenant_id, card.anchor_id, budget=anchor_budget
                    )
                    total_pruned += pruned
            summary["memories_pruned"] = total_pruned

            # 5. Global cap
            max_memories = int(cfg._file_config.get("max_memories", 50_000))
            cap_result = enforce_global_cap(store, tenant_id, max_memories=max_memories)
            summary["memories_pruned"] += cap_result.get("archived", 0)
            summary["memories_hard_deleted"] += cap_result.get("hard_deleted", 0)

            # === SLOW PATH (only when health triggers AND we have new data) ===
            monitor = AtlasHealthMonitor()
            health = monitor.check_health(atlas, store, tenant_id=tenant_id)

            if (has_new_data and health.should_rebuild) or force_rebuild:
                logger.info("Slow path: rebuilding atlas (reasons: %s)",
                            "; ".join(health.rebuild_reasons) if health.rebuild_reasons else "forced")

                # Cosine fallback during rebuild
                self._write_selector_mode(tenant.root_dir, "cosine")

                try:
                    atlas = atlas_mgr.rebuild(store)
                    summary["atlas_rebuilt"] = True
                    logger.info("Atlas v%d rebuilt successfully", atlas.version)
                except Exception as e:
                    logger.error("Atlas rebuild failed: %s", e)

                # Retrain selector if training gates are met
                if summary["atlas_rebuilt"]:
                    retrained = self._retrain_selector(tenant, store, atlas, state)
                    summary["selector_retrained"] = retrained

        # 6. Save state
        state["last_memory_count"] = store.count(tenant_id)
        state["last_run"] = now_iso
        self._save_state(tenant.root_dir, state)

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

    def _mine_sessions(self, session_files: List[str], tenant_id: str = "default",
                        store=None, tenant=None) -> int:
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

        total = 0
        for fpath in session_files:
            if fpath in already_processed:
                continue
            try:
                text = Path(fpath).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not text.strip():
                continue

            memories = pipeline.consolidate(
                raw_text=text,
                source_type="conversation",
                source_path=fpath,
                title=Path(fpath).name,
            )
            total += len(memories)

        return total

    def _retrain_selector(self, tenant, store, atlas, state) -> bool:
        """Retrain the selector if training gates are met. Returns True if retrained."""
        try:
            from .selector import check_training_gates
            from .training.oracle_labeler import OracleLabeler, _load_queries_from_sessions
            from .training.data_generator import DataGenerator
            from .training.train_selector import SelectorTrainer
        except ImportError:
            logger.warning("Training dependencies not available. Skipping selector retrain.")
            return False

        # Check if we have enough oracle labels
        labels_path = os.path.join(tenant.root_dir, "training", "oracle_labels.jsonl")
        label_count = 0
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                label_count = sum(1 for line in f if line.strip())

        if label_count < 50:
            logger.info("Only %d oracle labels (need 50+ for training). Staying in cosine mode.", label_count)
            return False

        # Generate training data from existing labels
        training_path = os.path.join(tenant.root_dir, "training", "training_data.jsonl")
        anchor_cards = {c.anchor_id: c.to_stable_card_text() for c in atlas.cards}
        gen = DataGenerator(oracle_labels_path=labels_path, anchor_cards=anchor_cards)
        n_written = gen.generate(output_path=training_path, n_examples=max(1000, label_count * 20))

        if n_written < 100:
            logger.info("Only %d training examples generated. Staying in cosine mode.", n_written)
            return False

        # Train
        selector_version = state.get("selector_version", 0) + 1
        output_dir = os.path.join(tenant.root_dir, "models", f"selector_v{selector_version}")
        trainer = SelectorTrainer(
            output_dir=output_dir,
            atlas_version=atlas.version,
        )

        try:
            sv = trainer.train(train_data_path=training_path, version=selector_version)
            logger.info("Selector v%d trained → %s", sv.version, sv.model_path)

            # Activate trained selector
            self._write_selector_mode(tenant.root_dir, "trained", sv.model_path)
            state["selector_version"] = selector_version
            return True
        except Exception as e:
            logger.error("Selector training failed: %s", e)
            return False
