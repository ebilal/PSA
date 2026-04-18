"""
config.py — typed view over the advertisement_decay block in MempalaceConfig.

Validation: tracking_enabled=true requires trace_queries=true. Without it,
ledger writes (gated on trace-first success) accumulate zero events, and
rebuild-ledger cannot be canonical. Fail fast at config load.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class AdvertisementDecayConfigError(ValueError):
    """Raised when advertisement_decay settings are inconsistent."""


def _env_bool(name: str, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in ("0", "", "false", "False")


def _env_float(name: str, default):
    v = os.environ.get(name)
    return float(v) if v is not None else default


def _env_int(name: str, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default


@dataclass(frozen=True)
class ShadowConfig:
    selector_decline_penalty: float = 0.5
    sustained_cycles: int = 7


@dataclass(frozen=True)
class AdvertisementDecayConfig:
    tracking_enabled: bool = False
    removal_enabled: bool = False

    retrieval_credit: float = 1.0
    selector_pick_credit: float = 2.0
    selector_decline_penalty: float = 0.25
    tau_days: int = 45
    grace_days: int = 21
    removal_threshold: float = 0.0
    sustained_cycles: int = 14
    min_patterns_floor: int = 3
    epsilon: float = 0.05
    bm25_topk_floor: int = 48

    shadow: ShadowConfig = field(default_factory=ShadowConfig)

    @classmethod
    def from_mempalace(cls, mempalace_cfg) -> "AdvertisementDecayConfig":
        block = mempalace_cfg._file_config.get("advertisement_decay", {}) or {}
        shadow_block = block.get("shadow", {}) or {}
        shadow = ShadowConfig(
            selector_decline_penalty=float(
                shadow_block.get("selector_decline_penalty", 0.5)
            ),
            sustained_cycles=int(shadow_block.get("sustained_cycles", 7)),
        )
        cfg = cls(
            tracking_enabled=_env_bool(
                "PSA_AD_DECAY_TRACKING_ENABLED",
                bool(block.get("tracking_enabled", False)),
            ),
            removal_enabled=_env_bool(
                "PSA_AD_DECAY_REMOVAL_ENABLED",
                bool(block.get("removal_enabled", False)),
            ),
            retrieval_credit=_env_float(
                "PSA_AD_DECAY_RETRIEVAL_CREDIT",
                float(block.get("retrieval_credit", 1.0)),
            ),
            selector_pick_credit=_env_float(
                "PSA_AD_DECAY_PICK_CREDIT",
                float(block.get("selector_pick_credit", 2.0)),
            ),
            selector_decline_penalty=_env_float(
                "PSA_AD_DECAY_DECLINE_PENALTY",
                float(block.get("selector_decline_penalty", 0.25)),
            ),
            tau_days=_env_int("PSA_AD_DECAY_TAU_DAYS", int(block.get("tau_days", 45))),
            grace_days=_env_int(
                "PSA_AD_DECAY_GRACE_DAYS", int(block.get("grace_days", 21))
            ),
            removal_threshold=_env_float(
                "PSA_AD_DECAY_REMOVAL_THRESHOLD",
                float(block.get("removal_threshold", 0.0)),
            ),
            sustained_cycles=_env_int(
                "PSA_AD_DECAY_SUSTAINED_CYCLES",
                int(block.get("sustained_cycles", 14)),
            ),
            min_patterns_floor=_env_int(
                "PSA_AD_DECAY_MIN_PATTERNS_FLOOR",
                int(block.get("min_patterns_floor", 3)),
            ),
            epsilon=_env_float(
                "PSA_AD_DECAY_EPSILON", float(block.get("epsilon", 0.05))
            ),
            bm25_topk_floor=_env_int(
                "PSA_AD_DECAY_BM25_TOPK_FLOOR",
                int(block.get("bm25_topk_floor", 48)),
            ),
            shadow=shadow,
        )
        cfg._validate(mempalace_cfg)
        return cfg

    def _validate(self, mempalace_cfg) -> None:
        if self.tracking_enabled and not mempalace_cfg.trace_queries:
            raise AdvertisementDecayConfigError(
                "advertisement_decay.tracking_enabled=true requires "
                "trace_queries=true. Ledger writes fire only when the "
                "preceding trace write succeeds, so tracing must be "
                "enabled for ledger accumulation to work. "
                "Set trace_queries=true or tracking_enabled=false."
            )
