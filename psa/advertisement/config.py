"""
config.py — typed view over the advertisement_decay block in MempalaceConfig.

Validation: tracking_enabled=true requires trace_queries=true. Without it,
ledger writes (gated on trace-first success) accumulate zero events, and
rebuild-ledger cannot be canonical. Fail fast at config load.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class AdvertisementDecayConfigError(ValueError):
    """Raised when advertisement_decay settings are inconsistent."""


@dataclass(frozen=True)
class ShadowConfig:
    selector_decline_penalty: float = 0.5
    sustained_cycles: int = 7


@dataclass(frozen=True)
class AdvertisementDecayConfig:
    tracking_enabled: bool = True
    removal_enabled: bool = True

    retrieval_credit: float = 1.0
    selector_pick_credit: float = 2.0
    selector_decline_penalty: float = 0.25
    tau_days: int = 45
    grace_days: int = 30
    removal_threshold: float = 0.0
    sustained_cycles: int = 21
    min_patterns_floor: int = 5
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
            tracking_enabled=bool(block.get("tracking_enabled", True)),
            removal_enabled=bool(block.get("removal_enabled", True)),
            retrieval_credit=float(block.get("retrieval_credit", 1.0)),
            selector_pick_credit=float(block.get("selector_pick_credit", 2.0)),
            selector_decline_penalty=float(block.get("selector_decline_penalty", 0.25)),
            tau_days=int(block.get("tau_days", 45)),
            grace_days=int(block.get("grace_days", 30)),
            removal_threshold=float(block.get("removal_threshold", 0.0)),
            sustained_cycles=int(block.get("sustained_cycles", 21)),
            min_patterns_floor=int(block.get("min_patterns_floor", 5)),
            epsilon=float(block.get("epsilon", 0.05)),
            bm25_topk_floor=int(block.get("bm25_topk_floor", 48)),
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
