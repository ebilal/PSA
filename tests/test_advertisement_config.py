"""Tests for psa.advertisement.config — typed view + tracking→trace validation."""

from __future__ import annotations

import json

import pytest


def _write_config(tmp_path, body):
    d = tmp_path / ".psa"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(body))
    return d


def test_defaults(tmp_path):
    from psa.config import MempalaceConfig
    from psa.advertisement.config import AdvertisementDecayConfig

    _write_config(tmp_path, {})
    cfg = MempalaceConfig(config_dir=tmp_path / ".psa")
    ad = AdvertisementDecayConfig.from_mempalace(cfg)

    assert ad.tracking_enabled is False
    assert ad.removal_enabled is False
    assert ad.retrieval_credit == 1.0
    assert ad.selector_pick_credit == 2.0
    assert ad.selector_decline_penalty == 0.25
    assert ad.tau_days == 45
    assert ad.grace_days == 21
    assert ad.removal_threshold == 0.0
    assert ad.sustained_cycles == 14
    assert ad.min_patterns_floor == 3
    assert ad.epsilon == 0.05
    assert ad.bm25_topk_floor == 48
    assert ad.shadow.selector_decline_penalty == 0.5
    assert ad.shadow.sustained_cycles == 7


def test_tracking_requires_trace(tmp_path):
    from psa.config import MempalaceConfig
    from psa.advertisement.config import (
        AdvertisementDecayConfig,
        AdvertisementDecayConfigError,
    )

    _write_config(
        tmp_path,
        {
            "trace_queries": False,
            "advertisement_decay": {"tracking_enabled": True},
        },
    )
    cfg = MempalaceConfig(config_dir=tmp_path / ".psa")
    with pytest.raises(AdvertisementDecayConfigError) as exc:
        AdvertisementDecayConfig.from_mempalace(cfg)
    assert "trace_queries" in str(exc.value)


def test_env_overrides(tmp_path, monkeypatch):
    from psa.config import MempalaceConfig
    from psa.advertisement.config import AdvertisementDecayConfig

    _write_config(tmp_path, {})
    monkeypatch.setenv("PSA_AD_DECAY_TRACKING_ENABLED", "1")
    monkeypatch.setenv("PSA_AD_DECAY_TAU_DAYS", "30")
    monkeypatch.setenv("PSA_AD_DECAY_DECLINE_PENALTY", "0.5")
    # trace_queries defaults True, so tracking can be enabled
    cfg = MempalaceConfig(config_dir=tmp_path / ".psa")
    ad = AdvertisementDecayConfig.from_mempalace(cfg)

    assert ad.tracking_enabled is True
    assert ad.tau_days == 30
    assert ad.selector_decline_penalty == 0.5
