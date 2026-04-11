"""
test_llm.py — Unit tests for psa.llm routing and fallback logic.

All tests are fully mocked — no network calls, no real LLM required.
"""

from unittest.mock import MagicMock

import pytest

import psa.llm as llm_mod
from psa.llm import call_llm


# ── Helpers ───────────────────────────────────────────────────────────────────

MESSAGES = [{"role": "user", "content": "hello"}]


def _cloud_config(local_fallback: bool = True) -> dict:
    return {
        "provider": "cloud",
        "cloud_model": "azure/gpt-5.4-mini",
        "cloud_api_key": "test-key",
        "cloud_api_base": "https://example.cognitiveservices.azure.com/",
        "cloud_api_version": "2024-12-01-preview",
        "local_endpoint": "http://localhost:11434/v1/chat/completions",
        "local_model": "qwen2.5:7b",
        "local_fallback": local_fallback,
    }


@pytest.fixture(autouse=True)
def clear_config_cache():
    """Reset the module-level config cache before each test."""
    llm_mod._config_cache = None
    yield
    llm_mod._config_cache = None


# ── Cloud succeeds — no fallback needed ───────────────────────────────────────


def test_cloud_success_returns_response(monkeypatch):
    monkeypatch.setattr(llm_mod, "_load_config", lambda: _cloud_config())
    monkeypatch.setattr(llm_mod, "_call_cloud", lambda *a, **kw: '{"status": "ok"}')

    result = call_llm(MESSAGES)
    assert result == '{"status": "ok"}'


def test_cloud_success_does_not_call_local(monkeypatch):
    local_called = []
    monkeypatch.setattr(llm_mod, "_load_config", lambda: _cloud_config())
    monkeypatch.setattr(llm_mod, "_call_cloud", lambda *a, **kw: '{"ok": 1}')
    monkeypatch.setattr(
        llm_mod, "_call_local", lambda *a, **kw: local_called.append(1) or "should not reach"
    )

    call_llm(MESSAGES)
    assert local_called == [], "Local should not be called when cloud succeeds"


# ── Cloud fails — fallback to local ──────────────────────────────────────────


def test_cloud_failure_falls_back_to_local(monkeypatch):
    """Core fallback test: cloud raises, local returns valid response."""
    monkeypatch.setattr(llm_mod, "_load_config", lambda: _cloud_config(local_fallback=True))
    monkeypatch.setattr(
        llm_mod, "_call_cloud", MagicMock(side_effect=RuntimeError("API unavailable"))
    )
    monkeypatch.setattr(llm_mod, "_call_local", lambda *a, **kw: '{"status": "ok"}')

    result = call_llm(MESSAGES)
    assert result == '{"status": "ok"}'


def test_cloud_timeout_falls_back_to_local(monkeypatch):
    """Timeout on cloud should also trigger fallback."""
    monkeypatch.setattr(llm_mod, "_load_config", lambda: _cloud_config(local_fallback=True))
    monkeypatch.setattr(llm_mod, "_call_cloud", MagicMock(side_effect=TimeoutError("timed out")))
    monkeypatch.setattr(llm_mod, "_call_local", lambda *a, **kw: '{"result": "from_local"}')

    result = call_llm(MESSAGES)
    assert result == '{"result": "from_local"}'


def test_fallback_disabled_raises_without_calling_local(monkeypatch):
    """When local_fallback=False, cloud failure raises immediately — no local call."""
    local_called = []
    monkeypatch.setattr(llm_mod, "_load_config", lambda: _cloud_config(local_fallback=False))
    monkeypatch.setattr(llm_mod, "_call_cloud", MagicMock(side_effect=RuntimeError("API down")))
    monkeypatch.setattr(llm_mod, "_call_local", lambda *a, **kw: local_called.append(1) or "nope")

    with pytest.raises(RuntimeError, match="All LLM endpoints failed"):
        call_llm(MESSAGES)
    assert local_called == [], "Local must not be called when local_fallback=False"


# ── Both fail ─────────────────────────────────────────────────────────────────


def test_both_fail_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(llm_mod, "_load_config", lambda: _cloud_config(local_fallback=True))
    monkeypatch.setattr(llm_mod, "_call_cloud", MagicMock(side_effect=RuntimeError("cloud down")))
    monkeypatch.setattr(
        llm_mod, "_call_local", MagicMock(side_effect=ConnectionRefusedError("ollama not running"))
    )

    with pytest.raises(RuntimeError, match="All LLM endpoints failed"):
        call_llm(MESSAGES)


def test_both_fail_error_message_includes_both(monkeypatch):
    monkeypatch.setattr(llm_mod, "_load_config", lambda: _cloud_config(local_fallback=True))
    monkeypatch.setattr(llm_mod, "_call_cloud", MagicMock(side_effect=RuntimeError("cloud down")))
    monkeypatch.setattr(
        llm_mod, "_call_local", MagicMock(side_effect=ConnectionRefusedError("ollama not running"))
    )

    with pytest.raises(RuntimeError) as exc:
        call_llm(MESSAGES)
    msg = str(exc.value)
    assert "cloud" in msg
    assert "local" in msg


# ── provider=local skips cloud entirely ───────────────────────────────────────


def test_provider_local_skips_cloud(monkeypatch):
    cloud_called = []
    cfg = _cloud_config()
    cfg["provider"] = "local"
    monkeypatch.setattr(llm_mod, "_load_config", lambda: cfg)
    monkeypatch.setattr(llm_mod, "_call_cloud", lambda *a, **kw: cloud_called.append(1) or "nope")
    monkeypatch.setattr(llm_mod, "_call_local", lambda *a, **kw: '{"local": true}')

    result = call_llm(MESSAGES)
    assert result == '{"local": true}'
    assert cloud_called == [], "Cloud must not be called when provider=local"
