"""
llm.py — Unified LLM caller for PSA.

Tries a cloud API via litellm first, falls back to local Ollama via HTTP.
All PSA components that need an LLM (consolidation, oracle labeling,
card generation) should use call_llm() instead of hitting endpoints directly.

Configuration is read from ~/.psa/llm.json (created on first run):

    {
        "provider": "cloud",          # "cloud" or "local"
        "cloud_model": "azure/grok-4-20-reasoning",
        "cloud_api_key": "...",
        "cloud_api_base": "https://...",
        "cloud_api_version": "2024-12-01-preview",
        "local_endpoint": "http://localhost:11434/v1/chat/completions",
        "local_model": "qwen2.5:7b"
    }

Environment variables override the config file:
    PSA_LLM_MODEL, PSA_LLM_API_KEY, PSA_LLM_API_BASE, PSA_LLM_API_VERSION,
    QWEN_ENDPOINT, QWEN_MODEL
"""

import json
import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger("psa.llm")

# ── Config loading ──────────────────────────────────────────────────────────

_CONFIG_PATH = Path(os.path.expanduser("~/.psa/llm.json"))

_DEFAULT_CONFIG = {
    "provider": "local",
    "cloud_model": "",
    "cloud_api_key": "",
    "cloud_api_base": "",
    "cloud_api_version": "2024-12-01-preview",
    "local_endpoint": "http://localhost:11434/v1/chat/completions",
    "local_model": "qwen2.5:7b",
    "local_fallback": True,
}

_config_cache = None


def _load_config() -> dict:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config = dict(_DEFAULT_CONFIG)

    # Read from file
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH) as f:
                file_config = json.load(f)
            config.update(file_config)
        except (json.JSONDecodeError, OSError):
            pass
    else:
        # Create default config file on first run
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(_CONFIG_PATH, "w") as f:
                json.dump(_DEFAULT_CONFIG, f, indent=2)
            logger.info("Created LLM config at %s", _CONFIG_PATH)
        except OSError:
            pass

    # Environment variables override file config
    if os.environ.get("PSA_LLM_MODEL"):
        config["cloud_model"] = os.environ["PSA_LLM_MODEL"]
    if os.environ.get("PSA_LLM_API_KEY"):
        config["cloud_api_key"] = os.environ["PSA_LLM_API_KEY"]
    if os.environ.get("PSA_LLM_API_BASE"):
        config["cloud_api_base"] = os.environ["PSA_LLM_API_BASE"]
    if os.environ.get("PSA_LLM_API_VERSION"):
        config["cloud_api_version"] = os.environ["PSA_LLM_API_VERSION"]
    if os.environ.get("QWEN_ENDPOINT"):
        config["local_endpoint"] = os.environ["QWEN_ENDPOINT"]
    if os.environ.get("QWEN_MODEL"):
        config["local_model"] = os.environ["QWEN_MODEL"]

    _config_cache = config
    return config


# ── Cloud caller (litellm) ──────────────────────────────────────────────────


def _call_cloud(
    messages: List[dict],
    temperature: float = 0.1,
    max_tokens: int = 2048,
    json_mode: bool = True,
    timeout: int = 120,
) -> str:
    """Call cloud LLM via litellm."""
    config = _load_config()
    try:
        import litellm

        litellm.suppress_debug_info = True

        kwargs = {
            "model": config["cloud_model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": config["cloud_api_key"],
            "api_base": config["cloud_api_base"],
            "api_version": config["cloud_api_version"],
            "timeout": timeout,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content
    except ImportError:
        raise RuntimeError("litellm not installed. Run: uv pip install litellm")


# ── Local caller (Ollama) ──────────────────────────────────────────────────


def _call_local(
    messages: List[dict],
    temperature: float = 0.1,
    max_tokens: int = 2048,
    json_mode: bool = True,
    timeout: int = 120,
) -> str:
    """Call local Ollama via HTTP."""
    import urllib.request

    config = _load_config()
    payload = {
        "model": config["local_model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        config["local_endpoint"],
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"]


# ── Public API ──────────────────────────────────────────────────────────────


def call_llm(
    messages: List[dict],
    temperature: float = 0.1,
    max_tokens: int = 2048,
    json_mode: bool = True,
    timeout: int = 120,
) -> str:
    """
    Call an LLM. Tries cloud first (if configured), falls back to local Ollama.

    Configure in ~/.psa/llm.json:
        "provider": "cloud"  → cloud first, local fallback
        "provider": "local"  → local only (no cloud calls)

    Returns the response text. Raises RuntimeError if all endpoints fail.
    """
    config = _load_config()
    # _PSA_ORACLE_LLM_OVERRIDE lets oracle-label --mode local/api force a
    # specific backend without editing llm.json.
    provider = os.environ.get("_PSA_ORACLE_LLM_OVERRIDE") or config.get("provider", "cloud")
    errors = []

    # Try cloud first (unless provider is "local")
    if provider != "local" and config.get("cloud_api_key"):
        try:
            return _call_cloud(messages, temperature, max_tokens, json_mode, timeout)
        except Exception as e:
            logger.debug("Cloud LLM failed: %s. Trying local.", e)
            errors.append(f"cloud ({config['cloud_model']}): {e}")

    # Fall back to local (unless disabled)
    if config.get("local_fallback", True):
        try:
            return _call_local(messages, temperature, max_tokens, json_mode, timeout)
        except Exception as e:
            errors.append(f"local ({config['local_model']}): {e}")

    raise RuntimeError(f"All LLM endpoints failed: {'; '.join(errors)}")


def is_any_llm_available() -> bool:
    """Check if any LLM endpoint is available."""
    config = _load_config()
    if config.get("provider") != "local" and config.get("cloud_api_key"):
        return True
    return _is_local_available()


def _is_local_available() -> bool:
    config = _load_config()
    try:
        import urllib.request

        base_url = config["local_endpoint"].rsplit("/v1", 1)[0]
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=3)
        return True
    except Exception:
        return False
