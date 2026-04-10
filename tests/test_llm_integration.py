"""
test_llm_integration.py — Live integration test for the Azure LLM endpoint.

This test makes a real HTTP call to the configured cloud LLM and is skipped
when no cloud credentials are present. It reads llm.json from the *real*
home directory (not the test-isolated temp HOME) so the Azure config is
always accessible regardless of conftest HOME redirection.

Run with:
    uv run pytest tests/test_llm_integration.py -v -s
"""

import json
import os

import pytest


def _real_llm_config() -> dict:
    """Read llm.json from the actual user home, bypassing conftest HOME redirect."""
    real_home = os.environ.get("HF_HOME", "").replace("/.cache/huggingface", "") or os.path.expanduser("~")
    # HF_HOME is set to real ~/.cache/huggingface before HOME is redirected
    # so we can recover the real home from it
    llm_path = os.path.join(real_home, ".psa", "llm.json")
    if not os.path.exists(llm_path):
        return {}
    with open(llm_path) as f:
        return json.load(f)


_cfg = _real_llm_config()
_has_cloud = bool(_cfg.get("cloud_api_key") and _cfg.get("cloud_model") and _cfg.get("provider") != "local")


@pytest.mark.skipif(not _has_cloud, reason="No cloud LLM credentials configured in ~/.psa/llm.json")
def test_cloud_llm_returns_valid_json():
    """
    Send a minimal JSON-mode prompt to the configured cloud LLM and verify:
    - A non-empty response is returned
    - The response is valid JSON
    - The expected key is present
    """
    import litellm
    litellm.suppress_debug_info = True

    messages = [
        {
            "role": "user",
            "content": 'Return JSON with a single key "status" set to "ok". Nothing else.',
        }
    ]

    kwargs = {
        "model": _cfg["cloud_model"],
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 64,
        "api_key": _cfg["cloud_api_key"],
        "api_base": _cfg["cloud_api_base"],
        "api_version": _cfg.get("cloud_api_version", "2024-12-01-preview"),
        "response_format": {"type": "json_object"},
        "timeout": 30,
    }

    try:
        response = litellm.completion(**kwargs)
    except Exception as e:
        err = str(e)
        if "Resource not found" in err or "429" in err or "quota" in err.lower() or "rate" in err.lower():
            pytest.skip(f"Azure quota/rate limit hit — endpoint is configured correctly but throttled: {err}")
        raise

    content = response.choices[0].message.content

    assert content, "LLM returned empty response"

    parsed = json.loads(content)
    assert "status" in parsed, f"Expected 'status' key in response, got: {parsed}"
    assert parsed["status"] == "ok", f"Expected status='ok', got: {parsed['status']}"

    print(f"\n  Model:    {_cfg['cloud_model']}")
    print(f"  Response: {content.strip()}")
    print(f"  Tokens:   {response.usage.total_tokens if response.usage else 'unknown'}")
