import os
import json
import tempfile
from psa.config import MempalaceConfig


def test_default_config():
    cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
    assert "palace" in cfg.palace_path
    assert cfg.collection_name == "psa_drawers"


def test_config_from_file():
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump({"palace_path": "/custom/palace"}, f)
    cfg = MempalaceConfig(config_dir=tmpdir)
    assert cfg.palace_path == "/custom/palace"


def test_env_override():
    os.environ["PSA_PALACE_PATH"] = "/env/palace"
    cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
    assert cfg.palace_path == "/env/palace"
    del os.environ["PSA_PALACE_PATH"]


def test_init():
    tmpdir = tempfile.mkdtemp()
    cfg = MempalaceConfig(config_dir=tmpdir)
    cfg.init()
    path = os.path.join(tmpdir, "config.json")
    assert os.path.exists(path)
    with open(path) as f:
        body = json.load(f)

    assert body["tenant_id"] == "default"
    assert body["psa_mode"] == "primary"
    assert body["token_budget"] == 6000
    assert body["max_memories"] == 50000
    assert body["anchor_memory_budget"] == 100
    assert body["trace_queries"] is True
    assert body["nightly_hour"] == 0
    assert body["advertisement_decay"] == {
        "tracking_enabled": True,
        "removal_enabled": True,
        "tau_days": 45,
        "grace_days": 30,
        "sustained_cycles": 21,
        "min_patterns_floor": 5,
    }
