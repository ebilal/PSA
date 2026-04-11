"""test_fingerprints.py — Tests for FingerprintStore."""

from psa.fingerprints import FingerprintStore


def test_fingerprint_store_empty_on_new_dir(tmp_path):
    store = FingerprintStore(str(tmp_path))
    assert store.get(1) == []


def test_fingerprint_store_append_and_get(tmp_path):
    store = FingerprintStore(str(tmp_path))
    store.append(1, "query A")
    store.append(1, "query B")
    assert store.get(1) == ["query A", "query B"]


def test_fingerprint_store_fifo_eviction(tmp_path):
    store = FingerprintStore(str(tmp_path))
    for i in range(55):
        store.append(1, f"query {i}")
    result = store.get(1)
    assert len(result) == 50
    assert result[0] == "query 5"  # oldest 5 evicted
    assert result[-1] == "query 54"


def test_fingerprint_store_save_and_reload(tmp_path):
    store = FingerprintStore(str(tmp_path))
    store.append(1, "query A")
    store.append(2, "query B")
    store.save()

    store2 = FingerprintStore(str(tmp_path))
    assert store2.get(1) == ["query A"]
    assert store2.get(2) == ["query B"]


def test_fingerprint_store_inherit_from(tmp_path):
    store = FingerprintStore(str(tmp_path))
    store.append(10, "old query")
    store.inherit_from(old_anchor_id=10, new_anchor_id=20)
    assert store.get(20) == ["old query"]
    assert store.get(10) == ["old query"]  # original unchanged


def test_fingerprint_store_inherit_from_missing_anchor(tmp_path):
    """inherit_from a non-existent old anchor is a no-op."""
    store = FingerprintStore(str(tmp_path))
    store.inherit_from(old_anchor_id=999, new_anchor_id=1)
    assert store.get(1) == []
