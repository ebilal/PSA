"""
test_anchor.py — Tests for psa.anchor.AnchorCard and AnchorIndex.

Covers:
- AnchorCard creation, card_text rendering, dict round-trip
- AnchorIndex build / search (numpy fallback path)
- Top-k search returns correct count and ordering (higher score first)
- Save / load round-trip
- Empty index raises on build
- FAISS path is tested implicitly if faiss-cpu is installed; numpy is
  always tested via monkeypatching the _use_faiss flag
"""

import os

import numpy as np
import pytest

from psa.anchor import AnchorCard, AnchorIndex


# ── Helpers ───────────────────────────────────────────────────────────────────


def _unit_vec(dim: int = 768, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _make_card(anchor_id: int, centroid: list, is_novelty: bool = False) -> AnchorCard:
    return AnchorCard(
        anchor_id=anchor_id,
        name=f"anchor_{anchor_id}",
        meaning=f"Test anchor {anchor_id}.",
        memory_types=["semantic"],
        include_terms=["term_a"],
        exclude_terms=["term_b"],
        prototype_examples=[f"Example {anchor_id}"],
        near_but_different=[],
        centroid=centroid,
        memory_count=10,
        is_novelty=is_novelty,
    )


# ── AnchorCard ────────────────────────────────────────────────────────────────


def test_anchor_card_card_text():
    card = _make_card(0, _unit_vec())
    text = card.to_card_text()
    assert "anchor_0" in text
    assert "Test anchor 0" in text
    assert "term_a" in text
    assert "term_b" in text
    assert "Example 0" in text


def test_anchor_card_dict_roundtrip():
    centroid = _unit_vec()
    card = _make_card(42, centroid)
    d = card.to_dict()
    card2 = AnchorCard.from_dict(d)
    assert card2.anchor_id == 42
    assert card2.name == "anchor_42"
    assert len(card2.centroid) == 768
    for a, b in zip(card.centroid, card2.centroid):
        assert abs(a - b) < 1e-7


def test_novelty_card():
    card = _make_card(255, _unit_vec(seed=99), is_novelty=True)
    assert card.is_novelty is True
    text = card.to_card_text()
    assert "anchor_255" in text


# ── AnchorIndex (numpy path) ──────────────────────────────────────────────────


@pytest.fixture
def numpy_index():
    """AnchorIndex forced onto the numpy path (no FAISS)."""
    idx = AnchorIndex(dim=768)
    idx._use_faiss = False
    return idx


def test_build_and_size(numpy_index):
    cards = [_make_card(i, _unit_vec(seed=i)) for i in range(10)]
    numpy_index.build(cards)
    assert numpy_index.size == 10


def test_search_returns_top_k(numpy_index):
    cards = [_make_card(i, _unit_vec(seed=i)) for i in range(20)]
    numpy_index.build(cards)
    query = _unit_vec(seed=999)
    results = numpy_index.search(query, top_k=5)
    assert len(results) == 5


def test_search_ordered_by_score(numpy_index):
    cards = [_make_card(i, _unit_vec(seed=i)) for i in range(10)]
    numpy_index.build(cards)
    results = numpy_index.search(_unit_vec(seed=77), top_k=10)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_search_self_similarity_is_highest(numpy_index):
    """Query with the same vector as anchor 3 → anchor 3 should be rank 1."""
    cards = [_make_card(i, _unit_vec(seed=i)) for i in range(10)]
    numpy_index.build(cards)
    query = _unit_vec(seed=3)  # same as anchor 3
    results = numpy_index.search(query, top_k=3)
    best_id, best_score = results[0]
    assert best_id == 3
    assert abs(best_score - 1.0) < 1e-5


def test_search_top_k_capped_at_index_size(numpy_index):
    cards = [_make_card(i, _unit_vec(seed=i)) for i in range(5)]
    numpy_index.build(cards)
    results = numpy_index.search(_unit_vec(), top_k=100)
    assert len(results) == 5  # capped at 5


def test_search_before_build_raises(numpy_index):
    with pytest.raises(RuntimeError, match="not built"):
        numpy_index.search(_unit_vec())


def test_build_empty_raises():
    idx = AnchorIndex(dim=768)
    with pytest.raises(ValueError, match="empty"):
        idx.build([])


def test_get_card(numpy_index):
    cards = [_make_card(i, _unit_vec(seed=i)) for i in range(5)]
    numpy_index.build(cards)
    card = numpy_index.get_card(3)
    assert card is not None
    assert card.anchor_id == 3


def test_get_card_missing_returns_none(numpy_index):
    cards = [_make_card(i, _unit_vec(seed=i)) for i in range(5)]
    numpy_index.build(cards)
    assert numpy_index.get_card(999) is None


# ── AnchorIndex normalization ─────────────────────────────────────────────────


def test_unnormalized_centroid_is_normalized_on_build(numpy_index):
    """build() must L2-normalize centroids even if the card has a non-unit vector."""
    raw = [3.0] + [0.0] * 767  # not unit norm
    card = _make_card(0, raw)
    numpy_index.build([card])
    # Search with the corresponding unit vector → should get sim ≈ 1.0
    query = [1.0] + [0.0] * 767
    results = numpy_index.search(query)
    assert abs(results[0][1] - 1.0) < 1e-5


# ── Save / load ───────────────────────────────────────────────────────────────


def test_save_and_load(tmp_dir):
    idx = AnchorIndex(dim=768)
    idx._use_faiss = False
    cards = [_make_card(i, _unit_vec(seed=i)) for i in range(8)]
    idx.build(cards)

    save_path = os.path.join(tmp_dir, "anchor_idx")
    idx.save(save_path)

    loaded = AnchorIndex.load(save_path, dim=768)
    loaded._use_faiss = False  # force numpy on load too

    assert loaded.size == 8

    # Verify search produces same top result
    q = _unit_vec(seed=4)
    orig_result = idx.search(q, top_k=1)
    loaded_result = loaded.search(q, top_k=1)
    assert orig_result[0][0] == loaded_result[0][0]
    assert abs(orig_result[0][1] - loaded_result[0][1]) < 1e-5


def test_load_missing_path_raises(tmp_dir):
    with pytest.raises(FileNotFoundError):
        AnchorIndex.load(os.path.join(tmp_dir, "nonexistent"), dim=768)


def test_anchor_index_load_prefers_refined_cards(tmp_path):
    """When both files exist, AnchorIndex.load() loads anchor_cards_refined.json."""
    import json
    from psa.anchor import AnchorIndex

    raw = [
        {
            "anchor_id": 1,
            "name": "raw-name",
            "meaning": "raw meaning",
            "memory_types": ["semantic"],
            "include_terms": [],
            "exclude_terms": [],
            "prototype_examples": [],
            "near_but_different": [],
            "centroid": [0.0] * 768,
            "memory_count": 1,
            "is_novelty": False,
            "status": "active",
            "metadata": {},
            "generated_query_patterns": ["raw pattern"],
            "query_fingerprint": [],
        }
    ]
    refined = [
        {
            "anchor_id": 1,
            "name": "raw-name",
            "meaning": "raw meaning",
            "memory_types": ["semantic"],
            "include_terms": [],
            "exclude_terms": [],
            "prototype_examples": [],
            "near_but_different": [],
            "centroid": [0.0] * 768,
            "memory_count": 1,
            "is_novelty": False,
            "status": "active",
            "metadata": {},
            "generated_query_patterns": ["raw pattern", "refined pattern"],
            "query_fingerprint": [],
        }
    ]
    (tmp_path / "anchor_cards.json").write_text(json.dumps(raw))
    (tmp_path / "anchor_cards_refined.json").write_text(json.dumps(refined))
    np.save(tmp_path / "centroids.npy", np.zeros((1, 768), dtype=np.float32))

    idx = AnchorIndex.load(str(tmp_path))

    assert len(idx._cards) == 1
    assert "refined pattern" in idx._cards[0].generated_query_patterns


def test_anchor_index_load_falls_back_to_raw_when_no_refined(tmp_path):
    """When only anchor_cards.json exists, it is loaded as before."""
    import json
    from psa.anchor import AnchorIndex

    raw = [
        {
            "anchor_id": 1,
            "name": "raw-name",
            "meaning": "raw meaning",
            "memory_types": ["semantic"],
            "include_terms": [],
            "exclude_terms": [],
            "prototype_examples": [],
            "near_but_different": [],
            "centroid": [0.0] * 768,
            "memory_count": 1,
            "is_novelty": False,
            "status": "active",
            "metadata": {},
            "generated_query_patterns": ["raw pattern"],
            "query_fingerprint": [],
        }
    ]
    (tmp_path / "anchor_cards.json").write_text(json.dumps(raw))
    np.save(tmp_path / "centroids.npy", np.zeros((1, 768), dtype=np.float32))

    idx = AnchorIndex.load(str(tmp_path))

    assert len(idx._cards) == 1
    assert idx._cards[0].generated_query_patterns == ["raw pattern"]


def test_anchor_index_load_raises_when_neither_file_exists(tmp_path):
    """If neither raw nor refined cards file exists, raise FileNotFoundError."""
    from psa.anchor import AnchorIndex

    np.save(tmp_path / "centroids.npy", np.zeros((1, 768), dtype=np.float32))

    with pytest.raises(FileNotFoundError):
        AnchorIndex.load(str(tmp_path))


def test_anchor_card_has_generated_query_patterns_field():
    card = AnchorCard(
        anchor_id=1,
        name="test",
        meaning="test meaning",
        memory_types=["semantic"],
        include_terms=[],
        exclude_terms=[],
        prototype_examples=[],
        near_but_different=[],
        centroid=[0.1] * 768,
    )
    assert hasattr(card, "generated_query_patterns")
    assert card.generated_query_patterns == []


def test_anchor_card_has_query_fingerprint_field():
    card = AnchorCard(
        anchor_id=1,
        name="test",
        meaning="test meaning",
        memory_types=["semantic"],
        include_terms=[],
        exclude_terms=[],
        prototype_examples=[],
        near_but_different=[],
        centroid=[0.1] * 768,
    )
    assert hasattr(card, "query_fingerprint")
    assert card.query_fingerprint == []


def test_to_stable_card_text_includes_generated_query_patterns():
    card = AnchorCard(
        anchor_id=1,
        name="schema-decisions",
        meaning="Covers schema choices.",
        memory_types=["semantic"],
        include_terms=["migration"],
        exclude_terms=[],
        prototype_examples=[],
        near_but_different=[],
        centroid=[0.0] * 768,
        generated_query_patterns=["What did we decide about migrations?", "Why postgres?"],
    )
    text = card.to_stable_card_text()
    assert "What did we decide about migrations?" in text
    assert "Why postgres?" in text


def test_to_stable_card_text_excludes_query_fingerprint():
    card = AnchorCard(
        anchor_id=1,
        name="schema-decisions",
        meaning="Covers schema choices.",
        memory_types=["semantic"],
        include_terms=[],
        exclude_terms=[],
        prototype_examples=[],
        near_but_different=[],
        centroid=[0.0] * 768,
        generated_query_patterns=[],
        query_fingerprint=["a real user query"],
    )
    text = card.to_stable_card_text()
    assert "a real user query" not in text


def test_to_card_text_includes_query_fingerprint():
    card = AnchorCard(
        anchor_id=1,
        name="schema-decisions",
        meaning="Covers schema choices.",
        memory_types=["semantic"],
        include_terms=[],
        exclude_terms=[],
        prototype_examples=[],
        near_but_different=[],
        centroid=[0.0] * 768,
        query_fingerprint=["a real user query"],
    )
    text = card.to_card_text()
    assert "a real user query" in text


def test_from_dict_ignores_unknown_keys():
    """Extra keys in persisted dict (e.g., from future versions) must be silently dropped."""
    d = {
        "anchor_id": 1,
        "name": "test",
        "meaning": "test",
        "memory_types": [],
        "include_terms": [],
        "exclude_terms": [],
        "prototype_examples": [],
        "near_but_different": [],
        "centroid": [0.0] * 768,
        "future_field_not_in_dataclass": "some_value",  # unknown key
    }
    card = AnchorCard.from_dict(d)
    assert card.anchor_id == 1


def test_from_dict_backward_compat_missing_query_fields():
    """Old atlas JSON without new fields should load without KeyError."""
    old_dict = {
        "anchor_id": 99,
        "name": "old-anchor",
        "meaning": "old meaning",
        "memory_types": ["semantic"],
        "include_terms": [],
        "exclude_terms": [],
        "prototype_examples": [],
        "near_but_different": [],
        "centroid": [0.0] * 768,
        "memory_count": 5,
        "is_novelty": False,
        "status": "active",
        "metadata": {},
    }
    card = AnchorCard.from_dict(old_dict)
    assert card.generated_query_patterns == []
    assert card.query_fingerprint == []


def test_anchor_index_load_logs_source_unknown_when_meta_missing(tmp_path, caplog):
    """Loading a refined file with no sibling .meta.json logs source=unknown at INFO."""
    import json
    import logging
    from psa.anchor import AnchorIndex

    cards = [
        {
            "anchor_id": 1,
            "name": "a",
            "meaning": "m",
            "memory_types": ["semantic"],
            "include_terms": [],
            "exclude_terms": [],
            "prototype_examples": [],
            "near_but_different": [],
            "centroid": [0.0] * 768,
            "memory_count": 1,
            "is_novelty": False,
            "status": "active",
            "metadata": {},
            "generated_query_patterns": [],
            "query_fingerprint": [],
        }
    ]
    (tmp_path / "anchor_cards_refined.json").write_text(json.dumps(cards))
    np.save(tmp_path / "centroids.npy", np.zeros((1, 768), dtype=np.float32))

    caplog.set_level(logging.INFO, logger="psa.anchor")
    AnchorIndex.load(str(tmp_path))

    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("source=unknown" in m for m in msgs), msgs


def test_anchor_index_load_logs_source_from_meta_file(tmp_path, caplog):
    """Loading a refined file with a sibling .meta.json logs its source."""
    import json
    import logging
    from psa.anchor import AnchorIndex

    cards = [
        {
            "anchor_id": 1,
            "name": "a",
            "meaning": "m",
            "memory_types": ["semantic"],
            "include_terms": [],
            "exclude_terms": [],
            "prototype_examples": [],
            "near_but_different": [],
            "centroid": [0.0] * 768,
            "memory_count": 1,
            "is_novelty": False,
            "status": "active",
            "metadata": {},
            "generated_query_patterns": [],
            "query_fingerprint": [],
        }
    ]
    (tmp_path / "anchor_cards_refined.json").write_text(json.dumps(cards))
    (tmp_path / "anchor_cards_refined.meta.json").write_text(
        json.dumps({"source": "oracle", "promoted": True})
    )
    np.save(tmp_path / "centroids.npy", np.zeros((1, 768), dtype=np.float32))

    caplog.set_level(logging.INFO, logger="psa.anchor")
    AnchorIndex.load(str(tmp_path))

    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("source=oracle" in m for m in msgs), msgs


def test_anchor_index_load_ignores_candidate_when_no_refined(tmp_path):
    """A candidate file alone does NOT cause auto-load; falls back to anchor_cards.json."""
    import json
    from psa.anchor import AnchorIndex

    raw = [
        {
            "anchor_id": 1,
            "name": "raw",
            "meaning": "m",
            "memory_types": ["semantic"],
            "include_terms": [],
            "exclude_terms": [],
            "prototype_examples": [],
            "near_but_different": [],
            "centroid": [0.0] * 768,
            "memory_count": 1,
            "is_novelty": False,
            "status": "active",
            "metadata": {},
            "generated_query_patterns": ["raw pattern"],
            "query_fingerprint": [],
        }
    ]
    candidate = [
        {
            **raw[0],
            "generated_query_patterns": ["raw pattern", "candidate pattern"],
        }
    ]
    (tmp_path / "anchor_cards.json").write_text(json.dumps(raw))
    (tmp_path / "anchor_cards_candidate.json").write_text(json.dumps(candidate))
    np.save(tmp_path / "centroids.npy", np.zeros((1, 768), dtype=np.float32))

    idx = AnchorIndex.load(str(tmp_path))
    assert idx._cards[0].generated_query_patterns == ["raw pattern"]
