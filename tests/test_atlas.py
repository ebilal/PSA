"""
test_atlas.py — Tests for psa.atlas.AtlasBuilder, Atlas, AtlasManager.

Covers:
- MIN_MEMORIES_FOR_ATLAS gate (AtlasCorpusTooSmall)
- Spherical k-means produces non-empty clusters
- Fixed V1 sizing (224 learned + 32 novelty = 256)
- Stability gate (AtlasUnstable if seeds diverge)
- Atlas.assign_memory returns valid anchor IDs
- AtlasManager version management (latest, rebuild)
- Save / load round-trip
- Stability score calculation

Notes:
- Building a real 256-anchor atlas requires >= 500 memories, which is
  impractical in unit tests. Tests use a small-k variant by monkeypatching
  V1_LEARNED_ANCHORS to a tiny value (8).
- Stability tests check the _stability_score helper directly.
"""

import os

import numpy as np
import pytest

import psa.atlas as atlas_mod
from psa.atlas import (
    AtlasBuilder,
    AtlasCorpusTooSmall,
    AtlasManager,
    _generate_card_via_qwen as _real_generate_card_via_qwen,
    _l2_normalize_rows,
    _spherical_kmeans,
    _stability_score,
)
from psa.memory_object import MemoryObject, MemoryStore, MemoryType


# ── Helpers ───────────────────────────────────────────────────────────────────


def _unit_vecs(n: int, dim: int = 64, seed: int = 0) -> np.ndarray:
    """Generate n random L2-normalized vectors of dimension dim."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    return _l2_normalize_rows(X)


def _make_store_with_memories(
    tmp_dir: str,
    n: int,
    dim: int = 64,
    tenant_id: str = "test",
) -> MemoryStore:
    """Create a MemoryStore pre-loaded with n memories that have dim-dimensional embeddings."""
    db_path = os.path.join(tmp_dir, "mem.sqlite3")
    store = MemoryStore(db_path=db_path)
    vecs = _unit_vecs(n, dim=dim)
    for i in range(n):
        mo = MemoryObject.create(
            tenant_id=tenant_id,
            memory_type=MemoryType.SEMANTIC,
            title=f"Memory {i}",
            body=f"Body of memory {i}.",
            summary=f"Summary {i}.",
            source_ids=[],
            classification_reason="test",
            embedding=vecs[i].tolist(),
        )
        store.add(mo)
    return store


# ── _l2_normalize_rows ────────────────────────────────────────────────────────


def test_l2_normalize_rows_unit_norm():
    X = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    Xn = _l2_normalize_rows(X)
    norms = np.linalg.norm(Xn, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_l2_normalize_rows_zero_vector():
    X = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    Xn = _l2_normalize_rows(X)
    # Zero vector stays zero (no NaN)
    assert not np.any(np.isnan(Xn))


# ── _spherical_kmeans ─────────────────────────────────────────────────────────


def test_spherical_kmeans_shape():
    X = _unit_vecs(100, dim=32)
    centroids, assignments = _spherical_kmeans(X, k=8, seed=0, max_iterations=50)
    assert centroids.shape == (8, 32)
    assert assignments.shape == (100,)


def test_spherical_kmeans_centroids_normalized():
    X = _unit_vecs(200, dim=32)
    centroids, _ = _spherical_kmeans(X, k=10, seed=0, max_iterations=50)
    norms = np.linalg.norm(centroids, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_spherical_kmeans_all_samples_assigned():
    n = 150
    X = _unit_vecs(n, dim=32)
    _, assignments = _spherical_kmeans(X, k=6, seed=0, max_iterations=50)
    # Every sample is assigned to exactly one cluster
    assert len(assignments) == n
    assert assignments.min() >= 0
    assert assignments.max() < 6


def test_spherical_kmeans_nonempty_clusters():
    X = _unit_vecs(100, dim=32)
    _, assignments = _spherical_kmeans(X, k=5, seed=0, max_iterations=50)
    cluster_sizes = [(assignments == k).sum() for k in range(5)]
    assert all(s > 0 for s in cluster_sizes), f"Empty clusters: {cluster_sizes}"


# ── _stability_score ──────────────────────────────────────────────────────────


def test_stability_score_identical_assignments():
    a = np.array([0, 0, 1, 1, 2, 2])
    score = _stability_score([a, a, a], n_samples=6)
    assert abs(score - 1.0) < 1e-6


def test_stability_score_single_seed():
    a = np.array([0, 1, 2])
    score = _stability_score([a], n_samples=3)
    assert score == 1.0


def test_stability_score_completely_different():
    a = np.array([0, 0, 1, 1])
    b = np.array([1, 0, 0, 1])  # partial mismatch
    score = _stability_score([a, b], n_samples=4)
    assert 0.0 <= score <= 1.0


# ── AtlasCorpusTooSmall gate ──────────────────────────────────────────────────


def test_atlas_too_small_raises(tmp_dir):
    store = _make_store_with_memories(tmp_dir, n=10)  # below MIN
    builder = AtlasBuilder(store=store, tenant_id="test")
    with pytest.raises(AtlasCorpusTooSmall, match="minimum"):
        builder.build_atlas(version=1, output_dir=os.path.join(tmp_dir, "atlas_v1"))


# ── AtlasBuilder with small k (monkeypatched) ────────────────────────────────


@pytest.fixture
def small_k(monkeypatch):
    """
    Patch atlas constants to build a tiny 8+4=12 anchor atlas for unit tests.

    - V1 sizing reduced to 12 anchors (8 learned + 4 novelty)
    - MIN_MEMORIES_FOR_ATLAS reduced to 50
    - N_SEEDS=1 so the stability check is trivially bypassed (single seed → stability=1.0)
    - MAX_ITERATIONS=30 for speed
    """
    monkeypatch.setattr(atlas_mod, "V1_LEARNED_ANCHORS", 8)
    monkeypatch.setattr(atlas_mod, "V1_NOVELTY_ANCHORS", 4)
    monkeypatch.setattr(atlas_mod, "V1_TOTAL_ANCHORS", 12)
    monkeypatch.setattr(atlas_mod, "MIN_MEMORIES_FOR_ATLAS", 50)
    monkeypatch.setattr(atlas_mod, "N_SEEDS", 1)   # 1 seed → stability trivially passes
    monkeypatch.setattr(atlas_mod, "MAX_ITERATIONS", 30)


def test_atlas_build_produces_correct_anchor_count(tmp_dir, small_k):
    store = _make_store_with_memories(tmp_dir, n=80, dim=64)
    builder = AtlasBuilder(store=store, tenant_id="test")
    atlas = builder.build_atlas(
        version=1, output_dir=os.path.join(tmp_dir, "atlas_v1")
    )
    # 8 learned + 4 novelty = 12
    assert atlas.anchor_index.size == 12
    assert atlas.stats.n_anchors_learned == 8
    assert atlas.stats.n_anchors_novelty == 4


def test_atlas_build_novelty_anchors_flagged(tmp_dir, small_k):
    store = _make_store_with_memories(tmp_dir, n=80, dim=64)
    builder = AtlasBuilder(store=store, tenant_id="test")
    atlas = builder.build_atlas(
        version=1, output_dir=os.path.join(tmp_dir, "atlas_v1")
    )
    novelty_cards = [c for c in atlas.cards if c.is_novelty]
    assert len(novelty_cards) == 4


def test_atlas_build_updates_anchor_assignments(tmp_dir, small_k):
    store = _make_store_with_memories(tmp_dir, n=80, dim=64)
    builder = AtlasBuilder(store=store, tenant_id="test")
    builder.build_atlas(
        version=1, output_dir=os.path.join(tmp_dir, "atlas_v1")
    )
    # All memories should have a primary_anchor_id assigned
    memories = store.get_all_with_embeddings("test")
    assigned = [m for m in memories if m.primary_anchor_id is not None]
    assert len(assigned) == len(memories)


def test_atlas_stats_sane(tmp_dir, small_k):
    store = _make_store_with_memories(tmp_dir, n=80, dim=64)
    builder = AtlasBuilder(store=store, tenant_id="test")
    atlas = builder.build_atlas(
        version=1, output_dir=os.path.join(tmp_dir, "atlas_v1")
    )
    assert atlas.stats.n_memories == 80
    assert atlas.stats.mean_cluster_size > 0
    assert atlas.stats.min_cluster_size >= 0
    assert atlas.stats.max_cluster_size <= 80
    assert 0.0 <= atlas.stats.stability_score <= 1.0


# ── Atlas.assign_memory ───────────────────────────────────────────────────────


def test_atlas_assign_memory(tmp_dir, small_k):
    store = _make_store_with_memories(tmp_dir, n=80, dim=64)
    builder = AtlasBuilder(store=store, tenant_id="test")
    atlas = builder.build_atlas(
        version=1, output_dir=os.path.join(tmp_dir, "atlas_v1")
    )
    # Create a fresh memory with an embedding
    rng = np.random.default_rng(999)
    embedding = rng.standard_normal(64).astype(np.float32)
    embedding /= np.linalg.norm(embedding)

    mo = MemoryObject.create(
        tenant_id="test",
        memory_type=MemoryType.EPISODIC,
        title="Test episode",
        body="Something happened.",
        summary="An episode.",
        source_ids=[],
        classification_reason="test",
        embedding=embedding.tolist(),
    )
    primary_id, secondary_id, confidence = atlas.assign_memory(mo)
    valid_ids = {c.anchor_id for c in atlas.cards}
    assert primary_id in valid_ids  # must be a valid anchor in this atlas
    assert 0.0 <= confidence <= 1.0


def test_atlas_assign_memory_no_embedding_raises(tmp_dir, small_k):
    store = _make_store_with_memories(tmp_dir, n=80, dim=64)
    builder = AtlasBuilder(store=store, tenant_id="test")
    atlas = builder.build_atlas(
        version=1, output_dir=os.path.join(tmp_dir, "atlas_v1")
    )
    mo = MemoryObject.create(
        tenant_id="test",
        memory_type=MemoryType.SEMANTIC,
        title="T",
        body="B",
        summary="S",
        source_ids=[],
        classification_reason="x",
    )  # no embedding
    with pytest.raises(ValueError, match="no embedding"):
        atlas.assign_memory(mo)


# ── Save / load round-trip ────────────────────────────────────────────────────


def test_atlas_save_and_load(tmp_dir, small_k):
    store = _make_store_with_memories(tmp_dir, n=80, dim=64)
    builder = AtlasBuilder(store=store, tenant_id="test")
    atlas_dir = os.path.join(tmp_dir, "atlas_v1")
    builder.build_atlas(version=1, output_dir=atlas_dir)

    loaded = atlas_mod.Atlas.load(atlas_dir)
    assert loaded.version == 1
    assert loaded.tenant_id == "test"
    assert loaded.anchor_index.size == 12
    assert loaded.stats.n_memories == 80


# ── AtlasManager ──────────────────────────────────────────────────────────────


def test_atlas_manager_no_atlas(tmp_dir):
    mgr = AtlasManager(tenant_dir=tmp_dir, tenant_id="test")
    assert mgr.latest_version() is None
    assert mgr.get_atlas() is None


def test_atlas_manager_rebuild_and_versioning(tmp_dir, small_k):
    store = _make_store_with_memories(tmp_dir, n=80, dim=64, tenant_id="test")
    mgr = AtlasManager(tenant_dir=tmp_dir, tenant_id="test")
    atlas_v1 = mgr.rebuild(store)
    assert atlas_v1.version == 1
    assert mgr.latest_version() == 1

    atlas_v2 = mgr.rebuild(store)
    assert atlas_v2.version == 2
    assert mgr.latest_version() == 2


def test_atlas_manager_get_latest(tmp_dir, small_k):
    store = _make_store_with_memories(tmp_dir, n=80, dim=64)
    mgr = AtlasManager(tenant_dir=tmp_dir, tenant_id="test")
    mgr.rebuild(store)
    mgr.rebuild(store)
    atlas = mgr.get_atlas()
    assert atlas.version == 2


def test_atlas_manager_get_or_build(tmp_dir, small_k):
    store = _make_store_with_memories(tmp_dir, n=80, dim=64)
    mgr = AtlasManager(tenant_dir=tmp_dir, tenant_id="test")
    assert mgr.get_atlas() is None
    atlas = mgr.get_or_build(store)
    assert atlas.version == 1
    # Second call returns existing
    atlas2 = mgr.get_or_build(store)
    assert atlas2.version == 1


# ── Novelty routing in assign_memory ─────────────────────────────────────────


def test_assign_memory_routes_distant_to_novelty(tmp_dir, small_k):
    """Memories far from all learned anchors should route to a novelty anchor."""
    store = _make_store_with_memories(tmp_dir, n=80, dim=64)
    builder = AtlasBuilder(store=store, tenant_id="test")
    atlas = builder.build_atlas(
        version=1, output_dir=os.path.join(tmp_dir, "atlas_v1")
    )

    novelty_ids = {c.anchor_id for c in atlas.cards if c.is_novelty}
    learned_ids = {c.anchor_id for c in atlas.cards if not c.is_novelty}
    assert len(novelty_ids) > 0, "Atlas should have novelty anchors"

    # Create a memory with a random embedding orthogonal to all learned centroids
    # Use a very different direction to ensure low similarity
    rng = np.random.default_rng(42424242)
    emb = rng.standard_normal(64).astype(np.float32)
    # Negate the average learned centroid direction to push far away
    learned_centroids = np.array(
        [c.centroid for c in atlas.cards if not c.is_novelty], dtype=np.float32
    )
    avg_direction = learned_centroids.mean(axis=0)
    emb = -avg_direction + 0.1 * emb  # point away from all clusters
    emb = emb / np.linalg.norm(emb)

    mo = MemoryObject.create(
        tenant_id="test",
        memory_type=MemoryType.EPISODIC,
        title="Very distant memory",
        body="Something completely unrelated.",
        summary="Distant.",
        source_ids=[],
        classification_reason="test",
        embedding=emb.tolist(),
    )

    # Compute best learned anchor score to know what routing should do
    from psa.atlas import NOVELTY_DISTANCE_THRESHOLD
    emb_arr = np.asarray(emb, dtype=np.float32)
    best_learned_score = max(
        float(emb_arr @ np.asarray(c.centroid, dtype=np.float32))
        for c in atlas.cards if not c.is_novelty
    )

    primary_id, secondary_id, confidence = atlas.assign_memory(mo)

    if best_learned_score < (1.0 - NOVELTY_DISTANCE_THRESHOLD):
        # Novelty routing should have triggered
        assert primary_id in novelty_ids, (
            f"Expected novelty anchor but got learned anchor {primary_id} "
            f"(best_learned_score={best_learned_score:.3f})"
        )
        assert secondary_id is None, "Secondary should be None for novelty routing"
    else:
        # Normal routing — just verify valid anchor
        assert primary_id in (novelty_ids | learned_ids)


# ── _generate_card_via_qwen query_patterns ───────────────────────────────────


def test_generate_card_has_query_patterns(monkeypatch):
    """_generate_card_via_qwen should populate generated_query_patterns from LLM response."""
    import json
    from unittest.mock import MagicMock

    import psa.llm as llm_module

    mock_response = json.dumps({
        "name": "auth-patterns",
        "meaning": "Authentication design choices.",
        "include_terms": ["auth", "token"],
        "exclude_terms": ["ui"],
        "query_patterns": [
            "What auth library did we use?",
            "How do tokens expire?",
        ],
    })

    monkeypatch.setattr(llm_module, "call_llm", lambda **kw: mock_response)

    mo = MagicMock(spec=MemoryObject)
    mo.title = "JWT setup"
    mo.summary = "We use JWT."
    mo.memory_type = MemoryType.PROCEDURAL

    # Call the real implementation directly (captured at module import time,
    # before conftest's autouse fixture replaces atlas_mod._generate_card_via_qwen
    # with a stub).
    card = _real_generate_card_via_qwen(
        anchor_id=1,
        centroid=[0.0] * 768,
        sample_memories=[mo],
    )
    assert card.generated_query_patterns == [
        "What auth library did we use?",
        "How do tokens expire?",
    ]

    # Verify the [:15] cap is enforced
    mock_response_16 = json.dumps({
        "name": "auth-patterns",
        "meaning": "Auth stuff.",
        "include_terms": [],
        "exclude_terms": [],
        "query_patterns": [f"question {i}?" for i in range(16)],
    })
    monkeypatch.setattr(llm_module, "call_llm", lambda **kw: mock_response_16)
    card2 = _real_generate_card_via_qwen(
        anchor_id=2,
        centroid=[0.0] * 768,
        sample_memories=[mo],
    )
    assert len(card2.generated_query_patterns) == 15
