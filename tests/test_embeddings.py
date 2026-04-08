"""
test_embeddings.py — Tests for psa.embeddings.EmbeddingModel.

Covers:
- dim=768, L2 normalization
- cosine_similarity on normalized vectors
- l2_normalize on raw vectors
- batch consistency (batch == single)
- ImportError when sentence-transformers is absent
"""

import importlib
import math
import sys

import pytest

# Eagerly check that sentence-transformers is installed so we can skip the
# whole module if it's not, without getting a hard import error.
try:
    import sentence_transformers  # noqa: F401

    HAS_ST = True
except ImportError:
    HAS_ST = False

pytestmark = pytest.mark.skipif(not HAS_ST, reason="sentence-transformers not installed")


from psa.embeddings import EmbeddingModel  # noqa: E402


@pytest.fixture(scope="module")
def model():
    """Load the embedding model once for the whole module."""
    return EmbeddingModel()


# ── Single embed ─────────────────────────────────────────────────────────────


def test_embed_dim(model):
    vec = model.embed("Hello, PSA.")
    assert len(vec) == EmbeddingModel.DIM


def test_embed_l2_normalized(model):
    vec = model.embed("Normalize me.")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


def test_embed_returns_floats(model):
    vec = model.embed("type check")
    assert all(isinstance(x, float) for x in vec)


# ── Batch embed ──────────────────────────────────────────────────────────────


def test_embed_batch_dim(model):
    texts = ["alpha", "beta", "gamma"]
    vecs = model.embed_batch(texts)
    assert len(vecs) == 3
    for vec in vecs:
        assert len(vec) == EmbeddingModel.DIM


def test_embed_batch_l2_normalized(model):
    vecs = model.embed_batch(["one", "two", "three"])
    for vec in vecs:
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-5


def test_batch_consistency(model):
    """
    embed() and a single-item embed_batch() must produce identical vectors.

    Note: BGE-family models change an item's embedding when it is padded to match
    a longer batch-mate (bidirectional attention sees padded tokens even when they
    are masked from the mean-pooling step).  This is expected model behaviour.
    We therefore only test consistency for single-item batches, where no
    padding occurs.
    """
    texts = ["PSA rocks", "memory is key"]
    for text in texts:
        single = model.embed(text)
        via_batch = model.embed_batch([text])[0]  # 1-item batch: no padding
        sim = EmbeddingModel.cosine_similarity(single, via_batch)
        assert sim > 0.9999, f"Single vs 1-item batch similarity too low: {sim}"


# ── cosine_similarity ────────────────────────────────────────────────────────


def test_cosine_self_similarity(model):
    vec = model.embed("identical text")
    sim = EmbeddingModel.cosine_similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-5


def test_cosine_similar_texts(model):
    a = model.embed("The cat sat on the mat")
    b = model.embed("A cat was sitting on a mat")
    sim = EmbeddingModel.cosine_similarity(a, b)
    assert sim > 0.8, f"Similar texts should have high cosine similarity, got {sim}"


def test_cosine_dissimilar_texts(model):
    a = model.embed("JWT authentication with 24h expiry")
    b = model.embed("tropical rainforest ecology")
    sim = EmbeddingModel.cosine_similarity(a, b)
    assert sim < 0.7, f"Dissimilar texts should have lower cosine similarity, got {sim}"


def test_cosine_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        EmbeddingModel.cosine_similarity([0.1, 0.2], [0.1, 0.2, 0.3])


def test_cosine_clamped_to_one():
    # A vector dotted with itself (after floating-point error) should be clamped.
    v = [1.0, 0.0, 0.0]
    # Manually introduce drift that would make dot > 1.0
    v_drift = [1.0000001, 0.0, 0.0]
    sim = EmbeddingModel.cosine_similarity(v, v_drift)
    assert sim <= 1.0


# ── l2_normalize ─────────────────────────────────────────────────────────────


def test_l2_normalize():
    raw = [3.0, 4.0]  # norm = 5.0
    normalized = EmbeddingModel.l2_normalize(raw)
    assert abs(normalized[0] - 0.6) < 1e-6
    assert abs(normalized[1] - 0.8) < 1e-6


def test_l2_normalize_zero_vector():
    """Zero vector cannot be normalized; returned unchanged."""
    zero = [0.0, 0.0, 0.0]
    result = EmbeddingModel.l2_normalize(zero)
    assert result == zero


def test_l2_normalize_does_not_mutate():
    original = [3.0, 4.0]
    copy = original[:]
    EmbeddingModel.l2_normalize(original)
    assert original == copy


# ── ImportError guard ─────────────────────────────────────────────────────────


def test_import_error_without_sentence_transformers(monkeypatch):
    """EmbeddingModel.__init__ must raise ImportError when ST is unavailable."""
    import psa.embeddings as emb_module

    original = emb_module._require_sentence_transformers

    def _fake_require():
        raise ImportError("sentence-transformers is required for PSA embeddings.")

    monkeypatch.setattr(emb_module, "_require_sentence_transformers", _fake_require)

    with pytest.raises(ImportError, match="sentence-transformers"):
        EmbeddingModel()

    monkeypatch.setattr(emb_module, "_require_sentence_transformers", original)
