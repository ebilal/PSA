"""
embeddings.py — Embedding model wrapper for PSA.

Uses BAAI/bge-base-en-v1.5 via sentence-transformers.
L2-normalized, 768-dimensional embeddings.

No silent fallback — embedding space consistency is critical.
If sentence-transformers is not installed, raises ImportError immediately.
"""

import math
from typing import List, Optional


def _require_sentence_transformers():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for PSA embeddings.\n"
            "Install it with: pip install sentence-transformers\n"
            "or: pip install 'psa[training]'"
        )


class EmbeddingModel:
    """
    Wrapper around BAAI/bge-base-en-v1.5 for PSA embeddings.

    All embeddings are L2-normalized. Dot product == cosine similarity.
    Model is loaded lazily on first use.
    """

    MODEL_NAME = "BAAI/bge-base-en-v1.5"
    DIM = 768

    def __init__(self, model_name: Optional[str] = None):
        _require_sentence_transformers()
        self._model_name = model_name or self.MODEL_NAME
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

    def embed(self, text: str) -> List[float]:
        """Embed a single string. Returns an L2-normalized float list."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings. Returns L2-normalized float lists.

        Empty strings are embedded as zero-vectors (already normalized to
        length 0, which can't be unit-normalized; callers should filter them).
        """
        self._load()
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """
        Cosine similarity between two L2-normalized vectors.

        Since embeddings from embed() / embed_batch() are already L2-normalized,
        this is equivalent to the dot product.
        """
        if len(a) != len(b):
            raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
        dot = sum(x * y for x, y in zip(a, b))
        # Clamp to [-1, 1] to guard against floating-point drift.
        return max(-1.0, min(1.0, dot))

    @staticmethod
    def l2_normalize(vec: List[float]) -> List[float]:
        """L2-normalize a raw (non-normalized) vector in place. Returns new list."""
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec[:]
        return [x / norm for x in vec]
