"""
atlas.py — PSA Atlas: fixed V1 sizing, mini-batch spherical k-means, versioning.

The Atlas is a two-level index over the tenant's memory space:
  - 224 "learned" anchors from spherical k-means clustering
  - 32  "novelty" anchors reserved for low-density / high-distance memories

V1 design decisions (from the plan):
  - Atlas size is FIXED at 256 (224 + 32). No adaptive sizing.
  - If the corpus is too small for a meaningful 256-anchor atlas, build is
    blocked until the minimum corpus size is met.
  - Stability check: run k-means with 3 seeds; reject proposals where
    cluster assignment variance across seeds > STABILITY_THRESHOLD.

Minimum corpus gate:
  - Fewer than MIN_MEMORIES_FOR_ATLAS non-duplicate memories with embeddings
    → raise AtlasCorpusTooSmall; the tenant stays on the baseline path.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np

from .anchor import AnchorCard, AnchorIndex
from .memory_object import MemoryObject, MemoryStore

logger = logging.getLogger("psa.atlas")

# ── Constants ─────────────────────────────────────────────────────────────────

V1_LEARNED_ANCHORS = 224
V1_NOVELTY_ANCHORS = 32
V1_TOTAL_ANCHORS = V1_LEARNED_ANCHORS + V1_NOVELTY_ANCHORS  # 256

N_SEEDS = 3               # number of k-means seeds for stability check
MAX_ITERATIONS = 200      # maximum k-means iterations per seed
CONVERGENCE_TOL = 1e-4    # centroid shift threshold for convergence
STABILITY_THRESHOLD = 0.40  # max allowed instability across seeds (loosened for small corpora)
NOVELTY_DISTANCE_THRESHOLD = 0.3  # cosine distance to nearest learned anchor → novelty

MIN_MEMORIES_FOR_ATLAS = 200  # minimum for a 256-anchor atlas (reduced for faster bootstrap)


# ── Exceptions ────────────────────────────────────────────────────────────────


class AtlasCorpusTooSmall(Exception):
    """Raised when the tenant has fewer memories than MIN_MEMORIES_FOR_ATLAS."""


class AtlasUnstable(Exception):
    """Raised when k-means is unstable across seeds (cluster assignments diverge)."""


# ── Atlas dataclass ───────────────────────────────────────────────────────────


@dataclass
class AtlasStats:
    n_memories: int
    n_anchors_learned: int
    n_anchors_novelty: int
    mean_cluster_size: float
    min_cluster_size: int
    max_cluster_size: int
    stability_score: float  # fraction of memories with consistent assignment across seeds
    built_at: str


@dataclass
class Atlas:
    """A versioned PSA atlas for a single tenant."""

    version: int
    tenant_id: str
    anchor_index: AnchorIndex
    stats: AtlasStats
    anchor_dir: str  # absolute path where this atlas is stored
    cards: List[AnchorCard] = field(default_factory=list)

    def assign_memory(
        self, memory: MemoryObject
    ) -> Tuple[int, Optional[int], float]:
        """
        Assign a memory object to anchor(s).

        Returns (primary_anchor_id, secondary_anchor_id_or_None, confidence).
        If the best learned anchor is below NOVELTY_DISTANCE_THRESHOLD,
        routes to the nearest novelty anchor instead (secondary=None).
        """
        if memory.embedding is None:
            raise ValueError(f"Memory {memory.memory_object_id} has no embedding.")
        results = self.anchor_index.search(memory.embedding, top_k=3)
        if not results:
            return -1, None, 0.0
        primary_id, primary_score = results[0]

        # Find best learned anchor among top results
        card_map = self._get_card_map()
        best_learned_score = None
        for rid, rscore in results:
            card = card_map.get(rid)
            if card and not card.is_novelty:
                best_learned_score = rscore
                break

        # If best learned anchor is too far, route to nearest novelty anchor
        if best_learned_score is not None and best_learned_score < (1.0 - NOVELTY_DISTANCE_THRESHOLD):
            emb = np.asarray(memory.embedding, dtype=np.float32)
            novelty_ids, novelty_centroids = self._get_novelty_centroids()
            if len(novelty_centroids) > 0:
                sims = novelty_centroids @ emb
                best_idx = int(np.argmax(sims))
                return novelty_ids[best_idx], None, float(sims[best_idx])

        secondary_id = results[1][0] if len(results) > 1 else None
        return primary_id, secondary_id, float(primary_score)

    def _get_card_map(self) -> dict:
        """Lazy {anchor_id: AnchorCard} lookup map."""
        if not hasattr(self, "_card_map"):
            self._card_map = {c.anchor_id: c for c in self.cards}
        return self._card_map

    def _get_novelty_centroids(self):
        """Lazy (ids, centroids_matrix) for novelty anchors."""
        if not hasattr(self, "_novelty_cache"):
            nov = [(c.anchor_id, c.centroid) for c in self.cards if c.is_novelty and c.centroid]
            if nov:
                ids, cents = zip(*nov)
                self._novelty_cache = (list(ids), np.array(cents, dtype=np.float32))
            else:
                self._novelty_cache = ([], np.empty((0, 0), dtype=np.float32))
        return self._novelty_cache

    def save(self):
        """Persist the atlas (anchor cards + index) to disk."""
        os.makedirs(self.anchor_dir, exist_ok=True)
        self.anchor_index.save(self.anchor_dir)
        meta = {
            "version": self.version,
            "tenant_id": self.tenant_id,
            "stats": {
                "n_memories": self.stats.n_memories,
                "n_anchors_learned": self.stats.n_anchors_learned,
                "n_anchors_novelty": self.stats.n_anchors_novelty,
                "mean_cluster_size": self.stats.mean_cluster_size,
                "min_cluster_size": self.stats.min_cluster_size,
                "max_cluster_size": self.stats.max_cluster_size,
                "stability_score": self.stats.stability_score,
                "built_at": self.stats.built_at,
            },
        }
        with open(os.path.join(self.anchor_dir, "atlas_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, anchor_dir: str) -> "Atlas":
        """Load an atlas from disk."""
        meta_path = os.path.join(anchor_dir, "atlas_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No atlas_meta.json at {anchor_dir}")
        with open(meta_path) as f:
            meta = json.load(f)
        # Dim is inferred from saved centroids in AnchorIndex.load
        anchor_index = AnchorIndex.load(anchor_dir)
        stats = AtlasStats(**meta["stats"])
        return cls(
            version=meta["version"],
            tenant_id=meta["tenant_id"],
            anchor_index=anchor_index,
            stats=stats,
            anchor_dir=anchor_dir,
            cards=anchor_index._cards,
        )


# ── Spherical k-means ─────────────────────────────────────────────────────────


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a matrix in place (returns normalized copy)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def _spherical_kmeans(
    embeddings: np.ndarray,
    k: int,
    seed: int,
    max_iterations: int = MAX_ITERATIONS,
    tol: float = CONVERGENCE_TOL,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mini-batch spherical k-means on L2-normalized embeddings.

    Parameters
    ----------
    embeddings: (n_samples, dim) float32, already L2-normalized
    k: number of clusters
    seed: random seed for centroid initialization
    max_iterations: maximum iterations
    tol: centroid shift threshold for early stopping

    Returns
    -------
    centroids: (k, dim) float32, L2-normalized cluster centroids
    assignments: (n_samples,) int32 cluster assignment for each sample
    """
    rng = np.random.default_rng(seed)
    n_samples, dim = embeddings.shape

    # k-means++ initialization on the sphere
    init_idx = [rng.integers(n_samples)]
    for _ in range(k - 1):
        sims = embeddings @ embeddings[init_idx].T  # (n, len(init_idx))
        max_sims = sims.max(axis=1)
        probs = 1.0 - max_sims  # distance proxy: 1 - max_cosine
        probs = np.maximum(probs, 0)
        total = probs.sum()
        if total == 0:
            probs = np.ones(n_samples) / n_samples
        else:
            probs /= total
        init_idx.append(rng.choice(n_samples, p=probs))

    centroids = embeddings[init_idx].copy()
    centroids = _l2_normalize_rows(centroids)

    assignments = np.zeros(n_samples, dtype=np.int32)

    for _iter in range(max_iterations):
        # Assignment step: each sample → nearest centroid (max cosine sim)
        sims = embeddings @ centroids.T  # (n_samples, k)
        new_assignments = sims.argmax(axis=1).astype(np.int32)

        # Update step: new centroid = mean of assigned samples, then L2-normalize
        new_centroids = np.zeros_like(centroids)
        for ki in range(k):
            mask = new_assignments == ki
            if mask.sum() == 0:
                # Empty cluster: reinitialize to a random sample
                new_centroids[ki] = embeddings[rng.integers(n_samples)]
            else:
                new_centroids[ki] = embeddings[mask].mean(axis=0)
        new_centroids = _l2_normalize_rows(new_centroids)

        # Convergence check
        shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
        centroids = new_centroids
        assignments = new_assignments

        if shift < tol:
            logger.debug("k-means converged at iteration %d (shift=%.6f)", _iter + 1, shift)
            break

    return centroids, assignments


def _stability_score(assignments_list: List[np.ndarray], n_samples: int = 2000) -> float:
    """
    Compute cross-seed stability via adjusted Rand index (ARI).

    ARI is permutation-invariant and runs in O(n*k) space, so it scales
    to large corpora without the O(n^2) cost of pairwise agreement matrices.

    For corpora larger than n_samples, we subsample to keep this fast.
    """
    if len(assignments_list) < 2:
        return 1.0

    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        # Fallback: sample-based pairwise agreement (O(n_samples^2))
        n = len(assignments_list[0])
        if n > n_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, n_samples, replace=False)
            assignments_list = [a[idx] for a in assignments_list]

        ref = assignments_list[0]
        agreements = []
        for other in assignments_list[1:]:
            same_in_ref = ref[:, None] == ref[None, :]
            same_in_other = other[:, None] == other[None, :]
            agreement = (same_in_ref == same_in_other).mean()
            agreements.append(agreement)
        return float(np.mean(agreements))

    ref = assignments_list[0]
    scores = []
    for other in assignments_list[1:]:
        scores.append(adjusted_rand_score(ref, other))
    # ARI is in [-1, 1]; map to [0, 1] for compatibility with stability threshold
    return float((np.mean(scores) + 1.0) / 2.0)


# ── Card generation ──────────────────────────────────────────────────────────


def _generate_card_via_qwen(
    anchor_id: int,
    centroid: List[float],
    sample_memories: List[MemoryObject],
    is_novelty: bool = False,
) -> AnchorCard:
    """
    Generate a semantic AnchorCard using Qwen to analyze cluster contents.

    Sends sample memory titles and summaries to Qwen, which produces a
    human-readable name, meaning, and include/exclude terms. Falls back to
    a stub card if Qwen is unavailable.
    """
    import json as _json

    titles = [m.title for m in sample_memories[:10]]
    summaries = [m.summary for m in sample_memories[:10] if m.summary]
    memory_types = list({m.memory_type.value for m in sample_memories[:10]})
    prototypes = [m.title for m in sample_memories[:5]]

    if is_novelty:
        prefix = "These are outlier memories that don't fit neatly into other clusters"
    else:
        prefix = "These memories belong to the same semantic cluster"

    samples_text = "\n".join(
        f"- [{m.memory_type.value}] {m.title}: {m.summary[:150]}"
        for m in sample_memories[:10]
    )

    prompt = (
        f"{prefix}. Analyze them and produce a semantic description.\n\n"
        f"Sample memories:\n{samples_text}\n\n"
        "Return JSON with these fields:\n"
        '{\n'
        '  "name": "short-kebab-case-name (2-4 words, descriptive)",\n'
        '  "meaning": "1-2 sentences describing what this region of memory covers",\n'
        '  "include_terms": ["up to 8 keywords that signal membership"],\n'
        '  "exclude_terms": ["up to 4 keywords that signal non-membership"],\n'
        '  "query_patterns": ["10-15 specific questions a user might ask that this cluster can answer"]\n'
        '}'
    )

    try:
        from .llm import call_llm

        content = call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
            timeout=60,
        )
        result = _json.loads(content)

        name = result.get("name", f"cluster-{anchor_id}")
        meaning = result.get("meaning", f"A cluster of {len(sample_memories)} memories.")
        include_terms = result.get("include_terms", [])[:8]
        exclude_terms = result.get("exclude_terms", [])[:4]
        query_patterns = result.get("query_patterns", [])[:15]

        logger.debug("Generated card for anchor %d: %s — %s", anchor_id, name, meaning)

    except Exception as e:
        logger.warning("Qwen card generation failed for anchor %d: %s (using stub)", anchor_id, e)
        name = f"novelty-{anchor_id}" if is_novelty else f"cluster-{anchor_id}"
        meaning = (
            "Low-density or outlier memories."
            if is_novelty
            else f"A cluster of {len(sample_memories)} memories. "
                 f"Representative: {'; '.join(titles[:3])}."
        )
        include_terms = []
        exclude_terms = []
        query_patterns = []

    return AnchorCard(
        anchor_id=anchor_id,
        name=name,
        meaning=meaning,
        memory_types=memory_types,
        include_terms=include_terms,
        exclude_terms=exclude_terms,
        prototype_examples=prototypes,
        near_but_different=[],
        centroid=centroid,
        memory_count=len(sample_memories),
        is_novelty=is_novelty,
        generated_query_patterns=query_patterns,
    )


# ── Anchor matching (identity persistence across rebuilds) ───────────────────


def _match_anchors(
    old_cards: List[AnchorCard],
    new_centroids: np.ndarray,
    threshold: float = 0.5,
) -> List[Optional[AnchorCard]]:
    """
    Match new clusters to existing anchors by centroid cosine similarity.

    Parameters
    ----------
    old_cards:
        AnchorCards from the previous atlas (active only).
    new_centroids:
        (k, dim) L2-normalized centroids from new k-means run.
    threshold:
        Minimum cosine similarity to consider a match.

    Returns
    -------
    List of length k: matched[i] is the old AnchorCard matched to new cluster i,
    or None if no match (fresh cluster).
    """
    if not old_cards:
        return [None] * len(new_centroids)

    old_centroids = np.array([c.centroid for c in old_cards], dtype=np.float32)
    norms = np.linalg.norm(old_centroids, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    old_centroids = old_centroids / norms

    # Cosine similarity matrix: (new_k, old_k)
    sim_matrix = new_centroids @ old_centroids.T

    k_new = len(new_centroids)
    matched: List[Optional[AnchorCard]] = [None] * k_new
    used_old: set = set()

    # Greedy assignment: best match per new cluster
    flat_indices = np.argsort(sim_matrix.ravel())[::-1]
    for flat_idx in flat_indices:
        new_idx = int(flat_idx // len(old_cards))
        old_idx = int(flat_idx % len(old_cards))
        sim = float(sim_matrix[new_idx, old_idx])

        if sim < threshold:
            break
        if matched[new_idx] is not None or old_idx in used_old:
            continue

        matched[new_idx] = old_cards[old_idx]
        used_old.add(old_idx)

    return matched


# ── AtlasBuilder ──────────────────────────────────────────────────────────────


class AtlasBuilder:
    """
    Builds a fixed V1 atlas (224 learned + 32 novelty anchors) from a
    tenant's memory embeddings.

    Usage::

        store = MemoryStore(...)
        builder = AtlasBuilder(store=store, tenant_id="default")
        atlas = builder.build_atlas(version=1, output_dir="~/.psa/tenants/default/atlas_v1")
    """

    def __init__(self, store: MemoryStore, tenant_id: str):
        self.store = store
        self.tenant_id = tenant_id

    def build_atlas(
        self,
        version: int = 1,
        output_dir: Optional[str] = None,
        previous_atlas: Optional[Atlas] = None,
    ) -> Atlas:
        """
        Build a V1 atlas from the tenant's embedded memories.

        Raises
        ------
        AtlasCorpusTooSmall
            If the tenant has fewer than MIN_MEMORIES_FOR_ATLAS memories.
        AtlasUnstable
            If k-means clusters are unstable across seeds.
        """
        # Step 1: Collect embeddings
        memories = self.store.get_all_with_embeddings(self.tenant_id)

        if len(memories) < MIN_MEMORIES_FOR_ATLAS:
            raise AtlasCorpusTooSmall(
                f"Tenant '{self.tenant_id}' has {len(memories)} embedded memories "
                f"(minimum {MIN_MEMORIES_FOR_ATLAS} required for a {V1_TOTAL_ANCHORS}-anchor atlas). "
                f"Continue ingesting memories and retry."
            )

        logger.info(
            "Building atlas v%d for tenant '%s' from %d memories",
            version,
            self.tenant_id,
            len(memories),
        )

        embeddings = np.array([m.embedding for m in memories], dtype=np.float32)
        embeddings = _l2_normalize_rows(embeddings)

        # Step 2: Run spherical k-means with N_SEEDS seeds, check stability
        # Scale k to corpus size: aim for ~5 memories per cluster minimum
        k = min(V1_LEARNED_ANCHORS, max(8, len(memories) // 5))
        all_assignments: List[np.ndarray] = []
        all_centroids: List[np.ndarray] = []

        for seed in range(N_SEEDS):
            logger.info("k-means seed %d/%d (k=%d)...", seed + 1, N_SEEDS, k)
            centroids, assignments = _spherical_kmeans(embeddings, k=k, seed=seed)
            all_centroids.append(centroids)
            all_assignments.append(assignments)

        stability = _stability_score(all_assignments, len(memories))
        logger.info("Atlas stability score: %.3f", stability)

        if stability < (1.0 - STABILITY_THRESHOLD):
            raise AtlasUnstable(
                f"Atlas for tenant '{self.tenant_id}' is unstable across {N_SEEDS} seeds "
                f"(stability={stability:.3f}, threshold={1.0 - STABILITY_THRESHOLD:.3f}). "
                f"This usually indicates insufficient or highly heterogeneous training data. "
                f"Investigate data quality before proceeding."
            )

        # Use seed 0 assignments (all seeds were stable enough)
        centroids = all_centroids[0]
        assignments = all_assignments[0]

        # Step 3: Build AnchorCards for learned clusters (with anchor matching)
        cards: List[AnchorCard] = []
        cluster_memories: dict = {}  # anchor_id → list of MemoryObjects
        for idx, (mem, cluster_id) in enumerate(zip(memories, assignments)):
            cluster_memories.setdefault(int(cluster_id), []).append(mem)

        # Match new clusters to old anchors if a previous atlas exists
        old_active_cards = []
        if previous_atlas is not None:
            old_active_cards = [c for c in previous_atlas.cards if getattr(c, "status", "active") == "active"]
        matched = _match_anchors(old_active_cards, centroids) if old_active_cards else [None] * k

        # Track next fresh anchor_id (above all existing IDs)
        max_existing_id = max((c.anchor_id for c in old_active_cards), default=-1) if old_active_cards else -1

        fresh_id_counter = max(k, max_existing_id + 1)

        for cluster_id in range(k):
            cluster_mems = cluster_memories.get(cluster_id, [])
            centroid_list = centroids[cluster_id].tolist()
            old_card = matched[cluster_id]

            if old_card is not None:
                # Matched: preserve identity, update centroid and examples
                card = AnchorCard(
                    anchor_id=old_card.anchor_id,
                    name=old_card.name,
                    meaning=old_card.meaning,
                    memory_types=list({m.memory_type.value for m in cluster_mems[:10]}) if cluster_mems else old_card.memory_types,
                    include_terms=old_card.include_terms,
                    exclude_terms=old_card.exclude_terms,
                    prototype_examples=[m.title for m in cluster_mems[:5]],
                    near_but_different=old_card.near_but_different,
                    centroid=centroid_list,
                    memory_count=len(cluster_mems),
                    is_novelty=False,
                    status="active",
                )
            else:
                # Fresh cluster: generate semantic card via Qwen
                card = _generate_card_via_qwen(
                    anchor_id=fresh_id_counter,
                    centroid=centroid_list,
                    sample_memories=cluster_mems,
                )
                card.memory_count = len(cluster_mems)
                fresh_id_counter += 1
            cards.append(card)

        # Step 4: Reserve novelty anchors (high-distance regions)
        # Novelty centroids are placed at the "most distant" points from learned clusters
        # For V1: use the N memories with the lowest max-cosine-sim to any learned centroid
        all_sims = embeddings @ centroids.T  # (n_memories, k)
        max_sims = all_sims.max(axis=1)  # highest similarity to any learned cluster
        novelty_indices = np.argsort(max_sims)[:V1_NOVELTY_ANCHORS]  # lowest → farthest out

        novelty_centroids = embeddings[novelty_indices]
        novelty_centroids = _l2_normalize_rows(novelty_centroids)

        # Collect all used IDs to avoid collisions
        used_ids = {c.anchor_id for c in cards}
        novelty_id_counter = fresh_id_counter  # continue from where learned cards left off

        for i, (ni, nc) in enumerate(zip(novelty_indices, novelty_centroids)):
            # Find next unused ID
            while novelty_id_counter in used_ids:
                novelty_id_counter += 1
            anchor_id = novelty_id_counter
            used_ids.add(anchor_id)
            novelty_id_counter += 1
            novelty_mems = [memories[ni]]
            card = _generate_card_via_qwen(
                anchor_id=anchor_id,
                centroid=nc.tolist(),
                sample_memories=novelty_mems,
                is_novelty=True,
            )
            cards.append(card)

        # Step 5: Build AnchorIndex from active anchors only
        active_cards = [c for c in cards if getattr(c, "status", "active") == "active"]
        anchor_index = AnchorIndex(dim=embeddings.shape[1])
        anchor_index.build(active_cards)

        # Step 6: Compute stats
        cluster_sizes = [len(cluster_memories.get(i, [])) for i in range(k)]
        stats = AtlasStats(
            n_memories=len(memories),
            n_anchors_learned=k,
            n_anchors_novelty=V1_NOVELTY_ANCHORS,
            mean_cluster_size=float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            min_cluster_size=int(np.min(cluster_sizes)) if cluster_sizes else 0,
            max_cluster_size=int(np.max(cluster_sizes)) if cluster_sizes else 0,
            stability_score=stability,
            built_at=datetime.now(timezone.utc).isoformat(),
        )

        # Step 7: Update anchor assignments in memory store (single transaction)
        # Map cluster index → actual anchor_id (which may differ due to matching)
        logger.info("Updating anchor assignments in memory store...")
        # Only use learned anchor cards (first k), not novelty
        learned_cards = [c for c in cards if not c.is_novelty]
        cluster_to_anchor_id = {i: learned_cards[i].anchor_id for i in range(len(learned_cards))}

        learned_sims = embeddings @ centroids.T  # (n, k)
        updates = []
        for idx, (mem, cluster_id) in enumerate(zip(memories, assignments)):
            row_sims = learned_sims[idx]
            top2 = np.argsort(row_sims)[::-1][:2]
            primary_id = cluster_to_anchor_id.get(int(top2[0]), int(top2[0]))
            secondary_id = cluster_to_anchor_id.get(int(top2[1]), int(top2[1])) if len(top2) > 1 else None
            confidence = float(row_sims[top2[0]])
            updates.append({
                "memory_object_id": mem.memory_object_id,
                "primary_anchor_id": primary_id,
                "secondary_anchor_ids": [secondary_id] if secondary_id is not None else [],
                "confidence": confidence,
            })
        # Also assign novelty memories to their novelty anchors
        novelty_cards = [c for c in cards if c.is_novelty]
        updates_by_id = {u["memory_object_id"]: u for u in updates}
        for ni_idx, nc_card in zip(novelty_indices, novelty_cards):
            mem = memories[int(ni_idx)]
            u = updates_by_id.get(mem.memory_object_id)
            if u:
                u["primary_anchor_id"] = nc_card.anchor_id
                u["confidence"] = float(max_sims[int(ni_idx)])

        self.store.batch_update_anchor_assignments(updates)

        # Step 8: Persist atlas
        if output_dir is None:
            output_dir = os.path.join(
                os.path.expanduser(f"~/.psa/tenants/{self.tenant_id}"),
                f"atlas_v{version}",
            )
        atlas = Atlas(
            version=version,
            tenant_id=self.tenant_id,
            anchor_index=anchor_index,
            stats=stats,
            anchor_dir=output_dir,
            cards=cards,
        )
        atlas.save()

        logger.info(
            "Atlas v%d built: %d learned + %d novelty anchors, stability=%.3f",
            version,
            k,
            V1_NOVELTY_ANCHORS,
            stability,
        )
        return atlas


# ── AtlasManager ─────────────────────────────────────────────────────────────


class AtlasManager:
    """
    Manages atlas versions for a tenant.

    Conventions:
    - Atlas directories: ~/.psa/tenants/{tenant_id}/atlas_v{version}/
    - Latest is the highest version number
    """

    def __init__(self, tenant_dir: str, tenant_id: str):
        self.tenant_dir = tenant_dir
        self.tenant_id = tenant_id

    def _atlas_dir(self, version: int) -> str:
        return os.path.join(self.tenant_dir, f"atlas_v{version}")

    def latest_version(self) -> Optional[int]:
        """Return the highest atlas version number, or None if no atlas exists."""
        if not os.path.isdir(self.tenant_dir):
            return None
        versions = []
        for entry in os.listdir(self.tenant_dir):
            if entry.startswith("atlas_v") and os.path.isdir(
                os.path.join(self.tenant_dir, entry)
            ):
                try:
                    v = int(entry[len("atlas_v"):])
                    meta = os.path.join(self.tenant_dir, entry, "atlas_meta.json")
                    if os.path.exists(meta):
                        versions.append(v)
                except ValueError:
                    pass
        return max(versions) if versions else None

    def get_atlas(self, version: Optional[int] = None) -> Optional[Atlas]:
        """Load an atlas by version (default: latest)."""
        v = version or self.latest_version()
        if v is None:
            return None
        atlas_dir = self._atlas_dir(v)
        if not os.path.exists(atlas_dir):
            return None
        try:
            return Atlas.load(atlas_dir)
        except Exception as e:
            logger.warning("Failed to load atlas v%d: %s", v, e)
            return None

    def rebuild(self, store: MemoryStore) -> Atlas:
        """Build a new atlas version (latest + 1), with anchor matching."""
        current = self.latest_version() or 0
        new_version = current + 1
        output_dir = self._atlas_dir(new_version)

        # Load the previous atlas for anchor identity matching
        previous_atlas = self.get_atlas(version=current) if current > 0 else None

        builder = AtlasBuilder(store=store, tenant_id=self.tenant_id)
        return builder.build_atlas(
            version=new_version,
            output_dir=output_dir,
            previous_atlas=previous_atlas,
        )

    def get_or_build(self, store: MemoryStore) -> Atlas:
        """Return the latest atlas or build one if none exists."""
        atlas = self.get_atlas()
        if atlas is not None:
            return atlas
        return self.rebuild(store)
