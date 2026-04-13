"""
coactivation_data.py — Generate training data for CoActivationModel.

Reads oracle labels (JSONL), embeds queries, scores all atlas anchors,
and writes a single coactivation_train.npz file for use by
CoActivationTrainer.
"""

import json
import logging
import os

import numpy as np

logger = logging.getLogger("psa.training.coactivation_data")


def _ensure_cpu_default():
    """Force CPU as default torch device to prevent MPS SIGSEGV."""
    try:
        import torch

        torch.set_default_device("cpu")
    except (ImportError, RuntimeError):
        pass


def generate_coactivation_data(
    oracle_labels_path: str,
    output_path: str,
    full_atlas_scorer,
    embedding_model,
    atlas,
) -> int:
    """
    Generate coactivation training data from oracle labels.

    Parameters
    ----------
    oracle_labels_path:
        Path to JSONL file containing oracle labels.
        Each line must have ``query`` and ``winning_oracle_set`` fields.
    output_path:
        Directory where ``coactivation_train.npz`` will be written.
    full_atlas_scorer:
        ``FullAtlasScorer`` instance — used to score all atlas anchors.
    embedding_model:
        ``EmbeddingModel`` instance — used to embed queries.
    atlas:
        ``Atlas`` instance — provides anchor ordering via ``atlas.cards``.

    Returns
    -------
    int
        Number of training examples written.
    """
    _ensure_cpu_default()

    cards = atlas.cards
    n_anchors = len(cards)
    anchor_id_to_idx = {card.anchor_id: idx for idx, card in enumerate(cards)}

    query_vecs_list = []
    ce_scores_list = []
    gold_masks_list = []
    gold_ks_list = []

    # Precompute centroids array (n_anchors, 768) in card order
    centroids = np.array(
        [np.asarray(card.centroid, dtype=np.float32) for card in cards],
        dtype=np.float32,
    )

    n_written = 0

    with open(oracle_labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                label = json.loads(line)
            except json.JSONDecodeError:
                continue

            query = label.get("query")
            winning_set = label.get("winning_oracle_set")

            if not query or not winning_set:
                continue

            # Embed query
            query_vec = np.asarray(embedding_model.embed(query), dtype=np.float32)

            # Score all atlas anchors
            anchor_scores = full_atlas_scorer.score_all(query, query_vec=query_vec)

            # Build ce_scores in card order (by anchor_id_to_idx)
            ce_scores = np.zeros(n_anchors, dtype=np.float32)
            for as_ in anchor_scores:
                idx = anchor_id_to_idx.get(as_.anchor_id)
                if idx is not None:
                    ce_scores[idx] = float(as_.ce_score)

            # Build gold_mask: 1.0 for gold anchors, 0.0 for rest
            gold_mask = np.zeros(n_anchors, dtype=np.float32)
            gold_count = 0
            for anchor_id in winning_set:
                # winning_oracle_set may contain int or string keys
                aid = int(anchor_id) if not isinstance(anchor_id, int) else anchor_id
                idx = anchor_id_to_idx.get(aid)
                if idx is not None:
                    gold_mask[idx] = 1.0
                    gold_count += 1

            query_vecs_list.append(query_vec)
            ce_scores_list.append(ce_scores)
            gold_masks_list.append(gold_mask)
            gold_ks_list.append(gold_count)

            n_written += 1
            if n_written % 50 == 0:
                logger.info("Processed %d oracle labels", n_written)
                # Free MPS memory to prevent SIGSEGV from accumulation
                import gc

                gc.collect()
                try:
                    import torch

                    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                except ImportError:
                    pass

    if n_written == 0:
        logger.warning("No valid oracle labels found in %s", oracle_labels_path)
        return 0

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "coactivation_train.npz")

    np.savez(
        out_file,
        query_vecs=np.stack(query_vecs_list, axis=0).astype(np.float32),
        ce_scores=np.stack(ce_scores_list, axis=0).astype(np.float32),
        gold_masks=np.stack(gold_masks_list, axis=0).astype(np.float32),
        gold_ks=np.array(gold_ks_list, dtype=np.int32),
        centroids=centroids,
        anchor_ids=np.array([card.anchor_id for card in cards], dtype=np.int32),
    )

    logger.info(
        "Saved %d coactivation training examples to %s",
        n_written,
        out_file,
    )
    return n_written
