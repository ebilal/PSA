"""
coactivation.py — Co-activation model for joint anchor selection.

CoActivationModel is a 2-layer transformer that takes cross-encoder scores,
atlas centroids, and query embedding as input, learns inter-anchor co-activation
patterns, and outputs refined scores plus a per-query adaptive threshold.

CoActivationSelector wraps the model for inference, consuming AnchorScore
lists and producing SelectedAnchor lists.
"""

import json
import logging
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .full_atlas_scorer import AnchorScore
from .selector import SelectedAnchor

logger = logging.getLogger("psa.coactivation")


# ── CoActivationModel ─────────────────────────────────────────────────────────


class CoActivationModel(nn.Module):
    """
    2-layer transformer over the full anchor set.

    Input per forward call:
      ce_scores  : (B, N)         — cross-encoder scores, one per anchor
      centroids  : (B, N, D)      — L2-normalised atlas centroids (D=768)
      query_vec  : (B, D)         — L2-normalised query embedding

    Output:
      refined_scores : (B, N)     — sigmoid-normalised scores in [0, 1]
      thresholds     : (B,)       — adaptive per-query threshold in (0, 1)

    Architecture
    ------------
    1. anchor_proj: (ce_score || centroid) -> d_model//2
    2. query_proj:  query_vec              -> d_model//2
    3. Concatenate  -> (B, N, d_model) tokens
    4. TransformerEncoder (batch_first=True)
    5. score_head: -> sigmoid -> refined_scores (B, N)
    6. threshold via dot-product attention:
         threshold_query (1,1,d_model) @ refined tokens.T -> softmax -> weighted sum
         -> threshold_head (MLP) -> sigmoid -> thresholds (B,)
    """

    def __init__(
        self,
        n_anchors: int = 256,
        centroid_dim: int = 768,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        anchor_feature_dim: int = 8,
    ):
        super().__init__()
        self.n_anchors = n_anchors
        self.centroid_dim = centroid_dim
        self.d_model = d_model
        self.anchor_feature_dim = anchor_feature_dim
        half = d_model // 2

        # Per-anchor projection: (1 scalar ce_score + centroid_dim + anchor_feature_dim) -> d_model//2
        self.anchor_proj = nn.Linear(1 + centroid_dim + anchor_feature_dim, half)

        # Query projection: centroid_dim -> d_model//2  (broadcast across anchors)
        self.query_proj = nn.Linear(centroid_dim, half)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Score head: d_model -> 1 (sigmoid applied afterwards)
        self.score_head = nn.Linear(d_model, 1)

        # Learnable threshold query token for attention pooling
        self.threshold_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Threshold head: d_model -> d_model//2 -> 1
        self.threshold_head = nn.Sequential(
            nn.Linear(d_model, half),
            nn.ReLU(),
            nn.Linear(half, 1),
        )

    def forward(
        self,
        ce_scores: torch.Tensor,
        centroids: torch.Tensor,
        query_vec: torch.Tensor,
        anchor_features: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        ce_scores       : (B, N)
        centroids       : (B, N, D)
        query_vec       : (B, D)
        anchor_features : (B, N, anchor_feature_dim) or None

        Returns
        -------
        refined_scores : (B, N)
        thresholds     : (B,)
        """
        B, N, D = centroids.shape

        # build per-anchor tokens
        # ce_scores: (B, N) -> (B, N, 1)
        ce = ce_scores.unsqueeze(-1)

        # anchor_input: (B, N, 1+D+anchor_feature_dim)
        if anchor_features is not None:
            anchor_input = torch.cat([ce, centroids, anchor_features], dim=-1)
        else:
            anchor_input = torch.cat(
                [
                    ce,
                    centroids,
                    torch.zeros(B, N, self.anchor_feature_dim, device=ce.device),
                ],
                dim=-1,
            )

        # anchor tokens: (B, N, 1+D+anchor_feature_dim) -> (B, N, d_model//2)
        anchor_tokens = self.anchor_proj(anchor_input)

        # query tokens: (B, D) -> (B, d_model//2) -> (B, 1, d_model//2) -> (B, N, d_model//2)
        query_tokens = self.query_proj(query_vec).unsqueeze(1).expand(B, N, -1)

        # concatenate -> (B, N, d_model)
        tokens = torch.cat([anchor_tokens, query_tokens], dim=-1)

        # transformer
        hidden = self.transformer(tokens)  # (B, N, d_model)

        # score head
        raw_scores = self.score_head(hidden).squeeze(-1)  # (B, N)
        refined_scores = torch.sigmoid(raw_scores)

        # threshold via attention pooling
        # threshold_query: (1, 1, d_model) -> (B, 1, d_model)
        tq = self.threshold_query.expand(B, 1, -1)

        # dot-product attention: (B, 1, d_model) @ (B, d_model, N) -> (B, 1, N)
        attn_logits = torch.bmm(tq, hidden.transpose(1, 2))  # (B, 1, N)
        attn_weights = torch.softmax(attn_logits, dim=-1)

        # weighted sum of hidden: (B, 1, N) @ (B, N, d_model) -> (B, 1, d_model)
        pooled = torch.bmm(attn_weights, hidden).squeeze(1)  # (B, d_model)

        thresholds = torch.sigmoid(self.threshold_head(pooled)).squeeze(-1)  # (B,)

        return refined_scores, thresholds


# ── CoActivationSelector ──────────────────────────────────────────────────────


class CoActivationSelector:
    """
    Inference wrapper around CoActivationModel.

    Usage::

        selector = CoActivationSelector(model=model, device="cpu")
        selected = selector.select(query_vec=qvec, anchor_scores=scores)

    Or load from disk::

        selector = CoActivationSelector.from_model_path("/path/to/dir", device="cpu")
    """

    def __init__(self, model: CoActivationModel, device: str = "cpu"):
        self.device = device
        self.model = model
        self.model.to(device)
        self.model.eval()

    def select(
        self,
        query_vec: np.ndarray,
        anchor_scores: List[AnchorScore],
        anchor_features: Optional[np.ndarray] = None,
    ) -> List[SelectedAnchor]:
        """
        Run the co-activation model and return selected anchors.

        Parameters
        ----------
        query_vec:
            L2-normalized query embedding, shape (D,).
        anchor_scores:
            Full-atlas scored anchors (from FullAtlasScorer).
        anchor_features:
            Optional per-anchor feature matrix, shape (N, anchor_feature_dim).

        Returns
        -------
        List of SelectedAnchor sorted by selector_score descending.
        Always returns at least 1 anchor.
        """
        if not anchor_scores:
            return []

        n = len(anchor_scores)

        # Build tensors (batch=1)
        ce_arr = np.array([a.ce_score for a in anchor_scores], dtype=np.float32)
        centroid_arr = np.stack([a.centroid for a in anchor_scores], axis=0).astype(np.float32)
        qvec_arr = np.asarray(query_vec, dtype=np.float32)

        ce_t = torch.from_numpy(ce_arr).unsqueeze(0).to(self.device)  # (1, N)
        centroids_t = torch.from_numpy(centroid_arr).unsqueeze(0).to(self.device)  # (1, N, D)
        qvec_t = torch.from_numpy(qvec_arr).unsqueeze(0).to(self.device)  # (1, D)

        af_t: Optional[torch.Tensor] = None
        if anchor_features is not None:
            af_t = (
                torch.from_numpy(np.asarray(anchor_features, dtype=np.float32))
                .unsqueeze(0)
                .to(self.device)
            )  # (1, N, anchor_feature_dim)

        with torch.no_grad():
            refined_scores, thresholds = self.model(ce_t, centroids_t, qvec_t, af_t)

        scores_np = refined_scores[0].cpu().numpy()  # (N,)
        threshold = float(thresholds[0].cpu().item())

        # Apply adaptive threshold — always keep at least 1
        selected_indices = [i for i in range(n) if scores_np[i] >= threshold]
        if not selected_indices:
            # Fallback: return best by refined score
            selected_indices = [int(np.argmax(scores_np))]

        # Build SelectedAnchor list sorted by score desc
        result = [
            SelectedAnchor(
                anchor_id=anchor_scores[i].anchor_id,
                selector_score=float(scores_np[i]),
                mode="coactivation",
                candidate=None,
            )
            for i in selected_indices
        ]
        result.sort(key=lambda sa: sa.selector_score, reverse=True)
        return result

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        device: str = "cpu",
    ) -> "CoActivationSelector":
        """
        Load a CoActivationModel from disk.

        Expects:
          <model_path>/coactivation_model.pt       -- model state_dict
          <model_path>/coactivation_version.json   -- hyperparameters

        Parameters
        ----------
        model_path:
            Directory containing the saved model artifacts.
        device:
            Torch device string, e.g. "cpu" or "cuda".
        """
        version_path = os.path.join(model_path, "coactivation_version.json")
        weights_path = os.path.join(model_path, "coactivation_model.pt")

        with open(version_path) as f:
            cfg = json.load(f)

        model = CoActivationModel(
            n_anchors=cfg.get("n_anchors", 256),
            centroid_dim=cfg.get("centroid_dim", 768),
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 512),
            dropout=cfg.get("dropout", 0.1),
            anchor_feature_dim=cfg.get("anchor_feature_dim", 8),
        )
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        return cls(model=model, device=device)
