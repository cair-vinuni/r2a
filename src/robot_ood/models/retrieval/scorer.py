"""ActionEncoder, the SEG_FEATURES builder, and SegmentScorer.

Continuation compatibility: which segment should I execute next? See docs/method-runtime.md."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from robot_ood.core.schemas import (
    ACTION_DIM,
    ACTION_EMBED_DIM,
    SEG_FEATURE_DIM,
    SEG_FEATURES,
)
from robot_ood.data.metrics import ee_pose_delta_norm
from robot_ood.models.base import CheckpointedModule, i64

__all__ = ["STAT_CLAMP", "ActionEncoder", "SegmentScorer", "segment_features"]

# Clamp at this many sigma: rollouts visit (d_state, d_future) outside training corpus.
STAT_CLAMP: Final[float] = 4.0
_STD_FLOOR: Final[float] = 1e-6
_EPS: Final[float] = 1e-12

# Column stacking order: must match SEG_FEATURES or assertion fails at import time.
_STACK_ORDER: Final[tuple[str, ...]] = ("d_state", "d_future", "phase_diff", "task_affinity")
assert SEG_FEATURES == _STACK_ORDER, (
    f"robot_ood.core.schemas.SEG_FEATURES is {SEG_FEATURES} but robot_ood.models.retrieval.scorer.segment_features "
    f"still stacks columns in {_STACK_ORDER} order -- update the torch.stack call in "
    f"segment_features to match before touching anything downstream."
)


class ActionEncoder(nn.Module):
    """Encodes a fixed-length action chunk into a direction-comparable embedding.

    Args:
        chunk_len: T, the fixed number of action-chunk rows this encoder reads. A chunk of a
            different length must be resampled to this one before it is encoded.
        hidden: Width of the single hidden layer.
        out_dim: Width of the output embedding.
    """

    def __init__(self, chunk_len: int, hidden: int = 128, out_dim: int = ACTION_EMBED_DIM) -> None:
        super().__init__()
        self.chunk_len = int(chunk_len)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.chunk_len * ACTION_DIM, int(hidden)), nn.ReLU(),
            nn.Linear(int(hidden), self.out_dim),
        )

    def forward(self, chunk: Tensor) -> Tensor:
        """Encode one batch of chunks.

        Args:
            chunk: [..., T, ACTION_DIM], T == self.chunk_len.

        Returns:
            [..., out_dim] embedding.

        Raises:
            ValueError: chunk's last two dims do not match (chunk_len, ACTION_DIM).
        """
        if chunk.shape[-2:] != (self.chunk_len, ACTION_DIM):
            raise ValueError(
                f"ActionEncoder built for chunks of shape [.., {self.chunk_len}, {ACTION_DIM}], "
                f"got {tuple(chunk.shape[-2:])}"
            )
        return self.net(chunk.reshape(*chunk.shape[:-2], self.chunk_len * ACTION_DIM))


def _cos(a: Tensor, b: Tensor) -> Tensor:
    """cos(a, b) along the last axis, floored so a zero row stays finite rather than NaN."""
    return (F.normalize(a, dim=-1, eps=_EPS) * F.normalize(b, dim=-1, eps=_EPS)).sum(-1)


def segment_features(
    *,
    ee_query: Tensor,
    e_future_q: Tensor,
    phase_query: Tensor,
    seg_ee_start: Tensor,
    e_future_seg: Tensor,
    seg_phase: Tensor,
    cand_task_p: Tensor,
) -> Tensor:
    """Build the SEG_FEATURES vector for every (query, candidate) pair.

    Args:
        ee_query: [B, 6] current end-effector pose, base frame.
        e_future_q: [B, ACTION_EMBED_DIM] ActionEncoder applied to the query's predicted future.
        phase_query: [B] the query's phase.
        seg_ee_start: [B, K, 6] each candidate segment's start pose.
        e_future_seg: [B, K, ACTION_EMBED_DIM] ActionEncoder applied to each segment's action chunk.
        seg_phase: [B, K] each candidate's phase_start.
        cand_task_p: [B, K] the locator belief's mass on each candidate's task.

    Returns:
        [B, K, SEG_FEATURE_DIM] features, columns in SEG_FEATURES order.
    """
    d_state = ee_pose_delta_norm(ee_query[:, None, :], seg_ee_start)                  # [B, K]
    d_future = 1.0 - _cos(e_future_q[:, None, :], e_future_seg)
    phase_diff = phase_query[:, None] - seg_phase                                     # [B, K]

    # Continuous, not an identity test: the belief spreads mass over the tasks it cannot separate,
    # so a corpus thin in this task still offers its neighbours something to say.
    affinity = cand_task_p.to(d_state.dtype)

    return torch.stack([d_state, d_future, phase_diff, affinity], dim=-1)


class SegmentScorer(CheckpointedModule, nn.Module):
    """MLP over SEG_FEATURES: compatibility logits for segment candidates.

    Args:
        hidden: Width of the single hidden layer.
    """

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.trunk = nn.Sequential(nn.Linear(SEG_FEATURE_DIM, self.hidden), nn.ReLU())
        self.head = nn.Linear(self.hidden, 1)
        self.register_buffer("feat_mean", torch.zeros(SEG_FEATURE_DIM))
        self.register_buffer("feat_std", torch.ones(SEG_FEATURE_DIM))

    def _standardise(self, feats: Tensor) -> Tensor:
        standardised = (feats - self.feat_mean) / self.feat_std
        return standardised.clamp(-STAT_CLAMP, STAT_CLAMP)

    def forward(self, feats: Tensor) -> Tensor:
        """Score every candidate.

        Args:
            feats: [B, K, SEG_FEATURE_DIM], columns in SEG_FEATURES order.

        Returns:
            [B, K] compatibility logits, higher is more compatible.
        """
        return self.head(self.trunk(self._standardise(feats))).squeeze(-1)

    @torch.no_grad()
    def fit_feature_stats(self, feats: Tensor) -> "SegmentScorer":
        """Measure the standardisation statistics off a training sample.

        Args:
            feats: [N, SEG_FEATURE_DIM] or [..., SEG_FEATURE_DIM] features over the training set.

        Returns:
            self, so the fit can be chained onto construction.

        Raises:
            ValueError: Fewer than two rows, which cannot give a standard deviation.
        """
        flat = feats.reshape(-1, SEG_FEATURE_DIM)
        if flat.shape[0] < 2:
            raise ValueError(f"need >= 2 rows to fit statistics, got {flat.shape[0]}")
        self.feat_mean.copy_(flat.mean(dim=0))
        self.feat_std.copy_(flat.std(dim=0).clamp_min(_STD_FLOOR))
        return self

    # ---- persistence -------------------------------------------------------------------------
    def checkpoint_meta(self) -> dict[str, np.int64]:
        """The hidden width, plus the feature width the fit assumed."""
        # feature_dim is stamped because the state dict shapes alone cannot name a SEG_FEATURES
        # change; from_checkpoint reads it back to say so.
        return {"hidden": i64(self.hidden), "feature_dim": i64(SEG_FEATURE_DIM)}

    @classmethod
    def from_checkpoint(cls, meta: Mapping[str, Any], src: Path) -> "SegmentScorer":
        """Rebuild the module an archive describes.

        Args:
            meta: The archive; read for hidden and feature_dim.
            src: Where it came from, for the error message.

        Returns:
            The module, before its weights are loaded.

        Raises:
            ValueError: The artifact was fitted on a different SEG_FEATURE_DIM than this build
                assembles -- reported by name with the retrain command, rather than dying inside
                load_state_dict on a bare shape mismatch.
        """
        feature_dim = int(meta["feature_dim"]) if "feature_dim" in meta else -1
        if feature_dim != SEG_FEATURE_DIM:
            raise ValueError(
                f"{src}: this scorer was fitted on a {feature_dim}-wide SEG_FEATURES vector and "
                f"this build assembles a {SEG_FEATURE_DIM}-wide one (robot_ood.core.schemas.SEG_FEATURES "
                f"changed). Its first layer's columns do not line up with the channels behind them. "
                f"Retrain with `python -m robot_ood.cli train fcsr --suite long`."
            )
        return cls(hidden=int(meta["hidden"]))
