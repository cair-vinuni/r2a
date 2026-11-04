"""C3's query-side encoders: the query window, its phase code, and the tier-1 segment key."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from robot_ood.core.schemas import ACTION_DIM, ACTION_EMBED_DIM, SEGMENT_LEN, QueryContext
from robot_ood.core.env import env_bool

# ROBOT_OOD_KEY_FUTURE=0 drops the future embedding from the tier-1 segment key, so the key is a
# function of state (start pose, phase) alone. With the future in the key, training matches a query
# to its OWN segment through the future the query encoder predicts from the same demo, and the
# embedding never becomes smooth in pose: a deploy query, whose own demo is not in the table, gets
# a shortlist that is random in pose (results_dbg/audit_tier1.py). The tier-2 scorer still reads
# the future features.
KEY_FUTURE: bool = env_bool("ROBOT_OOD_KEY_FUTURE", True)
from robot_ood.data import schema

__all__ = ["QueryEncoder", "SegmentKeyEncoder", "observed"]


def observed(value: Tensor | None, name: str) -> Tensor:
    """Return an observed QueryContext field, or say which one the caller forgot.

    Args:
        value: The field.
        name: Its attribute name, for the message.

    Returns:
        The field unchanged.

    Raises:
        ValueError: The field is None. Deliberately NOT a learned fallback: ee follows from q by
            forward kinematics and phase comes from the belief, so estimating either would put this
            module's own error into the two columns that carry the same-pose/different-continuation
            distinction.
    """
    if value is None:
        raise ValueError(
            f"QueryContext.{name} is None, but segment scoring needs it: {name} is OBSERVED "
            "(ee by forward kinematics from q_now, phase from the belief filter), not predicted. "
            "Build the QueryContext with it."
        )
    return value


# ==== QueryEncoder ====

DEFAULT_QUERY_WORD_DROPOUT = 0.2       # robot_ood.core.constants key: seg_query_word_dropout.
DEFAULT_QUERY_PHASE_GATE_BIAS = 2.0    # robot_ood.core.constants key: seg_query_phase_gate_bias.


def _query_calib(name: str, default: float) -> float:
    """A C0-calibrated QueryEncoder magnitude, or default when no calibration is loaded."""
    from robot_ood.core.config import calibrated_constants, warn_uncalibrated

    constants = calibrated_constants()
    if constants is None or name not in constants:
        warn_uncalibrated(name, default)
        return default
    return float(constants.value(name, default))


class _ActionAbsentEmbedding(nn.Module):
    """Wraps the "no action stream" vector so it participates in state_dict like any other tensor."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.zeros(hidden))


class _CyclicPhaseEncoding(nn.Module):
    """sin/cos(2*pi*phase) -> hidden: continuous across the p=0/p=1 wrap, unlike phase itself."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2, hidden)

    def forward(self, phase: Tensor) -> Tensor:
        angle = 2.0 * torch.pi * phase
        return self.proj(torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1))


class QueryEncoder(nn.Module):
    """Encodes a QueryContext into a future-conditioned, phase-aware query embedding.

    Args:
        dim: Joint-space dimensionality of qh/vh.
        hist: Window length H.
        hidden: Width of every embedding this module produces.
        future_len: T, rows in the predicted future chunk; must match the corpus's segment length.
        lang_dim: Width of e_lang, or 0 to disable the language token entirely (no parameters).
        frame_dropout: Training-time probability of masking a frame out of the pooling softmax.
    """

    def __init__(self, dim: int, hist: int, hidden: int, *, future_len: int = SEGMENT_LEN,
                 lang_dim: int = 0,
                 frame_dropout: float = schema.FRAME_DROPOUT) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hist = int(hist)
        self.hidden = int(hidden)
        self.future_len = int(future_len)
        self.lang_dim = int(lang_dim)
        self.word_dropout = _query_calib("seg_query_word_dropout", DEFAULT_QUERY_WORD_DROPOUT)
        self.frame_dropout = float(frame_dropout)

        frame_in = 2 * self.dim + 1                                   # q, qdot, grip
        self.frame_proj = nn.Sequential(
            nn.Linear(frame_in, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.hidden),
        )
        self.act_proj = nn.Sequential(
            nn.Linear(ACTION_DIM, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.hidden),
        )
        self.action_absent = _ActionAbsentEmbedding(self.hidden)
        self.pos_embed = nn.Parameter(torch.zeros(self.hist, self.hidden))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.phase_embed = _CyclicPhaseEncoding(self.hidden)
        self.lang_proj = nn.Linear(self.lang_dim, self.hidden) if self.lang_dim > 0 else None

        n_head = 8 if self.hidden % 8 == 0 else 1
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden, nhead=n_head, dim_feedforward=4 * self.hidden,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)

        self.pool_query = nn.Parameter(torch.zeros(self.hidden))
        nn.init.normal_(self.pool_query, std=0.02)

        self.phase_gate = nn.Linear(self.hidden, self.hidden)
        nn.init.zeros_(self.phase_gate.weight)
        nn.init.constant_(self.phase_gate.bias, _query_calib("seg_query_phase_gate_bias", DEFAULT_QUERY_PHASE_GATE_BIAS))

        self.future_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.future_len * ACTION_DIM),
        )

    def _action_embed(self, ctx: QueryContext) -> Tensor:
        """Per-frame action tokens, [B, H, hidden].

        Args:
            ctx: The query context.

        Returns:
            [B, H, hidden] -- the action_absent vector broadcast to every frame when a_hist is None
            (batch- and frame-invariant), else act_proj applied per step.
        """
        if ctx.a_hist is None:
            batch_shape = ctx.qh.shape[:-2]
            return self.action_absent.value.expand(*batch_shape, self.hist, self.hidden)
        return self.act_proj(ctx.a_hist)

    def _frame_tokens(self, ctx: QueryContext) -> Tensor:
        feat = torch.cat([ctx.qh, ctx.vh, ctx.gh[..., None]], dim=-1)
        return self.frame_proj(feat) + self._action_embed(ctx) + self.pos_embed

    def _drop_frames(self, logits: Tensor) -> Tensor:
        """Mask frames out of the pooling softmax while training.

        Args:
            logits: Pooling logits, [B, H].

        Returns:
            The logits with dropped frames set to -inf, or the input in eval mode. One frame is
            always kept, so no row can soft-max over an all -inf window.
        """
        if not self.training or self.frame_dropout <= 0.0:
            return logits
        drop = torch.rand_like(logits) < self.frame_dropout
        drop[..., torch.randint(logits.shape[-1], (1,), device=logits.device)] = False
        return logits.masked_fill(drop, float("-inf"))

    def _fuse(self, ctx: QueryContext) -> Tensor:
        phase = observed(ctx.phase, "phase")
        tokens = [self._frame_tokens(ctx), self.phase_embed(phase)[..., None, :]]
        if self.lang_proj is not None and ctx.e_lang is not None:
            e_lang = ctx.e_lang
            if self.training and self.word_dropout > 0.0:
                keep = (torch.rand(e_lang.shape[:-1], device=e_lang.device) >= self.word_dropout)
                e_lang = e_lang * keep[..., None].to(e_lang.dtype)
            tokens.append(self.lang_proj(e_lang)[..., None, :])
        seq = self.encoder(torch.cat(tokens, dim=-2))              # [B, H+1(+1), hidden]

        frame_out, phase_out = seq[..., :self.hist, :], seq[..., self.hist, :]
        attn = torch.softmax(self._drop_frames(frame_out @ self.pool_query), dim=-1)
        return (attn[..., None] * frame_out).sum(-2) * torch.sigmoid(self.phase_gate(phase_out))

    def forward(self, ctx: QueryContext) -> Tensor:
        """The fused query embedding.

        Args:
            ctx: The query context.

        Returns:
            [B, hidden].
        """
        return self._fuse(ctx)

    def predict_future(self, ctx: QueryContext) -> Tensor:
        """The predicted future action chunk, for ActionEncoder/segment_features' d_future term.

        Args:
            ctx: The query context.

        Returns:
            [B, future_len, ACTION_DIM].
        """
        flat = self.future_head(self._fuse(ctx))
        return flat.reshape(*flat.shape[:-1], self.future_len, ACTION_DIM)


class SegmentKeyEncoder(nn.Module):
    """Tier-1 segment key encoder: segment-side features to query embedding space.

    Args:
        hidden: Output width, matching QueryEncoder's.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        in_dim = 6 + 1 + ACTION_EMBED_DIM      # ee_start, phase_start, e_future_seg
        self.mlp = nn.Sequential(nn.Linear(in_dim, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.hidden))

    def forward(self, ee_start: Tensor, phase_start: Tensor, e_future: Tensor) -> Tensor:
        if not KEY_FUTURE:
            e_future = torch.zeros_like(e_future)
        return self.mlp(torch.cat([ee_start, phase_start[..., None], e_future], dim=-1))
