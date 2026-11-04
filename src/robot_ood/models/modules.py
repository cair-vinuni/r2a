"""Standalone building blocks of robot_ood.models.retrieval.gnn.GraphRetrieverNet: sinusoidal phase
encoding, query-window pooling, and query-gated relation embeddings.

Both modules are rank-agnostic -- one forward serves the single-query deploy path and the
batched training path, because every reduction is addressed from the end of the shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from robot_ood.data import schema

__all__ = ["QueryEncoder", "QueryEncoding", "RelationGate", "sinusoidal_psi"]

REL_INIT_STD = 0.02
POOL_INIT_STD = 0.02


def sinusoidal_psi(t_frac: Tensor, n_psi: int) -> Tensor:
    """Fixed (non-learned) sin/cos encoding of a phase fraction.

    Args:
        t_frac: Phase fractions in [0, 1], any shape [...].
        n_psi: Number of sin/cos octaves.

    Returns:
        Encoding of shape [..., 2*n_psi], dtype and device of t_frac.
    """
    freqs = 2.0 ** torch.arange(n_psi, dtype=t_frac.dtype, device=t_frac.device)   # [n_psi]
    ang = torch.pi * t_frac[..., None] * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)                    # [..., 2*n_psi]


@dataclass(frozen=True, eq=False)
class QueryEncoding:
    """One window's encoding, in the three pieces the pipeline reads separately.

    Attributes:
        e_q: Pooled query embedding, [hidden] or [B, hidden].
        z: Per-step states BEFORE pooling, [H, hidden] or [B, H, hidden] -- the learned seed's
            tier-1 keys, which must stay per row because its mixture is per row.
        pool_w: Pooling weights over the window, [H] or [B, H], summing to 1. The learned seed
            reads them too, so they weight WHICH nodes get seeded, not just e_q.
    """

    e_q: Tensor
    z: Tensor
    pool_w: Tensor


class QueryEncoder(nn.Module):
    """Pool a window of recent joint states into one query embedding e_Q.

    Args:
        dim: Joint-space dimensionality.
        hidden: Width of the hidden and output embeddings.
        frame_dropout: Training-time probability of masking a frame out of the pooling softmax.
    """

    def __init__(self, dim: int, hidden: int = schema.HIDDEN, *,
                 frame_dropout: float = schema.FRAME_DROPOUT):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.frame_dropout = float(frame_dropout)
        in_dim = 2 * self.dim + 1                                # per-step [q, qdot, grip]
        self.step = nn.Sequential(
            nn.Linear(in_dim, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.hidden),
        )
        # Three views per frame -- itself, its window's consensus, its offset from the latest frame
        # -- so a frame that disagrees with its own window can lose weight.
        self.pool = nn.Sequential(
            nn.Linear(3 * self.hidden, self.hidden), nn.ReLU(),
            # Bias-free: a constant on every logit cancels in the softmax.
            nn.Linear(self.hidden, 1, bias=False),
        )
        nn.init.normal_(self.pool[-1].weight, std=POOL_INIT_STD)
        self.out = nn.Linear(self.hidden, self.hidden)

    def _pool_logits(self, z: Tensor) -> Tensor:
        """Per-frame pooling logits, [..., H], from the per-frame states alone.

        Args:
            z: Per-step states, [..., H, hidden].

        Returns:
            Unnormalised logits over the window.
        """
        consensus = z.mean(-2, keepdim=True).expand_as(z)
        feat = torch.cat([z, consensus, z - z[..., -1:, :]], dim=-1)
        return self.pool(feat).squeeze(-1)

    def _drop_frames(self, logits: Tensor) -> Tensor:
        """Mask frames out of the pooling softmax while training.

        Args:
            logits: Pooling logits, [..., H].

        Returns:
            The logits with dropped frames set to -inf, or the input itself when the module is in
            eval mode. At least one frame always survives, so the softmax cannot be all -inf.
        """
        if not self.training or self.frame_dropout <= 0.0:
            return logits
        drop = torch.rand_like(logits) < self.frame_dropout
        drop[..., torch.randint(logits.shape[-1], (1,), device=logits.device)] = False
        return logits.masked_fill(drop, float("-inf"))

    def encode(self, qh: Tensor, vh: Tensor, gh: Tensor) -> QueryEncoding:
        """Encode and pool one window, batched or unbatched, keeping every piece.

        Args:
            qh: Joint positions, [H, dim] or [B, H, dim].
            vh: Joint velocities, same shape as qh.
            gh: Gripper states, [H] or [B, H].

        Returns:
            The encoding: e_Q, the per-step states and the pooling weights.
        """
        x = torch.cat([qh, vh, gh[..., None]], dim=-1)                     # [..., H, 2*dim+1]
        z = self.step(x)
        pool_w = F.softmax(self._drop_frames(self._pool_logits(z)), dim=-1)
        return QueryEncoding(
            e_q=self.out((pool_w[..., None] * z).sum(-2)), z=z, pool_w=pool_w,
        )

    def forward(self, qh: Tensor, vh: Tensor, gh: Tensor) -> Tensor:
        """Encode one window down to e_Q alone, batched or unbatched.

        Args:
            qh: Joint positions, [H, dim] or [B, H, dim].
            vh: Joint velocities, same shape as qh.
            gh: Gripper states, [H] or [B, H].

        Returns:
            Query embedding e_Q, [hidden] or [B, hidden].
        """
        return self.encode(qh, vh, gh).e_q


class RelationGate(nn.Module):
    """Query-gated relation embeddings: g_rel = MLP([h_rel ; e_Q]).

    Args:
        hidden: Embedding width.
        n_rel: Number of relation types.
    """

    def __init__(self, hidden: int, n_rel: int):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(n_rel, hidden) * REL_INIT_STD)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    # gnn.py recomputes this at every propagation layer rather than caching one table: a cached
    # table is no longer a function of e_q, and the gate silently stops being query-dependent.
    def forward(self, e_q: Tensor) -> Tensor:
        """Gate the relation table by a query embedding, batched or unbatched.

        Args:
            e_q: Query embedding, [hidden] or [B, hidden].

        Returns:
            Gated relation embeddings, [n_rel, hidden] or [B, n_rel, hidden].
        """
        target = (*e_q.shape[:-1], *self.emb.shape)                        # [..., R, hidden]
        emb = self.emb.expand(target)
        # view, not copy: each batch row's own e_q keeps the gate query-dependent per row.
        q = e_q[..., None, :].expand(target)
        return self.mlp(torch.cat([emb, q], dim=-1))
