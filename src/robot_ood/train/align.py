"""Dynamic time warping between position segments, for the hard-negative miner.

Warping accounts for speed differences: two segments tracing the same path at different speeds
have large pointwise MSE but small warped distance. Positions only; rotation would be unscalable.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

__all__ = ["dtw_cost"]

_BIG: Final[float] = 1e9


def dtw_cost(a: Tensor, b: Tensor) -> Tensor:
    """Mean-per-step dynamic-time-warping distance between two batches of position segments.

    Args:
        a: [B, Ka, 3] positions, metres.
        b: [B, Kb, 3] positions, metres.

    Returns:
        [B] mean warped distance, metres. Path length is used as the normaliser, so the value is
        comparable across different K.

    Raises:
        ValueError: Either input is not [B, K, 3] with K >= 1, or the batch sizes disagree.
    """
    for name, tensor in (("a", a), ("b", b)):
        if tensor.dim() != 3 or tensor.shape[-1] != 3 or tensor.shape[1] < 1:
            raise ValueError(f"{name} must be [B, K, 3] with K >= 1, got {tuple(tensor.shape)}")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"batch mismatch: a has {a.shape[0]} rows, b has {b.shape[0]}")

    local = torch.cdist(a, b)                                  # [B, Ka, Kb]
    n_b, k_a, k_b = local.shape
    big = torch.full((n_b,), _BIG, device=local.device, dtype=local.dtype)
    # (accumulated cost, path length) per cell; only the previous row is ever read.
    prev_cost = [big] * k_b
    prev_len = [torch.zeros_like(big)] * k_b

    for i in range(k_a):
        row_cost: list[Tensor] = []
        row_len: list[Tensor] = []
        for j in range(k_b):
            if i == 0 and j == 0:
                best_cost, best_len = torch.zeros_like(big), torch.zeros_like(big)
            else:
                cands = [(prev_cost[j], prev_len[j])]                       # from above
                if j > 0:
                    cands.append((row_cost[j - 1], row_len[j - 1]))         # from the left
                    cands.append((prev_cost[j - 1], prev_len[j - 1]))       # diagonal
                stacked = torch.stack([c for c, _ in cands])
                lengths = torch.stack([n for _, n in cands])
                pick = stacked.argmin(dim=0)
                best_cost = stacked.gather(0, pick[None])[0]
                best_len = lengths.gather(0, pick[None])[0]
            row_cost.append(best_cost + local[:, i, j])
            row_len.append(best_len + 1.0)
        prev_cost, prev_len = row_cost, row_len

    return prev_cost[-1] / prev_len[-1].clamp_min(1.0)

