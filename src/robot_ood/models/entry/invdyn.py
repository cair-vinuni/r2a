"""A learned inverse-dynamics map: desired joint-space displacement -> the policy-space action.

Fit on demonstration (q_t, q_t+1, a_t) triples with no rollout or episode outcome supervision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from robot_ood.models.base import CheckpointedModule, i64

__all__ = ["INVDYN_NPZ", "N_ACTION", "N_JOINT", "InverseDynamicsNet"]

INVDYN_NPZ: Final[str] = "g_invdyn.npz"
N_JOINT: Final[int] = 7
N_ACTION: Final[int] = 7
_HIDDEN: Final[int] = 64
_STD_FLOOR: Final[float] = 1e-6


class InverseDynamicsNet(CheckpointedModule, nn.Module):
    """(q_t, delta_q) -> a_t. A two-layer MLP fit by MSE regression on demo triples, clipped to observed action range.

    Attributes:
        hidden: Width of the two hidden layers.
    """

    NPZ = INVDYN_NPZ
    MISSING = ("no trained inverse-dynamics map. Train one with `OMP_NUM_THREADS=4 "
               "python -m robot_ood.cli train entry --graph-dir ...`.")

    def __init__(self, hidden: int = _HIDDEN) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.trunk = nn.Sequential(
            nn.Linear(N_JOINT + N_JOINT, self.hidden), nn.SiLU(),
            nn.Linear(self.hidden, self.hidden), nn.SiLU(),
        )
        self.head = nn.Linear(self.hidden, N_ACTION)
        self.register_buffer("q_mean", torch.zeros(N_JOINT))
        self.register_buffer("q_std", torch.ones(N_JOINT))
        self.register_buffer("dq_mean", torch.zeros(N_JOINT))
        self.register_buffer("dq_std", torch.ones(N_JOINT))
        self.register_buffer("a_lo", -torch.ones(N_ACTION))
        self.register_buffer("a_hi", torch.ones(N_ACTION))

    @torch.no_grad()
    def fit_stats(self, q: Tensor, dq: Tensor, a: Tensor) -> "InverseDynamicsNet":
        """Measure standardisation statistics and the output clip range off a training sample.

        Args:
            q: [N, N_JOINT].
            dq: [N, N_JOINT].
            a: [N, N_ACTION], the demo corpus's own observed action range.

        Returns:
            self, chainable onto construction.

        Raises:
            ValueError: Fewer than two rows.
        """
        if q.shape[0] < 2:
            raise ValueError(f"need >= 2 rows to fit statistics, got {q.shape[0]}")
        self.q_mean.copy_(q.mean(dim=0))
        self.q_std.copy_(q.std(dim=0).clamp_min(_STD_FLOOR))
        self.dq_mean.copy_(dq.mean(dim=0))
        self.dq_std.copy_(dq.std(dim=0).clamp_min(_STD_FLOOR))
        self.a_lo.copy_(a.min(dim=0).values)
        self.a_hi.copy_(a.max(dim=0).values)
        return self

    def forward(self, q: Tensor, dq: Tensor) -> Tensor:
        """Map (q, delta_q) to action.

        Args:
            q: [..., N_JOINT] current configuration.
            dq: [..., N_JOINT] desired displacement.

        Returns:
            [..., N_ACTION] action, clipped to the corpus's own observed range.
        """
        zq = (q - self.q_mean) / self.q_std
        zd = (dq - self.dq_mean) / self.dq_std
        raw = self.head(self.trunk(torch.cat([zq, zd], dim=-1)))
        return torch.clamp(raw, self.a_lo, self.a_hi)

    def checkpoint_meta(self) -> dict[str, np.int64]:
        """The trunk width, plus the joint and action widths the fit assumed."""
        return {"hidden": i64(self.hidden), "n_joint": i64(N_JOINT), "n_action": i64(N_ACTION)}

    @classmethod
    def from_checkpoint(cls, meta: Mapping[str, Any], src: Path) -> "InverseDynamicsNet":
        """Rebuild the module an archive describes.

        Args:
            meta: The archive; read for hidden, n_joint and n_action.
            src: Where it came from, for the error message.

        Returns:
            The module, before its weights are loaded.

        Raises:
            ValueError: The artifact's joint or action width does not match this build's.
        """
        n_joint = int(meta["n_joint"]) if "n_joint" in meta else N_JOINT
        n_action = int(meta["n_action"]) if "n_action" in meta else N_ACTION
        if n_joint != N_JOINT or n_action != N_ACTION:
            raise ValueError(
                f"{src}: trained against {n_joint} joints / {n_action} actions, this build "
                f"reads {N_JOINT} / {N_ACTION}. Retrain with `python -m robot_ood.cli train entry`."
            )
        return cls(hidden=int(meta["hidden"]))
