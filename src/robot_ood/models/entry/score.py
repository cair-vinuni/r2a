"""A learned score field over the demonstrated entry configuration; the correction vanishes on-manifold by construction.

Design writeup: docs/method-runtime.md.

    s(q, sigma) ~ grad_q log p_sigma(q), the ScoreNet output (denoising score matching).
    q_hat = q + sigma^2 * s(q, sigma), Tweedie's formula under the Gaussian noise model DSM trains under.
    delta_hat = |q - q_clean|, the level head's target, supervised on the free label the corruption draws.
    Anneal delta_hat / sqrt(D) -> sigma_lo, re-estimating q_hat at each level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from robot_ood.models.base import CheckpointedModule, i64

__all__ = [
    "SCORE_NPZ", "N_JOINT", "ScoreNet", "dsm_loss", "level_loss",
    "tweedie_denoise",
]

SCORE_NPZ: Final[str] = "g_score.npz"
N_JOINT: Final[int] = 7
_HIDDEN: Final[int] = 64
_STD_FLOOR: Final[float] = 1e-6


def tweedie_denoise(score: Tensor | np.ndarray, q: Tensor | np.ndarray,
                     sigma: float) -> Tensor | np.ndarray:
    """Return q + sigma^2 * score, the Tweedie posterior mean of the clean configuration.

    Args:
        score: [..., D] s(q, sigma).
        q: [..., D] the noisy (perturbed) configuration.
        sigma: The noise scale score was evaluated at.

    Returns:
        [..., D] denoised estimate, same type as the inputs.
    """
    # Exact under the Gaussian corruption model DSM trains against, not a tunable step size.
    return q + (sigma ** 2) * score


def dsm_loss(net: "ScoreNet", q_clean: Tensor, sigma: Tensor) -> Tensor:
    """Perturb q_clean by sigma and regress the noise kernel's own score (denoising score matching).

    Args:
        net: The score network, called as net(x, sigma).
        q_clean: [N, D] float32 clean configurations.
        sigma: [N] float32 per-row noise scale.

    Returns:
        Scalar loss.
    """
    eps = torch.randn_like(q_clean)
    x = q_clean + sigma[:, None] * eps
    pred = net(x, sigma)
    # Target score is -eps/sigma; the sigma^2 weight cancels one factor of sigma, leaving this.
    return ((sigma[:, None] * pred + eps) ** 2).mean()


# sigma is read from the data, not assumed; a single fixed sigma_hi pulls every start toward
# the barycentre.
def level_loss(net: "ScoreNet", q_clean: Tensor, sigma: Tensor) -> Tensor:
    """Regress how far the corruption moved each row, in log space, against the free displacement label.

    Args:
        net: The score network, whose level head is called as net.level_estimate(x).
        q_clean: [N, D] float32 clean configurations.
        sigma: [N] float32 per-row noise scale.

    Returns:
        Scalar loss.
    """
    eps = torch.randn_like(q_clean)
    x = q_clean + sigma[:, None] * eps
    delta = (sigma[:, None] * eps).norm(dim=-1)
    # Log space: displacements span decades log-uniformly, so a linear residual would fit
    # almost entirely to the widest (least consequential) rows.
    return ((net.level_estimate(x).log() - delta.clamp_min(_STD_FLOOR).log()) ** 2).mean()


class ScoreNet(CheckpointedModule, nn.Module):
    """(q, sigma) -> approximate grad_q log p_sigma(q); an MLP conditioned on log(sigma).

    Attributes:
        hidden: Width of the two hidden layers.
    """

    NPZ = SCORE_NPZ
    MISSING = ("no trained score field. Train one with `OMP_NUM_THREADS=4 PYTHONPATH=src "
               "python -m robot_ood.cli train entry --graph-dir ...`.")

    def __init__(self, hidden: int = _HIDDEN) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.trunk = nn.Sequential(
            nn.Linear(N_JOINT + 1, self.hidden), nn.SiLU(),
            nn.Linear(self.hidden, self.hidden), nn.SiLU(),
        )
        self.head = nn.Linear(self.hidden, N_JOINT)
        # Reads q alone, not log(sigma) (sigma is unknown at deploy); its own trunk, not a probe
        # on score's, so fitting it cannot perturb the score field (they share only the std buffers).
        self.level = nn.Sequential(
            nn.Linear(N_JOINT, self.hidden), nn.SiLU(),
            nn.Linear(self.hidden, self.hidden), nn.SiLU(),
            nn.Linear(self.hidden, 1),
        )
        # Standardises by the corpus's own mean/std, so descent does not have to first learn the
        # corpus's scale.
        self.register_buffer("q_mean", torch.zeros(N_JOINT))
        self.register_buffer("q_std", torch.ones(N_JOINT))

    @torch.no_grad()
    def fit_stats(self, q: Tensor) -> "ScoreNet":
        """Measure standardisation statistics off a training sample.

        Args:
            q: [N, N_JOINT].

        Returns:
            self, chainable onto construction.

        Raises:
            ValueError: Fewer than two rows.
        """
        if q.shape[0] < 2:
            raise ValueError(f"need >= 2 rows to fit statistics, got {q.shape[0]}")
        self.q_mean.copy_(q.mean(dim=0))
        self.q_std.copy_(q.std(dim=0).clamp_min(_STD_FLOOR))
        return self

    def forward(self, q: Tensor, sigma: Tensor) -> Tensor:
        """Predict s(q, sigma), the score in raw q-space units.

        Args:
            q: [..., N_JOINT] raw (unstandardised) configuration, possibly noisy.
            sigma: [...] or [..., 1] noise scale this q was drawn at.

        Returns:
            [..., N_JOINT] score, in raw q-space units (rad^-1), not standardised, so
            tweedie_denoise can add it to q directly.
        """
        sigma = sigma if sigma.dim() == q.dim() else sigma[..., None]
        z = (q - self.q_mean) / self.q_std
        feat = torch.cat([z, torch.log(sigma)], dim=-1)
        return self.head(self.trunk(feat))

    def level_estimate(self, q: Tensor) -> Tensor:
        """How far this configuration sits off the demonstrated entry manifold, in radians.

        Args:
            q: [..., N_JOINT] raw configuration.

        Returns:
            [...] displacement estimate, same units as q.
        """
        z = (q - self.q_mean) / self.q_std
        # Strictly positive by construction (exp of the head's output); no clamp needed to
        # keep the schedule it seeds well formed.
        return self.level(z).squeeze(-1).exp()

    def checkpoint_meta(self) -> dict[str, np.int64]:
        """The two constants a reader needs: the trunk width, and the joint width it was fit for."""
        return {"hidden": i64(self.hidden), "n_joint": i64(N_JOINT)}

    @classmethod
    def from_checkpoint(cls, meta: Mapping[str, Any], src: Path) -> "ScoreNet":
        """Rebuild the module an archive describes.

        Args:
            meta: The archive; read for hidden and n_joint.
            src: Where it came from, for the error message.

        Returns:
            The module, before its weights are loaded.

        Raises:
            ValueError: The artifact's joint width does not match this build's.
        """
        n_joint = int(meta["n_joint"]) if "n_joint" in meta else N_JOINT
        if n_joint != N_JOINT:
            raise ValueError(
                f"{src}: trained against {n_joint} joints, this build reads {N_JOINT}. "
                "Retrain with `python -m robot_ood.cli train entry`."
            )
        return cls(hidden=int(meta["hidden"]))

    @classmethod
    def check_state(cls, state: Mapping[str, Tensor], src: Path) -> None:
        """Refuse an artifact that predates the level head.

        Args:
            state: The archive's state half.
            src: Where it came from, for the error message.

        Raises:
            ValueError: The artifact has no level head.
        """
        if not any(k.startswith("level.") for k in state):
            # Explicit check: load_state_dict's own KeyError would not say a pre-level artifact
            # loads fine but cannot seed a schedule.
            raise ValueError(
                f"{src}: score field predates the level head, so it cannot seed a schedule. "
                "Retrain with `python -m robot_ood.cli train entry "
                f"--graph-dir {src.parent}`."
            )
