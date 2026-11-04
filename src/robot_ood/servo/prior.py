"""The servo's model of a nominal start: a per-task low-rank Gaussian over the policy's mid-layer features.

The policy's mid-layer image tokens separate a displaced robot from a nominal one far better than its
actions do. Offline, from nominal start frames only (no shift labels), the prior models those features
per task as a task mean plus a nuisance subspace V spanned by how nominal starts differ from each other
(object layout, lighting). The part of a feature outside mu + span(V), r = (I - V V^T)(z - mu), is what
the scene cannot explain and the robot's placement can; its squared norm is the energy.
"""

from __future__ import annotations

import pathlib
from typing import Final

import numpy as np

PRIOR_NPZ: Final[str] = "servo_prior.npz"
PRIOR_JSON: Final[str] = "servo_prior.json"


class ServoPrior:
    """Per-task low-rank Gaussian over nominal start features: mean mu_k and a shared nuisance basis V."""

    def __init__(self, mean: np.ndarray, basis: np.ndarray, scale: float, tasks: list[str], tau: float,
                 cols: tuple[int, int] | None = None, energy_std: float = float("nan")) -> None:
        self.mean, self.basis, self.scale = mean, basis, float(scale)
        self.tasks, self.tau = list(tasks), float(tau)
        self.cols = None if cols is None else (int(cols[0]), int(cols[1]))
        # Spread of held-out nominal energies: how much E varies from one nominal scene to the next.
        self.energy_std = float(energy_std)

    @staticmethod
    def _moments(feats: np.ndarray, task_ids: np.ndarray, n_tasks: int, rank: int, scale: float | None = None):
        mean = np.stack([feats[task_ids == k].mean(0) for k in range(n_tasks)])
        centered = feats - mean[task_ids]
        scale = float(np.sqrt((centered ** 2).sum(1).mean())) if scale is None else scale
        _, _, vt = np.linalg.svd(centered / scale, full_matrices=False)
        return mean, vt[:rank], scale

    @classmethod
    def fit(cls, feats: np.ndarray, task_ids: np.ndarray, tasks: list[str], *, rank: int = 32,
            cols: tuple[int, int] | None = None, folds: np.ndarray | None = None,
            alpha: float = 0.05) -> "ServoPrior":
        """Fit on the nominal bank; tau is the (1 - alpha) conformal quantile of energies held out by fold."""
        feats = np.asarray(feats if cols is None else feats[:, cols[0]:cols[1]], np.float32)
        task_ids = np.asarray(task_ids)
        mean, basis, scale = cls._moments(feats, task_ids, len(tasks), rank)
        tau, energy_std = np.inf, float("nan")
        if folds is not None:
            scores = np.zeros(len(feats))
            for f in np.unique(folds):
                tr = folds != f
                m, b, _ = cls._moments(feats[tr], task_ids[tr], len(tasks), rank, scale)
                held = cls(m, b, scale, tasks, np.inf)
                scores[~tr] = [held.energy(feats[i:i + 1], int(task_ids[i]), raw=True)[0] for i in np.where(~tr)[0]]
            n = len(scores)
            tau = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))
            energy_std = float(np.std(scores))
        return cls(mean, basis, scale, tasks, tau, cols, energy_std)

    def residual(self, feats: np.ndarray, task: int, raw: bool = False) -> np.ndarray:
        """[B, F] off-nuisance residual (I - V V^T)(z - mu_k) / scale."""
        feats = np.atleast_2d(np.asarray(feats, np.float32))
        if self.cols is not None and not raw:
            feats = feats[:, self.cols[0]:self.cols[1]]
        x = (feats - self.mean[task]) / self.scale
        return x - (x @ self.basis.T) @ self.basis

    def energy(self, feats: np.ndarray, task: int, raw: bool = False) -> np.ndarray:
        return (self.residual(feats, task, raw) ** 2).sum(-1)

    def save(self, path: "str | pathlib.Path") -> None:
        np.savez(path, mean=self.mean.astype(np.float32), basis=self.basis.astype(np.float32),
                 scale=self.scale, tasks=np.array(self.tasks), tau=self.tau,
                 cols=np.array(self.cols if self.cols is not None else (-1, -1)), energy_std=self.energy_std)

    @classmethod
    def load(cls, path: "str | pathlib.Path") -> "ServoPrior":
        d = np.load(path)
        cols = tuple(int(c) for c in d["cols"]) if d["cols"][0] >= 0 else None
        energy_std = float(d["energy_std"]) if "energy_std" in d.files else float("nan")
        return cls(d["mean"], d["basis"], float(d["scale"]), [str(x) for x in d["tasks"]], float(d["tau"]), cols,
                   energy_std)
