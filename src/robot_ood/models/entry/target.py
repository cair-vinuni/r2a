"""The corpus's own episode-start configuration, read off a built graph's node table.

Design writeup: docs/method-runtime.md.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np

from robot_ood.core.constants import STILL_TOL

_NODES_FILENAME = "g_nodes.npz"


def _ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrink a sample covariance toward a scaled identity using Ledoit-Wolf closed-form estimate.

    Args:
        x: [N, D] centred samples.

    Returns:
        (shrunk_cov [D, D], shrinkage in [0, 1]).
    """
    n, d = x.shape
    emp_cov = (x.T @ x) / n
    mu = np.trace(emp_cov) / d
    target = mu * np.eye(d)
    delta = float(np.sum((emp_cov - target) ** 2)) / d
    x2 = x ** 2
    beta_raw = float(np.sum((x2.T @ x2) / n - emp_cov ** 2)) / (d * n)
    beta = min(max(beta_raw, 0.0), delta)
    shrinkage = 0.0 if delta <= 0.0 else beta / delta
    shrunk_cov = (1.0 - shrinkage) * emp_cov + shrinkage * target
    return shrunk_cov, float(shrinkage)


@dataclasses.dataclass(frozen=True)
class EntryTarget:
    """Where the arm should be standing when the frozen policy takes its first step.

    Attributes:
        q_entry: [D] float64 barycentre of every real demo's t=0 raw frame.
        loo: {"p50", "p90", "p95", "p99", "max"} leave-one-demo-out Mahalanobis distance from each
            demo's own start to the barycentre of all the others. The spread of the support this
            target sits in the middle of, and therefore the scale of every radius compared against
            it.
        n_demos: Demos the barycentre was taken over.
        precision: [D, D] float64 inverse of the shrunk covariance, floored at the corpus's still
            tolerance squared on the diagonal -- the metric error() applies.
        shrinkage: Ledoit-Wolf shrinkage intensity in [0, 1], estimated from this same corpus.
    """

    q_entry: np.ndarray
    loo: dict
    n_demos: int
    precision: np.ndarray
    shrinkage: float

    @classmethod
    def from_graph_dir(cls, graph_dir: str | os.PathLike) -> "EntryTarget":
        """Build from the g_nodes.npz beside a graph.

        Args:
            graph_dir: The artifact directory holding g_nodes.npz.

        Returns:
            The entry target.

        Raises:
            FileNotFoundError: No g_nodes.npz in graph_dir.
        """
        path = os.path.join(os.fspath(graph_dir), _NODES_FILENAME)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"EntryTarget: {path} not found -- build the graph first "
                f"(python -m robot_ood.cli build graph --suite <suite>)"
            )
        with np.load(path) as nodes:
            return cls.from_raw(np.asarray(nodes["q_raw"], np.float64),
                                np.asarray(nodes["raw_ptr"], np.int64))

    @classmethod
    def from_raw(cls, q_raw: np.ndarray, raw_ptr: np.ndarray) -> "EntryTarget":
        """Build from a node table's raw-frame columns (each demo's first raw frame).

        Args:
            q_raw: [N, D] float64 raw joint frames, demos concatenated.
            raw_ptr: [J+1] int64 start offset of each demo into q_raw.

        Returns:
            The entry target.

        Raises:
            ValueError: Fewer than two demos, so no leave-one-demo-out spread exists.
        """
        raw_ptr = np.asarray(raw_ptr, np.int64)
        n_demos = len(raw_ptr) - 1
        if n_demos < 2:
            raise ValueError("EntryTarget: need >= 2 demos for a leave-one-demo-out spread")
        starts = np.asarray(q_raw, np.float64)[raw_ptr[:-1]]
        total = starts.sum(0)
        q_entry = total / float(n_demos)
        shrunk_cov, shrinkage = _ledoit_wolf(starts - q_entry)
        # Demos that all start on one configuration have no spread to invert; a displacement
        # below the still tolerance is one the corpus itself does not resolve.
        precision = np.linalg.inv(shrunk_cov + STILL_TOL ** 2 * np.eye(shrunk_cov.shape[0]))
        # Leave-one-demo-out in closed form: the barycentre without demo j is (total - s_j)/(J-1).
        others = (total[None, :] - starts) / float(n_demos - 1)
        diff = others - starts
        floors = np.sqrt(np.einsum("ij,jk,ik->i", diff, precision, diff))
        loo = {f"p{p}": float(np.percentile(floors, p)) for p in (50, 90, 95, 99)}
        loo["max"] = float(floors.max())
        return cls(q_entry=q_entry, loo=loo, n_demos=n_demos, precision=precision,
                   shrinkage=shrinkage)

    def error(self, q_now: np.ndarray) -> float:
        """Mahalanobis distance from q_now to the entry configuration, under precision."""
        diff = np.asarray(q_now, np.float64) - self.q_entry
        return float(np.sqrt(diff @ self.precision @ diff))
