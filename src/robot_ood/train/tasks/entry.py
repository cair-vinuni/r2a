"""The two nets at t = 0: the entry score field and the inverse dynamics that servos along it.

Both fit on the demo corpus's own raw frames -- no rollout, no episode outcome, anywhere in this
file. They share a module because they share their data, their held-out split and their shape:
one net, one loss, no model selection.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import numpy as np
import torch

from robot_ood.core.config import entry_sigma_hi, entry_sigma_lo
from robot_ood.models.entry.invdyn import InverseDynamicsNet
from robot_ood.models.entry.score import ScoreNet, dsm_loss, level_loss
from robot_ood.data import schema
from robot_ood.train.task import Requirements, TrainContext, TrainUnit

__all__ = ["InvDynTask", "ScoreTask", "demo_frames", "held_out_owners", "triples"]

DEFAULT_EPOCHS = 6_000


def held_out_owners(n_demos: int) -> np.ndarray:
    """The demos neither net trains on -- robot_ood.train.loop.heldout_demos's split.

    Args:
        n_demos: Number of demos in the corpus.

    Returns:
        [j] held-out demo ids.
    """
    from robot_ood.train.loop import heldout_demos

    return heldout_demos(n_demos)


def demo_frames(store: Any, *keys: str) -> tuple[np.ndarray, ...]:
    """Read raw per-frame columns straight out of the node table, plus its demo boundaries.

    Args:
        store: The artifact store holding g_nodes.npz.
        *keys: Column names to read, e.g. "q_raw", "a_raw".

    Returns:
        The requested columns as float64, then raw_ptr as int64.
    """
    with np.load(store.path(schema.NODES_NPZ)) as nodes:
        out = [np.asarray(nodes[key], np.float64) for key in keys]
        out.append(np.asarray(nodes["raw_ptr"], np.int64))
    return tuple(out)


def triples(q_raw: np.ndarray, a_raw: np.ndarray, raw_ptr: np.ndarray,
            owners: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every consecutive (q_t, delta_q, a_t) triple inside the given demos.

    Args:
        q_raw: [M, 7] raw joint frames, demos concatenated.
        a_raw: [M, 7] raw actions, aligned with q_raw (a_raw[t] executed between frame t and t+1).
        raw_ptr: [J+1] demo boundaries into q_raw/a_raw.
        owners: Demo indices to draw from.

    Returns:
        (q [N,7], dq [N,7], a [N,7]) float32.
    """
    q_l, dq_l, a_l = [], [], []
    for j in owners:
        lo, hi = int(raw_ptr[j]), int(raw_ptr[j + 1])
        if hi - lo < 2:
            continue
        q_l.append(q_raw[lo:hi - 1])
        dq_l.append(q_raw[lo + 1:hi] - q_raw[lo:hi - 1])
        a_l.append(a_raw[lo:hi - 1])
    return (np.concatenate(q_l).astype(np.float32), np.concatenate(dq_l).astype(np.float32),
            np.concatenate(a_l).astype(np.float32))


def _split(n_demos: int) -> tuple[list[int], list[int]]:
    """(training demos, held-out demos) under the stride-HELDOUT_STRIDE split."""
    heldout = set(held_out_owners(n_demos).tolist())
    return [j for j in range(n_demos) if j not in heldout], sorted(heldout)


class _EntryTask:
    """What the two t = 0 nets do identically: one net, one drawn batch per epoch, no selection."""

    def __init__(self, hidden: int, batch: int, seed: int = 0) -> None:
        """Bind the task to its architecture and batch size.

        Args:
            hidden: Hidden width.
            batch: Rows drawn per step, with replacement; 0 uses the whole training set.
            seed: Seeds the draw. The runner seeds torch from its own spec.
        """
        self.hidden, self.batch, self.seed = hidden, batch, seed
        self.ctx: TrainContext | None = None
        self.net: Any = None

    def requires(self) -> Requirements:
        """The node table, which is where both nets' raw frames live."""
        return Requirements(artifacts=(schema.NODES_NPZ,))

    def batches(self, ctx: TrainContext, rng: np.random.RandomState) -> Iterator[np.ndarray]:
        """One drawn batch of row indices; an epoch here is a single gradient step.

        Args:
            ctx: The run's context.
            rng: The generator the draw comes from.

        Yields:
            [b] row indices, drawn with replacement.
        """
        n = self._n_rows
        yield rng.randint(0, n, size=(self.batch or n,))

    def select(self, metrics: dict[str, float]) -> float | None:
        """No epoch is selected: both nets keep their final weights, as they always have."""
        return None

    def save(self, unit: TrainUnit, store: Any) -> None:
        """Write the net through its own saver, which stamps the shapes load needs.

        Args:
            unit: The trained net.
            store: Where it goes.
        """
        unit.trained["net"].save(store.dir)


class ScoreTask(_EntryTask):
    """Fit ScoreNet by denoising score matching on the corpus's own t = 0 configurations.

    Attributes:
        name: The task's registry name.
    """

    name = "entry_score"

    def __init__(self, hidden: int = 128, batch: int = 256, seed: int = 0) -> None:
        """Bind the task; the noise band is read from C0's calibration in build.

        Args:
            hidden: Hidden width.
            batch: Rows per step.
            seed: Seeds the draw.
        """
        super().__init__(hidden, batch, seed)
        self.sigma_lo = self.sigma_hi = 0.0

    def build(self, ctx: TrainContext) -> TrainUnit:
        """Read the demo starts, standardise on them, and resolve the calibrated noise band.

        Args:
            ctx: The run's store, config, device and generator.

        Returns:
            The one net this task trains.
        """
        self.ctx = ctx
        q_raw, raw_ptr = demo_frames(ctx.store, "q_raw")
        starts = q_raw[raw_ptr[:-1]].astype(np.float32)
        train_idx, ho_idx = _split(len(raw_ptr) - 1)
        self.q = torch.as_tensor(starts[train_idx], dtype=torch.float32)
        self.q_heldout = (torch.as_tensor(starts[ho_idx], dtype=torch.float32) if ho_idx else None)
        self.sigma_lo, self.sigma_hi = entry_sigma_lo(), entry_sigma_hi()
        self.net = ScoreNet(hidden=self.hidden)
        # Standardised on the corrupted rows it trains on: a corpus whose starts coincide has no
        # spread of its own to standardise by.
        self.net.fit_stats(self.q + self._sigma(self.q.shape[0])[:, None] * torch.randn_like(self.q))
        return TrainUnit(trained={"net": self.net})

    def loss(self, unit: TrainUnit, batch: np.ndarray) -> tuple[torch.Tensor, dict[str, float]]:
        """The summed denoising-score-matching and level losses on one drawn batch.

        Args:
            unit: The net being trained.
            batch: [b] row indices.

        Returns:
            (the summed loss, the two terms separately).
        """
        net = unit.trained["net"]
        q_batch = self.q[torch.as_tensor(batch, dtype=torch.long)]
        sigma = self._sigma(q_batch.shape[0])
        # Summed, not weighted: the two heads share no parameters (only the standardisation
        # buffers), so the sum's gradient on each is exactly its own term's.
        dsm, level = dsm_loss(net, q_batch, sigma), level_loss(net, q_batch, sigma)
        return dsm + level, {"dsm": float(dsm.detach()), "level": float(level.detach())}

    def evaluate(self, unit: TrainUnit, evalset: Any) -> dict[str, float]:
        """The two losses on the held-out starts, plus what the head reads on a clean one.

        Args:
            unit: The net being trained.
            evalset: Unread; the held-out starts were read in build.

        Returns:
            ho_dsm, ho_level and delta_on_clean, or nothing when no demo was held out.
        """
        if self.q_heldout is None:
            return {}
        net = unit.trained["net"]
        sigma = self._sigma(self.q_heldout.shape[0])
        # delta_on_clean is the number deployment depends on: anything near the manifold's own
        # width means the schedule this net seeds is a no-op.
        return {
            "ho_dsm": float(dsm_loss(net, self.q_heldout, sigma)),
            "ho_level": float(level_loss(net, self.q_heldout, sigma)),
            "delta_on_clean": float(net.level_estimate(self.q_heldout).median()),
        }

    def _sigma(self, n: int) -> torch.Tensor:
        """[n] noise scales, drawn log-uniformly across the calibrated band."""
        drawn = self.ctx.rng.uniform(np.log(self.sigma_lo), np.log(self.sigma_hi), size=n)
        return torch.as_tensor(np.exp(drawn), dtype=torch.float32)

    @property
    def _n_rows(self) -> int:
        """Training rows."""
        return self.q.shape[0]


class InvDynTask(_EntryTask):
    """Fit InverseDynamicsNet: which action moved the arm from one raw frame to the next.

    Attributes:
        name: The task's registry name.
    """

    name = "entry_invdyn"

    def __init__(self, hidden: int = 64, batch: int = 512, seed: int = 0) -> None:
        """Bind the task to its architecture and batch size.

        Args:
            hidden: Hidden width.
            batch: Rows per step.
            seed: Seeds the draw.
        """
        super().__init__(hidden, batch, seed)

    def build(self, ctx: TrainContext) -> TrainUnit:
        """Extract every consecutive triple and standardise on the training half of them.

        Args:
            ctx: The run's store, config, device and generator.

        Returns:
            The one net this task trains.
        """
        self.ctx = ctx
        q_raw, a_raw, raw_ptr = demo_frames(ctx.store, "q_raw", "a_raw")
        train_owners, ho_owners = _split(len(raw_ptr) - 1)
        self.train = tuple(
            torch.as_tensor(x, dtype=torch.float32)
            for x in triples(q_raw, a_raw, raw_ptr, train_owners)
        )
        self.heldout = tuple(
            torch.as_tensor(x, dtype=torch.float32)
            for x in triples(q_raw, a_raw, raw_ptr, ho_owners)
        ) if ho_owners else None
        self.net = InverseDynamicsNet(hidden=self.hidden)
        # a_lo/a_hi are MEASURED bounds -- the corpus's own action min/max, not a hand-picked clamp.
        self.net.fit_stats(*self.train)
        return TrainUnit(trained={"net": self.net})

    def loss(self, unit: TrainUnit, batch: np.ndarray) -> tuple[torch.Tensor, dict[str, float]]:
        """Mean squared error between the predicted and the recorded action.

        Args:
            unit: The net being trained.
            batch: [b] row indices.

        Returns:
            (the mse, no extra scalars).
        """
        q, dq, a = self.train
        idx = torch.as_tensor(batch, dtype=torch.long)
        return ((unit.trained["net"](q[idx], dq[idx]) - a[idx]) ** 2).mean(), {}

    def evaluate(self, unit: TrainUnit, evalset: Any) -> dict[str, float]:
        """Mean squared error on the held-out demos' triples.

        Args:
            unit: The net being trained.
            evalset: Unread; the held-out triples were extracted in build.

        Returns:
            {"heldout_mse": ...}, or nothing when no demo was held out.
        """
        if self.heldout is None:
            return {}
        q, dq, a = self.heldout
        return {"heldout_mse": float(((unit.trained["net"](q, dq) - a) ** 2).mean())}

    @property
    def _n_rows(self) -> int:
        """Training rows."""
        return self.train[0].shape[0]
