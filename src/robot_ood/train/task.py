"""What a trainable objective has to say for itself, and what the runner does with the answer.

This repo grew three ways to train something -- the staged loop, the fitters under build/, and
three standalone scripts -- sharing the same skeleton and none of the code. Task is the contract
that lets one runner drive all of them.

Adding a head should cost one file. Anything the runner would have to branch on is a property of
the task that has not been put into requires(), build() or select() yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor, nn

from robot_ood.data.store import ArtifactStore


@dataclass(frozen=True)
class Requirements:
    """What must exist before a task can start.

    Attributes:
        artifacts: Filenames the task reads out of the store, e.g. ("g_head.npz",). The runner
            checks all of them before the first epoch, so a missing input fails in a second with a
            message naming the file, not an hour in with a KeyError.
        upstream: Tasks that must have run first, by name.
    """

    artifacts: tuple[str, ...] = ()
    upstream: tuple[str, ...] = ()


@dataclass
class TrainUnit:
    """What is being trained, and what is along for the ride frozen.

    Attributes:
        trained: Modules the optimizer sees and the best-epoch snapshot captures.
        frozen: Modules held in eval mode under no_grad. Never snapshotted, never saved -- a frozen
            Stage 1 head is an input to Stage 2, not an output of it.
        param_groups: Optimizer groups, when a task needs its own lr or weight decay. Empty means
            one group over every trained parameter.
        optimizer: The live optimizer, set by the runner before the first epoch so an on_epoch
            hook can retune group learning rates mid-run.
    """

    trained: dict[str, nn.Module]
    frozen: dict[str, nn.Module] = field(default_factory=dict)
    param_groups: list[dict[str, Any]] = field(default_factory=list)
    optimizer: Any = None

    def parameters(self) -> list[nn.Parameter]:
        """Every trained parameter, in module insertion order."""
        return [p for module in self.trained.values() for p in module.parameters()]

    def groups(self) -> list[dict[str, Any]]:
        """The optimizer's parameter groups: the task's own, or one group over everything."""
        return self.param_groups or [{"params": self.parameters()}]

    def train_mode(self) -> None:
        """Put trained modules in train mode and frozen ones in eval, before an epoch."""
        for module in self.trained.values():
            module.train()
        for module in self.frozen.values():
            module.eval()

    def eval_mode(self) -> None:
        """Put every module in eval mode, before measuring."""
        for module in (*self.trained.values(), *self.frozen.values()):
            module.eval()

    def snapshot(self) -> dict[str, dict[str, Tensor]]:
        """A detached clone of every trained module's weights."""
        return {
            name: {k: v.detach().clone() for k, v in module.state_dict().items()}
            for name, module in self.trained.items()
        }

    def restore(self, snapshot: Mapping[str, Mapping[str, Tensor]]) -> None:
        """Load a snapshot back into the trained modules.

        Args:
            snapshot: What snapshot returned.
        """
        for name, state in snapshot.items():
            self.trained[name].load_state_dict(state)


@dataclass(frozen=True)
class TrainContext:
    """Everything a task needs to build itself that is not the task's own decision.

    Attributes:
        store: The artifacts beside the graph being trained on.
        cfg: The active graph configuration.
        device: Torch device every module is placed on.
        rng: The run's numpy generator; the only source of randomness a task should draw from.
        logger: Where progress goes, or None for stdout.
        parallel: This rank's place in a data-parallel run (robot_ood.train.parallel.Parallel), or
            None for a single-process run. A task that draws per-epoch data divides its own row
            budget by parallel.world_size, so an epoch is the same amount of data on any number of
            ranks.
    """

    store: ArtifactStore
    cfg: Any
    device: torch.device
    rng: np.random.RandomState
    logger: Any = None
    parallel: Any = None


@runtime_checkable
class Task(Protocol):
    """One trainable objective.

    Attributes:
        name: How the runner, the requirements and the CLI refer to this task.
        report: Optional, and so listed here rather than declared below -- a declared member is
            required of every task. report(unit, ctx) is a closing diagnostic the runner calls once
            at the end under no_grad, which the task logs rather than returns.
        on_epoch: Optional. on_epoch(unit, epoch) runs after train_mode and before the epoch's
            batches, for a task whose objective changes with the epoch index.
    """

    name: str

    def requires(self) -> Requirements:
        """What must exist before this task can start."""

    def build(self, ctx: TrainContext) -> TrainUnit:
        """Construct the modules, splitting what is trained from what is frozen.

        Args:
            ctx: The run's store, config, device and generator.

        Returns:
            The unit the runner optimizes.
        """

    def batches(self, ctx: TrainContext, rng: np.random.RandomState) -> Iterable[Any]:
        """One epoch's worth of training batches -- epoch length is a property of the task's data.

        Args:
            ctx: The run's context.
            rng: The generator to draw from.

        Yields:
            Whatever loss() takes.
        """

    def loss(self, unit: TrainUnit, batch: Any) -> tuple[Tensor, dict[str, float]]:
        """Score one batch.

        Args:
            unit: The modules being trained.
            batch: What batch() returned.

        Returns:
            (the tensor to backward, per-batch scalars to log).
        """

    def evaluate(self, unit: TrainUnit, evalset: Any) -> dict[str, float]:
        """Measure one epoch on held-out data.

        Args:
            unit: The modules being trained.
            evalset: The held-out set this task built.

        Returns:
            Named metrics.
        """

    def select(self, metrics: dict[str, float]) -> float | None:
        """This epoch's model-selection score, lower being better.

        Args:
            metrics: What evaluate returned.

        Returns:
            The score, or None when this epoch cannot be scored -- the runner then admits no
            candidate rather than selecting on a proxy the objective did not ask for.
        """

    def save(self, unit: TrainUnit, store: ArtifactStore) -> None:
        """Write the trained weights.

        Args:
            unit: The modules to save, already restored to the best epoch.
            store: Where they go. A task is handed the store, not a path, so it never has to know
                the artifact directory's layout.
        """
