"""The one training loop: seed, epochs, optimizer, best epoch, save.

The runner never learns an objective's name. What it needs (requires), what it trains (build) and
how an epoch scores (select) are all the Task's to declare; a branch on which objective is running
belongs in the task, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from robot_ood.data.store import ArtifactStore
from robot_ood.train.parallel import Parallel, world
from robot_ood.train.task import Task, TrainContext, TrainUnit


class MissingRequirement(FileNotFoundError):
    """A task's declared input is not beside the graph."""


@dataclass
class BestEpoch:
    """The lowest-scoring epoch seen, and its weights.

    Attributes:
        score: The winning score, or None before any epoch is admitted.
        epoch: Its zero-based index, -1 before any epoch is admitted.
        state: Its cloned per-module state dicts, or None.
    """

    score: float | None = None
    epoch: int = -1
    state: dict[str, dict[str, torch.Tensor]] | None = None

    def consider(self, unit: TrainUnit, epoch: int, score: float | None) -> None:
        """Snapshot this epoch if it scores lowest.

        Args:
            unit: The modules whose weights would be snapshotted.
            epoch: Zero-based epoch index.
            score: What the task's select() returned.
        """
        # A None or non-finite score admits no epoch: a run the objective could not score keeps its
        # final weights rather than selecting on something it never asked to be judged by.
        if score is None or not np.isfinite(score):
            return
        if self.score is not None and score >= self.score:
            return
        self.score, self.epoch, self.state = float(score), epoch, unit.snapshot()

    def restore_into(self, unit: TrainUnit, final_epoch: int) -> bool:
        """Load the winning weights back, unless the final epoch already won.

        Args:
            unit: The modules to load into.
            final_epoch: Zero-based index of the run's last epoch.

        Returns:
            True if weights were actually restored.
        """
        if self.state is None or self.epoch == final_epoch:
            return False
        unit.restore(self.state)
        return True


@dataclass(frozen=True)
class RunSpec:
    """How to run a task, independent of which task it is.

    Attributes:
        epochs: How many passes over the task's batches.
        lr: Learning rate for groups that do not set their own.
        weight_decay: Weight decay, likewise.
        seed: Seeds torch and the run's numpy generator.
        save: Write the trained weights when the run finishes.
        log_every: Epochs between progress lines. The last epoch always reports.
        patience: Stop after this many epochs without a new best score. 0 runs every epoch.
        eval_every: Epochs between held-out measurements; the last epoch always measures. Above 1
            only for an objective that selects no epoch, since a skipped epoch cannot be chosen.
    """

    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    save: bool = True
    log_every: int = 1
    patience: int = 0
    eval_every: int = 1


@dataclass
class RunResult:
    """What one run produced.

    Attributes:
        unit: The trained modules, restored to the best epoch.
        best: Which epoch won and what it scored.
        history: One entry per epoch: its mean loss and its metrics.
        final: The winning epoch's metrics.
    """

    unit: TrainUnit
    best: BestEpoch
    history: list[dict[str, Any]] = field(default_factory=list)
    final: dict[str, float] = field(default_factory=dict)


class Runner:
    """Drives one Task through one RunSpec."""

    def __init__(self, spec: RunSpec, logger: Any = None, parallel: Parallel | None = None) -> None:
        """Bind a runner to a spec.

        Args:
            spec: Epochs, batches, optimizer settings and the seed.
            logger: Where progress goes; None prints.
            parallel: This rank's place in a data-parallel run; None reads the launcher's
                environment (robot_ood.train.parallel.world).
        """
        self.spec = spec
        self.logger = logger
        self.parallel = world() if parallel is None else parallel

    def run(self, task: Task, store: ArtifactStore, cfg: Any, device: torch.device,
            evalset: Any = None) -> RunResult:
        """Train one task to completion.

        Args:
            task: The objective.
            store: The artifacts beside the graph.
            cfg: The active graph configuration.
            device: Where modules are placed.
            evalset: The held-out set passed to task.evaluate, or None to skip measuring.

        Returns:
            The trained unit, the winning epoch and the per-epoch history.

        Raises:
            MissingRequirement: A file the task declared is not beside the graph.
        """
        self._check(task, store)
        par = self.parallel
        torch.manual_seed(self.spec.seed)
        device = par.device(device)
        # Every rank draws its OWN rows off its own stream; the weights stay identical because the
        # gradients are averaged before each step (and broadcast once below).
        ctx = TrainContext(store=store, cfg=cfg, device=device,
                           rng=np.random.RandomState(par.seed(self.spec.seed)), logger=self.logger,
                           parallel=par)

        unit = task.build(ctx)
        for module in unit.trained.values():
            par.broadcast(module)
        opt = torch.optim.AdamW(unit.groups(), lr=self.spec.lr,
                                weight_decay=self.spec.weight_decay)
        unit.optimizer = opt
        result = RunResult(unit=unit, best=BestEpoch())
        metrics: dict[str, float] = {}
        last = self.spec.epochs - 1

        for epoch in range(self.spec.epochs):
            unit.train_mode()
            if hasattr(task, "on_epoch"):
                task.on_epoch(unit, epoch)
            params = unit.parameters()
            losses = []
            for batch in task.batches(ctx, ctx.rng):
                opt.zero_grad(set_to_none=True)
                loss, _ = task.loss(unit, batch)
                loss.backward()
                par.average_grads(params)
                opt.step()
                losses.append(float(loss.detach()))

            mean_loss = par.mean(float(np.mean(losses)) if losses else float("nan"))
            # Only rank 0 measures, and every other rank waits for it below: an eval the run does
            # not read is idle time on every GPU but one.
            due = epoch % max(1, self.spec.eval_every) == 0 or epoch == self.spec.epochs - 1
            metrics = self._measure(task, unit, evalset) if (par.is_main and due) else {}
            result.best.consider(unit, epoch, task.select(metrics))
            result.history.append({"epoch": epoch, "loss": mean_loss, **metrics})
            self._emit(task.name, epoch, mean_loss, metrics)
            if due or epoch % max(1, self.spec.log_every) == 0 or epoch == self.spec.epochs - 1:
                self._log(f"[{task.name}] epoch {epoch + 1}/{self.spec.epochs} loss={mean_loss:.4f} "
                          + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            par.barrier()
            last = epoch
            if self.spec.patience and epoch - result.best.epoch >= self.spec.patience:
                self._log(f"[{task.name}] stopped at epoch {epoch + 1}: "
                          f"{self.spec.patience} epochs without a new best")
                break

        restored = result.best.restore_into(unit, last)
        if par.is_main:
            result.final = self._measure(task, unit, evalset) if restored else metrics
            self._log(f"[{task.name}] done: best epoch {result.best.epoch + 1} "
                      f"score={result.best.score}")
            self._report(task, unit, ctx)
            if self.spec.save:
                task.save(unit, store)
        par.barrier()
        return result

    @staticmethod
    def _report(task: Task, unit: TrainUnit, ctx: TrainContext) -> None:
        """Run the task's closing diagnostic, when it has one.

        Args:
            task: The objective.
            unit: Its modules, restored to the best epoch.
            ctx: The run's context.
        """
        # Optional and duck-typed rather than a branch: a diagnostic only one objective produces is
        # still that objective's business, so the runner asks rather than knowing which ones have it.
        hook = getattr(task, "report", None)
        if hook is None:
            return
        unit.eval_mode()
        with torch.no_grad():
            hook(unit, ctx)

    @staticmethod
    def _measure(task: Task, unit: TrainUnit, evalset: Any) -> dict[str, float]:
        """The epoch's held-out metrics, or nothing when there is no held-out set."""
        if evalset is None:
            return {}
        unit.eval_mode()
        with torch.no_grad():
            return task.evaluate(unit, evalset)

    @staticmethod
    def _check(task: Task, store: ArtifactStore) -> None:
        """Refuse before the first epoch when a declared input is absent.

        Args:
            task: The objective.
            store: Where its inputs should be.

        Raises:
            MissingRequirement: Naming every file that is not there.
        """
        absent = [name for name in task.requires().artifacts if not store.has(name)]
        if absent:
            raise MissingRequirement(
                f"{task.name}: {', '.join(absent)} not beside the graph at {store.dir}. "
                "Run the stage that produces them first."
            )

    def _log(self, message: str) -> None:
        """Send one line to the logger, or print it; every rank but the first stays quiet."""
        if not self.parallel.is_main:
            return
        (self.logger.log(message) if self.logger is not None else print(message))

    def _emit(self, name: str, epoch: int, loss: float, metrics: Mapping[str, float]) -> None:
        """Hand the epoch's numbers to a logger that records them structurally.

        Args:
            name: The task's name, which prefixes every key.
            epoch: Zero-based epoch index.
            loss: Its mean training loss.
            metrics: Its held-out metrics.
        """
        emit = getattr(self.logger, "metric", None)
        if emit is None or not self.parallel.is_main:
            return
        emit(**{f"{name}_ep{epoch}_loss": loss},
             **{f"{name}_ep{epoch}_{k}": v for k, v in metrics.items()})
