"""The runner drives a task without knowing which task it is.

Every property the old staged loop branched on -- what inputs exist, what is frozen, which metric
picks the best epoch -- is asserted here through the contract instead, on a toy task small enough
that a wrong answer is unambiguous.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from robot_ood.data.store import ArtifactStore
from robot_ood.train.runner import MissingRequirement, Runner, RunSpec
from robot_ood.train.task import Requirements, TrainUnit


class _Toy:
    """Fits one scalar to a target. select() reports the distance, so lower is genuinely better."""

    name = "toy"

    def __init__(self, target: float = 3.0, *, needs: tuple[str, ...] = (), scores=None) -> None:
        self.target = target
        self.needs = needs
        self.scores = scores          # when given, the per-epoch select() values to hand back
        self.saved: list[float] = []
        self.epoch = 0

    def requires(self):
        return Requirements(artifacts=self.needs)

    def build(self, ctx):
        net = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            net.weight.fill_(0.0)
        frozen = nn.Linear(1, 1, bias=False)
        return TrainUnit(trained={"net": net}, frozen={"ref": frozen})

    def batches(self, ctx, rng):
        return [torch.ones(4, 1)] * 8

    def loss(self, unit, batch):
        pred = unit.trained["net"](batch)
        return ((pred - self.target) ** 2).mean(), {}

    def evaluate(self, unit, evalset):
        return {"dist": abs(float(unit.trained["net"].weight) - self.target)}

    def select(self, metrics):
        if self.scores is None:
            return metrics.get("dist")
        score = self.scores[min(self.epoch, len(self.scores) - 1)]
        self.epoch += 1
        return score

    def save(self, unit, store):
        self.saved.append(float(unit.trained["net"].weight))


def _spec(**kw):
    return RunSpec(**{"epochs": 3, "lr": 0.2, "save": False, **kw})


def test_a_declared_input_that_is_absent_fails_before_the_first_epoch(tmp_path):
    task = _Toy(needs=("g_head.npz", "g_missing.npz"))
    with pytest.raises(MissingRequirement, match="g_head.npz, g_missing.npz"):
        Runner(_spec()).run(task, ArtifactStore(tmp_path), cfg=None, device=torch.device("cpu"))


def test_training_moves_the_weight_towards_the_target(tmp_path):
    task = _Toy()
    result = Runner(_spec()).run(task, ArtifactStore(tmp_path), cfg=None,
                                 device=torch.device("cpu"), evalset=object())
    assert result.history[-1]["dist"] < result.history[0]["dist"]
    assert len(result.history) == 3


def test_the_best_epoch_is_the_one_the_task_scored_lowest(tmp_path):
    # Epoch 1 is the winner; the run then continues and ends on a worse epoch.
    task = _Toy(scores=[5.0, 1.0, 9.0])
    result = Runner(_spec()).run(task, ArtifactStore(tmp_path), cfg=None,
                                 device=torch.device("cpu"), evalset=object())
    assert (result.best.epoch, result.best.score) == (1, 1.0)


def test_patience_stops_a_run_that_stops_improving(tmp_path):
    """An epoch budget is a ceiling, not a schedule: a fit that peaked at epoch 1 must not spend the
    other 8 epochs walking away from the weights it is going to keep anyway."""
    task = _Toy(scores=[5.0, 1.0, 9.0, 9.0, 9.0, 9.0])
    result = Runner(_spec(epochs=9, patience=2)).run(
        task, ArtifactStore(tmp_path), cfg=None, device=torch.device("cpu"), evalset=object(),
    )
    assert (result.best.epoch, len(result.history)) == (1, 4)


def test_patience_zero_runs_every_epoch(tmp_path):
    """The default: nothing that trained before this option existed changes length."""
    task = _Toy(scores=[1.0, 9.0, 9.0, 9.0])
    result = Runner(_spec(epochs=4)).run(
        task, ArtifactStore(tmp_path), cfg=None, device=torch.device("cpu"), evalset=object(),
    )
    assert len(result.history) == 4


def test_an_unscorable_run_keeps_its_final_weights(tmp_path):
    """None and NaN admit no candidate -- the run must not fall back to some other metric."""
    task = _Toy(scores=[None, float("nan"), None])
    result = Runner(_spec()).run(task, ArtifactStore(tmp_path), cfg=None,
                                 device=torch.device("cpu"), evalset=object())
    assert result.best.epoch == -1 and result.best.state is None


def test_frozen_modules_are_never_trained_and_never_saved(tmp_path):
    task = _Toy()
    result = Runner(_spec(save=True)).run(task, ArtifactStore(tmp_path), cfg=None,
                                          device=torch.device("cpu"), evalset=object())
    assert result.best.state is not None and set(result.best.state) == {"net"}
    assert [p is q for p, q in zip(result.unit.parameters(),
                                   result.unit.trained["net"].parameters())] == [True]
    assert len(task.saved) == 1


def test_the_runner_names_no_objective():
    """§15.7's real acceptance criterion: a branch on the objective means the property is misplaced."""
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / "src/robot_ood/train/runner.py"
    text = source.read_text()
    for needle in ("RETRIEVAL", "CHUNK", "SEGMENT", "FUSION", "objective =="):
        assert needle not in text, f"runner.py branches on {needle!r}; move it into the task"


def test_the_seed_makes_a_run_reproducible(tmp_path):
    runs = [Runner(_spec(seed=7)).run(_Toy(), ArtifactStore(tmp_path), cfg=None,
                                      device=torch.device("cpu"), evalset=object())
            for _ in range(2)]
    np.testing.assert_allclose([r.history[-1]["dist"] for r in runs][0],
                               [r.history[-1]["dist"] for r in runs][1])
