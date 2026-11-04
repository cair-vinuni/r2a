"""The two t = 0 nets, driven by the runner instead of two hand-written loops in scripts/.

The scripts each carried their own copy of the same twenty lines. The test that matters is that
the transfer changed no number: a reference loop written here, step for step as the scripts had it,
must produce bit-identical weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robot_ood.data.store import ArtifactStore
from robot_ood.models.entry.invdyn import InverseDynamicsNet
from robot_ood.models.entry.score import ScoreNet, dsm_loss, level_loss
from robot_ood.data import schema
from robot_ood.train.runner import MissingRequirement, Runner, RunSpec
from robot_ood.train.task import Task
from robot_ood.train.tasks.entry import (
    InvDynTask, ScoreTask, demo_frames, held_out_owners, triples,
)

N_DEMOS, LENGTH, SEED = 6, 12, 0


@pytest.fixture
def store(tmp_path) -> ArtifactStore:
    """A node table carrying only the raw columns these two tasks read."""
    rng = np.random.RandomState(0)
    raw_ptr = np.arange(N_DEMOS + 1, dtype=np.int64) * LENGTH
    n = N_DEMOS * LENGTH
    np.savez(tmp_path / schema.NODES_NPZ,
             q_raw=rng.randn(n, 7).astype(np.float64),
             a_raw=rng.randn(n, 7).astype(np.float64),
             raw_ptr=raw_ptr)
    return ArtifactStore(tmp_path)


def _run(task, store, epochs=5, **kw):
    return Runner(RunSpec(epochs=epochs, lr=1e-3, seed=SEED, save=kw.pop("save", False))).run(
        task, store, cfg=None, device=torch.device("cpu"), evalset=object())


def test_both_satisfy_the_task_protocol():
    assert isinstance(ScoreTask(), Task) and isinstance(InvDynTask(), Task)


def test_a_missing_node_table_fails_before_the_first_step(tmp_path):
    with pytest.raises(MissingRequirement, match=schema.NODES_NPZ):
        _run(ScoreTask(), ArtifactStore(tmp_path))


def _band(store) -> tuple[float, float]:
    """The calibrated sigma band ScoreTask resolves in build."""
    from robot_ood.train.task import TrainContext

    task = ScoreTask()
    task.build(TrainContext(store=store, cfg=None, device=torch.device("cpu"),
                            rng=np.random.RandomState(SEED)))
    return task.sigma_lo, task.sigma_hi


def test_the_score_fit_is_the_loop_it_replaced(store):
    """Same draw order, same optimiser, same seed -- so the same weights, to the last bit."""
    epochs, hidden, batch = 5, 16, 8
    lo, hi = _band(store)
    result = _run(ScoreTask(hidden=hidden, batch=batch, seed=SEED), store, epochs=epochs)

    q_raw, raw_ptr = demo_frames(store, "q_raw")
    starts = q_raw[raw_ptr[:-1]].astype(np.float32)
    heldout = sorted(set(held_out_owners(len(raw_ptr) - 1).tolist()))
    q = torch.as_tensor(starts[[j for j in range(len(raw_ptr) - 1) if j not in heldout]],
                        dtype=torch.float32)
    q_ho = torch.as_tensor(starts[heldout], dtype=torch.float32)

    def draw(n):
        return torch.as_tensor(np.exp(rng.uniform(np.log(lo), np.log(hi), size=n)),
                               dtype=torch.float32)

    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    net = ScoreNet(hidden=hidden)
    net.fit_stats(q + draw(q.shape[0])[:, None] * torch.randn_like(q))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        idx = torch.as_tensor(rng.randint(0, q.shape[0], size=(batch,)), dtype=torch.long)
        q_batch = q[idx]
        sigma = draw(q_batch.shape[0])
        opt.zero_grad()
        (dsm_loss(net, q_batch, sigma) + level_loss(net, q_batch, sigma)).backward()
        opt.step()
        # Both losses draw torch noise, so the held-out report has to run here too or the two
        # streams diverge after epoch 1 -- which is what makes this a real check.
        with torch.no_grad():
            sigma_ho = draw(q_ho.shape[0])
            dsm_loss(net, q_ho, sigma_ho)
            level_loss(net, q_ho, sigma_ho)
            net.level_estimate(q_ho)

    for key, value in net.state_dict().items():
        assert torch.equal(value, result.unit.trained["net"].state_dict()[key]), key


def test_the_invdyn_fit_is_the_loop_it_replaced(store):
    epochs, hidden, batch = 5, 16, 8
    result = _run(InvDynTask(hidden=hidden, batch=batch, seed=SEED), store, epochs=epochs)

    q_raw, a_raw, raw_ptr = demo_frames(store, "q_raw", "a_raw")
    heldout = set(held_out_owners(len(raw_ptr) - 1).tolist())
    train = tuple(torch.as_tensor(x, dtype=torch.float32) for x in triples(
        q_raw, a_raw, raw_ptr, [j for j in range(len(raw_ptr) - 1) if j not in heldout]))

    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    net = InverseDynamicsNet(hidden=hidden)
    net.fit_stats(*train)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    q, dq, a = train
    for _ in range(epochs):
        idx = torch.as_tensor(rng.randint(0, q.shape[0], size=(batch,)), dtype=torch.long)
        opt.zero_grad()
        ((net(q[idx], dq[idx]) - a[idx]) ** 2).mean().backward()
        opt.step()

    for key, value in net.state_dict().items():
        assert torch.equal(value, result.unit.trained["net"].state_dict()[key]), key


def test_the_measured_action_bounds_survive_the_move(store):
    """a_lo/a_hi are the corpus's own action min/max, not a hand-picked clamp."""
    q_raw, a_raw, raw_ptr = demo_frames(store, "q_raw", "a_raw")
    heldout = set(held_out_owners(len(raw_ptr) - 1).tolist())
    _, _, a = triples(q_raw, a_raw, raw_ptr,
                      [j for j in range(len(raw_ptr) - 1) if j not in heldout])
    net = _run(InvDynTask(hidden=8, batch=4), store, epochs=1).unit.trained["net"]
    np.testing.assert_allclose(net.a_lo.numpy(), a.min(axis=0), rtol=1e-6)
    np.testing.assert_allclose(net.a_hi.numpy(), a.max(axis=0), rtol=1e-6)


def test_neither_selects_an_epoch(store):
    assert ScoreTask().select({"ho_dsm": 1.0}) is None
    assert InvDynTask().select({"heldout_mse": 1.0}) is None


def test_saving_writes_each_net_where_the_store_says(store):
    _run(ScoreTask(hidden=8, batch=4), store, epochs=1, save=True)
    _run(InvDynTask(hidden=8, batch=4), store, epochs=1, save=True)
    assert ScoreNet.load(store.dir) is not None
    assert InverseDynamicsNet.load(store.dir) is not None
