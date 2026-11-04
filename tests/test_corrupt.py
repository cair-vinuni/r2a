"""The corrupted-retrieval sampler: branch shares, the two phase constraints, and the magnitude law.

robot_ood.train.augment had no dedicated test until this file. It is the module the whole blend
weight depends on -- with no corrupted rows the alpha gradient is one-sided and alpha can only learn
a monotone function of displacement -- so its shares and its phase bands being right is not a detail.

The magnitude test is the one that matters for the axis. The sampler used to draw uniformly over
(0.05, 0.25) m against a measured real displacement of 2.32 cm, so no training row ever sat near the
deployed regime; the log-uniform draw is what puts mass there.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_ood.train.augment import (
    CLEAN,
    KIND_NAMES,
    OFFSET,
    TARGET,
    WRONG_PHASE,
    WRONG_TASK,
    RetrievalCorruptor,
)


class _Nodes:
    """The five columns RetrievalCorruptor reads, as a plain stand-in for a NodeTable.

    Two demos per task over n_tasks tasks, each a straight phase ramp, so every branch has a legal
    candidate for every row and the rejection loop is never the thing under test.
    """

    def __init__(self, n_tasks: int = 4, per_demo: int = 25) -> None:
        demos = 2 * n_tasks
        self.phase = np.tile(np.linspace(0.0, 1.0, per_demo), demos).astype(np.float64)
        self.task_id = np.repeat(np.arange(n_tasks), 2 * per_demo).astype(np.int64)
        self.owner = np.repeat(np.arange(demos), per_demo).astype(np.int64)
        self.demo_ptr = (np.arange(demos + 1) * per_demo).astype(np.int64)
        self.n = demos * per_demo


@pytest.fixture()
def nodes() -> _Nodes:
    return _Nodes()


def _draw(nodes: _Nodes, n: int = 4000, seed: int = 0, **kwargs):
    corruptor = RetrievalCorruptor(nodes, **kwargs)
    rng = np.random.default_rng(seed)
    tgt = rng.integers(0, nodes.n, size=n)
    return corruptor, corruptor.draw(tgt, rng), tgt


def test_branch_shares_match_the_configured_probabilities(nodes):
    """Each branch claims its configured share, and the rest stays clean."""
    _, corruption, _ = _draw(nodes)
    counts = corruption.counts()
    assert set(counts) == set(KIND_NAMES)
    assert sum(counts.values()) == 4000
    for name in ("wrong_task", "offset", "wrong_phase"):
        assert counts[name] / 4000 == pytest.approx(0.10, abs=0.02), name
    assert counts["target"] / 4000 == pytest.approx(0.15, abs=0.02)
    assert counts["clean"] / 4000 == pytest.approx(0.55, abs=0.03)


def test_zero_shares_leave_every_row_clean(nodes):
    """A sampler with every share at zero is a no-op, not a differently-shaped corruption."""
    _, corruption, _ = _draw(nodes, p_wrong_task=0.0, p_offset=0.0, p_wrong_phase=0.0, p_target=0.0)
    assert corruption.counts()["clean"] == 4000
    assert np.all(corruption.node == -1)
    assert not np.any(corruption.offset)


def test_shares_summing_past_one_are_refused(nodes):
    """The constructor rejects an over-subscribed simplex rather than silently truncating it."""
    with pytest.raises(ValueError, match="must sum into"):
        RetrievalCorruptor(nodes, p_wrong_task=0.5, p_offset=0.4, p_wrong_phase=0.3)


def test_wrong_task_picks_another_task_at_a_comparable_phase(nodes):
    """WRONG_TASK is a genuine alias: different task, same place in the demo."""
    corruptor, corruption, tgt = _draw(nodes)
    rows = np.flatnonzero(corruption.kind == WRONG_TASK)
    assert rows.size > 0
    picked = corruption.node[rows]
    assert np.all(picked >= 0)
    assert np.all(nodes.task_id[picked] != nodes.task_id[tgt[rows]])
    gap = np.abs(nodes.phase[picked] - nodes.phase[tgt[rows]])
    assert np.all(gap <= corruptor.phase_tol + 1e-9)


def test_wrong_phase_stays_in_the_same_demo_but_moves_far_along_it(nodes):
    """WRONG_PHASE is the self-intersection case: right demo, wrong place."""
    corruptor, corruption, tgt = _draw(nodes)
    rows = np.flatnonzero(corruption.kind == WRONG_PHASE)
    assert rows.size > 0
    picked = corruption.node[rows]
    assert np.all(picked >= 0)
    assert np.array_equal(nodes.owner[picked], nodes.owner[tgt[rows]])
    gap = np.abs(nodes.phase[picked] - nodes.phase[tgt[rows]])
    assert np.all(gap >= corruptor.phase_min - 1e-9)


def test_offset_keeps_the_node_and_moves_only_the_pose(nodes):
    """OFFSET's node is right -- only where it is has changed -- so it must not be swapped."""
    _, corruption, _ = _draw(nodes)
    rows = np.flatnonzero(corruption.kind == OFFSET)
    assert rows.size > 0
    assert np.all(corruption.node[rows] == -1)
    assert np.all(np.linalg.norm(corruption.offset[rows], axis=1) > 0.0)
    clean = np.flatnonzero(corruption.kind == CLEAN)
    assert not np.any(corruption.offset[clean])


def test_a_branch_with_no_legal_candidate_falls_back_to_clean(nodes):
    """An impossible constraint downgrades the row rather than corrupting it some other way.

    phase_min > 1 cannot be satisfied by any node of a demo whose phase spans [0, 1].
    """
    _, corruption, _ = _draw(
        nodes, p_wrong_task=0.0, p_offset=0.0, p_wrong_phase=1.0, p_target=0.0, phase_min=2.0,
    )
    assert corruption.counts()["clean"] == 4000
    assert np.all(corruption.node == -1)


def test_a_single_task_graph_cannot_produce_a_wrong_task_row(nodes):
    """With one task there is no other task to alias to, so the branch must yield, not invent."""
    single = _Nodes(n_tasks=1)
    _, corruption, _ = _draw(single, p_wrong_task=1.0, p_offset=0.0, p_wrong_phase=0.0, p_target=0.0)
    assert corruption.counts()["wrong_task"] == 0
    assert corruption.counts()["clean"] == 4000


def test_offset_magnitudes_are_log_uniform_across_the_whole_range(nodes):
    """The magnitude law: mass in every half-decade, not piled at the top.

    A LINEAR draw over (0.005, 0.5) puts ~90% of its mass above 0.05 m and would fail the lower two
    bins here. That is the exact starvation the deployed head was trained under, against a real
    displacement of 2.32 cm.
    """
    lo, hi = 0.005, 0.5
    _, corruption, _ = _draw(
        nodes, n=20000, p_wrong_task=0.0, p_offset=1.0, p_wrong_phase=0.0, p_target=0.0, offset_m=(lo, hi),
    )
    magnitude = np.linalg.norm(corruption.offset[:, :3], axis=1)
    assert magnitude.min() >= lo - 1e-6 and magnitude.max() <= hi + 1e-6

    # log10 spans two decades; each of the four half-decade bins should hold about a quarter.
    share, _ = np.histogram(np.log10(magnitude), bins=4, range=(np.log10(lo), np.log10(hi)))
    assert np.all(share / magnitude.size > 0.15), share

    # The decisive contrast with the old sampler: a real share of rows lands near the real 2.32 cm.
    assert (magnitude < 0.05).mean() > 0.4


def test_target_rows_carry_a_displacement_in_their_own_ranges(nodes):
    """TARGET draws its end displacement from target_m / target_rad, and only TARGET and OFFSET rows
    carry one at all."""
    _, corruption, _ = _draw(nodes, n=20000, target_m=(0.02, 0.15), target_rad=(0.05, 0.5))
    rows = corruption.kind == TARGET
    pos = np.linalg.norm(corruption.offset[rows, :3], axis=1)
    rot = np.linalg.norm(corruption.offset[rows, 3:], axis=1)
    assert pos.min() >= 0.02 - 1e-6 and pos.max() <= 0.15 + 1e-6
    assert rot.min() >= 0.05 - 1e-6 and rot.max() <= 0.5 + 1e-6
    assert np.all(corruption.node[rows] == -1)
    still = ~rows & (corruption.kind != OFFSET)
    assert np.all(corruption.offset[still] == 0.0)


def test_the_draw_is_reproducible_and_seed_dependent(nodes):
    """Same seed, same corruption -- the set is drawn once per query set and then only sliced."""
    _, first, _ = _draw(nodes, seed=7)
    _, again, _ = _draw(nodes, seed=7)
    _, other, _ = _draw(nodes, seed=8)
    assert np.array_equal(first.kind, again.kind)
    assert np.array_equal(first.node, again.node)
    assert np.array_equal(first.offset, again.offset)
    assert not np.array_equal(first.kind, other.kind)


# ==================================================================================================
# PolicyCorruptor -- the other operand
#
# The retrieval branches teach alpha to go DOWN when the demonstration is invalid. These teach it to
# go UP when the POLICY is, which is the majority case on this benchmark: four of the seven long
# axes perturb only the image. Without them a large |a_track - a_policy| has exactly one meaning in
# training and the opposite meaning on most of the deployed axes.
# ==================================================================================================
class _PolicyNodes(_Nodes):
    """_Nodes plus the two columns PolicyCorruptor reads: raw_ptr and t_raw."""

    def __init__(self, n_tasks: int = 4, per_demo: int = 25, frames: int = 40) -> None:
        super().__init__(n_tasks, per_demo)
        demos = 2 * n_tasks
        self.raw_ptr = (np.arange(demos + 1) * frames).astype(np.int64)
        # One node per frame-block, chronological within each demo.
        self.t_raw = np.concatenate([
            (np.linspace(0, frames - 1, per_demo).astype(np.int64) + d * frames)
            for d in range(demos)
        ])
        self.n_frames = demos * frames


class _Cache:
    """A PolicyActionCache stand-in whose chunk at frame f is filled with the value f."""

    def __init__(self, n_frames: int, dim: int = 7) -> None:
        self.n_frames, self.dim = n_frames, dim

    def chunks(self, raw_idx: np.ndarray, k: int) -> np.ndarray:
        idx = np.asarray(raw_idx, dtype=np.int64)
        return np.broadcast_to(
            idx[:, None, None].astype(np.float32), (idx.size, k, self.dim)
        ).copy()


@pytest.fixture()
def pol_nodes() -> _PolicyNodes:
    return _PolicyNodes()


def _draw_policy(nodes: _PolicyNodes, n: int = 4000, seed: int = 0, **kwargs):
    from robot_ood.train.augment import PolicyCorruptor

    corruptor = PolicyCorruptor(nodes, RetrievalCorruptor(nodes), **kwargs)
    rng = np.random.default_rng(seed)
    tgt = rng.integers(0, nodes.n, size=n)
    now = nodes.t_raw[tgt]
    return corruptor, corruptor.draw(tgt, now, rng), tgt, now


def test_policy_branch_shares_match_the_configured_probabilities(pol_nodes):
    from robot_ood.train.augment import P_CLEAN, P_GAIN, P_LAG, P_SHIFT, P_TASK

    _, corruption, _, _ = _draw_policy(pol_nodes, n=20000, p_lag=0.2, p_task=0.1, p_gain=0.3, p_shift=0.1)
    share = {k: float((corruption.kind == k).mean()) for k in (P_CLEAN, P_LAG, P_TASK, P_GAIN, P_SHIFT)}
    assert share[P_LAG] == pytest.approx(0.2, abs=0.02)
    assert share[P_TASK] == pytest.approx(0.1, abs=0.02)
    assert share[P_GAIN] == pytest.approx(0.3, abs=0.02)
    assert share[P_SHIFT] == pytest.approx(0.1, abs=0.02)
    assert share[P_CLEAN] == pytest.approx(0.3, abs=0.02)


def test_shift_rows_carry_a_displacement_in_range_and_nothing_else_does(pol_nodes):
    """A SHIFT row moves the live pose, within SHIFT_M / SHIFT_RAD, and leaves the chunk alone."""
    from robot_ood.train.augment import P_SHIFT, SHIFT_M, SHIFT_RAD

    _, corruption, _, _ = _draw_policy(pol_nodes, n=4000, p_shift=0.3)
    rows = corruption.kind == P_SHIFT
    assert rows.sum() > 1000
    pos = np.linalg.norm(corruption.shift[rows, :3], axis=-1)
    rot = np.linalg.norm(corruption.shift[rows, 3:], axis=-1)
    assert pos.min() >= SHIFT_M[0] * 0.999 and pos.max() <= SHIFT_M[1] * 1.001
    assert rot.min() >= SHIFT_RAD[0] * 0.999 and rot.max() <= SHIFT_RAD[1] * 1.001
    assert np.all(corruption.frame[rows] == -1) and np.all(corruption.gain[rows] == 1.0)
    assert np.all(corruption.shift[~rows] == 0.0)


def test_zero_shares_leave_every_policy_chunk_untouched(pol_nodes):
    """The default must be free: a run that draws no policy corruption gets the clean chunk back
    bit-identically, so every existing arm is unaffected."""
    _, corruption, _, _ = _draw_policy(pol_nodes, p_lag=0.0, p_task=0.0, p_gain=0.0)
    clean = np.random.RandomState(0).randn(len(corruption), 8, 7).astype(np.float32)
    out = corruption.apply(clean, _Cache(pol_nodes.n_frames), 8)
    assert np.array_equal(out, clean)


def test_lag_stays_inside_its_own_demo(pol_nodes):
    """A lag that ran off the end would hand the row ANOTHER demo's action, which is the P_TASK
    branch wearing the wrong label -- and would corrupt the two branches' separation."""
    from robot_ood.train.augment import P_LAG

    _, corruption, _, now = _draw_policy(pol_nodes, n=20000, p_lag=1.0, p_task=0.0, p_gain=0.0, p_shift=0.0)
    rows = corruption.kind == P_LAG
    demo_now = np.searchsorted(pol_nodes.raw_ptr, now[rows], side="right") - 1
    demo_new = np.searchsorted(pol_nodes.raw_ptr, corruption.frame[rows], side="right") - 1
    assert np.array_equal(demo_now, demo_new)
    # And it really does move. A row whose lag clipped back onto its own frame is not counted as
    # corrupt at all -- a no-op labelled corrupt would sit in the strata table as fake evidence.
    assert np.all(corruption.frame[rows] != now[rows])


def test_policy_task_takes_another_task_at_a_comparable_phase(pol_nodes):
    from robot_ood.train.augment import P_TASK

    _, corruption, tgt, _ = _draw_policy(pol_nodes, n=20000, p_lag=0.0, p_task=1.0, p_gain=0.0, p_shift=0.0)
    rows = np.flatnonzero(corruption.kind == P_TASK)
    # The substitute frame belongs to a demo of a different task than the row's own label node.
    demo = np.searchsorted(pol_nodes.raw_ptr, corruption.frame[rows], side="right") - 1
    sub_task = pol_nodes.task_id[pol_nodes.demo_ptr[demo]]
    assert np.all(sub_task != pol_nodes.task_id[tgt[rows]])


def test_gain_is_log_uniform_outside_the_dead_zone_and_spares_the_gripper(pol_nodes):
    from robot_ood.train.augment import GAIN_DEAD, GAIN_RANGE, P_GAIN

    _, corruption, _, _ = _draw_policy(pol_nodes, n=20000, p_lag=0.0, p_task=0.0, p_gain=1.0, p_shift=0.0)
    gain = corruption.gain[corruption.kind == P_GAIN]
    lo, hi = GAIN_RANGE
    dlo, dhi = GAIN_DEAD
    assert gain.min() >= lo - 1e-6 and gain.max() <= hi + 1e-6
    # No corrupt row is a near-copy of a clean one: nothing lands inside the dead zone.
    assert not np.any((gain > dlo + 1e-6) & (gain < dhi - 1e-6))
    # Log-uniform on each side of it: both half-decades of a side hold a real share of that side.
    for a, b in ((lo, dlo), (dhi, hi)):
        side = gain[(gain >= a - 1e-6) & (gain <= b + 1e-6)]
        assert side.size > 0.1 * gain.size
        share, _ = np.histogram(np.log10(side), bins=2, range=(np.log10(a), np.log10(b)))
        assert np.all(share / side.size > 0.3), share

    clean = np.ones((len(corruption), 8, 7), dtype=np.float32)
    out = corruption.apply(clean, _Cache(pol_nodes.n_frames), 8)
    rows = corruption.kind == P_GAIN
    # Dim 6 is a binary gripper command; scaling it commands something no gripper can obey.
    assert np.array_equal(out[rows][:, :, 6], clean[rows][:, :, 6])
    expect = np.broadcast_to(corruption.gain[rows][:, None, None], out[rows][:, :, :6].shape)
    assert out[rows][:, :, :6] == pytest.approx(expect)


def test_apply_reads_the_substituted_frame_and_leaves_clean_rows_alone(pol_nodes):
    from robot_ood.train.augment import P_CLEAN

    _, corruption, _, _ = _draw_policy(pol_nodes, n=2000)
    clean = np.full((len(corruption), 8, 7), -1.0, dtype=np.float32)
    out = corruption.apply(clean, _Cache(pol_nodes.n_frames), 8)
    swapped = corruption.frame >= 0
    assert np.all(out[swapped][:, 0, 0] == corruption.frame[swapped].astype(np.float32))
    untouched = (corruption.kind == P_CLEAN) & ~swapped
    assert np.array_equal(out[untouched], clean[untouched])


def test_policy_and_retrieval_corruption_are_drawn_independently(pol_nodes):
    """All four quadrants must occur, or the head cannot learn to tell a wrong demo from a wrong
    policy -- they share one observable, the disagreement, and differ only in the confidence
    scalars."""
    from robot_ood.train.augment import P_CLEAN

    rng = np.random.default_rng(3)
    retr = RetrievalCorruptor(pol_nodes)
    pol = _draw_policy(pol_nodes, n=20000, seed=3)[0]
    tgt = rng.integers(0, pol_nodes.n, size=20000)
    now = pol_nodes.t_raw[tgt]
    r_kind = retr.draw(tgt, rng).kind
    p_kind = pol.draw(tgt, now, rng).kind
    for r_bad in (False, True):
        for p_bad in (False, True):
            cell = ((r_kind != CLEAN) == r_bad) & ((p_kind != P_CLEAN) == p_bad)
            assert cell.sum() > 200, (r_bad, p_bad, int(cell.sum()))
