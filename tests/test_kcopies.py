"""K fixed corruption copies per query -- the scale axis that costs no demonstrations.

The rule this must not break: corruption is drawn ONCE and never redrawn per epoch, because a
redrawn row is a noise PROCESS and the only fit consistent with all of its draws is the mean. That
argument is about redrawing, not about how many copies exist, so K copies of a query -- each with
its own fixed corruption -- multiply the training distribution while every row stays an example.

The two properties asserted here are exactly those: K=1 reproduces the pre-existing draw row for
row, and a K>1 plan is reproducible from (seed, copy) alone.
"""

from __future__ import annotations

import numpy as np
import torch

from robot_ood.core.config import GraphConfig
from robot_ood.train.augment import CopyPlan, PolicyCorruptor, RetrievalCorruptor
from robot_ood.train.data import tile_queries
from robot_ood.train.tasks.query import QuerySet
from test_corrupt import _PolicyNodes

HIST, DIM, N = 4, 3, 64


def _queries(nodes: _PolicyNodes, n: int = N, seed: int = 0) -> dict[str, np.ndarray]:
    """The columns QuerySet.from_queries reads, with no chunk targets.

    Args:
        nodes: The node stand-in.
        n: Rows.
        seed: Random state.

    Returns:
        A make_queries-shaped mapping.
    """
    rng = np.random.RandomState(seed)
    tgt = rng.randint(0, nodes.n, size=n).astype(np.int64)
    return {
        "q_hist": rng.randn(n, HIST, DIM).astype(np.float32),
        "qdot_hist": rng.randn(n, HIST, DIM).astype(np.float32),
        "grip_hist": rng.randn(n, HIST).astype(np.float32),
        "src_node": tgt,
        "tgt_node": tgt,
        "t_raw_now": nodes.t_raw[tgt],
    }


def _query_set(queries, nodes, copies, seed: int = 5) -> QuerySet:
    """One QuerySet with both corruptors drawn and no chunk targets.

    Args:
        queries: The mapping.
        nodes: The node stand-in.
        copies: A CopyPlan, or None.
        seed: Seed of the shared generator the K=1 path draws from.

    Returns:
        The set.
    """
    picker = RetrievalCorruptor(nodes)
    return QuerySet.from_queries(
        queries, "cpu", None, picker, np.random.default_rng(seed),
        policy_corruptor=PolicyCorruptor(nodes, picker), copies=copies,
    )


def test_tile_queries_is_copy_major_and_free_at_k1():
    nodes = _PolicyNodes()
    queries = _queries(nodes)
    assert tile_queries(queries, 1) is queries
    tiled = tile_queries(queries, 3)
    assert len(tiled["tgt_node"]) == 3 * N
    for k in range(3):
        np.testing.assert_array_equal(tiled["tgt_node"][k * N : (k + 1) * N], queries["tgt_node"])
        np.testing.assert_array_equal(tiled["q_hist"][k * N : (k + 1) * N], queries["q_hist"])


def test_copy_plan_rows_match_the_tiling():
    assert np.array_equal(CopyPlan(3, 0).rows(N), np.tile(np.arange(N), 3))
    assert np.array_equal(CopyPlan(1, 0).rows(N), np.arange(N))


def test_a_copy_stream_depends_only_on_seed_and_copy_index():
    """Not on anything else the run has drawn: two plans with the same seed hand copy k the same
    stream, and two copies of one plan get different ones."""
    first, again = CopyPlan(4, 11), CopyPlan(4, 11)
    assert np.array_equal(first.rng(2).random(8), again.rng(2).random(8))
    assert not np.array_equal(first.rng(2).random(8), first.rng(3).random(8))
    assert not np.array_equal(first.rng(2).random(8), CopyPlan(4, 12).rng(2).random(8))


def test_k_one_reproduces_the_single_draw_exactly():
    """K=1 must be the behaviour that predates this feature, row for row -- the whole point of the
    lever is that turning it off costs nothing."""
    nodes = _PolicyNodes()
    queries = _queries(nodes)
    before = _query_set(queries, nodes, copies=None)
    after = _query_set(queries, nodes, copies=CopyPlan(1, 0))
    assert len(before) == len(after) == N
    np.testing.assert_array_equal(before.corrupt.kind, after.corrupt.kind)
    np.testing.assert_array_equal(before.corrupt.node, after.corrupt.node)
    np.testing.assert_array_equal(before.corrupt.offset, after.corrupt.offset)
    np.testing.assert_array_equal(before.policy_corrupt.kind, after.policy_corrupt.kind)
    np.testing.assert_array_equal(before.policy_corrupt.frame, after.policy_corrupt.frame)


def test_k_copies_multiply_the_rows_and_differ_between_copies():
    """Ten copies is ten times the data; copies that all drew the same corruption would be ten
    times the same example."""
    nodes = _PolicyNodes()
    queries = _queries(nodes)
    every = _query_set(queries, nodes, copies=CopyPlan(10, 3))
    assert len(every) == 10 * N
    kind = every.corrupt.kind.reshape(10, N)
    assert not np.array_equal(kind[0], kind[1])
    # Every copy still describes the SAME source queries, in the same order.
    np.testing.assert_array_equal(
        every.tgt.reshape(10, N), np.tile(queries["tgt_node"], (10, 1))
    )
    np.testing.assert_array_equal(
        every.qh.numpy()[:N], queries["q_hist"]
    )


def test_a_copy_is_an_example_not_a_process():
    """Rebuilding the set redraws nothing: copy k of seed s is one fixed corruption forever, which
    is what keeps a corrupted row a training example rather than a noise process."""
    nodes = _PolicyNodes()
    queries = _queries(nodes)
    first = _query_set(queries, nodes, copies=CopyPlan(4, 7))
    again = _query_set(queries, nodes, copies=CopyPlan(4, 7), seed=999)
    np.testing.assert_array_equal(first.corrupt.kind, again.corrupt.kind)
    np.testing.assert_array_equal(first.corrupt.offset, again.corrupt.offset)
    np.testing.assert_array_equal(first.policy_corrupt.gain, again.policy_corrupt.gain)


def test_held_out_sets_stay_at_one_copy():
    """The default is one copy, so a set built without a plan -- which is what every held-out set
    is -- keeps the row count its table was always reported at."""
    nodes = _PolicyNodes()
    held = _query_set(_queries(nodes), nodes, copies=None)
    assert len(held) == N
    assert GraphConfig().query_copies == 1
    assert CopyPlan().copies == 1


def test_the_batch_slicing_survives_the_expansion():
    """Every downstream consumer slices by row index, so the expanded set has to batch like any
    other."""
    nodes = _PolicyNodes()
    every = _query_set(_queries(nodes), nodes, copies=CopyPlan(3, 1))
    seen = sum(len(batch.tgt) for batch in every.ordered_batches(16))
    assert seen == 3 * N
    batch = every.batch(np.arange(N, N + 8))
    assert batch.qh.shape == (8, HIST, DIM)
    assert isinstance(batch.corrupt_kind, np.ndarray) and len(batch.corrupt_kind) == 8
    assert torch.is_tensor(batch.corrupt_offset)
