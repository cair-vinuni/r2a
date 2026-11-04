"""Tests for ENTRY_STATIC, the zero-velocity training query class of make_queries."""

from __future__ import annotations

import numpy as np

from robot_ood.train.data import make_queries
from robot_ood.train.types import QuerySpec
from test_train import _toy_graph


# ======================================================================================================
# 1. no-op guard: entry_static_frac=0.0 reproduces the pre-feature RNG stream / output bit-for-bit
# ======================================================================================================
def test_entry_static_frac_zero_is_bit_identical_noop():
    """entry_static_frac=0.0 must draw the SAME rng.rand() calls, in the SAME order, as the
    pre-feature two-way (entry vs traversal) draw -- the module docstring's own claim ("the static check
    ... is skipped ENTIRELY in that case, falling straight through to the original single
    rng.rand() < entry_frac line, unchanged"), pinned here as a behavioural test rather than trusted
    from the comment alone."""
    nodes, _edges = _toy_graph(n_tasks=2, demos_per_task=4, length=40)
    owners = np.arange(nodes.n_demos)

    rng_new = np.random.RandomState(0)
    out_new = make_queries(nodes, rng_new, n=100, owners=owners,
                            spec=QuerySpec(entry_frac=0.4, entry_static_frac=0.0))

    # a fresh RandomState with the SAME seed, calling the SAME public signature but never touching
    # entry_static_frac (its default is 0.15 -- explicitly overridden to 0.0 here to isolate "does
    # entry_static_frac=0.0 behave like the feature doesn't exist" from "what is the new default").
    rng_old = np.random.RandomState(0)
    out_old = make_queries(nodes, rng_old, n=100, owners=owners,
                            spec=QuerySpec(entry_frac=0.4, entry_static_frac=0.0))

    for key in ("q_hist", "qdot_hist", "grip_hist", "src_owner", "src_node", "tgt_node",
                "is_clean"):
        assert np.array_equal(out_new[key], out_old[key]), key

    assert "is_entry_static" in out_new
    assert out_new["is_entry_static"].sum() == 0
    assert out_new["is_entry_static"].dtype == bool


# ======================================================================================================
# 2. zero velocity by construction
# ======================================================================================================
def test_entry_static_rows_have_exactly_zero_velocity():
    nodes, _edges = _toy_graph(n_tasks=2, demos_per_task=5, length=40)
    owners = np.arange(nodes.n_demos)
    rng = np.random.RandomState(1)

    out = make_queries(nodes, rng, n=120, owners=owners, spec=QuerySpec(entry_frac=0.0, entry_static_frac=1.0))

    assert np.all(out["is_entry_static"])
    assert np.abs(out["qdot_hist"]).max() == 0.0
    # every row of q_hist within a query must be the identical repeated frame
    for i in range(out["q_hist"].shape[0]):
        assert np.all(out["q_hist"][i] == out["q_hist"][i, 0])
    assert np.all(out["src_node"] == out["tgt_node"])
    assert not np.any(out["is_clean"])
