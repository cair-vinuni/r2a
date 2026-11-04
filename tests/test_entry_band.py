"""Tests for the ENTRY-stratum candidate-pool mask (GraphConfig.entry_band) — masks the retrieval
pool to phase <= entry_band before softmax, bounding the readout barycentre inside the demo-start
cloud rather than letting mid-trajectory nodes leak into it.

GraphRetriever.observe(): when cfg.entry_band > 0, logits are masked to nodes with
phase <= entry_band (masked_fill(-inf)), applied AFTER the task mask, BEFORE softmax. If no
node survives, the existing "no node is reachable" RuntimeError still raises (no silent un-mask).

Reuses the synthetic toy-graph fixture from tests/test_retrieve.py (straight-line demos,
NodeTable.from_demos + EdgeSet.build + a randomly-initialised GraphRetrieverNet — no dataset
dependency).
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_ood.core.config import GraphConfig
from robot_ood.runtime.retrieve import CandidateMasker, GraphRetriever
from test_retrieve import (
    _declare_ee_base, _loaded_retriever, _toy_net, _toy_nodes_edges, _write_graph_dir,
)


# ======================================================================================================
# 1a. entry_band=0 (default) is a documented no-op — bit-identical to the field not existing
# ======================================================================================================
def test_entry_band_default_is_bit_identical_noop(tmp_path):
    """entry_band=0.0 (the default) must retrieve identically to a config that never mentions the
    field at all — the whole point of "0.0 = off" as documented on GraphConfig.entry_band."""
    # Build ONE (nodes, edges, net) triple and write it to two graph dirs -- _loaded_retriever's
    # _toy_net seeds nothing, so two independent calls would give two DIFFERENT random nets and the
    # comparison below would be meaningless (an earlier version of this test learned that the hard way).
    nodes, edges = _toy_nodes_edges()
    net = _toy_net(nodes)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    d_plain = _write_graph_dir(tmp_path / "a", nodes, edges, net)
    d_zero = _write_graph_dir(tmp_path / "b", nodes, edges, net)
    r_plain = GraphRetriever.load(d_plain, cfg=GraphConfig(device="cpu"))
    r_zero = GraphRetriever.load(d_zero, cfg=GraphConfig(device="cpu", entry_band=0.0))
    _declare_ee_base(r_plain.nodes)
    _declare_ee_base(r_zero.nodes)

    q0 = nodes.q[len(nodes) // 2].astype(np.float64)
    q_hist = np.repeat(q0[None, :], r_plain.cfg.hist, axis=0)

    res_plain = r_plain.retrieve(q_hist)
    res_zero = r_zero.retrieve(q_hist)

    assert np.array_equal(res_plain.q_star, res_zero.q_star)
    assert np.array_equal(res_plain.p_node, res_zero.p_node)
    assert res_plain.depth == res_zero.depth


# ======================================================================================================
# 1b. entry_band > 0 restricts the surviving candidate pool to phase <= entry_band
# ======================================================================================================
def test_entry_band_restricts_candidates_to_entry_stratum(tmp_path):
    """Every node with nonzero posterior mass must satisfy phase <= entry_band, and the returned
    q_star (an average over that surviving pool, for readouts that blend — euc_raw) must lie
    inside the per-dim [min, max] box of the surviving nodes' q — a convex-hull-superset check
    cheap enough not to depend on the readout arm's exact blending rule."""
    nodes, edges = _toy_nodes_edges()
    band = float(np.median(nodes.phase))  # keeps roughly half the graph, definitely nonempty
    assert 0.0 < band < 1.0, "fixture sanity: need a band that is neither empty nor everything"

    retriever = _loaded_retriever(tmp_path, entry_band=band)
    q0 = nodes.q[nodes.owner == 0][0].astype(np.float64)  # first node of demo 0 (near phase 0)
    q_hist = np.repeat(q0[None, :], retriever.cfg.hist, axis=0)

    result = retriever.retrieve(q_hist)

    survivors = np.flatnonzero(result.p_node > 0)
    assert survivors.size > 0, "band was chosen to keep some nodes"
    assert np.all(nodes.phase[survivors] <= band + 1e-6), (
        "every node carrying nonzero posterior mass must be within the entry stratum"
    )

    q_surv = nodes.q[survivors].astype(np.float64)
    lo, hi = q_surv.min(axis=0), q_surv.max(axis=0)
    assert np.all(result.q_star >= lo - 1e-6) and np.all(result.q_star <= hi + 1e-6), (
        "q_star must lie inside the surviving entry-stratum nodes' bounding box"
    )
    assert np.all(np.isfinite(result.q_star))


# ======================================================================================================
# 1c. entry_band so tight (combined with a task filter) that no node survives -> RuntimeError
# ======================================================================================================
def test_entry_band_empty_pool_raises_runtime_error(tmp_path):
    """A band tighter than the smallest phase in the graph, filtered to a single task, must leave no
    finite logit and raise the existing 'no node is reachable' error — no silent un-mask fallback
    inside GraphRetriever.observe/retrieve; that fallback is GraphTarget's job."""
    # A real graph always has phase-0 nodes (every demo's first frame), so a positive band can never
    # empty the pool via the band alone -- emptiness only arises through the mask COMPOSING with the
    # task / reachability masks, and NodeTable.validate (correctly) refuses to load a table whose
    # phases don't span [0,1]. To exercise the raise path in isolation, load a VALID graph and then
    # shift the retriever's own cached phase tensor above a positive band: entry_band > 0 keeps the
    # mask enabled (a negative band would just disable it, which is exactly the bug an earlier version
    # of this test had) while no node satisfies it.
    tiny_band = 0.25
    retriever = _loaded_retriever(tmp_path, entry_band=tiny_band)
    # The band mask is precomputed once in CandidateMasker.__init__ (it is a constant [V] boolean, and
    # recomputing it every control step bought nothing), so rebuild the masker from the shifted phase
    # rather than mutating the cached tensor behind its back.
    retriever._phase_t = retriever._phase_t * 0.5 + 0.5  # every node now at phase >= 0.5 > band
    retriever.masker = CandidateMasker(
        device=retriever.device, entry_band=tiny_band,
        owner_ids=retriever._owner_t, phase=retriever._phase_t,
    )
    nodes = retriever.nodes
    q0 = nodes.q[0].astype(np.float64)
    q_hist = np.repeat(q0[None, :], retriever.cfg.hist, axis=0)

    with pytest.raises(RuntimeError, match="no node is reachable"):
        retriever.retrieve(q_hist)
