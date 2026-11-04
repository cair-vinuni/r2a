"""Tests for robot_ood.train.loop — the demonstration-graph retriever's training curriculum.

No HDF5/suite dependency: every test builds a small, synthetic, fully in-memory graph (straight-line
toy demos through robot_ood.data.nodes.NodeTable.from_demos + robot_ood.data.edges.EdgeSet.build,
mirroring tests/test_retrieve.py's fixture style) so this runs fast and standalone in .venv-core.
coarsen=1 throughout (rather than the production default) sidesteps a real landmine unrelated to
this module: NodeTable.validate requires the table-wide node phase to reach >0.99, which needs
a raw length congruent to 1 (mod coarsen) to land a coarse group of size 1 on the final raw frame
(see tests/test_retrieve.py's N_FRAMES/COARSEN comment) — coarsen=1 makes every raw
frame its own node and removes that constraint entirely, which is all these tests need.
"""

from __future__ import annotations


import numpy as np
import pytest
import torch

from robot_ood.core.paths import default_paths

from robot_ood.core.config import GraphConfig
from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.data.store import ArtifactManifest
from robot_ood.models.retrieval.gnn import GraphRetrieverNet
from robot_ood.data.nodes import NodeTable
from robot_ood.train.types import QuerySpec, TrainGraphSpec
from robot_ood.models.retrieval.gnn import save_checkpoint
from robot_ood.train.loop import train_graph
from robot_ood.train.data import make_queries
from robot_ood.train.target import TargetBuilder
from robot_ood.train.metrics import evaluate

D = 4   # small synthetic joint dim -- any works, node_encode/edges don't care


# ======================================================================================================
# synthetic-graph fixtures
# ======================================================================================================
def _make_demo(task_id: int, seed: int, n: int, dim: int = D, base: np.ndarray | None = None) -> dict:
    """A straight-line synthetic demo (mirrors tests/test_retrieve.py's _make_demo): joint
    config drifts linearly along a random unit direction from an optional per-task base offset, so
    different tasks occupy well-separated regions of joint space and different demos of one task are
    still geometrically distinguishable (different random direction)."""
    rng = np.random.RandomState(seed)
    start = (base if base is not None else np.zeros(dim)) + rng.randn(dim) * 0.01
    direction = rng.randn(dim)
    direction /= np.linalg.norm(direction)
    t = np.arange(n, dtype=np.float32)
    q = (start[None, :] + 0.05 * t[:, None] * direction[None, :]).astype(np.float32)
    qdot = np.zeros_like(q)
    qdot[:-1] = q[1:] - q[:-1]
    qdot[-1] = qdot[-2] if n > 1 else 0.0
    grip = np.zeros(n, dtype=np.uint8)
    stage = np.zeros(n, dtype=np.int16)
    return {"q": q, "qdot": qdot, "grip": grip, "stage": stage, "task_id": task_id}


def _frame_floors(nodes: NodeTable) -> np.ndarray:
    """Every raw frame's distance to the nearest frame of another demo, by brute force."""
    raw_ptr = np.asarray(nodes.raw_ptr)
    q = np.asarray(nodes.q_raw, dtype=np.float64)
    owner = np.repeat(np.arange(len(raw_ptr) - 1), np.diff(raw_ptr))
    d = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=-1)
    d[owner[:, None] == owner[None, :]] = np.inf
    floors = d.min(axis=1)
    return floors[floors > 0.0]


def test_entry_perturbation_floor_is_measured_over_every_frame_and_refuses_a_flat_corpus():
    from robot_ood.data.geometry import entry_perturbation_floor

    shared = np.zeros(D)
    demos = [_make_demo(0, seed=s, n=20, dim=D, base=shared) for s in range(4)]
    for d in demos:
        d["q"][0] = shared.astype(np.float32)
    nodes = NodeTable.from_demos(demos, task_names=["t0"], coarsen=1)
    floor = entry_perturbation_floor(nodes)
    assert floor["p99"] > 0.0
    assert floor["p50"] == pytest.approx(float(np.percentile(_frame_floors(nodes), 50)), rel=1e-4)
    twin = _make_demo(0, seed=1, n=20, dim=D)
    flat = NodeTable.from_demos([twin, dict(twin)], task_names=["t0"], coarsen=1)
    with pytest.raises(ValueError, match="coincides"):
        entry_perturbation_floor(flat)


def _toy_graph(n_tasks: int = 2, demos_per_task: int = 6, length: int = 40, coarsen: int = 1,
               dim: int = D, seed: int = 0) -> tuple[NodeTable, EdgeSet]:
    rng = np.random.RandomState(seed)
    demos = []
    for ti in range(n_tasks):
        base = rng.randn(dim) * 2.0
        for di in range(demos_per_task):
            demos.append(_make_demo(ti, seed=1000 * ti + di + seed, n=length, dim=dim, base=base))
    nodes = NodeTable.from_demos(demos, task_names=[f"task{i}" for i in range(n_tasks)], coarsen=coarsen)
    edges = EdgeSet.build(nodes, k_sib=4, k_align=2, device="cpu")
    return nodes, edges


def _write_graph(tmp_path, nodes: NodeTable, edges: EdgeSet):
    d = tmp_path / "graph"
    d.mkdir(exist_ok=True)
    nodes.save(d)
    edges.save(d / schema.EDGES_NPZ)
    _write_segments(nodes, d)
    ArtifactManifest(d, suite="long").save()
    return d


def _write_segments(nodes: NodeTable, graph_dir) -> None:
    """The segment table graph_plan_rows reads to size the locator's policy channel."""
    from robot_ood.core.schemas import ACTION_DIM, SEGMENT_LEN, SegmentTable
    from robot_ood.data.segments import save_segments

    n_real = int(nodes.n_real)
    owner = np.asarray(nodes.owner[:n_real], dtype=np.int32)
    start = np.arange(n_real, dtype=np.int32)
    end = start + SEGMENT_LEN - 1
    keep = (end < n_real) & (owner[np.minimum(end, n_real - 1)] == owner)
    start, end = start[keep], end[keep]
    s = len(start)
    save_segments(SegmentTable(
        start_node=start, end_node=end, owner=owner[start],
        task_id=np.asarray(nodes.task_id[start], dtype=np.int16),
        phase_start=np.asarray(nodes.phase[start], dtype=np.float32),
        phase_end=np.asarray(nodes.phase[end], dtype=np.float32),
        action_chunk=np.zeros((s, SEGMENT_LEN, ACTION_DIM), dtype=np.float32),
        ee_start=np.zeros((s, 6), dtype=np.float32),
        ee_end=np.zeros((s, 6), dtype=np.float32),
        node_to_segment=np.zeros(len(nodes), dtype=np.int32),
    ), graph_dir)


def _write_locator_artifacts(nodes: NodeTable, graph_dir) -> None:
    """The policy cache, action scale and task embeddings a locator run reads beside the graph."""
    from robot_ood.core.action import ActionScale
    from robot_ood.models.fusion.target import ACTIONS_NPZ
    from robot_ood.models.language.encoder import EMBED_DIM, N_AUGMENT, TaskEmbeddings

    n_raw = int(nodes.n_raw)
    rng = np.random.RandomState(0)
    a_pi_raw = rng.randn(n_raw, 7).astype(np.float32)
    a_pi_raw[:, 6] = np.sign(rng.randn(n_raw)).astype(np.float32)
    np.savez(graph_dir / ACTIONS_NPZ, a_pi_raw=a_pi_raw,
             raw_ptr=np.asarray(nodes.raw_ptr, dtype=np.int64))
    ActionScale(pos=0.0116, rot=0.1092, r2_pos=1.0, r2_rot=1.0).to_json(
        graph_dir / schema.ACTION_SCALE_JSON)

    names = tuple(nodes.task_names)
    n = len(names)
    unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-8)
    e_task = unit(rng.randn(n, EMBED_DIM)).astype(np.float32)
    e_aug = unit(rng.randn(n * N_AUGMENT, EMBED_DIM)).astype(np.float32)
    table = TaskEmbeddings(
        e_task=e_task, mean=rng.randn(EMBED_DIM).astype(np.float32),
        model_id="toy", task_names=names, e_aug=e_aug,
        aug_ptr=np.arange(0, n * N_AUGMENT + 1, N_AUGMENT, dtype=np.int64),
        head_w=(4.0 * e_task.T).astype(np.float32), head_b=np.zeros(n, dtype=np.float32),
        temperature=1.0, p_task=np.zeros((0, 0), dtype=np.float32), p_aug=np.zeros((0, 0), dtype=np.float32),
    )
    TaskEmbeddings(**{**table.__dict__, "p_task": table.posterior(e_task),
                      "p_aug": table.posterior(e_aug)}).save(graph_dir)


# ======================================================================================================
# make_queries
# ======================================================================================================
def test_make_queries_shapes_dtypes():
    nodes, _edges = _toy_graph(n_tasks=2, demos_per_task=4, length=40)
    rng = np.random.RandomState(0)
    owners = np.arange(nodes.n_demos)
    out = make_queries(nodes, rng, n=25, owners=owners, spec=QuerySpec(window=8, advance=4))

    assert out["q_hist"].shape == (25, 8, nodes.dim)
    assert out["qdot_hist"].shape == (25, 8, nodes.dim)
    assert out["grip_hist"].shape == (25, 8)
    assert out["q_hist"].dtype == np.float32
    assert out["qdot_hist"].dtype == np.float32
    assert out["tgt_node"].dtype == np.int64
    assert out["src_node"].dtype == np.int64
    assert out["src_owner"].dtype == np.int32
    assert not np.any(np.isnan(out["q_hist"]))


def test_make_queries_targets_ahead_same_demo():
    nodes, _edges = _toy_graph(n_tasks=2, demos_per_task=4, length=40)
    rng = np.random.RandomState(1)
    owners = np.arange(nodes.n_demos)
    out = make_queries(nodes, rng, n=80, owners=owners, spec=QuerySpec(window=8, advance=4))

    for i in range(80):
        src_v, tgt_v = int(out["src_node"][i]), int(out["tgt_node"][i])
        assert nodes.owner[src_v] == nodes.owner[tgt_v] == out["src_owner"][i]
        assert nodes.t_idx[tgt_v] >= nodes.t_idx[src_v]
        assert nodes.t_raw[tgt_v] >= nodes.t_raw[src_v]


def test_make_queries_restricted_to_requested_owners():
    nodes, _edges = _toy_graph(n_tasks=2, demos_per_task=5, length=40)
    rng = np.random.RandomState(2)
    owners = np.array([0, 2, 4])
    out = make_queries(nodes, rng, n=60, owners=owners, spec=QuerySpec(window=8, advance=4))

    assert set(out["src_owner"].tolist()) <= set(owners.tolist())
    assert set(nodes.owner[out["src_node"]].tolist()) <= set(owners.tolist())
    assert set(nodes.owner[out["tgt_node"]].tolist()) <= set(owners.tolist())


# ======================================================================================================
# make_queries: variable-length history + episode-start (entry) query class
# ======================================================================================================
def test_make_queries_variable_window_length_matches_edge_pad():
    """The true history length W varies 1..window per TRAVERSAL query (entry_frac=0 isolates it), and
    the padding is byte-identical to robot_ood.data.geometry.edge_pad_hist -- the single source of truth
    shared with QueryPreprocessor.build."""
    from robot_ood.data.geometry import edge_pad_hist

    nodes, _edges = _toy_graph(n_tasks=1, demos_per_task=3, length=60)
    rng = np.random.RandomState(0)
    owners = np.arange(nodes.n_demos)
    window = 8
    out = make_queries(nodes, rng, n=300, owners=owners, spec=QuerySpec(window=window, advance=4, entry_frac=0.0))

    assert np.all(out["is_clean"])
    eff_len = np.array([len(np.unique(out["q_hist"][i], axis=0)) for i in range(300)])
    assert eff_len.min() == 1, "no query drew the shortest (W=1) history"
    assert eff_len.max() == window, "no query drew the full (W=window) history"

    # padding parity: for a query whose effective length is w, edge_pad_hist(the last w raw rows, window)
    # must reproduce q_hist[i] exactly.
    for i in range(300):
        w = int(eff_len[i])
        raw_tail = out["q_hist"][i, -w:]
        padded, _ = edge_pad_hist(raw_tail, window)
        assert np.array_equal(padded, out["q_hist"][i])


def test_make_queries_entry_class_targets_same_demo_opening():
    """Entry (episode-start) queries: W ~ Uniform(1, window) REAL consecutive frames from the demo's
    own start (a constant offset added to all of them), target = advance raw frames past the window's
    OWN (unperturbed) end, on the SAME demo -- is_clean=False (provenance)."""
    nodes, _edges = _toy_graph(n_tasks=2, demos_per_task=5, length=40)
    rng = np.random.RandomState(3)
    owners = np.arange(nodes.n_demos)
    window = 8
    out = make_queries(nodes, rng, n=200, owners=owners, spec=QuerySpec(window=window, advance=4, entry_frac=1.0))

    assert not np.any(out["is_clean"])
    # window length is ALSO variable for entry queries now (no longer pinned at W=1) -- some queries
    # must draw the shortest and some the longest history, exactly like TRAVERSAL.
    eff_len = np.array([len(np.unique(out["q_hist"][i], axis=0)) for i in range(200)])
    assert eff_len.min() == 1, "no entry query drew the shortest (W=1) history"
    assert eff_len.max() == window, "no entry query drew the full (W=window) history"

    for i in range(200):
        owner = int(out["src_owner"][i])
        src_v, tgt_v = int(out["src_node"][i]), int(out["tgt_node"][i])
        lo = int(nodes.raw_ptr[owner])
        assert nodes.owner[src_v] == nodes.owner[tgt_v] == owner
        # src sits within the demo's OWN opening window raw frames (the window is drawn from t=0)
        assert lo <= nodes.t_raw[src_v] <= lo + window - 1 + schema.COARSEN
        assert nodes.t_raw[tgt_v] >= nodes.t_raw[src_v]                 # target is ahead-or-tied


def test_make_queries_entry_and_traversal_vh_distributions_overlap():
    """An ENTRY window is a real multi-frame window under a constant offset, so its velocity
    distribution overlaps TRAVERSAL's -- "is this window flat" must not separate the two classes.
    """
    nodes, _edges = _toy_graph(n_tasks=2, demos_per_task=8, length=60)
    owners = np.arange(nodes.n_demos)

    rng_e = np.random.RandomState(11)
    entry = make_queries(nodes, rng_e, n=300, owners=owners, spec=QuerySpec(window=8, advance=4, entry_frac=1.0))
    rng_t = np.random.RandomState(12)
    trav = make_queries(nodes, rng_t, n=300, owners=owners, spec=QuerySpec(window=8, advance=4, entry_frac=0.0))

    entry_norms = np.linalg.norm(entry["qdot_hist"], axis=-1).mean(axis=-1)   # [n] mean ||vh|| per query
    trav_norms = np.linalg.norm(trav["qdot_hist"], axis=-1).mean(axis=-1)

    assert entry_norms.max() > 1e-8, "entry vh collapsed back to all-zero -- the shortcut is back"

    # IQRs overlap: NOT the old regime where entry sits at a single point (0) strictly below traversal.
    e_lo, e_hi = np.percentile(entry_norms, [25, 75])
    t_lo, t_hi = np.percentile(trav_norms, [25, 75])
    assert e_lo <= t_hi and t_lo <= e_hi, (
        f"entry vh IQR=({e_lo:.4f},{e_hi:.4f}) and traversal vh IQR=({t_lo:.4f},{t_hi:.4f}) do not "
        "overlap -- window flatness is still a usable class tell"
    )


def test_make_queriesentry_perturbation_magnitude_matches_documented_range():
    """The entry perturbation radius lies in entry_sigma_range: from a quarter of the corpus's OWN
    leave-one-demo-out floor to the benchmark's robot-init magnitude -- measured here, not asserted
    against a hand-picked constant.

    Uses a SMALL-magnitude toy graph (unlike _toy_graph's default base*2.0 scale) so every
    demo start sits well inside the real Panda joint limits entry_perturb clips to -- with
    _toy_graph's normal scale some synthetic demo starts already sit OUTSIDE those limits (an
    artifact of "any magnitude works" toy data, unrelated to a real robot corpus), which would make
    clipping, not sampling, determine the observed radius and defeat this test's purpose.
    """
    from robot_ood.train.drift import PANDA_JOINT_HI, PANDA_JOINT_LO
    from robot_ood.data.geometry import entry_perturbation_floor
    from robot_ood.train.data import entry_sigma_range

    # Panda joint 4 (index 3) has an ASYMMETRIC range that excludes 0 ([-3.07, -0.07] rad) -- centering
    # every toy demo at each joint's OWN mid-range (rather than 0) keeps this small-amplitude synthetic
    # graph safely inside every real limit entry_perturb clips to, so the observed radius reflects the
    # sampling distribution rather than an accidental per-joint clip.
    center = ((PANDA_JOINT_LO[:D] + PANDA_JOINT_HI[:D]) / 2.0).astype(np.float32)
    rng0 = np.random.RandomState(7)
    demos = []
    for ti in range(2):
        base = rng0.randn(D) * 0.05
        for di in range(6):
            demos.append(_make_demo(ti, seed=1000 * ti + di, n=40, dim=D, base=base))
    for d in demos:                                   # shrink the within-demo drift too (default 0.05*t)
        d["q"] = d["q"] * 0.05 + center
        d["qdot"] *= 0.05
    nodes = NodeTable.from_demos(demos, task_names=["t0", "t1"], coarsen=1)

    floor = entry_perturbation_floor(nodes)
    lo, hi = entry_sigma_range(nodes)
    assert 0.0 < lo < hi
    assert floor["p99"] == pytest.approx(float(np.percentile(_frame_floors(nodes), 99)), rel=1e-4)

    rng = np.random.RandomState(4)
    owners = np.arange(nodes.n_demos)
    window = 8
    raw_ptr = np.asarray(nodes.raw_ptr)
    q_raw = np.asarray(nodes.q_raw, dtype=np.float64)
    n = 200
    out = make_queries(nodes, rng, n=n, owners=owners, spec=QuerySpec(window=window, advance=4, entry_frac=1.0))

    # the perturbation is now a CONSTANT offset over a REAL multi-frame window (module docstring,
    # "EPISODE-START AS A TRAINED CASE"), so recovering the sampled radius means reading the window's
    # OWN earliest row (which corresponds to raw frame lo, perturbed by delta) rather than always
    # q_hist[i, -1] -- the earliest real row sits at index window - w_use, w_use recovered the
    # same way test_make_queries_variable_window_length_matches_edge_pad does (unique-row count).
    radii = np.empty(n)
    for i in range(n):
        owner = int(out["src_owner"][i])
        q0 = q_raw[raw_ptr[owner]]
        w_use = len(np.unique(out["q_hist"][i], axis=0))
        q0_pert = out["q_hist"][i, window - w_use].astype(np.float64)
        radii[i] = np.linalg.norm(q0_pert - q0)

    assert radii.min() >= lo - 1e-6
    assert radii.max() <= hi + 1e-6
    # spans a real range, not collapsed to one corner by clipping
    assert radii.max() - radii.min() > 0.1 * (hi - lo)


# ======================================================================================================
# make_queries: ADVANCE=0 and the DRIFT query class
# ======================================================================================================
class _CountingRandomState:
    """np.random.RandomState proxy that counts rand() -- the class-mix draws -- and forwards the rest."""

    def __init__(self, seed: int):
        self._rng = np.random.RandomState(seed)
        self.n_rand = 0

    def rand(self, *args, **kwargs):
        self.n_rand += 1
        return self._rng.rand(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._rng, name)


def test_make_queries_drift_frac_zero_is_bit_identical_noop():
    """drift_frac=0.0 must be opt-in to the point of invisibility: identical arrays at the same seed,
    and -- the part an equality check alone would not catch -- NOT ONE rng.rand() spent on the drift
    check, so the pre-DRIFT stream is reproduced draw for draw (the same property
    tests/test_entry_static.py pins for entry_static_frac)."""
    nodes, _edges = _toy_graph(n_tasks=2, demos_per_task=4, length=40)
    owners = np.arange(nodes.n_demos)
    n = 200
    spec = dict(window=8, entry_frac=1 / 3, entry_static_frac=0.15)

    off = make_queries(nodes, _CountingRandomState(0), n=n, owners=owners,
                       spec=QuerySpec(drift_frac=0.0, **spec))
    base = make_queries(nodes, np.random.RandomState(0), n=n, owners=owners, spec=QuerySpec(**spec))
    for key in base:
        assert np.array_equal(off[key], base[key]), key
    assert not off["is_drift"].any()

    # one rand() per row for the static check, one more for every row the static check declined --
    # and nothing else.
    rng_off = _CountingRandomState(0)
    off = make_queries(nodes, rng_off, n=n, owners=owners, spec=QuerySpec(drift_frac=0.0, **spec))
    n_static = int(off["is_entry_static"].sum())
    assert rng_off.n_rand == n + (n - n_static)

    # with the class on, exactly one further rand() per row that is neither static nor entry
    rng_on = _CountingRandomState(0)
    on = make_queries(nodes, rng_on, n=n, owners=owners, spec=QuerySpec(drift_frac=0.5, **spec))
    n_static, n_drift, n_trav = (int(on[k].sum()) for k in ("is_entry_static", "is_drift", "is_clean"))
    assert rng_on.n_rand == n + (n - n_static) + (n_drift + n_trav)
    assert n_drift > 0


def test_make_queries_drift_rows_are_a_clean_prefix_then_a_ramped_tail():
    """The property no other class has: a breakpoint INSIDE the window. Rows before it are the demo's
    own raw frames bit-for-bit; rows after it deviate by an offset that RAMPS -- equal increments per
    row, reaching the full radius at T -- rather than stepping, which would put the whole radius into
    one finite-difference step. Uses demos centred in the real Panda joint limits so entry_perturb's
    clip is not what determines the offset (same reason as
    test_make_queriesentry_perturbation_magnitude_matches_documented_range)."""
    from robot_ood.data.geometry import edge_pad_hist
    from robot_ood.train.drift import PANDA_JOINT_HI, PANDA_JOINT_LO

    center = ((PANDA_JOINT_LO[:D] + PANDA_JOINT_HI[:D]) / 2.0).astype(np.float32)
    demos = [_make_demo(0, seed=di, n=60, dim=D) for di in range(6)]
    for d in demos:
        d["q"] = d["q"] * 0.05 + center
        d["qdot"] *= 0.05
    nodes = NodeTable.from_demos(demos, task_names=["t0"], coarsen=1)
    q_raw = np.asarray(nodes.q_raw, dtype=np.float64)

    window, n = 8, 200
    out = make_queries(nodes, np.random.RandomState(5), n=n, owners=np.arange(nodes.n_demos),
                       spec=QuerySpec(window=window, entry_frac=0.0, drift_frac=1.0))
    assert np.all(out["is_drift"]) and not np.any(out["is_clean"])

    for i in range(n):
        t_end = int(nodes.t_raw[out["src_node"][i]])
        w_use = len(np.unique(out["q_hist"][i], axis=0))          # every row of a drift window differs
        clean = q_raw[t_end - w_use + 1 : t_end + 1].astype(np.float32)
        ref = edge_pad_hist(clean, window)[0].astype(np.float64)
        deviation = np.linalg.norm(out["q_hist"][i].astype(np.float64) - ref, axis=1)

        breakpoint_row = int((deviation > 0).argmax())
        assert 0 < breakpoint_row <= window - 2, (i, breakpoint_row)   # >=1 clean and >=2 drifted rows
        assert np.all(deviation[:breakpoint_row] == 0.0), i
        step = np.diff(deviation[breakpoint_row - 1 :])
        assert np.all(step > 0), i
        assert np.allclose(step, step[0], rtol=0.05), (i, step)        # a ramp, not a step


def test_make_queries_advance_zero_labels_the_windows_own_end():
    """ADVANCE=0: the head answers "where am I NOW", so tgt_node IS the window's own end node for
    every class -- the deployed transition kernel owns the lookahead."""
    nodes, _edges = _toy_graph(n_tasks=2, demos_per_task=4, length=40)
    owners = np.arange(nodes.n_demos)
    spec = dict(window=8, entry_static_frac=0.15, drift_frac=0.5)
    assert schema.ADVANCE == 0

    now = make_queries(nodes, np.random.RandomState(6), n=150, owners=owners, spec=QuerySpec(**spec))
    assert np.array_equal(now["src_node"], now["tgt_node"])
    assert now["is_drift"].any() and now["is_entry_static"].any() and now["is_clean"].any()

    ahead = make_queries(nodes, np.random.RandomState(6), n=150, owners=owners,
                          spec=QuerySpec(advance=4, **spec))
    assert not np.array_equal(ahead["src_node"], ahead["tgt_node"])


def test_finite_diff_vel_shared_by_make_queries_and_retrieve(tmp_path):
    """Divergence 6 (module docstring, 'SHARED TRAIN/DEPLOY PLUMBING'): the query-construction path
    (make_queries, via robot_ood.data.geometry.finite_diff_vel) and
    robot_ood.runtime.retrieve.GraphRetriever.retrieve's own live path must compute the IDENTICAL
    velocity for the IDENTICAL raw window. Before the shared helper existed, train.py
    edge-extrapolated row 0 while retrieve() zero-padded it, so the SAME window produced two
    DIFFERENT query embeddings depending on whether it was seen at train time or live.
    """
    from robot_ood.runtime.retrieve import GraphRetriever
    from robot_ood.data.geometry import finite_diff_vel

    nodes, edges = _toy_graph(n_tasks=1, demos_per_task=2, length=30)
    net = _small_net(nodes)
    d = _write_graph(tmp_path, nodes, edges)
    save_checkpoint(net, nodes, edges, d / schema.HEAD_NPZ)
    cfg = GraphConfig(hidden=16, layers=2, seed_topk=32, device="cpu")
    retriever = GraphRetriever.load(d, cfg=cfg)

    window = cfg.hist
    q_window = nodes.q_raw[5 : 5 + window].astype(np.float64)      # a real, non-padded raw window
    assert q_window.shape[0] == window

    v_shared = finite_diff_vel(q_window.astype(np.float32))                        # (1) train's path
    _w = retriever.preprocessor.build(q_window, None)                              # (2) deploy's path
    q_pad, padded = _w.q, _w.padded
    assert not padded
    v_deploy = finite_diff_vel(q_pad)

    assert np.array_equal(v_shared.astype(np.float64), v_deploy.astype(np.float64))

    qh = torch.as_tensor(q_pad, dtype=torch.float32)
    gh = torch.zeros(window)
    e_q_train = retriever.net.enc(qh, torch.as_tensor(v_shared, dtype=torch.float32), gh)
    e_q_deploy = retriever.net.enc(qh, torch.as_tensor(v_deploy, dtype=torch.float32), gh)
    assert torch.equal(e_q_train, e_q_deploy)

    # regression guard: the RETIRED zero-padded convention must actually DISAGREE with the shared one,
    # proving this is a real parity check and not a tautology that would pass no matter what.
    v_old_zero_pad = torch.zeros_like(qh)
    v_old_zero_pad[1:] = qh[1:] - qh[:-1]
    e_q_old = retriever.net.enc(qh, v_old_zero_pad, gh)
    assert not torch.allclose(e_q_train, e_q_old)



# ======================================================================================================
# TargetBuilder and check_loss
# ======================================================================================================
def test_target_sums_to_one_and_peaks_at_the_true_node():
    nodes, _edges = _toy_graph(n_tasks=2, demos_per_task=3, length=20)
    true_node = 5
    target = TargetBuilder(nodes)(true_node)
    assert np.isclose(float(target.weight.sum()), 1.0, atol=1e-5)
    assert int(target.nodes[np.argmax(target.weight)]) == true_node
    assert true_node in target


def test_target_stays_inside_the_phase_and_joint_band():
    from robot_ood.core.constants import PHASE_TOL

    nodes, _edges = _toy_graph(n_tasks=2, demos_per_task=3, length=20)
    true_node = 5
    target = TargetBuilder(nodes, joint_tol=0.2)(true_node)
    assert np.all(np.abs(nodes.phase[target.nodes] - nodes.phase[true_node]) <= PHASE_TOL + 1e-6)
    dist = np.linalg.norm(nodes.q[target.nodes].astype(np.float64) - nodes.q[true_node], axis=1)
    assert np.all(dist <= 0.2 + 1e-6)
    assert np.all(nodes.task_id[target.nodes] == nodes.task_id[true_node])


def test_target_spans_other_strands_of_the_same_task():
    nodes, _edges = _toy_graph(n_tasks=1, demos_per_task=4, length=30)
    target = TargetBuilder(nodes, joint_tol=1.0)(15)
    assert len(set(nodes.owner[target.nodes].tolist())) > 1


def test_target_without_an_instruction_opens_to_every_task_and_boosts_the_true_one():
    from robot_ood.core.constants import TRUE_TASK_BOOST

    demos = [_make_demo(task_id=t, seed=t, n=20) for t in range(2)]
    nodes = NodeTable.from_demos(demos, task_names=["a", "b"], coarsen=1)
    builder = TargetBuilder(nodes, joint_tol=10.0)
    true_node = 5
    with_instr = builder(true_node, has_instr=True)
    without = builder(true_node, has_instr=False)
    assert np.all(nodes.task_id[with_instr.nodes] == 0)
    assert set(nodes.task_id[without.nodes].tolist()) == {0, 1}
    w = dict(zip(without.nodes.tolist(), without.weight.tolist()))
    peer_other = [v for v in without.nodes if nodes.task_id[v] == 1 and nodes.t_idx[v] == nodes.t_idx[true_node]]
    assert peer_other
    d_same = np.linalg.norm(nodes.q[true_node].astype(np.float64) - nodes.q[true_node])
    assert w[true_node] > w[int(peer_other[0])]


def _query_batch_for(nodes, cfg, true_node: int):
    from robot_ood.train.types import QueryBatch

    window = cfg.hist
    q_hist = np.repeat(nodes.q_raw[int(nodes.t_raw[true_node])][None, :], window, axis=0).astype(np.float32)
    qh = torch.as_tensor(q_hist)[None]
    return QueryBatch(qh=qh, vh=torch.zeros_like(qh), gh=torch.zeros(1, window),
                      tgt=np.array([true_node]))


def test_check_loss_reaches_the_message_passing_trunk():
    from robot_ood.train.loss import QueryLossComputer
    from robot_ood.train.tensors import prepare_tensors

    nodes, edges = _toy_graph(n_tasks=1, demos_per_task=2, length=30)
    net = _small_net(nodes)
    cfg = GraphConfig(hidden=16, layers=2, seed_topk=32, device="cpu")
    gt = prepare_tensors(nodes, edges, "cpu", cfg=cfg)

    net.zero_grad()
    loss = QueryLossComputer(net, cfg)(gt, _query_batch_for(nodes, cfg, int(nodes.demo_nodes(0)[5])))
    assert torch.isfinite(loss)
    loss.backward()
    for mod_name in ("rel", "node", "upd", "out", "enc"):
        grads = [p.grad for p in getattr(net, mod_name).parameters()]
        assert all(g is not None for g in grads), f"{mod_name}: no gradient reached at all"
        assert any(float(g.abs().sum()) > 0 for g in grads), f"{mod_name}: gradient is all-zero"


def test_check_loss_is_zero_when_no_target_node_is_reached():
    from robot_ood.train.loss import check_loss
    from robot_ood.train.target import Target

    logit = torch.tensor([0.5, -1.0, float("-inf"), float("-inf")])
    reached = torch.isfinite(logit)
    target = Target(nodes=np.array([2, 3]), weight=np.array([0.5, 0.5], dtype=np.float32))
    assert float(check_loss(logit, target, reached)) == 0.0


def test_check_loss_prefers_the_target_over_the_negatives():
    from robot_ood.train.loss import check_loss
    from robot_ood.train.target import Target

    target = Target(nodes=np.array([0]), weight=np.array([1.0], dtype=np.float32))
    reached = torch.ones(3, dtype=torch.bool)
    good = check_loss(torch.tensor([4.0, -4.0, -4.0]), target, reached)
    bad = check_loss(torch.tensor([-4.0, 4.0, 4.0]), target, reached)
    assert float(good) < float(bad)

# ======================================================================================================
# train_graph
# ======================================================================================================
def test_train_graph_survives_zero_reached_pathological_seeding(tmp_path):
    """Integration-level companion to
    test_query_loss_handles_zero_reached_via_fallback_anchor: the whole curriculum (stage 1, the
    only stage left) must run to completion, with finite losses throughout, even when EVERY query
    seeds zero nodes.
    """
    nodes, edges = _toy_graph(n_tasks=2, demos_per_task=4, length=28, dim=7)
    d = _write_graph(tmp_path, nodes, edges)
    _write_locator_artifacts(nodes, d)
    cfg = GraphConfig(hidden=12, layers=1, seed_topk=0, device="cpu")

    result = train_graph(
        d, TrainGraphSpec(stages=(1,), epochs=2, seed=0, device="cpu", save=False, cfg=cfg,
                          n_query=40, n_eval=10, task="locator"),
    )
    for entry in result.history["stage1"]:
        if entry["epoch"] != "final":
            assert np.isfinite(entry["loss"])


def test_train_graph_determinism_same_seed(tmp_path):
    """Same seed -> bit-identical final weights. Forced single-threaded: GraphRetrieverNet's message
    passing (a sibling module, robot_ood.models.retrieval.gnn) uses index_add_ scatter-accumulation, whose summation
    ORDER (and hence exact floating-point result) is a known PyTorch non-determinism under multi-
    threaded CPU execution -- not a bug in this module, just a precondition for exact reproducibility
    that RandomState(seed)/torch.manual_seed(seed) alone do not cover.
    """
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        nodes, edges = _toy_graph(n_tasks=2, demos_per_task=5, length=32, dim=7)
        d = _write_graph(tmp_path, nodes, edges)
        _write_locator_artifacts(nodes, d)
        cfg = GraphConfig(hidden=12, layers=2, seed_topk=32, device="cpu")

        spec = TrainGraphSpec(
            stages=(1,), epochs=2, seed=0, device="cpu", save=False, cfg=cfg, n_query=60, n_eval=10,
            task="locator",
        )
        r1 = train_graph(d, spec)
        r2 = train_graph(d, spec)
    finally:
        torch.set_num_threads(prev_threads)

    sd1, sd2 = r1.net.state_dict(), r2.net.state_dict()
    assert set(sd1) == set(sd2)
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), k


# ======================================================================================================
# checkpoint round-trip
# ======================================================================================================
def _small_net(nodes: NodeTable) -> GraphRetrieverNet:
    net = GraphRetrieverNet(dim=nodes.dim, hidden=16, layers=2, n_rel=schema.N_RELATIONS, agg="sum")
    mean, std = nodes.feature_stats()
    net.set_stats(mean, std)
    net.eval()
    return net


def test_checkpoint_roundtrip(tmp_path):
    """save_checkpoint writes a state dict GraphRetrieverNet.load_npz can read back
    bit-for-bit -- the graph_hash STAMP itself (and refusing a mismatched one) is
    robot_ood.runtime.retrieve.GraphRetriever.load's job, tested alongside that class."""
    nodes, edges = _toy_graph(n_tasks=2, demos_per_task=4, length=30)
    net = _small_net(nodes)

    path = save_checkpoint(net, nodes, edges, tmp_path / "g_head.npz")
    loaded = GraphRetrieverNet.load_npz(path)

    sd1, sd2 = net.state_dict(), loaded.state_dict()
    assert set(sd1) == set(sd2)
    for k in sd1:
        assert torch.allclose(sd1[k], sd2[k])


# ======================================================================================================
# evaluate
# ======================================================================================================
def test_evaluate_returns_documented_keys():
    nodes, edges = _toy_graph(n_tasks=2, demos_per_task=6, length=40)
    net = _small_net(nodes)

    rng = np.random.RandomState(0)
    owners = np.arange(nodes.n_demos)
    queries = make_queries(nodes, rng, n=15, owners=owners, spec=QuerySpec())

    cfg = GraphConfig(hidden=16, layers=2, seed_topk=32, device="cpu")
    out = evaluate(net, nodes, edges, queries, cfg=cfg, device="cpu")

    for key in (
        "top1", "top1_target", "top10", "mrr", "when_mae_steps", "when_mae_phase", "where_rmse",
        "own_task_frac", "move_dist_mean", "move_dist_median",
        "when_phase_argmax_corr", "when_phase_argmax_mae", "when_phase_expect_corr", "when_phase_expect_mae",
    ):
        assert key in out, key
    assert np.isfinite(out["move_dist_mean"])
    assert np.isfinite(out["when_phase_expect_mae"])


def test_evaluate_transfer_across_differently_sized_graph():
    """The inductive claim, made executable: one trained net runs unmodified on a graph built from a
    different suite with a different node count."""
    nodes_a, _edges_a = _toy_graph(n_tasks=2, demos_per_task=4, length=30, seed=1)
    nodes_b, edges_b = _toy_graph(n_tasks=3, demos_per_task=7, length=50, seed=2)
    assert len(nodes_a) != len(nodes_b)
    assert nodes_a.dim == nodes_b.dim

    net = _small_net(nodes_a)

    rng = np.random.RandomState(0)
    queries_b = make_queries(nodes_b, rng, n=10, owners=np.arange(nodes_b.n_demos))
    cfg = GraphConfig(hidden=16, layers=2, seed_topk=32, device="cpu")

    out = evaluate(net, nodes_b, edges_b, queries_b, cfg=cfg, device="cpu")
    assert "top1" in out and np.isfinite(out["top1"])


# ==================================================================================================
# STAGE 2 -- the chunk-blend target (robot_ood.core.kinematics, robot_ood.models.fusion.target) and the loss over it.
# ==================================================================================================
# Resolved the way every other consumer resolves it -- Paths.graph, which honours GR_GRAPH_DIR.
# This used to be hardcoded to outputs/long/latest/artifacts, which is both the OLD layout (the
# shipping one is outputs/<suite>/artifacts, see docs/artifacts.md) and a path
# robot_ood.data.store.assert_not_latest refuses outright: `latest` is repointed by every graph build, so
# a test pinned to it silently changes corpus underneath itself.
ARTIFACT_DIR = default_paths().graph("long")
skip_no_corpus = pytest.mark.skipif(
    not (ARTIFACT_DIR / "g_nodes.npz").exists(),
    reason=f"needs the recorded corpus at {ARTIFACT_DIR} to check FK against stored obs/ee_*",
)
# A fresh `python -m robot_ood.cli build graph` writes only the graph itself. The C0 calibration arrives at
# step 5 of the fourteen (docs/SETUP.md §5), so a half-built directory must SKIP here rather than
# raise FileNotFoundError out of the test body.
skip_no_action_scale = pytest.mark.skipif(
    not (ARTIFACT_DIR / "action_scale.json").exists(),
    reason=f"needs {ARTIFACT_DIR}/action_scale.json -- run `python -m robot_ood.cli build calibrate --suite long`",
)


def _real_nodes() -> NodeTable:
    return NodeTable.load(ARTIFACT_DIR)


@skip_no_corpus
def test_forward_kinematics_round_trips_the_recorded_end_effector_pose():
    """The FK model behind ee_now, checked against the poses LIBERO actually recorded.

    Three claims, because three different things could be wrong. (1) The absolute position model
    ee_pos = base_t + fk_pos + R @ tool_d holds, with tool_d the rigid grip-site offset -- fitted
    per TASK, because base_t is a property of the SCENE and not of the robot: it is
    [-0.660, 0, 0.912] on the kitchen scenes, [-0.510, 0, 0.420] on the living-room ones and
    [-0.750, 0, 0.912] on the study one, so one pooled fit lands 26 cm out. (2) The absolute
    orientation reproduces obs/ee_ori in ITS unnormalized, unwrapped convention, not merely up to
    that convention. (3) The claim NodeTable.ee_base rests on: world minus base is a PURE
    TRANSLATION, constant within a scene and large across scenes. If it were not, moving the
    reference segment into the base frame would rotate it, and the action deltas built from it
    would no longer be the world-frame deltas the policy's chunk is stated in.
    """
    pytest.importorskip("pytorch_kinematics")
    from robot_ood.core.action import matrix_to_rotvec, rotvec_to_matrix
    from robot_ood.core.kinematics import PandaKinematics, matrix_to_ee_rotvec

    nodes = _real_nodes()
    kin = PandaKinematics()
    demo_task = np.array([int(nodes.task_id[nodes.demo_ptr[j]]) for j in range(nodes.n_demos)])
    # Several demos per task, not one: base_t and R @ tool_d are only separable across a spread of
    # wrist orientations, and a single demo does not span enough of them to condition the fit.
    demos = [int(j) for t in np.unique(demo_task) for j in np.flatnonzero(demo_task == t)[:5]]
    frames = np.concatenate(
        [np.arange(nodes.raw_ptr[j], nodes.raw_ptr[j + 1]) for j in demos]
    )
    pose = kin.flange(np.asarray(nodes.q_raw, dtype=np.float64)[frames])
    stored = np.asarray(nodes.ee_raw, dtype=np.float64)[frames]

    # (1) absolute position, one [I | R] least-squares fit per task.
    owner = np.searchsorted(nodes.raw_ptr, frames, side="right") - 1
    for t in np.unique(demo_task):
        rows = np.isin(owner, np.flatnonzero(demo_task == t))
        flange = pose.position[rows] - np.einsum("nij,j->ni", pose.rotation[rows], kin.tool_offset)
        design = np.concatenate(
            [np.tile(np.eye(3), (int(rows.sum()), 1, 1)), pose.rotation[rows]], axis=2
        ).reshape(-1, 6)
        target = (stored[rows, :3] - flange).reshape(-1)
        fit, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = np.linalg.norm((design @ fit - target).reshape(-1, 3), axis=1)
        assert fit[3:] == pytest.approx(kin.tool_offset, abs=5e-4), (t, fit)
        assert np.sqrt((residual ** 2).mean()) < 5e-4, (t, residual.max())

    # (2) absolute orientation, in obs/ee_ori's own convention.
    ori_err = np.linalg.norm(matrix_to_ee_rotvec(pose.rotation) - stored[:, 3:], axis=1)
    assert np.sqrt((ori_err ** 2).mean()) < 3e-3
    assert ori_err.max() < 3e-2

    # (3) world minus base, per scene. The position offset is a constant within a task and the
    # rotation between the two frames is the identity, so the offset drops out of every DIFFERENCE
    # the tracking chunk takes -- but only once both operands are on the same side of it.
    base = kin.pose(np.asarray(nodes.q_raw, dtype=np.float64)[frames])
    rot_err = np.linalg.norm(
        matrix_to_rotvec(
            rotvec_to_matrix(base[:, 3:]) @ np.swapaxes(rotvec_to_matrix(stored[:, 3:]), -1, -2)
        ),
        axis=1,
    )
    assert np.sqrt((rot_err ** 2).mean()) < 3e-3 and rot_err.max() < 3e-2
    offsets = []
    for t in np.unique(demo_task):
        rows = np.isin(owner, np.flatnonzero(demo_task == t))
        delta = stored[rows, :3] - base[rows, :3]
        assert delta.std(axis=0).max() < 5e-4, (t, delta.std(axis=0))
        offsets.append(delta.mean(axis=0))
    spread = np.linalg.norm(np.asarray(offsets)[:, None] - np.asarray(offsets)[None, :], axis=-1)
    assert spread.max() > 0.4, spread          # the teleport ee_base exists to remove


@skip_no_corpus
@skip_no_action_scale
def test_a_corr_on_an_unperturbed_window_is_the_pose_derived_demo_track():
    """On a clean window the target must reduce to the demo's OWN pose-derived track.

    a_corr is EeTrack.chunk(demo segment at the label node, the window's current pose), and on a
    TRAVERSAL row that current pose is a frame the corpus recorded. So a_corr has to equal the same
    chunk built straight off that frame's own base-frame pose -- if it does not, ee_now has drifted
    off the frame the reference segment lives in and every row of the chunk is measured against the
    wrong origin.
    """
    pytest.importorskip("pytorch_kinematics")
    from robot_ood.core.action import ACTION_LIMIT, ActionScale
    from robot_ood.runtime.ee_track import EeTrack
    from robot_ood.core.kinematics import PandaKinematics
    from robot_ood.models.fusion.target import ChunkTargetBuilder, PolicyActionCache

    nodes = _real_nodes()
    scale = ActionScale.from_json(ARTIFACT_DIR / "action_scale.json")
    builder = ChunkTargetBuilder(
        nodes=nodes, scale=scale, policy=PolicyActionCache.load(ARTIFACT_DIR).check_against(nodes),
        kinematics=PandaKinematics(), chunk_len=schema.SEG_K,
    )
    queries = make_queries(
        nodes, np.random.RandomState(0), 64, np.arange(nodes.n_demos), QuerySpec(entry_frac=0.0)
    )
    assert queries["perturb_radius"].max() == 0.0, "entry_frac=0 must give a wholly clean set"

    targets = builder.build(queries)
    track = EeTrack(scale, bound=ACTION_LIMIT)
    for i, (tgt, t_now) in enumerate(zip(queries["tgt_node"], queries["t_raw_now"])):
        expected = track.chunk(
            nodes.ee_segment(int(tgt), schema.SEG_K),
            np.asarray(nodes.ee_base, dtype=np.float64)[int(t_now)],
        )
        # The tolerance is float32's: reference and live pose now come from the SAME forward
        # kinematics, and ee_base is stored f32, so nothing else separates the two.
        assert targets.a_corr[i] == pytest.approx(expected, abs=1e-3), i

    # a_policy must be the frozen policy's chunk at the window's OWN frame, not the label node's.
    assert targets.a_policy.shape == (64, schema.SEG_K, 7)
    assert targets.a_policy[:, 0, :] == pytest.approx(
        np.load(ARTIFACT_DIR / "a_pi_raw.npz")["a_pi_raw"][queries["t_raw_now"]]
    )


def test_a_policy_corruptor_actually_reaches_the_cached_chunks():
    """The wiring, not the sampler: PolicyCorruptor -> QuerySet -> ChunkTargets.a_policy.

    Every piece of this can be correct while the corruption never reaches the array the loss reads,
    and the symptom would be an artifact that trains cleanly and contains no change at all.
    """
    from robot_ood.core.action import ActionScale
    from robot_ood.core.kinematics import PandaKinematics
    from robot_ood.models.fusion.target import ChunkTargetBuilder, PolicyActionCache
    from robot_ood.train.augment import P_CLEAN, P_SHIFT, PolicyCorruptor, RetrievalCorruptor
    from robot_ood.train.tasks.query import QuerySet

    # dim=7: ChunkTargetBuilder reads nodes.ee_segment, which forward-kinematicks a Panda.
    nodes, _edges = _toy_graph(n_tasks=3, demos_per_task=3, length=40, dim=7)
    n_raw = int(nodes.n_raw)
    rng = np.random.RandomState(0)
    cache = PolicyActionCache(
        a_pi_raw=np.tile(np.arange(n_raw, dtype=np.float32)[:, None], (1, 7)),
        raw_ptr=np.asarray(nodes.raw_ptr, dtype=np.int64),
    )

    builder = ChunkTargetBuilder(
        nodes=nodes, scale=ActionScale(pos=0.0116, rot=0.1092, r2_pos=1.0, r2_rot=1.0),
        policy=cache, kinematics=PandaKinematics(), chunk_len=4,
    )
    queries = make_queries(nodes, rng, 256, np.arange(nodes.n_demos), QuerySpec(window=8))
    corruptor = PolicyCorruptor(nodes, RetrievalCorruptor(nodes))

    clean = QuerySet.from_queries(queries, torch.device("cpu"), builder)
    dirty = QuerySet.from_queries(
        queries, torch.device("cpu"), builder, rng=np.random.default_rng(0),
        policy_corruptor=corruptor,
    )
    kind = dirty.policy_corrupt.kind
    assert (kind != P_CLEAN).sum() > 40, "the sampler produced nothing to check"

    moved = ~np.all(clean.targets.a_policy == dirty.targets.a_policy, axis=(1, 2))
    # Every chunk-corrupted row moved, and no other row did. A GAIN of ~1.0 is possible but
    # vanishingly unlikely over a log-uniform draw across two decades. A SHIFT row moves the live
    # pose instead, and its correction with it.
    chunk_rows = (kind != P_CLEAN) & (kind != P_SHIFT)
    assert np.array_equal(moved, chunk_rows), (moved.sum(), int(chunk_rows.sum()))
    shifted = kind == P_SHIFT
    assert shifted.sum() > 10
    assert np.array_equal(clean.targets.a_corr[~shifted], dirty.targets.a_corr[~shifted])
    assert np.array_equal(clean.targets.ee_now[~shifted], dirty.targets.ee_now[~shifted])
    assert not np.any(np.all(clean.targets.ee_now[shifted] == dirty.targets.ee_now[shifted], axis=-1))


def test_a_shift_row_moves_the_live_pose_and_its_correction_but_not_the_chunk():
    """P_SHIFT is the arm landing away from the demos: ee_now moves by the drawn displacement,
    a_corr becomes the recovery tracked from there, and a_policy stays what the policy emitted."""
    from robot_ood.core.action import ACTION_LIMIT, ActionScale
    from robot_ood.core.kinematics import PandaKinematics
    from robot_ood.models.fusion.target import ChunkTargetBuilder, PolicyActionCache
    from robot_ood.runtime.ee_track import EeTrack
    from robot_ood.train.augment import P_CLEAN, P_SHIFT, PolicyCorruption

    nodes, _edges = _toy_graph(n_tasks=2, demos_per_task=2, length=40, dim=7)
    n_raw = int(nodes.n_raw)
    cache = PolicyActionCache(
        a_pi_raw=np.tile(np.arange(n_raw, dtype=np.float32)[:, None], (1, 7)),
        raw_ptr=np.asarray(nodes.raw_ptr, dtype=np.int64),
    )
    scale = ActionScale(pos=0.0116, rot=0.1092, r2_pos=1.0, r2_rot=1.0)
    builder = ChunkTargetBuilder(nodes=nodes, scale=scale, policy=cache, kinematics=PandaKinematics(), chunk_len=4)
    queries = make_queries(nodes, np.random.RandomState(0), 32, np.arange(nodes.n_demos), QuerySpec(window=8))

    kind = np.full(32, P_CLEAN, dtype=np.int8)
    kind[::4] = P_SHIFT
    shift = np.zeros((32, 6), dtype=np.float32)
    shift[::4] = np.array([0.05, -0.03, 0.02, 0.0, 0.1, -0.05], dtype=np.float32)
    corruption = PolicyCorruption(kind=kind, frame=np.full(32, -1, dtype=np.int64),
                                  gain=np.ones(32, dtype=np.float32), shift=shift)

    clean = builder.build(queries)
    dirty = builder.build(queries, corruption)

    moved = kind == P_SHIFT
    assert dirty.ee_now[moved] == pytest.approx(clean.ee_now[moved] + shift[moved], abs=1e-6)
    assert np.array_equal(dirty.ee_now[~moved], clean.ee_now[~moved])
    assert np.array_equal(dirty.a_policy, clean.a_policy)
    assert np.array_equal(dirty.a_corr[~moved], clean.a_corr[~moved])
    track = EeTrack(scale, bound=ACTION_LIMIT)
    for i in np.flatnonzero(moved):
        expected = track.chunk(clean.ee_ref[i].astype(np.float64), dirty.ee_now[i].astype(np.float64))
        assert dirty.a_corr[i] == pytest.approx(expected, abs=1e-4), i
    assert not np.allclose(dirty.a_corr[moved], clean.a_corr[moved])
