"""Query construction for the locator's training curriculum: make_queries and its four window classes."""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple

import numpy as np

from robot_ood.core.constants import ENTRY_STATIC_BAND, STILL_TOL
from robot_ood.data import schema
from robot_ood.data.geometry import entry_perturbation_floor
from robot_ood.train.drift import PANDA_JOINT_HI, PANDA_JOINT_LO
from robot_ood.data.nodes import NodeTable
from robot_ood.data.geometry import edge_pad_hist, finite_diff_vel
from robot_ood.train.types import QuerySpec

ENTRY_FRAC = 1.0 / 3.0
ENTRY_LO_MULT = 0.25
ENTRY_LOGUNIFORM = True

ENTRY_STATIC_FRAC = 0.15

DRIFT_FRAC = 0.15
DRIFT_MIN_HIST = 3

SEGMENT_FRAC = 0.0
SEGMENT_SIGMA_FRAC = 0.2

SEGMENT_ENTRY_FRAC_DEFAULT = 0.0


def _node_at_raw(nodes: NodeTable, owner: int, raw_idx: int) -> int:
    """The node on demo owner whose raw span contains raw_idx, clamped to the demo's own range.

    Args:
        nodes: Node table to look up in.
        owner: Demo index.
        raw_idx: Raw frame index.

    Returns:
        The node id.
    """
    node_ids = np.asarray(nodes.demo_nodes(int(owner)))
    t_raws = np.asarray(nodes.t_raw)[node_ids]
    pos = int(np.searchsorted(t_raws, int(raw_idx), side="right")) - 1
    pos = min(max(pos, 0), len(node_ids) - 1)
    return int(node_ids[pos])




def entry_sigma_range(nodes: NodeTable) -> tuple[float, float]:
    """The radius band (rad) every off-manifold draw shares, the one the entry field trains on.

    Args:
        nodes: Node table providing raw_ptr and q_raw.

    Returns:
        (lo, hi): a quarter of the corpus's median floor, never below what the corpus calls still,
        up to the benchmark's recorded robot-init magnitude.
    """
    from robot_ood.core.config import entry_radius_hi

    floor = entry_perturbation_floor(nodes)
    return max(ENTRY_LO_MULT * floor["p50"], STILL_TOL), entry_radius_hi()


def joint_clip_bounds(dim: int, joint_space: bool) -> tuple[np.ndarray, np.ndarray]:
    """(q_lo, q_hi) for a dim-axis config: the Panda joint limits, or +/-inf off a joint-space arm."""
    if joint_space:
        return PANDA_JOINT_LO[:dim], PANDA_JOINT_HI[:dim]
    return np.full(dim, -np.inf), np.full(dim, np.inf)


def entry_perturb(
    q0: np.ndarray, rng: np.random.RandomState, dim: int, sigma_range: tuple[float, float],
    q_lo: np.ndarray, q_hi: np.ndarray,
) -> np.ndarray:
    """q0 -> q0 + a random-direction offset of radius drawn from sigma_range, clipped to the limits.

    Args:
        q0: [dim] unperturbed configuration.
        rng: Random generator, consuming a direction draw then a radius draw.
        dim: Degrees of freedom.
        sigma_range: (lo, hi) radius range in radians.

    Returns:
        [dim] perturbed configuration.
    """
    direction = rng.randn(dim)
    n = np.linalg.norm(direction)
    direction = direction / n if n > 1e-12 else direction
    lo_r, hi_r = float(sigma_range[0]), float(sigma_range[1])
    # Log-uniform: a linear draw over two decades starves the small-offset end.
    radius = float(np.exp(rng.uniform(np.log(max(lo_r, 1e-6)), np.log(max(hi_r, 1e-6))))) \
        if ENTRY_LOGUNIFORM and hi_r > lo_r > 0 else float(rng.uniform(lo_r, hi_r))
    return np.clip(np.asarray(q0, np.float64) + radius * direction, q_lo, q_hi)


def _node_feature_matrix(nodes: NodeTable) -> np.ndarray:
    """[q, qdot, grip, t_frac] raw columns -- GraphRetrieverNet.node_encode's expected input.

    Args:
        nodes: Node table to read.

    Returns:
        [V, 2*dim+2] float32 feature matrix.
    """
    grip = np.asarray(nodes.grip, dtype=np.float32)[:, None]
    t_frac = np.asarray(nodes.t_frac, dtype=np.float32)[:, None]
    return np.concatenate([nodes.q, nodes.qdot, grip, t_frac], axis=1).astype(np.float32)


class _QueryRow(NamedTuple):
    """One row of make_queries output, as returned by each per-class row-builder.

    Attributes:
        q_hist: [window, dim] float32 joint-position window.
        src_owner: Demo the window was drawn from.
        src_node: Node at the window's own end, T.
        tgt_node: Node at T+advance -- the retrieval label.
        t_raw_now: The window's own current RAW frame, T. Not recoverable from src_node, whose
            t_raw is up to coarsen-1 frames behind it.
        is_clean: Provenance: True for TRAVERSAL, False for the perturbed classes.
        is_entry_static: Whether this row is an ENTRY_STATIC row.
        is_drift: Whether this row is a DRIFT row.
        break_row: First perturbed row of the window, or -1 when every row is clean.
        perturb_radius: Realised (post-clip) offset radius at the window's end, rad; 0 when clean.
        seg_id: The C1 segment id this row's ground-truth continuation is (SEGMENT rows only), or
            -1 for every other class.
    """

    q_hist: np.ndarray
    src_owner: int
    src_node: int
    tgt_node: int
    t_raw_now: int
    is_clean: bool
    is_entry_static: bool
    is_drift: bool
    break_row: int
    perturb_radius: float
    seg_id: int = -1


class _QueryCorpus(NamedTuple):
    """The per-call corpus context shared by the three row-builders.

    Attributes:
        nodes: The node table.
        raw_ptr: [J+1] per-demo boundaries into the raw arrays.
        q_raw: [N_raw, dim] float32 raw joint configs.
        dim: Degrees of freedom.
        window: This call's window length.
        q_lo: [dim] lower clip bound per config axis; -inf where the corpus has no joint limit.
        q_hi: [dim] upper clip bound per config axis; +inf where the corpus has no joint limit.
    """

    nodes: NodeTable
    raw_ptr: np.ndarray
    q_raw: np.ndarray
    dim: int
    window: int
    q_lo: np.ndarray
    q_hi: np.ndarray


def _entry_static_row(
    corpus: _QueryCorpus, rng: np.random.RandomState, pool_entry: np.ndarray,
    entry_sigma_range: tuple[float, float],
) -> _QueryRow:
    """Build one ENTRY_STATIC row -- see make_queries.

    Args:
        corpus: The per-call corpus context.
        rng: Random generator; draws rng.randint (pick demo) then entry_perturb's own draws.
        pool_entry: Demos long enough to serve as entry queries.
        entry_sigma_range: (lo, hi) perturbation radius range.

    Returns:
        The row.
    """
    nodes, raw_ptr, q_raw, dim, window = corpus[:5]
    j = int(pool_entry[rng.randint(len(pool_entry))])
    lo = int(raw_ptr[j])
    q0 = q_raw[lo].astype(np.float64)
    q0_pert = entry_perturb(q0, rng, dim, entry_sigma_range, corpus.q_lo, corpus.q_hi)
    # Every row of this window is a copy of the SAME perturbed frame, so finite_diff_vel yields
    # identically zero velocity -- the exact shape a live t=0 call's repeat(q0, hist) produces.
    q_hist_row = np.repeat(q0_pert[None, :].astype(np.float32), window, axis=0)
    tgt = _node_at_raw(nodes, j, lo)
    return _QueryRow(q_hist=q_hist_row, src_owner=j, src_node=tgt, tgt_node=tgt, t_raw_now=lo,
                      is_clean=False, is_entry_static=True, is_drift=False, break_row=0,
                      perturb_radius=float(np.linalg.norm(q0_pert - q0)))


def _entry_row(
    corpus: _QueryCorpus, rng: np.random.RandomState, pool_entry: np.ndarray, advance: int,
    entry_sigma_range: tuple[float, float],
) -> _QueryRow:
    """Build one ENTRY row -- see make_queries.

    Args:
        corpus: The per-call corpus context.
        rng: Random generator; draws rng.randint (pick demo), entry_perturb's own draws, then
            rng.randint (window length).
        pool_entry: Demos long enough to serve as entry queries.
        advance: Raw frames from the window end to the target.
        entry_sigma_range: (lo, hi) perturbation radius range.

    Returns:
        The row.
    """
    nodes, raw_ptr, q_raw, dim, window = corpus[:5]
    j = int(pool_entry[rng.randint(len(pool_entry))])
    lo = int(raw_ptr[j])
    demo_hi = int(raw_ptr[j + 1])
    q0 = q_raw[lo].astype(np.float64)
    q0_pert = entry_perturb(q0, rng, dim, entry_sigma_range, corpus.q_lo, corpus.q_hi)
    delta = q0_pert - q0                             # the REALIZED constant offset (post-clip)

    w_use = min(int(rng.randint(1, window + 1)), demo_hi - lo)   # same W distribution as TRAVERSAL
    t_raw_end = lo + w_use - 1                                   # window's OWN (current) end
    raw_window = q_raw[lo : lo + w_use].astype(np.float64) + delta[None, :]
    raw_window = np.clip(raw_window, corpus.q_lo, corpus.q_hi)
    q_hist_row = edge_pad_hist(raw_window.astype(np.float32), window)[0]
    return _QueryRow(
        q_hist=q_hist_row, src_owner=j, src_node=_node_at_raw(nodes, j, t_raw_end),
        tgt_node=_node_at_raw(nodes, j, t_raw_end + advance), t_raw_now=t_raw_end, is_clean=False,
        is_entry_static=False, is_drift=False, break_row=0,
        perturb_radius=float(np.linalg.norm(delta)),
    )


def _drift_row(
    corpus: _QueryCorpus, rng: np.random.RandomState, pool_trav: np.ndarray, advance: int,
    entry_sigma_range: tuple[float, float],
) -> _QueryRow:
    """Build one DRIFT row -- see make_queries.

    Args:
        corpus: The per-call corpus context.
        rng: Random generator; draws rng.randint (pick demo), rng.randint (T), rng.randint (window
            length), rng.randint (breakpoint), then entry_perturb's own draws.
        pool_trav: Demos long enough for a full window plus advance.
        advance: Raw frames from the window end to the target.
        entry_sigma_range: (lo, hi) perturbation radius range.

    Returns:
        The row.
    """
    nodes, raw_ptr, q_raw, dim, window = corpus[:5]
    j = int(pool_trav[rng.randint(len(pool_trav))])
    lo, hi = int(raw_ptr[j]), int(raw_ptr[j + 1])
    t_raw_end = lo + int(rng.randint(window - 1, (hi - lo) - advance))   # same T draw as TRAVERSAL
    w_use = int(rng.randint(DRIFT_MIN_HIST, window + 1))
    n_clean = int(rng.randint(1, w_use - 1))                             # leaves >= 2 perturbed rows
    raw_window = q_raw[t_raw_end - w_use + 1 : t_raw_end + 1].astype(np.float64)

    delta = entry_perturb(raw_window[-1], rng, dim, entry_sigma_range, corpus.q_lo, corpus.q_hi) - raw_window[-1]
    # A RAMP to the full radius at T, not a step at the breakpoint.
    ramp = np.arange(1, w_use - n_clean + 1, dtype=np.float64) / (w_use - n_clean)
    drifted = raw_window.copy()
    drifted[n_clean:] += ramp[:, None] * delta[None, :]
    drifted = np.clip(drifted, corpus.q_lo, corpus.q_hi)
    q_hist_row = edge_pad_hist(drifted.astype(np.float32), window)[0]
    return _QueryRow(
        q_hist=q_hist_row, src_owner=j, src_node=_node_at_raw(nodes, j, t_raw_end),
        tgt_node=_node_at_raw(nodes, j, t_raw_end + advance), t_raw_now=t_raw_end, is_clean=False,
        is_entry_static=False, is_drift=True,
        # edge_pad_hist right-aligns, so a w_use-row window's own row i sits at window - w_use + i.
        break_row=window - w_use + n_clean, perturb_radius=float(np.linalg.norm(delta)),
    )


def _segment_row(
    corpus: _QueryCorpus, rng: np.random.RandomState, seg_pool: np.ndarray, segments: Any,
    segment_sigma_range: tuple[float, float], *,
    seg_pool_entry: np.ndarray | None = None, segment_entry_frac: float = 0.0,
) -> _QueryRow:
    """Build one SEGMENT row -- see make_queries.

    Args:
        corpus: The per-call corpus context.
        rng: Random generator; draws the entry-stratum coin, the segment pick, then entry_perturb.
        seg_pool: Segment ids whose owner is in this call's owners.
        segments: The C1 SegmentTable (robot_ood.core.schemas.SegmentTable).
        segment_sigma_range: (lo, hi) perturbation radius range -- SEGMENT_SIGMA_FRAC of ENTRY's.
        seg_pool_entry: The subset of seg_pool at phase_start <= segment_entry_band, or None.
        segment_entry_frac: Share of SEGMENT rows drawn from seg_pool_entry instead of the pool.

    Returns:
        The row, whose ground truth is the segment's own continuation.
    """
    nodes, _raw_ptr, q_raw, dim, window = corpus[:5]
    use_entry = (
        segment_entry_frac > 0 and seg_pool_entry is not None and len(seg_pool_entry) > 0
        and rng.rand() < segment_entry_frac
    )
    pool = seg_pool_entry if use_entry else seg_pool
    s = int(pool[rng.randint(len(pool))])
    start_node = int(segments.start_node[s])
    owner = int(nodes.owner[start_node])
    t_raw = int(nodes.t_raw[start_node])
    q0 = q_raw[t_raw].astype(np.float64)
    q0_pert = entry_perturb(q0, rng, dim, segment_sigma_range, corpus.q_lo, corpus.q_hi)
    # A single perturbed instant repeated across the window, same shape as ENTRY_STATIC.
    q_hist_row = np.repeat(q0_pert[None, :].astype(np.float32), window, axis=0)
    return _QueryRow(
        q_hist=q_hist_row, src_owner=owner, src_node=start_node, tgt_node=start_node,
        t_raw_now=t_raw, is_clean=False, is_entry_static=False, is_drift=False, break_row=0,
        perturb_radius=float(np.linalg.norm(q0_pert - q0)), seg_id=s,
    )


def _traversal_row(corpus: _QueryCorpus, rng: np.random.RandomState, pool_trav: np.ndarray, advance: int) -> _QueryRow:
    """Build one TRAVERSAL row -- see make_queries.

    Args:
        corpus: The per-call corpus context.
        rng: Random generator; draws rng.randint (pick demo), rng.randint (T), rng.randint (window
            length).
        pool_trav: Demos long enough for a full window plus advance.
        advance: Raw frames from the window end to the target.

    Returns:
        The row.
    """
    nodes, raw_ptr, q_raw, _dim, window = corpus[:5]
    j = int(pool_trav[rng.randint(len(pool_trav))])
    lo, hi = int(raw_ptr[j]), int(raw_ptr[j + 1])
    length = hi - lo
    t_local = int(rng.randint(window - 1, length - advance))   # T, local to this demo
    t_raw_end = lo + t_local
    w_use = int(rng.randint(1, window + 1))                    # variable true history length
    start = t_raw_end - w_use + 1
    raw_window = q_raw[start : t_raw_end + 1]
    q_hist_row = edge_pad_hist(raw_window, window)[0]
    return _QueryRow(
        q_hist=q_hist_row, src_owner=j, src_node=_node_at_raw(nodes, j, t_raw_end),
        tgt_node=_node_at_raw(nodes, j, t_raw_end + advance), t_raw_now=t_raw_end, is_clean=True,
        is_entry_static=False, is_drift=False, break_row=-1, perturb_radius=0.0,
    )


def make_queries(
    nodes: NodeTable, rng: np.random.RandomState, n: int, owners: np.ndarray, spec: QuerySpec | None = None,
    *, joint_space: bool = True,
) -> dict[str, np.ndarray]:
    """Draw n training queries from demos owned by owners, four classes mixed per spec.

    Args:
        nodes: Node table to draw from.
        rng: Random generator driving the class mix and every row builder.
        n: Number of queries.
        owners: Demo indices eligible for this set.
        spec: Keyword options; None is exactly QuerySpec().
        joint_space: True clips perturbed windows to the Panda joint limits; False (a pose read off
            the observation) leaves them unclipped.

    Returns:
        A dict of length-n arrays:
            q_hist: [n, window, D] f32.
            qdot_hist: [n, window, D] f32 (robot_ood.data.geometry.finite_diff_vel).
            grip_hist: [n, window] f32, broadcast from the query's own T node.
            src_owner: [n] i32.
            src_node: [n] i64, the T node.
            tgt_node: [n] i64, the T+advance node and the retrieval label.
            t_raw_now: [n] i64, the T raw frame.
            is_clean: [n] bool, which branch built the window.
            is_entry_static: [n] bool.
            is_drift: [n] bool.
            break_row: [n] i32, first perturbed window row, -1 when clean.
            perturb_radius: [n] f32, realised offset radius at T, rad.
            seg_id: [n] i64, the row's own segment id for SEGMENT rows, -1 otherwise.
            a_hist: [n, window, ACTION_DIM] f32, the demo's recorded actions ending at T.
            phase_hat: [n] f32, the row's segment phase, 0.0 on rows with no segment.

    Raises:
        ValueError: If owners is empty, no demo is long enough for the traversal queries the
            requested class mix still needs, or DRIFT is requested on a window too short to hold a
            breakpoint.
    """
    spec = spec if spec is not None else QuerySpec()
    advance = schema.ADVANCE if spec.advance is None else spec.advance
    window = schema.HIST_H if spec.window is None else spec.window
    entry_frac = ENTRY_FRAC if spec.entry_frac is None else spec.entry_frac
    sigma_range = spec.entry_sigma_range
    entry_static_frac = spec.entry_static_frac
    drift_frac = float(spec.drift_frac)
    segments = spec.segments
    segment_frac = float(spec.segment_frac) if segments is not None else 0.0
    segment_entry_frac = float(spec.segment_entry_frac) if segments is not None else 0.0
    segment_entry_band = ENTRY_STATIC_BAND if spec.segment_entry_band is None else float(spec.segment_entry_band)

    owners = np.asarray(owners, dtype=np.int64)
    if len(owners) == 0:
        raise ValueError("make_queries: `owners` is empty")
    raw_ptr = np.asarray(nodes.raw_ptr, dtype=np.int64)
    lengths = np.diff(raw_ptr)

    min_len_trav = window + advance
    pool_trav = owners[lengths[owners] >= min_len_trav]
    min_len_entry = advance + 1
    pool_entry = owners[lengths[owners] >= min_len_entry]

    entry_frac = float(entry_frac)
    entry_static_frac = float(entry_static_frac)
    if len(pool_entry) == 0 and (entry_frac > 0 or entry_static_frac > 0):
        warnings.warn(
            f"make_queries: no demo in `owners` has >= {min_len_entry} raw frames -- entry_frac "
            f"{entry_frac} / entry_static_frac {entry_static_frac} forced to 0 for this call", stacklevel=2,
        )
        entry_frac = 0.0
        entry_static_frac = 0.0
    if len(pool_trav) == 0:
        if entry_frac + entry_static_frac < 1.0:
            raise ValueError(
                f"make_queries: no demo in `owners` has >= {min_len_trav} raw frames (window={window} "
                f"+ advance={advance}), and entry_frac={entry_frac}+entry_static_frac={entry_static_frac} "
                "< 1 still needs traversal queries"
            )
        pool_trav = pool_entry   # unused when entry_frac+entry_static_frac == 1.0, kept non-empty defensively
    if drift_frac > 0 and window < DRIFT_MIN_HIST:
        raise ValueError(
            f"make_queries: drift_frac={drift_frac} needs window >= {DRIFT_MIN_HIST} for a window "
            f"that can hold a clean-to-perturbed breakpoint, got window={window}"
        )

    seg_pool = None
    seg_pool_entry = None
    if segment_frac > 0:
        seg_pool = np.flatnonzero(np.isin(np.asarray(segments.owner, dtype=np.int64), owners))
        if seg_pool.size == 0:
            warnings.warn(
                f"make_queries: no segment in `segments` belongs to a demo in `owners` -- "
                f"segment_frac {segment_frac} forced to 0 for this call", stacklevel=2,
            )
            segment_frac = 0.0
            seg_pool = None
        elif segment_entry_frac > 0:
            phase_start = np.asarray(segments.phase_start, dtype=np.float64)[seg_pool]
            seg_pool_entry = seg_pool[phase_start <= segment_entry_band]
            if seg_pool_entry.size == 0:
                warnings.warn(
                    f"make_queries: no segment in `owners`'s pool has phase_start <= "
                    f"segment_entry_band={segment_entry_band} -- segment_entry_frac "
                    f"{segment_entry_frac} forced to 0 for this call", stacklevel=2,
                )
                segment_entry_frac = 0.0
                seg_pool_entry = None

    if (
        entry_frac > 0 or entry_static_frac > 0 or drift_frac > 0 or segment_frac > 0
    ) and sigma_range is None:
        sigma_range = entry_sigma_range(nodes)
    segment_sigma_range = (
        None if sigma_range is None
        else (SEGMENT_SIGMA_FRAC * sigma_range[0], SEGMENT_SIGMA_FRAC * sigma_range[1])
    )

    dim = nodes.dim
    q_raw = np.asarray(nodes.q_raw, dtype=np.float32)
    a_raw = np.asarray(nodes.a_raw, dtype=np.float32)
    q_lo, q_hi = joint_clip_bounds(dim, joint_space)
    corpus = _QueryCorpus(nodes=nodes, raw_ptr=raw_ptr, q_raw=q_raw, dim=dim, window=window,
                          q_lo=q_lo, q_hi=q_hi)
    q_hist = np.empty((n, window, dim), dtype=np.float32)
    a_hist = np.empty((n, window, a_raw.shape[1]), dtype=np.float32)
    src_owner = np.empty(n, dtype=np.int32)
    src_node = np.empty(n, dtype=np.int64)
    tgt_node = np.empty(n, dtype=np.int64)
    t_raw_now = np.empty(n, dtype=np.int64)
    is_clean = np.empty(n, dtype=bool)
    is_entry_static = np.zeros(n, dtype=bool)
    is_drift = np.zeros(n, dtype=bool)
    break_row = np.full(n, -1, dtype=np.int32)
    perturb_radius = np.zeros(n, dtype=np.float32)
    seg_id = np.full(n, -1, dtype=np.int64)

    for i in range(n):
        # A zero fraction must reproduce the RNG stream of the mix that predates its class, so every
        # check short-circuits its own rng.rand() away rather than drawing and discarding it.
        if entry_static_frac > 0:
            is_static = rng.rand() < entry_static_frac
        else:
            is_static = False
        is_entry = False if is_static else (entry_frac > 0 and rng.rand() < entry_frac)
        drifts = False if (is_static or is_entry) else (drift_frac > 0 and rng.rand() < drift_frac)
        is_segment = False if (is_static or is_entry or drifts) else (
            segment_frac > 0 and rng.rand() < segment_frac
        )

        if is_static:
            row = _entry_static_row(corpus, rng, pool_entry, sigma_range)
        elif is_entry:
            row = _entry_row(corpus, rng, pool_entry, advance, sigma_range)
        elif drifts:
            row = _drift_row(corpus, rng, pool_trav, advance, sigma_range)
        elif is_segment:
            row = _segment_row(
                corpus, rng, seg_pool, segments, segment_sigma_range,
                seg_pool_entry=seg_pool_entry, segment_entry_frac=segment_entry_frac,
            )
        else:
            row = _traversal_row(corpus, rng, pool_trav, advance)

        q_hist[i] = row.q_hist
        # a_hist is the demo's OWN recorded actions (nodes.a_raw), never a_pi_raw.npz's replay.
        # Same right-aligned edge padding as q_hist.
        lo_owner = int(raw_ptr[row.src_owner])
        hi_excl = int(row.t_raw_now) + 1
        start = max(lo_owner, hi_excl - window)
        a_hist[i] = edge_pad_hist(a_raw[start:hi_excl], window)[0]
        src_owner[i] = row.src_owner
        src_node[i] = row.src_node
        tgt_node[i] = row.tgt_node
        t_raw_now[i] = row.t_raw_now
        is_clean[i] = row.is_clean
        is_entry_static[i] = row.is_entry_static
        is_drift[i] = row.is_drift
        break_row[i] = row.break_row
        perturb_radius[i] = row.perturb_radius
        seg_id[i] = row.seg_id

    qdot_hist = finite_diff_vel(q_hist)
    grip_hist = np.repeat(np.asarray(nodes.grip, dtype=np.float32)[src_node][:, None], window, axis=1)

    # The phase the row's QueryContext will carry. Non-SEGMENT rows carry 0.0 and are unread.
    phase_true = np.zeros(n, dtype=np.float32)
    if segments is not None:
        has_seg = seg_id >= 0
        table = segments.phase_start
        phase_true[has_seg] = np.asarray(table, dtype=np.float32)[seg_id[has_seg]]
    phase_hat = (
        phase_true if spec.phase_residual is None else spec.phase_residual.apply(phase_true, rng)
    )

    return {
        "q_hist": q_hist, "qdot_hist": qdot_hist, "grip_hist": grip_hist,
        "src_owner": src_owner, "src_node": src_node, "tgt_node": tgt_node,
        "t_raw_now": t_raw_now, "is_clean": is_clean, "is_entry_static": is_entry_static,
        "is_drift": is_drift, "break_row": break_row, "perturb_radius": perturb_radius,
        "seg_id": seg_id, "a_hist": a_hist, "phase_hat": phase_hat,
    }

def tile_queries(queries: dict[str, np.ndarray], copies: int) -> dict[str, np.ndarray]:
    """Repeat a query set copies times, copy-major, so each source row can take K corruptions.

    Args:
        queries: A make_queries result.
        copies: K, at least 1. 1 returns the mapping unchanged, not a copy of it.

    Returns:
        The tiled mapping, every array K times longer along axis 0.

    Raises:
        ValueError: copies is below 1.
    """
    if int(copies) < 1:
        raise ValueError(f"tile_queries: copies must be >= 1, got {copies}")
    if int(copies) == 1:
        return queries
    # Copy-major: copy k's rows land at [k*n:(k+1)*n], matching CopyPlan.rows() so each
    # copy's corruption draws align position-for-position with its tiled rows.
    return {k: np.concatenate([v] * int(copies), axis=0) for k, v in queries.items()}
