"""Manufactures labeled failures so the corrector learns when the retrieved demos are worth following.

Retrieval and policy corruptions are drawn independently, so every quadrant of (retrieval wrong or
right, policy wrong or right) is seen in training. The correct correction (a_corr, the query's own
demo continued from the live pose) is never changed by any corruption: the loss then lowers the
dose where the reference is corrupted and raises it where the policy is. TARGET is the retrieval
corruption a scene change produces: the demos start where the arm is and end somewhere else, which
must read as "leave the policy alone". P_SHIFT is its mirror on the policy side: the arm is not
where the demos were, the policy acts as if it were, and the demos are right. BeliefPhaseResidual
at the bottom is the fourth axis: query phase, drawn from the filter's measured residual.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Sequence

import numpy as np

from robot_ood.core.action import POSE_DIM
from robot_ood.data import schema
from robot_ood.data.schema import PHASE_SKEW_JSON

if TYPE_CHECKING:                                    # pragma: no cover -- typing only
    from robot_ood.data.nodes import NodeTable

__all__ = [
    "BeliefPhaseResidual",
    "CLEAN",
    "WRONG_TASK",
    "OFFSET",
    "WRONG_PHASE",
    "TARGET",
    "KIND_NAMES",
    "Corruption",
    "RetrievalCorruptor",
    "P_CLEAN",
    "P_LAG",
    "P_TASK",
    "P_GAIN",
    "P_SHIFT",
    "POLICY_KIND_NAMES",
    "PolicyCorruption",
    "PolicyCorruptor",
    "CopyPlan",
]

CLEAN: Final[int] = 0
WRONG_TASK: Final[int] = 1
OFFSET: Final[int] = 2
WRONG_PHASE: Final[int] = 3
TARGET: Final[int] = 4
KIND_NAMES: Final[tuple[str, ...]] = ("clean", "wrong_task", "offset", "wrong_phase", "target")

# Rejection draws per row before a branch gives up and leaves the row clean. Both branches accept
# roughly 9 draws in 10, so exhausting this is a property of a degenerate graph, not of bad luck.
_MAX_TRIES: Final[int] = 8

# Draw the OFFSET and TARGET displacement magnitudes log-uniformly over their ranges.
OFFSET_LOGUNIFORM: Final[bool] = True


@dataclass(frozen=True, eq=False)
class Corruption:
    """Which rows of a query set are corrupted, and how.

    Attributes:
        kind: [n] int8, one of CLEAN/WRONG_TASK/OFFSET/WRONG_PHASE/TARGET.
        node: [n] int64 node whose reference segment replaces the posterior's expectation, or -1 to
            keep the posterior's. OFFSET and TARGET rows keep it: their node is right and only its
            poses moved.
        offset: [n, 6] float32 Cartesian displacement (metres, then a rotation vector in radians)
            of the reference segment. OFFSET moves the whole segment by it; TARGET keeps every
            candidate's start and moves every later row and end by it. Zero elsewhere.
            A joint-space lane maps it through the arm's Jacobian (robot_ood.train.fcsr.realize_column0).
    """

    kind: np.ndarray
    node: np.ndarray
    offset: np.ndarray

    def __len__(self) -> int:
        return int(self.kind.shape[0])

    def counts(self) -> dict[str, int]:
        """Rows per branch, for the log line that proves the sampler ran.

        Returns:
            {branch name: row count}, including branches with zero rows.
        """
        return {name: int((self.kind == k).sum()) for k, name in enumerate(KIND_NAMES)}

    @classmethod
    def concat(cls, parts: "Sequence[Corruption]") -> "Corruption":
        """Stack the per-copy draws of one CopyPlan into one row-aligned corruption.

        Args:
            parts: One draw per copy, each over the same source rows in the same order.

        Returns:
            The concatenation, in copy-major order -- CopyPlan.rows's own order.
        """
        return cls(
            kind=np.concatenate([p.kind for p in parts]),
            node=np.concatenate([p.node for p in parts]),
            offset=np.concatenate([p.offset for p in parts]),
        )


@dataclass(eq=False)
class RetrievalCorruptor:
    """Draws a Corruption for any query set over one fixed graph.

    Attributes:
        nodes: The node table queries index into.
        p_wrong_task: Share of rows given a different task's node.
        p_offset: Share of rows given a displaced reference segment.
        p_wrong_phase: Share of rows given the right demo at a distant phase.
        p_target: Share of rows whose every candidate starts where the arm is and then runs
            displaced: the retrieval a moved object produces, where the demos agree with each
            other and the policy, which sees the scene, is right.
        phase_tol: WRONG_TASK's node must sit within this much phase of the true one, so the
            failure is a genuine alias and not a trivially-detectable phase error.
        phase_min: WRONG_PHASE's node must sit at least this much phase away.
        offset_m: (low, high) metres for the OFFSET displacement magnitude, drawn log-uniformly.
            Two decades, from well under the measured 2.32 cm median displacement to a shelf away, so
            the head learns the dose as a function OF the magnitude rather than a verdict on one
            band. See OFFSET_LOGUNIFORM.
        offset_rad: (low, high) radians for the rotation part of the same displacement.
        target_m: (low, high) metres for the TARGET end displacement: the range an object moves
            under a layout change, from within the demos' own spread to well outside it.
        target_rad: (low, high) radians for the TARGET end rotation.
    """

    nodes: "NodeTable"
    p_wrong_task: float = 0.10
    p_offset: float = 0.10
    p_wrong_phase: float = 0.10
    p_target: float = 0.15
    phase_tol: float = 0.05
    phase_min: float = 0.15
    offset_m: tuple[float, float] = (0.005, 0.5)
    offset_rad: tuple[float, float] = (0.02, 0.5)
    target_m: tuple[float, float] = (0.02, 0.15)
    target_rad: tuple[float, float] = (0.05, 0.5)

    def __post_init__(self) -> None:
        total = self.p_wrong_task + self.p_offset + self.p_wrong_phase + self.p_target
        if not 0.0 <= total <= 1.0:
            raise ValueError(f"corruption shares must sum into [0, 1], got {total}")
        self._phase = np.asarray(self.nodes.phase, dtype=np.float64)
        self._task = np.asarray(self.nodes.task_id, dtype=np.int64)
        self._owner = np.asarray(self.nodes.owner, dtype=np.int64)
        self._demo_ptr = np.asarray(self.nodes.demo_ptr, dtype=np.int64)
        # Phase-sorted view, so the WRONG_TASK candidate band is a searchsorted slice rather than a
        # [n, V] mask -- on the long graph that mask would be 28476 booleans per query.
        self._order = np.argsort(self._phase, kind="stable")
        self._sorted_phase = self._phase[self._order]

    def draw(self, tgt_node: np.ndarray, rng: np.random.Generator) -> Corruption:
        """Assign a branch to every row of a query set and realise it.

        Args:
            tgt_node: [n] int64 the label node of each query.
            rng: Random state. Drawing the branch assignment first, in one call, keeps the
                assignment independent of how many rejection draws each branch then needs.

        Returns:
            The corruption, in query-set row order.
        """
        tgt = np.asarray(tgt_node, dtype=np.int64)
        n = int(tgt.shape[0])
        kind = self._assign(n, rng)
        node = np.full(n, -1, dtype=np.int64)
        offset = np.zeros((n, 6), dtype=np.float32)

        rows = np.flatnonzero(kind == WRONG_TASK)
        node[rows] = self.pick_other_task(tgt[rows], rng)
        rows = np.flatnonzero(kind == WRONG_PHASE)
        node[rows] = self._pick_wrong_phase(tgt[rows], rng)
        rows = np.flatnonzero(kind == OFFSET)
        offset[rows] = self._pick_offset(rows.size, rng, self.offset_m, self.offset_rad)
        rows = np.flatnonzero(kind == TARGET)
        offset[rows] = self._pick_offset(rows.size, rng, self.target_m, self.target_rad)

        # A branch that could not find a legal node leaves the row clean rather than silently
        # corrupting it in some other way.
        kind[np.flatnonzero((kind != CLEAN) & (kind != OFFSET) & (kind != TARGET) & (node < 0))] = CLEAN
        return Corruption(kind=kind, node=node, offset=offset)

    # ---- branch assignment ---------------------------------------------------------------------
    def _assign(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw each row's branch from the configured shares.

        Args:
            n: Number of rows.
            rng: Random state.

        Returns:
            [n] int8 branch ids.
        """
        u = rng.random(n)
        edges = np.cumsum([self.p_wrong_task, self.p_offset, self.p_wrong_phase, self.p_target])
        kind = np.full(n, CLEAN, dtype=np.int8)
        kind[u < edges[3]] = TARGET
        kind[u < edges[2]] = WRONG_PHASE
        kind[u < edges[1]] = OFFSET
        kind[u < edges[0]] = WRONG_TASK
        return kind

    # ---- the three branches --------------------------------------------------------------------
    def pick_other_task(self, tgt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """A node of a DIFFERENT task at a comparable phase, per row.

        Args:
            tgt: [m] int64 label nodes.
            rng: Random state.

        Returns:
            [m] int64 node ids, -1 where the phase band held no other task.
        """
        if tgt.size == 0:
            return np.zeros(0, dtype=np.int64)
        phase = self._phase[tgt]
        lo = np.searchsorted(self._sorted_phase, phase - self.phase_tol, side="left")
        hi = np.searchsorted(self._sorted_phase, phase + self.phase_tol, side="right")
        return self._reject_sample(
            lo, hi, rng, lambda cand, row: self._task[cand] != self._task[tgt[row]]
        )

    def _pick_wrong_phase(self, tgt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """A node of the SAME demo at least phase_min away, per row.

        Args:
            tgt: [m] int64 label nodes.
            rng: Random state.

        Returns:
            [m] int64 node ids, -1 where the demo was too short to hold one.
        """
        if tgt.size == 0:
            return np.zeros(0, dtype=np.int64)
        owner = self._owner[tgt]
        lo, hi = self._demo_ptr[owner], self._demo_ptr[owner + 1]
        return self._reject_sample(
            lo, hi, rng,
            lambda cand, row: np.abs(self._phase[cand] - self._phase[tgt[row]]) >= self.phase_min,
            sorted_view=False,
        )

    @staticmethod
    def _pick_offset(m: int, rng: np.random.Generator, metres: tuple[float, float],
                     radians: tuple[float, float]) -> np.ndarray:
        """Isotropic Cartesian displacements: a position offset and an independent rotation vector.

        Args:
            m: Number of rows.
            rng: Random state.
            metres: (low, high) magnitude range of the position part.
            radians: (low, high) magnitude range of the rotation part.

        Returns:
            [m, 6] float32 displacements, each block's magnitude log-uniform in its range. Direction
            is drawn first and magnitude second, matching robot_ood.train.data._entry_perturb, so
            the samplers consume their generator the same way.
        """
        out = np.zeros((m, 6), dtype=np.float32)
        for block, (lo, hi) in ((slice(0, 3), metres), (slice(3, 6), radians)):
            direction = rng.standard_normal((m, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True).clip(1e-12)
            lo, hi = float(lo), float(hi)
            if OFFSET_LOGUNIFORM and hi > lo > 0.0:
                magnitude = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(m, 1)))
            else:
                magnitude = rng.uniform(lo, hi, size=(m, 1))
            out[:, block] = direction * magnitude
        return out

    def _reject_sample(self, lo, hi, rng, accept, *, sorted_view: bool = True) -> np.ndarray:
        """Draw one accepted candidate per row from that row's [lo, hi) index band.

        Args:
            lo: [m] int64 band starts.
            hi: [m] int64 band ends, exclusive.
            rng: Random state.
            accept: (candidate ids [k], row indices [k]) -> [k] bool.
            sorted_view: True when the band indexes the phase-sorted order rather than node ids.

        Returns:
            [m] int64 node ids, -1 where no draw was accepted.
        """
        m = int(np.asarray(lo).shape[0])
        out = np.full(m, -1, dtype=np.int64)
        width = np.maximum(hi - lo, 0)
        pending = np.flatnonzero(width > 0)
        for _ in range(_MAX_TRIES):
            if pending.size == 0:
                break
            pos = lo[pending] + (rng.random(pending.size) * width[pending]).astype(np.int64)
            cand = self._order[pos] if sorted_view else pos
            ok = accept(cand, pending)
            out[pending[ok]] = cand[ok]
            pending = pending[~ok]
        return out


# ==== The other operand: a policy that is wrong, on a query whose retrieval is right ====
P_CLEAN: Final[int] = 0
P_LAG: Final[int] = 1
P_TASK: Final[int] = 2
P_GAIN: Final[int] = 3
P_SHIFT: Final[int] = 4
POLICY_KIND_NAMES: Final[tuple[str, ...]] = ("clean", "lag", "policy_task", "gain", "shift")

# Both magnitudes are drawn LOG-uniformly. GAIN is applied to the POSE block only. A lag of at
# least 8 frames and a gain outside [0.5, 2]: a corrupt row is never a near-copy of a clean one,
# which the label would otherwise call wrong on evidence the heads cannot see.
LAG_FRAMES: Final[tuple[int, int]] = (8, 64)
GAIN_RANGE: Final[tuple[float, float]] = (0.02, 4.0)
GAIN_DEAD: Final[tuple[float, float] | None] = (0.5, 2.0)
# P_SHIFT moves the live pose, not the chunk: the range an arm lands away from the demos' poses.
SHIFT_M: Final[tuple[float, float]] = (0.02, 0.15)
SHIFT_RAD: Final[tuple[float, float]] = (0.05, 0.5)


def _log_uniform(low: float, high: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """n draws log-uniform on [low, high]."""
    return np.exp(rng.uniform(np.log(low), np.log(high), size=n))


def _gain_draw(n: int, rng: np.random.Generator) -> np.ndarray:
    """n gains log-uniform on GAIN_RANGE, with GAIN_DEAD (when set) cut out of the support."""
    if GAIN_DEAD is None:
        return _log_uniform(*GAIN_RANGE, n, rng)
    lo, hi = GAIN_RANGE
    dlo, dhi = GAIN_DEAD
    w_low, w_high = np.log(dlo / lo), np.log(hi / dhi)
    low_side = rng.random(n) < w_low / (w_low + w_high)
    out = np.empty(n, dtype=np.float64)
    out[low_side] = _log_uniform(lo, dlo, int(low_side.sum()), rng)
    out[~low_side] = _log_uniform(dhi, hi, int((~low_side).sum()), rng)
    return out


@dataclass(frozen=True, eq=False)
class PolicyCorruption:
    """Per-row instructions for corrupting the frozen policy's cached chunk.

    Attributes:
        kind: [n] int8, one of P_CLEAN/P_LAG/P_TASK/P_GAIN/P_SHIFT.
        frame: [n] int64 raw frame to read the substitute chunk from, or -1 to keep the row's own.
            Both LAG and P_TASK are realised this way -- a real chunk the policy really emitted
            somewhere, never a synthetic one, on the same principle that makes a_corr pose-derived.
        gain: [n] float32 multiplier on the pose block; 1.0 off the GAIN branch.
        shift: [n, 6] float32 Cartesian displacement (metres, then a rotation vector in radians)
            of the live pose on the SHIFT branch, zero elsewhere. The chunk is left as the policy
            emitted it; ChunkTargetBuilder moves ee_now by it, so a_corr becomes the recovery
            from where the arm is to the demo, and the reference, read from that same pose,
            agrees with a_corr.
    """

    kind: np.ndarray
    frame: np.ndarray
    gain: np.ndarray
    shift: np.ndarray

    def __len__(self) -> int:
        return int(self.kind.shape[0])

    def shift_rows(self) -> np.ndarray:
        """The rows whose policy chunk was displaced (kind P_SHIFT), ascending."""
        return np.flatnonzero(self.kind == P_SHIFT)

    def counts(self) -> dict[str, int]:
        """Rows per branch, for the log line that proves the sampler ran.

        Returns:
            {branch name: row count}, including branches with zero rows.
        """
        return {name: int((self.kind == k).sum()) for k, name in enumerate(POLICY_KIND_NAMES)}

    @classmethod
    def concat(cls, parts: "Sequence[PolicyCorruption]") -> "PolicyCorruption":
        """Stack the per-copy draws of one CopyPlan into one row-aligned corruption.

        Args:
            parts: One draw per copy, each over the same source rows in the same order.

        Returns:
            The concatenation, in copy-major order -- CopyPlan.rows's own order.
        """
        return cls(
            kind=np.concatenate([p.kind for p in parts]),
            frame=np.concatenate([p.frame for p in parts]),
            gain=np.concatenate([p.gain for p in parts]),
            shift=np.concatenate([p.shift for p in parts]),
        )

    def apply(self, a_policy: np.ndarray, cache: Any, chunk_len: int) -> np.ndarray:
        """Realise this corruption against a policy-action cache.

        Args:
            a_policy: [n, K, 7] the clean chunks, unchanged by this call.
            cache: robot_ood.models.fusion.target.PolicyActionCache, for the substituted frames.
            chunk_len: K.

        Returns:
            [n, K, 7] float32 the corrupted chunks.
        """
        out = np.array(a_policy, dtype=np.float32, copy=True)
        rows = np.flatnonzero(self.frame >= 0)
        if rows.size:
            out[rows] = cache.chunks(self.frame[rows], chunk_len).astype(np.float32)
        rows = np.flatnonzero(self.kind == P_GAIN)
        if rows.size:
            out[rows, :, :POSE_DIM] *= self.gain[rows, None, None]
        return out


@dataclass(eq=False)
class PolicyCorruptor:
    """Draws a PolicyCorruption for any query set over one fixed corpus.

    Attributes:
        nodes: The node table queries index into; supplies the phase-banded lookup P_TASK needs.
        picker: A RetrievalCorruptor over the same nodes, reused purely for that lookup rather than
            duplicating the rejection sampler.
        p_lag: Share of rows given the policy's own chunk from elsewhere on this demo.
        p_task: Share of rows given a chunk from a different task at a comparable phase.
        p_gain: Share of rows whose pose block is rescaled.
        p_shift: Share of rows whose live pose is displaced from the demos', SHIFT_M / SHIFT_RAD
            log-uniform, with the policy's chunk unchanged.
    """

    nodes: "NodeTable"
    picker: RetrievalCorruptor
    p_lag: float = 0.12
    p_task: float = 0.12
    p_gain: float = 0.12
    p_shift: float = 0.12

    def __post_init__(self) -> None:
        total = self.p_lag + self.p_task + self.p_gain + self.p_shift
        if not 0.0 <= total <= 1.0:
            raise ValueError(f"policy-corruption shares must sum into [0, 1], got {total}")
        self._raw_ptr = np.asarray(self.nodes.raw_ptr, dtype=np.int64)
        self._t_raw = np.asarray(self.nodes.t_raw, dtype=np.int64)

    def draw(
        self, tgt_node: np.ndarray, t_raw_now: np.ndarray, rng: np.random.Generator
    ) -> PolicyCorruption:
        """Assign a branch to every row and realise it.

        Args:
            tgt_node: [n] int64 label node of each query, for the P_TASK phase band.
            t_raw_now: [n] int64 the query window's own current raw frame.
            rng: Random state.

        Returns:
            The corruption, in query-set row order.
        """
        tgt = np.asarray(tgt_node, dtype=np.int64)
        now = np.asarray(t_raw_now, dtype=np.int64)
        n = int(tgt.shape[0])
        kind = self._assign(n, rng)
        frame = np.full(n, -1, dtype=np.int64)
        gain = np.ones(n, dtype=np.float32)
        shift = np.zeros((n, 6), dtype=np.float32)

        rows = np.flatnonzero(kind == P_LAG)
        frame[rows] = self._lag_frames(now[rows], rng)
        # A lag that clipped back onto the row's own frame is a no-op, and a no-op row labelled
        # corrupt would sit in the strata table pretending to be evidence.
        frame[rows[frame[rows] == now[rows]]] = -1
        rows = np.flatnonzero(kind == P_TASK)
        node = self.picker.pick_other_task(tgt[rows], rng)
        frame[rows] = np.where(node >= 0, self._t_raw[np.maximum(node, 0)], -1)
        rows = np.flatnonzero(kind == P_GAIN)
        gain[rows] = _gain_draw(rows.size, rng).astype(np.float32)
        rows = np.flatnonzero(kind == P_SHIFT)
        shift[rows] = RetrievalCorruptor._pick_offset(rows.size, rng, SHIFT_M, SHIFT_RAD)

        # A branch that found nothing, or that realised as a no-op, leaves the row clean rather
        # than being counted as a corruption that never happened.
        kind[np.flatnonzero((kind != P_CLEAN) & (kind != P_GAIN) & (kind != P_SHIFT) & (frame < 0))] = P_CLEAN
        return PolicyCorruption(kind=kind, frame=frame, gain=gain, shift=shift)

    def _assign(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw each row's branch from the configured shares.

        Args:
            n: Number of rows.
            rng: Random state.

        Returns:
            [n] int8 branch ids.
        """
        u = rng.random(n)
        edges = np.cumsum([self.p_lag, self.p_task, self.p_gain, self.p_shift])
        kind = np.full(n, P_CLEAN, dtype=np.int8)
        kind[u < edges[3]] = P_SHIFT
        kind[u < edges[2]] = P_GAIN
        kind[u < edges[1]] = P_TASK
        kind[u < edges[0]] = P_LAG
        return kind

    def _lag_frames(self, now: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """A frame LAG_FRAMES away on the same demo, clipped to that demo's own span.

        Args:
            now: [m] int64 each row's current raw frame.
            rng: Random state.

        Returns:
            [m] int64 substitute raw frames. Clipping keeps the substitute inside the demo, so a
            row near a boundary gets a smaller lag rather than another demo's action -- that case
            belongs to P_TASK and should not leak in here. A row clipped all the way back onto its
            own frame is dropped to P_CLEAN by the caller.
        """
        if now.size == 0:
            return np.zeros(0, dtype=np.int64)
        demo = np.searchsorted(self._raw_ptr, now, side="right") - 1
        lo, hi = self._raw_ptr[demo], self._raw_ptr[demo + 1] - 1
        step = _log_uniform(*LAG_FRAMES, now.size, rng)
        step = np.round(step * rng.choice([-1.0, 1.0], size=now.size)).astype(np.int64)
        return np.clip(now + step, lo, hi)


# ==== K fixed copies: the scale axis that costs no demonstrations ====
@dataclass(frozen=True)
class CopyPlan:
    """K fixed corruption copies of one source query set.

    Attributes:
        copies: K, at least 1. K=1 is the pre-existing single-draw behaviour.
        seed: The run seed the per-copy streams are derived from.
    """

    copies: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.copies) < 1:
            raise ValueError(f"CopyPlan: copies must be >= 1, got {self.copies}")

    def rows(self, n: int) -> np.ndarray:
        """Which source row each expanded row came from.

        Args:
            n: Source rows.

        Returns:
            [n * copies] int64, copy-major: copy 0's n rows, then copy 1's.
        """
        return np.tile(np.arange(n, dtype=np.int64), int(self.copies))

    def rng(self, copy: int) -> np.random.Generator:
        """The generator copy copy draws its corruption from.

        Args:
            copy: Copy index in [0, copies).

        Returns:
            A generator seeded by (seed, copy) and nothing else.
        """
        return np.random.default_rng(np.random.SeedSequence([int(self.seed), int(copy)]))


# ==== query phase: drawn from the belief filter's own measured residual ====

# The quantile levels robot_ood.eval.phase_skew records, in increasing order. Not a magnitude:
# these are the report's own KEYS, and reading a different set would read a different file.
_PHASE_LEVELS: tuple[tuple[str, float], ...] = (
    ("p5", 0.05), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p95", 0.95),
)


class BeliefPhaseResidual:
    """Samples phase_hat - phase_true from the measured per-bin residual distribution."""

    def __init__(self, graph_dir: str | Path) -> None:
        """Loads the per-bin residual quantiles from one graph's own replay report.

        Args:
            graph_dir: The artifact dir. The report is read out of g_head.npz, where train
                locator's post-save replay embeds it; a g_phase_skew.json sidecar (written by
                build skew) is the fallback when the head carries none. Each suite measures
                its own skew, so this is never a shared file.

        Raises:
            FileNotFoundError: That graph has no report in either place.
            ValueError: The report has no per-bin signed quantiles, or a bin's edges are unreadable.
        """
        data, source = self._load_report(Path(graph_dir))
        bins = data.get("by_phase_bin")
        if not bins:
            raise ValueError(
                f"BeliefPhaseResidual: {source} carries no 'by_phase_bin' block, so there is no "
                "per-bin residual to sample. Retrain the locator (or rerun build skew)."
            )
        self._path = source
        uppers: list[float] = []
        curves: list[np.ndarray] = []
        for key, stats in bins.items():
            signed = stats.get("signed")
            if signed is None or any(name not in signed for name, _ in _PHASE_LEVELS):
                raise ValueError(
                    f"BeliefPhaseResidual: {source} bin {key!r} has no complete 'signed' quantile "
                    f"set {[name for name, _ in _PHASE_LEVELS]}; this is not a report this class can read."
                )
            uppers.append(_bin_upper(key))
            curves.append(np.array([float(signed[name]) for name, _ in _PHASE_LEVELS], dtype=np.float64))
        order = np.argsort(uppers)
        self._upper = np.asarray(uppers, dtype=np.float64)[order]
        self._curve = np.stack(curves)[order]                      # [n_bins, len(_PHASE_LEVELS)]
        self._u = np.array([u for _, u in _PHASE_LEVELS], dtype=np.float64)

    @staticmethod
    def _load_report(root: Path) -> tuple[dict, Path]:
        """The replay report and where it came from: g_head.npz first, the sidecar file second.

        Args:
            root: The artifact dir.

        Returns:
            (the parsed report, the path it was read from).

        Raises:
            FileNotFoundError: Neither the head carries a report nor the sidecar file exists.
        """
        head = root / schema.HEAD_NPZ
        if head.is_file():
            with np.load(head, allow_pickle=False) as archive:
                if schema.PHASE_SKEW_KEY in archive.files:
                    return json.loads(str(archive[schema.PHASE_SKEW_KEY])), head
        sidecar = root / PHASE_SKEW_JSON
        if sidecar.is_file():
            with open(sidecar) as f:
                return json.load(f), sidecar
        raise FileNotFoundError(
            f"BeliefPhaseResidual: no replay report in {head} (key {schema.PHASE_SKEW_KEY!r}) and "
            f"no {sidecar}. Run `python -m robot_ood.cli train locator` -- its save embeds the "
            "report -- or build skew to write the sidecar; this class only reads the report, "
            "it does not run the replay itself."
        )

    @property
    def path(self) -> Path:
        """The report this was read from, for provenance in a run log."""
        return self._path

    def sample(self, phase_true: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Draw one residual per row.

        Args:
            phase_true: [n] any dtype; each row's true phase, which selects its bin.
            rng: The caller's random state, so a run's draws stay reproducible from its own seed.

        Returns:
            [n] float32 signed residuals, to be added to phase_true.
        """
        true = np.asarray(phase_true, dtype=np.float64).ravel()
        which = np.searchsorted(self._upper, true, side="left").clip(0, len(self._upper) - 1)
        u = rng.uniform(self._u[0], self._u[-1], size=true.shape[0])
        out = np.empty(true.shape[0], dtype=np.float64)
        for b in np.unique(which):
            rows = which == b
            out[rows] = np.interp(u[rows], self._u, self._curve[b])
        return out.astype(np.float32)

    def apply(self, phase_true: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Adds a sampled residual to phase_true, clipped to [0, 1].

        Args:
            phase_true: [n] each row's true phase.
            rng: The caller's random state.

        Returns:
            [n] float32 estimated phases.
        """
        true = np.asarray(phase_true, dtype=np.float32).ravel()
        return np.clip(true + self.sample(true, rng), 0.0, 1.0).astype(np.float32)


def _bin_upper(key: str) -> float:
    """The upper edge of a report bin key such as 'entry(<=0.05)', '0.05-0.25' or '0.75-1.0'.

    Args:
        key: The bin's key in the report's by_phase_bin mapping.

    Returns:
        Its upper edge.

    Raises:
        ValueError: The key carries no readable number.
    """
    # '-' is a SEPARATOR in these keys ('0.05-0.25'), never a sign: every bin edge is a phase in
    # [0, 1], so a negative edge would already be a report this parser should refuse.
    digits = "".join(c if (c.isdigit() or c == ".") else " " for c in key).split()
    edges = [float(d) for d in digits if d.strip(".")]
    if not edges:
        raise ValueError(
            f"BeliefPhaseResidual: cannot read an upper phase edge out of the bin key {key!r}. "
            "robot_ood.eval.phase_skew names its bins 'entry(<=0.05)', '0.05-0.25', ...; a "
            "report using other names needs this parser updated alongside it."
        )
    return max(edges)
