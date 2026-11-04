"""The measuring functions C0 runs. One class per procedure, each emitting named Constants.

Every class here answers one question about a corpus and nothing else, so a suspect number can be
re-derived on its own. None of them chooses a threshold: the quantiles are fields of
CalibrationSpec, which is written into the run's config record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from robot_ood.core.action import ActionScale, DEFAULT_MIN_R2, action_limit, fit_blocks
from robot_ood.core.constants import ASSUMED, MEASURED, RECORDED, Constant
from robot_ood.data.libero import DemoCorpus
from robot_ood.core.config import GraphConfig
from robot_ood.core.schemas import SEG_TIER1_M, SEG_TOPK, SEGMENT_LEN
from robot_ood.data import schema
from robot_ood.data.binning import phase_bins
from robot_ood.data.corpus import CorpusGeometry
from robot_ood.data.geometry import entry_perturbation_floor, leave_one_demo_out_floors
from robot_ood.data.nodes import NodeTable
from robot_ood.data.segments import SegmentBuilder
from robot_ood.models.types import (
    NodeCloud, PipelineParams, QueryMetricColumns, SeedQuery,
)
from robot_ood.train.align import dtw_cost

__all__ = [
    "ActionScaleFit",
    "CalibrationSpec",
    "DilationReach",
    "EntryPerturbation",
    "EntryStepCap",
    "NeighbourSpacing",
    "RobotInitMagnitude",
    "SeedSupport",
    "SegmentEntryStratum",
    "SegmentHardNegatives",
    "SegmentLen",
    "SegTier1M",
    "FutureTemp",
    "bisect_temperature",
    "mean_perplexity",
]



@dataclass(frozen=True, slots=True)
class CalibrationSpec:
    """Every choice a calibration run makes, as one recorded value object.

    Attributes:
        entry_lo_mult: Multiplier on the corpus's median leave-one-demo-out frame distance giving
            the off-manifold radius floor.
        support_mass: Cumulative mass a truncated support must retain.
        support_quantile: Quantile over the calibration query set of the node count that mass needs.
        n_queries: Calibration queries drawn for the support measurement.
        min_r2: Pre-registered floor on either action-scale block's R2.
        device: Torch device for the support measurement; CPU by default, since CUDA reductions
            are not bit-for-bit reproducible.
        direction_coherence_thresh: Mean cosine similarity below which a demo's action direction
            counts as decorrelated.
        direction_max_lag: Largest lag (raw frames) SegmentLen searches before giving up.
        future_temp_target_perplexity: exp(entropy) FutureTemp's softmax is fitted to match; None
            uses robot_ood.core.schemas.SEG_TOPK.
        hard_neg_max_pairs: Cap on segment pairs SegmentHardNegatives runs its DTW distance over.
        seg_entry_band: Phase ceiling defining the episode-start segment stratum.
        seg_entry_oversample: Factor multiplying the entry stratum's natural share to get its
            training share.
        seg_entry_frac_cap: Ceiling on SegmentEntryStratum's output share.
    """

    entry_lo_mult: float = 0.25
    support_mass: float = 0.99
    support_quantile: float = 0.95
    n_queries: int = 256
    min_r2: float = DEFAULT_MIN_R2
    device: str = "cpu"
    direction_coherence_thresh: float = 0.5
    direction_max_lag: int = 64
    future_temp_target_perplexity: float | None = None
    hard_neg_max_pairs: int = 4096
    seg_entry_band: float = 0.05
    seg_entry_oversample: float = 2.0
    seg_entry_frac_cap: float = 0.5


# ==== already type (b) -- gathered, not migrated ====
class ActionScaleFit:
    """Metres/radians per action unit, by least squares through the origin over every transition.

    Args:
        corpus: The corpus to read demos from.
        spec: Calibration settings; only min_r2 is read.
    """

    def __init__(self, corpus: DemoCorpus, spec: CalibrationSpec) -> None:
        self._corpus = corpus
        self._spec = spec
        pooled = corpus.pooled_deltas()
        self._pos, self._rot = fit_blocks(pooled.deltas)
        self._limit = action_limit(pooled.deltas)
        self._n_files, self._n_demos = pooled.n_files, pooled.n_demos
        self._n_transitions = len(pooled.deltas)

    @property
    def scale(self) -> ActionScale:
        """The fit as the value object robot_ood.runtime.ee_track.EeTrack consumes."""
        return ActionScale.from_blocks(self._pos, self._rot, pose_space=self._corpus.pose_space,
                                       limit=self._limit)

    def check(self) -> None:
        """Enforce the pre-registered R2 bar.

        Raises:
            ValueError: Either block fitted below spec.min_r2.
        """
        self.scale.check(self._spec.min_r2)

    def constants(self) -> dict[str, Constant]:
        """pos_scale and rot_scale, with each block's R2 and per-axis fit."""
        procedure = (
            "least squares through the origin of the measured end-effector change of every demo "
            "transition onto the commands of the RESPONSE_TAPS control steps ending in it; the "
            "scale is the summed gain and the rotation block composes in the world frame"
        )
        shared = {
            "n_transitions": self._n_transitions, "n_demos": self._n_demos,
            "n_hdf5_files": self._n_files,
        }
        return {
            "pos_scale": Constant(
                value=float(self._pos.scale), kind=MEASURED, procedure=procedure,
                stats={**shared, "unit": "m/action-unit", "r2": float(self._pos.r2),
                       "gains": list(self._pos.gains),
                       "axis_scale": [float(s) for s in self._pos.axis_scale]},
            ),
            "rot_scale": Constant(
                value=float(self._rot.scale), kind=MEASURED, procedure=procedure,
                stats={**shared, "unit": "rad/action-unit", "r2": float(self._rot.r2),
                       "gains": list(self._rot.gains),
                       "axis_scale": [float(s) for s in self._rot.axis_scale]},
            ),
        }


class EntryPerturbation:
    """The off-manifold radius floor, from the corpus's leave-one-demo-out frame distance.

    Args:
        nodes: Built node table.
        spec: Calibration settings; the floor multiplier is read.

    Raises:
        ValueError: Fewer than two demos, or a corpus with no frame another demo does not repeat.
    """

    def __init__(self, nodes: NodeTable, spec: CalibrationSpec) -> None:
        self._floor = entry_perturbation_floor(nodes)
        self._spec = spec

    def constants(self) -> dict[str, Constant]:
        """entry_perturb_lo, in radians."""
        procedure = (
            "leave-one-demo-out 1-NN distance from every raw frame no other demo repeats verbatim "
            "to the other demos' frames; lo = entry_lo_mult * p50"
        )
        stats = {**self._floor, "entry_lo_mult": self._spec.entry_lo_mult, "unit": "rad"}
        return {
            "entry_perturb_lo": Constant(
                value=self._spec.entry_lo_mult * self._floor["p50"], kind=MEASURED,
                procedure=procedure, stats=stats,
            ),
        }


class EntryStepCap:
    """entry_score_step_cap_rad: the corpus's own p99 per-control-step joint displacement.

    Args:
        nodes: Built node table.

    Raises:
        ValueError: Fewer than two demos, so no per-step displacement exists.
    """

    def __init__(self, nodes: NodeTable) -> None:
        raw_ptr = np.asarray(nodes.raw_ptr, dtype=np.int64)
        q_raw = np.asarray(nodes.q_raw, dtype=np.float64)
        if len(raw_ptr) - 1 < 2:
            raise ValueError("EntryStepCap: need >= 2 demos for a per-step displacement sample")
        steps = np.concatenate([
            np.linalg.norm(np.diff(q_raw[a:b], axis=0), axis=1)
            for a, b in zip(raw_ptr[:-1], raw_ptr[1:]) if b - a >= 2
        ])
        self._p99 = float(np.percentile(steps, 99))
        self._n_steps = int(steps.shape[0])

    def constants(self) -> dict[str, Constant]:
        """entry_score_step_cap_rad, in radians per control step."""
        procedure = (
            "p99 of ||q_t+1 - q_t|| over every consecutive raw-frame pair of every demo -- the "
            "corpus's own per-control-step joint displacement ceiling"
        )
        return {
            "entry_score_step_cap_rad": Constant(
                value=self._p99, kind=MEASURED, procedure=procedure,
                stats={"n_steps": self._n_steps, "unit": "rad"},
            ),
        }


class RobotInitMagnitude:
    """Robot-initialization range from benchmark variants, not corpus-measurable."""

    def constants(self) -> dict[str, Constant]:
        """entry_robotinit_lo_rad / entry_robotinit_hi_rad, in radians."""
        procedure = (
            "||init_qpos - nominal Panda|| over LIBERO-Plus's 500 generated Robot_Initial_States "
            "robot-variant classes; a 0.1-rad grid from 0.1 to 0.5, read off the generated tables"
        )
        stats = {"grid_step_rad": 0.1, "n_variants": 500, "unit": "rad"}
        return {
            "entry_robotinit_lo_rad": Constant(value=0.1, kind=RECORDED, procedure=procedure,
                                               stats=stats),
            "entry_robotinit_hi_rad": Constant(value=0.5, kind=RECORDED, procedure=procedure,
                                               stats=stats),
        }


class NeighbourSpacing:
    """How far apart independent demonstrations sit in joint space -- one number, two consumers.

    Args:
        nodes: Built node table.
    """

    def __init__(self, nodes: NodeTable) -> None:
        geometry = CorpusGeometry(nodes)
        self._spacing = float(geometry.spacing)
        self._n_nodes = int(len(nodes))

    @property
    def spacing(self) -> float:
        """The measured spacing, in radians."""
        return self._spacing

    def constants(self) -> dict[str, Constant]:
        """kernel_bw and edge_sigma_q -- the same measurement, named at both consumers."""
        procedure = (
            "median leave-one-demo-out nearest-neighbour distance between node configs "
            "(robot_ood.data.corpus.CorpusGeometry.spacing)"
        )
        stats = {"n_nodes": self._n_nodes, "unit": "rad"}
        return {
            "kernel_bw": Constant(
                value=self._spacing, kind=MEASURED, procedure=procedure, stats=stats),
            "edge_sigma_q": Constant(
                value=self._spacing, kind=MEASURED,
                procedure=procedure + "; the sibling/align Gaussian reads the same scale",
                stats=stats),
        }


class DilationReach:
    """The dilation schedule's constraint check: does message passing reach a whole demo?

    Args:
        nodes: Built node table.
        contract: The graph contract whose layers/dilations/coarsen are being checked.
    """

    def __init__(self, nodes: NodeTable, contract: schema.GraphContract = schema.DEFAULTS) -> None:
        raw_per_demo = np.diff(np.asarray(nodes.raw_ptr, dtype=np.int64))
        self._median_raw = float(np.median(raw_per_demo))
        self._contract = contract
        self._reach = int(contract.reach_raw_frames)

    @property
    def satisfied(self) -> bool:
        """Whether reach covers the median demo length."""
        return self._reach >= self._median_raw

    def constants(self) -> dict[str, Constant]:
        """reach_raw_frames, carrying the constraint's verdict in its stats."""
        return {
            "reach_raw_frames": Constant(
                value=self._reach,
                kind=MEASURED,
                procedure="layers * max(dilations) * coarsen, checked against the median demo "
                          "length in raw frames",
                stats={
                    "layers": int(self._contract.network.layers),
                    "max_dilation": int(max(self._contract.vocab.dilations)),
                    "coarsen": int(self._contract.build.coarsen),
                    "median_raw_frames_per_demo": self._median_raw,
                    "satisfied": self.satisfied,
                },
            )
        }


class SeedSupport:
    """How many nodes actually carry the mass -- the two top-K truncations, measured.

    Args:
        corpus: The corpus, providing the graph directory and node table.
        spec: Calibration settings.
        kernel_bw: Bandwidth the seeding kernel is measured under; the calibrated value, so C0's
            own output is self-consistent.
    """

    def __init__(self, corpus: DemoCorpus, spec: CalibrationSpec, *, kernel_bw: float) -> None:
        self._corpus = corpus
        self._spec = spec
        self._kernel_bw = float(kernel_bw)
        self._seed_counts: np.ndarray | None = None
        self._post_counts: np.ndarray | None = None

    @property
    def measurable(self) -> bool:
        """Whether the graph carries a trained head to run the query set through."""
        return self._corpus.head_path is not None

    def run(self) -> None:
        """Run the calibration query set and record both support-size distributions."""
        from robot_ood.runtime.retrieve import GraphRetriever

        spec, nodes = self._spec, self._corpus.nodes
        cfg = GraphConfig(device=spec.device)
        retriever = GraphRetriever.load(self._corpus.graph_dir, cfg=cfg, device=spec.device)
        node_cloud = NodeCloud(
            q_nodes_n=retriever.node_cloud.q_nodes_n,
            n_v=torch.ones_like(retriever.node_cloud.n_v),
            columns=retriever.node_cloud.columns,
        )
        e_q_ones = torch.ones(1, device=retriever.device)
        seed_counts, post_counts = [], []
        for q_hist in _calibration_windows(nodes, cfg.hist, spec.n_queries):
            window = retriever.preprocessor.build(q_hist, None)
            qh = torch.as_tensor(window.q, dtype=torch.float32, device=retriever.device)
            vh = torch.as_tensor(window.vel, dtype=torch.float32, device=retriever.device)
            gh = torch.as_tensor(window.grip, dtype=torch.float32, device=retriever.device)
            with torch.no_grad():
                # All-ones e_q/n_v with topk = tier1_m = V recovers the seed's raw amplitude a
                # untruncated -- this is still the deployed seed, not a reimplementation.
                enc = retriever.net.enc.encode(qh, vh, gh)
                h0 = retriever.net.seed(
                    SeedQuery(
                        e_q=e_q_ones, z_h=enc.z, pool_w=enc.pool_w,
                        qh_n=retriever.net.normalize_q(qh),
                        columns=QueryMetricColumns(qdot=vh, grip=gh),
                    ),
                    node_cloud,
                    PipelineParams(topk=len(nodes), tier1_m=len(nodes)),
                )[0]
                assert torch.equal(h0, h0[..., :1].expand_as(h0)), (
                    "seed's h^0 is not constant across hidden columns -- the all-ones e_q/n_v "
                    "broadcast this measure depends on no longer holds"
                )
                activation = h0[..., 0]
                posterior = torch.softmax(retriever.observe(q_hist).logits64, dim=-1)
            seed_counts.append(_nodes_for_mass(activation.double(), spec.support_mass))
            post_counts.append(_nodes_for_mass(posterior, spec.support_mass))
        self._seed_counts = np.asarray(seed_counts, dtype=np.int64)
        self._post_counts = np.asarray(post_counts, dtype=np.int64)

    def constants(self) -> dict[str, Constant]:
        """seed_topk and topm -- measured when a trained head exists, else left as type (a)."""
        if not self.measurable:
            note = (
                f"NOT MEASURED: no {schema.HEAD_NPZ} beside the graph, so no posterior exists to "
                "count support over. Train the retrieval head and re-run C0."
            )
            return {
                "seed_topk": Constant(schema.SEED_TOPK, ASSUMED, note),
                "topm": Constant(GraphConfig().topm, ASSUMED, note),
            }
        if self._seed_counts is None or self._post_counts is None:
            self.run()
        procedure = (
            f"quantile {self._spec.support_quantile} over {self._spec.n_queries} calibration "
            f"queries of the node count needed to reach {self._spec.support_mass} cumulative mass"
        )
        return {
            "seed_topk": Constant(
                value=_quantile_count(self._seed_counts, self._spec.support_quantile),
                kind=MEASURED, procedure=procedure + ", on the seeding kernel's activation",
                stats={**_count_stats(self._seed_counts), "kernel_bw": self._kernel_bw},
            ),
            "topm": Constant(
                value=_quantile_count(self._post_counts, self._spec.support_quantile),
                kind=MEASURED, procedure=procedure + ", on the readout posterior",
                stats=_count_stats(self._post_counts),
            ),
        }


# ==== FCSR (C1 segment memory) calibration ====
class SegmentLen:
    """Raw-frame span over which a demo's action direction stays autocorrelated.

    Args:
        nodes: Built node table.
        spec: Calibration settings; direction_coherence_thresh and direction_max_lag are read.

    Raises:
        ValueError: No demo has >= 2 raw frames, so no lag is measurable.
    """

    def __init__(self, nodes: NodeTable, spec: CalibrationSpec) -> None:
        a = np.asarray(nodes.a_raw, dtype=np.float64)[:, :3]
        norm = np.linalg.norm(a, axis=1)
        valid = norm > 1e-9
        direction = np.zeros_like(a)
        direction[valid] = a[valid] / norm[valid, None]

        raw_ptr = np.asarray(nodes.raw_ptr, dtype=np.int64)
        owner_raw = np.repeat(np.arange(len(raw_ptr) - 1), np.diff(raw_ptr))

        max_lag = min(spec.direction_max_lag, len(a) - 1)
        if max_lag < 1:
            raise ValueError("SegmentLen: no demo has >= 2 raw frames to measure a lag over")

        lags = np.arange(1, max_lag + 1)
        cos_by_lag = np.full(len(lags), np.nan, dtype=np.float64)
        for i, lag in enumerate(lags):
            same_demo = owner_raw[:-lag] == owner_raw[lag:]
            paired = valid[:-lag] & valid[lag:] & same_demo
            if not paired.any():
                continue
            cos_by_lag[i] = float(
                np.einsum("ij,ij->i", direction[:-lag][paired], direction[lag:][paired]).mean()
            )

        below = np.flatnonzero(cos_by_lag < spec.direction_coherence_thresh)
        self._coherence_len = int(lags[below[0]]) if below.size else int(lags[-1])
        self._lags, self._cos_by_lag, self._spec = lags, cos_by_lag, spec

    def constants(self) -> dict[str, Constant]:
        """segment_len, in raw frames."""
        default_idx = SEGMENT_LEN - 1
        cos_at_default = (
            float(self._cos_by_lag[default_idx]) if default_idx < len(self._cos_by_lag)
            else float("nan")
        )
        return {
            "segment_len": Constant(
                value=self._coherence_len, kind=MEASURED,
                procedure=(
                    "smallest lag L, raw frames, at which the corpus-wide mean cosine similarity "
                    "between the translation-channel action direction at t and t+L (same demo) "
                    f"drops below {self._spec.direction_coherence_thresh}"
                ),
                stats={
                    "coherence_thresh": self._spec.direction_coherence_thresh,
                    "max_lag": int(self._lags[-1]),
                    "architectural_default": SEGMENT_LEN,
                    "cos_at_architectural_default": cos_at_default,
                },
            )
        }


def mean_perplexity(score: np.ndarray, temp: float) -> float:
    """exp(entropy(softmax(score / temp))), averaged over score's leading (query) axis.

    Args:
        score: [..., S] logits/scores, any finite real scale.
        temp: Softmax temperature, > 0.

    Returns:
        Mean perplexity over the leading axis/axes.
    """
    s = score / temp
    s = s - np.max(s, axis=-1, keepdims=True)
    p = np.exp(s)
    p = p / p.sum(axis=-1, keepdims=True)
    # log(0) is masked out rather than let through and caught by np.where: np.where still evaluates
    # both branches eagerly, so a zero-probability tail warns on every call otherwise.
    safe_p = np.where(p > 0, p, 1.0)
    entropy = -np.sum(np.where(p > 0, p * np.log(safe_p), 0.0), axis=-1)
    return float(np.mean(np.exp(entropy)))


def bisect_temperature(
    score: np.ndarray, target: float, *, lo: float = 1e-4, hi: float = 1e4, iters: int = 60,
) -> tuple[float, float]:
    """Geometric bisection on T so mean_perplexity(score, T) approaches target perplexity.

    Args:
        score: [..., S] logits/scores, any finite real scale.
        target: Perplexity to hit.
        lo: Bracket floor.
        hi: Bracket ceiling.
        iters: Bisection steps; 60 resolves the [1e-4, 1e4] default bracket to ~1e-18 relative
            precision, far past floating-point noise.

    Returns:
        (temp, achieved_perplexity) at the bisection's final midpoint.
    """
    for _ in range(iters):
        mid = math.sqrt(lo * hi)              # geometric bisection: T spans many decades
        if mean_perplexity(score, mid) < target:
            lo = mid
        else:
            hi = mid
    temp = math.sqrt(lo * hi)
    return temp, mean_perplexity(score, temp)


class FutureTemp:
    """Softmax temperature over segment compatibility scores -- PROXY-SCORE bisection, superseded.

    Args:
        nodes: Built node table.
        spec: Calibration settings; n_queries and future_temp_target_perplexity are read.

    Raises:
        ValueError: Fewer than 2 segments exist to score against.
    """

    def __init__(self, nodes: NodeTable, spec: CalibrationSpec) -> None:
        # proxy score geometry cannot reach target perplexity on real corpora;
        # use the trained scorer from robot_ood.train.tasks.fcsr.FcsrTask instead.
        segments = SegmentBuilder(nodes).build()
        if segments.n_segments < 2:
            raise ValueError("FutureTemp: need >= 2 segments to score a query against")

        stride = max(segments.n_segments // spec.n_queries, 1)
        query_idx = np.arange(0, segments.n_segments, stride)[: spec.n_queries]
        pos = segments.ee_start[:, :3].astype(np.float64)
        d2 = np.sum((pos[query_idx, None, :] - pos[None, :, :]) ** 2, axis=-1)  # [Q, S]
        rows = np.arange(len(query_idx))
        d2[rows, query_idx] = np.inf  # a query never scores against itself
        self._score = -d2
        self._target = (
            float(spec.future_temp_target_perplexity) if spec.future_temp_target_perplexity
            is not None else float(SEG_TOPK)
        )

    # Relative tolerance the bisection's achieved perplexity must clear against target_perplexity.
    _CONVERGENCE_RTOL: float = 0.15

    def constants(self) -> dict[str, Constant]:
        """future_temp, fitted by bisection.

        Raises:
            ValueError: The bisection did not converge -- the achieved perplexity sits more than
                _CONVERGENCE_RTOL away from the target, meaning the loop bottomed out at a bracket
                endpoint (1e-4 or 1e4) instead of finding a T that actually matches. This proxy score
                (geometric start-pose L2, not the trained SegmentScorer) is not guaranteed to be able
                to reach every target perplexity, and reporting the endpoint as if it were a fit would
                fabricate a MEASURED constant that was never actually measured.
        """
        temp, achieved = bisect_temperature(self._score, self._target)
        if not math.isclose(achieved, self._target, rel_tol=self._CONVERGENCE_RTOL):
            raise ValueError(
                f"FutureTemp: bisection over T in [1e-4, 1e4] did NOT converge -- achieved mean "
                f"perplexity {achieved:.2f} against target {self._target:.2f} "
                f"(rel_tol={self._CONVERGENCE_RTOL}), landing at T={temp:.6g}. The proxy score "
                "(negative squared start-pose L2) cannot reach the target perplexity anywhere in the "
                "swept range on this corpus, so the bisection bottomed out at a bracket endpoint "
                "rather than finding a real fit. This is NOT a valid future_temp -- do not report "
                "the bracket endpoint as MEASURED. Use the trained SegmentScorer's own logit "
                "distribution instead: robot_ood.train.tasks.fcsr.FcsrTask fits future_temp "
                "from held-out queries after S1' training and stamps it into the FCSR "
                "checkpoint, which robot_ood.models.retrieval.retriever.SegmentRetriever.load reads directly."
            )
        return {
            "future_temp": Constant(
                value=temp, kind=MEASURED,
                procedure=(
                    "geometric bisection on T so the mean perplexity of softmax(-||ee_start_query - "
                    "ee_start_candidate||^2 / T) over a held-out query sample matches "
                    "future_temp_target_perplexity, a proxy score standing in for C2's untrained "
                    "SegmentScorer"
                ),
                stats={
                    "target_perplexity": self._target,
                    "n_queries": int(self._score.shape[0]),
                    "n_segments": int(self._score.shape[1]),
                    "achieved_perplexity": achieved,
                },
            )
        }


class SegmentHardNegatives:
    """p90 DTW distance between action chunks of segments sharing a phase bin but differing task.

    Args:
        nodes: Built node table.
        spec: Calibration settings; hard_neg_max_pairs is read.

    Raises:
        ValueError: No same-phase-bin, cross-task segment pair exists.
    """

    def __init__(self, nodes: NodeTable, spec: CalibrationSpec, *, seed: int = 0) -> None:
        segments = SegmentBuilder(nodes).build()
        pbin = phase_bins(segments.phase_start.astype(np.float64), schema.NBINS_ALIGN)
        task_id = segments.task_id.astype(np.int64)

        order = np.argsort(pbin, kind="stable")
        bounds = np.flatnonzero(np.diff(pbin[order])) + 1
        pair_a: list[np.ndarray] = []
        pair_b: list[np.ndarray] = []
        for group in np.split(order, bounds):
            if len(group) < 2:
                continue
            gi, gj = np.meshgrid(group, group, indexing="ij")
            keep = (gi < gj) & (task_id[gi] != task_id[gj])
            pair_a.append(gi[keep])
            pair_b.append(gj[keep])

        a_idx = np.concatenate(pair_a) if pair_a else np.zeros(0, dtype=np.int64)
        b_idx = np.concatenate(pair_b) if pair_b else np.zeros(0, dtype=np.int64)
        if a_idx.size == 0:
            raise ValueError(
                "SegmentHardNegatives: no same-phase-bin, cross-task segment pair exists in this "
                "corpus -- it has too few tasks or too few phase bins populated per task"
            )
        self._n_pairs_total = int(a_idx.size)

        rng = np.random.RandomState(seed)
        if a_idx.size > spec.hard_neg_max_pairs:
            keep = rng.choice(a_idx.size, size=spec.hard_neg_max_pairs, replace=False)
            a_idx, b_idx = a_idx[keep], b_idx[keep]

        a = torch.as_tensor(segments.action_chunk[a_idx, :, :3], dtype=torch.float64)
        b = torch.as_tensor(segments.action_chunk[b_idx, :, :3], dtype=torch.float64)
        self._dist = dtw_cost(a, b).numpy()

    def constants(self) -> dict[str, Constant]:
        """seg_hard_neg_thresh, in the action command's own translation units."""
        quantiles = {f"p{p}": float(np.percentile(self._dist, p)) for p in (10, 50, 90, 99)}
        return {
            "seg_hard_neg_thresh": Constant(
                value=quantiles["p90"], kind=MEASURED,
                procedure=(
                    "p90 of the mean-per-step DTW distance (robot_ood.train.align.dtw_cost) between "
                    "the translation channel of action_chunk pairs sharing a phase bin but differing "
                    "task_id"
                ),
                stats={
                    **quantiles, "n_pairs_sampled": int(self._dist.size),
                    "n_pairs_total": self._n_pairs_total, "unit": "action command units",
                },
            )
        }


class SegmentEntryStratum:
    """seg_entry_stratum_frac: the SEGMENT curriculum's oversampled share of entry-region rows.

    Args:
        nodes: Built node table.
        spec: Calibration settings; seg_entry_band, seg_entry_oversample and seg_entry_frac_cap are
            read.

    Raises:
        ValueError: No segment exists to measure a share over.
    """

    def __init__(self, nodes: NodeTable, spec: CalibrationSpec) -> None:
        segments = SegmentBuilder(nodes).build()
        if segments.n_segments == 0:
            raise ValueError("SegmentEntryStratum: no segment to measure an entry share over")
        phase = segments.phase_start.astype(np.float64)
        self._band = float(spec.seg_entry_band)
        self._oversample = float(spec.seg_entry_oversample)
        self._cap = float(spec.seg_entry_frac_cap)
        self._natural_share = float(np.mean(phase <= self._band))
        self._n_segments = segments.n_segments

    def constants(self) -> dict[str, Constant]:
        """seg_entry_stratum_frac, in [0, seg_entry_frac_cap]."""
        target = min(self._cap, self._oversample * self._natural_share)
        return {
            "seg_entry_stratum_frac": Constant(
                value=target, kind=MEASURED,
                procedure=(
                    "min(seg_entry_frac_cap, seg_entry_oversample * natural_share), natural_share = "
                    "mean(segment.phase_start <= seg_entry_band) over this corpus's segment table"
                ),
                stats={
                    "natural_share": self._natural_share, "seg_entry_band": self._band,
                    "seg_entry_oversample": self._oversample, "seg_entry_frac_cap": self._cap,
                    "n_segments": self._n_segments,
                },
            )
        }


class SegTier1M:
    """Tier-1 segment-retrieval candidate count."""

    def constants(self) -> dict[str, Constant]:
        """seg_tier1_m."""
        note = (
            "Snapshot from one trained checkpoint's deployed own-in-top-8 curve (see "
            "robot_ood.core.schemas.SEG_TIER1_M), not a per-suite measurement this class computes -- "
            "re-run that curve for a new suite or checkpoint rather than trusting this default."
        )
        return {"seg_tier1_m": Constant(SEG_TIER1_M, ASSUMED, note)}


# ==== helpers ====
def _calibration_windows(nodes: NodeTable, hist: int, n_queries: int) -> Sequence[np.ndarray]:
    """Deterministic clean query windows: hist consecutive raw frames ending at strided nodes.

    Args:
        nodes: Built node table.
        hist: Window length in raw frames.
        n_queries: How many windows to draw.

    Returns:
        Up to n_queries [<=hist, D] float64 windows, oldest row first. No RNG: the node indices are
        a fixed stride over the table, which is what makes the measurement reproducible.
    """
    raw_ptr = np.asarray(nodes.raw_ptr, dtype=np.int64)
    q_raw = np.asarray(nodes.q_raw, dtype=np.float64)
    owner = np.asarray(nodes.owner, dtype=np.int64)
    t_raw = np.asarray(nodes.t_raw, dtype=np.int64)
    total = len(nodes)
    stride = max(total // int(n_queries), 1)
    windows = []
    for v in range(0, total, stride)[: int(n_queries)]:
        # t_raw indexes the CONCATENATED q_raw, not the demo -- raw_ptr[owner] is only the floor
        # that stops a short window from reaching back into the previous demo.
        end = int(t_raw[v]) + 1
        windows.append(q_raw[max(int(raw_ptr[owner[v]]), end - int(hist)) : end])
    return windows


def _nodes_for_mass(distribution: torch.Tensor, mass: float) -> int:
    """How many of the largest entries are needed to reach mass of the total.

    Args:
        distribution: [V] non-negative, not necessarily normalised. Non-finite entries (the -inf
            logits' softmax leaves none, but a dead seed row can) count as zero.
        mass: Cumulative-mass target.

    Returns:
        The smallest K whose top-K sum reaches mass, at least 1.
    """
    values = torch.nan_to_num(distribution, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    total = float(values.sum())
    if total <= 0.0:
        return 1
    ordered = torch.sort(values, descending=True).values
    fraction = torch.cumsum(ordered, dim=0) / total
    return int((fraction < mass).sum().item()) + 1


def _quantile_count(counts: np.ndarray, quantile: float) -> int:
    """The quantile of a support-size distribution, rounded UP to a whole node."""
    return int(math.ceil(float(np.quantile(counts.astype(np.float64), quantile))))


def _count_stats(counts: np.ndarray) -> dict[str, Any]:
    """Percentiles of one support-size distribution, for the constant's stats block."""
    return {
        "n_queries": int(counts.size),
        **{f"p{p}": float(np.percentile(counts, p)) for p in (50, 90, 95, 99)},
        "max": int(counts.max()),
    }
