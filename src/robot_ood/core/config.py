"""Typed knobs for graph, sentinel and entry, read from the environment and the C0 calibration.

The filesystem roots that used to sit beside these live in robot_ood.core.paths.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

from robot_ood.core.env import env_bool, env_float, env_int, env_str
from robot_ood.core import schemas as fcsr
from robot_ood.core.constants import JOINT_TOL, STILL_TOL
from robot_ood.data import schema as graph_schema

__all__ = ["ABLATIONS", "Ablation", "EntryConfig", "GraphConfig", "RADIUS_TO_SIGMA", "SentinelConfig", "calibrated",
           "calibrated_constants", "entry_sigma_hi", "entry_sigma_lo", "plan_rows", "set_constants",
           "warn_uncalibrated"]

# Rows a policy executes per query, when no benchmark protocol says otherwise.
_POLICY_CHUNK = 8

ABLATIONS = ("graph", "memory", "entry", "corrector")
_ABLATE_ENV = "ROBOT_OOD_ABLATE"


@dataclass(frozen=True)
class Ablation:
    graph: bool = False
    memory: bool = False
    entry: bool = False
    corrector: bool = False

    @classmethod
    def from_env(cls) -> "Ablation":
        names = [n.strip() for n in env_str(_ABLATE_ENV).split(",") if n.strip()]
        unknown = sorted(set(names) - set(ABLATIONS))
        if unknown:
            raise ValueError(f"{_ABLATE_ENV} names {unknown}; known: {list(ABLATIONS)}")
        return cls(**{n: True for n in names})

    def __str__(self) -> str:
        return ",".join(n for n in ABLATIONS if getattr(self, n)) or "none"

_TRAIN_WINDOW_STRIDE = 1


def _known_fields(cls: type, data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Filter a mapping down to cls's declared dataclass fields.

    Args:
        cls: A dataclass type.
        data: Raw mapping, possibly carrying unknown keys, possibly None.

    Returns:
        The mapping restricted to real fields, so a stale key in a YAML file is ignored rather than
        raising TypeError from the constructor.
    """
    return {k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__}



# The calibration a training run measured for itself and installed via set_constants. It outranks
# the environment: the run that measured it is the one process that must not read anything else.
_installed_constants: Any = None


def set_constants(constants: Any) -> None:
    """Install (or with None, clear) the process's active C0 calibration.

    Args:
        constants: A robot_ood.core.constants.Constants, or None to fall back to the resolution
            chain below.

    train fcsr measures C0 itself and installs the result here before any consumer reads a
    constant, then embeds the same bytes in its checkpoint -- so training and the deploy that
    later loads that checkpoint read the identical calibration.
    """
    global _installed_constants
    _installed_constants = constants
    calibrated_constants.cache_clear()


@lru_cache(maxsize=1)
def calibrated_constants() -> Any:
    """The process's C0 calibration (robot_ood.core.constants.Constants), or None.

    Returns:
        The first of: what set_constants installed (a training run's own in-process measurement),
        the $ROBOT_OOD_CONSTANTS file, the calibration train fcsr embedded in $GR_GRAPH_DIR's
        g_fcsr.npz -- why a deploy needs only the artifact directory -- or None.

    Raises:
        ValueError: A resolved file is stamped with an unknown schema version or fails its own hash.
    """
    if _installed_constants is not None:
        return _installed_constants
    path = env_str("ROBOT_OOD_CONSTANTS")
    if path:
        from robot_ood.core.constants import Constants
        return Constants.from_json(path)
    return _embedded_constants()


def _embedded_constants(graph_dir: "str | os.PathLike | None" = None) -> Any:
    """The calibration inside graph_dir's ($GR_GRAPH_DIR's) g_fcsr.npz, or None when there is none to read."""
    graph_dir = os.fspath(graph_dir) if graph_dir else env_str("GR_GRAPH_DIR")
    if not graph_dir:
        return None
    path = os.path.join(graph_dir, graph_schema.FCSR_NPZ)
    if not os.path.isfile(path):
        return None
    import json

    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        if graph_schema.CONSTANTS_KEY not in archive.files:
            return None
        data = json.loads(str(archive[graph_schema.CONSTANTS_KEY]))
    from robot_ood.core.constants import Constants
    return Constants.from_dict(data)


_uncalibrated_warned: set[str] = set()


def warn_uncalibrated(key: str, default: Any) -> None:
    """Emit ONE unmissable warning per process when key falls back to its hardcoded default.

    Args:
        key: The constant's name, as it would appear in constants.json.
        default: The hardcoded value being substituted.
    """
    if key in _uncalibrated_warned:
        return
    _uncalibrated_warned.add(key)
    import warnings
    warnings.warn(
        f"C0 calibration not applied to {key!r}: using the hardcoded default {default!r} instead "
        f"of a corpus-measured value. Run `python -m robot_ood.cli build calibrate --suite <suite>` and point "
        "ROBOT_OOD_CONSTANTS at the result to pick this up. (Fires once per process per key -- see "
        "robot_ood.core.config.warn_uncalibrated.)",
        stacklevel=3,
    )


def calibrated(key: str, default: Any, graph_dir: "str | os.PathLike | None" = None) -> Any:
    """The calibrated value for key, or default -- warning once per key when it falls back.

    Args:
        key: The constant's name.
        default: What to return when no calibration carries the key.
        graph_dir: An artifact directory whose embedded calibration is read instead of the
            process's; a deploy with several graphs reads each one's own.
    """
    constants = _embedded_constants(graph_dir) if graph_dir else calibrated_constants()
    if constants is None or key not in constants:
        warn_uncalibrated(key, default)
        return default
    return constants.value(key, default)



# ==== Demonstration-graph knobs (Track GR) ====
@dataclass
class GraphConfig:
    """Demonstration-graph build/train/readout knobs (Track GR)."""

    graph_dir: str = ""
    coarsen: int = graph_schema.COARSEN
    dilations: tuple[int, ...] = graph_schema.DILATIONS
    k_sibling: int = graph_schema.K_SIBLING
    k_align: int = graph_schema.K_ALIGN
    virtual_per_entry: int = graph_schema.VIRTUAL_PER_ENTRY
    hidden: int = graph_schema.HIDDEN
    layers: int = graph_schema.LAYERS
    agg: str = graph_schema.AGG
    hist: int = graph_schema.HIST_H
    seg_k: int = graph_schema.SEG_K
    seed_topk: int = graph_schema.SEED_TOPK
    seed_tier1_m: int = graph_schema.SEED_TIER1_M
    idf: float = 1.0
    entry_band: float = 0.0

    seg_topk: int = fcsr.SEG_TOPK
    seg_tier1_m: int = fcsr.SEG_TIER1_M
    future_temp: float = 1.0
    action_hist: bool = True

    temp: float = 1.0
    topm: int = 32
    device: str = ""
    move_cost: bool = True
    joint_tol: float = JOINT_TOL
    query_copies: int = 1
    qbatch: int = 8

    @classmethod
    def from_env(cls, graph_dir: "str | os.PathLike | None" = None) -> "GraphConfig":
        """Build from the GR_* environment variables; every default mirrors the field above.

        Args:
            graph_dir: The artifact directory whose embedded calibration the calibrated knobs
                default to, and the config's graph_dir; None reads $GR_GRAPH_DIR for both.
        """
        return cls(
            graph_dir=os.fspath(graph_dir) if graph_dir else env_str("GR_GRAPH_DIR"),
            coarsen=env_int("GR_COARSEN", graph_schema.COARSEN),
            k_sibling=env_int("GR_K_SIBLING", graph_schema.K_SIBLING),
            k_align=env_int("GR_K_ALIGN", graph_schema.K_ALIGN),
            virtual_per_entry=env_int("GR_VIRTUAL_PER_ENTRY", graph_schema.VIRTUAL_PER_ENTRY),
            hidden=env_int("GR_HIDDEN", graph_schema.HIDDEN),
            layers=env_int("GR_LAYERS", graph_schema.LAYERS),
            agg=env_str("GR_AGG", graph_schema.AGG),
            hist=env_int("GR_HIST", graph_schema.HIST_H),
            seg_k=env_int("GR_SEG_K", graph_schema.SEG_K),
            seed_topk=env_int("GR_SEED_TOPK", calibrated("seed_topk", graph_schema.SEED_TOPK, graph_dir)),
            seed_tier1_m=env_int("GR_SEED_TIER1_M", graph_schema.SEED_TIER1_M),
            idf=env_float("GR_IDF", 1.0),
            entry_band=env_float("GR_ENTRY_BAND", 0.0),
            temp=env_float("GR_TEMP", 1.0),
            topm=env_int("GR_TOPM", calibrated("topm", 32, graph_dir)),
            seg_topk=env_int("GR_SEG_TOPK", fcsr.SEG_TOPK),
            seg_tier1_m=env_int("GR_SEG_TIER1_M", calibrated("seg_tier1_m", fcsr.SEG_TIER1_M, graph_dir)),
            future_temp=env_float("GR_FUTURE_TEMP", calibrated("future_temp", 1.0, graph_dir)),
            action_hist=env_bool("GR_ACTION_HIST", True),
            device=env_str("GR_DEVICE"),
            move_cost=env_bool("GR_MOVE_COST", True),
            query_copies=env_int("GR_QUERY_COPIES", 1),
            qbatch=env_int("GR_QBATCH", 8),
            joint_tol=env_float("GR_JOINT_TOL", JOINT_TOL),
        )


# ==== Sentinel knobs (Track SENTINEL) ====
def plan_rows(policy_chunk: int, seg_frames: int) -> int:
    """Rows one plan may cover: the largest divisor of the chunk that fits inside a segment.

    Args:
        policy_chunk: Rows the policy executes per query.
        seg_frames: Demo frames one reference segment spans (data.segments.segment_frames).

    Returns:
        The row count; a divisor of policy_chunk, at least one.

    Raises:
        ValueError: Either argument is below one.
    """
    # A plan's reference is one segment spread over the plan's rows, so a plan longer than its
    # segment asks the arm to cover the segment's distance in more steps than the demo took.
    if policy_chunk < 1 or seg_frames < 1:
        raise ValueError(f"plan_rows needs positive arguments, got {policy_chunk=} {seg_frames=}")
    return max(n for n in range(1, policy_chunk + 1) if policy_chunk % n == 0 and n <= seg_frames)


@dataclass
class SentinelConfig:
    """Runtime out-of-support monitor for a frozen policy (Track SENTINEL)."""

    policy_chunk: int = _POLICY_CHUNK
    plan_rows: int = _POLICY_CHUNK
    graph_dir: str = ""
    graph_win: int = 8
    graph_stride: int = 0
    graph_topm: int = 32
    graph_temp: float = 1.0
    graph_device: str = ""
    action_scale_path: str = ""
    ablation: Ablation = Ablation()

    def __post_init__(self) -> None:
        """Resolve graph_stride to _TRAIN_WINDOW_STRIDE if 0; enforce window/chunk invariants.

        Raises:
            ValueError: graph_stride not in {0, _TRAIN_WINDOW_STRIDE}, or policy_chunk not
                divisible by plan_rows.
        """
        if self.graph_stride == 0:
            self.graph_stride = _TRAIN_WINDOW_STRIDE
        elif self.graph_stride != _TRAIN_WINDOW_STRIDE:
            raise ValueError(
                f"SentinelConfig.graph_stride={self.graph_stride} != {_TRAIN_WINDOW_STRIDE}: a training "
                f"window is {_TRAIN_WINDOW_STRIDE} raw frames apart per row, and "
                "robot_ood.train.loop's finite_diff_vel differences neighbouring rows, so any other "
                "spacing hands the network a window shape it never trained on. Leave graph_stride 0, "
                "which resolves to the training stride."
            )
        if self.policy_chunk < 1:
            raise ValueError(f"SentinelConfig.policy_chunk={self.policy_chunk} must be positive")
        if self.plan_rows < 1 or self.policy_chunk % self.plan_rows != 0:
            raise ValueError(
                f"SentinelConfig.policy_chunk={self.policy_chunk} is not a multiple of "
                f"plan_rows={self.plan_rows}: a check re-plans the rows of the policy's chunk "
                "that have not been executed yet, so its period must divide the chunk."
            )

    @classmethod
    def from_env(cls, policy_chunk: int = _POLICY_CHUNK) -> "SentinelConfig":
        """Build from the SN_* environment variables, around the rows the policy executes per query.

        Args:
            policy_chunk: Rows of one policy chunk. How many of them one plan covers is the
                graph's call, not an environment knob; see plan_rows.

        Returns:
            The config.

        Raises:
            ValueError: SN_BLEND_SCALE is set (this knob is deleted).
        """
        if "SN_BLEND_SCALE" in os.environ:
            raise ValueError(
                "SN_BLEND_SCALE is deleted. It was one global throttle standing in for a per-check "
                "question -- is the demonstration or the policy the wrong operand -- which the head "
                "now answers itself, through the corrector's trust head "
                "(robot_ood.models.fusion.corrector). Unset it."
            )
        return cls(
            policy_chunk=policy_chunk,
            plan_rows=policy_chunk,
            graph_dir=env_str("SN_GRAPH_DIR"),
            graph_win=env_int("SN_GRAPH_WIN", 8),
            graph_stride=env_int("SN_GRAPH_STRIDE", 0),
            graph_topm=env_int("SN_GRAPH_TOPM", 32),
            graph_temp=env_float("SN_GRAPH_TEMP", 1.0),
            graph_device=env_str("SN_GRAPH_DEVICE"),
            action_scale_path=env_str("SN_ACTION_SCALE"),
            ablation=Ablation.from_env(),
        )


    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SentinelConfig":
        """Build from a mapping, ignoring unknown keys."""
        return cls(**_known_fields(cls, data))


# robot_ood.core.kinematics.ARM_JOINTS' own length -- kept local to avoid mujoco/torch dependencies.
_ENTRY_ARM_DOF: int = 7
# Convert fixed-radius offset to Gaussian sigma: E[||sigma*eps||] = sigma*sqrt(D) for D dimensions.
RADIUS_TO_SIGMA: float = 1.0 / math.sqrt(_ENTRY_ARM_DOF)

# Tweedie iteration count: NOT a calibrated key (no corpus measure exists), overridable via EN_REFINE_STEPS.
_ENTRY_REFINE_STEPS: int = 3
_ENTRY_PERTURB_LO_DEFAULT: float = 0.004858336877077818


def entry_radius_lo(graph_dir: "str | os.PathLike | None" = None) -> float:
    """The finest off-manifold radius (rad): the calibrated corpus floor, never below what the corpus calls still."""
    lo = float(calibrated("entry_perturb_lo", _ENTRY_PERTURB_LO_DEFAULT, graph_dir))
    return max(lo, STILL_TOL)


def entry_radius_hi(graph_dir: "str | os.PathLike | None" = None) -> float:
    """The widest off-manifold radius (rad): the benchmark's recorded robot-init magnitude."""
    return float(calibrated("entry_robotinit_hi_rad", 0.5, graph_dir))


def entry_sigma_lo(graph_dir: "str | os.PathLike | None" = None) -> float:
    """The entry score field's finest noise scale, entry_radius_lo as a Gaussian sigma."""
    return entry_radius_lo(graph_dir) * RADIUS_TO_SIGMA


def entry_sigma_hi(graph_dir: "str | os.PathLike | None" = None) -> float:
    """The entry score field's widest noise scale, entry_radius_hi as a Gaussian sigma."""
    return entry_radius_hi(graph_dir) * RADIUS_TO_SIGMA


@dataclass(frozen=True)
class EntryConfig:
    """The one-shot entry correction's knobs (robot_ood.runtime.entry -- the t=0 regime).

    Attributes:
        score_sigma_hi: EN_SIGMA_HI — upper clamp on estimated noise scale.
        score_sigma_lo: EN_SIGMA_LO — lower clamp on estimated noise scale.
        score_refine_steps: EN_REFINE_STEPS — Tweedie iteration count.
        score_step_cap: EN_STEP_CAP — maximum joint displacement per control step, in radians.
    """

    score_sigma_hi: float = 0.5 * RADIUS_TO_SIGMA
    score_sigma_lo: float = _ENTRY_PERTURB_LO_DEFAULT * RADIUS_TO_SIGMA
    score_refine_steps: int = _ENTRY_REFINE_STEPS
    score_step_cap: float = 0.06644450331284547

    @classmethod
    def from_env(cls, graph_dir: "str | os.PathLike | None" = None) -> "EntryConfig":
        """Build from the EN_* environment variables, defaulting to the C0 calibration.

        Args:
            graph_dir: The artifact directory whose embedded calibration the defaults are read
                from; None reads the process's.
        """
        return cls(
            score_sigma_hi=env_float("EN_SIGMA_HI", entry_sigma_hi(graph_dir)),
            score_sigma_lo=env_float("EN_SIGMA_LO", entry_sigma_lo(graph_dir)),
            score_refine_steps=env_int("EN_REFINE_STEPS", _ENTRY_REFINE_STEPS),
            score_step_cap=env_float("EN_STEP_CAP",
                                      calibrated("entry_score_step_cap_rad", 0.06644450331284547, graph_dir)),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "EntryConfig":
        """Build from a mapping, ignoring unknown keys."""
        return cls(**_known_fields(cls, data))
