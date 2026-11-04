"""Contract for the demonstration-graph subsystem.

Dataclasses define the configuration and artifact schema; flat names at the bottom are derived from DEFAULTS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Final, Mapping

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "NpzSchema",
    "ArtifactSchemas",
    "RelationVocab",
    "GripperConfig",
    "GraphBuildConfig",
    "NetworkConfig",
    "RetrievalConfig",
    "GraphContract",
    "DEFAULTS",
    "grip_flag",
]


# ==== I/O schema ====
@dataclass(frozen=True, slots=True)
class NpzSchema:
    """Filename and column dtypes for one on-disk npz artifact.

    Args:
        filename: Basename of the artifact within the graph directory.
        keys: Column name -> numpy dtype string.
        hash_stamped: Whether loaders must verify a graph_hash stamp and refuse on mismatch.
        optional_keys: Columns a loader may synthesise a zero default for instead of refusing the
            file. Everything else is mandatory: a zero-filled ee_raw is a dead end-effector track
            that nothing reports.
    """

    filename: str
    keys: Mapping[str, str] = field(default_factory=dict)
    hash_stamped: bool = False
    optional_keys: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.filename:
            raise ValueError("NpzSchema.filename must be non-empty")
        for name, dtype in self.keys.items():
            try:
                np.dtype(dtype)
            except TypeError as exc:
                raise ValueError(f"{self.filename}: key {name!r} has bad dtype {dtype!r}") from exc
        unknown = set(self.optional_keys) - set(self.keys)
        if unknown:
            raise ValueError(f"{self.filename}: optional_keys names no such column {sorted(unknown)}")
        object.__setattr__(self, "keys", MappingProxyType(dict(self.keys)))

    def required_keys(self) -> tuple[str, ...]:
        """Column names a loader must find in the archive.

        Returns:
            Every key that is not optional, in declaration order.
        """
        return tuple(k for k in self.keys if k not in self.optional_keys)


class ArtifactSchemas:
    """Namespace of the on-disk npz artifact schemas. Not instantiated."""

    # q_raw/qdot_raw/raw_ptr keep every raw frame so segment length k stays a runtime choice.
    NODES: Final[NpzSchema] = NpzSchema(
        filename="g_nodes.npz",
        keys={
            "q": "f4", "qdot": "f4", "grip": "u1", "stage": "i2", "t_idx": "i4", "t_frac": "f4",
            "t_raw": "i4", "n_members": "i2", "owner": "i4", "task_id": "i2", "phase": "f4",
            "demo_ptr": "i4", "q_raw": "f4", "qdot_raw": "f4", "raw_ptr": "i4",
            "ee_raw": "f4", "a_raw": "f4", "is_virtual": "u1", "arm": "u1", "pair": "i4",
        },
        # is_virtual postdates every graph on disk, and an absent one loads as all-real.
        # arm and pair are bimanual: a single-arm graph is arm 0 and pair -1 throughout.
        optional_keys=frozenset({"is_virtual", "arm", "pair"}),
    )

    # Sorted by dst so indptr is a valid CSR offset array and index_add_ collisions coalesce.
    EDGES: Final[NpzSchema] = NpzSchema(
        filename="g_edges.npz",
        keys={"src": "i4", "dst": "i4", "rel": "i1", "w": "f4", "indptr": "i4", "log_idf": "f4"},
    )

    HEAD: Final[NpzSchema] = NpzSchema(filename="g_head.npz", keys={}, hash_stamped=True)

    # FCSR in one file: the segment retriever, the C2 scorer, the fusion gate and the dose head,
    # each under its own key prefix. Segment ids are positional, so the stamp is mandatory.
    FCSR: Final[NpzSchema] = NpzSchema(filename="g_fcsr.npz", keys={}, hash_stamped=True)

    # robot_ood.core.schemas.SegmentTable, one row per trajectory segment. Derived from
    # g_nodes.npz, so it is stamped and refused the same way.
    SEGMENTS: Final[NpzSchema] = NpzSchema(
        filename="g_segments.npz",
        keys={
            "start_node": "i4", "end_node": "i4", "owner": "i4", "task_id": "i2",
            "arm": "i2", "phase_start": "f4", "phase_end": "f4", "action_chunk": "f4",
            "ee_start": "f4", "ee_end": "f4", "node_to_segment": "i4",
        },
        hash_stamped=True,
    )

    def __init__(self) -> None:
        raise TypeError("ArtifactSchemas is a namespace and must not be instantiated")


# ==== Relation vocabulary ====
@dataclass(frozen=True, slots=True)
class RelationVocab:
    """Edge-relation alphabet. Order is load-bearing: rel indexes into relations.

    Args:
        dilations: Temporal hop lengths in coarse node steps.
        rel_dtype: Numpy dtype string of the rel edge column, used for a width check.
    """

    # Dilation is load-bearing, not an optimisation: depth L reaches only L hops, while a
    # pre-grasp approach is ~80 raw frames. Changing this invalidates every g_edges.npz.
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    rel_dtype: str = ArtifactSchemas.EDGES.keys["rel"]

    @property
    def next_relations(self) -> tuple[str, ...]:
        """Forward-along-demo relation names, dilated.

        Returns:
            Relation names in dilation order.
        """
        return tuple(f"next{d}" for d in self.dilations)

    @property
    def prev_relations(self) -> tuple[str, ...]:
        """Backward-along-demo relation names, dilated.

        Returns:
            Relation names in dilation order.
        """
        return tuple(f"prev{d}" for d in self.dilations)

    @property
    def relations(self) -> tuple[str, ...]:
        """Full relation alphabet in rel-index order.

        Returns:
            All relation names. sibling is kNN within a task; align crosses tasks at a
            shared phase bin; approach runs virtual -> real at the entry stratum. There is
            deliberately no stage relation.

            APPEND ONLY. The index is the row of the relation embedding table, so inserting a name
            silently re-labels every edge of every relation after it in every g_edges.npz on disk.
        """
        return (
            *self.next_relations, *self.prev_relations, "sibling", "align", "approach",
            # The bimanual relations. They are empty on a single-arm graph, and their slots come
            # last so a relation id means the same thing in every checkpoint ever trained.
            "partner", "cross_arm", "cross_arm_align",
        )

    @property
    def rel_index(self) -> Mapping[str, int]:
        """Relation name to its rel column value.

        Returns:
            Read-only name -> index mapping.
        """
        return MappingProxyType({r: i for i, r in enumerate(self.relations)})

    @property
    def n_relations(self) -> int:
        """Size of the relation alphabet.

        Returns:
            Number of distinct relations.
        """
        return len(self.relations)

    def __post_init__(self) -> None:
        if not self.dilations:
            raise ValueError("RelationVocab.dilations must be non-empty")
        if any(d <= 0 for d in self.dilations):
            raise ValueError(f"dilations must be positive, got {self.dilations}")
        if len(set(self.dilations)) != len(self.dilations):
            raise ValueError(f"dilations contains duplicates: {self.dilations}")

        names = self.relations
        if len(set(names)) != len(names):
            raise ValueError(f"relation alphabet contains duplicates: {names}")

        rel_max = int(np.iinfo(np.dtype(self.rel_dtype)).max)
        if len(names) > rel_max + 1:
            raise ValueError(
                f"{len(names)} relations do not fit dtype {self.rel_dtype!r} (max {rel_max})"
            )


# ==== Configs ====
@dataclass(frozen=True, slots=True)
class GripperConfig:
    """Gripper-state thresholding.

    Args:
        open_thresh: Mean absolute finger qpos at or below which the gripper counts as closed.
    """

    # Single source of truth for the gripper-closed threshold.
    open_thresh: float = 0.035

    def __post_init__(self) -> None:
        if self.open_thresh <= 0.0:
            raise ValueError(f"open_thresh={self.open_thresh} must be positive")


@dataclass(frozen=True, slots=True)
class GraphBuildConfig:
    """Node-table and edge-construction parameters.

    Args:
        coarsen: Raw frames merged into one node; splits at every gripper-state change.
        psi_freqs: Sinusoidal time-encoding octaves; yields 2 * psi_freqs feature slots.
        k_sibling: kNN degree for the sibling relation.
        k_align: kNN degree for the align relation.
        virtual_per_entry: Virtual nodes synthesised per entry-stratum node (M). 0 = off, which is
            the pre-virtual graph bit-for-bit.
        virtual_phase_band: Phase ceiling defining the entry stratum virtual nodes are grown from.
        nbins_align: Phase bins for the align bucket. Exported as NBINS_ALIGN, which
            every other reader of the binning (edges, loss, metrics, the sentinel's phase gate)
            must agree on -- see robot_ood.data.binning.phase_bins.
        cdist_bs: Row-block size for batched torch.cdist.
        kernel_bw: Joint-space kernel bandwidth.
        phase_conv: Phase convention stamped into the npz and re-checked on load.
    """

    coarsen: int = 5
    psi_freqs: int = 4
    k_sibling: int = 8
    k_align: int = 4
    # Both arms of gate 4 / gate 6 default OFF: they change V and the edge set, and the artifact
    # tree is shared. A measurement run turns them on explicitly.
    virtual_per_entry: int = 0
    virtual_phase_band: float = 0.05
    nbins_align: int = 20
    cdist_bs: int = 2048
    kernel_bw: float = 0.15
    # Full-episode convention: phase[0] == 0 and phase[-1] == 1 exactly, so align bins phase
    # across tasks.
    phase_conv: str = "travelled(q)/travelled(T-1) full"

    def __post_init__(self) -> None:
        positive_ints = {
            "coarsen": self.coarsen, "psi_freqs": self.psi_freqs,
            "k_sibling": self.k_sibling, "k_align": self.k_align,
            "nbins_align": self.nbins_align, "cdist_bs": self.cdist_bs,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name}={value} must be a positive integer")
        if self.virtual_per_entry < 0:
            raise ValueError(f"virtual_per_entry={self.virtual_per_entry} must be non-negative")
        if not 0.0 <= self.virtual_phase_band < 1.0:
            raise ValueError(f"virtual_phase_band={self.virtual_phase_band} must lie in [0, 1)")
        if self.kernel_bw <= 0.0:
            raise ValueError(f"kernel_bw={self.kernel_bw} must be positive")
        if not self.phase_conv:
            raise ValueError("phase_conv must be non-empty")


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Message-passing network hyperparameters.

    Args:
        hidden: Hidden width.
        layers: Message-passing depth; reach is layers * max(dilations) * coarsen raw frames.
        agg: Neighbour aggregation, "sum" or "logsumexp".
        agg_temp: Temperature for logsumexp aggregation.
        frame_dropout: Probability a query-window frame is masked out of the pooling softmax
            during training.
    """

    hidden: int = 64
    # 3 gives ~240 raw frames of reach, just under the long suite median demo length (p50 263).
    # Deeper dilutes: from a 256-node seed set, 6 layers reach 100% of a long graph's nodes.
    layers: int = 3
    agg: str = "sum"          # logsumexp with m = -cost recovers DTW's soft-min
    agg_temp: float = 1.0
    # The pooling head has H=8 near-identical frames to choose between. Masking one per draw is
    # what stops the softmax collapsing onto the latest frame, which is the parameter-free control.
    frame_dropout: float = 0.15

    # ClassVar, not a field: a slots dataclass would otherwise put this in __init__.
    _VALID_AGG: ClassVar[frozenset[str]] = frozenset({"sum", "logsumexp"})

    def __post_init__(self) -> None:
        if self.hidden <= 0:
            raise ValueError(f"hidden={self.hidden} must be positive")
        if self.layers <= 0:
            raise ValueError(f"layers={self.layers} must be positive")
        if self.agg not in self._VALID_AGG:
            raise ValueError(f"agg={self.agg!r} must be one of {sorted(self._VALID_AGG)}")
        if self.agg_temp <= 0.0:
            raise ValueError(f"agg_temp={self.agg_temp} must be positive")
        if not 0.0 <= self.frame_dropout < 1.0:
            raise ValueError(f"frame_dropout={self.frame_dropout} must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Query construction and seeding.

    Args:
        hist_h: Query history window length W, one policy action chunk.
        seg_k: Reference-segment length; a segment restores velocity, a single point does not.
        seed_vel_w: Weight on the velocity-direction factor in the seeding kernel; 0 disables.
        seed_topk: Only the top-K nodes receive a nonzero initial hidden state.
        seed_tier1_m: Tier-1 candidates per window row in the two-tier seed. It lives here rather
            than beside LearnedSeed so that robot_ood.core.config can default from it without importing torch
            (tests/test_layering.py). Gated on tier1_recall >= 0.99, measured after S1.
        advance: Ground truth sits this many raw frames ahead; 0 labels the window's own end.
    """

    hist_h: int = 8
    seg_k: int = 8
    # Drift is nearly a pure translation of the query window, corrupting q but not qdot: at the
    # deployment-median 0.30 rad displacement, phase localisation is 4x better by velocity.
    seed_vel_w: float = 2.0
    seed_topk: int = 128
    seed_tier1_m: int = 2048
    # The head answers "where am I NOW"; the fitted transition kernel owns the lookahead. Any
    # advance > 0 is added to the kernel's own E[a] (~9.6 raw frames), overshooting the chunk.
    advance: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("hist_h", self.hist_h), ("seg_k", self.seg_k), ("seed_topk", self.seed_topk),
        ):
            if value <= 0:
                raise ValueError(f"{name}={value} must be positive")
        if self.advance < 0:
            raise ValueError(f"advance={self.advance} must be non-negative")
        if self.seed_vel_w < 0.0:
            raise ValueError(f"seed_vel_w={self.seed_vel_w} must be non-negative")


@dataclass(frozen=True, slots=True)
class GraphContract:
    """Composed configuration for the demonstration-graph subsystem.

    Args:
        vocab: Relation alphabet.
        gripper: Gripper thresholding.
        build: Node-table and edge construction.
        network: Message-passing hyperparameters.
        retrieval: Query construction and seeding.
    """

    vocab: RelationVocab = field(default_factory=RelationVocab)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    build: GraphBuildConfig = field(default_factory=GraphBuildConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @property
    def reach_raw_frames(self) -> int:
        """Raw-frame reach of the message-passing stack.

        Returns:
            layers * max(dilations) * coarsen.
        """
        return self.network.layers * max(self.vocab.dilations) * self.build.coarsen



# Convenience default. Prefer injecting a GraphContract explicitly in new code.
DEFAULTS: Final[GraphContract] = GraphContract()


# ==== Utilities ====
def grip_flag(gripper_qpos: ArrayLike | None, thresh: float | None = None) -> float:
    """Map gripper joint positions to the graph's grip channel.

    Args:
        gripper_qpos: Finger joint positions, any array-like shape, or None if unavailable.
        thresh: Closed-gripper threshold. Defaults to DEFAULTS.gripper.open_thresh.

    Returns:
        1.0 if the gripper is closed (grasping), else 0.0. None yields 0.0.
    """
    if gripper_qpos is None:
        return 0.0
    limit = DEFAULTS.gripper.open_thresh if thresh is None else thresh
    mean_abs = float(np.abs(np.asarray(gripper_qpos, np.float64)).mean())
    return 1.0 if mean_abs <= limit else 0.0


# ==== Flat view -- derived from DEFAULTS, never assigned a literal ====

# ---- relation vocabulary ----
DILATIONS: Final = DEFAULTS.vocab.dilations
RELATIONS: Final = DEFAULTS.vocab.relations
REL_INDEX: Final = DEFAULTS.vocab.rel_index
N_RELATIONS: Final = DEFAULTS.vocab.n_relations

# ---- node-table construction ----
COARSEN: Final = DEFAULTS.build.coarsen
PSI_FREQS: Final = DEFAULTS.build.psi_freqs
GRIP_OPEN_THR: Final = DEFAULTS.gripper.open_thresh
PHASE_CONV: Final = DEFAULTS.build.phase_conv

# ---- edge construction ----
K_SIBLING: Final = DEFAULTS.build.k_sibling
K_ALIGN: Final = DEFAULTS.build.k_align
VIRTUAL_PER_ENTRY: Final = DEFAULTS.build.virtual_per_entry
VIRTUAL_PHASE_BAND: Final = DEFAULTS.build.virtual_phase_band
NBINS_ALIGN: Final = DEFAULTS.build.nbins_align
CDIST_BS: Final = DEFAULTS.build.cdist_bs
KERNEL_BW: Final = DEFAULTS.build.kernel_bw

# ---- query / retrieval ----
HIST_H: Final = DEFAULTS.retrieval.hist_h
SEG_K: Final = DEFAULTS.retrieval.seg_k
SEED_VEL_W: Final = DEFAULTS.retrieval.seed_vel_w
SEED_TOPK: Final = DEFAULTS.retrieval.seed_topk
SEED_TIER1_M: Final = DEFAULTS.retrieval.seed_tier1_m
ADVANCE: Final = DEFAULTS.retrieval.advance

# ---- network ----
HIDDEN: Final = DEFAULTS.network.hidden
FRAME_DROPOUT: Final = DEFAULTS.network.frame_dropout
LAYERS: Final = DEFAULTS.network.layers
AGG: Final = DEFAULTS.network.agg
AGG_TEMP: Final = DEFAULTS.network.agg_temp

# ---- artifact layout ----
NODES_NPZ: Final = ArtifactSchemas.NODES.filename
EDGES_NPZ: Final = ArtifactSchemas.EDGES.filename
HEAD_NPZ: Final = ArtifactSchemas.HEAD.filename
FCSR_NPZ: Final = ArtifactSchemas.FCSR.filename
SEGMENTS_NPZ: Final = ArtifactSchemas.SEGMENTS.filename
# Not an ArtifactSchemas entry: a scalar json the blend layer writes, not a column table.
ACTION_SCALE_JSON: Final = "action_scale.json"
# Not an ArtifactSchemas entry either: written by build.policy_cache, read by models.fusion.target.
A_PI_RAW_NPZ: Final = "a_pi_raw.npz"
# The replay report's sidecar file, read only when a head carries no embedded report; graph_hash
# does not cover it, so writing it into an artifact dir leaves trained checkpoints loadable.
PHASE_SKEW_JSON: Final = "g_phase_skew.json"
# The report's home: a JSON string train locator's post-save replay embeds under this key in g_head.npz.
PHASE_SKEW_KEY: Final = "phase_skew_json"
# The C0 calibration a train fcsr run trained under, a JSON string under this key in g_fcsr.npz;
# deploy resolves constants from here unless an ROBOT_OOD_CONSTANTS file overrides it.
CONSTANTS_KEY: Final = "constants_json"
NODES_KEYS: Final = ArtifactSchemas.NODES.keys

__all__ += [
    "DILATIONS", "RELATIONS", "REL_INDEX", "N_RELATIONS",
    "COARSEN", "PSI_FREQS", "GRIP_OPEN_THR", "PHASE_CONV",
    "K_SIBLING", "K_ALIGN", "NBINS_ALIGN", "CDIST_BS", "KERNEL_BW",
    "VIRTUAL_PER_ENTRY", "VIRTUAL_PHASE_BAND",
    "HIST_H", "SEG_K", "SEED_VEL_W", "SEED_TOPK", "ADVANCE",
    "HIDDEN", "LAYERS", "AGG", "AGG_TEMP",
    "NODES_NPZ", "EDGES_NPZ", "HEAD_NPZ", "FCSR_NPZ", "SEGMENTS_NPZ",
    "ACTION_SCALE_JSON",
    "A_PI_RAW_NPZ", "PHASE_SKEW_JSON", "PHASE_SKEW_KEY", "CONSTANTS_KEY",
    "NODES_KEYS",
]
