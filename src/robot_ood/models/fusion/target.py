"""The fusion loss's cached inputs and its pose-derived regression target a_corr.

a_policy is the policy's own cached action (not the demo's); a_corr is pose-derived (not recorded); track_chunk is the torch restatement of the numpy version so gradients flow to a_track and alpha.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from robot_ood.core.rotation import so3_exp, so3_log
from robot_ood.core.action import EUCLIDEAN, ActionScale, POSE_DIM, POS_BLOCK, ROT_BLOCK
from robot_ood.runtime.ee_track import EeTrack

if TYPE_CHECKING:
    from robot_ood.core.kinematics import PandaKinematics
    from robot_ood.data.nodes import NodeTable

__all__ = [
    "ACTIONS_NPZ",
    "saturate",
    "track_chunk",
    "consensus_reference",
    "node_pose_ref",
    "PolicyActionCache",
    "ChunkTargets",
    "ChunkTargetBuilder",
]

from robot_ood.data.schema import A_PI_RAW_NPZ as ACTIONS_NPZ  # one owner for the filename


def saturate(row: torch.Tensor, bound: torch.Tensor | float) -> torch.Tensor:
    """Squash one chunk row into (-bound, bound) smoothly.

    Args:
        row: [..., 6] action-unit row.
        bound: Saturation amplitude in action units.

    Returns:
        [..., 6] bound * tanh(row / bound) -- identity to first order near zero, so a row already
        inside the bound is passed through essentially unchanged.
    """
    return bound * torch.tanh(row / bound)

def track_chunk(
    ee_ref: torch.Tensor,
    ee_now: torch.Tensor,
    scale: ActionScale,
    bound: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Batched, differentiable robot_ood.runtime.ee_track.EeTrack.chunk.

    Args:
        ee_ref: [B, K, 6] reference segments, concat(position, axis-angle orientation).
        ee_now: [B, 6] live end-effector poses.
        scale: Metres/radians per action unit.
        bound: Row-0 saturation amplitude in action units, or None for none. Normally
            scale.limit; see EeTrack.bound for why row 0 alone needs it.

    Returns:
        [B, K, 6] action-unit deltas, in the layout of action dims 0:6.

    Raises:
        ValueError: ee_ref is not [B, K, 6] or ee_now is not [B, 6].
    """
    if ee_ref.dim() != 3 or ee_ref.shape[-1] != POSE_DIM:
        raise ValueError(f"ee_ref must be [B, K, {POSE_DIM}], got shape {tuple(ee_ref.shape)}")
    if ee_now.shape != (ee_ref.shape[0], POSE_DIM):
        raise ValueError(
            f"ee_now must be [{ee_ref.shape[0]}, {POSE_DIM}], got shape {tuple(ee_now.shape)}"
        )
    source = torch.cat([ee_now[:, None, :], ee_ref[:, :-1, :]], dim=1)
    position = (ee_ref[..., POS_BLOCK] - source[..., POS_BLOCK]) / scale.pos
    if scale.pose_space == EUCLIDEAN:     # a joint-space proxy: dims 3:6 are coordinates
        rotation = ee_ref[..., ROT_BLOCK] - source[..., ROT_BLOCK]
    else:
        # COMPOSED, never subtracted: measured on the demos, world-frame composition regresses
        # onto the policy's rotation block at R2 0.84-0.93 and subtraction at R2 -0.03.
        relative = so3_exp(ee_ref[..., ROT_BLOCK]) @ so3_exp(source[..., ROT_BLOCK]).transpose(-1, -2)
        rotation = so3_log(relative)
    track = torch.cat([position, rotation / scale.rot], dim=-1)
    if bound is None:
        return track
    head = saturate(track[:, :1, :], bound)
    return torch.cat([head, track[:, 1:, :]], dim=1)


# The reference lives where the retrieved demos agree. Directions of the candidates' Cartesian
# spread with a standard deviation well below these pass through; directions the scene moved
# (an object placed elsewhere in every demo) are left to the policy.
CONSENSUS_POS: float = 0.01      # metres
CONSENSUS_ROT: float = 0.05      # radians
_CONSENSUS_DAMP: float = 1e-4    # damped least squares on J J^T, metres^2


def _soft_projector(dev: torch.Tensor, p: torch.Tensor, tau: float) -> torch.Tensor:
    """[B, D, D] tau^2 (Sigma + tau^2 I)^-1 for the score-weighted spread of dev [B, K, C, D] over K."""
    w = p[:, :, None, None, None]
    cov = (w * dev[..., :, None] * dev[..., None, :]).sum(dim=1).mean(dim=1)             # [B, D, D]
    eye = torch.eye(dev.shape[-1], dtype=dev.dtype, device=dev.device)
    return tau * tau * torch.linalg.inv(cov + tau * tau * eye)


def _consensus_terms(
    ee_ref: torch.Tensor, p: torch.Tensor, ee_now: torch.Tensor, cart: Any,
    a_policy: torch.Tensor, scale: ActionScale,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The policy's planned joints [B, C, 6] and the projected Cartesian offset [B, C, 6] toward the candidates.

    Args:
        ee_ref: [B, K, C, 6] each candidate's own joint trajectory.
        p: [B, K] the candidates' posterior weights.
        ee_now: [B, 6] the live joints.
        cart: robot_ood.core.kinematics.Arx5Kinematics of the lane.
        a_policy: [B, C, >=6] the policy's chunk, per-row deltas in action units.
        scale: Metres/radians per action unit.

    Returns:
        q_pol, the policy's planned joints per row, and offset: the candidates' score-weighted
        mean tip pose minus the policy's planned tip pose, per row, projected onto the directions
        the candidates hold to within CONSENSUS_POS / CONSENSUS_ROT (see consensus_reference).
    """
    weights = p.reshape(*p.shape, 1, 1)
    b, c = a_policy.shape[0], a_policy.shape[1]
    units = torch.tensor([scale.pos] * 3 + [scale.rot] * 3, dtype=ee_ref.dtype, device=ee_ref.device)
    q_pol = ee_now[:, None, :] + torch.cumsum(a_policy[..., :POSE_DIM].to(ee_ref.dtype) * units, dim=1)   # [B, C, 6]
    x_ref = cart.pose(ee_ref)                                                              # [B, K, C, 6]
    x_pol = cart.pose(q_pol)                                                               # [B, C, 6]
    x_mean = (weights * x_ref).sum(dim=1)                                                  # [B, C, 6]
    # Rotation deviations as the log of the relative rotation, so nothing wraps at pi.
    r_mean = so3_exp(x_mean[..., ROT_BLOCK])
    d_pos = x_ref[..., POS_BLOCK] - x_mean[:, None, :, POS_BLOCK]
    d_rot = so3_log(so3_exp(x_ref[..., ROT_BLOCK]) @ r_mean[:, None].transpose(-1, -2))
    proj_pos = _soft_projector(d_pos, p, CONSENSUS_POS)                                    # [B, 3, 3]
    proj_rot = _soft_projector(d_rot, p, CONSENSUS_ROT)
    off_pos = x_mean[..., POS_BLOCK] - x_pol[..., POS_BLOCK]                                # [B, C, 3]
    off_rot = so3_log(r_mean @ so3_exp(x_pol[..., ROT_BLOCK]).transpose(-1, -2))
    offset = torch.cat([off_pos @ proj_pos.transpose(-1, -2), off_rot @ proj_rot.transpose(-1, -2)], dim=-1)
    return q_pol, offset


def consensus_reference(
    ee_ref: torch.Tensor, p: torch.Tensor, ee_now: torch.Tensor, cart: Any,
    a_policy: torch.Tensor | None = None, scale: ActionScale | None = None,
) -> torch.Tensor:
    """[B, C, 6] the shortlist's reference trajectory, following the demos only where they agree.

    On a Cartesian lane (cart None) this is the score-weighted mean of the candidates' own pose
    trajectories. On a joint-space lane the reference is the policy's own planned configuration
    at each row, moved toward the candidates' mean tip pose only along the directions the
    candidates hold to within CONSENSUS_POS metres / CONSENSUS_ROT radians: per position and
    rotation block the projector tau^2 (Sigma + tau^2 I)^-1 of their score-weighted spread, so a
    direction the demos scatter over (an object placed elsewhere in every demo) stays with the
    policy and a direction they hold (a grasp height, a wrist attitude) pulls toward them. The
    projected Cartesian offset comes back to joint space through the Jacobian at the policy's
    planned configuration.

    Args:
        ee_ref: [B, K, C, 6] each candidate's own pose trajectory, in the lane's pose block.
        p: [B, K] the candidates' posterior weights.
        ee_now: [B, 6] the live pose, in the lane's pose block.
        cart: robot_ood.core.kinematics.Arx5Kinematics for a joint-space lane, else None.
        a_policy: [B, C, >=6] the policy's chunk, per-row deltas in action units; joint-space
            lanes only.
        scale: Metres/radians per action unit, converting a_policy's rows; joint-space lanes only.

    Returns:
        [B, C, 6] in the lane's pose block, ee_ref's dtype.

    Raises:
        ValueError: A joint-space lane without a_policy and scale.
    """
    weights = p.reshape(*p.shape, 1, 1)
    if cart is None:
        return (weights * ee_ref).sum(dim=1)
    if a_policy is None or scale is None:
        raise ValueError("consensus_reference on a joint-space lane needs the policy chunk and its scale")
    b, c = a_policy.shape[0], a_policy.shape[1]
    q_pol, offset = _consensus_terms(ee_ref, p, ee_now, cart, a_policy, scale)
    jac = cart.jacobian(q_pol.reshape(b * c, POSE_DIM)).reshape(b, c, POSE_DIM, POSE_DIM)  # [B, C, 6, 6]
    eye = torch.eye(POSE_DIM, dtype=jac.dtype, device=jac.device)
    gain = jac.transpose(-1, -2) @ torch.linalg.inv(jac @ jac.transpose(-1, -2) + _CONSENSUS_DAMP * eye)
    return q_pol + (gain @ offset[..., None])[..., 0]


def node_pose_ref(
    node_ee: np.ndarray, start_node: np.ndarray, end_node: np.ndarray, chunk_len: int,
) -> np.ndarray:
    """Segment's pose trajectory via NODE-id interpolation (always legal time-ordered samples within a demo).

    Args:
        node_ee: [V, 6] per-node pose, from robot_ood.data.segments.node_ee_table.
        start_node: [...] int, a segment's first node id.
        end_node: [...] int, the segment's last node id (inclusive).
        chunk_len: C, the number of pose samples to produce.

    Returns:
        [..., C, 6] float32, node_ee's own dtype promoted.

    Raises:
        ValueError: chunk_len < 1.
    """
    if chunk_len < 1:
        raise ValueError(f"chunk_len must be >= 1, got {chunk_len}")
    denom = float(max(chunk_len - 1, 1))
    frac = np.arange(chunk_len, dtype=np.float64) / denom
    start = np.asarray(start_node, dtype=np.float64)[..., None]
    end = np.asarray(end_node, dtype=np.float64)[..., None]
    node_idx = np.round(start + frac * (end - start)).astype(np.int64)
    return node_ee[node_idx].astype(np.float32)


# ROBOT_OOD_REF=raw: a candidate's reference is its own demo's raw frames from the segment's start,
# one frame per plan row -- the same construction as the training target a_corr
# (NodeTable.ee_segment). The default "node" keeps node_pose_ref, which spreads the segment's
# 2-3 nodes over the rows as a staircase.
REF_MODE: str = os.environ.get("ROBOT_OOD_REF", "node")


@dataclass(frozen=True, eq=False)
class RawPoseRef:
    """Raw-frame pose references for segments, vectorised over a batch of start nodes.

    Attributes:
        t_raw: [V] int64 each node's first raw frame.
        last: [V] int64 the last raw frame of each node's own demo (edge padding stops there).
        ee_base: [N_raw, 6] float32 base-frame poses.
    """

    t_raw: np.ndarray
    last: np.ndarray
    ee_base: np.ndarray
    first: np.ndarray
    relative: bool = False
    owner: np.ndarray | None = None
    phase: np.ndarray | None = None
    demo_ptr: np.ndarray | None = None
    raw_ptr: np.ndarray | None = None        # [n_demo + 1] each demo's raw-frame range
    demo_pause: np.ndarray | None = None     # [n_demo] filled by dwell() on first use

    @classmethod
    def from_nodes(cls, nodes: "NodeTable", relative: bool = False) -> "RawPoseRef":
        owner = np.asarray(nodes.owner, dtype=np.int64)
        raw_ptr = np.asarray(nodes.raw_ptr, dtype=np.int64)
        return cls(t_raw=np.asarray(nodes.t_raw, dtype=np.int64), last=raw_ptr[owner + 1] - 1,
                   ee_base=np.asarray(nodes.ee_base, dtype=np.float32), first=raw_ptr[owner],
                   relative=bool(relative), owner=owner, phase=np.asarray(nodes.phase, dtype=np.float32),
                   demo_ptr=np.asarray(nodes.demo_ptr, dtype=np.int64), raw_ptr=raw_ptr)

    def _phase_nodes(self, start_node: np.ndarray, phase: float) -> np.ndarray:
        """[K] int64 for each start node, the node of its own demo whose phase is nearest `phase`."""
        if self.owner is None or self.phase is None or self.demo_ptr is None:
            raise ValueError("at_phase needs a RawPoseRef built by from_nodes")
        s = np.asarray(start_node, dtype=np.int64).reshape(-1)
        nodes = np.empty(len(s), dtype=np.int64)
        for k, d in enumerate(self.owner[s]):
            lo, hi = int(self.demo_ptr[d]), int(self.demo_ptr[d + 1])
            nodes[k] = lo + int(np.argmin(np.abs(self.phase[lo:hi] - float(phase))))
        return nodes

    def _still(self, t: int, move_eps: float) -> bool:
        """Whether the demo's step out of raw frame t is below move_eps."""
        return bool(np.linalg.norm(self.ee_base[t + 1] - self.ee_base[t]) < move_eps)

    def dwell(self, start_node: np.ndarray, still_eps: float = 0.003) -> np.ndarray:
        """[K] int64 the longest pause in each start node's own demo, in frames.

        A pause is a run of frames whose mean absolute step over the pose block is below
        still_eps, the same test the runtime applies to a policy chunk, between the demo's first
        and last moving frame; a demo that never moves (the idle arm of a one-armed task) counts
        as one pause over its whole length. Read per demo rather than at a phase because a pause
        is a single point of the motion-covered phase and the belief may miss it by more than a
        node.
        """
        if self.owner is None or self.raw_ptr is None:
            raise ValueError("dwell needs a RawPoseRef built by from_nodes")
        if self.demo_pause is None:
            step = np.abs(np.diff(self.ee_base, axis=0)).mean(axis=-1)                  # [N_raw - 1]
            pause = np.zeros(len(self.raw_ptr) - 1, dtype=np.int64)
            for d in range(len(pause)):
                still = step[int(self.raw_ptr[d]):int(self.raw_ptr[d + 1]) - 1] < still_eps
                moving = np.flatnonzero(~still)
                if len(moving) == 0:
                    pause[d] = len(still) + 1
                    continue
                inner = still[moving[0]:moving[-1]]
                edges = np.diff(np.concatenate(([0], inner.astype(np.int64), [0])))
                runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
                pause[d] = int(runs.max()) if len(runs) else 0
            object.__setattr__(self, "demo_pause", pause)
        s = np.asarray(start_node, dtype=np.int64).reshape(-1)
        return self.demo_pause[self.owner[s]]

    def pose_at_phase(self, start_node: np.ndarray, phase: float) -> np.ndarray:
        """[K, 6] each start node's demo pose at its frame nearest `phase` (absolute, whatever `relative` says)."""
        return self.ee_base[self.t_raw[self._phase_nodes(start_node, phase)]]

    def between(self, start_node: np.ndarray, p0: float, p1: float) -> np.ndarray:
        """[K, 6] each start node's demo displacement from its frame at phase p0 to its frame at p1."""
        t0 = self.t_raw[self._phase_nodes(start_node, p0)]
        t1 = self.t_raw[self._phase_nodes(start_node, p1)]
        return self.ee_base[t1] - self.ee_base[t0]

    def at_phase(self, start_node: np.ndarray, phase: float, chunk_len: int,
                 move_eps: float = 0.005, max_skip: int | None = 30) -> np.ndarray:
        """[K, chunk_len, 6] each start node's demo, displaced from its own frame at `phase`.

        The frame is the first raw frame of the demo's node whose phase is nearest `phase`, moved
        forward past a pause there (frames whose step is below move_eps; up to max_skip of them,
        or the whole pause when max_skip is None), so the rows are what the demo does NEXT: its
        poses over the following chunk_len frames minus the pose one frame before. Anchoring them
        at the live pose gives the demo's motion at this phase, wherever the demo's scene put it.
        """
        nodes = self._phase_nodes(start_node, phase)
        t = self.t_raw[nodes].copy()
        last = self.last[nodes]
        for k in range(len(t)):
            stop = last[k] - chunk_len if max_skip is None else min(last[k] - chunk_len, t[k] + max_skip)
            while t[k] < stop and self._still(int(t[k]), move_eps):
                t[k] += 1
        idx = np.minimum(t[:, None] + np.arange(chunk_len, dtype=np.int64), last[:, None])
        prev = np.maximum(t - 1, self.first[nodes])
        return self.ee_base[idx] - self.ee_base[prev][:, None, :]

    def __call__(self, start_node: np.ndarray, chunk_len: int) -> np.ndarray:
        """[...] start nodes -> [..., chunk_len, 6] float32 raw poses, edge-padded at the demo's end.

        With relative=True (ROBOT_OOD_REF=rawnj) the poses are the demo's own displacements from the
        frame before the segment start, so anchoring them at the live pose gives a pure delta chunk:
        row 0 is the demo's step into the segment instead of a saturated jump toward the demo's pose.
        """
        if chunk_len < 1:
            raise ValueError(f"chunk_len must be >= 1, got {chunk_len}")
        s = np.asarray(start_node, dtype=np.int64)
        idx = np.minimum(self.t_raw[s][..., None] + np.arange(chunk_len, dtype=np.int64),
                         self.last[s][..., None])
        poses = self.ee_base[idx]
        if not self.relative:
            return poses
        prev = np.maximum(self.t_raw[s] - 1, self.first[s])
        return poses - self.ee_base[prev][..., None, :]


def raw_pose_ref_for(nodes: "NodeTable | None") -> "RawPoseRef | None":
    """The RawPoseRef ROBOT_OOD_REF=raw asks for, or None under the default node reference."""
    if REF_MODE not in ("raw", "rawnj") or nodes is None:
        return None
    return RawPoseRef.from_nodes(nodes, relative=REF_MODE == "rawnj")


# ==== Cached inputs ====
@dataclass(frozen=True, eq=False)
class PolicyActionCache:
    """The frozen policy's replayed per-frame actions, aligned 1:1 with NodeTable.q_raw.

    Attributes:
        a_pi_raw: [N_raw, 7] float32 commanded actions, gripper already binarized to +-1.
        raw_ptr: [n_demos + 1] int64 CSR boundaries, equal to NodeTable.raw_ptr.
    """

    a_pi_raw: np.ndarray
    raw_ptr: np.ndarray

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "PolicyActionCache":
        """Read a_pi_raw.npz.

        Args:
            path: The .npz, or the directory holding it.

        Returns:
            The cache.

        Raises:
            FileNotFoundError: Nothing is at the resolved path.
        """
        src = Path(path)
        src = src if src.suffix == ".npz" else src / ACTIONS_NPZ
        if not src.is_file():
            raise FileNotFoundError(
                f"{src}: no cached policy actions. Build them with "
                f"`python -m robot_ood.cli cache-policy --suite <suite>`."
            )
        with np.load(os.fspath(src)) as archive:
            return cls(
                a_pi_raw=np.asarray(archive["a_pi_raw"], dtype=np.float32),
                raw_ptr=np.asarray(archive["raw_ptr"], dtype=np.int64),
            )

    def check_against(self, nodes: "NodeTable") -> "PolicyActionCache":
        """Refuse a cache built against a different corpus.

        Args:
            nodes: The node table the cache must align with.

        Returns:
            self, so the check can be chained onto a load.

        Raises:
            ValueError: The frame count or the demo boundaries disagree.
        """
        if len(self.a_pi_raw) != nodes.n_raw:
            raise ValueError(
                f"cached policy actions have {len(self.a_pi_raw)} frames but the node table has "
                f"{nodes.n_raw}: the cache was built against a different corpus. Rebuild it."
            )
        if not np.array_equal(self.raw_ptr, np.asarray(nodes.raw_ptr, dtype=np.int64)):
            raise ValueError(
                "cached policy actions' raw_ptr does not match the node table's: the cache was "
                "built against a different demo ordering. Rebuild it."
            )
        return self

    def chunks(self, raw_idx: np.ndarray, k: int) -> np.ndarray:
        """The k actions the policy commands from each of several raw frames.

        Args:
            raw_idx: [n] global raw-frame offsets.
            k: Chunk length.

        Returns:
            [n, k, 7] float32, edge-padded at a demo's last frame so a chunk never reads across a
            demo boundary -- the same clipping NodeTable.ee_segment applies.
        """
        idx = np.asarray(raw_idx, dtype=np.int64)
        demo = np.searchsorted(self.raw_ptr, idx, side="right") - 1
        last = self.raw_ptr[demo + 1] - 1
        reach = np.minimum(idx[:, None] + np.arange(k, dtype=np.int64)[None, :], last[:, None])
        return self.a_pi_raw[reach]


# ==== The per-query-set target ====
@dataclass(frozen=True, eq=False)
class ChunkTargets:
    """One query set's chunk-blend inputs and target, built once and then only sliced.

    Attributes:
        a_policy: [n, K, 7] float32 the frozen policy's own chunk at each query's current frame.
        a_corr: [n, K, 6] float32 the pose-derived correction chunk, the regression target.
        ee_now: [n, 6] float32 each query window's own current base-frame end-effector pose.
        ee_ref: [n, K, 6] float32 the LABEL node's own reference segment -- what a_corr was built
            from. Carried so the alignment teacher can warp whatever segment the blend ends up
            following onto the correct one; see robot_ood.train.align.
    """

    a_policy: np.ndarray
    a_corr: np.ndarray
    ee_now: np.ndarray
    ee_ref: np.ndarray

    def __len__(self) -> int:
        return int(self.a_policy.shape[0])

    @property
    def chunk_len(self) -> int:
        """K, the action-chunk length."""
        return int(self.a_policy.shape[1])


@dataclass(eq=False)
class ChunkTargetBuilder:
    """Builds a ChunkTargets for any query set drawn from one fixed corpus.

    Attributes:
        nodes: The node table the queries index into.
        scale: Metres/radians per action unit.
        policy: The frozen policy's replayed actions.
        kinematics: Forward kinematics; poses the query window, and via NodeTable.ee_base the
            reference segment too, so both sides of the chunk are in the robot base frame.
        chunk_len: K, the action-chunk length; matches the deployed policy's chunk size.
        cart: The arm's robot_ood.core.kinematics.Arx5Kinematics on a joint-space lane, through
            whose Jacobian a P_SHIFT displacement becomes a joint displacement; None on a
            Cartesian lane.
    """

    nodes: "NodeTable"
    scale: ActionScale
    policy: PolicyActionCache
    kinematics: "PandaKinematics"
    chunk_len: int
    cart: Any = None

    def build(self, queries: Any, policy_corruption: Any = None) -> ChunkTargets:
        """Assemble one query set's targets.

        Args:
            queries: A mapping from robot_ood.train.data.make_queries. It must carry t_raw_now,
                which is the query window's OWN current raw frame -- the label node's t_raw is up
                to coarsen-1 frames behind it and would misalign the policy chunk.
            policy_corruption: A PolicyCorruption (robot_ood.train.augment), or None to leave every
                row's policy chunk as the policy really emitted it. Its P_SHIFT rows move ee_now
                instead of the chunk, and a_corr follows: the correct correction from where the
                arm now is does not depend on what the policy did.

        Returns:
            The targets, in query-set row order.

        Raises:
            KeyError: The mapping predates the t_raw_now column.
        """
        if "t_raw_now" not in queries:
            raise KeyError(
                "make_queries result has no t_raw_now column -- it predates the chunk-blend "
                "target and cannot be aligned to the policy-action cache."
            )
        t_now = np.asarray(queries["t_raw_now"], dtype=np.int64)
        q_now = np.asarray(queries["q_hist"], dtype=np.float64)[:, -1, :]
        ee_now = self.kinematics.pose(q_now)
        if policy_corruption is not None:
            ee_now = self._shifted(ee_now, policy_corruption)
        # Saturated exactly as a_track is: the target must be a chunk the controller could
        # actually emit, or the fit spends itself reaching for values the simulator clips.
        track = EeTrack(self.scale, bound=self.scale.limit)
        ee_ref = np.stack(
            [
                np.asarray(self.nodes.ee_segment(int(v), self.chunk_len), dtype=np.float64)
                for v in np.asarray(queries["tgt_node"], dtype=np.int64)
            ],
            axis=0,
        )
        a_corr = np.stack(
            [track.chunk(ee_ref[i], ee_now[i]) for i in range(len(ee_now))], axis=0
        )
        a_policy = self.policy.chunks(t_now, self.chunk_len).astype(np.float32)
        if policy_corruption is not None:
            a_policy = policy_corruption.apply(a_policy, self.policy, self.chunk_len)
        return ChunkTargets(
            a_policy=a_policy,
            a_corr=a_corr.astype(np.float32),
            ee_now=ee_now.astype(np.float32),
            ee_ref=ee_ref.astype(np.float32),
        )

    def _shifted(self, ee_now: np.ndarray, policy_corruption: Any) -> np.ndarray:
        """ee_now with the corruption's P_SHIFT rows displaced in the lane's own pose coordinates."""
        from robot_ood.core.kinematics import lane_displacement

        rows = policy_corruption.shift_rows()
        if rows.size == 0:
            return ee_now
        delta = lane_displacement(
            torch.as_tensor(policy_corruption.shift[rows], dtype=torch.float64),
            torch.as_tensor(ee_now[rows], dtype=torch.float64), self.cart,
        ).numpy()
        out = np.array(ee_now, dtype=np.float64, copy=True)
        out[rows] += delta
        return out
