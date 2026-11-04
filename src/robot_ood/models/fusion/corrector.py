"""Corrector: the one correction learner.

    G      = d * A                               d in (0, 1) per check, A in (0, 1)^{C x 6} per row and dim
    a_exec = a_policy + G * (a_ref - a_policy)   pose dims 0:6; the gripper passes through verbatim

a_ref is the shortlist's posterior-weighted pose target off the detached retriever, so the step is a
constant and one MSE moves d (trust, per-check features: follow the retrieval?) and A (shape,
per-row features: where the correction acts) together; corrector_features serves train and deploy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from robot_ood.core.action import ActionScale, POSE_DIM
from robot_ood.core.schemas import GRIPPER_COL, SEG_TOPK
from robot_ood.models.base import CheckpointedModule, i64
from robot_ood.models.fusion.target import REF_MODE, consensus_reference, track_chunk

__all__ = [
    "N_CHK_FEAT", "N_GATE", "N_ROW_FEAT", "N_TASK", "Corrector", "CorrectorFeatures",
    "CorrectorOutputs",
    "confidence_features", "consensus_outlier", "corrector_features", "optimal_dose",
    "resample_torch",
]

_POS = slice(0, 3)
_ROT = slice(3, GRIPPER_COL)
_POSE = slice(0, GRIPPER_COL)

_N_CONF: Final[int] = 3
_N_EE_REL: Final[int] = 2 * POSE_DIM
N_ROW_FEAT: Final[int] = 2 * GRIPPER_COL + _N_EE_REL       # [pose_policy | pose_agg | ee_rel] = 24
N_GATE: Final[int] = 2
N_TASK: Final[int] = 2
# conf, phase, dphase, top1, step, offset, outlier, spread, gate, task
N_CHK_FEAT: Final[int] = _N_CONF + 3 + 2 + 2 + 1 + 1 + N_GATE + N_TASK

_SHAPE_INIT_BIAS: Final[float] = -1.0     # sigmoid(-1) ~ 0.27: an untrained shape head trusts the policy
_TRUST_INIT_BIAS: Final[float] = 0.0      # sigmoid(0) = 0.5
_STD_FLOOR: Final[float] = 1e-6
SCORE_CLAMP: Final[float] = 1e4


def resample_torch(chunk: Tensor, length: int) -> Tensor:
    """Differentiable linear interpolation of [..., T, D] along time to [..., length, D]."""
    if chunk.shape[-2] == length:
        return chunk
    lead, t, d = chunk.shape[:-2], chunk.shape[-2], chunk.shape[-1]
    flat = chunk.reshape(-1, t, d).transpose(1, 2)
    out = F.interpolate(flat, size=length, mode="linear", align_corners=True)
    return out.transpose(1, 2).reshape(*lead, length, d)


def confidence_features(weights: Tensor, cand_actions: Tensor, a_policy: Tensor) -> Tensor:
    """[top1 mass, entropy, mean|candidate - policy|] over the shortlist.

    Args:
        weights: [B, K] posterior over the candidates (softmax of their scores).
        cand_actions: [B, K, T, ACTION_DIM] candidate chunks, already on the policy's time axis.
        a_policy: [B, C, ACTION_DIM] the policy's own chunk.

    Returns:
        [B, 3] float tensor.
    """
    eps = torch.finfo(weights.dtype).tiny
    top1 = weights.max(dim=-1).values
    entropy = -(weights.clamp_min(eps) * weights.clamp_min(eps).log()).sum(dim=-1)
    disagree = (cand_actions - a_policy[:, None, ..., : cand_actions.shape[-1]]).abs().mean(dim=(1, 2, 3))
    return torch.stack([top1, entropy, disagree], dim=-1)


# A geometric heuristic, so it is an INPUT to the trust head; the label comes from the corruptor.
def consensus_outlier(a_policy: Tensor, cand_actions: Tensor) -> Tensor:
    """Whether the policy's chunk falls outside the spread of what the retrieved demos do here.

    Args:
        a_policy: [B, C, ACTION_DIM] the policy's chunk.
        cand_actions: [B, K, C, ACTION_DIM] the retrieved candidates' chunks, resampled to C.

    Returns:
        [B] float32, 1 where the chunk's pose block lies farther from the candidates' mean than
        the farthest candidate does, 0 where a demo already behaves like the policy.
    """
    mean = cand_actions[..., _POSE].mean(dim=1)
    spread = (cand_actions[..., _POSE] - mean[:, None]).flatten(2).norm(dim=-1).max(dim=1).values
    reach = (a_policy[..., _POSE] - mean).flatten(1).norm(dim=-1)
    return (reach > spread).to(torch.float32)


def optimal_dose(a_policy: Tensor, a_ref: Tensor, target: Tensor) -> Tensor:
    """The scalar dose minimising mse(a_policy + d * (a_ref - a_policy), target), per row -- an eval oracle.

    Args:
        a_policy: [B, C, ACTION_DIM] the policy's chunk.
        a_ref: [B, C, 6] the reference chunk.
        target: [B, C, 6] the target chunk.

    Returns:
        [B] float in [0, 1].
    """
    b = a_policy.shape[0]
    step = (a_ref[..., _POSE] - a_policy[..., _POSE]).reshape(b, -1)
    gap = (target[..., _POSE] - a_policy[..., _POSE]).reshape(b, -1)
    denom = (step * step).sum(dim=-1)
    ratio = (gap * step).sum(dim=-1) / denom.clamp_min(1e-12)
    return torch.where(denom > 0, ratio, torch.zeros_like(denom)).clamp(0.0, 1.0)


def _weighted(p: Tensor, values: Tensor) -> Tensor:
    """sum_k p_k * values_k over the K candidate axis (axis 1)."""
    return (p.reshape(*p.shape, *([1] * (values.dim() - p.dim()))) * values).sum(dim=1)


@dataclass(frozen=True, slots=True)
class CorrectorFeatures:
    """What one check hands the two heads.

    Attributes:
        z_row: [B, C, N_ROW_FEAT] per-row features, the shape head's input.
        z_chk: [B, N_CHK_FEAT] per-check features, the trust head's input.
        a_ref: [B, C, 6] the posterior-weighted tracking chunk the correction steps toward.
        authority: [B] the candidate mass that is not a pause, in [0, 1]. A pause is a hold
            segment of a demo whose arm moved: a scripted wait or the park after the arm's last
            move, which says nothing about where the arm should go and so cannot command a step.
    """

    z_row: Tensor
    z_chk: Tensor
    a_ref: Tensor
    authority: Tensor


@dataclass(frozen=True, slots=True)
class CorrectorOutputs:
    """One corrector forward pass.

    Attributes:
        a_exec: [B, C, ACTION_DIM] a_policy + (trust * authority * shape) * (a_ref - a_policy),
            gripper verbatim.
        trust: [B] d in (0, 1).
        shape: [B, C, 6] A in (0, 1).
        features: What the heads read, kept so diagnostics see the same numbers.
    """

    a_exec: Tensor
    trust: Tensor
    shape: Tensor
    features: CorrectorFeatures


def _check_shapes(
    a_policy: Tensor, cand_actions: Tensor, cand_score: Tensor, cand_phase: Tensor,
    query_phase: Tensor, ee_ref: Tensor, ee_now: Tensor, cand_ee_start: Tensor, cand_ee_end: Tensor,
    cand_pause: Tensor | None = None,
) -> None:
    if cand_actions.shape[-2] != a_policy.shape[-2]:
        raise ValueError(
            f"cand_actions must already be resampled to a_policy's chunk length "
            f"({a_policy.shape[-2]}), got T={cand_actions.shape[-2]} -- resample at the call site "
            "(robot_ood.train.loss.align_candidates / CorrectionAuthority.apply), not here."
        )
    if cand_score.shape != cand_phase.shape or cand_score.shape[1] != cand_actions.shape[1]:
        raise ValueError(
            f"cand_score {tuple(cand_score.shape)}, cand_phase {tuple(cand_phase.shape)} and "
            f"cand_actions' K={cand_actions.shape[1]} must all agree on K"
        )
    if query_phase.shape != cand_score.shape[:1]:
        raise ValueError(f"query_phase must be [B]={tuple(cand_score.shape[:1])}, got {tuple(query_phase.shape)}")
    if ee_ref.shape[:2] != cand_actions.shape[:2] or ee_ref.shape[2] != a_policy.shape[1]:
        raise ValueError(
            f"ee_ref must be [B, K, C, 6]={(*cand_actions.shape[:2], a_policy.shape[1], POSE_DIM)}, "
            f"got {tuple(ee_ref.shape)}"
        )
    if ee_now.shape != (a_policy.shape[0], POSE_DIM):
        raise ValueError(f"ee_now must be [B, {POSE_DIM}]={(a_policy.shape[0], POSE_DIM)}, got {tuple(ee_now.shape)}")
    expect = (*cand_actions.shape[:2], POSE_DIM)
    if cand_ee_start.shape != expect or cand_ee_end.shape != expect:
        raise ValueError(
            f"cand_ee_start/cand_ee_end must be [B, K, {POSE_DIM}]={expect}, got "
            f"{tuple(cand_ee_start.shape)}/{tuple(cand_ee_end.shape)}"
        )
    if cand_pause is not None and cand_pause.shape != cand_score.shape:
        raise ValueError(f"cand_pause must be [B, K]={tuple(cand_score.shape)}, got {tuple(cand_pause.shape)}")


# Everything here is a constant to the heads (the retriever is detached upstream), hence no_grad.
@torch.no_grad()
def corrector_features(
    a_policy: Tensor,
    cand_actions: Tensor,
    cand_score: Tensor,
    cand_phase: Tensor,
    query_phase: Tensor,
    ee_ref: Tensor,
    ee_now: Tensor,
    scale: ActionScale,
    cand_ee_start: Tensor,
    cand_ee_end: Tensor,
    gate: Tensor | None = None,
    cand_task_p: Tensor | None = None,
    cand_task_id: Tensor | None = None,
    cand_pause: Tensor | None = None,
    cart: Any = None,
) -> CorrectorFeatures:
    """The two feature rows and the reference chunk, computed identically by training and deploy.

    Args:
        a_policy: [B, C, ACTION_DIM] the policy's own chunk.
        cand_actions: [B, K, C, ACTION_DIM] candidate chunks, already resampled to C.
        cand_score: [B, K] raw compatibility scores.
        cand_phase: [B, K] each candidate's phase_start.
        query_phase: [B] the query's own current phase estimate.
        ee_ref: [B, K, C, 6] each candidate's own node-interpolated pose trajectory.
        ee_now: [B, 6] the query window's own live base-frame end-effector pose.
        scale: Metres/radians per action unit.
        cand_ee_start: [B, K, 6] each candidate's own ee_start.
        cand_ee_end: [B, K, 6] each candidate's own ee_end.
        gate: [B, N_GATE] the locator's own view of this check -- the arm's distance to the nearest
            demo configuration and the belief's entropy -- or None for zeros.
        cand_task_p: [B, K] the belief's mass on each candidate's task, or None for zeros.
        cand_task_id: [B, K] each candidate's task id, needed with cand_task_p.
        cand_pause: [B, K] whether each candidate is a pause (robot_ood.data.segments.segment_pause),
            or None for no pause anywhere.
        cart: The arm's robot_ood.core.kinematics.Arx5Kinematics on a joint-space lane, whose
            reference then follows the candidates only along the Cartesian directions they agree
            on (robot_ood.models.fusion.target.consensus_reference), or None on a Cartesian lane.

    Returns:
        CorrectorFeatures.

    Raises:
        ValueError: A shape does not line up.
    """
    _check_shapes(a_policy, cand_actions, cand_score, cand_phase, query_phase, ee_ref, ee_now,
                  cand_ee_start, cand_ee_end, cand_pause)
    b, c = a_policy.shape[0], a_policy.shape[1]
    pose_policy = a_policy[..., _POSE]
    p = torch.softmax(cand_score.clamp(-SCORE_CLAMP, SCORE_CLAMP), dim=-1)

    pose_agg = _weighted(p, cand_actions[..., _POSE])
    if REF_MODE == "rawnj":
        ee_ref = ee_ref + ee_now[:, None, None, :]       # demo displacements anchored at the live pose
    ee_ref_abs = consensus_reference(ee_ref, p, ee_now, cart, a_policy, scale)
    a_ref = track_chunk(ee_ref_abs, ee_now, scale, bound=scale.limit)
    ee_rel = torch.cat([_weighted(p, cand_ee_start) - ee_now, _weighted(p, cand_ee_end) - ee_now], dim=-1)
    z_row = torch.cat([pose_policy, pose_agg, ee_rel[:, None, :].expand(b, c, _N_EE_REL)], dim=-1)

    rows = torch.arange(b, device=a_policy.device)
    top1 = cand_score.argmax(dim=-1)
    step = a_ref - pose_policy
    offset = cand_ee_start[rows, top1] - ee_now
    # Weighted spread of the candidates' end poses: do the retrieved continuations agree on a goal.
    end_mean = _weighted(p, cand_ee_end)
    spread = (p * (cand_ee_end[..., _POS] - end_mean[:, None, _POS]).norm(dim=-1) ** 2).sum(-1).sqrt()
    if gate is None:
        gate = torch.zeros(b, N_GATE, dtype=a_policy.dtype, device=a_policy.device)
    task = torch.zeros(b, N_TASK, dtype=a_policy.dtype, device=a_policy.device)
    if cand_task_p is not None and cand_task_id is not None:
        # Is the chunk about to be blended drawn from the task the locator believes we are doing,
        # and do the candidates agree on one task at all.
        on_task = (p * cand_task_p.to(p.dtype)).sum(-1)
        n_task = int(cand_task_id.max().item()) + 1 if cand_task_id.numel() else 1
        by_task = torch.zeros(b, n_task, dtype=p.dtype, device=p.device)
        by_task.scatter_add_(1, cand_task_id.to(torch.int64), p)
        task = torch.stack([on_task, by_task.max(dim=-1).values], dim=-1).to(a_policy.dtype)
    z_chk = torch.cat(
        [
            confidence_features(p, cand_actions, a_policy),
            query_phase[:, None],
            ((p * cand_phase).sum(dim=-1) - query_phase)[:, None],
            cand_score.max(dim=-1).values[:, None],
            step[..., _POS].norm(dim=-1).mean(dim=-1)[:, None],
            step[..., _ROT].norm(dim=-1).mean(dim=-1)[:, None],
            offset[..., _POS].norm(dim=-1)[:, None],
            offset[..., _ROT].norm(dim=-1)[:, None],
            consensus_outlier(a_policy, cand_actions)[:, None],
            spread[:, None],
            gate.to(a_policy.dtype),
            task,
        ],
        dim=-1,
    )
    authority = torch.ones(b, dtype=p.dtype, device=p.device)
    if cand_pause is not None:
        authority = authority - (p * cand_pause.to(p.dtype)).sum(dim=-1)
    return CorrectorFeatures(z_row=z_row.to(torch.float32), z_chk=z_chk.to(torch.float32),
                             a_ref=a_ref.to(torch.float32), authority=authority.to(torch.float32))


def _unit_where_flat(std: Tensor) -> Tensor:
    """A standard deviation per feature, unit where the sample showed none, so deploy cannot amplify it."""
    return torch.where(std > _STD_FLOOR, std, torch.ones_like(std))


def _mlp(n_in: int, hidden: int, n_out: int, last_bias: float) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, n_out))
    nn.init.zeros_(net[-1].weight)
    nn.init.constant_(net[-1].bias, last_bias)
    return net


class Corrector(CheckpointedModule, nn.Module):
    """The shape head A and the trust head d, over standardised features.

    Attributes:
        chunk_len: C, the policy's action-chunk length -- a checkpoint stamp; forward reads a_policy.
        k: The candidate count the corrector was built expecting -- a stamp; forward reads cand_score.
        hidden: Width of the shape head's hidden layer.
        trust_hidden: Width of the trust head's hidden layer.
        row_mean, row_std, chk_mean, chk_std: Feature standardisation, fitted by fit_feature_stats
            off a training sample and carried in the checkpoint; a feature that did not vary over
            the sample keeps unit scale.
    """

    def __init__(self, chunk_len: int, *, k: int = SEG_TOPK, hidden: int = 128,
                 trust_hidden: int = 64) -> None:
        super().__init__()
        self.chunk_len, self.k = int(chunk_len), int(k)
        self.hidden, self.trust_hidden = int(hidden), int(trust_hidden)
        self.register_buffer("row_mean", torch.zeros(N_ROW_FEAT))
        self.register_buffer("row_std", torch.ones(N_ROW_FEAT))
        self.register_buffer("chk_mean", torch.zeros(N_CHK_FEAT))
        self.register_buffer("chk_std", torch.ones(N_CHK_FEAT))
        self.shape = _mlp(N_ROW_FEAT, self.hidden, GRIPPER_COL, _SHAPE_INIT_BIAS)
        self.trust = _mlp(N_CHK_FEAT, self.trust_hidden, 1, _TRUST_INIT_BIAS)

    def fit_feature_stats(self, z_row: Tensor, z_chk: Tensor) -> "Corrector":
        """Measure the standardisation statistics off a training sample.

        Args:
            z_row: [..., N_ROW_FEAT] row features over the sample.
            z_chk: [..., N_CHK_FEAT] check features over the sample.

        Returns:
            self, so the fit can be chained onto construction.

        Raises:
            ValueError: Fewer than two rows, which cannot give a standard deviation.
        """
        row, chk = z_row.reshape(-1, N_ROW_FEAT), z_chk.reshape(-1, N_CHK_FEAT)
        if row.shape[0] < 2 or chk.shape[0] < 2:
            raise ValueError(f"need >= 2 rows to fit statistics, got {row.shape[0]}/{chk.shape[0]}")
        with torch.no_grad():
            self.row_mean.copy_(row.mean(dim=0))
            self.row_std.copy_(_unit_where_flat(row.std(dim=0)))
            self.chk_mean.copy_(chk.mean(dim=0))
            self.chk_std.copy_(_unit_where_flat(chk.std(dim=0)))
        return self

    def heads(self, z_row: Tensor, z_chk: Tensor) -> tuple[Tensor, Tensor]:
        """(trust [B], shape [B, C, 6]) off already-built features."""
        shape = torch.sigmoid(self.shape((z_row - self.row_mean) / self.row_std))
        trust = torch.sigmoid(self.trust((z_chk - self.chk_mean) / self.chk_std)).squeeze(-1)
        return trust, shape

    @staticmethod
    def blend(a_policy: Tensor, a_ref: Tensor, gain: Tensor) -> Tensor:
        """a_policy + gain * (a_ref - a_policy) on the pose block, gripper from a_policy.

        Args:
            a_policy: [B, C, ACTION_DIM] the policy's chunk.
            a_ref: [B, C, 6] the reference chunk.
            gain: [B, C, 6] per-row, per-dim gain in [0, 1].

        Returns:
            [B, C, ACTION_DIM].
        """
        pose = a_policy[..., _POSE] + gain * (a_ref - a_policy[..., _POSE])
        return torch.cat([pose, a_policy[..., GRIPPER_COL:]], dim=-1)

    def forward(
        self,
        a_policy: Tensor,
        cand_actions: Tensor,
        cand_score: Tensor,
        cand_phase: Tensor,
        query_phase: Tensor,
        ee_ref: Tensor,
        ee_now: Tensor,
        scale: ActionScale,
        cand_ee_start: Tensor,
        cand_ee_end: Tensor,
        gate: Tensor | None = None,
        cand_task_p: Tensor | None = None,
        cand_task_id: Tensor | None = None,
        cand_pause: Tensor | None = None,
        cart: Any = None,
    ) -> CorrectorOutputs:
        """Build the features, run both heads, blend.

        Args: see corrector_features.

        Returns:
            CorrectorOutputs; a_exec's pose block lies on the closed segment between a_policy and
            a_ref elementwise, because trust * authority * shape is in [0, 1].
        """
        feats = corrector_features(a_policy, cand_actions, cand_score, cand_phase, query_phase,
                                   ee_ref, ee_now, scale, cand_ee_start, cand_ee_end, gate,
                                   cand_task_p, cand_task_id, cand_pause, cart)
        trust, shape = self.heads(feats.z_row, feats.z_chk)
        a_exec = self.blend(a_policy, feats.a_ref, (trust * feats.authority)[:, None, None] * shape)
        return CorrectorOutputs(a_exec=a_exec, trust=trust, shape=shape, features=feats)

    def checkpoint_meta(self) -> dict[str, Any]:
        """The shape stamps, in the order the archive stores them."""
        return {"chunk_len": i64(self.chunk_len), "k": i64(self.k), "hidden": i64(self.hidden),
                "trust_hidden": i64(self.trust_hidden), "row_dim": i64(N_ROW_FEAT),
                "chk_dim": i64(N_CHK_FEAT)}

    @classmethod
    def from_checkpoint(cls, meta: Mapping[str, Any], src: Path) -> "Corrector":
        """Rebuild the module an archive describes, refusing a feature-width skew.

        Args:
            meta: The archive; read for the stamps.
            src: Where it came from, for the error message.

        Returns:
            The module, before its weights are loaded.

        Raises:
            ValueError: The checkpoint's feature widths are not this build's.
        """
        try:
            row_dim, chk_dim = int(meta["row_dim"]), int(meta["chk_dim"])
            net = cls(chunk_len=int(meta["chunk_len"]), k=int(meta["k"]), hidden=int(meta["hidden"]),
                      trust_hidden=int(meta["trust_hidden"]))
        except KeyError as exc:
            raise ValueError(
                f"{src}: no {exc.args[0]} stamped for the corrector. Retrain with "
                "`python -m robot_ood.cli train fcsr --suite <suite>`."
            ) from exc
        if (row_dim, chk_dim) != (N_ROW_FEAT, N_CHK_FEAT):
            raise ValueError(
                f"{src}: this checkpoint's heads read {row_dim}/{chk_dim}-wide inputs, this build "
                f"assembles {N_ROW_FEAT}/{N_CHK_FEAT} (corrector_features). Retrain with "
                "`python -m robot_ood.cli train fcsr --suite <suite>`."
            )
        return net
