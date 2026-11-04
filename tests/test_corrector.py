"""The Corrector: one gain G = trust * shape over one step a_ref - a_policy.

What these pin: the geometric guarantee (a_exec lies on the closed segment between a_policy and
a_ref, gripper untouched), that gradient reaches both heads and nothing upstream, that the feature
builder is one function shared by training and deploy (CorrectionAuthority.apply reproduces
Corrector.forward bit for bit), the standardisation stamp, persistence through the FCSR archive,
and that an archive carrying the retired safn./dose. halves is refused by name.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robot_ood.core.action import ActionScale, POSE_DIM
from robot_ood.core.schemas import GRIPPER_COL, SegmentCandidates
from robot_ood.models.fusion.authority import CorrectionAuthority
from robot_ood.models.fusion.corrector import (
    N_CHK_FEAT, N_ROW_FEAT, Corrector, consensus_outlier, corrector_features, optimal_dose,
)
from robot_ood.models.fusion.target import node_pose_ref

C, K, AD = 8, 4, 7
_SCALE = ActionScale(pos=0.0116, rot=0.1092, r2_pos=1.0, r2_rot=1.0)
_NODE_EE = (np.arange(64 * 6, dtype=np.float32).reshape(64, 6) / 400.0)


def _net(**kw) -> Corrector:
    return Corrector(chunk_len=C, k=K, hidden=16, trust_hidden=8, **kw)


def _trained_ish_net() -> Corrector:
    """Both last layers are zero-initialised, so nudge them off the constant for wiring tests."""
    net = _net()
    torch.nn.init.normal_(net.shape[-1].weight, std=0.5)
    torch.nn.init.normal_(net.trust[-1].weight, std=0.5)
    return net


def _batch(seed: int = 0):
    rng = torch.Generator().manual_seed(seed)
    a_policy = torch.randn(2, C, AD, generator=rng)
    a_policy[..., GRIPPER_COL] = torch.tensor([1.0, 0.0]).repeat(2, C // 2)
    cand = torch.randn(2, K, C, AD, generator=rng)
    score = torch.randn(2, K, generator=rng)
    phase = torch.rand(2, K, generator=rng)
    query_phase = torch.rand(2, generator=rng)
    ee_ref = torch.randn(2, K, C, POSE_DIM, generator=rng) * 0.1
    ee_now = torch.randn(2, POSE_DIM, generator=rng) * 0.1
    cand_ee_start = torch.randn(2, K, POSE_DIM, generator=rng) * 0.1
    cand_ee_end = torch.randn(2, K, POSE_DIM, generator=rng) * 0.1
    return a_policy, cand, score, phase, query_phase, ee_ref, ee_now, cand_ee_start, cand_ee_end


# ==================================================================================================
# geometry
# ==================================================================================================
def test_a_exec_lies_on_the_segment_between_a_policy_and_a_ref():
    net = _trained_ish_net()
    args = _batch()
    out = net(*args[:6], *args[6:7], _SCALE, *args[7:])
    pose, ref = args[0][..., :GRIPPER_COL], out.features.a_ref
    lo, hi = torch.minimum(pose, ref), torch.maximum(pose, ref)
    assert torch.all(out.a_exec[..., :GRIPPER_COL] >= lo - 1e-6)
    assert torch.all(out.a_exec[..., :GRIPPER_COL] <= hi + 1e-6)
    assert torch.all((out.trust > 0) & (out.trust < 1))
    assert torch.all((out.shape > 0) & (out.shape < 1))


def test_a_pause_candidate_carries_no_authority():
    net = _trained_ish_net()
    a_policy, cand, score, phase, query_phase, ee_ref, ee_now, s0, s1 = _batch()
    pause = torch.zeros(2, K, dtype=torch.bool)
    pause[0] = True
    pause[1, 0] = True
    out = net(a_policy, cand, score, phase, query_phase, ee_ref, ee_now, _SCALE, s0, s1, cand_pause=pause)
    free = net(a_policy, cand, score, phase, query_phase, ee_ref, ee_now, _SCALE, s0, s1)
    p = torch.softmax(score, dim=-1)
    torch.testing.assert_close(out.features.authority, 1.0 - (p * pause).sum(-1))
    torch.testing.assert_close(free.features.authority, torch.ones(2))
    torch.testing.assert_close(out.a_exec[0], a_policy[0])
    step_free = free.a_exec[1, :, :GRIPPER_COL] - a_policy[1, :, :GRIPPER_COL]
    step = out.a_exec[1, :, :GRIPPER_COL] - a_policy[1, :, :GRIPPER_COL]
    torch.testing.assert_close(step, step_free * out.features.authority[1])
    with pytest.raises(ValueError, match="cand_pause"):
        net(a_policy, cand, score, phase, query_phase, ee_ref, ee_now, _SCALE, s0, s1, cand_pause=pause[:, :1])


def test_the_gripper_passes_through_verbatim():
    net = _trained_ish_net()
    args = _batch()
    out = net(*args[:6], *args[6:7], _SCALE, *args[7:])
    assert torch.equal(out.a_exec[..., GRIPPER_COL], args[0][..., GRIPPER_COL])


def test_blend_at_gain_zero_and_one_reproduces_the_two_endpoints():
    a_policy, *_ = _batch()
    a_ref = torch.randn(2, C, GRIPPER_COL)
    assert torch.equal(Corrector.blend(a_policy, a_ref, torch.zeros(2, C, GRIPPER_COL)), a_policy)
    one = Corrector.blend(a_policy, a_ref, torch.ones(2, C, GRIPPER_COL))
    assert torch.allclose(one[..., :GRIPPER_COL], a_ref, atol=1e-6)


def test_an_untrained_corrector_trusts_the_policy():
    """shape bias -1 (sigmoid 0.27), trust bias 0 (0.5): the same constant on every input."""
    net = _net()
    args = _batch()
    out = net(*args[:6], *args[6:7], _SCALE, *args[7:])
    assert torch.allclose(out.trust, torch.full((2,), 0.5))
    assert torch.allclose(out.shape, torch.full((2, C, GRIPPER_COL), float(torch.sigmoid(torch.tensor(-1.0)))))


# ==================================================================================================
# gradient
# ==================================================================================================
def test_gradient_reaches_both_heads_and_nothing_upstream():
    net = _trained_ish_net()
    a_policy, cand, score, phase, qp, ee_ref, ee_now, s, e = _batch()
    cand = cand.clone().requires_grad_(True)
    ee_ref = ee_ref.clone().requires_grad_(True)
    score = score.clone().requires_grad_(True)
    out = net(a_policy, cand, score, phase, qp, ee_ref, ee_now, _SCALE, s, e)
    out.a_exec.sum().backward()
    assert torch.any(net.shape[-1].weight.grad != 0)
    assert torch.any(net.trust[-1].weight.grad != 0)
    for t in (cand, ee_ref, score):
        assert t.grad is None or torch.all(t.grad == 0)


# ==================================================================================================
# features
# ==================================================================================================
def test_feature_widths_are_the_stamped_constants():
    args = _batch()
    feats = corrector_features(*args[:6], *args[6:7], _SCALE, *args[7:])
    assert feats.z_row.shape == (2, C, N_ROW_FEAT)
    assert feats.z_chk.shape == (2, N_CHK_FEAT)
    assert feats.a_ref.shape == (2, C, GRIPPER_COL)
    assert torch.isfinite(feats.z_row).all() and torch.isfinite(feats.z_chk).all()


def test_ee_rel_is_relative_never_absolute():
    """Shifting every pose operand by the same offset must leave z_row's ee_rel block unchanged."""
    a_policy, cand, score, phase, qp, ee_ref, ee_now, s, e = _batch()
    shift = torch.tensor([0.3, -0.2, 0.1, 0.0, 0.0, 0.0])
    f0 = corrector_features(a_policy, cand, score, phase, qp, ee_ref, ee_now, _SCALE, s, e)
    f1 = corrector_features(a_policy, cand, score, phase, qp, ee_ref + shift, ee_now + shift, _SCALE,
                            s + shift, e + shift)
    torch.testing.assert_close(f0.z_row[..., 2 * GRIPPER_COL:], f1.z_row[..., 2 * GRIPPER_COL:], atol=1e-5, rtol=1e-5)


def test_consensus_outlier_flags_a_chunk_outside_the_candidates_spread():
    cand = torch.zeros(1, K, C, AD)
    cand[0, 0, :, 0] = 1.0                       # spread = 0.75 * sqrt(C) around the mean
    inside = cand.mean(dim=1)
    outside = inside.clone()
    outside[..., 0] += 10.0
    assert consensus_outlier(inside, cand).item() == 0.0
    assert consensus_outlier(outside, cand).item() == 1.0


def test_fit_feature_stats_standardises_to_zero_mean_unit_variance():
    net = _net()
    rng = torch.Generator().manual_seed(1)
    z_row = torch.randn(64, C, N_ROW_FEAT, generator=rng) * 3.0 + 2.0
    z_chk = torch.randn(64, N_CHK_FEAT, generator=rng) * 0.5 - 1.0
    net.fit_feature_stats(z_row, z_chk)
    rows = (z_row.reshape(-1, N_ROW_FEAT) - net.row_mean) / net.row_std
    chk = (z_chk - net.chk_mean) / net.chk_std
    assert torch.allclose(rows.mean(dim=0), torch.zeros(N_ROW_FEAT), atol=1e-4)
    assert torch.allclose(rows.std(dim=0), torch.ones(N_ROW_FEAT), atol=1e-3)
    assert torch.allclose(chk.mean(dim=0), torch.zeros(N_CHK_FEAT), atol=1e-4)
    with pytest.raises(ValueError, match=">= 2 rows"):
        net.fit_feature_stats(z_row[:1], z_chk[:1])


def test_fit_feature_stats_leaves_a_flat_feature_at_unit_scale():
    net = _net()
    rng = torch.Generator().manual_seed(2)
    z_row = torch.randn(32, C, N_ROW_FEAT, generator=rng)
    z_chk = torch.randn(32, N_CHK_FEAT, generator=rng)
    z_chk[:, 0] = 0.0
    net.fit_feature_stats(z_row, z_chk)
    assert float(net.chk_std[0]) == 1.0
    assert float(((z_chk[:, 1:] - net.chk_mean[1:]) / net.chk_std[1:]).std(dim=0).mean()) == pytest.approx(1.0, abs=1e-3)


def test_optimal_dose_is_the_one_dimensional_minimiser():
    rng = torch.Generator().manual_seed(3)
    a_policy = torch.randn(5, C, AD, generator=rng)
    a_ref = torch.randn(5, C, GRIPPER_COL, generator=rng)
    target = torch.randn(5, C, GRIPPER_COL, generator=rng)

    def err(d):
        g = d.reshape(-1, 1, 1).expand(5, C, GRIPPER_COL)
        return ((Corrector.blend(a_policy, a_ref, g)[..., :GRIPPER_COL] - target) ** 2).reshape(5, -1).sum(-1)

    best = err(optimal_dose(a_policy, a_ref, target))
    for t in np.linspace(0.0, 1.0, 21):
        assert torch.all(best <= err(torch.full((5,), float(t))) + 1e-5)
    assert torch.all(optimal_dose(a_policy, a_ref, a_policy[..., :GRIPPER_COL]) == 0.0)
    assert torch.all(optimal_dose(a_policy, a_ref, a_ref) == 1.0)


# ==================================================================================================
# deploy parity
# ==================================================================================================
def _cands(seg_len: int = 16) -> SegmentCandidates:
    rng = np.random.default_rng(0)
    return SegmentCandidates(
        idx=np.arange(K, dtype=np.int32),
        score=rng.normal(size=K).astype(np.float32),
        action_chunk=rng.normal(size=(K, seg_len, AD)).astype(np.float32),
        phase=rng.random(K).astype(np.float32),
        ee_start=rng.normal(size=(K, 6)).astype(np.float32) * 0.1,
        ee_end=rng.normal(size=(K, 6)).astype(np.float32) * 0.1,
        start_node=np.arange(K, dtype=np.int32),
        end_node=(np.arange(K, dtype=np.int32) + 10),
    )


def test_authority_apply_reproduces_corrector_forward():
    """Deploy builds the same tensors training does; the resampled 16-row candidates, the
    node-interpolated ee_ref and the live pose all reach one forward."""
    net = _trained_ish_net().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    authority = CorrectionAuthority(corrector=net, node_ee=_NODE_EE, scale=_SCALE)
    cands = _cands()
    a_policy = np.random.default_rng(1).normal(size=(C, AD))
    ee = np.array([0.1, 0.2, 0.3, 0.0, 0.1, 0.0])
    out = authority.apply(a_policy, cands, ee, query_phase=0.4)

    from robot_ood.models.fusion.corrector import resample_torch

    policy = torch.as_tensor(a_policy.astype(np.float32))[None]
    actions = resample_torch(torch.as_tensor(cands.action_chunk)[None], C)
    ee_ref = torch.as_tensor(node_pose_ref(_NODE_EE, cands.start_node, cands.end_node, C))[None]
    ref = net(policy, actions, torch.as_tensor(cands.score)[None], torch.as_tensor(cands.phase)[None],
              torch.tensor([0.4]), ee_ref, torch.as_tensor(ee.astype(np.float32))[None], _SCALE,
              torch.as_tensor(cands.ee_start)[None], torch.as_tensor(cands.ee_end)[None])
    np.testing.assert_allclose(out.a_exec, ref.a_exec[0].numpy(), atol=1e-6)
    np.testing.assert_allclose(out.trust, ref.trust[0].numpy(), atol=1e-6)
    assert out.a_exec.shape == (C, AD) and out.shape.shape == (C, GRIPPER_COL)
    np.testing.assert_allclose(out.a_exec[:, GRIPPER_COL], a_policy[:, GRIPPER_COL], atol=1e-6)


def test_authority_apply_reads_pauses_by_segment_id():
    net = _trained_ish_net().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    cands = _cands()
    pause = np.zeros(K + 3, dtype=bool)
    pause[cands.idx[:2]] = True
    a_policy = np.random.default_rng(1).normal(size=(C, AD))
    ee = np.array([0.1, 0.2, 0.3, 0.0, 0.1, 0.0])
    out = CorrectionAuthority(corrector=net, node_ee=_NODE_EE, scale=_SCALE, pause=pause).apply(
        a_policy, cands, ee, query_phase=0.4)
    free = CorrectionAuthority(corrector=net, node_ee=_NODE_EE, scale=_SCALE).apply(
        a_policy, cands, ee, query_phase=0.4)
    p = torch.softmax(torch.as_tensor(cands.score), dim=-1).numpy()
    np.testing.assert_allclose(out.authority, 1.0 - p[:2].sum(), atol=1e-6)
    np.testing.assert_allclose(free.authority, 1.0)
    np.testing.assert_allclose(out.a_exec[:, :GRIPPER_COL] - a_policy[:, :GRIPPER_COL],
                               (free.a_exec[:, :GRIPPER_COL] - a_policy[:, :GRIPPER_COL]) * out.authority,
                               atol=1e-6)


def test_authority_apply_requires_query_phase():
    authority = CorrectionAuthority(corrector=_net(), node_ee=_NODE_EE, scale=_SCALE)
    with pytest.raises(KeyError, match="query_phase"):
        authority.apply(np.zeros((C, AD)), _cands(), np.zeros(6))


# ==================================================================================================
# persistence
# ==================================================================================================
def test_round_trip_through_the_fcsr_archive_keeps_weights_and_stats(tmp_path):
    from conftest import fcsr_round_trip

    net = _trained_ish_net()
    net.fit_feature_stats(torch.randn(8, N_ROW_FEAT) * 2 + 1, torch.randn(8, N_CHK_FEAT) * 3 - 1)
    back = fcsr_round_trip(net, tmp_path, "corrector")
    for k, v in net.state_dict().items():
        assert torch.equal(v, back.state_dict()[k]), k
    assert (back.chunk_len, back.k, back.hidden, back.trust_hidden) == (C, K, 16, 8)


def test_a_feature_width_skew_is_refused_by_name(tmp_path):
    from conftest import fcsr_round_trip

    def bump(arrays):
        arrays["corrector.chk_dim"] = np.int64(N_CHK_FEAT + 1)

    with pytest.raises(ValueError, match="train fcsr"):
        fcsr_round_trip(_net(), tmp_path, "corrector", mutate=bump)


def test_an_archive_with_the_retired_heads_is_refused_by_name(tmp_path):
    from conftest import write_fcsr_archive
    from robot_ood.models.fcsr_net import FcsrNet

    def retire(arrays):
        for k in [k for k in arrays if k.startswith("corrector.")]:
            arrays["safn." + k[len("corrector."):]] = arrays.pop(k)

    path = write_fcsr_archive(tmp_path, mutate=retire)
    with pytest.raises(ValueError, match="safn"):
        FcsrNet.load(path)


def test_a_direct_authority_executes_the_fused_reference_at_full_gain():
    net = _trained_ish_net().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    cands = _cands()
    pause = np.zeros(K + 3, dtype=bool)
    pause[cands.idx[:2]] = True
    a_policy = np.random.default_rng(1).normal(size=(C, AD))
    ee = np.array([0.1, 0.2, 0.3, 0.0, 0.1, 0.0])
    out = CorrectionAuthority(corrector=net, node_ee=_NODE_EE, scale=_SCALE, pause=pause, direct=True).apply(
        a_policy, cands, ee, query_phase=0.4)
    learned = CorrectionAuthority(corrector=net, node_ee=_NODE_EE, scale=_SCALE, pause=pause).apply(
        a_policy, cands, ee, query_phase=0.4)
    np.testing.assert_allclose(out.trust, 1.0)
    np.testing.assert_allclose(out.shape, 1.0)
    np.testing.assert_allclose(out.a_ref, learned.a_ref, atol=1e-6)
    np.testing.assert_allclose(out.authority, learned.authority, atol=1e-6)
    np.testing.assert_allclose(out.a_exec[:, :GRIPPER_COL],
                               a_policy[:, :GRIPPER_COL] + out.authority * (out.a_ref - a_policy[:, :GRIPPER_COL]),
                               atol=1e-6)
    np.testing.assert_allclose(out.a_exec[:, GRIPPER_COL], a_policy[:, GRIPPER_COL], atol=1e-6)
    assert not np.allclose(out.a_exec, learned.a_exec)
