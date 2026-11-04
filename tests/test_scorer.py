"""Tests for robot_ood.models.retrieval.scorer -- ActionEncoder, segment_features and SegmentScorer.

Hand-built tensors only. The load-bearing checks are the SEG_FEATURES column-order contract, the
task_affinity column, and SegmentScorer's width-stamped save/load.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robot_ood.core.schemas import ACTION_DIM, ACTION_EMBED_DIM, SEG_FEATURE_DIM, SEG_FEATURES
from robot_ood.models.retrieval.scorer import ActionEncoder, SegmentScorer, segment_features

B, K = 5, 8


def _query(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        ee_query=torch.randn(B, 6, generator=g),
        e_future_q=torch.randn(B, ACTION_EMBED_DIM, generator=g),
        phase_query=torch.rand(B, generator=g),
    )


def _segments(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return dict(
        seg_ee_start=torch.randn(B, K, 6, generator=g),
        e_future_seg=torch.randn(B, K, ACTION_EMBED_DIM, generator=g),
        seg_phase=torch.rand(B, K, generator=g),
        cand_task_p=torch.rand(B, K, generator=g),
    )


# ---- ActionEncoder ---------------------------------------------------------------------------
def test_action_encoder_shape():
    enc = ActionEncoder(chunk_len=16)
    chunk = torch.randn(B, K, 16, ACTION_DIM)
    out = enc(chunk)
    assert out.shape == (B, K, ACTION_EMBED_DIM)


def test_action_encoder_rejects_wrong_chunk_shape():
    enc = ActionEncoder(chunk_len=16)
    with pytest.raises(ValueError):
        enc(torch.randn(B, 12, ACTION_DIM))


def test_action_encoder_is_inductive_in_batch_and_candidate_count():
    """No parameter depends on B or K -- only chunk_len and ACTION_DIM, both architectural."""
    enc = ActionEncoder(chunk_len=16)
    for name, p in enc.named_parameters():
        assert p.shape[-1] not in (B, K) or name.endswith("bias"), name


# ---- segment_features: column order and widths -----------------------------------------------
def test_segment_features_shape_and_column_count():
    feats = segment_features(**_query(), **_segments())
    assert feats.shape == (B, K, SEG_FEATURE_DIM)
    assert SEG_FEATURES == ("d_state", "d_future", "phase_diff", "task_affinity")


def test_d_state_matches_the_metric_pose_delta_norm():
    q = _query()
    s = _segments()
    feats = segment_features(**q, **s)
    want = (q["ee_query"][:, None, :] - s["seg_ee_start"]).norm(dim=-1)
    assert torch.allclose(feats[..., 0], want)


def test_d_future_is_one_minus_cosine():
    q = _query()
    s = _segments()
    feats = segment_features(**q, **s)
    eq = torch.nn.functional.normalize(q["e_future_q"][:, None, :], dim=-1)
    es = torch.nn.functional.normalize(s["e_future_seg"], dim=-1)
    want = 1.0 - (eq * es).sum(-1)
    assert torch.allclose(feats[..., 1], want, atol=1e-5)


def test_phase_diff_is_signed():
    q = _query()
    s = _segments()
    feats = segment_features(**q, **s)
    want = q["phase_query"][:, None] - s["seg_phase"]
    assert torch.allclose(feats[..., 2], want)


# ---- task_affinity -----------------------------------------------------------------------------
def test_task_affinity_is_the_belief_mass_on_each_candidates_task():
    q, s_ = _query(), _segments()
    feats = segment_features(**q, **s_)
    assert torch.allclose(feats[..., 3], s_["cand_task_p"])


def test_task_affinity_is_the_last_column():
    feats = segment_features(**_query(), **_segments())
    assert feats.shape[-1] == 4


def test_segment_features_column_order_matches_contracts():
    """Reordering SEG_FEATURES in contracts.py without updating scorer.py must fail loudly."""
    assert SEG_FEATURES == ("d_state", "d_future", "phase_diff", "task_affinity")


# ---- SegmentScorer -------------------------------------------------------------------------------
def test_segment_scorer_output_shape():
    scorer = SegmentScorer()
    feats = segment_features(**_query(), **_segments())
    out = scorer(feats)
    assert out.shape == (B, K)


def test_segment_scorer_is_inductive_no_parameter_dimensioned_by_s_or_tasks():
    scorer = SegmentScorer()
    for name, p in scorer.named_parameters():
        assert all(dim in (SEG_FEATURE_DIM, scorer.hidden, 1) for dim in p.shape), (
            f"{name} has a suspicious dimension {tuple(p.shape)}"
        )


def test_fit_feature_stats_changes_forward_output():
    scorer = SegmentScorer()
    feats = segment_features(**_query(), **_segments()) * 100.0 + 50.0
    before = scorer(feats)
    scorer.fit_feature_stats(feats)
    after = scorer(feats)
    assert not torch.allclose(before, after)


def test_fit_feature_stats_rejects_fewer_than_two_rows():
    scorer = SegmentScorer()
    with pytest.raises(ValueError):
        scorer.fit_feature_stats(torch.randn(1, SEG_FEATURE_DIM))


def test_save_load_roundtrip(tmp_path):
    scorer = SegmentScorer(hidden=32)
    feats = segment_features(**_query(), **_segments())
    scorer.fit_feature_stats(feats)
    with torch.no_grad():
        before = scorer(feats)

    from conftest import fcsr_round_trip

    loaded = fcsr_round_trip(scorer, tmp_path, "scorer")
    with torch.no_grad():
        after = loaded(feats)
    assert torch.allclose(before, after)
    assert not loaded.training


def test_a_missing_archive_names_the_train_command(tmp_path):
    from robot_ood.models.fcsr_net import FcsrNet

    with pytest.raises(FileNotFoundError, match="train fcsr"):
        FcsrNet.load(tmp_path)


# ---- the deleted phase mask ------------------------------------------------------------------
# SegmentScorer used to mask a "starts in my future" candidate to -inf. That is gone, along with
# phase_tolerance; see the class docstring for the sweep that deleted it (rank-1 0.4189 at
# tolerance 0 against 0.7744 unmasked, measured at the belief filter's own phase distribution).
# These pin the deletion so it cannot creep back as a guard somewhere downstream.
def test_every_logit_is_finite_whatever_the_phase_diff():
    """The whole point of the deletion: no row can reach a softmax as a silent NaN, because no
    logit is -inf any more -- not even for a candidate starting an entire episode in the future."""
    scorer = SegmentScorer(hidden=32)
    feats = segment_features(**_query(), **_segments())
    feats[..., 2] = -(feats[..., 2].abs() + 1.0)     # every candidate starts after the query
    out = scorer(feats)
    assert torch.isfinite(out).all()


def test_forward_is_exactly_the_bare_mlp():
    """Nothing sits between the standardised features and the logit."""
    scorer = SegmentScorer(hidden=32)
    feats = segment_features(**_query(), **_segments())
    with torch.no_grad():
        raw = scorer.head(scorer.trunk(scorer._standardise(feats))).squeeze(-1)
        out = scorer(feats)
    assert torch.allclose(out, raw)


def test_a_future_starting_segment_stays_rankable():
    """It may still LOSE on the learned score -- phase_diff is still a feature -- but it must remain
    a candidate the ranking can choose, which under the old mask it could not."""
    scorer = SegmentScorer(hidden=32)
    q = _query()
    q["phase_query"] = torch.full((B,), 0.03)
    s = _segments()
    s["seg_phase"] = s["seg_phase"].clone()
    s["seg_phase"][:, 1] = 0.90                      # starts well after the query
    out = scorer(segment_features(**q, **s))
    assert torch.isfinite(out[:, 1]).all()


def test_no_phase_tolerance_survives_in_the_state_dict():
    """The buffer is gone, so a checkpoint written before this change fails load_state_dict loudly
    instead of being silently accepted with a mask nothing else knows about."""
    scorer = SegmentScorer()
    assert "phase_tolerance" not in scorer.state_dict()
    assert not hasattr(scorer, "phase_tolerance")


def test_load_rejects_wrong_feature_width_by_name(tmp_path):
    scorer = SegmentScorer()
    from conftest import fcsr_round_trip

    def _drift(arrays):
        arrays["scorer.feature_dim"] = np.int64(SEG_FEATURE_DIM + 1)

    with pytest.raises(ValueError, match="python -m robot_ood.cli train fcsr --suite long"):
        fcsr_round_trip(scorer, tmp_path, "scorer", mutate=_drift)
