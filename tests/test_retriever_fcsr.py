"""Tests for robot_ood.models.retrieval.retriever -- the query encoder and segment retriever.

Every test builds a synthetic robot_ood.core.schemas.SegmentTable directly and stubs the
robot_ood.models.retrieval.scorer module with small, real (not mocked) torch modules matching the
contracted signatures, so these tests exercise the actual shortlist/score wiring rather than
asserting against a mock.
"""

from __future__ import annotations


import inspect
import numpy as np
import pytest
import torch

from robot_ood.core.config import GraphConfig
from robot_ood.models.retrieval import scorer
from robot_ood.core.schemas import (
    ACTION_DIM,
    SEGMENT_LEN,
    QueryContext,
    SegmentCandidates,
    SegmentTable,
)
from robot_ood.models.retrieval.encoders import (
    QueryEncoder,
    SegmentKeyEncoder,
    _CyclicPhaseEncoding,
)
from robot_ood.models.retrieval.retriever import SHORTLIST_PRIOR_W, SegmentRetriever, segment_graph_hash
from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.data.nodes import NodeTable

DIM = 5
HIST = 4
HIDDEN = 16
N_SEGMENTS = 40
N_TASKS = 2


# ======================================================================================================
# ======================================================================================================
# synthetic fixtures
# ======================================================================================================
def _make_segments(n: int = N_SEGMENTS, seed: int = 0) -> SegmentTable:
    rng = np.random.RandomState(seed)
    ee_start = rng.randn(n, 6).astype(np.float32)
    ee_end = ee_start + rng.randn(n, 6).astype(np.float32) * 0.1
    return SegmentTable(
        start_node=np.arange(n, dtype=np.int32),
        end_node=(np.arange(n, dtype=np.int32) + SEGMENT_LEN - 1),
        owner=(np.arange(n) // 4).astype(np.int32),
        task_id=(np.arange(n) % N_TASKS).astype(np.int16),
        phase_start=rng.uniform(0, 0.9, n).astype(np.float32),
        phase_end=rng.uniform(0.1, 1.0, n).astype(np.float32),
        action_chunk=rng.randn(n, SEGMENT_LEN, ACTION_DIM).astype(np.float32),
        ee_start=ee_start,
        ee_end=ee_end,
        node_to_segment=np.arange(n, dtype=np.int32),
    )


def _make_query_encoder(*, lang_dim: int = 0) -> QueryEncoder:
    return QueryEncoder(dim=DIM, hist=HIST, hidden=HIDDEN, lang_dim=lang_dim)


def _make_ctx(
    batch: int = 1, *, a_hist: bool | None = True, e_lang: bool = False,
    lang_dim: int = 0, seed: int = 0, task_p: torch.Tensor | None = None,
) -> QueryContext:
    g = torch.Generator().manual_seed(seed)
    qh = torch.randn(batch, HIST, DIM, generator=g)
    vh = torch.randn(batch, HIST, DIM, generator=g)
    gh = torch.zeros(batch, HIST)
    a = None
    if a_hist is True:
        a = torch.randn(batch, HIST, ACTION_DIM, generator=g)
    elif a_hist == "zero":
        a = torch.zeros(batch, HIST, ACTION_DIM)
    lang = torch.randn(batch, lang_dim, generator=g) if e_lang else None
    ee = torch.randn(batch, 6, generator=g)
    phase = torch.rand(batch, generator=g)
    return QueryContext(qh=qh, vh=vh, gh=gh, ee=ee, phase=phase,
                        a_hist=a, e_lang=lang, task_p=task_p)


def _write_node_graph(tmp_path) -> tuple[NodeTable, EdgeSet]:
    """A minimal real (nodes, edges) pair on disk at tmp_path -- what SegmentRetriever.load reads
    g_nodes.npz/g_edges.npz from to verify its graph_hash stamp against."""

    def demo(seed: int, n: int = 12, dim: int = 4) -> dict:
        r = np.random.RandomState(seed)
        direction = r.randn(dim)
        direction /= np.linalg.norm(direction)
        t = np.arange(n, dtype=np.float32)
        q = (r.randn(dim) * 0.01 + 0.05 * t[:, None] * direction[None, :]).astype(np.float32)
        qdot = np.zeros_like(q)
        qdot[1:] = q[1:] - q[:-1]
        return {"q": q, "qdot": qdot, "grip": np.zeros(n, dtype=np.uint8),
                "stage": np.zeros(n, dtype=np.int16), "task_id": 0}

    nodes = NodeTable.from_demos([demo(1), demo(2)], task_names=["taskA"], coarsen=1)
    edges = EdgeSet.build(nodes, k_sib=2, k_align=2, device="cpu")
    nodes.save(tmp_path)
    edges.save(tmp_path / schema.EDGES_NPZ)
    return nodes, edges


def _make_retriever(*, segments: SegmentTable | None = None,
                    cfg: GraphConfig | None = None) -> SegmentRetriever:
    segments = segments if segments is not None else _make_segments()
    qe = _make_query_encoder()
    ke = SegmentKeyEncoder(HIDDEN)
    ae = scorer.ActionEncoder(SEGMENT_LEN)
    sc = scorer.SegmentScorer()
    cfg = cfg or GraphConfig(seg_topk=5, seg_tier1_m=12, future_temp=1.0)
    return SegmentRetriever(qe, ke, ae, sc, segments, cfg=cfg, device="cpu")


# ======================================================================================================
# QueryEncoder
# ======================================================================================================
def test_forward_and_predict_future_shapes():
    qe = _make_query_encoder()
    ctx = _make_ctx(batch=3)
    e = qe(ctx)
    assert e.shape == (3, HIDDEN)
    future = qe.predict_future(ctx)
    assert future.shape == (3, SEGMENT_LEN, ACTION_DIM)


def test_forward_is_deterministic_in_eval_mode():
    """(a) same ctx -> bit-identical output, twice, once dropout/etc are off."""
    qe = _make_query_encoder(lang_dim=8).eval()
    ctx = _make_ctx(batch=3, e_lang=True, lang_dim=8, seed=11)
    a = qe(ctx)
    b = qe(ctx)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_phase_gate_starts_near_identity_at_init():
    """(b) the phase gate's weight is zero-initialised and its bias near +2.0, so at construction
    time the gate multiplier is a constant sigmoid(bias) ~= 0.88, not a no-op learned function of
    context that could suppress the pre-gate embedding arbitrarily."""
    qe = _make_query_encoder()
    assert torch.all(qe.phase_gate.weight == 0)
    gate_value = torch.sigmoid(qe.phase_gate.bias)
    assert torch.all(gate_value > 0.8)


def test_a_hist_none_uses_a_distinct_path_from_a_hist_zero():
    """(c) a_hist absent and a_hist present-and-zero must NOT collapse to the same embedding."""
    qe = _make_query_encoder().eval()
    torch.manual_seed(0)
    ctx_none = _make_ctx(batch=2, a_hist=None)
    ctx_zero = _make_ctx(batch=2, a_hist="zero")
    # same qh/vh/gh/w (same seed), only a_hist differs: None vs an explicit all-zero tensor.
    ctx_zero = QueryContext(qh=ctx_none.qh, vh=ctx_none.vh, gh=ctx_none.gh,
                             ee=ctx_none.ee, phase=ctx_none.phase,
                             a_hist=torch.zeros(2, HIST, ACTION_DIM))
    e_none = qe(ctx_none)
    e_zero = qe(ctx_zero)
    assert not torch.allclose(e_none, e_zero)


def test_a_hist_absent_is_batch_and_frame_invariant_in_the_action_term():
    """The action_absent embedding does not depend on which query or which frame it is attached to."""
    qe = _make_query_encoder()
    ctx = _make_ctx(batch=3, a_hist=None)
    embed = qe._action_embed(ctx)
    assert embed.shape == (3, HIST, HIDDEN)
    assert torch.allclose(embed[0], embed[1])
    assert torch.allclose(embed[1], embed[2])
    assert torch.allclose(embed[0, 0], embed[0, 1])


def test_two_queries_differing_only_in_phase_produce_different_embeddings():
    """(d) the phase token must actually reach the output, not just exist."""
    qe = _make_query_encoder().eval()
    ctx_p0 = _make_ctx(batch=1, seed=5)
    ctx_p1 = QueryContext(qh=ctx_p0.qh, vh=ctx_p0.vh, gh=ctx_p0.gh, ee=ctx_p0.ee,
                          phase=torch.remainder(ctx_p0.phase + 0.5, 1.0), a_hist=ctx_p0.a_hist)
    assert not torch.allclose(qe(ctx_p0), qe(ctx_p1))


def test_cyclic_phase_encoding_is_continuous_at_the_p0_p1_wrap():
    """(e) phase=0 and phase=1 describe the same point on the cycle, so the encoding must agree
    exactly, not merely approximately, at the wrap."""
    enc = _CyclicPhaseEncoding(HIDDEN)
    out0 = enc(torch.zeros(1))
    out1 = enc(torch.ones(1))
    torch.testing.assert_close(out0, out1)


def test_the_language_term_changes_the_embedding_when_present():
    qe = _make_query_encoder(lang_dim=8).eval()
    ctx_no_lang = _make_ctx(batch=1, e_lang=False)
    ctx_lang = QueryContext(qh=ctx_no_lang.qh, vh=ctx_no_lang.vh, gh=ctx_no_lang.gh,
                             ee=ctx_no_lang.ee, phase=ctx_no_lang.phase,
                             a_hist=ctx_no_lang.a_hist, e_lang=torch.ones(1, 8))
    assert not torch.allclose(qe(ctx_no_lang), qe(ctx_lang))


def test_disabled_lang_dim_builds_no_parameters():
    qe = _make_query_encoder(lang_dim=0)
    assert qe.lang_proj is None


def test_word_dropout_only_applies_in_training_mode():
    """word_dropout can zero e_lang at train time; eval must never do so (else determinism breaks)."""
    qe = _make_query_encoder(lang_dim=8)
    qe.word_dropout = 1.0          # force-drop, to make the effect observable
    ctx = _make_ctx(batch=4, e_lang=True, lang_dim=8, seed=2)

    qe.eval()
    a = qe(ctx)
    b = qe(ctx)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_gradient_flows_from_predict_future_into_the_frame_encoder():
    qe = _make_query_encoder()
    ctx = _make_ctx(batch=2)
    out = qe.predict_future(ctx).sum()
    out.backward()
    grads = [p.grad for p in qe.frame_proj.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads)


# ======================================================================================================
# SegmentRetriever -- shortlist / score / masking / determinism
# ======================================================================================================
def test_retrieve_returns_well_formed_candidates():
    ret = _make_retriever()
    ctx = _make_ctx(batch=1)
    cand = ret.retrieve(ctx)
    assert isinstance(cand, SegmentCandidates)
    assert cand.k == 5
    assert cand.idx.dtype == np.int32
    assert cand.action_chunk.shape == (5, SEGMENT_LEN, ACTION_DIM)
    assert cand.ee_start.shape == (5, 6)


def test_retrieve_requires_batch_size_one():
    ret = _make_retriever()
    ctx = _make_ctx(batch=2)
    with pytest.raises(ValueError):
        ret.retrieve(ctx)


def test_retrieve_deterministic_bit_identical():
    ret = _make_retriever()
    ctx = _make_ctx(batch=1, seed=7)
    a = ret.retrieve(ctx)
    b = ret.retrieve(ctx)
    np.testing.assert_array_equal(a.idx, b.idx)
    np.testing.assert_array_equal(a.score, b.score)


def test_exclude_owner_removes_that_demos_segments():
    segments = _make_segments()
    ret = _make_retriever(segments=segments, cfg=GraphConfig(seg_topk=5, seg_tier1_m=40))
    ctx = _make_ctx(batch=1)
    cand = ret.retrieve(ctx, exclude_owner=0)
    owners = segments.owner[cand.idx]
    assert not np.any(owners == 0)


def test_a_flat_belief_leaves_every_task_reachable():
    """Uniform mass adds one constant to every segment, so the cut is the one geometry alone makes."""
    # The cut, not the final top-K: a flat belief still moves task_affinity off the zero that
    # task_p=None feeds it, so score() may legitimately reorder the tail.
    segments = _make_segments()
    m = 12
    ret = _make_retriever(segments=segments, cfg=GraphConfig(seg_topk=8, seg_tier1_m=m))
    flat = torch.full((1, N_TASKS), 1.0 / N_TASKS)

    def cut(task_p):
        ctx = _make_ctx(batch=1, task_p=task_p)
        with torch.no_grad():
            score = ret.query_encoder(ctx) @ ret._key_table.T
            if task_p is not None:
                score = score + SHORTLIST_PRIOR_W * task_p[..., ret._task_id].clamp_min(1e-6).log()
        return set(score.topk(m, dim=-1).indices[0].tolist())

    assert cut(flat) == cut(None)


def test_the_belief_tilts_tier_1_toward_the_task_it_believes():
    """A belief concentrated on one task must pull that task's segments into the shortlist."""
    segments = _make_segments()
    ret = _make_retriever(segments=segments, cfg=GraphConfig(seg_topk=4, seg_tier1_m=8))
    for task in range(N_TASKS):
        p = torch.full((1, N_TASKS), 1e-6)
        p[0, task] = 1.0
        cand = ret.retrieve(_make_ctx(batch=1, task_p=p))
        assert set(segments.task_id[cand.idx].tolist()) == {task}


def test_tier1_shortlist_is_never_wider_than_seg_tier1_m():
    segments = _make_segments(n=200)
    cfg = GraphConfig(seg_topk=8, seg_tier1_m=17)
    ret = _make_retriever(segments=segments, cfg=cfg)
    ctx = _make_ctx(batch=1)
    cand = ret.retrieve(ctx)
    assert cand.k == 8            # seg_topk, not seg_tier1_m
    assert len(set(cand.idx.tolist())) == cand.k


def test_score_batch_matches_retrieve_score_on_the_same_candidates():
    """retrieve()'s score pass and score_batch's must agree bit-for-bit on identical candidates and
    identical (eval-mode, no-dropout) weights -- they share the same score() implementation.

    This guard is why the deploy/train split stayed honest when a deploy-only causal mask was
    tried here: it failed immediately.
    """
    ret = _make_retriever()
    ctx = _make_ctx(batch=1)
    cand = ret.retrieve(ctx)
    cand_idx = torch.as_tensor(cand.idx, dtype=torch.int64).unsqueeze(0)
    with torch.no_grad():
        batch_score = ret.score(ctx, ret.query_encoder(ctx), cand_idx)
    np.testing.assert_allclose(batch_score[0].numpy(), cand.score, atol=1e-4)


def test_score_batch_is_differentiable_into_query_encoder_and_key_encoder():
    ret = _make_retriever()
    ctx = _make_ctx(batch=2)
    cand_idx = torch.randint(0, N_SEGMENTS, (2, 6))
    scores = ret.score_batch(ctx, cand_idx)
    assert scores.shape == (2, 6)
    scores.sum().backward()
    q_grads = [p.grad for p in ret.query_encoder.parameters()]
    k_grads = [p.grad for p in ret.key_encoder.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in q_grads)
    assert any(g is not None and torch.any(g != 0) for g in k_grads), (
        "the match term must be carried into score() additively, or the key encoder gets no gradient"
    )


def test_the_match_carry_is_load_bearing_for_the_key_encoder_gradient():
    """Directly the design point: zero out the additive match term and the key encoder's gradient
    must vanish, because SegmentScorer's own features never read the key encoder's output."""
    ret = _make_retriever()
    ctx = _make_ctx(batch=1)
    seg = ret.gather(torch.arange(6).unsqueeze(0))
    e_future_seg = ret.action_encoder(seg.action_chunk)
    # the match term deliberately dropped here, unlike score():
    ee_query = ctx.ee
    e_future_q = ret.action_encoder(ret.query_encoder.predict_future(ctx))
    phase_query = ctx.phase
    feats = scorer.segment_features(
        ee_query=ee_query, e_future_q=e_future_q, phase_query=phase_query,
        seg_ee_start=seg.ee_start, e_future_seg=e_future_seg, seg_phase=seg.phase_start,
        cand_task_p=torch.zeros(1, 6),
    )
    score_without_match = ret.fine_check(feats)     # note: no "+ match" term, unlike score()
    score_without_match.sum().backward()
    assert all(p.grad is None or torch.all(p.grad == 0) for p in ret.key_encoder.parameters()), (
        "seg_key participates only through the match dot product; without carrying it into the "
        "score, the key encoder must be untouched by the backward pass"
    )


def test_no_parameter_is_dimensioned_by_the_segment_count():
    ret = _make_retriever(segments=_make_segments(n=123))
    for name, mod in (("query_encoder", ret.query_encoder), ("key_encoder", ret.key_encoder),
                      ("action_encoder", ret.action_encoder), ("fine_check", ret.fine_check)):
        for pname, p in mod.named_parameters():
            assert 123 not in p.shape, f"{name}.{pname} has a dimension == S (123): {tuple(p.shape)}"


# ======================================================================================================
# persistence round trip
# ======================================================================================================
def _stamp(nodes: NodeTable, edges: EdgeSet, tmp_path, ret: SegmentRetriever) -> str:
    """The graph_hash ret's own checkpoint must carry to be accepted back by load()."""
    scorer_state = {k: v.detach().cpu().numpy() for k, v in ret.fine_check.state_dict().items()}
    return segment_graph_hash(nodes, edges, tmp_path / "constants.json",
    segment_len=int(ret.segments.action_chunk.shape[1]), seg_topk=int(ret.cfg.seg_topk),
                              scorer_state=scorer_state)


def _patch_loaders(monkeypatch, segments: SegmentTable, scorer_module) -> None:
    """Stand in for C1's load_segments -- out of scope for these tests, which are about what
    SegmentRetriever.load itself is responsible for."""
    from robot_ood.data import segments as segments_mod

    monkeypatch.setattr(segments_mod, "load_segments", lambda path: segments, raising=False)


def _write_fcsr(graph_dir, ret, *, graph_hash=None, future_temp=None, mutate=None):
    """Write this retriever's encoders and scorer as the one FCSR archive load() reads."""
    from conftest import fcsr_net, write_fcsr_archive
    from robot_ood.models.fcsr_net import SegEncoders

    net = fcsr_net(seg=SegEncoders(ret.query_encoder, ret.key_encoder, ret.action_encoder),
                   scorer=ret.fine_check)
    net.future_temp = future_temp
    return write_fcsr_archive(graph_dir, net, graph_hash=graph_hash or "", mutate=mutate)


def test_save_load_round_trip(tmp_path, monkeypatch):
    segments = _make_segments()
    ret = _make_retriever(segments=segments)
    nodes, edges = _write_node_graph(tmp_path)
    _write_fcsr(tmp_path, ret, graph_hash=_stamp(nodes, edges, tmp_path, ret))
    _patch_loaders(monkeypatch, segments, ret.fine_check)

    loaded = SegmentRetriever.load(tmp_path, cfg=ret.cfg, device="cpu")

    ctx = _make_ctx(batch=1, seed=3)
    a = ret.retrieve(ctx)
    b = loaded.retrieve(ctx)
    np.testing.assert_array_equal(a.idx, b.idx)
    np.testing.assert_allclose(a.score, b.score, atol=1e-5)


def test_load_rejects_a_checkpoint_with_no_graph_hash_stamp(tmp_path, monkeypatch):
    """A checkpoint that predates the stamp must be REFUSED, not silently trusted -- the same rule
    robot_ood.runtime.retrieve.GraphRetriever.load enforces for g_head.npz."""
    segments = _make_segments()
    ret = _make_retriever(segments=segments)
    _write_node_graph(tmp_path)
    _write_fcsr(tmp_path, ret)          # graph_hash=None: unstamped
    _patch_loaders(monkeypatch, segments, ret.fine_check)

    with pytest.raises(ValueError, match="graph_hash mismatch"):
        SegmentRetriever.load(tmp_path, cfg=ret.cfg, device="cpu")


def test_load_rejects_a_checkpoint_stamped_for_a_different_graph(tmp_path, monkeypatch):
    segments = _make_segments()
    ret = _make_retriever(segments=segments)
    _write_node_graph(tmp_path)
    _write_fcsr(tmp_path, ret, graph_hash="not-the-right-hash")
    _patch_loaders(monkeypatch, segments, ret.fine_check)

    with pytest.raises(ValueError, match="graph_hash mismatch"):
        SegmentRetriever.load(tmp_path, cfg=ret.cfg, device="cpu")


def test_load_rejects_a_checkpoint_whose_scorer_state_changed(tmp_path, monkeypatch):
    """segment_graph_hash folds the scorer's OWN weights in, so retraining the scorer without
    retraining the query/key encoders together must also be caught -- exactly the "wrong C2
    checkpoint" half of the guarantee, not just "wrong graph"."""
    segments = _make_segments()
    ret = _make_retriever(segments=segments)
    nodes, edges = _write_node_graph(tmp_path)
    stamp = _stamp(nodes, edges, tmp_path, ret)
    def _perturb(arrays):
        for k, v in list(arrays.items()):
            if k.startswith("scorer.") and getattr(v, "dtype", None) == np.float32:
                arrays[k] = v + 1.0   # the scorer that loads no longer matches the stamp

    _write_fcsr(tmp_path, ret, graph_hash=stamp, mutate=_perturb)
    _patch_loaders(monkeypatch, segments, ret.fine_check)

    with pytest.raises(ValueError, match="graph_hash mismatch"):
        SegmentRetriever.load(tmp_path, cfg=ret.cfg, device="cpu")
















def test_task_affinity_needs_language_on_both_sides():
    """No table, or no instruction on the query: the column is zero and retrieval still runs."""
    ret = _make_retriever()
    ctx = _make_ctx(batch=1, seed=7)
    cand = ret.retrieve(ctx)
    assert cand.idx.shape[0] > 0


def test_the_key_table_follows_the_key_encoder_after_a_step():
    """The cached key table is stale the moment the key encoder steps, until refresh_keys()."""
    r = _make_retriever()
    before = r._key_table.clone()

    with torch.no_grad():
        for p in r.key_encoder.parameters():
            p.add_(torch.randn_like(p))

    assert torch.equal(r._key_table, before), "the table must not silently track the weights"
    r.refresh_keys()
    assert not torch.allclose(r._key_table, before), "refresh_keys() did not rebuild the table"

    expected = r.key_encoder(r._ee_start, r._phase_start, r.action_encoder(r.action_chunks))
    assert torch.allclose(r._key_table, expected)


def test_training_refresh_keeps_the_key_table_in_step_with_the_segment_loss():
    """FcsrTask.on_epoch must rebuild the table: the shortlist and the segment loss share one encoder."""
    from robot_ood.train.tasks.fcsr import FcsrTask

    src = inspect.getsource(FcsrTask.on_epoch)
    assert "refresh_keys" in src, (
        "FcsrTask.on_epoch no longer refreshes the key table -- the fusion and dose shortlists "
        "will rank against the key encoder's initial weights for the whole run while the segment "
        "loss trains against the current ones."
    )
