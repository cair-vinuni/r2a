"""Future-conditioned query encoder and segment retrieval.

Query encoding, key caching, and two-stage retrieval: a shortlist cut to M, then a rescore to K.
See docs/method-runtime.md."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from robot_ood.core import npz
from robot_ood.data.store import ArtifactStore
from robot_ood.models.retrieval.encoders import QueryEncoder, SegmentKeyEncoder, observed
from robot_ood.core.schemas import QueryContext, SegmentCandidates

if TYPE_CHECKING:
    from robot_ood.core.config import GraphConfig
    from robot_ood.core.schemas import SegmentTable
    from robot_ood.data.edges import EdgeSet
    from robot_ood.data.nodes import NodeTable

# Weight of the belief's task log-prior on the shortlist cut. It never enters the final score.
SHORTLIST_PRIOR_W = 1.0

__all__ = ["BatchedSegmentCandidates", "SegmentFields", "SegmentRetriever", "segment_graph_hash"]


@dataclass(frozen=True, slots=True)
class BatchedSegmentCandidates:
    """retrieve_batch()'s output: the shortlist ids and scores training reads.

    Attributes:
        idx: [B, K] int64 segment ids.
        score: [B, K] float32 raw segment scores.
    """

    idx: Tensor
    score: Tensor


class SegmentFields(NamedTuple):
    """The per-candidate segment columns one shortlist needs.

    Attributes:
        ee_start: [B, K, 6] each candidate's start pose.
        phase_start: [B, K] its phase.
        task_id: [B, K] its task.
        action_chunk: [B, K, SEGMENT_LEN, ACTION_DIM] its recorded actions.
    """

    ee_start: Tensor
    phase_start: Tensor
    task_id: Tensor
    action_chunk: Tensor


def segment_graph_hash(
    nodes: "NodeTable", edges: "EdgeSet", constants_path: "Path | str", *,
    segment_len: int, seg_topk: int, scorer_state: dict,
) -> str:
    """Extend graph_hash_v2 to include segment_len and seg_topk in the stamp.

    Args:
        nodes: The node table.
        edges: Its edge set.
        constants_path: Path to constants.json beside the graph.
        segment_len: The segment memory's own chunk length, off SegmentTable.action_chunk.
        seg_topk: K, the structured candidate count fusion sees.
        scorer_state: The SegmentScorer's state_dict, as {name: array}.

    Returns:
        The hex digest.
    """
    from robot_ood.data.geometry import graph_hash

    c2_state = {
        **scorer_state,
        "seg_segment_len": np.asarray(int(segment_len)),
        "seg_seg_topk": np.asarray(int(seg_topk)),
    }
    return graph_hash(nodes, edges, constants=constants_path, c2_state=c2_state)

class SegmentRetriever:
    """Deploy-side: future-conditioned query -> K structured segment candidates.

    Args:
        query_encoder: The future-conditioned query encoder.
        key_encoder: The shortlist segment key encoder.
        action_encoder: The ActionEncoder, shared between the query's predicted future and every
            segment's own action chunk so both land in the same embedding space.
        fine_check: The SegmentScorer.
        segments: The segment memory.
        cfg: seg_topk/seg_tier1_m.
        device: Torch device; None resolves to CPU.
    """

    def __init__(
        self,
        query_encoder: QueryEncoder,
        key_encoder: SegmentKeyEncoder,
        action_encoder: nn.Module,
        fine_check: nn.Module,
        segments: "SegmentTable",
        *,
        cfg: "GraphConfig",
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.cfg = cfg
        self.query_encoder = query_encoder.to(self.device).eval()
        self.key_encoder = key_encoder.to(self.device).eval()
        self.action_encoder = action_encoder.to(self.device).eval()
        self.fine_check = fine_check.to(self.device).eval()
        self.segments = segments

        self._ee_start = torch.as_tensor(np.asarray(segments.ee_start, np.float32), device=self.device)
        self._phase_start = torch.as_tensor(np.asarray(segments.phase_start, np.float32), device=self.device)
        self._task_id = torch.as_tensor(np.asarray(segments.task_id, np.int64), device=self.device)
        self._owner = torch.as_tensor(np.asarray(segments.owner, np.int64), device=self.device)
        self.action_chunks = torch.as_tensor(np.asarray(segments.action_chunk, np.float32), device=self.device)

        self.refresh_keys()

    @torch.no_grad()
    def refresh_keys(self) -> None:
        """Recompute the shortlist key table, one O(S) pass, from the current encoder weights."""
        # Only _shortlist reads this table. The training loss rebuilds its own keys from the live
        # weights every step, so a stale table here can never leak into a gradient.
        e_future_all = self.action_encoder(self.action_chunks)
        self._key_table = self.key_encoder(self._ee_start, self._phase_start, e_future_all)

    # ---- construction ------------------------------------------------------------------------------
    @classmethod
    def load(
        cls, graph_dir: str | os.PathLike, *, cfg: "GraphConfig", device: str | torch.device | None = None,
    ) -> "SegmentRetriever":
        """Load a trained retriever from <graph_dir>/{g_segments.npz, g_fcsr.npz}.

        Args:
            graph_dir: The artifact directory the graph, segment and FCSR builds write into.
            cfg: seg_topk/seg_tier1_m/future_temp.
            device: Torch device; None resolves to CPU.

        Returns:
            The loaded retriever.

        Raises:
            ValueError: The stamped graph_hash (or its absence) does not match this graph.
        """
        graph_dir = Path(graph_dir)
        from robot_ood.data.geometry import CONSTANTS_JSON
        from robot_ood.models.fcsr_net import FcsrNet

        store = ArtifactStore(graph_dir)
        segments = store.segments()
        resolved = torch.device(device) if device is not None else torch.device("cpu")

        _, archive = npz.read(graph_dir, FcsrNet.NPZ, FcsrNet.MISSING)
        net = FcsrNet.from_arrays(archive, graph_dir, resolved)
        fine_check = net.scorer
        got = str(archive["graph_hash"]) if "graph_hash" in archive else None

        nodes = store.nodes()
        edges = store.edges()
        scorer_state = {k: v.detach().cpu().numpy() for k, v in fine_check.state_dict().items()}
        # A checkpoint that embeds its own calibration is verified against those bytes; one
        # without falls back to a constants.json beside the graph.
        from robot_ood.data import schema as graph_schema
        constants_src = (str(archive[graph_schema.CONSTANTS_KEY])
                         if graph_schema.CONSTANTS_KEY in archive else graph_dir / CONSTANTS_JSON)
        want = segment_graph_hash(
            nodes, edges, constants_src,
            segment_len=int(segments.action_chunk.shape[1]), seg_topk=int(cfg.seg_topk),
            scorer_state=scorer_state,
        )
        if got != want:
            raise ValueError(
                f"SegmentRetriever.load({graph_dir}): graph_hash mismatch -- "
                f"checkpoint={got if got is not None else '<none: predates graph_hash stamping>'!r} "
                f"graph={want!r}. This retriever was not trained on THIS (graph, scorer, "
                "segment_len, seg_topk) combination: segment ids are positional, so a foreign "
                "checkpoint retrieves confidently WRONG continuations with no shape error anywhere "
                "downstream. Retrain with `python -m robot_ood.cli train fcsr --suite <suite>`, "
                "or point graph_dir at the artifacts this retriever was "
                "actually trained on."
            )

        return cls(net.seg.query, net.seg.key, net.seg.action, fine_check, segments,
                   cfg=cfg, device=resolved)

    # ---- scoring -------------------------------------------------------------------------------------
    def gather(self, cand_idx: Tensor) -> "SegmentFields":
        """Raw per-candidate segment fields, gathered fresh -- O(K), never O(S)."""
        return SegmentFields(self._ee_start[cand_idx], self._phase_start[cand_idx],
                             self._task_id[cand_idx], self.action_chunks[cand_idx])

    def candidate_task_p(self, ctx: QueryContext, cand_idx: Tensor) -> Tensor:
        """[B, K] the belief's mass on each candidate's task, zeros before the locator's first check."""
        if ctx.task_p is None:
            return torch.zeros(cand_idx.shape, dtype=torch.float32, device=self.device)
        return ctx.task_p.gather(-1, self._task_id[cand_idx])

    def score(self, ctx: QueryContext, e_q: Tensor, cand_idx: Tensor) -> Tensor:
        """The final score over cand_idx: the embedding match plus the scorer's fine check.

        Args:
            ctx: The query context; its task_p drives the task_affinity feature.
            e_q: [B, hidden] this query's fused embedding.
            cand_idx: [B, K] candidate segment ids.

        Returns:
            [B, K] scores.
        """
        from robot_ood.models.retrieval import scorer

        seg = self.gather(cand_idx)
        e_future_seg = self.action_encoder(seg.action_chunk)
        seg_key = self.key_encoder(seg.ee_start, seg.phase_start, e_future_seg)
        match = (e_q.unsqueeze(-2) * seg_key).sum(-1)

        ee_query = observed(ctx.ee, "ee")
        e_future_q = self.action_encoder(self.query_encoder.predict_future(ctx))
        phase_query = observed(ctx.phase, "phase")

        feats = scorer.segment_features(
            ee_query=ee_query, e_future_q=e_future_q, phase_query=phase_query,
            seg_ee_start=seg.ee_start, e_future_seg=e_future_seg, seg_phase=seg.phase_start,
            cand_task_p=self.candidate_task_p(ctx, cand_idx),
        )
        return self.fine_check(feats) + match

    def score_batch(self, ctx: QueryContext, cand_idx: Tensor) -> Tensor:
        """Differentiable final scores over a caller-chosen candidate set.

        Args:
            ctx: The query context.
            cand_idx: [B, K] int64 candidate segment ids.

        Returns:
            [B, K] scores.
        """
        e_q = self.query_encoder(ctx)
        return self.score(ctx, e_q, cand_idx)

    # ---- the one deploy call ---------------------------------------------------------------------
    def tier1(self, ctx: QueryContext, e_q: Tensor) -> Tensor:
        """The shortlist score over the whole segment table, before any caller-supplied prior.

        Args:
            ctx: The query context; its task_p tilts the score in log space.
            e_q: [B, hidden] the query embedding.

        Returns:
            [B, S] the key match plus the task tilt, the same units a seg_prior is added in.
        """
        shortlist_score = e_q @ self._key_table.T                     # [B, S] -- one matmul, never a loop
        if ctx.task_p is not None:
            # The belief tilts the cut in log space, so a segment of an unlikely task drops out
            # here instead of surviving to be rescored. It never reaches score().
            shortlist_score = (shortlist_score
                               + SHORTLIST_PRIOR_W * ctx.task_p[..., self._task_id].clamp_min(1e-6).log())
        return shortlist_score

    @torch.no_grad()
    def tier1_scores(self, ctx: QueryContext) -> np.ndarray:
        """One query's tier-1 score over every segment, as a two-armed robot's other lane reads it.

        Args:
            ctx: The query context, B == 1.

        Returns:
            [S] float32.
        """
        e_q = self.query_encoder(ctx)
        return self.tier1(ctx, e_q)[0].detach().cpu().numpy().astype(np.float32)

    def _shortlist(
        self, ctx: QueryContext, e_q: Tensor, *, exclude_owner: "int | Tensor | None",
    ) -> tuple[Tensor, Tensor]:
        """Cut the table to M on the shortlist score, rescore, keep the top K.

        Args:
            ctx: The query context.
            e_q: [B, hidden] this call's own query embedding.
            exclude_owner: Demo id to exclude from the shortlist: one id for every row, a [B] tensor
                of one id per row (leave-one-demo-out at training), or None.

        Returns:
            (final_idx [B, k] int64, top_score [B, k]).
        """
        shortlist_score = self.tier1(ctx, e_q)
        if ctx.seg_prior is not None:
            shortlist_score = (shortlist_score
                               + SHORTLIST_PRIOR_W * ctx.seg_prior.to(shortlist_score.device, shortlist_score.dtype))
        if exclude_owner is not None:
            excluded = torch.as_tensor(exclude_owner, device=self._owner.device).reshape(-1, 1)
            shortlist_score = shortlist_score.masked_fill(self._owner[None, :] == excluded, float("-inf"))
        # Clamp m: with exclude_owner set, masked -inf entries still rank if m == S, so use min avail.
        avail = int(torch.isfinite(shortlist_score).sum(dim=-1).min().item())
        m = min(int(self.cfg.seg_tier1_m), avail)
        _, cand_idx = shortlist_score.topk(m, dim=-1)

        final_score = self.score(ctx, e_q, cand_idx)
        k = min(int(self.cfg.seg_topk), m)
        top_score, order = final_score.topk(k, dim=-1)
        final_idx = cand_idx.gather(-1, order)
        return final_idx, top_score

    def retrieve(
        self, ctx: QueryContext, *, exclude_owner: int | None = None,
    ) -> SegmentCandidates:
        """One query -> K structured candidates, ordered by descending score.

        Args:
            ctx: The query context, B == 1.
            exclude_owner: Demo id to exclude (leave-one-demo-out at training) -- the one HARD mask,
                applied at the cut so an excluded demo's segments never enter the shortlist.

        Returns:
            The K candidates.

        Raises:
            ValueError: ctx does not describe exactly one query.
        """
        with torch.no_grad():
            e_q = self.query_encoder(ctx)
            if e_q.shape[0] != 1:
                raise ValueError(f"SegmentRetriever.retrieve expects one query (B=1), got batch {e_q.shape[0]}")
            final_idx, top_score = self._shortlist(ctx, e_q, exclude_owner=exclude_owner)

        idx_np = final_idx[0].detach().cpu().numpy().astype(np.int32)
        return SegmentCandidates(
            idx=idx_np,
            score=top_score[0].detach().cpu().numpy().astype(np.float32),
            action_chunk=np.asarray(self.segments.action_chunk[idx_np], np.float32),
            phase=np.asarray(self.segments.phase_start[idx_np], np.float32),
            ee_start=np.asarray(self.segments.ee_start[idx_np], np.float32),
            ee_end=np.asarray(self.segments.ee_end[idx_np], np.float32),
            start_node=np.asarray(self.segments.start_node[idx_np], np.int32),
            end_node=np.asarray(self.segments.end_node[idx_np], np.int32),
        )

    def retrieve_batch(self, ctx: QueryContext, *, exclude_owner: "Tensor | None" = None) -> BatchedSegmentCandidates:
        """Batched retrieve() over a batch of queries.

        Args:
            ctx: The query context, any batch size B.
            exclude_owner: [B] demo id each row may not retrieve from (its own, at training), or None.

        Returns:
            The K candidates per row, K = min(cfg.seg_topk, cfg.seg_tier1_m, S).
        """
        with torch.no_grad():
            e_q = self.query_encoder(ctx)
            final_idx, top_score = self._shortlist(ctx, e_q, exclude_owner=exclude_owner)
        return BatchedSegmentCandidates(idx=final_idx, score=top_score)
