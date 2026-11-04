"""One retrieval for a two-armed robot: each arm's shortlist is cut on both arms' queries.

The two lanes of a bimanual tree share their episodes frame for frame, so a segment of one lane
pairs with the segments of the other lane that overlap it in time within the same demo. Each
check, every lane scores its whole segment table on its own query (tier 1); a lane's cut is then
made on its own score plus the best score its partner gave any paired segment, so both arms are
drawn to the same demo at the same moment. Nothing about the other arm is modelled: no motion
threshold, no lagged phase, only the partner lane's own retrieval score. A lane with no partner
(one arm) is unchanged.

Off by default; the harness turns it on under ROBOT_OOD_PAIR=1.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robot_ood.data import schema
from robot_ood.data.nodes import NodeTable
from robot_ood.data.segments import load_segments
from robot_ood.runtime.partner import node_spans

__all__ = ["PairedRetrieval", "pair_table", "segment_spans"]


def segment_spans(table: NodeTable, start_node: np.ndarray, end_node: np.ndarray) -> np.ndarray:
    """Each segment's [first frame, one past its last frame) within the lane's raw episodes.

    Args:
        table: The lane's node table.
        start_node: [S] the segment's first node.
        end_node: [S] the segment's last node.

    Returns:
        [S, 2] int64 frames.
    """
    nodes = node_spans(table.t_raw, table.owner, table.raw_ptr)
    s, e = np.asarray(start_node, dtype=np.int64), np.asarray(end_node, dtype=np.int64)
    return np.stack([nodes[s, 0], nodes[e, 1]], axis=1)


def pair_table(spans_a: np.ndarray, owner_a: np.ndarray, spans_b: np.ndarray, owner_b: np.ndarray,
               ) -> tuple[np.ndarray, np.ndarray]:
    """Which segments of lane b overlap each segment of lane a, as a CSR list.

    Args:
        spans_a: [S_a, 2] frames of lane a's segments; owner_a: [S_a] their demos.
        spans_b: [S_b, 2] and owner_b: [S_b], lane b's.

    Returns:
        (indptr [S_a + 1], indices) so that indices[indptr[i]:indptr[i + 1]] are the lane-b segments
        of the same demo whose frame span intersects segment i's.

    Raises:
        ValueError: A segment of lane a has no partner: the lanes do not share their episodes.
    """
    owner_a, owner_b = np.asarray(owner_a, dtype=np.int64), np.asarray(owner_b, dtype=np.int64)
    spans_b = np.asarray(spans_b, dtype=np.int64)
    order_b = np.argsort(owner_b, kind="stable")
    demos, first = np.unique(owner_b[order_b], return_index=True)
    stops = np.append(first[1:], len(order_b))
    by_demo = {int(d): order_b[s:e] for d, s, e in zip(demos, first, stops)}
    lists = []
    for i, (a0, a1) in enumerate(np.asarray(spans_a, dtype=np.int64)):
        demo = int(owner_a[i])
        cand = by_demo.get(demo, np.zeros(0, dtype=np.int64))
        b = spans_b[cand]
        hit = cand[(b[:, 0] < a1) & (a0 < b[:, 1])]
        if len(hit) == 0:
            raise ValueError(f"segment {i} of demo {demo} ({a0}:{a1}) overlaps no segment of the other lane")
        lists.append(np.asarray(hit, dtype=np.int64))
    indptr = np.zeros(len(lists) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(x) for x in lists])
    return indptr, np.concatenate(lists)


class PairedRetrieval:
    """Cut both lanes' shortlists on the sum of their tier-1 scores over paired segments."""

    def __init__(self, pairs: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Initialize from each arm's pair table.

        Args:
            pairs: arm -> (indptr, indices) of the partner segments of each of its own segments.
        """
        if len(pairs) != 2:
            raise ValueError(f"paired retrieval names exactly two arms, got {sorted(pairs)}")
        self.pairs = {arm: (np.asarray(p, dtype=np.int64), np.asarray(i, dtype=np.int64)) for arm, (p, i) in pairs.items()}
        self.partner = {a: b for a in pairs for b in pairs if b != a}

    @classmethod
    def from_lanes(cls, graph_dirs: Mapping[str, str | os.PathLike[str]]) -> "PairedRetrieval":
        """Build from the two lanes' node and segment tables.

        Args:
            graph_dirs: arm -> its artifact directory.

        Returns:
            The paired retrieval.

        Raises:
            ValueError: Not two lanes, or lanes whose episodes do not line up frame for frame.
        """
        if len(graph_dirs) != 2:
            raise ValueError(f"paired retrieval needs exactly two lanes, got {sorted(graph_dirs)}")
        spans, owner = {}, {}
        for arm, d in graph_dirs.items():
            table = NodeTable.load(Path(d) / schema.NODES_NPZ, strict=False)
            segs = load_segments(Path(d))
            spans[arm] = segment_spans(table, segs.start_node, segs.end_node)
            owner[arm] = np.asarray(segs.owner, dtype=np.int64)
        (a, ta), (b, tb) = spans.items()
        pairs = {a: pair_table(ta, owner[a], tb, owner[b]), b: pair_table(tb, owner[b], ta, owner[a])}
        return cls(pairs)

    def prior(self, arm: str, partner_scores: np.ndarray) -> np.ndarray:
        """This arm's shortlist prior: the best partner score over each of its segments' pairs.

        Args:
            arm: The arm the prior is for.
            partner_scores: [S_partner] the other lane's tier-1 scores this check.

        Returns:
            [S_arm] float32, in the partner's tier-1 units (the cut adds it at weight one).
        """
        indptr, indices = self.pairs[arm]
        vals = np.asarray(partner_scores, dtype=np.float32)[indices]
        return np.maximum.reduceat(vals, indptr[:-1]).astype(np.float32)

    def plan(self, chains: Mapping[str, Any], envs: Mapping[str, Any], states: Mapping[str, np.ndarray],
             step: int, *, shard: int, task: str) -> dict[str, Any]:
        """Both arms' blends for this step, each cut on both arms' queries.

        Args:
            chains: arm -> its DeployHooks.
            envs: arm -> the lane environment its chain reads joints from.
            states: arm -> concat(ee_pos, ee_ori, gripper_qpos) for this step.
            step: The policy step counter.
            shard: This client's shard id.
            task: The task description.

        Returns:
            arm -> its BlendPlan, or None off a check step.
        """
        checked = {arm: chains[arm].check(envs[arm], states[arm], step, shard=shard, task=task) for arm in chains}
        if all(c is None for c in checked.values()):
            return {arm: None for arm in chains}
        scores = {}
        for arm, c in checked.items():
            if c is not None:
                ctx = chains[arm].query_context(c, shard=shard, task=task)
                scores[arm] = chains[arm].sentinel.bundle.segment_retriever().tier1_scores(ctx)
        blends = {}
        for arm, c in checked.items():
            if c is None:
                blends[arm] = None
                continue
            other = self.partner[arm]
            prior = self.prior(arm, scores[other]) if other in scores else None
            blends[arm] = chains[arm].plan_checked(c, shard=shard, task=task, seg_prior=prior)
        return blends
