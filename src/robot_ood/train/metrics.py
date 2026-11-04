"""Held-out evaluation for the locator: retrieval quality, target-set hit rate and the phase readout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from robot_ood.core.config import GraphConfig
from robot_ood.core.device import resolve_device
from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.models.retrieval.gnn import GraphRetrieverNet
from robot_ood.data.metrics import NodeMetricColumns, QueryMetricColumns
from robot_ood.models.types import GraphEdges, MoveCost, NodeCloud, PipelineParams, QueryWindow
from robot_ood.data.nodes import NodeTable
from robot_ood.train.data import _node_feature_matrix
from robot_ood.train.target import TargetBuilder


def task_accuracy(belief: torch.Tensor, task_id: torch.Tensor, task: np.ndarray) -> float:
    """Share of rows whose belief puts most mass on the true task.

    Args:
        belief: [B, V] beliefs.
        task_id: [V] int64 per-node task id.
        task: [B] true task per row.

    Returns:
        The accuracy in [0, 1].
    """
    n_tasks = int(task_id.max().item()) + 1
    marginal = torch.zeros(belief.shape[0], n_tasks, dtype=belief.dtype, device=belief.device)
    marginal.index_add_(1, task_id, belief)
    pred = marginal.argmax(dim=-1).cpu().numpy()
    return float(np.mean(pred == np.asarray(task)))


def _retrieval_metrics(
    top1_pred: np.ndarray, ranks: np.ndarray, tgt: np.ndarray, nodes: NodeTable,
    move_dist: np.ndarray, hit: np.ndarray,
) -> dict:
    """top1/top1_target/top10/mrr, when-MAE (steps and phase), where-rmse, own_task_frac and movement distance.

    Args:
        top1_pred: [n] argmax node id per query.
        ranks: [n] 1-based rank of the true target.
        tgt: [n] target node ids.
        nodes: Node table the ids index into.
        move_dist: [n] ||q_top1 - q_now||.
        hit: [n] bool, top1_pred inside the row's target set.

    Returns:
        The metric mapping.
    """
    t_idx = np.asarray(nodes.t_idx)
    phase = np.asarray(nodes.phase)
    task_id = np.asarray(nodes.task_id)
    q_node = np.asarray(nodes.q, dtype=np.float64)

    out: dict[str, float] = {
        "top1": float(np.mean(top1_pred == tgt)),
        "top1_target": float(np.mean(hit)),
        "top10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
    }
    out["when_mae_steps"] = float(np.abs(t_idx[top1_pred] - t_idx[tgt]).mean())
    out["when_mae_phase"] = float(np.abs(phase[top1_pred] - phase[tgt]).mean())
    where_err = q_node[top1_pred] - q_node[tgt]
    out["where_rmse"] = float(np.sqrt((where_err**2).sum(axis=1).mean()))
    out["own_task_frac"] = float(np.mean(task_id[top1_pred] == task_id[tgt]))
    out["move_dist_mean"] = float(move_dist.mean())
    out["move_dist_median"] = float(np.median(move_dist))
    return out


def _phase_metrics(phase_argmax_pred: np.ndarray, phase_expect_pred: np.ndarray, nodes: NodeTable, tgt: np.ndarray) -> dict:
    """The argmax-bin phase estimate and the belief expectation against the true phase.

    Args:
        phase_argmax_pred: [n] argmax-bin phase estimate.
        phase_expect_pred: [n] expectation phase estimate.
        nodes: Node table the targets index into.
        tgt: [n] target node ids.

    Returns:
        Correlation and MAE for each estimator over this query set.
    """
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    phase = np.asarray(nodes.phase)
    tgt_phase = phase[tgt].astype(np.float64)
    return {
        "when_phase_argmax_corr": _corr(phase_argmax_pred, tgt_phase),
        "when_phase_argmax_mae": float(np.abs(phase_argmax_pred - tgt_phase).mean()),
        "when_phase_expect_corr": _corr(phase_expect_pred, tgt_phase),
        "when_phase_expect_mae": float(np.abs(phase_expect_pred - tgt_phase).mean()),
    }


@dataclass(frozen=True)
class _QueryOutcome:
    """One query's raw readouts.

    Attributes:
        top1: Argmax node id.
        rank: 1-based rank of the true target.
        hit: Whether top1 lies in the row's target set.
        phase_argmax: The argmax-bin phase estimate (bin centre).
        phase_expect: The belief expectation of phase.
        move_dist: ||q_top1 - q_now||.
    """

    top1: int
    rank: int
    hit: bool
    phase_argmax: float
    phase_expect: float
    move_dist: float


@dataclass(frozen=True)
class _Predictions:
    """The per-query outcomes as column arrays.

    Attributes:
        top1: [n] i64.
        rank: [n] i64.
        hit: [n] bool.
        phase_argmax: [n] f64.
        phase_expect: [n] f64.
        move_dist: [n] f64.
    """

    top1: np.ndarray
    rank: np.ndarray
    hit: np.ndarray
    phase_argmax: np.ndarray
    phase_expect: np.ndarray
    move_dist: np.ndarray

    @classmethod
    def stack(cls, rows: list[_QueryOutcome]) -> "_Predictions":
        def col(name: str, dtype: Any) -> np.ndarray:
            return np.asarray([getattr(r, name) for r in rows], dtype=dtype)

        return cls(
            top1=col("top1", np.int64), rank=col("rank", np.int64), hit=col("hit", bool),
            phase_argmax=col("phase_argmax", np.float64),
            phase_expect=col("phase_expect", np.float64),
            move_dist=col("move_dist", np.float64),
        )


class _QueryScorer:
    """Scores held-out queries against one fixed set of network weights."""

    def __init__(
        self, net: GraphRetrieverNet, nodes: NodeTable, edges: EdgeSet, cfg: GraphConfig, dev: Any,
    ) -> None:
        self.net, self.nodes, self.cfg, self.dev = net, nodes, cfg, dev
        self.targets = TargetBuilder(nodes, joint_tol=cfg.joint_tol)

        node_x_t = torch.as_tensor(_node_feature_matrix(nodes), device=dev)
        et = edges.torch(dev)
        self._log_idf = et["log_idf"]
        self._edges = GraphEdges(
            node_x=node_x_t, src=et["src"].long(), dst=et["dst"].long(), rel=et["rel"],
            log_idf=self._log_idf,
        )

        self._n_v = net.node_encode(node_x_t)
        self._q_nodes_n = net.normalize_q(node_x_t[:, : net.dim])
        self._cloud = NodeCloud(
            q_nodes_n=self._q_nodes_n, n_v=self._n_v,
            columns=NodeMetricColumns.from_node_table(nodes, dev),
        )
        self._seed_params = PipelineParams(topk=cfg.seed_topk, tier1_m=cfg.seed_tier1_m)

        n_bins = schema.NBINS_ALIGN
        self._n_bins = n_bins
        self._phase_t = torch.as_tensor(nodes.phase.astype(np.float32), device=dev)
        self._bin_idx_t = torch.clamp((self._phase_t * n_bins).long(), max=n_bins - 1)
        self._bin_centers = (
            torch.arange(n_bins, device=dev, dtype=self._phase_t.dtype) + 0.5
        ) / n_bins

    def seed_guards(self, queries: dict) -> dict[str, float]:
        """The two registered seed guards, measured on this eval set's first query window."""
        net, dev = self.net, self.dev

        def col(name: str) -> torch.Tensor:
            return torch.as_tensor(queries[name][0], dtype=torch.float32, device=dev)

        qh, vh, gh = col("q_hist"), col("qdot_hist"), col("grip_hist")
        enc = net.enc.encode(qh, vh, gh)
        report = net.seed_head.guard_report(
            net.normalize_q(qh), QueryMetricColumns(qdot=vh, grip=gh), enc.z, enc.pool_w,
            self._q_nodes_n, self._cloud.columns, self._n_v,
            tier1_m=self._seed_params.tier1_m,
        )
        return {"seed_entropy_gap": report.entropy_gap, "sigma_decades": report.decades}

    def score(self, queries: dict, tgt: np.ndarray) -> _Predictions:
        """Run the forward pass for every query and collect its readouts."""
        return _Predictions.stack([self._score_one(queries, i, int(tgt[i])) for i in range(len(tgt))])

    def _score_one(self, queries: dict, i: int, tgt_i: int) -> _QueryOutcome:
        net, cfg, dev = self.net, self.cfg, self.dev

        def col(name: str) -> torch.Tensor:
            return torch.as_tensor(queries[name][i], dtype=torch.float32, device=dev)

        qh, vh, gh = col("q_hist"), col("qdot_hist"), col("grip_hist")

        query = net.query(QueryWindow(qh, vh, gh))
        h0, seeded = net.seed(query, self._cloud, self._seed_params)
        h_l, reached = net.propagate(h0, self._n_v, query.e_q, self._edges, checkpoint=False,
                                     seeded=seeded)
        logits = net.logits(
            h_l, self._n_v, reached, self._log_idf, cfg.idf,
            MoveCost(q_nodes=self._q_nodes_n, q_now=query.qh_n[-1]),
            net.extra_features(query, self._edges),
        )

        order = torch.argsort(logits, descending=True).cpu().numpy()
        top1 = int(order[0])
        rank = int(np.nonzero(order == tgt_i)[0][0]) + 1
        move_dist = float(np.linalg.norm(
            self.nodes.q[top1].astype(np.float64) - qh[-1].detach().cpu().numpy().astype(np.float64)
        ))

        probs_full = torch.nan_to_num(F.softmax(logits, dim=0), nan=0.0)
        p_phase_bins = torch.zeros(
            self._n_bins, device=dev, dtype=probs_full.dtype
        ).index_add_(0, self._bin_idx_t, probs_full)
        phase_argmax = float(self._bin_centers[int(torch.argmax(p_phase_bins).item())].item())
        phase_expect = float((probs_full * self._phase_t).sum().item())

        return _QueryOutcome(
            top1=top1, rank=rank, hit=top1 in self.targets(tgt_i),
            phase_argmax=phase_argmax, phase_expect=phase_expect, move_dist=move_dist,
        )


def evaluate(
    net: GraphRetrieverNet, nodes: NodeTable, edges: EdgeSet, queries: dict, *,
    cfg: GraphConfig | None = None, device: Any = None,
) -> dict:
    """Evaluate net against a graph and query set.

    Args:
        net: The retriever to evaluate.
        nodes: Node table being retrieved over.
        edges: Its edge set.
        queries: The query mapping to evaluate on.
        cfg: Readout/seed configuration; None builds a default GraphConfig.
        device: Torch device; None resolves to the project default.

    Returns:
        seed_entropy_gap/sigma_decades, top1/top1_target/top10/mrr, when_mae_steps/when_mae_phase,
        where_rmse, own_task_frac, move_dist_{mean,median} and
        when_phase_{argmax,expect}_{corr,mae}.
    """
    cfg = cfg or GraphConfig()
    dev = resolve_device(device)
    net = net.to(dev).eval()

    tgt = np.asarray(queries["tgt_node"], dtype=np.int64)
    with torch.no_grad():
        scorer = _QueryScorer(net, nodes, edges, cfg, dev)
        preds = scorer.score(queries, tgt)

    out: dict[str, float] = dict(scorer.seed_guards(queries))
    out.update(_retrieval_metrics(preds.top1, preds.rank, tgt, nodes, preds.move_dist, preds.hit))
    out.update(_phase_metrics(preds.phase_argmax, preds.phase_expect, nodes, tgt))
    return out
