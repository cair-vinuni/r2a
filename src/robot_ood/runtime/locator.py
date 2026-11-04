"""The deploy side of the locator: one belief per episode, carried into the next check as memory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from robot_ood.core.config import GraphConfig
from robot_ood.data.binning import phase_hist
from robot_ood.models.retrieval.gnn import MEMORY_TOPK
from robot_ood.runtime.retrieve import GraphRetrieval, GraphRetriever

__all__ = ["BeliefTopK", "LocatorStep", "GraphLocator"]


def _finalize(belief: torch.Tensor, obs, retriever: GraphRetriever) -> GraphRetrieval:
    """Reduce a dense [V] belief to a GraphRetrieval."""
    cfg, nodes = retriever.cfg, retriever.nodes
    finite = torch.isfinite(obs.logits64)
    r_t = int(finite.sum().item())
    topm = min(cfg.topm, r_t)
    topv, topidx = torch.topk(belief, max(topm, 1), largest=True, sorted=True)
    p_node_t = torch.zeros_like(belief)
    p_node_t[topidx] = topv
    p_node_t = p_node_t / p_node_t.sum().clamp_min(1e-300)

    node = int(topidx[0].item())
    conf = float(p_node_t[node].item())
    q_seg, qdot_seg = nodes.segment(node, cfg.seg_k)
    p_node = p_node_t.detach().cpu().numpy()
    nll = float(-np.log(max(float(belief[node].item()), 1e-300)))
    p_phase = phase_hist(nodes.phase, p_node)
    nz = p_node > 0
    entropy = float(-(p_node[nz] * np.log(p_node[nz])).sum())
    return GraphRetrieval(
        p_node=p_node, p_phase=p_phase, node=node, owner=int(nodes.owner[node]),
        task_id=int(nodes.task_id[node]), stage=int(nodes.stage[node]),
        t_idx=int(nodes.t_idx[node]), q_star=q_seg[0].copy(), q_seg=q_seg, qdot_seg=qdot_seg,
        depth=float(nodes.phase[node]), conf=conf, entropy=entropy, nll=nll,
        off_manifold=obs.off_manifold,
        info={"reached": r_t, "topm": int(topm), "hist_len": obs.hist_len, "padded": obs.padded,
              "readout": "belief_argmax", "graph_hash": retriever.graph_hash},
    )


@dataclass(frozen=True, slots=True)
class BeliefTopK:
    """Host-side top-K view of a belief.

    Attributes:
        idx: [K] int64 node ids, descending by weight.
        weight: [K] float64 belief mass at each node.
        first_step: True when this is the first check of the episode.
    """

    idx: np.ndarray
    weight: np.ndarray
    first_step: bool


class LocatorStep:
    """One check's answer: phase_hat, the task distribution, and lazily computed belief views.

    Attributes:
        phase_hat: The belief's phase expectation, the one field deploy reads.
        task_prob: [n_tasks] float64 the belief's task marginal.
    """

    def __init__(self, phase_hat: float, task_prob: np.ndarray, belief: torch.Tensor, obs: Any,
                 retriever: GraphRetriever, first_step: bool) -> None:
        self.phase_hat = phase_hat
        self.task_prob = task_prob
        self._b = belief
        self._obs = obs
        self._retriever = retriever
        self._first_step = first_step

    @property
    def task_p(self) -> np.ndarray:
        """[n_tasks] float64 the belief's task marginal, the prior retrieval and fusion inherit."""
        return self.task_prob

    @property
    def gate(self) -> np.ndarray:
        """[N_GATE] float64 the trust head's view of this check: off-manifold distance, belief entropy."""
        return np.array([float(self._obs.off_manifold), self.belief_entropy], dtype=np.float64)

    @property
    def belief_entropy(self) -> float:
        nz = self._b > 0
        return float(-(self._b[nz] * self._b[nz].log()).sum().item())

    @property
    def topk(self) -> BeliefTopK:
        k = min(MEMORY_TOPK, int(self._b.shape[0]))
        top_v, top_idx = torch.topk(self._b, k, largest=True, sorted=True)
        return BeliefTopK(idx=top_idx.cpu().numpy().astype(np.int64),
                          weight=top_v.cpu().numpy().astype(np.float64),
                          first_step=self._first_step)

    @property
    def belief_view(self) -> GraphRetrieval:
        return _finalize(self._b, self._obs, self._retriever)


class GraphLocator:
    """The trained locator, driving one episode's belief."""

    def __init__(self, retriever: GraphRetriever, *, memory: bool = True) -> None:
        self.retriever = retriever
        self.memory = memory
        self.reset()

    @classmethod
    def load(cls, graph_dir: str | os.PathLike, *, cfg: GraphConfig | None = None,
             device: str | torch.device | None = None, memory: bool = True) -> "GraphLocator":
        return cls(GraphRetriever.load(graph_dir, cfg=cfg, device=device), memory=memory)

    def reset(self) -> None:
        """Start a new episode: no memory, no policy chunk."""
        self._memory: torch.Tensor | None = None
        self._policy_chunk: np.ndarray | None = None

    def record_chunk(self, a_policy: np.ndarray) -> None:
        """Keep the chunk the policy issued after this check; the next check reads it."""
        self._policy_chunk = np.asarray(a_policy, dtype=np.float32)[: self.retriever.net.chunk]

    @torch.no_grad()
    def step(self, q_hist: np.ndarray, *, g_hist=None, exclude_owner: int | None = None,
             task: str = "") -> LocatorStep:
        """One check: one observation under the carried memory, one belief.

        Args:
            q_hist: Query history, [T, D].
            g_hist: Optional gripper history.
            exclude_owner: Optional demo owner to mask out.
            task: This episode's instruction string, or "" for none.

        Returns:
            The LocatorStep.
        """
        retriever = self.retriever
        first_step = self._memory is None
        obs = retriever.observe(
            q_hist, g_hist=g_hist, exclude_owner=exclude_owner,
            instr=retriever.language_embedding(task), policy=self._policy_chunk,
            memory=self._memory, instr_p=retriever.language_posterior(task),
        )
        belief = torch.softmax(obs.logits64, dim=0)
        phase_hat = float((belief * retriever.phase64).sum().item())
        task_prob = torch.zeros(retriever.nodes.n_tasks, dtype=belief.dtype,
                                device=belief.device).index_add_(0, retriever.task_id_t, belief)
        k = min(MEMORY_TOPK, int(belief.shape[0]))
        top_v, top_idx = torch.topk(belief, k)
        memory = torch.zeros_like(belief).scatter_(0, top_idx, top_v)
        self._memory = (memory / memory.sum().clamp_min(1e-300)).to(torch.float32) if self.memory else None
        return LocatorStep(phase_hat=phase_hat, task_prob=task_prob.cpu().numpy(), belief=belief,
                           obs=obs, retriever=retriever, first_step=first_step)
