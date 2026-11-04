"""GraphRetrieverNet: the query-dependent GNN that answers a belief over the demonstration graph."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as _checkpoint

from robot_ood.core import npz
from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.data.geometry import CONSTANTS_JSON, graph_hash
from robot_ood.data.nodes import NodeTable
from robot_ood.data.metrics import C2Metric, LearnedSeed, NodeMetricColumns, QueryMetricColumns
from robot_ood.models.modules import QueryEncoder, RelationGate, sinusoidal_psi
from robot_ood.models.types import (
    EdgeIndex, GraphEdges, MoveCost, NodeCloud, PipelineParams, QueryWindow, SeedLanguage,
    SeedQuery,
)

__all__ = ["GraphRetrieverNet", "RetrieverConfig", "QueryEncoder", "RelationGate",
           "save_checkpoint", "sinusoidal_psi", "INSTR_DIM", "INSTR_LOW", "CHUNK", "ACTION_DIM",
           "MEMORY_SEED", "MEMORY_TOPK", "MEMORY_FLOOR"]

LOGSUMEXP_INIT = -1e30
LOG_FLOOR = 1e-30
NEG_INF = float("-inf")
C2_PREFIX = "seed."

INSTR_DIM = 896
INSTR_LOW = 32
CHUNK = 8
ACTION_DIM = 7
N_EXTRA = 5
MEMORY_SEED = 128
MEMORY_TOPK = 2048
MEMORY_FLOOR = -12.0


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    """Architecture hyper-parameters of one GraphRetrieverNet.

    Attributes:
        dim: Joint-space dimensionality.
        hidden: Width of every hidden embedding.
        layers: Number of shared-weight message-passing hops.
        n_rel: Size of the relation vocabulary.
        agg: Message aggregator, "sum" or "logsumexp".
        n_psi: Number of sin/cos octaves in the phase encoding.
        instr_dim: Instruction embedding width.
        instr_low: Width the instruction is projected to.
        chunk: Policy chunk length.
    """

    dim: int
    hidden: int = schema.HIDDEN
    layers: int = schema.LAYERS
    n_rel: int = schema.N_RELATIONS
    agg: str = schema.AGG
    n_psi: int = schema.PSI_FREQS
    instr_dim: int = INSTR_DIM
    instr_low: int = INSTR_LOW
    chunk: int = CHUNK

    @property
    def prefix_dim(self) -> int:
        return 2 * self.dim + 1

    @property
    def node_in_dim(self) -> int:
        return self.prefix_dim + 2 * self.n_psi

    @classmethod
    def from_npz(cls, arrays: Mapping[str, np.ndarray]) -> "RetrieverConfig":
        dim = (len(arrays["mean"]) - 1) // 2
        prefix_dim = 2 * dim + 1
        instr_low, instr_dim = (arrays["instr_proj.weight"].shape if "instr_proj.weight" in arrays
                                else (INSTR_LOW, INSTR_DIM))
        chunk = (int(arrays["policy_proj.weight"].shape[1]) // ACTION_DIM
                 if "policy_proj.weight" in arrays else CHUNK)
        return cls(
            dim=dim,
            hidden=int(arrays["rel.emb"].shape[1]),
            layers=int(arrays["layers"]),
            n_rel=int(arrays["rel.emb"].shape[0]),
            agg=str(arrays["agg"]),
            n_psi=(int(arrays["node.0.weight"].shape[1]) - prefix_dim) // 2,
            instr_dim=int(instr_dim), instr_low=int(instr_low), chunk=chunk,
        )


class GraphRetrieverNet(nn.Module):
    """Query-conditioned message passing over demonstration frames."""

    def __init__(
        self,
        dim: int,
        hidden: int = schema.HIDDEN,
        layers: int = schema.LAYERS,
        n_rel: int = schema.N_RELATIONS,
        agg: str = schema.AGG,
        n_psi: int = schema.PSI_FREQS,
        sigma_init: float = schema.KERNEL_BW,
        instr_dim: int = INSTR_DIM,
        instr_low: int = INSTR_LOW,
        chunk: int = CHUNK,
    ):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.n_rel = int(n_rel)
        self.agg = str(agg)
        self.n_psi = int(n_psi)
        self.instr_dim = int(instr_dim)
        self.instr_low = int(instr_low)
        self.chunk = int(chunk)
        self.prefix_dim = self.config.prefix_dim

        self.enc = QueryEncoder(self.dim, self.hidden)
        self.seed_head = LearnedSeed(self.hidden, C2Metric(), sigma_init=sigma_init)
        self.node = nn.Sequential(
            nn.Linear(self.config.node_in_dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.rel = RelationGate(self.hidden, self.n_rel)
        self.upd = nn.Sequential(
            nn.Linear(3 * self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.hidden),
        )
        self.out = nn.Sequential(
            nn.Linear(2 * self.hidden + 1 + N_EXTRA, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )
        self.instr_proj = nn.Linear(self.instr_dim, self.instr_low)
        self.instr_q = nn.Linear(self.instr_low, self.hidden)
        # Language on the nodes: each node's projected task instruction, folded into n_v so the
        # seed product e_q * n_v and every message carry an instruction match, not only the readout.
        self.node_lang = nn.Linear(self.instr_low, self.hidden)
        self.lang_graph = True
        self.policy_proj = nn.Linear(self.chunk * ACTION_DIM, self.hidden)
        self.mem = nn.Linear(1, self.hidden)
        nn.init.zeros_(self.node_lang.weight)
        nn.init.zeros_(self.node_lang.bias)
        nn.init.zeros_(self.instr_q.weight)
        nn.init.zeros_(self.instr_q.bias)
        nn.init.zeros_(self.policy_proj.weight)
        nn.init.zeros_(self.policy_proj.bias)
        nn.init.normal_(self.mem.weight, std=0.02)
        nn.init.zeros_(self.mem.bias)

        self.register_buffer("mean", torch.zeros(self.prefix_dim))
        self.register_buffer("std", torch.ones(self.prefix_dim))

    @property
    def config(self) -> RetrieverConfig:
        return RetrieverConfig(
            dim=self.dim, hidden=self.hidden, layers=self.layers, n_rel=self.n_rel, agg=self.agg,
            n_psi=self.n_psi, instr_dim=self.instr_dim, instr_low=self.instr_low, chunk=self.chunk,
        )

    @classmethod
    def load_npz(cls, path: str | os.PathLike) -> "GraphRetrieverNet":
        return cls.from_npz_dict(np.load(os.fspath(path)))

    @classmethod
    def from_npz_dict(cls, arrays: Mapping[str, Any]) -> "GraphRetrieverNet":
        cfg = RetrieverConfig.from_npz(arrays)
        net = cls(dim=cfg.dim, hidden=cfg.hidden, layers=cfg.layers, n_rel=cfg.n_rel,
                  agg=cfg.agg, n_psi=cfg.n_psi, instr_dim=cfg.instr_dim,
                  instr_low=cfg.instr_low, chunk=cfg.chunk)
        net.set_stats(arrays["mean"].astype(np.float32), arrays["std"].astype(np.float32))
        # A head written before language reached the graph keeps the graph language-free.
        net.lang_graph = bool(int(arrays["lang_graph"])) if "lang_graph" in arrays else False
        for name, mod in net._named_modules():
            state = {k.split(".", 1)[1]: torch.tensor(v)
                     for k, v in arrays.items() if k.startswith(name + ".")}
            if not state:
                continue
            # A head written before the readout grew its extra columns: pad them with zeros.
            if name == "out" and state["0.weight"].shape[1] < net.out[0].weight.shape[1]:
                pad = net.out[0].weight.shape[1] - state["0.weight"].shape[1]
                state["0.weight"] = torch.cat([state["0.weight"], torch.zeros(state["0.weight"].shape[0], pad)], 1)
            mod.load_state_dict(state)
        net.eval()
        return net

    def set_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean.copy_(torch.as_tensor(np.asarray(mean, np.float32)))
        self.std.copy_(torch.as_tensor(np.asarray(std, np.float32)))

    def _named_modules(self) -> list[tuple[str, nn.Module]]:
        return [("enc", self.enc), ("seed", self.seed_head), ("node", self.node),
                ("rel", self.rel), ("upd", self.upd), ("out", self.out),
                ("instr_proj", self.instr_proj), ("instr_q", self.instr_q),
                ("node_lang", self.node_lang),
                ("policy_proj", self.policy_proj), ("mem", self.mem)]

    def c2_state(self) -> dict[str, np.ndarray]:
        return {
            f"{C2_PREFIX}{k}": v.detach().cpu().numpy()
            for k, v in self.seed_head.state_dict().items()
        }

    def state_npz_dict(self) -> dict:
        out = {
            "mean": self.mean.detach().cpu().numpy(),
            "std": self.std.detach().cpu().numpy(),
            "layers": np.array(self.layers),
            "agg": np.array(self.agg),
            "lang_graph": np.array(int(self.lang_graph)),
        }
        for name, mod in self._named_modules():
            for k, v in mod.state_dict().items():
                out[f"{name}.{k}"] = v.detach().cpu().numpy()
        return out

    # ---- node features ----
    def normalize_q(self, q: Tensor) -> Tensor:
        return (q - self.mean[: self.dim]) / self.std[: self.dim]

    def normalize_nodes(self, x: Tensor) -> Tensor:
        prefix = (x[:, : self.prefix_dim] - self.mean) / self.std
        return torch.cat([prefix, x[:, self.prefix_dim :]], dim=-1)

    def node_encode(self, x: Tensor) -> Tensor:
        """[V, 2*dim+2] raw rows ([q, qdot, grip, t_frac]) -> n_v [V, hidden]."""
        xn = self.normalize_nodes(x)
        t_frac = xn[:, self.prefix_dim]
        psi = sinusoidal_psi(t_frac, self.n_psi)
        feat = torch.cat([xn[:, : self.prefix_dim], psi], dim=-1)
        return self.node(feat)

    def node_instr(self, e_task: Tensor, task_id: Tensor) -> Tensor:
        """[V, instr_low] each node's projected task instruction."""
        return F.normalize(self.instr_proj(e_task), dim=-1)[task_id]

    def node_repr(self, x: Tensor, e_task: Tensor | None, task_id: Tensor | None) -> Tensor:
        """n_v with the node's own task instruction folded in, when the graph carries language."""
        n_v = self.node_encode(x)
        if self.lang_graph and e_task is not None and task_id is not None:
            n_v = n_v + self.node_lang(self.node_instr(e_task, task_id))
        return n_v

    # ---- query ----
    def query(self, qw: QueryWindow, instr: Tensor | None = None,
              policy: Tensor | None = None, has_instr: Tensor | None = None,
              has_policy: Tensor | None = None, instr_p: Tensor | None = None) -> SeedQuery:
        """Encode one window plus its optional instruction and policy chunk.

        Args:
            qw: The query window; [H, ...] unbatched or [B, H, ...] batched.
            instr: [instr_dim] or [B, instr_dim] instruction embedding, or None.
            policy: [chunk, 7] or [B, chunk, 7] the chunk the policy just issued, or None.
            has_instr: [] or [B] float mask on instr; None means every row carries one.
            has_policy: [] or [B] float mask on policy; None means every row carries one.

            instr_p: [n_tasks] or [B, n_tasks] the instruction's task posterior, or None.

        Returns:
            The seed's query bundle; e_q carries both extra channels.
        """
        enc = self.enc.encode(qw.qh, qw.vh, qw.gh)
        e_q = enc.e_q
        u_q = None
        if instr is not None:
            u_q = F.normalize(self.instr_proj(instr), dim=-1)
            if has_instr is not None:
                u_q = u_q * has_instr[..., None]
            e_q = e_q + self.instr_q(u_q) * (1.0 if has_instr is None else has_instr[..., None])
        if policy is not None:
            if policy.shape[-2] != self.chunk:
                raise ValueError(
                    f"policy chunk has {policy.shape[-2]} rows but this net was built for "
                    f"{self.chunk}: the tree's plan rows and the head's policy channel must agree "
                    "(robot_ood.data.segments.graph_plan_rows)."
                )
            p = self.policy_proj(policy.reshape(*policy.shape[:-2], -1))
            e_q = e_q + p * (1.0 if has_policy is None else has_policy[..., None])
        return SeedQuery(
            e_q=e_q, z_h=enc.z, pool_w=enc.pool_w, qh_n=self.normalize_q(qw.qh),
            columns=QueryMetricColumns(qdot=qw.vh, grip=qw.gh), instr=u_q, policy=policy,
            has_instr=has_instr, has_policy=has_policy,
            instr_p=None if (instr_p is None or not self.lang_graph) else instr_p,
        )

    # ---- seeding ----
    def seed(
        self, query: SeedQuery, nodes: NodeCloud, params: PipelineParams,
        language: SeedLanguage = SeedLanguage(), memory: Tensor | None = None,
        succ: Tensor | None = None, has_memory: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Build h^0 from the learned seed and, when a memory is given, from its top nodes one hop ahead.

        Args:
            query: The encoded query.
            nodes: The graph's node cloud.
            params: topk/tier1_m.
            language: Optional instruction prior on the seed.
            memory: [V] or [B, V] the previous check's belief, or None.
            succ: [V, n_next] successor table along the next relations; needed with memory.
            has_memory: [] or [B] float mask on memory; None means every row carries one.
            mask: [V] bool nodes kept out of the seed, or None.

        Returns:
            (h^0 [..., V, hidden], seeded [..., V] bool).
        """
        factor = language.factor
        if factor is None and query.instr_p is not None and nodes.columns.task_id is not None:
            # The instruction's task posterior tilts the seed: strands of the likely tasks get seeded.
            factor = query.instr_p[..., nodes.columns.task_id]
            if query.has_instr is not None:
                flat = torch.full_like(factor, 1.0 / query.instr_p.shape[-1])
                factor = torch.where(query.has_instr[..., None] > 0, factor, flat)
        a = self.seed_head(
            query.qh_n, query.columns, query.z_h, query.pool_w, nodes.q_nodes_n, nodes.columns,
            nodes.n_v, topk=params.topk, tier1_m=params.tier1_m, factor=factor,
            mask=mask,
        )
        h0 = a[..., None] * query.e_q[..., None, :] * nodes.n_v
        seeded = a > 0
        if memory is None:
            return h0, seeded
        k = min(MEMORY_SEED, memory.shape[-1])
        idx = memory.topk(k, dim=-1).indices
        grown = torch.cat([idx, succ[idx].reshape(*idx.shape[:-1], -1)], dim=-1)
        mem_seed = torch.zeros_like(seeded).scatter_(-1, grown, True)
        if has_memory is not None:
            mem_seed = mem_seed & (has_memory[..., None] > 0)
        if mask is not None:
            mem_seed = mem_seed & ~mask
        mem_feat = memory.clamp_min(1e-12).log().clamp(MEMORY_FLOOR, 0.0)[..., None]
        h0 = h0 + mem_seed[..., None] * self.mem(mem_feat) * nodes.n_v
        return h0, seeded | mem_seed

    # ---- one shared-weight message-passing layer ----
    def _layer(self, h: Tensor, n_v: Tensor, e_q: Tensor, reached: Tensor, edges: EdgeIndex) -> tuple[Tensor, Tensor]:
        src, dst = edges.src, edges.dst
        n_nodes, hid = h.shape[-2], h.shape[-1]
        g_rel = self.rel(e_q)
        if self.agg == "sum" and edges.adj is not None:
            out = self._aggregate_sparse(h, g_rel, edges)
        elif self.agg == "sum":
            out = torch.zeros_like(h)
            for d_r, m_r in self._messages(h, g_rel, edges):
                out.index_add_(-2, d_r, m_r)
        elif self.agg == "logsumexp":
            # The stabiliser is per destination node, not per relation block.
            blocks = self._messages(h, g_rel, edges)
            mx = torch.full_like(h, LOGSUMEXP_INIT)
            s = torch.zeros_like(h)
            deg = torch.zeros(n_nodes, dtype=h.dtype, device=h.device)
            for d_r, m_r in blocks:
                mx.index_reduce_(-2, d_r, m_r.detach(), "amax", include_self=True)
            for d_r, m_r in blocks:
                s.index_add_(-2, d_r, (m_r - mx[..., d_r, :]).exp())
                deg.index_add_(0, d_r, torch.ones(d_r.shape[0], dtype=h.dtype, device=h.device))
            # deg > 0 guard: a zero-degree node would otherwise reach a Linear as garbage.
            out = torch.where(
                deg[:, None] > 0, schema.AGG_TEMP * (mx + s.clamp_min(LOG_FLOOR).log()), torch.zeros_like(mx)
            )
        else:
            raise ValueError(f"unknown agg {self.agg!r}")

        h_new = F.layer_norm(h + self.upd(torch.cat([h, out, n_v.expand_as(h)], dim=-1)), (hid,))
        # reached is tracked explicitly: layer_norm can zero a reached node's state.
        arrived = torch.zeros_like(reached, dtype=torch.int32).index_add_(
            -1, dst, reached[..., src].to(torch.int32)
        )
        return h_new, reached | (arrived > 0)

    def _aggregate_sparse(self, h: Tensor, g_rel: Tensor, edges: EdgeIndex) -> Tensor:
        """Sum aggregation as one sparse product per relation, with no per-edge tensor.

        Args:
            h: [..., V, hidden] node states.
            g_rel: [..., n_rel, hidden] this query's relation gates.
            edges: The graph, carrying one adjacency per relation.

        Returns:
            [..., V, hidden] the aggregated messages.
        """
        # The message h[src] * g_r is linear in h, so the gate comes OUT of the sum: the same
        # arithmetic as the edge loop, without ever holding an [E, hidden] tensor per relation.
        v, hid = h.shape[-2], h.shape[-1]
        flat = h.reshape(-1, v, hid)
        gates = g_rel.reshape(-1, g_rel.shape[-2], hid)
        cols = flat.permute(1, 0, 2).reshape(v, -1)
        out = torch.zeros_like(flat)
        for r, adj in enumerate(edges.adj):
            if adj is None:
                continue
            summed = torch.sparse.mm(adj, cols).reshape(v, flat.shape[0], hid)
            out = out + summed.permute(1, 0, 2) * gates[:, r : r + 1, :]
        return out.reshape(h.shape)

    def _messages(self, h: Tensor, g_rel: Tensor, edges: EdgeIndex) -> list[tuple[Tensor, Tensor]]:
        if edges.rel_offsets is None:
            return [(edges.dst, h[..., edges.src, :] * g_rel[..., edges.rel, :])]
        blocks: list[tuple[Tensor, Tensor]] = []
        for r in range(self.n_rel):
            lo, hi = edges.rel_offsets[r], edges.rel_offsets[r + 1]
            if hi > lo:
                s_r, d_r = edges.src_by_rel[lo:hi], edges.dst_by_rel[lo:hi]
                blocks.append((d_r, h[..., s_r, :] * g_rel[..., r : r + 1, :]))
        return blocks

    def propagate(self, h0: Tensor, n_v: Tensor, e_q: Tensor, ge: GraphEdges,
                  checkpoint: bool = True, seeded: Tensor | None = None,
                  mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Run self.layers shared-weight hops from the seeded h^0.

        Args:
            h0: Seeded node states, [V, hidden] or [B, V, hidden].
            n_v: Encoded node features [V, hidden].
            e_q: Query embedding, [hidden] or [B, hidden].
            ge: The graph's edge tensors, unbatched.
            checkpoint: Recompute each layer on backward instead of retaining activations.
            seeded: [..., V] the seed mask; None infers it from a nonzero h0.
            mask: [V] bool nodes that never count as reached, or None.

        Returns:
            (h^L, reached), shaped like h0 and h0's node axis.
        """
        edges = EdgeIndex(src=ge.src, dst=ge.dst, rel=ge.rel.long(),
                          src_by_rel=ge.src_by_rel, dst_by_rel=ge.dst_by_rel,
                          rel_offsets=ge.rel_offsets, adj=getattr(ge, "adj", None))
        reached = (h0.abs().sum(-1) > 0) if seeded is None else seeded
        h = h0
        for _ in range(self.layers):
            if checkpoint:
                h, reached = _checkpoint(self._layer, h, n_v, e_q, reached, edges, use_reentrant=False)
            else:
                h, reached = self._layer(h, n_v, e_q, reached, edges)
            if mask is not None:
                reached = reached & ~mask
        return h, reached

    # ---- readout ----
    def extra_features(self, query: SeedQuery, ge: GraphEdges) -> Tensor:
        """[..., V, N_EXTRA]: instruction cosine and flag, policy-chunk distance and flag, task log-prior."""
        v = ge.node_x.shape[0]
        lead = query.e_q.shape[:-1]
        zeros = torch.zeros(*lead, v, dtype=query.e_q.dtype, device=query.e_q.device)
        cos, has_instr, dist, has_policy, logp = zeros, zeros, zeros, zeros, zeros
        if query.instr_p is not None and ge.task_id is not None:
            logp = query.instr_p[..., ge.task_id].clamp_min(1e-6).log()
            if query.has_instr is not None:
                logp = logp * query.has_instr[..., None]
        if query.instr is not None and ge.e_task is not None:
            u_v = self.node_instr(ge.e_task, ge.task_id)
            cos = query.instr @ u_v.T
            has_instr = torch.ones_like(zeros) if query.has_instr is None else (
                query.has_instr[..., None].expand_as(zeros))
        if query.policy is not None and ge.action is not None:
            diff = query.policy[..., None, :, :] - ge.action
            dist = -diff.norm(dim=-1).mean(dim=-1)
            has_policy = torch.ones_like(zeros) if query.has_policy is None else (
                query.has_policy[..., None].expand_as(zeros))
            dist = dist * has_policy
        return torch.stack([cos, has_instr, dist, has_policy, logp], dim=-1)

    def logits(
        self, h: Tensor, n_v: Tensor, reached: Tensor, log_idf: Tensor, idf_w: float,
        move_cost: MoveCost = MoveCost(), extra: Tensor | None = None,
    ) -> Tensor:
        """Score every node, masking out nodes the query cannot reach.

        Args:
            h: Node states, [V, hidden] or [B, V, hidden].
            n_v: Encoded node features [V, hidden].
            reached: Reachability mask, [V] or [B, V].
            log_idf: Per-node log-IDF prior [V].
            idf_w: Weight on that prior.
            move_cost: Optional movement-cost feature; zero when unset.
            extra: [..., V, N_EXTRA] from extra_features, or None for zeros.

        Returns:
            Unnormalized logits, [V] or [B, V], -inf where ~reached.
        """
        q_nodes, q_now = move_cost.q_nodes, move_cost.q_now
        if q_now is not None and q_nodes is not None:
            move = (q_nodes - q_now[..., None, :]).norm(dim=-1, keepdim=True)
        else:
            move = torch.zeros(*h.shape[:-1], 1, dtype=h.dtype, device=h.device)
        if extra is None:
            extra = torch.zeros(*h.shape[:-1], N_EXTRA, dtype=h.dtype, device=h.device)
        raw = self.out(torch.cat([h, n_v.expand_as(h), move, extra], dim=-1)).squeeze(-1) + idf_w * log_idf
        return raw.masked_fill(~reached, NEG_INF)

    def forward(
        self, qw: QueryWindow, ge: GraphEdges, params: PipelineParams,
        columns: NodeMetricColumns, instr: Tensor | None = None, policy: Tensor | None = None,
        memory: Tensor | None = None,
    ) -> Tensor:
        """Full pipeline: encode, seed, propagate, read out.

        Args:
            qw: The query window; [H, ...] unbatched or [B, H, ...] batched.
            ge: The graph's tensors, unbatched.
            params: Seeding/readout/checkpoint knobs.
            columns: The graph's per-node metric columns.
            instr: Instruction embedding, or None.
            policy: The policy chunk just issued, or None.
            memory: The previous check's belief, or None.

        Returns:
            Unnormalized logits, [V] or [B, V].
        """
        query = self.query(qw, instr, policy)
        n_v = self.node_encode(ge.node_x)
        q_nodes_n = self.normalize_q(ge.node_x[:, : self.dim])
        cloud = NodeCloud(q_nodes_n=q_nodes_n, n_v=n_v, columns=columns)
        h0, seeded = self.seed(query, cloud, params, memory=memory, succ=ge.succ)
        h_l, reached = self.propagate(h0, n_v, query.e_q, ge, checkpoint=params.checkpoint,
                                      seeded=seeded)
        return self.logits(h_l, n_v, reached, ge.log_idf, params.idf_w,
                           MoveCost(q_nodes=q_nodes_n, q_now=query.qh_n[..., -1, :]),
                           self.extra_features(query, ge))


def save_checkpoint(net: GraphRetrieverNet, nodes: NodeTable, edges: EdgeSet, path: str | os.PathLike) -> str:
    """Write the head to path, stamped with the graph hash."""
    out_path = npz.resolve(path, schema.HEAD_NPZ)
    out = net.state_npz_dict()
    out["graph_hash"] = np.array(graph_hash(
        nodes, edges, constants=out_path.parent / CONSTANTS_JSON, c2_state=net.c2_state(),
    ))
    return os.fspath(npz.write_arrays(out_path, schema.HEAD_NPZ, out))
