"""Runtime retrieval against a trained demonstration graph.

graph_hash is the critical check: node ids are positional, so a head trained on one graph loaded
against another would return confident, well-formed logits over the WRONG strands with no other
error signal. load refuses mismatches.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from robot_ood.core.config import GraphConfig
from robot_ood.data.store import ArtifactStore
from robot_ood.data import schema
from robot_ood.data.binning import phase_hist as _phase_hist_geom
from robot_ood.data.geometry import CONSTANTS_JSON, edge_pad_hist, finite_diff_vel, graph_hash, off_manifold_distance
from robot_ood.core.device import resolve_device as _resolve_device
from robot_ood.data.edges import EdgeSet
from robot_ood.data.metrics import NodeMetricColumns, QueryMetricColumns
from robot_ood.models.retrieval.gnn import C2_PREFIX, GraphRetrieverNet
from robot_ood.models.types import GraphEdges, MoveCost, NodeCloud, PipelineParams, QueryWindow as _QW
from robot_ood.data.nodes import NodeTable

from robot_ood.runtime.readout import ReadoutContext, ReadoutResult, read_out

# ==== GraphRetrieval — one call's full answer ====
@dataclass(frozen=True, slots=True)
class GraphRetrieval:
    """One retrieval call's answer: node id, reference segment, phase histogram, and posterior.

    Attributes:
        p_node: [V] float64 posterior truncated to the top cfg.topm nodes and renormalized.
        p_phase: [schema.NBINS_ALIGN] float64 p_node rebinned by phase.
        node: The chosen node id, or -1 when the readout arm produced a free-form target.
        owner: The node's demo id; -1 when node == -1.
        task_id: The node's task id; -1 when node == -1.
        stage: The node's stage index; -1 when node == -1.
        t_idx: The node's raw frame index; -1 when node == -1.
        q_star: [D] float64 first waypoint, the single-setpoint answer.
        q_seg: [k, D] float64 reference segment, cfg.seg_k raw frames from node.
        qdot_seg: [k, D] float64 reference velocities.
        depth: Phase of the chosen node, in [0, 1].
        conf: Top-1 mass of the readout's truncated support.
        entropy: Shannon entropy of p_node (nats).
        nll: -log p under the untruncated softmax; nan when node == -1.
        off_manifold: Geometric nearest-node distance from the most recent query frame.
        info: Reached-node count, realized top-M, history length, readout arm and graph_hash.
    """

    p_node: np.ndarray
    p_phase: np.ndarray
    node: int
    owner: int
    task_id: int
    stage: int
    t_idx: int
    q_star: np.ndarray
    q_seg: np.ndarray
    qdot_seg: np.ndarray
    depth: float
    conf: float
    entropy: float
    nll: float
    off_manifold: float
    info: dict


# ==== QueryWindow — the fixed-size, right-aligned window the network actually sees ====
@dataclass(frozen=True, slots=True)
class QueryWindow:
    """One query's padded window, with every channel the network needs already derived.

    Attributes:
        q: [hist, D] float64 padded joint configs, row 0 oldest.
        vel: [hist, D] float64 finite-difference velocities.
        grip: [hist] float64 gripper flag, constant across the window by construction.
        hist_len: Real frames supplied, BEFORE padding.
        padded: True iff the raw window was shorter than hist.
    """

    q: np.ndarray
    vel: np.ndarray
    grip: np.ndarray
    hist_len: int
    padded: bool

    @property
    def q_now(self) -> np.ndarray:
        """[D] float64 most recent joint reading."""
        return self.q[-1]


class QueryPreprocessor:
    """Turns a raw, variable-length joint history into a fixed-size QueryWindow.

    Args:
        dim: Joint-space dimensionality the graph was built at.
        hist: Target window length (cfg.hist).
    """

    def __init__(self, dim: int, hist: int) -> None:
        self.dim = int(dim)
        self.hist = int(hist)

    def build(self, q_hist: np.ndarray, g_hist: float | np.ndarray | None) -> QueryWindow:
        """Right-align, edge-pad, and derive the velocity and grip channels.

        Args:
            q_hist: [T, D] joint configs, row -1 most recent, T >= 1.
            g_hist: Current gripper flag, or None for open.

        Returns:
            The padded window.

        Raises:
            ValueError: q_hist is not [T, dim] with T >= 1.
        """
        raw = np.atleast_2d(np.asarray(q_hist, np.float64))
        if raw.ndim != 2 or raw.shape[1] != self.dim:
            raise ValueError(f"q_hist must be [T,{self.dim}], got shape {raw.shape}")
        if raw.shape[0] == 0:
            raise ValueError("q_hist must contain at least the current joint reading (T>=1)")

        q_pad, padded = edge_pad_hist(raw, self.hist)
        g_now = 0.0 if g_hist is None else float(np.asarray(g_hist, np.float64).reshape(-1)[-1])
        return QueryWindow(
            q=q_pad,
            # SAME helper robot_ood.train.data builds its windows with -- one implementation only.
            vel=finite_diff_vel(q_pad),
            grip=np.full(self.hist, g_now, dtype=np.float64),
            hist_len=int(raw.shape[0]),
            padded=bool(padded),
        )


# ==== CandidateMasker — which nodes are legal for this query ====
class CandidateMasker:
    """Restricts the candidate node pool: leave-one-demo-out, the ENTRY band, and virtual nodes.

    Args:
        device: Torch device the masks live on.
        entry_band: Phase ceiling for the ENTRY stratum; <= 0 disables the band.
        owner_ids: [V] per-node demo id.
        phase: [V] per-node phase.
        is_virtual: [V] bool, True on the synthesised entry nodes, or None on a graph without them.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        entry_band: float,
        owner_ids: torch.Tensor,
        phase: torch.Tensor,
        is_virtual: torch.Tensor | None = None,
    ) -> None:
        self.device = device
        self.entry_band = float(entry_band)
        self._owner_ids = owner_ids
        self._entry_band_mask = (phase <= self.entry_band) if self.entry_band > 0.0 else None
        self._virtual_mask = is_virtual if is_virtual is not None and bool(is_virtual.any()) else None

    def mask(self, logits: torch.Tensor, *, exclude_owner: int | None) -> torch.Tensor:
        """Fold every candidate-pool restriction into logits.

        Args:
            logits: [V] raw node logits.
            exclude_owner: Demo id to exclude, or None.

        Returns:
            [V] logits with illegal nodes set to -inf.
        """
        if exclude_owner is not None:
            logits = logits.masked_fill(self._owner_ids == int(exclude_owner), float("-inf"))
        if self._entry_band_mask is not None:
            logits = logits.masked_fill(~self._entry_band_mask, float("-inf"))
        if self._virtual_mask is not None:
            # A virtual node may be seeded and pass messages, but is never an answer.
            logits = logits.masked_fill(self._virtual_mask, float("-inf"))
        return logits


# ==== Observation — the network forward pass, before softmax / top-M truncation / readout ====
@dataclass(frozen=True, slots=True)
class Observation:
    """The network forward pass, before softmax / truncation / readout.

    Attributes:
        logits64: [V] float64 dense, untruncated, -inf where unreached, every mask
            (exclude_owner, entry band) folded in. This is what the action is read off, and what
            the task fusion observes.
        e_q: [hidden] pooled query embedding.
        off_manifold: Geometric nearest-node distance from the query.
        window: The padded window this observation was produced from.
    """

    logits64: torch.Tensor
    e_q: torch.Tensor
    off_manifold: float
    window: QueryWindow

    @property
    def hist_len(self) -> int:
        """Real frames supplied, BEFORE padding."""
        return self.window.hist_len

    @property
    def padded(self) -> bool:
        """True iff the raw window was shorter than cfg.hist."""
        return self.window.padded


@dataclass(frozen=True, slots=True)
class _ReadoutOutcome:
    """One readout arm's answer, in this module's vocabulary.

    Attributes:
        node: Chosen node id, or -1 for a free-form target (both shipped arms).
        q_star: [D] float64 target configuration.
        q_seg: [seg_k, D] float64 reference segment.
        qdot_seg: [seg_k, D] float64.
        depth: Phase of the target.
        conf: Top-1 mass of the arm's own truncated support.
        info: Arm-specific telemetry, plus target_off_manifold.
    """

    node: int
    q_star: np.ndarray
    q_seg: np.ndarray
    qdot_seg: np.ndarray
    depth: float
    conf: float
    info: dict


# ==== PosteriorAggregator — logits -> the full retrieval answer ====
class PosteriorAggregator:
    """Softmax -> ENTRY weighting -> top-M truncation -> readout -> summary statistics.

    Args:
        nodes: The graph's node table.
        cfg: Graph configuration (topm/temp/seg_k/entry_band).
        graph_hash_value: Fingerprint stamped into every answer's info.
    """

    def __init__(self, nodes: NodeTable, cfg: GraphConfig, graph_hash_value: str) -> None:
        self.nodes = nodes
        self.cfg = cfg
        self.graph_hash = graph_hash_value

    def aggregate(self, obs: "Observation", *, q_now: np.ndarray) -> "GraphRetrieval":
        """Turn one forward pass into the full retrieval answer.

        Args:
            obs: The network forward pass.
            q_now: The window's most recent joint reading; the target's off_manifold origin.

        Returns:
            The assembled answer.
        """
        # This stage runs in float64 throughout, so repeated calls on identical input are bit-identical.
        with torch.no_grad():
            logits64 = obs.logits64
            p_full = torch.softmax(logits64, dim=0)             # [V] float64, exact (no sampling)

            finite = torch.isfinite(logits64)
            entry = self._entry_aggregation(finite)             # None unless cfg.entry_band > 0
            if entry is not None:
                p_full, topm = entry
            else:
                topm = min(self.cfg.topm, int(finite.sum().item()))
            p_node_t = self._truncate(p_full, topm)

        p_full_np = p_full.detach().cpu().numpy()
        with torch.no_grad():       # readout arms build no autograd graph either
            outcome = self.readout(p_full, q_now=q_now,
                                    topm=None if entry is None else entry[1])

        p_node = p_node_t.detach().cpu().numpy()
        node = outcome.node
        owner, task_id, stage, t_idx, nll = self._node_columns(node, p_full_np)

        nz = p_node > 0
        info = {
            "reached": int(finite.sum().item()), "topm": int(topm), "hist_len": obs.hist_len,
            "padded": obs.padded, "graph_hash": self.graph_hash,
            **outcome.info,
        }
        return GraphRetrieval(
            p_node=p_node, p_phase=self.phase_hist(p_node), node=node, owner=owner, task_id=task_id,
            stage=stage, t_idx=t_idx, q_star=outcome.q_star, q_seg=outcome.q_seg,
            qdot_seg=outcome.qdot_seg, depth=outcome.depth, conf=outcome.conf,
            entropy=float(-(p_node[nz] * np.log(p_node[nz])).sum()), nll=nll,
            off_manifold=obs.off_manifold, info=info,
        )

    def _truncate(self, p_full: torch.Tensor, topm: int) -> torch.Tensor:
        """[V] float64 EXACT top-topm slice of p_full, renormalized, zero elsewhere.

        Args:
            p_full: [V] float64 distribution.
            topm: Truncation width.

        Returns:
            [V] float64, summing to 1.
        """
        topv, topidx = torch.topk(p_full, topm, largest=True, sorted=True)      # EXACT top-K
        p_node_t = torch.zeros_like(p_full)
        p_node_t[topidx] = topv
        return p_node_t / p_node_t.sum().clamp_min(1e-300)

    def _node_columns(
        self, node: int, p_full_np: np.ndarray
    ) -> tuple[int, int, int, int, float]:
        """The chosen node's bookkeeping columns, or the free-form sentinel row.

        Args:
            node: Chosen node id, or -1.
            p_full_np: [V] untruncated posterior, for the nll.

        Returns:
            (owner, task_id, stage, t_idx, nll); all -1 / nan for a free-form target.
        """
        if node < 0:
            return -1, -1, -1, -1, float("nan")    # no single node to read a likelihood off of
        return (
            int(self.nodes.owner[node]), int(self.nodes.task_id[node]),
            int(self.nodes.stage[node]), int(self.nodes.t_idx[node]),
            float(-np.log(max(p_full_np[node], 1e-300))),
        )

    def phase_hist(self, p_node: np.ndarray) -> np.ndarray:
        """[schema.NBINS_ALIGN] WHEN-readout: rebin p_node's mass by phase."""
        return _phase_hist_geom(self.nodes.phase, p_node)

    def _entry_aggregation(self, finite: torch.Tensor) -> tuple[torch.Tensor, int] | None:
        """(p, topm) for the t=0 ENTRY regime, or None when cfg.entry_band is off."""
        if float(getattr(self.cfg, "entry_band", 0.0)) <= 0.0:
            return None
        n = int(finite.sum().item())
        # topm widens to n so weighted_topk's own clamp cannot re-truncate the pool to an arbitrary index-ordered count.
        # torch.sort is stable on ties, so q* is the plain barycentre of that determinate node set.
        return finite.double() / float(max(n, 1)), max(self.cfg.topm, n)

    def readout(
        self, p_full: torch.Tensor, *, q_now: np.ndarray | None = None, topm: int | None = None,
    ) -> "_ReadoutOutcome":
        """Collapse the posterior into one target.

        Args:
            p_full: [V] float64 untruncated posterior.
            q_now: The current arm config, the origin off_manifold is measured from.
            topm: Truncation ceiling handed to the readout; None uses cfg.topm (ENTRY widens it).

        Returns:
            The answer, flattened into this module's own vocabulary.
        """
        ctx = ReadoutContext(
            self.nodes,
            self.cfg.topm if topm is None else topm,      # entry mode widens it; see _entry_aggregation
            self.cfg.temp, self.cfg.seg_k, q_now=q_now,
        )
        result: ReadoutResult = read_out(p_full.detach().cpu().numpy(), ctx)
        info = dict(result.info)
        # The TARGET's own distance to the nearest node; GraphRetrieval.off_manifold is the
        # QUERY's. Different quantities -- keep both.
        info["target_off_manifold"] = float(result.off_manifold)
        return _ReadoutOutcome(
            node=-1,
            q_star=np.asarray(result.q_star, np.float64),
            q_seg=np.asarray(result.q_seg, np.float64),
            qdot_seg=np.asarray(result.qdot_seg, np.float64),
            depth=float(result.depth),
            conf=float(result.conf),
            info=info,
        )


# ==== GraphRetriever ====
@dataclass(frozen=True, slots=True)
class _Language:
    """The deploy-side instruction channel: the task table and its embeddings.

    Attributes:
        tasks: robot_ood.models.language.encoder.TaskEmbeddings for this graph.
        e_task: [n_tasks, embed_dim] the table's embeddings on device.
    """

    tasks: Any
    e_task: torch.Tensor


def _load_language(graph_dir: Any, nodes: NodeTable, device: torch.device) -> "_Language | None":
    """Load the graph's task embeddings, or None when it ships none."""
    if graph_dir is None:
        return None
    from robot_ood.models.language.encoder import LANG_NPZ, TaskEmbeddings

    root = Path(graph_dir)
    if not (root / LANG_NPZ).is_file():
        return None
    tasks = TaskEmbeddings.load(root).check_against(nodes.task_names)
    return _Language(tasks=tasks, e_task=torch.as_tensor(tasks.e_task, dtype=torch.float32, device=device))


class GraphRetriever:
    """Load a trained demo-graph head + the graph it was trained on, and expose retrieve."""

    def __init__(
        self,
        net: GraphRetrieverNet,
        nodes: NodeTable,
        edges: EdgeSet,
        *,
        cfg: GraphConfig | None = None,
        device: str | torch.device | None = None,
        graph_dir: str | os.PathLike | None = None,
    ):
        self.cfg = cfg or GraphConfig()
        self.device = _resolve_device(device if device is not None else self.cfg.device)
        self.nodes = nodes
        self.edges = edges
        # Stage 1 of the pipeline: raw history -> fixed-size QueryWindow.
        self.preprocessor = QueryPreprocessor(nodes.dim, int(self.cfg.hist))
        self.net = net.to(self.device).eval()
        # graph_hash v2: geometry, C0's constants and the head's own C2 tensors. C2 sets the edge
        # weights, so a geometry-only hash would not move when the graph does.
        self.graph_hash = graph_hash(
            nodes, edges, constants=None if graph_dir is None else Path(graph_dir) / CONSTANTS_JSON,
            c2_state=self.net.c2_state(),
        )
        # Stage 3 of the pipeline: logits -> the full answer.
        self.aggregator = PosteriorAggregator(nodes, self.cfg, self.graph_hash)

        # ---- cache everything query-independent, ONCE, on self.device (the perf contract) --------
        nt = nodes.torch(self.device)
        node_x = torch.cat(
            [nt["q"], nt["qdot"], nt["grip"].to(nt["q"].dtype).unsqueeze(-1), nt["t_frac"].unsqueeze(-1)],
            dim=-1,
        )
        self._lang = _load_language(graph_dir, nodes, self.device)
        self.task_id_t = torch.as_tensor(np.asarray(nodes.task_id, dtype=np.int64), device=self.device)
        with torch.no_grad():
            self._n_v = self.net.node_repr(
                node_x, None if self._lang is None else self._lang.e_task, self.task_id_t)
            self._q_nodes_n = self.net.normalize_q(nt["q"])
            self._phase_t = torch.as_tensor(nodes.phase, dtype=torch.float32, device=self.device)
        self.phase64 = self._phase_t.double()
        self.task_id_t = torch.as_tensor(np.asarray(nodes.task_id, dtype=np.int64), device=self.device)
        # seed's graph-only, query-independent tensors (see NodeCloud's docstring) -- cached once
        # here, the metric columns included: C2 reads them on every check.
        self.node_cloud = NodeCloud(
            q_nodes_n=self._q_nodes_n, n_v=self._n_v,
            columns=NodeMetricColumns.from_node_table(nodes, self.device),
        )
        # Host mirror of the node owner ids, for the masker's exclude_owner filter.
        self._owner_t = nt["owner"]
        # Stage 2 of the pipeline: which nodes are legal for this query.
        self.masker = CandidateMasker(
            device=self.device, entry_band=float(getattr(self.cfg, "entry_band", 0.0)),
            owner_ids=self._owner_t, phase=self._phase_t,
            is_virtual=torch.as_tensor(
                np.asarray(self.nodes.virtual_mask, dtype=bool), device=self.device),
        )
        self._e_lang_cache: dict[str, torch.Tensor] = {}
        self._p_lang_cache: dict[str, torch.Tensor] = {}

        et = edges.torch(self.device)
        self._src = et["src"].to(torch.int32)
        self._dst = et["dst"].to(torch.int32)
        self._rel = et["rel"]
        self._log_idf = et["log_idf"]
        self._ge = GraphEdges(
            node_x=node_x, src=self._src, dst=self._dst, rel=self._rel, log_idf=self._log_idf,
            task_id=self.task_id_t, e_task=None if self._lang is None else self._lang.e_task,
            action=torch.as_tensor(nodes.action_chunks(self.net.chunk), device=self.device),
            succ=torch.as_tensor(edges.next_successors(len(nodes)), device=self.device),
        )

    # ---- construction ------------------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        graph_dir: str | os.PathLike,
        *,
        cfg: GraphConfig | None = None,
        device: str | torch.device | None = None,
        head_npz: str = schema.HEAD_NPZ,
    ) -> "GraphRetriever":
        """Load <graph_dir>/{g_nodes,g_edges,<head_npz>}.npz."""
        cfg = cfg or GraphConfig()
        graph_dir = Path(graph_dir)
        store = ArtifactStore(graph_dir)
        nodes = store.nodes()
        edges = store.edges()

        head_path = graph_dir / head_npz
        raw = np.load(os.fspath(head_path))
        got = str(raw["graph_hash"]) if "graph_hash" in raw.files else "<none: head predates graph_hash stamping>"
        c2 = {k: raw[k] for k in raw.files if k.startswith(C2_PREFIX)}
        want = graph_hash(
            nodes, edges, constants=graph_dir / CONSTANTS_JSON, c2_state=c2 or None,
        )
        if got != want:
            raise ValueError(
                f"GraphRetriever.load({graph_dir}): graph_hash mismatch -- head={got!r} graph={want!r}. "
                f"The head at {head_path} was not trained on THIS (g_nodes.npz, g_edges.npz) pair: node "
                "ids are positional and mean nothing across builder runs, so applying a foreign head "
                "yields confident, well-formed targets on the WRONG demonstration strands -- the worst "
                "failure mode in this subsystem, since nothing else in the pipeline would ever raise "
                "for it. Rebuild the head against this graph, or point `graph_dir` at the artifacts "
                "the head was actually trained on."
            )
        net = GraphRetrieverNet.load_npz(head_path)

        return cls(net, nodes, edges, cfg=cfg, device=device, graph_dir=graph_dir)

    # ---- the network forward pass, before softmax / top-M / readout -----------------------------------
    def language_embedding(self, task: str) -> torch.Tensor | None:
        """This episode's instruction embedding, in the task table's own centred space.

        Args:
            task: The harness's raw instruction string.

        Returns:
            [embed_dim] float32 on device, or None when this graph carries no instruction prior or
            the caller supplied no instruction.
        """
        if self._lang is None or not task:
            return None
        cached = self._e_lang_cache.get(task)
        if cached is None:
            from robot_ood.models.language.encoder import InstructionEncoder

            if getattr(self, "_encoder", None) is None:
                self._encoder = InstructionEncoder()
            # The canonical task embeddings, mixed by the head's task posterior.
            row = self._lang.tasks.embed_query_posterior(task, self._encoder)
            cached = torch.as_tensor(row, dtype=torch.float32, device=self.device)
            self._e_lang_cache[task] = cached
        return cached

    def language_posterior(self, task: str) -> torch.Tensor | None:
        """This episode's instruction as a posterior over the graph's tasks, or None without language."""
        if self._lang is None or not task or not self.net.lang_graph:
            return None
        cached = self._p_lang_cache.get(task)
        if cached is None:
            from robot_ood.models.language.encoder import InstructionEncoder

            if getattr(self, "_encoder", None) is None:
                self._encoder = InstructionEncoder()
            cached = torch.as_tensor(self._lang.tasks.task_posterior(task, self._encoder),
                                     dtype=torch.float32, device=self.device)
            self._p_lang_cache[task] = cached
        return cached

    def observe(
        self, q_hist: np.ndarray, *, g_hist: float | np.ndarray | None = None,
        exclude_owner: int | None = None, instr: torch.Tensor | None = None,
        policy: np.ndarray | None = None, memory: torch.Tensor | None = None,
        instr_p: torch.Tensor | None = None,
    ) -> "Observation":
        """Run the trained net forward; untruncated logits before softmax / readout.

        Args:
            q_hist: [T, D] joint history, row -1 most recent.
            g_hist: Current gripper flag, or None for open.
            exclude_owner: Demo id to mask out, or None.
            instr: [instr_dim] the episode's instruction embedding, or None.
            policy: [chunk, 7] the chunk the policy issued after the previous check, or None.
            memory: [V] the previous check's belief, or None on the first check.
            instr_p: [n_tasks] the instruction's task posterior, or None.

        Returns:
            The observation.
        """
        window = self.preprocessor.build(q_hist, g_hist)

        qh = torch.as_tensor(window.q, dtype=torch.float32, device=self.device)
        vh = torch.as_tensor(window.vel, dtype=torch.float32, device=self.device)
        gh = torch.as_tensor(window.grip, dtype=torch.float32, device=self.device)
        policy_t = None if policy is None else torch.as_tensor(
            np.asarray(policy, dtype=np.float32), device=self.device)

        with torch.no_grad():
            query = self.net.query(_QW(qh, vh, gh), instr, policy_t, instr_p=instr_p)
            h0, seeded = self.net.seed(
                query, self.node_cloud,
                PipelineParams(topk=self.cfg.seed_topk, tier1_m=self.cfg.seed_tier1_m),
                memory=None if memory is None else memory.to(self.device), succ=self._ge.succ,
            )
            h_l, reached = self.net.propagate(h0, self._n_v, query.e_q, self._ge, checkpoint=False,
                                              seeded=seeded)
            move_cost = bool(getattr(self.cfg, "move_cost", True))
            logits = self.net.logits(
                h_l, self._n_v, reached, self._log_idf, self.cfg.idf,
                MoveCost(q_nodes=self._q_nodes_n if move_cost else None,
                         q_now=query.qh_n[-1] if move_cost else None),
                self.net.extra_features(query, self._ge),
            )
            logits64 = self.masker.mask(logits, exclude_owner=exclude_owner).double()
            if not bool(torch.isfinite(logits64).any()):
                raise RuntimeError(
                    "GraphRetriever.observe: no node is reachable under the current mask (the history "
                    "window is too far from every strand, or exclude_owner filtered out the only "
                    "reachable one)"
                )
            off_manifold = off_manifold_distance(self.nodes, window.q_now)

        return Observation(logits64=logits64, e_q=query.e_q, off_manifold=off_manifold, window=window)

    # ---- the one runtime call -------------------------------------------------------------------------
    def retrieve(
        self, q_hist: np.ndarray, *, g_hist: float | np.ndarray | None = None,
        exclude_owner: int | None = None,
    ) -> GraphRetrieval:
        """A window of recent joint configs -> p(v|Q) -> a reference SEGMENT."""
        obs = self.observe(q_hist, g_hist=g_hist, exclude_owner=exclude_owner)
        # The padded window rides along on the Observation, so the readout's q_now costs nothing here.
        return self.aggregator.aggregate(obs, q_now=obs.window.q_now)
