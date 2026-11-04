"""Everything the check flow loads: graph artifacts, the nets, the action-unit calibration.

This is the only half of the runtime that touches disk or reads the environment. Each accessor
resolves on first use and caches, so the first plan() of a run pays for all of it and no later
control step does.
"""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from robot_ood.core.action import ActionScale
from robot_ood.core.paths import default_paths

from robot_ood.core.config import SentinelConfig

if TYPE_CHECKING:                                          # pragma: no cover -- typing only
    from robot_ood.core.kinematics import PandaKinematics
    from robot_ood.models.fcsr_net import FcsrNet
    from robot_ood.core.config import GraphConfig
    from robot_ood.models.retrieval.retriever import SegmentRetriever
    from robot_ood.models.fusion.authority import CorrectionAuthority
    from robot_ood.models.fusion.corrector import Corrector
    from robot_ood.runtime.locator import GraphLocator

# Written by python -m robot_ood.cli build calibrate (robot_ood.build.calibration) next to the graph artifacts.
SCALE_JSON = "action_scale.json"


class RuntimeBundle:
    """The loaded side of a Sentinel: one per Sentinel, shared by every worker it serves."""

    def __init__(self, cfg: SentinelConfig | None = None,
                 fwm_dir: str | os.PathLike | None = None):
        """Initialize this bundle.

        Args:
            cfg: Sentinel knobs; None reads them from the environment.
            fwm_dir: Field directory; None resolves robot_ood.core.paths.Paths.fwm.
        """
        self.cfg = cfg or SentinelConfig.from_env()
        self.fwm_dir = os.fspath(fwm_dir) if fwm_dir is not None else os.fspath(default_paths().fwm())
        self._scale: ActionScale | None = None
        self._kin: "PandaKinematics | None" = None
        self._node_ee_cache: "np.ndarray | None" = None
        self._language: Any = None
        self._encoder: Any = None
        self._e_lang: dict[str, "np.ndarray"] = {}
        self._fcsr: "FcsrNet | None" = None
        self._retriever: "SegmentRetriever | None" = None

    def suite(self) -> str:
        """The suite this bundle's field directory belongs to; "" for the fwm root (LIBERO object)."""
        suite = os.path.basename(os.path.normpath(self.fwm_dir))
        return "" if suite == "fwm" else suite

    def graph_config(self) -> tuple["GraphConfig", str]:
        """This bundle's GraphConfig and the graph directory it resolves to."""
        from robot_ood.core.config import GraphConfig

        c = self.cfg
        # A graph named by the sentinel reads its own embedded calibration: a two-armed deploy
        # runs one bundle per lane, and $GR_GRAPH_DIR can name at most one of them.
        base = GraphConfig.from_env(c.graph_dir or None)
        # Replace only SN_GRAPH_* fields; bare GraphConfig would silently reset other GR_* knobs.
        gcfg = dataclasses.replace(
            base, hist=c.graph_win,
            topm=c.graph_topm, temp=c.graph_temp, device=c.graph_device,
        )
        graph_dir = gcfg.graph_dir or default_paths().graph(self.suite())
        return gcfg, graph_dir

    def build_locator(self) -> "GraphLocator":
        """Construct this graph's locator. Callers cache it per worker; this always builds one."""
        from robot_ood.runtime.locator import GraphLocator

        gcfg, graph_dir = self.graph_config()
        return GraphLocator.load(graph_dir, cfg=gcfg, memory=not self.cfg.ablation.memory)

    def query_context(self, *, q_win: Any, a_win: Any, grip: float, task: str, ee_now: Any,
                      phase: float, task_p: "np.ndarray | None" = None,
                      seg_prior: "np.ndarray | None" = None) -> Any:
        """The QueryContext a check hands the retriever, shared by the fusion arm and the graph-off phase readout."""
        from robot_ood.core.schemas import QueryContext
        from robot_ood.data.geometry import finite_diff_vel

        gcfg, _ = self.graph_config()
        q_hist = np.asarray(q_win, dtype=np.float32)
        a_hist = None
        if gcfg.action_hist and len(a_win) == a_win.maxlen:
            a_hist = torch.as_tensor(np.asarray(a_win, dtype=np.float32))[None]
        e_lang = self.instruction_embedding(task)
        return QueryContext(
            qh=torch.as_tensor(q_hist)[None],
            vh=torch.as_tensor(finite_diff_vel(q_hist))[None],
            gh=torch.as_tensor(np.full(q_hist.shape[0], grip, dtype=np.float32))[None],
            ee=torch.as_tensor(np.asarray(ee_now, dtype=np.float32))[None],
            phase=torch.as_tensor(np.asarray([phase], dtype=np.float32)),
            a_hist=a_hist,
            e_lang=None if e_lang is None else torch.as_tensor(np.asarray(e_lang, np.float32))[None],
            task_p=None if task_p is None else torch.as_tensor(np.asarray(task_p, np.float32))[None],
            seg_prior=None if seg_prior is None else torch.as_tensor(np.asarray(seg_prior, np.float32))[None],
        )

    def topk_phase(self, ctx: Any) -> float:
        """Phase read straight off the top-k retrieved segments: the graph-off phase estimate.

        With the belief filter gone, phase is the score-weighted mean of the retrieved neighbours'
        own phase_start, a plain top-k readout over the demo segments, no propagation, no memory.
        """
        cands = self.segment_retriever().retrieve(ctx)
        w = np.asarray(cands.score, dtype=np.float64)
        w = np.exp(w - w.max())
        w = w / max(float(w.sum()), 1e-300)
        return float(np.dot(w, np.asarray(cands.phase, dtype=np.float64)))

    def instruction_embedding(self, task: str) -> "np.ndarray | None":
        """This episode's instruction, in the space the segment keys were trained against.

        Args:
            task: The instruction the policy was given, verbatim.

        Returns:
            [D_lang] float32, or None when the trained query encoder has no language branch.
        """
        if not task or not self.segment_retriever().query_encoder.lang_dim:
            return None
        if task not in self._e_lang:
            # One encoder forward per distinct instruction, not per check; the same posterior read
            # as GraphRetriever.language_embedding, so both consumers see one instruction.
            tasks, enc = self._task_embeddings(), self._instruction_encoder()
            self._e_lang[task] = tasks.embed_query_posterior(task, enc)
        return self._e_lang[task]

    def _task_embeddings(self) -> Any:
        """The graph's centred task embeddings, read once and cached."""
        from robot_ood.models.language.encoder import TaskEmbeddings

        if self._language is None:
            self._language = TaskEmbeddings.load(self.graph_config()[1])
        return self._language

    def _instruction_encoder(self) -> Any:
        """The frozen sentence encoder the graph's embeddings were built with, loaded once."""
        from robot_ood.models.language.encoder import InstructionEncoder

        if self._encoder is None:
            self._encoder = InstructionEncoder()
        return self._encoder

    def fcsr_net(self) -> "FcsrNet":
        """The trained FCSR net, read once and cached.

        Returns:
            The net. An unset cfg.device is CPU, never "cuda if one is visible": the runtime feeds
            these modules CPU tensors.

        Raises:
            FileNotFoundError: No checkpoint beside the graph. Never defaulted to a fixed-weight
                blend: a run that measures a different mechanism than the one it was asked for is
                worse than a run that does not start.
        """
        from robot_ood.models.fcsr_net import FcsrNet

        if self._fcsr is None:
            _, graph_dir = self.graph_config()
            self._fcsr = FcsrNet.load(graph_dir, self._device())
        return self._fcsr

    def _device(self) -> "torch.device":
        """Where the trained modules go. An unset cfg.device is CPU, never a visible CUDA one."""
        gcfg, _ = self.graph_config()
        return torch.device(gcfg.device) if gcfg.device else torch.device("cpu")

    def segment_retriever(self) -> "SegmentRetriever":
        """This bundle's segment retriever (robot_ood.models.retrieval)."""
        from robot_ood.data.store import ArtifactStore
        from robot_ood.models.retrieval.retriever import SegmentRetriever

        if self._retriever is None:
            gcfg, graph_dir = self.graph_config()
            net = self.fcsr_net()
            self._retriever = SegmentRetriever(
                net.seg.query, net.seg.key, net.seg.action, net.scorer,
                ArtifactStore(graph_dir).segments(), cfg=gcfg, device=self._device(),
            )
        return self._retriever

    def corrector(self) -> "Corrector":
        """The trained corrector (trust and shape heads), off the loaded FCSR checkpoint."""
        return self.fcsr_net().corrector

    def authority(self) -> "CorrectionAuthority":
        """The CorrectionAuthority this bundle's configuration and checkpoints describe."""
        from robot_ood.models.fusion.authority import CorrectionAuthority

        from robot_ood.core.kinematics import cartesian_for
        from robot_ood.data.nodes import NodeTable
        from robot_ood.models.fusion.authority import STALL_CHECKS
        from robot_ood.models.fusion.target import REF_MODE, RawPoseRef, raw_pose_ref_for

        _, graph_dir = self.graph_config()
        need_nodes = REF_MODE in ("raw", "rawnj") or STALL_CHECKS > 0
        nodes = NodeTable.load(graph_dir) if need_nodes else None
        raw_ref = raw_pose_ref_for(nodes) if REF_MODE in ("raw", "rawnj") else None
        rel_ref = RawPoseRef.from_nodes(nodes, relative=True) if STALL_CHECKS > 0 else None
        return CorrectionAuthority(
            corrector=self.corrector(), node_ee=self.node_ee(), scale=self.action_scale(),
            pause=self.segment_pause(), direct=self.cfg.ablation.corrector, raw_ref=raw_ref,
            rel_ref=rel_ref, cart=cartesian_for(self.suite()),
        )

    def segment_pause(self) -> "np.ndarray":
        """[S] bool which segments are pauses (robot_ood.data.segments.segment_pause), read once and cached."""
        from robot_ood.data.nodes import NodeTable
        from robot_ood.data.segments import segment_pause
        from robot_ood.data.store import ArtifactStore

        if getattr(self, "_seg_pause_cache", None) is None:
            _, graph_dir = self.graph_config()
            self._seg_pause_cache = segment_pause(NodeTable.load(graph_dir), ArtifactStore(graph_dir).segments())
        return self._seg_pause_cache

    def segment_task_ids(self) -> "np.ndarray":
        """[S] int64 each segment's task id, read once and cached."""
        from robot_ood.data.store import ArtifactStore

        if getattr(self, "_seg_task_cache", None) is None:
            _, graph_dir = self.graph_config()
            self._seg_task_cache = np.asarray(ArtifactStore(graph_dir).segments().task_id, dtype=np.int64)
        return self._seg_task_cache

    def node_ee(self) -> "np.ndarray":
        """This graph's per-node end-effector pose, read once and cached."""
        from robot_ood.data.nodes import NodeTable
        from robot_ood.data.segments import node_ee_table

        # node_pose_ref indexes into this table; same construction as training side (node_ee_table).
        # Segment ids stay positional against node table they were built from.
        if self._node_ee_cache is None:
            _, graph_dir = self.graph_config()
            self._node_ee_cache = node_ee_table(NodeTable.load(graph_dir))
        return self._node_ee_cache

    def kinematics(self) -> "PandaKinematics":
        """The forward kinematics behind both operands of the tracking chunk, built once and cached.

        Returns:
            The model. Its chain is built on first use, and the first plan() of a run already pays
            for loading the graph artifacts, so nothing new lands in a later control step.
        """
        if self._kin is None:
            from robot_ood.core.kinematics import kinematics_for

            self._kin = kinematics_for(self.suite())
        return self._kin

    def action_scale(self) -> ActionScale:
        """This suite's metres/radians per action unit, read once and cached.

        Returns:
            The calibration at action_scale_path, else action_scale.json beside the graph artifacts.

        Raises:
            FileNotFoundError: No calibration exists there. Never defaulted: a wrong scale does not
                fail, it silently mis-drives the arm.
        """
        if self._scale is None:
            path = self.cfg.action_scale_path or os.path.join(
                os.fspath(self.cfg.graph_dir or default_paths().graph(self.suite())), SCALE_JSON)
            if not os.path.exists(path):
                suite = self.suite() or "object"
                raise FileNotFoundError(
                    f"no action-unit calibration at {path}: the blend converts metres and radians "
                    f"into action units and has no safe default. Fit one with "
                    f"'python -m robot_ood.cli build calibrate --suite {suite}', or point SN_ACTION_SCALE at an "
                    f"existing {SCALE_JSON}."
                )
            self._scale = ActionScale.from_json(path)
        return self._scale
