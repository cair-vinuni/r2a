"""Train the locator: load the graph, draw check sequences, run the task, save the head."""

from __future__ import annotations

import os
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from robot_ood.core.config import GraphConfig
from robot_ood.core.constants import HELDOUT_STRIDE
from robot_ood.core.device import resolve_device
from robot_ood.core.kinematics import is_joint_space
from robot_ood.core.protocol import suite_chunk
from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.data.geometry import CONSTANTS_JSON, graph_hash
from robot_ood.data.nodes import NodeTable
from robot_ood.data.segments import graph_plan_rows
from robot_ood.data.store import ArtifactStore, recorded_suite
from robot_ood.models.fusion.target import PolicyActionCache
from robot_ood.models.retrieval.gnn import GraphRetrieverNet
from robot_ood.train.augment import PolicyCorruptor, RetrievalCorruptor
from robot_ood.train.runner import Runner, RunSpec
from robot_ood.train.sequences import SequenceSampler
from robot_ood.train.tasks.locator import LocatorTask
from robot_ood.train.tensors import prepare_tensors
from robot_ood.train.types import TrainGraphSpec

N_QUERY = 4_096
N_EVAL = 128
LR, WEIGHT_DECAY = 1e-3, 1e-5
LOCATOR: Final[str] = "locator"
SINGLE: Final[str] = "single"
TASKS: tuple[str, ...] = (LOCATOR, SINGLE)


def heldout_demos(n_demos: int) -> np.ndarray:
    """Every HELDOUT_STRIDE-th demo."""
    return np.arange(n_demos, dtype=np.int64)[::HELDOUT_STRIDE]


@dataclass
class TrainResult:
    """Outcome of one GraphTrainer.train call.

    Attributes:
        net: The trained retriever.
        graph_hash: Hash of the graph it was trained against.
        history: Per-stage epoch history.
        checkpoint_path: Path where the head was written, or None.
    """

    net: GraphRetrieverNet
    graph_hash: str
    history: dict[str, list[dict]] = dc_field(default_factory=dict)
    checkpoint_path: str | None = None


class GraphTrainer:
    """One training run, start to finish."""

    def __init__(self, graph_dir: str | os.PathLike, spec: TrainGraphSpec | None = None) -> None:
        self.graph_dir = Path(graph_dir)
        self.suite = recorded_suite(self.graph_dir)
        self.spec = spec if spec is not None else TrainGraphSpec()
        self.dev = resolve_device(self.spec.device)
        torch.manual_seed(self.spec.seed)
        self.cfg: GraphConfig = self.spec.cfg or GraphConfig()
        self.nodes: NodeTable | None = None
        self.edges: EdgeSet | None = None
        self.net: GraphRetrieverNet | None = None
        self.language: Any = None
        self._policy_chunk: int | None = None

    def train(self) -> TrainResult:
        """Run the curriculum and, if spec.save, write <graph_dir>/g_head.npz.

        Raises:
            ValueError: spec.stages is anything but (1,), spec.task is unknown, or the corpus is
                too small for the by-demo split.
        """
        if tuple(self.spec.stages) != (1,):
            raise ValueError(f"train_graph: stages={self.spec.stages!r} -- only (1,) is implemented")
        if self.spec.task not in TASKS:
            raise ValueError(f"train_graph: unknown task {self.spec.task!r}; expected one of {TASKS}")
        store = ArtifactStore(self.graph_dir)
        self.nodes, self.edges = store.nodes(), store.edges()
        self.edges.validate(self.nodes)
        self.language = self._resolve_language()
        policy = PolicyActionCache.load(self.graph_dir).check_against(self.nodes)
        train_owners, heldout_owners = self._split_owners()
        self._build_model()

        gt = prepare_tensors(self.nodes, self.edges, self.dev, cfg=self.cfg,
                             language=self.language, chunk=self.net.chunk)
        sampler = SequenceSampler(
            self.nodes, policy, self.language,
            # The locator reads chunks only, so a displaced live pose has no meaning here.
            PolicyCorruptor(self.nodes, RetrievalCorruptor(self.nodes), p_shift=0.0), gt.targets,
            hist=self.cfg.hist, chunk=self.net.chunk, joint_space=is_joint_space(self.suite),
            seed=self.spec.seed,
        )
        task = LocatorTask(
            gt, sampler, self.net, self.nodes, self.edges, train_owners, heldout_owners,
            epochs=self.spec.epochs, n_rows=self._n_query, single_check=self.spec.task == SINGLE,
            no_memory=self.spec.no_memory, seed=self.spec.seed,
        )
        run = Runner(
            RunSpec(epochs=self.spec.epochs, lr=LR, weight_decay=WEIGHT_DECAY,
                    seed=self.spec.seed, save=self.spec.save,
                    # $GR_EVAL_EVERY: the held-out measure is rank 0's alone and every other rank
                    # waits for it, so a long run measures every few epochs instead.
                    eval_every=int(os.environ.get("GR_EVAL_EVERY", "1"))),
            self.spec.logger,
        ).run(task, store, cfg=self.cfg, device=self.dev, evalset=object())
        return TrainResult(
            net=self.net, graph_hash=self._graph_hash(),
            history={"stage1": run.history + [{"epoch": "final", "best_epoch": run.best.epoch,
                                                "best_score": run.best.score, **run.final}]},
            checkpoint_path=os.fspath(self.graph_dir / schema.HEAD_NPZ) if self.spec.save else None,
        )

    def _log(self, msg: str) -> None:
        (self.spec.logger.log(msg) if self.spec.logger is not None else print(msg))

    def _resolve_language(self) -> Any:
        from robot_ood.models.language.encoder import LANG_NPZ, TaskEmbeddings

        if not (self.graph_dir / LANG_NPZ).is_file():
            raise ValueError(
                f"train locator: no {LANG_NPZ} beside the graph. Build it with "
                "`python -m robot_ood.cli build language --suite <suite>` first."
            )
        table = TaskEmbeddings.load(self.graph_dir).check_against(self.nodes.task_names)
        self._log(f"[graph.train] loaded {LANG_NPZ}: {table.n_tasks} tasks, "
                  f"{table.e_aug.shape[0]} bank phrasings")
        return table

    def _split_owners(self) -> tuple[np.ndarray, np.ndarray]:
        heldout_owners = heldout_demos(self.nodes.n_demos)
        train_owners = np.setdiff1d(np.arange(self.nodes.n_demos, dtype=np.int64), heldout_owners)
        if len(train_owners) == 0 or len(heldout_owners) == 0:
            raise ValueError(
                f"train_graph: n_demos={self.nodes.n_demos} too small for a stride-{HELDOUT_STRIDE} "
                "by-demo held-out split (need >= 2 demos)"
            )
        return train_owners, heldout_owners

    def _build_model(self) -> None:
        head = self.graph_dir / schema.HEAD_NPZ
        if head.exists():
            try:
                net = GraphRetrieverNet.load_npz(head).to(self.dev)
            except (KeyError, RuntimeError, ValueError) as exc:
                self._log(f"[graph.train] {head.name} is not loadable by this net ({exc}); fresh start")
            else:
                if net.chunk == self.policy_chunk:
                    self.net = net
                    self._log(f"[graph.train] warm start off {head.name}")
                    return
                self._log(f"[graph.train] {head.name} carries a {net.chunk}-row policy channel and this "
                          f"tree plans {self.policy_chunk} rows; fresh start")
        self.net = self._fresh_retriever()

    def _fresh_retriever(self) -> GraphRetrieverNet:
        net = GraphRetrieverNet(
            dim=self.nodes.dim, hidden=self.cfg.hidden, layers=self.cfg.layers,
            n_rel=schema.N_RELATIONS, agg=self.cfg.agg, n_psi=schema.PSI_FREQS,
            instr_dim=int(np.asarray(self.language.e_task).shape[1]),
            chunk=self.policy_chunk,
        ).to(self.dev)
        net.set_stats(*self.nodes.feature_stats())
        return net

    def _graph_hash(self) -> str:
        return graph_hash(
            self.nodes, self.edges, constants=self.graph_dir / CONSTANTS_JSON,
            c2_state=self.net.c2_state(),
        )

    @property
    def policy_chunk(self) -> int:
        """Rows of one policy chunk a plan covers -- the width of the net's policy channel."""
        if self._policy_chunk is None:
            self._policy_chunk = graph_plan_rows(self.graph_dir, suite_chunk(self.suite))
        return self._policy_chunk

    @property
    def _n_query(self) -> int:
        return N_QUERY if self.spec.n_query is None else self.spec.n_query


def train_graph(graph_dir: str | os.PathLike, spec: TrainGraphSpec | None = None) -> TrainResult:
    """Train GraphRetrieverNet on <graph_dir>/{g_nodes,g_edges}.npz and, if spec.save, write g_head.npz."""
    return GraphTrainer(graph_dir, spec).train()
