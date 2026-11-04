"""Embed one suite's task instructions into g_lang.npz, and gate the result before writing it.

The artifact is written outside ArtifactSchemas.NODES, so an existing g_nodes.npz stays valid and a
graph without g_lang.npz simply runs with the language prior switched off.
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_ood.data.libero import demo_files, task_name_of, task_phrasings
from robot_ood.data.store import record_artifacts
from robot_ood.data.nodes import NodeTable
from robot_ood.models.language.encoder import InstructionEncoder, TaskEmbeddings, normalise_instruction

__all__ = ["DEFAULT_MAX_CROSS", "SeparationReport", "build_task_embeddings", "measure_separation",
           "save_task_embeddings"]

# Above this mean off-diagonal cosine the embeddings carry no usable contrast. Centred Qwen scores
# well under it on long; uncentred it scores ~0.95 and the whole mechanism is inert.
DEFAULT_MAX_CROSS = 0.75


@dataclass(frozen=True)
class SeparationReport:
    """Both pre-registered gates, measured on one set of embeddings.

    Attributes:
        cross_mean: Mean cosine between two different tasks.
        cross_max: The largest such cosine.
        nearest: Per task, the index of its nearest OTHER task.
        max_cross: The separation gate this was measured against.
        separated: cross_mean is under the gate.
        distinct: Every task is its own nearest neighbour.
    """

    cross_mean: float
    cross_max: float
    nearest: np.ndarray
    max_cross: float
    separated: bool
    distinct: bool

    @property
    def passed(self) -> bool:
        """Whether both gates hold."""
        return self.separated and self.distinct

    def lines(self, emb: TaskEmbeddings) -> list[str]:
        """The full report, one line per element.

        Args:
            emb: The embeddings this report was measured on.

        Returns:
            The lines, in print order.
        """
        sim = emb.e_task @ emb.e_task.T
        out = [f"encoder    : {emb.model_id}",
               f"tasks      : {emb.n_tasks}",
               f"cross-task cosine: mean {self.cross_mean:+.4f}  max {self.cross_max:+.4f}  "
               f"(gate < {self.max_cross})",
               "",
               "nearest other task per task (the pairs this prior cannot separate):"]
        for i, name in enumerate(emb.task_names):
            j = int(self.nearest[i])
            out.append(f"  [{i}] {normalise_instruction(name)[:58]:58s} -> [{j}] cos {sim[i, j]:+.3f}")
        if not self.separated:
            out += ["", f"GATE FAILED: mean cross-task cosine {self.cross_mean:.4f} >= "
                        f"{self.max_cross}. The task embeddings are not separated; check that "
                        "centering was applied."]
        if not self.distinct:
            out += ["", "GATE FAILED: some task is not its own nearest neighbour -- two task "
                        "names collide."]
        return out


def measure_separation(emb: TaskEmbeddings, max_cross: float = DEFAULT_MAX_CROSS) -> SeparationReport:
    """Measure both gates on one set of embeddings.

    Args:
        emb: The freshly built embeddings.
        max_cross: Separation gate.

    Returns:
        The report.
    """
    sim = emb.e_task @ emb.e_task.T
    off = ~np.eye(emb.n_tasks, dtype=bool)
    nearest = np.argmax(np.where(off, sim, -np.inf), axis=1)
    return SeparationReport(
        cross_mean=float(sim[off].mean()), cross_max=float(sim[off].max()), nearest=nearest,
        max_cross=max_cross, separated=float(sim[off].mean()) < max_cross,
        distinct=bool(np.all(np.diag(sim) > sim[np.arange(emb.n_tasks), nearest])),
    )


def build_task_embeddings(graph_dir: str | Path, hdf5_dir: str | Path, *, model_dir: str = "",
                          device: str = "cpu") -> TaskEmbeddings:
    """Encode each task's recorded phrasings with the frozen local encoder and fit the task head.

    Args:
        graph_dir: Directory holding g_nodes.npz.
        hdf5_dir: The demo corpus the graph was built from, one file per task.
        model_dir: Encoder snapshot; empty means the local Qwen2.5-0.5B.
        device: Torch device the encoder runs on.

    Returns:
        The embeddings, not yet written.

    Raises:
        ValueError: A task of the graph has no demo file in hdf5_dir.
    """
    nodes = NodeTable.load(Path(graph_dir) / "g_nodes.npz")
    files = {task_name_of(f): f for f in demo_files(hdf5_dir)}
    missing = [n for n in nodes.task_names if n not in files]
    if missing:
        raise ValueError(f"{hdf5_dir} has no demo file for {missing[:3]}...")
    phrasings = [task_phrasings(files[n]) for n in nodes.task_names]
    encoder = InstructionEncoder(model_dir=model_dir, device=device or "cpu")
    return TaskEmbeddings.build(nodes.task_names, encoder, phrasings)


def save_task_embeddings(emb: TaskEmbeddings, destination: str | Path) -> Path:
    """Write the embeddings and record them in the artifact ledger.

    Args:
        emb: What to write.
        destination: The output path, or the directory to write LANG_NPZ into.

    Returns:
        The path written.
    """
    written = emb.save(Path(destination))
    record_artifacts(written.parent, stage="lang", command=shlex.join(sys.argv),
                     filenames=[written.name])
    return written
