"""The LIBERO demonstration corpus: which files, which demos, in which order, and what was read.

Five callers used to open these files themselves, each deciding demo order for itself. That order
is not cosmetic: node ids, owner ids and every artifact indexed by them are positional, so two
callers disagreeing means two artifacts that cannot be used together.

DemoCorpus below binds one suite's demo directory to the graph built from it and fingerprints
both, so a constants.json can be attributed to the bytes it came from.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Final, Iterator, Sequence

import numpy as np

from robot_ood.core.action import RESPONSE_TAPS, DemoDeltas
from robot_ood.data.geometry import graph_hash
from robot_ood.core.paths import Paths, default_paths
from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.data.nodes import NodeTable
from robot_ood.data.store import sha256_file

DEMO_GLOB = "*.hdf5"
TASK_NAME_SUFFIX = "_demo.hdf5"
DATA_GROUP = "data"

# Suite short-name -> the name passed to Paths.hdf5. long maps to LIBERO's
# historical libero_10, not libero_long.
SUITE_HDF5_DIRS: Final[dict[str, str]] = {
    "object": "libero_object",
    "spatial": "libero_spatial",
    "goal": "libero_goal",
    "long": "libero_10",
}

# Default suite when none is supplied.
DEFAULT_SUITE: Final[str] = "object"


def hdf5_name(suite: str) -> str:
    """suite -> the name passed to Paths.hdf5.

    Args:
        suite: Suite short-name, or an hdf5 dirname, or "".

    Returns:
        The dirname to pass to Paths.hdf5.
    """
    # A suite this table doesn't know about falls through unchanged, so a caller may also hand this
    # an already-correct hdf5 dirname directly.
    return SUITE_HDF5_DIRS.get(suite, suite or SUITE_HDF5_DIRS[DEFAULT_SUITE])


def demo_files(directory: str | os.PathLike[str]) -> list[Path]:
    """Every demonstration file in a corpus directory, in the order task ids are assigned.

    Args:
        directory: The corpus directory.

    Returns:
        The files, sorted by name.

    Raises:
        FileNotFoundError: The directory holds no demonstration files.
    """
    found = sorted(Path(directory).glob(DEMO_GLOB))
    if not found:
        raise FileNotFoundError(f"no {DEMO_GLOB} files found under {directory}")
    return found


def task_phrasings(path: str | os.PathLike[str]) -> list[str]:
    """The phrasings one demonstration file records for its task: its instruction pool, or its instruction.

    Args:
        path: The .hdf5 file.

    Returns:
        The phrasings, at least one.

    Raises:
        ValueError: The file records no problem_info.
    """
    import json

    with open_demos(path) as data:
        raw = data.attrs.get("problem_info")
        if raw is None:
            raise ValueError(f"{path}: HDF5 data group has no problem_info attribute")
        info = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    pool = [str(t) for t in info.get("instructions", [])]
    return pool or [str(info["language_instruction"]).strip()]


def task_name_of(path: str | os.PathLike[str]) -> str:
    """The task a demonstration file names: its filename without the demo suffix."""
    name = Path(path).name
    return name[: -len(TASK_NAME_SUFFIX)] if name.endswith(TASK_NAME_SUFFIX) else Path(path).stem


@contextmanager
def open_demos(path: str | os.PathLike[str]) -> Iterator[Any]:
    """Open one demonstration file and yield its data group, closing the handle on the way out.

    Args:
        path: The .hdf5 file.

    Yields:
        The h5py group holding one demo per key.
    """
    import h5py  # deferred: only the offline builders need it

    with h5py.File(os.fspath(path), "r") as handle:
        yield handle[DATA_GROUP]


@contextmanager
def open_hdf5(path: str | os.PathLike[str]) -> Iterator[Any]:
    """Open any hdf5 file read-only and yield its root, closing it on the way out.

    Args:
        path: The file; a benchmark's own recording a converter reads, so h5py keeps one home.

    Yields:
        The h5py file handle.
    """
    import h5py  # deferred: only the offline builders need it

    with h5py.File(os.fspath(path), "r") as handle:
        yield handle


def write_demo_file(path: str | os.PathLike[str], instruction: str,
                    demos: Iterator[dict[str, Any]], phrasings: Sequence[str] = ()) -> int:
    """Write one task's demonstration file in the layout open_demos reads back.

    Args:
        path: The .hdf5 file to create.
        instruction: The language instruction recorded in problem_info.
        demos: Per demo, a mapping of obs dataset name to array plus an "actions" array.
        phrasings: Every phrasing the task's demonstrations were recorded under, when the
            benchmark records more than the instruction.

    Returns:
        The number of demos written.
    """
    import json

    import h5py  # deferred: only the offline builders need it

    info: dict[str, Any] = {"language_instruction": instruction}
    if phrasings:
        info["instructions"] = [str(t) for t in phrasings]
    count = 0
    with h5py.File(os.fspath(path), "w") as handle:
        data = handle.create_group(DATA_GROUP)
        data.attrs["problem_info"] = json.dumps(info)
        for count, demo in enumerate(demos, start=1):
            group = data.create_group(f"demo_{count - 1}")
            obs = group.create_group("obs")
            for key, value in demo.items():
                if key == "actions":
                    group.create_dataset(key, data=value)
                elif value.ndim == 4:
                    obs.create_dataset(key, data=value, compression="gzip",
                                       compression_opts=4, chunks=(1, *value.shape[1:]))
                else:
                    obs.create_dataset(key, data=value)
    return count


def demo_keys(group: Any) -> list[str]:
    """The demo keys of one data group, in h5py's alphabetical order.

    Args:
        group: A data group from open_demos.

    Returns:
        The keys -- alphabetical, the order every shipped artifact's owner ids were indexed by.
    """
    return list(group.keys())


def _stamped_sha256(path: Path) -> str:
    """sha256 of one file, prefixed with its algorithm the way constants.json records it."""
    return f"sha256:{sha256_file(path)}"


# ==== one suite's corpus: the demos, the graph built from them, and the fingerprint of both ====

# The HDF5 datasets the action-unit fit reads per demo group. Nothing else in the group is touched.
_POS_KEY = "obs/ee_pos"
_ORI_KEY = "obs/ee_ori"
_ACT_KEY = "actions"


@dataclass(frozen=True, slots=True)
class PooledDeltas:
    """Every transition of one suite, plus what was read to get them.

    Attributes:
        deltas: Pooled per-transition commands and measured pose changes.
        n_files: HDF5 files read.
        n_demos: Demo groups read.
    """

    deltas: DemoDeltas
    n_files: int
    n_demos: int


class DemoCorpus:
    """One suite's demo HDF5s and built graph, resolved and read lazily.

    Args:
        suite: Suite short-name (object | spatial | goal | long), or an hdf5 dirname.
        hdf5_dir: Explicit demo directory, bypassing the suite mapping.
        graph_dir: Explicit graph-artifact directory, bypassing Paths.graph.
        limit_tasks: Cap on HDF5 files read. None means all.
        limit_demos: Cap on demo groups per file. None means all.
        paths: Path resolver. None uses default_paths().
    """

    def __init__(
        self,
        suite: str,
        *,
        hdf5_dir: str | Path | None = None,
        graph_dir: str | Path | None = None,
        limit_tasks: int | None = None,
        limit_demos: int | None = None,
        paths: Paths | None = None,
    ) -> None:
        resolver = paths or default_paths()
        self.suite = suite
        self.hdf5_dir = Path(hdf5_dir) if hdf5_dir is not None else Path(resolver.hdf5(hdf5_name(suite)))
        self.graph_dir = Path(graph_dir) if graph_dir is not None else Path(resolver.graph(suite))
        self.limit_tasks = limit_tasks
        self.limit_demos = limit_demos

    # ---- inputs --------------------------------------------------------------------------------
    @property
    def pose_space(self) -> str:
        """How this corpus's poses compose (configs/trees/<suite>.yaml pose_space); SO3 by default."""
        from robot_ood.core.action import pose_space_for

        return pose_space_for(self.suite)

    @cached_property
    def hdf5_files(self) -> tuple[Path, ...]:
        """The demo files this corpus reads, in sorted order.

        Raises:
            FileNotFoundError: No *.hdf5 files exist under hdf5_dir.
        """
        return tuple(demo_files(self.hdf5_dir)[: self.limit_tasks])

    @cached_property
    def nodes(self) -> NodeTable:
        """The built graph's node table.

        Raises:
            FileNotFoundError: The suite has no built graph.
        """
        path = self.graph_dir / schema.NODES_NPZ
        if not path.is_file():
            raise FileNotFoundError(
                f"{path}: no built graph for suite {self.suite!r} -- run "
                f"`python -m robot_ood.cli build graph --suite {self.suite}` before calibrating."
            )
        return NodeTable.load(path)

    @cached_property
    def head_path(self) -> Path | None:
        """The trained retrieval head beside the graph, or None when the graph is untrained."""
        path = self.graph_dir / schema.HEAD_NPZ
        return path if path.is_file() else None

    def pooled_deltas(self) -> PooledDeltas:
        """Difference every demo of every file into (action, measured pose change) transitions.

        Returns:
            The pooled transitions and the file/demo counts behind them; a demo too short to
            carry a full RESPONSE_TAPS history contributes nothing.
        """
        parts: list[DemoDeltas] = []
        for path in self.hdf5_files:
            with open_demos(path) as data:
                for key in demo_keys(data)[: self.limit_demos]:
                    group = data[key]
                    if group[_POS_KEY].shape[0] <= RESPONSE_TAPS:
                        continue
                    parts.append(
                        DemoDeltas.from_demo(
                            np.asarray(group[_POS_KEY]),
                            np.asarray(group[_ORI_KEY]),
                            np.asarray(group[_ACT_KEY]),
                            self.pose_space,
                        )
                    )
        return PooledDeltas(
            deltas=DemoDeltas.concat(parts), n_files=len(self.hdf5_files), n_demos=len(parts)
        )

    # ---- provenance ----------------------------------------------------------------------------
    def fingerprint(self) -> dict[str, Any]:
        """What this corpus is, as the corpus block of constants.json.

        Returns:
            The corpus block: suite, both directories, corpus sizes and the input fingerprint.
        """
        nodes = self.nodes
        edges = EdgeSet.load(self.graph_dir / schema.EDGES_NPZ)
        demo_manifest = "".join(f"{p.name}:{p.stat().st_size};" for p in self.hdf5_files)
        inputs = {
            "hdf5_manifest": f"sha256:{hashlib.sha256(demo_manifest.encode()).hexdigest()}",
            schema.NODES_NPZ: _stamped_sha256(self.graph_dir / schema.NODES_NPZ),
        }
        if self.head_path is not None:
            inputs[schema.HEAD_NPZ] = _stamped_sha256(self.head_path)
        return {
            "suite": self.suite,
            "hdf5_dir": str(self.hdf5_dir),
            "graph_dir": str(self.graph_dir),
            "n_hdf5_files": len(self.hdf5_files),
            "n_demos": int(nodes.n_demos),
            "n_nodes": int(len(nodes)),
            "n_tasks": int(nodes.n_tasks),
            "graph_hash": graph_hash(nodes, edges),
            "inputs": inputs,
        }
