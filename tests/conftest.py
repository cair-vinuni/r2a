"""Shared skip guard for the tests that need a TRAINED head, not just a built graph.

It lives here because two files need it. graph_hash covers the geometry, the C0 constants beside
the graph and the head's own C2 tensors, so the check must be built the same way
robot_ood.runtime.retrieve.GraphRetriever.load builds it -- a geometry-only hash never matches a
stamped one, and the tests would skip forever.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.data.geometry import CONSTANTS_JSON, graph_hash
from robot_ood.data.nodes import NodeTable
from robot_ood.models.retrieval.gnn import C2_PREFIX

RETRAIN_CMD = "python -m robot_ood.cli train locator --suite long"


def _head_matches_graph(graph_dir: Path) -> bool:
    """Whether the head beside graph_dir was trained on the graph currently beside it.

    Args:
        graph_dir: An artifact directory.

    Returns:
        True when the head's stamped graph_hash equals the one GraphRetriever.load recomputes.
    """
    graph_dir = Path(graph_dir)
    head_path = graph_dir / schema.HEAD_NPZ
    if not head_path.exists():
        return False
    with np.load(head_path) as raw:
        if "graph_hash" not in raw.files:
            return False
        stamped = str(raw["graph_hash"])
        c2 = {k: raw[k] for k in raw.files if k.startswith(C2_PREFIX)}
    want = graph_hash(
        NodeTable.load(graph_dir / schema.NODES_NPZ),
        EdgeSet.load(graph_dir / schema.EDGES_NPZ),
        constants=graph_dir / CONSTANTS_JSON,
        c2_state=c2 or None,
    )
    return stamped == want


def skip_if_head_is_stale(graph_dir: Path) -> None:
    """Skip the calling test when graph_dir's head predates this branch's graph_hash."""
    if not _head_matches_graph(graph_dir):
        pytest.skip(
            f"{graph_dir}/{schema.HEAD_NPZ} was trained against a different graph_hash than the one "
            f"this branch computes (graph_hash v2 + two appended relations). Re-run S1 -- "
            f"`{RETRAIN_CMD}` -- and re-capture tests/golden/parity.json."
        )


def fcsr_net(*, hidden: int = 8, chunk: int = 4, k: int = 3, **children):
    """A complete FcsrNet, with any child replaced by a keyword.

    Args:
        hidden: Width for the children built here.
        chunk: The corrector's chunk length.
        k: The corrector's candidate count.
        **children: seg / scorer / corrector overrides.

    Returns:
        The net.
    """
    from robot_ood.core.schemas import ACTION_EMBED_DIM, SEGMENT_LEN
    from robot_ood.models.fcsr_net import FcsrNet, SegEncoders
    from robot_ood.models.fusion.corrector import Corrector
    from robot_ood.models.retrieval.encoders import QueryEncoder, SegmentKeyEncoder
    from robot_ood.models.retrieval.scorer import ActionEncoder, SegmentScorer

    seg = children.get("seg") or SegEncoders(
        QueryEncoder(dim=7, hist=3, hidden=hidden, lang_dim=0),
        SegmentKeyEncoder(hidden),
        ActionEncoder(SEGMENT_LEN, out_dim=ACTION_EMBED_DIM),
    )
    return FcsrNet(seg,
                   children.get("scorer") or SegmentScorer(hidden=hidden),
                   children.get("corrector") or Corrector(chunk, k=k, hidden=hidden))


def write_fcsr_archive(tmp_path, net=None, *, graph_hash: str = "h", mutate=None, **children):
    """Write one FcsrNet to <tmp_path>/g_fcsr.npz, the file deploy reads.

    Args:
        tmp_path: Directory to write into.
        net: The net; None builds one from the children keywords.
        graph_hash: The stamp to write.
        mutate: Optional callable given the arrays before they are written.
        **children: Passed to fcsr_net when net is None.

    Returns:
        The written path.
    """
    from robot_ood.core import npz as npz_io

    arrays = (net or fcsr_net(**children)).arrays(graph_hash=graph_hash)
    if mutate is not None:
        mutate(arrays)
    return npz_io.write_arrays(tmp_path, schema.FCSR_NPZ, arrays)


def fcsr_round_trip(child, tmp_path, attr: str, *, mutate=None, device="cpu"):
    """Write one child inside a whole archive and read that child back.

    The three share a single checkpoint, so a child's persistence is exercised through the archive
    it actually lands in.

    Args:
        child: The module to write.
        tmp_path: Directory to write the archive into.
        attr: Which child this is -- seg, scorer or corrector.
        mutate: Optional callable given the arrays before they are written.
        device: Torch device to rebuild on.

    Returns:
        The rebuilt child.
    """
    from robot_ood.models.fcsr_net import FcsrNet

    path = write_fcsr_archive(tmp_path, mutate=mutate, **{attr: child})
    return getattr(FcsrNet.load(path, device), attr)
