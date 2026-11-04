"""The pure pieces of robot_ood.build: the parts that were unreachable while they lived in scripts/.

Both were argparse-only code paths before §9 moved them into src/, so neither had ever been called
by a test. Nothing here needs a corpus, a graph or an encoder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_ood.build.language import DEFAULT_MAX_CROSS, measure_separation      # noqa: E402
from robot_ood.build.segments import resolve_segment_len                        # noqa: E402
from robot_ood.core.constants import CONSTANTS_JSON                            # noqa: E402
from robot_ood.core.schemas import SEGMENT_LEN                                # noqa: E402


class _Embeddings:
    """The three fields measure_separation reads off a TaskEmbeddings."""

    def __init__(self, e_task: np.ndarray) -> None:
        """Bind to a row-normalised task matrix.

        Args:
            e_task: [n_tasks, dim], each row unit-norm.
        """
        self.e_task = e_task
        self.n_tasks = e_task.shape[0]
        self.task_names = [f"task {i}" for i in range(self.n_tasks)]
        self.model_id = "test"


def _unit(rows: list[list[float]]) -> np.ndarray:
    """Row-normalise a small matrix, so cosine is a plain dot product."""
    m = np.asarray(rows, dtype=np.float64)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_separation_passes_on_orthogonal_tasks():
    """Orthogonal task embeddings score 0 cross-cosine and every task is its own neighbour."""
    report = measure_separation(_Embeddings(_unit([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))
    assert report.cross_mean == pytest.approx(0.0)
    assert report.separated and report.distinct and report.passed


def test_the_separation_gate_fires_on_near_identical_tasks():
    """This is the gate's whole point: uncentred encoders put every pair near 1.0 and the prior is inert."""
    report = measure_separation(_Embeddings(_unit([[1, 0.02, 0], [1, 0, 0.02], [1, 0.01, 0.01]])))
    assert report.cross_mean > DEFAULT_MAX_CROSS
    assert not report.separated and not report.passed


def test_the_distinctness_gate_fires_when_two_task_names_collide():
    """A task closer to another task than to itself cannot be tilted apart by this mechanism at all."""
    e = _unit([[1, 0, 0], [1, 0, 0], [0, 0, 1]])
    # Two identical rows: self-similarity ties the cross-similarity instead of exceeding it.
    report = measure_separation(_Embeddings(e))
    assert not report.distinct and not report.passed


def test_segment_len_falls_back_to_the_contract_default_with_no_constants(tmp_path):
    """A graph built before calibration still builds segments, at the contract's own H."""
    assert resolve_segment_len(tmp_path) == SEGMENT_LEN


def test_segment_len_prefers_the_graph_s_own_calibrated_value(tmp_path):
    """A calibrated graph must not be segmented at the default -- the table would not match its own corpus."""
    from robot_ood.core.constants import MEASURED, Constant, Constants

    Constants(entries={"segment_len": Constant(11, MEASURED, "test")}).to_json(
        tmp_path / CONSTANTS_JSON)
    assert resolve_segment_len(tmp_path) == 11
