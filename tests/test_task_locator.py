from __future__ import annotations

import numpy as np

from robot_ood.train.tasks.locator import LocatorTask
from robot_ood.train.types import TrainGraphSpec


def _task(**kw):
    return LocatorTask(None, None, None, None, None, np.array([0]), np.array([1]),
                       epochs=1, n_rows=1, **kw)


def test_the_graph_spec_carries_the_no_memory_ablation():
    assert TrainGraphSpec().no_memory is False
    assert TrainGraphSpec(no_memory=True).no_memory is True


def test_the_locator_task_records_the_no_memory_ablation():
    assert _task().no_memory is False
    assert _task(no_memory=True).no_memory is True
    assert _task(single_check=True).no_memory is False
