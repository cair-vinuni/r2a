"""The rows one plan covers: a divisor of the policy chunk that fits inside a reference segment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_ood.core.config import SentinelConfig, plan_rows
from robot_ood.data.segments import segment_frames


def _tables(spans: list[int]) -> tuple[SimpleNamespace, SimpleNamespace]:
    """A node table and a segment table whose s-th segment spans spans[s] raw frames."""
    starts = np.arange(len(spans), dtype=np.int64)
    t_raw = np.concatenate([starts * 100, starts * 100 + np.asarray(spans, dtype=np.int64)])
    return (SimpleNamespace(t_raw=t_raw),
            SimpleNamespace(start_node=starts, end_node=starts + len(spans)))


def test_segment_frames_is_the_median_span_counted_inclusively():
    assert segment_frames(*_tables([4, 10, 10, 30])) == 11


def test_a_plan_never_outruns_its_segment_and_still_divides_the_chunk():
    # 25 divides RoboTwin's chunk too, but a 13-frame segment stretched over it under-travels.
    assert plan_rows(50, 13) == 10
    assert plan_rows(8, 16) == 8
    assert plan_rows(50, 1) == 1


def test_plan_rows_needs_positive_arguments():
    with pytest.raises(ValueError, match="positive arguments"):
        plan_rows(50, 0)


def test_a_config_read_from_the_environment_holds_any_chunk_the_protocol_names():
    cfg = SentinelConfig.from_env(50)
    assert (cfg.policy_chunk, cfg.plan_rows) == (50, 50)


def test_a_period_that_does_not_divide_the_chunk_is_refused():
    with pytest.raises(ValueError, match="not a multiple"):
        SentinelConfig(policy_chunk=50, plan_rows=8)
