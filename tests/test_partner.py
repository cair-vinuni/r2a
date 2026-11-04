import math

import numpy as np
import pytest

from robot_ood.core.constants import STILL_TOL
from robot_ood.runtime.partner import (PRIOR_FLOOR, PartnerPrior, frame_phase, lagged, moved_frames, moving_frames,
                                       moving_threshold, node_spans, pause_span)

STEP = 10 * STILL_TOL


def _walk(n: int, moving: np.ndarray) -> np.ndarray:
    q = np.zeros((n, 7))
    q[1:, 0] = np.cumsum(np.where(moving[1:], STEP, 0.0))
    return q


def _prior(span: int = 1) -> PartnerPrior:
    spans = {"left": np.array([[0, 4], [4, 8], [8, 10]]), "right": np.array([[0, 5], [5, 10]])}
    moving = {"right": np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=bool),
              "left": np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)}
    phase = {"right": np.array([0, 0, 0, 0, 0.2, 0.4, 0.6, 0.7, 0.7, 0.7]),
             "left": np.array([0, 0, 0.3, 0.6, 1, 1, 1, 1, 1, 1])}
    return PartnerPrior(spans, moving, phase, {"left": 0.1, "right": 0.1}, {"left": span, "right": span}, window=2)


def _feed(prior: PartnerPrior, rows: list, phase: dict) -> None:
    for row in rows:
        prior.observe({"left": np.full(7, row), "right": np.full(7, row)})
    prior.observe_phase(phase)


def test_moved_frames_reset_at_every_demo_start():
    q = _walk(6, np.array([0, 1, 1, 0, 1, 1], dtype=bool))
    moved = moved_frames(q, np.array([0, 3, 6]))
    assert moved.tolist() == [False, True, True, False, True, True]


def test_moving_threshold_is_a_low_moving_displacement():
    moving = np.tile(np.array([1] * 10 + [0] * 10, dtype=bool), 5)
    q = _walk(100, moving)
    moved = moved_frames(q, np.array([0, 100]))
    assert math.isclose(moving_threshold(q, moved, np.array([0, 100]), 10), 9 * STEP)


def test_pause_span_covers_the_pauses_a_moving_demo_arm_takes():
    moved = np.array([True] * 5 + [False] * 25 + [True] * 5)
    assert pause_span(moved, np.array([0, 35]), 10) == 3
    assert pause_span(np.ones(10, dtype=bool), np.array([0, 10]), 10) == 1


def test_node_spans_tile_the_episodes():
    spans = node_spans(np.array([0, 5, 12, 20, 25]), np.array([0, 0, 0, 1, 1]), np.array([0, 20, 30]))
    assert spans.tolist() == [[0, 5], [5, 12], [12, 20], [20, 25], [25, 30]]
    with pytest.raises(ValueError, match="tile"):
        node_spans(np.array([0, 5, 12, 21, 25]), np.array([0, 0, 0, 1, 1]), np.array([0, 20, 30]))


def test_frame_phase_lags_by_one_window_within_each_episode():
    spans = node_spans(np.array([0, 2, 4, 6]), np.array([0, 0, 1, 1]), np.array([0, 4, 8]))
    phase = frame_phase(np.array([0.0, 0.5, 0.0, 0.5]), spans)
    assert phase.tolist() == [0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5]
    assert lagged(phase, np.array([0, 4, 8]), 2).tolist() == [0, 0, 0, 0, 0, 0, 0, 0]
    assert lagged(phase, np.array([0, 4, 8]), 1).tolist() == [0, 0, 0, 0.5, 0, 0, 0, 0.5]


def test_the_corpus_detector_and_the_live_detector_agree():
    moving = np.array([0] * 12 + [1] * 10 + [0] * 8 + [1] * 10, dtype=bool)
    q = _walk(40, moving)
    frames = moving_frames(q, np.array([0, 40]), 5, 2, 2.5 * STEP)
    prior = PartnerPrior({"a": np.zeros((1, 2), dtype=np.int64), "b": np.zeros((1, 2), dtype=np.int64)},
                         {"a": frames, "b": frames}, {"a": np.zeros(40), "b": np.zeros(40)},
                         {"a": 2.5 * STEP, "b": 2.5 * STEP}, {"a": 2, "b": 2}, window=5)
    live = []
    for row in q:
        prior.observe({"a": row, "b": row})
        live.append(prior.moving("a"))
    assert live[:5] == [None] * 5
    assert live[5:] == frames[5:].tolist()
    assert not frames[:14].any() and frames[14:29].all() and not frames[29:32].any() and frames[32:].all()


def test_no_prior_before_the_partner_has_a_window_and_a_phase():
    prior = _prior()
    assert prior.prior("left") is None
    _feed(prior, [0.0, 0.0, 0.0], {})
    assert prior.prior("left") is None
    prior.observe_phase({"right": 0.0})
    assert prior.prior("left") is not None


def test_a_moving_partner_prefers_the_segments_where_it_moved_at_that_phase():
    prior = _prior()
    _feed(prior, [0.0, 0.0, 0.2], {"right": 0.6})
    left = prior.prior("left")
    assert int(np.argmax(left)) == 1
    assert left[0] == pytest.approx(math.log(PRIOR_FLOOR)) and left[2] == pytest.approx(math.log(PRIOR_FLOOR))
    assert left[1] == pytest.approx(math.log(np.mean(np.exp(-0.5 * ((np.array([0.2, 0.4, 0.6, 0.7]) - 0.6) / 0.05) ** 2))))


def test_a_still_partner_prefers_the_segments_where_it_stayed_at_that_phase():
    prior = _prior()
    _feed(prior, [0.0, 0.0, 0.0], {"right": 0.0})
    assert int(np.argmax(prior.prior("left"))) == 0
    prior.observe_phase({"right": 0.7})
    assert int(np.argmax(prior.prior("left"))) == 2


def test_each_arm_reads_its_own_partner():
    prior = _prior()
    _feed(prior, [0.0, 0.0, 0.2], {"left": 0.6, "right": 0.6})
    assert int(np.argmax(prior.prior("right"))) == 0


def test_a_pause_shorter_than_the_span_keeps_the_partner_moving():
    rows = [0.0, 0.0, 0.2, 0.2, 0.2]
    long, short = _prior(span=2), _prior(span=1)
    _feed(long, rows, {"right": 0.6})
    _feed(short, rows, {"right": 0.6})
    assert long.moving("right") and not short.moving("right")


def test_reset_forgets_the_episode():
    prior = _prior()
    _feed(prior, [0.0, 0.0, 0.2], {"right": 0.6})
    prior.reset()
    assert prior.moving("right") is None and prior.prior("left") is None


def test_a_prior_names_exactly_two_arms():
    with pytest.raises(ValueError, match="exactly two arms"):
        PartnerPrior({"a": np.zeros((1, 2))}, {"a": np.zeros(1, dtype=bool)}, {"a": np.zeros(1)}, {"a": 0.1},
                     {"a": 1}, window=1)
    with pytest.raises(ValueError, match="exactly two lanes"):
        PartnerPrior.from_lanes({"a": "."}, window=1)
