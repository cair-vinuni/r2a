"""The servo's two motions under the robot's limits, and the star search on a synthetic bowl."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from robot_ood.servo.limits import WAIST_RANGE, WAIST_RATE, walk_velocity
from robot_ood.servo.motion import CONTROL_PERIOD, UNIT, WaistMotion, WalkMotion

_spec = importlib.util.spec_from_file_location(
    "servo_cases", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "servo_cases.py")
cases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cases)


def _place(u):
    return {"u": np.asarray(u, float)}


def test_walk_command_below_the_deadband_does_not_move_the_base():
    assert np.all(walk_velocity(np.array([0.05, -0.07, 0.13])) == 0.0)
    assert np.all(walk_velocity(np.array([0.06, -0.10, 0.14])) != 0.0)


def test_walk_command_is_clipped_to_the_command_range():
    np.testing.assert_allclose(walk_velocity(np.array([5.0, 5.0, 5.0])), walk_velocity(np.array([1.2, 0.5, 1.0])))


def test_waist_never_moves_the_legs_and_respects_its_rate():
    motion = WaistMotion(_place, WaistMotion.default_box())
    poses = [motion.step(np.array([3.0, -3.0, 20.0]))["u"] for _ in range(100)]
    poses = np.array(poses)
    assert np.all(poses[:, :2] == 0.0)
    assert np.abs(np.diff(poses[:, 2])).max() <= WAIST_RATE * CONTROL_PERIOD + 1e-9
    assert poses[-1, 2] == pytest.approx(WAIST_RANGE)


def test_walk_arrives_without_hunting():
    motion = WalkMotion(_place, WalkMotion.default_box())
    target = np.array([1.0, -0.5, 0.3])
    for _ in range(200):
        motion.step(target)
        if motion.stopped_at(target):
            break
    assert motion.stopped_at(target)


@pytest.mark.parametrize("case", ["xoay_phai_30do", "xoay_trai_30do"])
def test_waist_servo_undoes_a_turn(case):
    shift = np.array(cases.CASES[case])
    last = cases.run("waist", "decoupled", shift, seed=0)
    assert last["stop"] == "certified"
    assert last["energy"] <= last["tau"]
    # The servo stops at the first certified reading, which may sit on the rim of the bowl.
    assert last["u"][2] == pytest.approx(-shift[2], abs=0.1)
    assert last["u"][:2] == [0.0, 0.0]


def test_walk_servo_undoes_a_forward_shift():
    shift = np.array(cases.CASES["lui_xuong_10cm"])
    last = cases.run("walk", "decoupled", shift, seed=0)
    assert last["stop"] == "certified"
    assert np.abs(np.asarray(last["u"]) + shift).max() < 2 * UNIT[0]


@pytest.mark.parametrize("turn", [-0.9, -0.52, -0.26, 0.0, 0.15, 0.26, 0.52, 0.9])
def test_operator_search_finds_the_bottom_of_a_wide_bowl_in_few_reads(turn):
    from robot_ood.servo.session import MAX_READS, WaistSearch

    # A bowl as wide as the finetuned checkpoint's: under tau 0.91 within about 0.2 rad of the bottom.
    def energy(u):
        return 0.6 + 7.0 * (u + turn) ** 2

    search = WaistSearch(tau=0.91)
    while search.next_target() is not None:
        u = search.next_target()
        search.add(u, energy(u))
    assert len(search.u) <= MAX_READS
    assert energy(search.final()) <= 0.91
    if search.stop != "already nominal":
        assert search.final() == pytest.approx(-turn, abs=0.08)
