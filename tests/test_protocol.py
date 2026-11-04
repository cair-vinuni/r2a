"""Rollout protocols: the vendor recipes the harness replays."""

import pytest

from robot_ood.core.protocol import (BENCH_STEP_LIMIT, BINARIZE, FIXED, NATIVE, NATIVE_SIZE, RANDOM,
                                     get_protocol, protocol_ids)


def test_the_seven_vendor_protocols_are_declared():
    assert protocol_ids() == ["gr00t_aloha", "gr00t_vrh31", "nvidia", "openpi", "openvla_oft", "pi05_robotwin",
                              "vlact_robotwin"]


def test_pi05_robotwin_is_the_xpolicylab_rollout():
    p = get_protocol("pi05_robotwin")
    assert (p.init_states, p.settle_steps, p.seed) == (RANDOM, 0, 0)
    assert p.step_budget("g1") == BENCH_STEP_LIMIT == 0      # the benchmark's own per-task limit
    assert (p.wire_image_size, p.exec_horizon, p.gripper) == (NATIVE_SIZE, 50, NATIVE)


def test_openvla_oft_is_stablevlas_own_rollout():
    p = get_protocol("openvla_oft")
    assert (p.init_states, p.settle_steps, p.seed) == (FIXED, 10, 7)
    assert p.step_budget("libero_10") == 520 and p.step_budget("libero_spatial") == 220
    assert (p.wire_image_size, p.exec_horizon, p.gripper) == (224, 8, BINARIZE)


def test_nvidia_is_gr00ts_own_rollout():
    p = get_protocol("nvidia")
    assert (p.init_states, p.settle_steps) == (RANDOM, 0)
    assert all(p.step_budget(s) == 504 for s in ("libero_10", "libero_goal", "libero_object"))
    assert (p.wire_image_size, p.exec_horizon) == (256, 8)


def test_openpi_is_pi05s_own_rollout():
    p = get_protocol("openpi")
    assert (p.init_states, p.settle_steps, p.seed) == (FIXED, 10, 7)
    assert p.step_budget("libero_10") == 520
    assert (p.exec_horizon, p.gripper) == (5, NATIVE)


def test_every_protocol_cites_a_vendor_file():
    for pid in protocol_ids():
        assert get_protocol(pid).source.endswith(".py")


def test_an_unknown_protocol_names_the_known_ones():
    with pytest.raises(KeyError, match="openvla_oft"):
        get_protocol("nonesuch")


def test_a_missing_budget_is_an_error():
    with pytest.raises(KeyError, match="libero_99"):
        get_protocol("openpi").step_budget("libero_99")
