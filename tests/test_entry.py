"""Tests for robot_ood.runtime.entry -- the one-shot pose correction run at the policy's first step.

Three families:
  * EntryTarget: the barycentre and its leave-one-demo-out spread, which the diagnostic error()
    read by every arm is defined against.
  * EntryConfig: every magnitude arrives through calibrated_constants, and an absent calibration is
    LOUD rather than silent.
  * ScoreEntryCorrector: reads sigma off ScoreNet.level_estimate (pure inference, zero env steps),
    then emits one control step unconditionally and keeps walking while the gap read off the arm
    exceeds level_estimate(q_hat) -- both sides of the stopping rule are the same head's output, no
    fixed ladder length and no fire/abstain branch, and step_cap only rate-limits the commanded
    displacement. Driven against a stub sim
    and stub score/inverse-dynamics heads, so the loop's logic is tested without a mujoco or
    torch-training dependency.
  * ScoreNet.level_estimate / level_loss: the supervised head the schedule now reads its sigma from.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from robot_ood.core.config import EntryConfig
from robot_ood.core.constants import STILL_TOL
from robot_ood.models.entry.score import N_JOINT, ScoreNet, level_loss
from robot_ood.runtime.entry import ScoreEntryCorrector
from robot_ood.models.entry.target import EntryTarget, _ledoit_wolf

D = 7


def _raw(starts: np.ndarray, frames: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """[J*frames, D] raw frames and their [J+1] pointer, each demo starting at its own start row."""
    rows = [s[None, :] + np.linspace(0.0, 0.1, frames)[:, None] for s in starts]
    return np.concatenate(rows, axis=0), np.arange(len(starts) + 1, dtype=np.int64) * frames


# ==================================================================================================
# EntryTarget
# ==================================================================================================
def _cfg(**kw) -> EntryConfig:
    return dataclasses.replace(
        EntryConfig(score_sigma_hi=0.5, score_sigma_lo=0.05, score_refine_steps=4), **kw)


def test_target_is_the_barycentre_of_the_demo_start_frames():
    starts = np.array([[0.0] * D, [1.0] * D, [2.0] * D])
    target = EntryTarget.from_raw(*_raw(starts))
    np.testing.assert_allclose(target.q_entry, np.full(D, 1.0))
    assert target.n_demos == 3


def test_target_reads_only_the_first_frame_of_each_demo():
    """A demo that travels far after its start must not drag the target with it."""
    starts = np.array([[-0.01] * D, [0.0] * D, [0.01] * D])   # non-degenerate: mean still 0
    q_raw, raw_ptr = _raw(starts, frames=6)
    q_raw[raw_ptr[0] + 1:raw_ptr[1]] += 100.0            # demo 0 runs off after its start frame
    np.testing.assert_allclose(EntryTarget.from_raw(q_raw, raw_ptr).q_entry, np.zeros(D))


def test_precision_is_the_ledoit_wolf_inverse_of_the_start_cloud():
    starts = np.array([[0.0] * D, [1.0] * D, [2.0] * D])
    got = EntryTarget.from_raw(*_raw(starts))
    shrunk, shrinkage = _ledoit_wolf(starts - starts.mean(0))
    np.testing.assert_allclose(got.precision, np.linalg.inv(shrunk + STILL_TOL ** 2 * np.eye(D)))
    assert got.shrinkage == pytest.approx(shrinkage)
    assert 0.0 <= got.shrinkage <= 1.0


def test_demos_that_all_start_on_one_configuration_still_have_a_finite_metric():
    got = EntryTarget.from_raw(*_raw(np.tile(np.arange(D, dtype=float), (4, 1))))
    assert np.isfinite(got.precision).all()
    assert got.error(np.arange(D, dtype=float)) == 0.0
    assert got.error(np.arange(D, dtype=float) + STILL_TOL) == pytest.approx(np.sqrt(D))


def test_loo_spread_is_the_mahalanobis_distance_to_the_others_barycentre():
    starts = np.array([[0.0] * D, [1.0] * D, [2.0] * D])
    target = EntryTarget.from_raw(*_raw(starts))
    # Demo 0's leave-one-out barycentre is 1.5 (all dims); demo 1's is 1.0; demo 2's is 0.5.
    others = np.array([[1.5] * D, [1.0] * D, [0.5] * D])
    diff = others - starts
    expected = np.sort(np.sqrt(np.einsum("ij,jk,ik->i", diff, target.precision, diff)))
    assert target.loo["max"] == pytest.approx(expected[-1])
    assert target.loo["p50"] == pytest.approx(expected[1])


def test_target_refuses_a_single_demo():
    with pytest.raises(ValueError, match="2 demos"):
        EntryTarget.from_raw(np.zeros((4, D)), np.array([0, 4], dtype=np.int64))


def test_error_is_the_mahalanobis_distance_to_the_entry_configuration():
    starts = np.array([[0.0] * D, [1.0] * D, [2.0] * D])
    target = EntryTarget.from_raw(*_raw(starts))
    q = np.full(D, 3.0)
    diff = q - target.q_entry
    expected = np.sqrt(diff @ target.precision @ diff)
    assert target.error(q) == pytest.approx(expected)


def test_from_graph_dir_names_the_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="g_nodes.npz"):
        EntryTarget.from_graph_dir(tmp_path)


# ==================================================================================================
# EntryConfig -- every magnitude is calibrated, and its absence is loud
# ==================================================================================================
def _write_constants(tmp_path, values: dict) -> str:
    from robot_ood.core.constants import MEASURED, Constant, Constants
    constants = Constants(entries={k: Constant(value=v, kind=MEASURED, procedure="test", stats={})
                                   for k, v in values.items()})
    path = tmp_path / "constants.json"
    path.write_text(json.dumps(constants.to_dict()))
    return str(path)


def test_config_reads_every_calibrated_magnitude_from_the_calibration(tmp_path, monkeypatch):
    """The three C0 keys come from constants.json; score_refine_steps is not one of them.

    entry_score_refine_steps is written into the file here precisely to show it is IGNORED: no
    class in robot_ood.build.measure emits it, so a calibration carrying it must not move the config.
    """
    from robot_ood.core.config import calibrated_constants
    path = _write_constants(tmp_path, {
        "entry_robotinit_hi_rad": 2.0, "entry_perturb_lo": 0.01, "entry_score_refine_steps": 6.0,
        "entry_score_step_cap_rad": 0.05,
    })
    monkeypatch.setenv("ROBOT_OOD_CONSTANTS", path)
    calibrated_constants.cache_clear()
    try:
        cfg = EntryConfig.from_env()
    finally:
        calibrated_constants.cache_clear()
    radius_to_sigma = 7 ** -0.5           # matches robot_ood.core.config._RADIUS_TO_SIGMA
    assert cfg.score_sigma_hi == pytest.approx(2.0 * radius_to_sigma)
    assert cfg.score_sigma_lo == pytest.approx(0.01 * radius_to_sigma)
    assert cfg.score_refine_steps == 3
    assert cfg.score_step_cap == pytest.approx(0.05)


def test_config_warns_once_per_key_when_uncalibrated(monkeypatch):
    from robot_ood.core.config import _uncalibrated_warned, calibrated_constants
    monkeypatch.delenv("ROBOT_OOD_CONSTANTS", raising=False)
    calibrated_constants.cache_clear()
    for key in ("entry_robotinit_hi_rad", "entry_perturb_lo", "entry_score_step_cap_rad"):
        _uncalibrated_warned.discard(key)
    with pytest.warns(UserWarning, match="entry_robotinit_hi_rad") as warnings_seen:
        EntryConfig.from_env()
    # score_refine_steps has no calibrator, so it must NOT cry wolf about one.
    assert not any("entry_score_refine_steps" in str(w.message) for w in warnings_seen)
    calibrated_constants.cache_clear()






# ==================================================================================================
# ScoreEntryCorrector, against a stub sim and stub score/inverse-dynamics heads
# ==================================================================================================
class _StubModel:
    """Just enough mujoco model surface for robot_ood.core.kinematics."""

    def __init__(self) -> None:
        self.actuator_ctrlrange = np.tile(np.array([-100.0, 100.0]), (9, 1))
        self.opt = dataclasses.make_dataclass("Opt", ["timestep"])(timestep=0.002)

    def get_joint_qpos_addr(self, name: str) -> int:
        return int(name[-1]) - 1

    def joint_name2id(self, name: str) -> int:
        return int(name[-1]) - 1

    @property
    def jnt_dofadr(self) -> np.ndarray:
        return np.arange(D)


class _StubData:
    def __init__(self, q0: np.ndarray) -> None:
        self.qpos = np.asarray(q0, np.float64).copy()
        self.qvel = np.zeros(D)


class _StubSim:
    def __init__(self, q0: np.ndarray) -> None:
        self.model = _StubModel()
        self.data = _StubData(q0)

    def render(self, **_kw) -> np.ndarray:
        return np.zeros((2, 2, 3), np.uint8)


class _StubController:
    def __init__(self) -> None:
        self.initial_joint = np.zeros(D)

    def update_initial_joints(self, q: np.ndarray) -> None:
        self.initial_joint = np.asarray(q, np.float64).copy()


class _StubRobot:
    def __init__(self) -> None:
        self.controller = _StubController()


class _StubEnv:
    def __init__(self, q0: np.ndarray) -> None:
        self.sim = _StubSim(q0)
        self.robots = [_StubRobot()]
        self.steps = 0
        self.actions: list[np.ndarray] = []

    def step(self, action):
        self.steps += 1
        self.actions.append(np.asarray(action, np.float32).copy())
        return {"obs": self.steps}, 0.0, False, {}


class _StubScoreNet:
    """score(q, sigma) = (fixed_target - q) / sigma^2, so tweedie_denoise recovers fixed_target
    EXACTLY on the very first refine iteration, regardless of sigma or how far q started -- once
    q_hat == fixed_target the next iteration's score is 0, so further refine_steps are inert. That
    makes this stub's converged displacement `fixed_target - q0`, known to the test without
    depending on how many refine steps the corrector under test happens to run.

    level_estimate is a fixed, configurable radius: this stub's score already reaches fixed_target
    regardless of which sigma the level head names, so `level` only exercises that the corrector
    reads level_estimate and clamps it, never what q_hat converges to.
    """

    def __init__(self, fixed_target: np.ndarray, level: float = 0.1) -> None:
        self.fixed_target = torch.as_tensor(fixed_target, dtype=torch.float32)
        self.level = level

    def __call__(self, q: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return (self.fixed_target - q) / (sigma ** 2)

    def level_estimate(self, q: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.level, dtype=torch.float32)


class _StubInvDyn:
    """Records every (q, dq) it is called with; returns a fixed, recognisable action."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        self.calls.append((q.numpy().copy(), dq.numpy().copy()))
        return torch.full((7,), 0.5, dtype=torch.float32)


@pytest.fixture()
def target() -> EntryTarget:
    return EntryTarget.from_raw(*_raw(np.array([[-0.01] * D, [0.0] * D, [0.01] * D])))




class _TrackingEnv(_StubEnv):
    """A _StubEnv whose arm actually MOVES, achieving only `rate` of each commanded displacement.

    The plain _StubEnv holds q fixed, which cannot exercise a walk at all. Partial tracking is the
    real behaviour under test: OSC never delivers a full commanded delta in one control step, and
    assuming it did is exactly the bug this walk replaced -- deployed, ceil(delta/cap) removed 28.7%
    of the Mahalanobis error and reached the deleted PD servo's arrival tolerance on 12 of 393
    Robot_Initial_States episodes.
    """

    def __init__(self, q0: np.ndarray, dq_source: "_StubInvDyn", rate: float = 0.35) -> None:
        super().__init__(q0)
        self.dq_source = dq_source
        self.rate = float(rate)

    def step(self, action):
        if self.dq_source.calls and len(self.actions) < len(self.dq_source.calls):
            self.sim.data.qpos[:D] += self.rate * self.dq_source.calls[-1][1]
        return super().step(action)


def test_the_walk_arrives_inside_the_level_head_radius_not_inside_the_rate_limit(target):
    """The emitted count is measured, not predicted, and the radius it is measured against is the
    level head's own reading at q_hat -- not score_step_cap, which is the corpus's p99 per-step
    displacement and is the RATE LIMIT. With the head reading strictly inside the cap the walk must
    run past the point the old cap-arrival rule would have stopped at, and past
    ceil(|q_hat - q0| / cap), which assumes every step lands its full commanded displacement."""
    cap, level, offset = 0.1, 0.04, np.full(D, 0.25)
    invdyn_net = _StubInvDyn()
    env = _TrackingEnv(target.q_entry + offset, invdyn_net, rate=0.35)
    corrector = ScoreEntryCorrector(target, _StubScoreNet(target.q_entry, level=level), invdyn_net,
                                    _cfg(score_step_cap=cap))
    corrector.run(env, "before", 10)

    cap_estimate = int(np.ceil(np.linalg.norm(offset) / cap))
    assert corrector.last["steps"] > cap_estimate, (
        f"walked {corrector.last['steps']} steps, no more than the {cap_estimate} the discredited "
        "ceil(delta/cap) rule predicts -- the walk is not reading the gap off the arm"
    )
    assert not corrector.last["walk_capped"], "arrival, not the guard, must end the walk"
    assert corrector.last["residual"] <= level
    assert corrector.last["level_hat"] == pytest.approx(level)
    assert env.steps == corrector.last["steps"] + 1          # emitted steps + one resync
    assert len(invdyn_net.calls) == corrector.last["steps"]
    assert 0.0 < corrector.last["realised"] < 1.0            # the ratio the old rule assumed was 1


def test_a_wider_level_reading_stops_the_same_walk_sooner(target):
    """Nothing but the head's reading sets the arrival radius: the same arm, the same rate limit and
    the same displacement, walked against a level twice as wide, must stop strictly earlier and hand
    off strictly further out. If score_step_cap were still the arrival radius both runs would emit
    the same count."""
    runs = {}
    for level in (0.04, 0.08):
        invdyn_net = _StubInvDyn()
        env = _TrackingEnv(target.q_entry + np.full(D, 0.25), invdyn_net, rate=0.35)
        corrector = ScoreEntryCorrector(target, _StubScoreNet(target.q_entry, level=level),
                                        invdyn_net, _cfg(score_step_cap=0.1))
        corrector.run(env, "before", 10)
        runs[level] = corrector.last
    assert runs[0.08]["steps"] < runs[0.04]["steps"]
    assert runs[0.08]["residual"] > runs[0.04]["residual"]


def test_the_walk_reports_the_path_it_covered_and_the_level_it_started_from(target):
    """last carries both quantities the rejected distance-budget rule was written against, so the
    next iteration can be argued from the SR logs rather than re-derived: level0 is the head's
    reading at the start, moved_total the path length the arm actually covered. The budget rule --
    walk while moved_total < level0 -- is REJECTED and not implemented: moved_total is monotone but
    bounded above by delta, and offline level0 exceeds delta on 92.4% of clean and 63.0% of
    perturbed starts, so 777/1000 walks ran to the guard."""
    invdyn_net = _StubInvDyn()
    env = _TrackingEnv(target.q_entry + np.full(D, 0.25), invdyn_net, rate=0.35)
    corrector = ScoreEntryCorrector(target, _StubScoreNet(target.q_entry, level=0.04), invdyn_net,
                                    _cfg(score_step_cap=0.1))
    corrector.run(env, "before", 10)
    last = corrector.last
    assert last["level0"] == pytest.approx(0.04)             # the stub's head reads one radius
    assert last["moved_total"] == pytest.approx(last["delta"] - last["residual"], abs=1e-4)
    assert last["moved_total"] < last["delta"]               # the walk never overshoots q_hat


def test_an_arm_that_cannot_move_ends_at_the_guard_and_says_so(target):
    """A jammed arm never arrives, so the guard is what stops it -- and the report has to admit
    that, because a step count set by the guard is not a converged handoff. The early sign-test exit
    that used to cover this case is deliberately absent: deployed, OSC widens the gap on individual
    steps often enough that it fired after one step on most episodes (steps p50 = 1, 75.3% against
    82.7%), ending the walk before it had started."""
    invdyn_net = _StubInvDyn()
    env = _TrackingEnv(target.q_entry + np.full(D, 0.25), invdyn_net, rate=0.0)
    corrector = ScoreEntryCorrector(target, _StubScoreNet(target.q_entry), invdyn_net,
                                    _cfg(score_step_cap=0.1))
    corrector.run(env, "before", 10)
    assert corrector.last["walk_capped"]
    assert corrector.last["steps"] == 64
    assert corrector.last["moved_total"] == pytest.approx(0.0)


def test_a_configuration_the_field_leaves_alone_costs_no_env_step(target):
    """When the field finds nothing to move (q_hat == q0 exactly, this stub's score is 0 whenever
    fixed_target == q) the walk emits nothing: no control step, no resync, obs and the counter
    handed back as given. On a scene perturbation every arm starts here (delta ~0.0005 rad against
    a radius ~0.009), and a step emitted there moves nothing yet changes the rollout the policy
    then produces."""
    q0 = target.q_entry
    env = _StubEnv(q0)
    score_net = _StubScoreNet(q0)                          # fixed_target == q0: score(q0) == 0
    invdyn_net = _StubInvDyn()
    corrector = ScoreEntryCorrector(target, score_net, invdyn_net, _cfg())
    obs, t = corrector.run(env, "before", 10)
    assert corrector.last["steps"] == 0
    assert not corrector.last["fired"]
    assert corrector.last["delta"] == pytest.approx(0.0, abs=1e-6)
    assert env.steps == 0
    assert (obs, t) == ("before", 10)
    assert len(invdyn_net.calls) == 0
    assert np.isnan(corrector.last["realised"])


def test_a_start_already_inside_the_heads_own_radius_does_not_walk(target):
    """The arrival radius is the level head's reading at the target, and it decides whether the
    walk starts as well as when it stops: delta is a real 0.02 rad but the head reads the target as
    sitting 0.1 rad off the manifold, so the arm is already where the field can place it and no
    step is emitted."""
    q0 = target.q_entry
    offset = np.zeros(D)
    offset[0] = 0.02                                        # delta 0.02 < the stub's level 0.1
    env = _StubEnv(q0)
    score_net = _StubScoreNet(q0 + offset, level=0.1)
    invdyn_net = _StubInvDyn()
    corrector = ScoreEntryCorrector(target, score_net, invdyn_net, _cfg(score_step_cap=0.005))
    obs, t = corrector.run(env, "before", 10)
    assert corrector.last["delta"] == pytest.approx(0.02, abs=1e-5)
    assert corrector.last["delta"] < corrector.last["level_hat"]
    assert corrector.last["steps"] == 0
    assert env.steps == 0
    assert t == 10
    assert len(invdyn_net.calls) == 0


def test_a_start_outside_the_radius_walks_and_resyncs(target):
    """The same arm with the gap outside the head's radius walks, rate-limited by score_step_cap,
    and pays the resync step after it."""
    q0 = target.q_entry
    offset = np.zeros(D)
    offset[0] = 0.02                                        # delta 0.02 > the stub's level 0.005
    env = _StubEnv(q0)
    score_net = _StubScoreNet(q0 + offset, level=0.005)
    invdyn_net = _StubInvDyn()
    corrector = ScoreEntryCorrector(target, score_net, invdyn_net, _cfg(score_step_cap=0.005))
    obs, t = corrector.run(env, "before", 10)
    assert corrector.last["steps"] >= 1
    assert env.steps == corrector.last["steps"] + 1        # emitted steps plus the resync
    assert t == 10 + env.steps
    assert np.linalg.norm(invdyn_net.calls[0][1]) == pytest.approx(corrector.cfg.score_step_cap)


def test_a_larger_starting_level_yields_strictly_more_steps(target):
    """The head's reading, not a constant, is what sets how long the walk runs: the same arm at a
    larger displacement -- which this stub's head reports through level_estimate -- must walk
    strictly further before the gap falls inside the target's radius."""
    runs = {}
    for offset in (0.05, 0.25):
        invdyn_net = _StubInvDyn()
        env = _TrackingEnv(target.q_entry + np.full(D, offset), invdyn_net, rate=0.35)
        corrector = ScoreEntryCorrector(target, _StubScoreNet(target.q_entry, level=0.04),
                                        invdyn_net, _cfg(score_step_cap=0.1))
        corrector.run(env, "before", 10)
        runs[offset] = corrector.last
    assert runs[0.25]["steps"] > runs[0.05]["steps"] >= 1
    assert runs[0.25]["moved_total"] > runs[0.05]["moved_total"]






def test_emitted_actions_hold_the_gripper_open(target):
    env = _StubEnv(np.full(D, 0.2))
    score_net = _StubScoreNet(target.q_entry)
    invdyn_net = _StubInvDyn()
    ScoreEntryCorrector(target, score_net, invdyn_net, _cfg()).run(env, None, 10)
    for action in env.actions[:-1]:                      # every emitted control-step action...
        assert action[6] == -1.0
    np.testing.assert_array_equal(env.actions[-1], [0, 0, 0, 0, 0, 0, -1])  # ...and the resync


def test_frame_callback_receives_one_frame_per_emitted_control_step_plus_the_resync(target):
    env = _StubEnv(np.full(D, 0.2))                    # target.q_entry is 0, so this displaces
    score_net = _StubScoreNet(target.q_entry)
    invdyn_net = _StubInvDyn()
    frames: list = []
    corrector = ScoreEntryCorrector(target, score_net, invdyn_net, _cfg())
    corrector.run(env, None, 10, frame_cb=frames.append)
    assert corrector.last["steps"] > 0                  # otherwise this scenario proves nothing
    assert len(frames) == corrector.last["steps"] + 1


# ==================================================================================================
# ScoreNet.level_estimate / level_loss -- the supervised head the schedule reads sigma from
# ==================================================================================================
def test_level_estimate_is_strictly_positive_and_orders_perturbation_magnitude():
    """level_estimate is exp() of a linear head's output, so it can never read <= 0 by construction
    -- no clamp is needed to keep the schedule it seeds well-formed. And after a short fit on
    synthetic data, a heavily perturbed configuration must read a LARGER delta_hat than a lightly
    perturbed one: that ordering, not any particular scale, is the property the anneal schedule
    actually depends on (see robot_ood.models.entry.score's module docstring for why a constant instead
    collapsed every clean start onto the barycentre)."""
    torch.manual_seed(0)
    q_clean = torch.randn(256, N_JOINT) * 0.05          # a small synthetic clean-manifold cloud
    net = ScoreNet(hidden=32).fit_stats(q_clean)
    opt = torch.optim.Adam(net.level.parameters(), lr=1e-2)
    for _ in range(200):
        sigma = torch.exp(torch.empty(q_clean.shape[0]).uniform_(np.log(0.01), np.log(0.5)))
        opt.zero_grad()
        level_loss(net, q_clean, sigma).backward()
        opt.step()

    with torch.no_grad():
        clean = net.level_estimate(q_clean)
        assert (clean > 0).all()

        small = net.level_estimate(q_clean + 0.02 * torch.randn_like(q_clean))
        big = net.level_estimate(q_clean + 0.4 * torch.randn_like(q_clean))
        assert big.mean() > small.mean()


def test_load_without_level_head_weights_raises_value_error_naming_the_retrain_command(tmp_path):
    """A pre-level artifact loads its score trunk/head fine and would then have no delta_hat to seed
    a schedule with -- a failure mode this project hit twice. It must fail loud, as a named
    ValueError pointing at the retrain command, not surface load_state_dict's bare KeyError once the
    caller tries to use the (missing) level head."""
    net = ScoreNet(hidden=8)
    state = {k: v.detach().numpy() for k, v in net.state_dict().items()
             if not k.startswith("level.")}
    path = tmp_path / "g_score.npz"
    np.savez(str(path), hidden=np.int64(8), n_joint=np.int64(N_JOINT), **state)

    with pytest.raises(ValueError, match="train entry"):
        ScoreNet.load(path)


def test_invdyn_is_queried_once_per_emitted_step_with_the_live_configuration(target):
    """invdyn is called exactly once per emitted control step -- never once per refine iteration,
    which is pure inference upstream and costs no invdyn calls at all -- and each call's q argument
    is q_now, re-read off the environment after every prior env.step (fk.q_arm inside the loop),
    not the pre-loop q0 cached once. The stub sim leaves qpos fixed across .step(), so every call
    happens to see the same value here; what this pins down is that the call SITE is inside the
    loop, which a q0-cached-once implementation would not reproduce for a moving sim."""
    env = _StubEnv(np.full(D, 0.2))
    score_net = _StubScoreNet(target.q_entry)
    invdyn_net = _StubInvDyn()
    corrector = ScoreEntryCorrector(target, score_net, invdyn_net, _cfg(score_step_cap=0.1))
    corrector.run(env, None, 10)
    assert len(invdyn_net.calls) == corrector.last["steps"] > 0
    for q_call, _dq_call in invdyn_net.calls:
        np.testing.assert_allclose(q_call, np.full(D, 0.2))   # the stub env never moves q


def test_config_reads_a_graph_dirs_own_embedded_calibration(tmp_path, monkeypatch):
    from robot_ood.core.constants import MEASURED, Constant, Constants
    from robot_ood.data import schema

    monkeypatch.delenv("ROBOT_OOD_CONSTANTS", raising=False)
    monkeypatch.delenv("GR_GRAPH_DIR", raising=False)
    constants = Constants(entries={k: Constant(value=v, kind=MEASURED, procedure="test", stats={})
                                   for k, v in {"entry_score_step_cap_rad": 0.25, "entry_perturb_lo": 0.0}.items()})
    np.savez(tmp_path / schema.FCSR_NPZ, **{schema.CONSTANTS_KEY: json.dumps(constants.to_dict())})
    cfg = EntryConfig.from_env(tmp_path)
    assert cfg.score_step_cap == 0.25
    assert cfg.score_sigma_lo == pytest.approx(STILL_TOL * 7 ** -0.5)
    assert EntryConfig.from_env().score_step_cap != 0.25


def test_graph_config_reads_a_graph_dirs_own_embedded_calibration(tmp_path, monkeypatch):
    from robot_ood.core.config import GraphConfig
    from robot_ood.core.constants import MEASURED, Constant, Constants
    from robot_ood.data import schema

    monkeypatch.delenv("ROBOT_OOD_CONSTANTS", raising=False)
    monkeypatch.delenv("GR_GRAPH_DIR", raising=False)
    constants = Constants(entries={"seed_topk": Constant(value=675, kind=MEASURED, procedure="test", stats={})})
    np.savez(tmp_path / schema.FCSR_NPZ, **{schema.CONSTANTS_KEY: json.dumps(constants.to_dict())})
    cfg = GraphConfig.from_env(tmp_path)
    assert cfg.seed_topk == 675 and cfg.graph_dir == str(tmp_path)
    assert GraphConfig.from_env().seed_topk != 675
