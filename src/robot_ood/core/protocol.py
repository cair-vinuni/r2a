"""Rollout protocols: each policy's published LIBERO evaluation recipe, replayed by the shared harness.

A protocol fixes what the simulator client does around the policy: how an episode starts, how many
settling steps run first, the step budget per suite, the seed, the frame side sent over the wire,
how many actions of a predicted chunk execute before the next query, and how the gripper command is
read. The policy descriptor names its protocol; the values below are copied from the vendor file
each entry cites.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

__all__ = ["BENCH_STEP_LIMIT", "BINARIZE", "DEFAULT_PROTOCOL", "FIXED", "NATIVE", "NATIVE_SIZE",
           "PROTOCOLS", "RANDOM", "Protocol", "get_protocol", "protocol_ids", "suite_chunk"]

FIXED = "fixed"
RANDOM = "random"
BINARIZE = "binarize"
NATIVE = "native"
NATIVE_SIZE = 0
BENCH_STEP_LIMIT = 0      # a step budget the benchmark sets itself, per task
ALL_SUITES = "all"
DEFAULT_PROTOCOL = "openvla_oft"

_OPENVLA_BUDGET = MappingProxyType({
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520,
    "libero_90": 400,
})


@dataclass(frozen=True)
class Protocol:
    """One vendor's rollout recipe.

    Attributes:
        id: The name a policy descriptor refers to.
        source: The vendor file the values are read from.
        init_states: fixed cycles the suite's stored initial states; random keeps env.reset() as is.
        settle_steps: Dummy steps before the policy acts.
        max_steps: Step budget by base suite; the key all covers every suite, and the value
            BENCH_STEP_LIMIT leaves the budget to the benchmark's own per-task limit.
        seed: The environment and numpy seed.
        wire_image_size: Side of the square frames the client sends; NATIVE_SIZE sends frames as
            the simulator rendered them.
        exec_horizon: Actions executed from each predicted chunk before the next query.
        gripper: binarize thresholds an open value at 0.5 into -1 or +1; native passes it on.
        bimanual: The chunk carries both arms' absolute joint targets rather than one arm's deltas.
    """

    id: str
    source: str
    init_states: str
    settle_steps: int
    max_steps: Mapping[str, int]
    seed: int
    wire_image_size: int
    exec_horizon: int
    gripper: str
    bimanual: bool = False

    def step_budget(self, base_suite: str) -> int:
        """The step budget of a base suite.

        Args:
            base_suite: A LIBERO suite name such as libero_10.

        Returns:
            The number of policy steps an episode may take.

        Raises:
            KeyError: The protocol declares no budget for the suite.
        """
        if base_suite in self.max_steps:
            return self.max_steps[base_suite]
        if ALL_SUITES in self.max_steps:
            return self.max_steps[ALL_SUITES]
        raise KeyError(f"protocol {self.id!r} has no step budget for {base_suite!r}")


PROTOCOLS: Mapping[str, Protocol] = MappingProxyType({
    "openvla_oft": Protocol(
        id="openvla_oft",
        source="stablevla/experiments/robot/libero/run_libero_eval.py",
        init_states=FIXED, settle_steps=10, max_steps=_OPENVLA_BUDGET, seed=7,
        wire_image_size=224, exec_horizon=8, gripper=BINARIZE,
    ),
    "nvidia": Protocol(
        id="nvidia",
        source="external/Isaac-GR00T/gr00t/eval/rollout_policy.py",
        init_states=RANDOM, settle_steps=0, max_steps=MappingProxyType({ALL_SUITES: 504}), seed=7,
        wire_image_size=256, exec_horizon=8, gripper=BINARIZE,
    ),
    "openpi": Protocol(
        id="openpi",
        source="external/openpi/examples/libero/main.py",
        init_states=FIXED, settle_steps=10, max_steps=_OPENVLA_BUDGET, seed=7,
        wire_image_size=256, exec_horizon=5, gripper=NATIVE,
    ),
    # RoboTwin 2.0 through XPolicyLab: the whole chunk of joint targets runs before the next
    # query, the budget is the benchmark's own per-task limit, episodes start at 2000 * (1 + seed).
    "pi05_robotwin": Protocol(
        id="pi05_robotwin",
        source="external/RoboTwin/scripts/eval_policy.py + XPolicyLab policy/Pi_05/model.py",
        init_states=RANDOM, settle_steps=0, max_steps=MappingProxyType({ALL_SUITES: BENCH_STEP_LIMIT}),
        seed=0, wire_image_size=NATIVE_SIZE, exec_horizon=50, gripper=NATIVE, bimanual=True,
    ),
    # VLAct on RoboTwin 2.0: a thirty-two step chunk of both arms' absolute joint targets, all of
    # which runs before the next query, the way its RoboTwin interface steps the environment.
    "vlact_robotwin": Protocol(
        id="vlact_robotwin",
        source="external/VLAct/examples/Robotwin/eval_files/model2robotwin_interface.py",
        init_states=RANDOM, settle_steps=0, max_steps=MappingProxyType({ALL_SUITES: BENCH_STEP_LIMIT}),
        seed=0, wire_image_size=NATIVE_SIZE, exec_horizon=32, gripper=NATIVE, bimanual=True,
    ),
    # GR00T N1.7 on one real ALOHA arm: 16 rows (delta_indices 0..15), native frames and an
    # absolute continuous gripper. No simulator; the offline cache and the runtime share this contract.
    "gr00t_aloha": Protocol(
        id="gr00t_aloha",
        source="deploy/aloha_right/serve.py",
        init_states=RANDOM, settle_steps=0, max_steps=MappingProxyType({ALL_SUITES: BENCH_STEP_LIMIT}),
        seed=0, wire_image_size=NATIVE_SIZE, exec_horizon=16, gripper=NATIVE,
    ),
    "gr00t_vrh31": Protocol(
        id="gr00t_vrh31",
        source="deploy/vrh31_bds_pick/serve.py",
        init_states=RANDOM, settle_steps=0, max_steps=MappingProxyType({ALL_SUITES: BENCH_STEP_LIMIT}),
        seed=0, wire_image_size=NATIVE_SIZE, exec_horizon=16, gripper=NATIVE, bimanual=True,
    ),
})


def protocol_ids() -> list[str]:
    """The declared protocol names, sorted."""
    return sorted(PROTOCOLS)


def get_protocol(protocol_id: str) -> Protocol:
    """The protocol a descriptor names.

    Args:
        protocol_id: One of protocol_ids().

    Returns:
        The protocol.

    Raises:
        KeyError: No such protocol.
    """
    try:
        return PROTOCOLS[protocol_id]
    except KeyError:
        raise KeyError(f"unknown protocol {protocol_id!r} (known: {protocol_ids()})") from None


def suite_chunk(suite: str) -> int:
    """Rows the policy a tree names executes per query, off that policy's rollout protocol.

    Args:
        suite: The tree, as configs/trees/<suite>.yaml names it.

    Returns:
        The protocol's exec_horizon.

    Raises:
        KeyError: The tree names no policy.
    """
    import yaml

    from robot_ood.core.paths import repo_root, suite_config

    tree = suite_config(suite)
    if "policy" not in tree:
        raise KeyError(f"configs/trees/{suite}.yaml names no policy, so no chunk length to read")
    path = repo_root() / "evals" / "policies" / f"{tree['policy']}.yaml"
    doc = yaml.safe_load(path.read_text()) or {}
    return get_protocol(doc.get("protocol", DEFAULT_PROTOCOL)).exec_horizon
