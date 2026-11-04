"""The deploy-time chain a benchmark client drives, assembled from the run mode's environment.

A simulator client holds one of these per process and calls it at four points of the rollout.
Two clients that both hold one run the same method, which is what makes a delta between two
benchmarks attributable to the benchmark.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from robot_ood.core.env import env_str

if TYPE_CHECKING:
    from robot_ood.runtime.fuse import BlendPlan

__all__ = ["DeployHooks"]


class DeployHooks:
    """The entry corrector and the sentinel: on under fcsr, off under base or (entry only) off a proxy robot.

    Attributes:
        entry: The t = 0 corrector, or None under base or on a robot whose corpus rows are not joints.
        sentinel: The t > 0 locator and corrector, or None under base.
    """

    def __init__(self, entry: Any = None, sentinel: Any = None) -> None:
        self.entry = entry
        self.sentinel = sentinel

    @classmethod
    def from_env(cls, suite: str) -> "DeployHooks":
        """Build the chain $MODE asks for: fcsr turns both halves on, base leaves both off.

        Args:
            suite: The suite whose artifact directory the entry nets load from, when GR_GRAPH_DIR
                does not name one.

        Returns:
            The assembled chain; nothing is imported under base.

        Raises:
            ValueError: $MODE names neither base nor fcsr.
        """
        mode = env_str("MODE", "base")
        if mode not in ("base", "fcsr"):
            raise ValueError(f"unknown MODE={mode!r}; expected 'base' or 'fcsr'")
        if mode == "base":
            return cls()
        from robot_ood.core.config import SentinelConfig

        cfg = SentinelConfig.from_env()
        entry = None if cfg.ablation.entry else cls._load_entry(suite)
        return cls(entry=entry, sentinel=cls._load_sentinel(cfg))

    @property
    def plan_rows(self) -> int:
        """Rows one plan covers, or the whole chunk under base, where nothing plans."""
        return 0 if self.sentinel is None else int(self.sentinel.plan_rows)

    @classmethod
    def for_graph(cls, tree: str, artifacts: "str | os.PathLike", chunk: int) -> "DeployHooks":
        """The sentinel of one tree's graph, checking once per plan_rows of the policy chunk.

        Args:
            tree: The tree the graph was built from; it names the corpus the bundle reads its
                kinematics and calibration for.
            artifacts: Its artifact directory.
            chunk: Rows the policy executes per query; the graph's own segments decide how many
                of them one plan covers.

        Returns:
            A chain with the sentinel on, and the entry corrector on when the tree's rows are
            joints it can walk.
        """
        import dataclasses

        from robot_ood.core.config import SentinelConfig
        from robot_ood.core.kinematics import entry_walks
        from robot_ood.core.paths import default_paths
        from robot_ood.data.segments import graph_plan_rows
        from robot_ood.runtime.sentinel import Sentinel

        # The tree's own field directory, not $QNDF_DIR: a two-armed robot runs one chain per arm
        # and each reads the corpus of its own arm.
        cfg = dataclasses.replace(SentinelConfig.from_env(int(chunk)), graph_dir=os.fspath(artifacts),
                                  plan_rows=graph_plan_rows(artifacts, int(chunk)))
        entry = cls._entry_for(artifacts) if entry_walks(tree) and not cfg.ablation.entry else None
        return cls(entry=entry, sentinel=Sentinel(cfg, fwm_dir=default_paths().fwm_root / tree))

    @property
    def enabled(self) -> bool:
        """Whether any mechanism is on."""
        return self.entry is not None or self.sentinel is not None

    def reset(self, shard: int) -> None:
        """Clear the per-episode state. Call after the policy's own reset, before env.reset()."""
        if self.sentinel is not None:
            self.sentinel.reset(shard)

    def correct_entry(self, env: Any, obs: dict, t: int,
                      frame_cb: Callable[[np.ndarray], None]) -> tuple[dict, int]:
        """Walk the arm onto the demonstrated entry manifold, once, before the policy first acts.

        Args:
            env: The simulator environment.
            obs: The observation the correction starts from.
            t: The environment step counter.
            frame_cb: Receives each rendered correction frame, for the replay video.

        Returns:
            The observation and step counter after the correction, unchanged under base.
        """
        if self.entry is None:
            return obs, t
        return self.entry.run(env, obs, t, frame_cb=frame_cb)

    def plan(self, env: Any, state: np.ndarray, step: int, *, shard: int,
             task: str, seg_prior: np.ndarray | None = None) -> "BlendPlan | None":
        """Feed this step's query window and, on a check step, plan the blend.

        Args:
            env: The simulator environment.
            state: concat(ee_pos, ee_ori, gripper_qpos) for this step.
            step: The policy step counter.
            shard: This client's shard id.
            task: The task description the policy was given.
            seg_prior: [S] log prior over this arm's segment table, or None; what the other arm
                of a two-armed robot is doing reaches retrieval through it.

        Returns:
            The blend to apply to the policy chunk's remaining rows, or None off a check step or
            with the sentinel off.
        """
        checked = self.check(env, state, step, shard=shard, task=task)
        if checked is None:
            return None
        return self.plan_checked(checked, shard=shard, task=task, seg_prior=seg_prior)

    def check(self, env: Any, state: np.ndarray, step: int, *, shard: int,
              task: str) -> "tuple[np.ndarray, float, float] | None":
        """The first half of plan(): feed the window and, on a check step, run the locator.

        Args:
            env: The simulator environment.
            state: concat(ee_pos, ee_ori, gripper_qpos) for this step.
            step: The policy step counter.
            shard: This client's shard id.
            task: The task description the policy was given.

        Returns:
            (q_now, grip, phase_hat) for plan_checked, or None off a check step or with the sentinel off.
        """
        if self.sentinel is None:
            return None
        from robot_ood.core.kinematics import q_arm
        from robot_ood.data.schema import grip_flag

        q_now = q_arm(env)
        self.sentinel.observe(q_now, step, wid=shard)
        grip = grip_flag(state[6:])
        phase_hat = self.sentinel.check(q_now, grip, step, wid=shard, task=task)
        return None if phase_hat is None else (q_now, grip, phase_hat)

    def plan_checked(self, checked: "tuple[np.ndarray, float, float]", *, shard: int, task: str,
                     seg_prior: np.ndarray | None = None) -> "BlendPlan":
        """The second half of plan(): the blend for a check() that returned a phase."""
        q_now, grip, phase_hat = checked
        return self.sentinel.plan_checked(q_now, grip, phase_hat, wid=shard, task=task, seg_prior=seg_prior)

    def query_context(self, checked: "tuple[np.ndarray, float, float]", *, shard: int, task: str) -> Any:
        """The retriever's QueryContext for a check() that returned a phase, without retrieving."""
        q_now, grip, phase_hat = checked
        return self.sentinel.query_context(q_now, grip, phase_hat, wid=shard, task=task)

    def record_action(self, action: np.ndarray, step: int, shard: int) -> None:
        """Feed the executed action into the commanded-action window, every environment step."""
        if self.sentinel is not None:
            self.sentinel.observe_action(action, step, wid=shard)

    @staticmethod
    def _load_sentinel(cfg: Any) -> Any:
        import dataclasses

        from robot_ood.data.segments import graph_plan_rows
        from robot_ood.runtime.bundle import RuntimeBundle
        from robot_ood.runtime.sentinel import Sentinel

        _, graph_dir = RuntimeBundle(cfg).graph_config()
        return Sentinel(dataclasses.replace(cfg, plan_rows=graph_plan_rows(graph_dir, cfg.policy_chunk)))

    @classmethod
    def _load_entry(cls, suite: str) -> Any:
        from robot_ood.core.kinematics import entry_walks
        from robot_ood.core.paths import default_paths

        if not entry_walks(suite):
            return None
        return cls._entry_for(default_paths().graph(suite))

    @staticmethod
    def _entry_for(graph_dir: "str | os.PathLike") -> Any:
        """The t = 0 corrector of one artifact directory: its entry target and its two trained nets."""
        from robot_ood.core.config import EntryConfig
        from robot_ood.models.entry.invdyn import InverseDynamicsNet
        from robot_ood.models.entry.score import ScoreNet
        from robot_ood.models.entry.target import EntryTarget
        from robot_ood.runtime.entry import ScoreEntryCorrector

        return ScoreEntryCorrector(
            EntryTarget.from_graph_dir(graph_dir), ScoreNet.load(graph_dir),
            InverseDynamicsNet.load(graph_dir), EntryConfig.from_env(graph_dir),
        )
