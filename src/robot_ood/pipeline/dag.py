"""The offline build as a declared graph: nine steps, each with its inputs and its outputs.

Every step names the files it reads and the files it writes, so pipeline.Manifest can decide what
is stale by hashing them rather than by whether an output file exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from robot_ood.core.paths import Paths, default_paths
from robot_ood.models.entry.invdyn import INVDYN_NPZ
from robot_ood.models.entry.score import SCORE_NPZ
from robot_ood.core.schemas import SEGMENTS_NPZ
from robot_ood.data import schema
from robot_ood.models.language.encoder import LANG_NPZ

__all__ = ["DEPLOY_ARTIFACTS", "PROFILES", "PipelineSpec", "STEPS", "Step"]

# Per-profile epoch counts and corpus limits. "fast" proves the nine steps run; it proves no
# number, because a locator at one epoch has not converged.
PROFILES: dict[str, dict] = {
    "full":  {"epochs": {"locator": 40, "fcsr": 30, "entry": 6_000}, "limits": {}},
    "fast":  {"epochs": {"locator": 2, "fcsr": 1, "entry": 200}, "limits": {}},
    "smoke": {"epochs": {"locator": 2, "fcsr": 1, "entry": 200},
              "limits": {"limit_tasks": 2, "limit_demos": 5}},
}

from robot_ood.data.schema import A_PI_RAW_NPZ


@dataclass(frozen=True)
class PipelineSpec:
    """Everything the pipeline steps resolve against, computed once.

    Attributes:
        suite: Suite short-name.
        device: torch device string.
        profile: full | fast | smoke.
        port: Policy-server port for the cache-policy step.
        gpu: CUDA devices the cache-policy step places its servers on; "" defers to the suite's
            build.cache_policy.gpus, so suites built side by side can each keep to their own.
        paths: The path resolver every command shares.
        artifacts: The one artifact directory the later steps read and write.
        log_dir: Where step logs and pid files go.
        python: The interpreter to spawn steps with.
        outputs_root: Overrides Paths.outputs(); the smoke profile builds into its own tree.
    """

    suite: str
    device: str
    profile: str = "full"
    port: int = 14190
    gpu: str = ""
    paths: Paths = field(default_factory=default_paths)
    python: str = ""
    outputs_root: Path | None = None

    @property
    def outputs(self) -> Path:
        """The run-bookkeeping root every step writes under."""
        return Path(self.outputs_root) if self.outputs_root else Path(self.paths.outputs())

    @property
    def artifacts(self) -> Path:
        """The one artifact directory (§6.0), pinned by GR_GRAPH_DIR for steps 3 onward."""
        return self.outputs / self.suite / "artifacts"

    @property
    def latest(self) -> Path:
        """Where build graph leaves its run before step 2b promotes it."""
        return self.outputs / self.suite / "latest" / "artifacts"

    @property
    def log_dir(self) -> Path:
        """Per-step logs and pid files."""
        return self.outputs / self.suite / "pipeline_logs"

    @property
    def knobs(self) -> dict:
        """This profile's epochs and corpus limits."""
        return PROFILES[self.profile]

    @property
    def clients(self) -> int:
        """Concurrent cache-policy clients this suite asks for."""
        from robot_ood.core.build_config import build_param

        return int(build_param("cache_policy", "clients", 0, suite=self.suite))

    @property
    def servers(self) -> int:
        """Policy servers those clients are spread over."""
        from robot_ood.core.build_config import build_param

        return int(build_param("cache_policy", "servers", 0, suite=self.suite))

    def art(self, name: str) -> str:
        """One artifact path inside the artifact directory.

        Args:
            name: The filename.

        Returns:
            The absolute path as a string.
        """
        return str(self.artifacts / name)

    def epochs(self, key: str) -> list[str]:
        """The --epochs flag for one training step, the suite's train file over the profile.

        Args:
            key: The profile key (locator, fcsr, entry).

        Returns:
            ["--epochs", "N"].
        """
        from robot_ood.core.train_config import train_param

        return ["--epochs", str(train_param(key, "epochs", self.knobs["epochs"][key],
                                            suite=self.suite))]

    def limits(self) -> list[str]:
        """The corpus-limit flags this profile applies to steps 2 and 7.

        Returns:
            The flags, empty outside the smoke profile.
        """
        out: list[str] = []
        for key, value in self.knobs["limits"].items():
            out += [f"--{key.replace('_', '-')}", str(value)]
        return out

    def env(self) -> dict[str, str]:
        """The environment every step inherits, with the artifact directory pinned.

        Returns:
            A copy of os.environ plus GR_GRAPH_DIR and ROBOT_OOD_OUTPUTS.
        """
        return {**os.environ, "GR_GRAPH_DIR": str(self.artifacts),
                "ROBOT_OOD_OUTPUTS": str(self.outputs)}

    def py(self) -> str:
        """The interpreter to run steps with."""
        import sys

        return self.python or sys.executable


@dataclass(frozen=True)
class Step:
    """One node of the build graph.

    Attributes:
        name: The step id, e.g. "02_graph".
        stage: Which verb group it belongs to (build, cache, train, check).
        argv: The command, resolved against the spec.
        inputs: Paths the step reads; a change to any of them makes it stale.
        outputs: Paths the step writes.
        accept_rc: Exit codes that count as success beyond 0.
        detach: Run under the job runner and poll process state, not file existence.
        forceable: The verb takes --force, so a stale step passes it and the verb's own
            config-identity skip cannot override this DAG's verdict.
        note: One line of --dry-run output, when the step needs one.
    """

    name: str
    stage: str
    argv: Callable[[PipelineSpec], list[str]]
    inputs: Callable[[PipelineSpec], list[str]]
    outputs: Callable[[PipelineSpec], list[str]]
    accept_rc: tuple[int, ...] = (0,)
    detach: bool = False
    forceable: bool = True
    note: str = ""


def _graph_inputs(spec: PipelineSpec) -> list[str]:
    """The promoted node and edge tables every step from 3 onward reads."""
    return [spec.art(schema.NODES_NPZ), spec.art(schema.EDGES_NPZ)]


STEPS: tuple[Step, ...] = (
    Step("02_graph", "build",
         lambda s: [s.py(), "-m", "robot_ood.cli", "build", "graph", "--suite", s.suite] + s.limits(),
         lambda s: [],
         _graph_inputs,
         note="writes a timestamped run for bookkeeping and seeds the one canonical artifact "
              "path every later step reads"),
    Step("04_segments", "build",
         lambda s: [s.py(), "-m", "robot_ood.cli", "build", "segments", "--suite", s.suite],
         _graph_inputs,
         lambda s: [s.art(SEGMENTS_NPZ)]),
    Step("05_calib", "build",
         lambda s: [s.py(), "-m", "robot_ood.cli", "build", "calibrate", "--suite", s.suite],
         _graph_inputs,
         lambda s: [s.art(schema.ACTION_SCALE_JSON)],
         accept_rc=(0, 2),
         note="exit 2 is a pre-registered gate failing, not an operational error: on `long` the "
              "dilation-reach gate cannot pass with the shipped schema, and the artifact is still "
              "written"),
    Step("06_lang", "build",
         lambda s: [s.py(), "-m", "robot_ood.cli", "build", "language", "--suite", s.suite],
         _graph_inputs,
         lambda s: [s.art(LANG_NPZ)]),
    Step("07_apiraw", "cache",
         lambda s: [s.py(), "-m", "robot_ood.cli", "cache-policy", "--suite", s.suite,
                    "--port", str(s.port), "--clients", str(s.clients),
                    "--servers", str(s.servers)]
                   + (["--gpu", s.gpu] if s.gpu else []) + s.limits(),
         _graph_inputs,
         lambda s: [s.art(A_PI_RAW_NPZ)],
         detach=True,
         note="hours long, in the stablevla env; the job runner polls its process, so a crash at "
              "minute five is seen at minute five"),
    # No 08: it trained a blend head no arm loaded. The rest keep their ids so older run
    # directories still line up.
    Step("09_locator", "train",
         lambda s: [s.py(), "-m", "robot_ood.cli", "train", "locator", "--suite", s.suite,
                    "--device", s.device] + s.epochs("locator"),
         lambda s: _graph_inputs(s) + [s.art(LANG_NPZ), s.art(A_PI_RAW_NPZ)],
         lambda s: [s.art(schema.HEAD_NPZ)],
         note="the retriever and the task fusion, co-trained in one job: the fusion's zero-init "
              "last layer is an implicit warmup, so no task gradient reaches the retriever at "
              "step 0. The job's save also replays the checkpoint over the held-out demos and "
              "embeds the phase-skew report in g_head.npz, which is why no skew step follows"),
    Step("13_entry", "train",
         lambda s: [s.py(), "-m", "robot_ood.cli", "train", "entry", "--suite", s.suite,
                    "--device", s.device] + s.epochs("entry"),
         _graph_inputs,
         lambda s: [s.art(SCORE_NPZ), s.art(INVDYN_NPZ)],
         note="§10 fits both t = 0 nets off the same node table, so this is one step and not two"),
    Step("14b_fcsr", "train",
         lambda s: [s.py(), "-m", "robot_ood.cli", "train", "fcsr", "--suite", s.suite,
                    "--device", s.device] + s.epochs("fcsr"),
         lambda s: _graph_inputs(s) + [s.art(SEGMENTS_NPZ), s.art(schema.HEAD_NPZ)],
         lambda s: [s.art(schema.FCSR_NPZ)],
         note="the job measures C0 itself when nothing resolves a calibration, trains under it, "
              "and embeds those exact bytes in g_fcsr.npz -- deploy then needs only the artifact "
              "directory, no constants.json"),
    Step("15_check", "check",
         lambda s: [s.py(), "-m", "robot_ood.cli", "artifacts", "check", str(s.artifacts),
                    "--stage", "deploy"],
         lambda s: _graph_inputs(s) + [s.art(n) for n in DEPLOY_ARTIFACTS],
         lambda s: [],
         forceable=False,
         note="the deploy-stage gate. It declares every artifact as an input and writes none, so it "
              "re-runs whenever any of them changes and never otherwise"),
)

# What step 15 gates on, and therefore what makes it stale.
DEPLOY_ARTIFACTS: tuple[str, ...] = (
    schema.HEAD_NPZ, schema.FCSR_NPZ, SEGMENTS_NPZ, schema.ACTION_SCALE_JSON, LANG_NPZ,
    A_PI_RAW_NPZ,
    SCORE_NPZ, INVDYN_NPZ,
)
