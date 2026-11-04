"""The benchmark registry: what a simulator benchmark tells the eval driver about itself.

One directory under evals/ per benchmark, each holding a bench.yaml. Nothing outside that file
names a benchmark's checkout, client, axes or result namespace.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from robot_ood.core.paths import repo_root

__all__ = ["Axis", "Bench", "EGL_RENDERER", "RENDERERS", "VULKAN_RENDERER", "bench_ids", "load_bench",
           "repo_root", "DEFAULT_BENCH"]

DEFAULT_BENCH = "libero_plus"
BENCH_FILE = "bench.yaml"
EGL_RENDERER = "egl"          # MuJoCo/robosuite: the client is told its EGL device
VULKAN_RENDERER = "vulkan"    # SAPIEN: the client picks its GPU from CUDA_VISIBLE_DEVICES alone
RENDERERS = (EGL_RENDERER, VULKAN_RENDERER)
EVALS_DIR = "evals"


@dataclass(frozen=True)
class Axis:
    """One perturbation axis of a benchmark.

    Attributes:
        name: The axis token used in tags, result paths and --axis.
        suffix: Appended to the suite's benchmark name to select this axis' task suite.
        category_value: Passed to a client that selects its axis at suite construction.
        label: Short column header for the scorer's grid.
    """

    name: str
    suffix: str = ""
    category_value: str = ""
    label: str = ""


@dataclass(frozen=True)
class Bench:
    """One benchmark's contract with scripts/run_sr.py.

    Attributes:
        id: Directory name under evals/, and the --bench token.
        root: The benchmark checkout, absolute.
        client: The simulator client entry point, absolute.
        conda_env: The environment its client runs in.
        pythonpath: Prepended to the client's PYTHONPATH, absolute, in order.
        env: Extra environment for every client process, roots already expanded.
        axes: Axis name -> Axis, in report order.
        task_suites: Suite short-name -> the benchmark name the client builds.
        results_prefix: Result dirs live at <results>/<prefix>_<task_suite>/<axis>/<tag>.
        default_rung: The ladder rung a bare run drives, or "" for ad hoc cells only.
        client_args: The client's argv after the interpreter and script, with placeholders.
        setup: The provisioning script, absolute.
        marker: A path under root that exists only in a correctly installed checkout.
        task_glob: Where the client's task files live, relative to root, or "" when unknown.
        task_list: A text file naming the client's tasks one per line, relative to the repo, with
            {task_suite} for the suite, or "" for a benchmark that ships its tasks as files.
        trials_per_task: Episodes one task contributes.
        renderer: egl (MuJoCo, told its EGL device) or vulkan (SAPIEN, its GPU alone).
        episodes_flag: The client flag that takes a json file of the [task, episode] pairs to run;
            a tyro client nests it under --args.
    """

    id: str
    root: Path
    client: Path
    conda_env: str
    pythonpath: tuple[Path, ...]
    env: Mapping[str, str]
    axes: Mapping[str, Axis]
    task_suites: Mapping[str, str]
    results_prefix: str
    default_rung: str
    client_args: tuple[str, ...]
    setup: Path
    marker: str
    task_glob: str
    task_list: str
    trials_per_task: int
    renderer: str = EGL_RENDERER
    episodes_flag: str = "--episodes"
    _repo: Path = field(default=Path("."), repr=False, compare=False)

    @classmethod
    def load(cls, bench_id: str, root: Path | None = None) -> "Bench":
        """Read evals/<bench_id>/bench.yaml.

        Args:
            bench_id: The benchmark directory name.
            root: The repository root; discovered from this file when absent.

        Returns:
            The parsed benchmark.

        Raises:
            FileNotFoundError: No such benchmark directory.
        """
        root = root or repo_root()
        path = root / EVALS_DIR / bench_id / BENCH_FILE
        if not path.is_file():
            raise FileNotFoundError(f"unknown bench {bench_id!r}: no {path} (known: {bench_ids(root)})")
        doc = yaml.safe_load(path.read_text())
        checkout = cls._checkout(doc, root)
        fields = {"root": str(checkout), "repo": str(root), "src": str(root / "src"),
                  "stablevla": os.environ.get("STABLEVLA_HOME", str(root / "stablevla"))}
        return cls(
            id=doc["id"],
            root=checkout,
            client=root / doc["client"],
            conda_env=doc["conda_env"],
            pythonpath=tuple(cls._resolve(p, root, fields) for p in doc.get("pythonpath", ())),
            env={k: v.format(**fields) for k, v in (doc.get("env") or {}).items()},
            axes=cls._axes(doc["axes"]),
            task_suites=dict(doc["task_suites"]),
            results_prefix=doc["results_prefix"],
            default_rung=doc.get("default_rung", ""),
            client_args=tuple(doc.get("client_args", ())),
            setup=root / doc["setup"],
            marker=doc["marker"],
            task_glob=doc.get("task_glob", ""),
            task_list=doc.get("task_list", ""),
            episodes_flag=doc.get("episodes_flag", "--episodes"),
            trials_per_task=int(doc.get("trials_per_task", 1)),
            renderer=cls._renderer(doc.get("renderer", EGL_RENDERER)),
            _repo=root,
        )

    @property
    def installed(self) -> bool:
        """Whether this benchmark's checkout is present and provisioned."""
        return (self.root / self.marker).exists()

    def axis(self, name: str) -> Axis:
        """The named axis.

        Raises:
            ValueError: This benchmark has no such axis.
        """
        if name not in self.axes:
            raise ValueError(f"{self.id}: unknown axis {name!r} (known: {list(self.axes)})")
        return self.axes[name]

    def task_suite(self, suite: str, axis: str) -> str:
        """The benchmark name the client builds for one (suite, axis).

        Raises:
            ValueError: This benchmark does not carry that suite.
        """
        if suite not in self.task_suites:
            raise ValueError(f"{self.id}: unknown suite {suite!r} (known: {list(self.task_suites)})")
        return self.task_suites[suite] + self.axis(axis).suffix

    def axis_dir(self, results_root: Path, suite: str, axis: str) -> Path:
        """Where one (suite, axis) cell's result tags live."""
        return results_root / f"{self.results_prefix}_{self.task_suites[suite]}" / axis

    def task_files(self, suite: str, axis: str) -> list[Path]:
        """The client's task files for one cell, empty when the benchmark does not declare a glob."""
        if not self.task_glob:
            return []
        pattern = self.task_glob.format(task_suite=self.task_suite(suite, axis))
        return sorted(self.root.glob(pattern))

    def task_names(self, suite: str, axis: str) -> list[str]:
        """The tasks of one cell as the task list names them, empty without a task list."""
        if not self.task_list:
            return []
        path = self._repo / self.task_list.format(task_suite=self.task_suite(suite, axis))
        return [line.strip() for line in path.read_text().splitlines()
                if line.strip() and not line.startswith("#")]

    def expected_episodes(self, suite: str, axis: str, trials: int | None = None) -> int | None:
        """How many episodes one cell should produce, or None when the benchmark cannot say.

        Args:
            suite: Suite short-name.
            axis: The perturbation axis.
            trials: Episodes per task; the benchmark's own default when absent.

        Returns:
            Task count times trials, or None for a benchmark that declares neither a task glob
            nor a task list.
        """
        if self.task_glob:
            n_tasks = len(self.task_files(suite, axis))
        elif self.task_list:
            n_tasks = len(self.task_names(suite, axis))
        else:
            return None
        return n_tasks * (trials or self.trials_per_task)

    def client_argv(self, **fields: Any) -> list[str]:
        """The client's argv, with every placeholder filled.

        Args:
            **fields: Placeholder values, at least checkpoint, host, port, task_suite,
                category_value, trials, shard, num_shards and result_dir.

        Returns:
            The argument list, without the interpreter or the script path.

        Raises:
            KeyError: An argument names a placeholder the caller did not supply.
        """
        return [arg.format(**fields) for arg in self.client_args]

    def client_env(self, inherited: Mapping[str, str] | None = None) -> dict[str, str]:
        """This benchmark's environment, with every other benchmark's checkout evicted.

        Args:
            inherited: The ambient environment; os.environ when absent.

        Returns:
            The variables to set for a client process, PYTHONPATH included.
        """
        # Two LIBERO forks both ship a top-level libero package, so an inherited PYTHONPATH naming
        # the other one would decide the import by accident.
        inherited = os.environ if inherited is None else inherited
        foreign = {str(b.root) for b in load_all() if b.id != self.id}
        kept = [p for p in (inherited.get("PYTHONPATH") or "").split(os.pathsep)
                if p and not any(p == f or p.startswith(f + os.sep) for f in foreign)]
        mine = [str(p) for p in self.pythonpath]
        env = dict(self.env)
        env["PYTHONPATH"] = os.pathsep.join(mine + [p for p in kept if p not in mine])
        return env

    @staticmethod
    def _renderer(value: str) -> str:
        if value not in RENDERERS:
            raise ValueError(f"renderer must be one of {RENDERERS}, got {value!r}")
        return value

    @staticmethod
    def _checkout(doc: Mapping[str, Any], root: Path) -> Path:
        """The benchmark checkout, honouring the override variable the bench file names."""
        named = os.environ.get(doc.get("root_env", ""))
        return Path(named) if named else root / doc["root"]

    @staticmethod
    def _resolve(value: str, root: Path, fields: Mapping[str, str]) -> Path:
        formatted = value.format(**fields)
        return Path(formatted) if os.path.isabs(formatted) else root / formatted

    @staticmethod
    def _axes(entries: Sequence[Mapping[str, Any]]) -> dict[str, Axis]:
        axes: dict[str, Axis] = {}
        for entry in entries:
            name = entry["name"]
            axes[name] = Axis(
                name=name,
                suffix=entry.get("suffix", ""),
                category_value=entry.get("category_value", name.replace("_", " ")),
                label=entry.get("label", name),
            )
        return axes


def bench_ids(root: Path | None = None) -> list[str]:
    """Every benchmark directory that carries a bench.yaml, sorted."""
    root = root or repo_root()
    return sorted(p.parent.name for p in (root / EVALS_DIR).glob(f"*/{BENCH_FILE}"))


def load_bench(bench_id: str | None = None, root: Path | None = None) -> Bench:
    """Load one benchmark, defaulting to $ROBOT_OOD_BENCH and then to DEFAULT_BENCH."""
    return Bench.load(bench_id or os.environ.get("ROBOT_OOD_BENCH") or DEFAULT_BENCH, root)


def load_all(root: Path | None = None) -> list[Bench]:
    """Every registered benchmark."""
    root = root or repo_root()
    return [Bench.load(name, root) for name in bench_ids(root)]
