"""Resolved filesystem roots, and the configs/robot_ood.yaml lookups that key off them.

Split out of robot_ood.core.config so that resolving a path costs neither torch nor the graph schema: this
module reaches only for robot_ood.core.env and, lazily, yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from robot_ood.core.env import env_str

__all__ = ["DEFAULT_POLICY", "Paths", "default_paths", "lane_graph_dirs", "lane_suites",
           "policy_tree", "project_config", "repo_root", "resolve_trunk_path", "suite_config",
           "suite_from_artifacts"]

# The policy a tree is built against when its yaml names none.
DEFAULT_POLICY = "stablevla"

# robot-ood/src/robot_ood/core/paths.py  ->  parents[3] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_YAML = _REPO_ROOT / "configs" / "robot_ood.yaml"

# Paths.hdf5() search roots, in order.
_HDF5_ROOTS = ("libero_hdf5", "libero_datasets")


def repo_root() -> Path:
    """The repository root."""
    return _REPO_ROOT


@lru_cache(maxsize=1)
def project_config() -> Mapping[str, Any]:
    """configs/robot_ood.yaml (or $ROBOT_OOD_CONFIG), parsed once.

    Returns:
        The parsed mapping -- filesystem roots and training defaults, with one suite per
        configs/trees/<name>.yaml merged in (a tree file wins over an inline suites: block);
        {} when the root file is absent or unreadable.
    """
    path = Path(env_str("ROBOT_OOD_CONFIG") or _PROJECT_YAML)
    try:
        import yaml
        with open(path) as f:
            doc = dict(yaml.safe_load(f) or {})
    except (ImportError, OSError):
        return MappingProxyType({})
    suites = dict(doc.get("suites") or {})
    for f in sorted((path.parent / "trees").glob("*.yaml")):
        try:
            block = yaml.safe_load(f.read_text()) or {}
        except OSError:
            continue
        if isinstance(block, dict):
            suites[f.stem] = block
    if suites:
        doc["suites"] = suites
    return MappingProxyType(doc)


def suite_config(suite: str) -> Mapping[str, Any]:
    """One suite's block of configs/robot_ood.yaml, {} when it has none."""
    block = (project_config().get("suites") or {}).get(suite)
    return block if isinstance(block, dict) else {}


def lane_suites(suite: str) -> Mapping[str, str]:
    """The per-arm trees a two-armed tree names, arm -> tree; {} when it names none."""
    lanes = suite_config(suite).get("lanes") or {}
    return MappingProxyType({str(arm): str(tree) for arm, tree in lanes.items()})


def policy_tree(suite: str, policy: str) -> str:
    """The tree of a suite built against a policy: the suite itself, or <suite>_<policy>.

    Args:
        suite: The suite a cell names (goal, g5).
        policy: The policy descriptor id the cell serves (stablevla, gr00t, pi05, vlact).

    Returns:
        The suite when its own tree names that policy, else the <suite>_<policy> tree.

    Raises:
        ValueError: No tree of that suite was built against that policy.
    """
    own = str(suite_config(suite).get("policy") or DEFAULT_POLICY)
    if policy == own:
        return suite
    tree = f"{suite}_{policy}"
    if str(suite_config(tree).get("policy") or "") != policy:
        raise ValueError(
            f"no tree of {suite} is built against {policy}: configs/trees/{suite}.yaml names {own}, "
            f"and configs/trees/{tree}.yaml is absent or names another policy"
        )
    return tree


def suite_from_artifacts(path) -> str:
    """The suite an artifact belongs to, read off its outputs/<suite>/... path, else "<suite>"."""
    parts = Path(path).parts
    if "outputs" in parts:
        index = parts.index("outputs") + 1
        if index < len(parts):
            return parts[index]
    return "<suite>"


def _configured_root(name: str) -> Path | None:
    """A roots.<name> entry off configs/robot_ood.yaml, resolved against the repo root, or None."""
    value = (project_config().get("roots") or {}).get(name)
    return None if not value else (_REPO_ROOT / Path(value).expanduser())


@lru_cache(maxsize=1)
def _suite_bench_names() -> Mapping[str, str]:
    """{suite: task_suite} mapping from configs/robot_ood.yaml, empty if absent.

    Returns:
        A read-only mapping.
    """
    suites = project_config().get("suites") or {}
    return MappingProxyType({
        name: str(spec["task_suite"])
        for name, spec in suites.items()
        if isinstance(spec, dict) and spec.get("task_suite")
    })


@lru_cache(maxsize=1)
def _suite_fwm_dirs() -> Mapping[str, str]:
    """{suite: data.fwm} mapping from configs/robot_ood.yaml, empty if absent.

    Returns:
        A read-only mapping.
    """
    suites = project_config().get("suites") or {}
    return MappingProxyType({
        name: str((spec.get("data") or {}).get("fwm"))
        for name, spec in suites.items()
        if isinstance(spec, dict) and (spec.get("data") or {}).get("fwm")
    })



# ==== filesystem roots ====
@dataclass(frozen=True)
class Paths:
    """Resolved filesystem roots. Construct once with resolve and pass around.

    Attributes:
        data: The data/ tree (holds fwm/<suite>/ fields, hdf5 demos, datasets).
        results: Where eval rollouts (*_success.mp4 / *.npz qlogs) and figures land.
    """

    data: Path
    results: Path

    @classmethod
    def resolve(cls, data: str | os.PathLike | None = None,
                results: str | os.PathLike | None = None) -> "Paths":
        """Resolve both roots: argument, else ROBOT_OOD_DATA/ROBOT_OOD_RESULTS, else configs/robot_ood.yaml, else repo-local.

        Args:
            data: Override for the data root.
            results: Override for the results root.

        Returns:
            Paths with both roots absolute.
        """
        data_root = Path(data or env_str("ROBOT_OOD_DATA") or _configured_root("data")
                         or (_REPO_ROOT / "data")).expanduser()
        results_root = Path(results or env_str("ROBOT_OOD_RESULTS") or _configured_root("results")
                            or (_REPO_ROOT / "results")).expanduser()
        return cls(data=data_root.resolve(), results=results_root.resolve())

    @property
    def fwm_root(self) -> Path:
        """Root of trained ONF fields (data/fwm)."""
        return self.data / "fwm"

    def fwm(self, suite: str | None = None) -> Path:
        """The per-suite field directory, honoring QNDF_DIR override and configs/robot_ood.yaml.

        Args:
            suite: Suite name. None/""/"default" resolves to the fwm root.

        Returns:
            The directory holding that suite's field.
        """
        override = env_str("QNDF_DIR")
        if override:
            return Path(override)
        if suite in (None, "", "default"):
            return self.fwm_root
        configured = _suite_fwm_dirs().get(suite)
        return (self.data / configured) if configured else (self.fwm_root / suite)

    def hdf5(self, name: str = "libero_object") -> Path:
        """Locate a LIBERO demonstration hdf5 directory.

        Args:
            name: Dataset directory name ("libero_10"), or a suite short-name ("long"), which is
                translated through configs/robot_ood.yaml's task_suite -- see _suite_bench_names.

        Returns:
            The first existing candidate across _HDF5_ROOTS, searched in order.

        Raises:
            FileNotFoundError: No candidate exists; the message names every path attempted.
        """
        configured = (suite_config(name).get("data") or {}).get("hdf5")
        if configured:
            direct = self.data / configured
            if direct.exists():
                return direct
        name = _suite_bench_names().get(name, name)
        candidates = [self.data / root / name for root in _HDF5_ROOTS]
        for c in candidates:
            if c.exists():
                return c
        tried = ", ".join(os.fspath(c) for c in candidates)
        raise FileNotFoundError(f"no hdf5 dir named {name!r} found; tried: {tried}")

    def outputs(self, *parts: str) -> Path:
        """The run-bookkeeping root.

        Args:
            *parts: Path components appended to the root.

        Returns:
            The joined path.
        """
        override = env_str("ROBOT_OOD_OUTPUTS")
        root = (Path(override).expanduser() if override
                else (_configured_root("outputs") or _REPO_ROOT / "outputs"))
        return root.joinpath(*parts)

    def graph(self, suite: str = "") -> Path:
        """The one artifact directory the arms and every build step read.

        Args:
            suite: Suite name.

        Returns:
            $GR_GRAPH_DIR when set, else outputs(suite, "artifacts"). A tree that has not been
            seeded yet falls back to the newest build, resolved past the latest link.
        """
        override = env_str("GR_GRAPH_DIR")
        if override:
            return Path(override)
        configured = suite_config(suite).get("artifacts")
        shipping = self.outputs(configured) if configured else self.outputs(suite, "artifacts")
        if shipping.exists():
            return shipping
        newest = self.outputs(suite, "latest", "artifacts")
        return newest.resolve() if newest.exists() else shipping


def lane_graph_dirs(suite: str) -> Mapping[str, Path]:
    """Each arm's artifact directory for a two-armed tree; {} when the tree names no lanes.

    Args:
        suite: The tree the cell runs.

    Returns:
        arm -> the artifact directory of that arm's lane tree.

    Raises:
        ValueError: $GR_GRAPH_DIR is set, which names one directory where two arms read two.
    """
    lanes = lane_suites(suite)
    if lanes and env_str("GR_GRAPH_DIR"):
        raise ValueError(
            f"GR_GRAPH_DIR is set, but {suite} runs one graph per arm ({', '.join(sorted(lanes.values()))}). "
            "Unset it and let each lane tree resolve its own artifacts."
        )
    paths = Paths.resolve()
    return MappingProxyType({arm: paths.graph(tree) for arm, tree in lanes.items()})


def default_paths() -> Paths:
    """Convenience singleton-style accessor used by scripts that don't thread a Paths through."""
    return Paths.resolve()


def trunk_home(name: str, root: Path | None = None) -> Path:
    """A checkout directory named in configs, from $<NAME>_HOME or inside the clone."""
    return Path(os.environ.get(f"{name.upper()}_HOME") or str((root or repo_root()) / name))


def resolve_trunk_path(value: str | Path, root: Path | None = None) -> Path:
    """Make a config path absolute; its first component may be relocated via $<NAME>_HOME."""
    p = Path(value)
    if p.is_absolute() or not p.parts:
        return p
    return trunk_home(p.parts[0], root).joinpath(*p.parts[1:])
