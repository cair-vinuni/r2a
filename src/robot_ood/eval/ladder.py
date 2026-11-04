"""The ladder: cells, gates, suites and everything configs/sr_ladder.yaml declares."""

from __future__ import annotations

import dataclasses
import enum
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from robot_ood.core.paths import repo_root
from robot_ood.eval.bench import DEFAULT_BENCH, Bench
from robot_ood.eval.policy import resolve_trunk_path
from robot_ood.eval.scoring import DirCount, ScoreModule

REPO_ROOT = repo_root()
LADDER_YAML = REPO_ROOT / "configs" / "sr_ladder.yaml"


def load_yaml(path: Path) -> dict:
    """Read a YAML file into a dict ({} when the document is empty)."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_suites_doc(path: Path) -> dict:
    """The suites/axes document: trees/*.yaml merged into suites:, then flattened per suite."""
    doc = load_yaml(path)
    suites = dict(doc.get("suites") or {})
    for f in sorted((Path(path).parent / "trees").glob("*.yaml")):
        block = load_yaml(f)
        if isinstance(block, dict):
            suites[f.stem] = block
    doc = {**doc, "suites": suites}
    if not any(isinstance(v, dict) and "eval" in v for v in suites.values()):
        return doc
    flat = {}
    for name, info in suites.items():
        ev, data = info.get("eval") or {}, info.get("data") or {}
        flat[name] = {
            "task_suite": info.get("task_suite"),
            "checkpoints": {k: v for k, v in ev.items() if k.endswith("_ckpt") and v},
            "fwm_dir": data.get("fwm"),
            "light_filter": ev.get("light_filter"),
        }
    return {**doc, "suites": flat}


@dataclass(frozen=True)
class LadderPaths:
    """Absolute paths to every external artifact this driver drives."""

    score_py: Path
    suites_yaml: Path

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any], root: Path = REPO_ROOT) -> "LadderPaths":
        """Resolve sr_ladder.yaml's paths: block (all entries relative to the repo root)."""
        raw = doc["paths"]
        names = [f.name for f in dataclasses.fields(cls)]
        missing = [n for n in names if n not in raw]
        if missing:
            raise KeyError(f"sr_ladder.yaml: paths block is missing {missing}")
        return cls(**{n: root / raw[n] for n in names})


@dataclass(frozen=True)
class SuiteInfo:
    """One suite's row of configs/robot_ood.yaml, with every path already made absolute."""

    name: str
    checkpoints: Mapping[str, Path]
    fwm_dir: Path
    light_filter: Path | None

    def checkpoint(self, key: str) -> Path:
        """The checkpoint the policy descriptor's ckpt_key names in this suite's tree yaml."""
        try:
            return self.checkpoints[key]
        except KeyError:
            raise KeyError(f"suite {self.name!r} declares no {key!r} checkpoint "
                           f"(has {sorted(self.checkpoints)})") from None


@dataclass(frozen=True)
class Defaults:
    """sr_ladder.yaml's defaults: block."""

    num_clients: int
    base_port: int
    server_ready_timeout_s: float

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any]) -> "Defaults":
        return cls(
            num_clients=int(doc["num_clients"]),
            base_port=int(doc["base_port"]),
            server_ready_timeout_s=float(doc["server_ready_timeout_s"]),
        )


@dataclass(frozen=True)
class PreflightLimits:
    """sr_ladder.yaml's preflight: block -- the thresholds that make this driver a polite neighbour on a shared box."""

    min_free_gib: float
    max_loadavg_factor: float
    large_rung_episodes: int

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any]) -> "PreflightLimits":
        return cls(
            min_free_gib=float(doc["min_free_gib"]),
            max_loadavg_factor=float(doc["max_loadavg_factor"]),
            large_rung_episodes=int(doc["large_rung_episodes"]),
        )


MODES = ("base", "fcsr")


class GateKind(str, enum.Enum):
    """The gate kinds sr_ladder.yaml may declare."""

    MIN_SUCCESSES = "min_successes"


@dataclass(frozen=True)
class GateVerdict:
    """Whether a cell cleared its bar, and the one-line reason."""

    passed: bool
    message: str


@dataclass(frozen=True)
class Gate:
    """One cell's pass/fail bar, as declared in sr_ladder.yaml."""

    kind: GateKind
    min_successes: int

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "Gate":
        return cls(kind=GateKind(doc["kind"]), min_successes=int(doc["min_successes"]))

    def evaluate(self, score: "CellScore") -> GateVerdict:
        return GateVerdict(
            score.successes >= self.min_successes,
            f"min_successes={self.min_successes}: got {score.successes}",
        )


def cell_tag(policy: str, mode: str, suite: str, axis: str, variant: str = "") -> str:
    """The one on-disk name for a cell: {policy}_{mode}_{suite}_{axis}[_{variant}]."""
    tag = f"{policy}_{mode}_{suite}_{axis}"
    return f"{tag}_{variant}" if variant else tag


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_log_dir(policy: str, mode: str, suite: str, axis: str, variant: str = "",
                  root: Path = REPO_ROOT) -> Path:
    """logs/{UTC ts}[_s{i}]_{tag}/ (spaces in the axis become underscores here).

    Under ROBOT_OOD_EPISODE_SLICE=i/n the slice index keeps apart the slices one cell's hosts
    start in the same second.
    """
    tag = cell_tag(policy, mode, suite, axis.replace(" ", "_"), variant)
    piece = os.environ.get("ROBOT_OOD_EPISODE_SLICE", "")
    return root / "logs" / f"{utc_stamp()}{'_s' + piece.split('/')[0] if piece else ''}_{tag}"


@dataclass(frozen=True)
class Cell:
    """One ladder cell: a (suite, axis, mode) triple plus everything needed to run and score it."""

    suite: str
    axis: str
    base_task_suite: str
    mode: str
    expected_n: int
    filter_file: Path | None = None
    extra_env: Mapping[str, str] = field(default_factory=dict)
    gate: Gate | None = None
    num_clients: int = 0
    servers: int = 0
    policy: str = "stablevla"
    variant: str = ""
    bench_id: str = DEFAULT_BENCH
    task_suite: str = ""
    results_prefix: str = "plus"
    category: str = ""
    trials: int = 1
    seed: int | None = None

    @property
    def tag(self) -> str:
        return cell_tag(self.policy, self.mode, self.suite, self.axis, self.variant)

    @property
    def key(self) -> str:
        """suite:axis -- what --cells matches on."""
        return f"{self.suite}:{self.axis}"

    @property
    def category_value(self) -> str:
        """The axis as a client that selects its axis at suite construction wants it."""
        return self.category or self.axis.replace("_", " ")

    @property
    def client_suite(self) -> str:
        """The benchmark name the client builds; the base task suite when no axis suffixes it."""
        return self.task_suite or self.base_task_suite

    def result_dir(self, results_root: Path) -> Path:
        return results_root / f"{self.results_prefix}_{self.base_task_suite}" / self.axis / self.tag

    def with_mode(self, mode: str) -> "Cell":
        """Swap this cell's mode. The tag follows automatically -- it is derived, not stored."""
        return replace(self, mode=mode)


@dataclass(frozen=True)
class CellScore:
    """What one cell's result directory currently contains."""

    cell: Cell
    count: DirCount

    @property
    def successes(self) -> int:
        return self.count.successes

    @property
    def actual_n(self) -> int:
        return self.count.n

    @property
    def rate(self) -> float:
        return self.count.rate

    @property
    def complete(self) -> bool:
        return self.count.n == self.cell.expected_n


@dataclass(frozen=True)
class CellReport:
    """A scored cell plus its verdict."""

    score: CellScore
    verdict: GateVerdict

    @property
    def line(self) -> str:
        """The one RESULT ... line printed per cell."""
        cell = self.score.cell
        rate = f"{self.score.rate:.1f}%" if self.score.actual_n else "--"
        status = "PASS" if self.verdict.passed else "FAIL"
        return (
            f"RESULT {cell.tag}: {self.score.successes}/{self.score.actual_n} = {rate}"
            f"  [{status}: {self.verdict.message}]"
        )


class LadderConfig:
    """Everything sr_ladder.yaml + robot_ood.yaml say, resolved once."""

    def __init__(self, doc: Mapping[str, Any], results_root: Path | None = None,
                 root: Path = REPO_ROOT, bench_id: str | None = None) -> None:
        self.doc = doc
        self.root = root
        self.bench = Bench.load(bench_id or doc.get("bench") or DEFAULT_BENCH, root)
        self.paths = LadderPaths.from_doc(doc, root)
        self.scorer = ScoreModule(self.paths.score_py)
        self.defaults = Defaults.from_doc(doc["defaults"])
        self.limits = PreflightLimits.from_doc(doc["preflight"])
        self.suites = self._load_suites()
        # Absolute: a client runs from its benchmark's own checkout, where a relative root would
        # name a different directory than the one the runner reads back.
        self.results_root = (Path(results_root).resolve() if results_root
                             else self.scorer.default_results_root())

    @classmethod
    def load(cls, ladder_yaml: Path | None = None, results_root: Path | None = None,
             bench_id: str | None = None) -> "LadderConfig":
        return cls(load_yaml(ladder_yaml or LADDER_YAML), results_root, bench_id=bench_id)

    @property
    def axes(self) -> list[str]:
        """The benchmark's perturbation axes, in report order."""
        return list(self.bench.axes)

    def recount(self, cell: Cell, trials: int) -> int:
        """A cell's completeness denominator at a different trial count."""
        return self.bench.expected_episodes(cell.suite, cell.axis, trials) or cell.expected_n

    def bench_fields(self, suite: str, axis: str) -> dict[str, Any]:
        """The benchmark half of a Cell: which task suite, category, namespace and trial count."""
        return {
            "base_task_suite": self.bench.task_suites[suite],
            "bench_id": self.bench.id,
            "task_suite": self.bench.task_suite(suite, axis),
            "results_prefix": self.bench.results_prefix,
            "category": self.bench.axis(axis).category_value,
            "trials": self.bench.trials_per_task,
        }

    def suite_policy(self, name: str) -> str:
        """The suite tree's own policy, stablevla when the tree names none."""
        from robot_ood.core.paths import suite_config
        return str(suite_config(name).get("policy") or "stablevla")

    def suite(self, name: str) -> SuiteInfo:
        if name not in self.suites:
            raise ValueError(f"unknown suite {name!r} (known: {sorted(self.suites)})")
        return self.suites[name]

    def default_mode(self, axis: str) -> str:
        """The per-axis default recipe from axis_default_mode."""
        table = self.doc["axis_default_mode"]
        return table.get(axis, table["default"])

    def expected_episodes(self, suite: str, axis: str) -> int:
        """The completeness denominator: the installed benchmark's own count when it can give one, else the ladder's table."""
        counted = self.bench.expected_episodes(suite, axis)
        if counted is not None:
            return counted
        return int(self.doc["expected_episodes"][suite][axis])

    def cells_for_rung(self, rung_name: str) -> list[Cell]:
        """Every cell of a named rung, either declared or auto-generated as the full grid."""
        if rung_name not in self.doc["rungs"]:
            raise ValueError(f"unknown rung {rung_name!r}; known: {sorted(self.doc['rungs'])}")
        rung_doc = self.doc["rungs"][rung_name]
        if rung_doc.get("auto_generate") == "grid":
            gate_doc = rung_doc.get("gate")
            return self._grid_cells(
                rung_name, Gate.from_dict(gate_doc) if gate_doc else None,
                suites=rung_doc.get("suites") or ("object", "spatial", "goal", "long"),
                axes=rung_doc.get("axes") or self.axes,
            )
        return [self.cell_from_doc(c) for c in rung_doc["cells"]]

    def cells_from_tokens(self, spec: str, mode: str | None = None) -> list[Cell]:
        """--cells suite:axis,..."""
        axes = self.axes
        cells: list[Cell] = []
        for token in (t.strip() for t in spec.split(",")):
            if not token:
                continue
            suite, _, axis = token.partition(":")
            if suite not in self.suites:
                raise ValueError(f"--cells: unknown suite {suite!r} in {token!r} (known: {sorted(self.suites)})")
            if axis not in axes:
                raise ValueError(f"--cells: unknown axis {axis!r} in {token!r} (known: {axes})")
            declared = self._find_declared_cell(suite, axis)
            cells.append(Cell(
                suite=suite, axis=axis, **self.bench_fields(suite, axis),
                mode=mode or self.default_mode(axis),
                expected_n=self.expected_episodes(suite, axis),
                filter_file=self._resolve_filter(None, axis, suite),
                gate=declared.gate if declared else None,
                num_clients=self.defaults.num_clients,
            ))
        return cells

    def cell_from_doc(self, cell_doc: Mapping[str, Any]) -> Cell:
        """One cells: entry -> a Cell, resolving its filter, gate and client count."""
        suite, axis = cell_doc["suite"], cell_doc["axis"]
        gate_doc = cell_doc.get("gate")
        fields = self.bench_fields(suite, axis)
        if cell_doc.get("trials"):
            fields["trials"] = int(cell_doc["trials"])
        return Cell(
            suite=suite, axis=axis, **fields,
            mode=cell_doc.get("mode") or self.default_mode(axis),
            policy=cell_doc.get("policy") or self.suite_policy(suite),
            expected_n=int(cell_doc["expected_n"]),
            variant=cell_doc.get("variant", ""),
            filter_file=self._resolve_filter(cell_doc.get("filter"), axis, suite),
            extra_env={k: str(v) for k, v in (cell_doc.get("env") or {}).items()},
            gate=Gate.from_dict(gate_doc) if gate_doc else None,
            num_clients=int(cell_doc.get("num_clients", self.defaults.num_clients)),
        )

    def _grid_cells(self, rung_name: str, gate: Gate | None, *, suites: Sequence[str],
                    axes: Sequence[str]) -> list[Cell]:
        """The suites x axes product, each cell resolved the one way full_grid resolves them."""
        unknown = [a for a in axes if a not in self.axes]
        if unknown:
            raise ValueError(f"rung {rung_name!r} names unknown axes {unknown}; "
                             f"known: {self.axes}")
        return [
            Cell(
                suite=suite, axis=axis, **self.bench_fields(suite, axis),
                mode=self.default_mode(axis),
                policy=self.suite_policy(suite),
                expected_n=self.expected_episodes(suite, axis),
                filter_file=self._resolve_filter(None, axis, suite),
                gate=gate, num_clients=self.defaults.num_clients,
            )
            for suite in suites
            for axis in axes
        ]

    def _find_declared_cell(self, suite: str, axis: str) -> Cell | None:
        for rung_name, rung_doc in self.doc["rungs"].items():
            if rung_doc.get("auto_generate"):
                continue
            for cell_doc in rung_doc["cells"]:
                if cell_doc["suite"] == suite and cell_doc["axis"] == axis:
                    return self.cell_from_doc(cell_doc)
        return None

    def _resolve_filter(self, filt: str | None, axis: str, suite: str) -> Path | None:
        if filt in (None, "auto"):
            if filt == "auto" or axis == "Light_Conditions":
                return self.suite(suite).light_filter
            return None
        return self.root / filt

    def _load_suites(self) -> dict[str, SuiteInfo]:
        doc = load_suites_doc(self.paths.suites_yaml)
        return {
            name: SuiteInfo(
                name=name,
                checkpoints={k: resolve_trunk_path(v) for k, v in
                             (info.get("checkpoints") or
                              {k: v for k, v in info.items()
                               if k.endswith("_ckpt") and v}).items()},
                # a tree without its own corpus (a bimanual tree naming lane trees) gets the default
                fwm_dir=data_root_dir() / (info["fwm_dir"] or f"fwm/{name}"),
                light_filter=(self.root / info["light_filter"]) if info["light_filter"] else None,
            )
            for name, info in doc["suites"].items()
        }



def data_root_dir() -> Path:
    """The data/ tree holding fwm/<suite>/ fields, from $ROBOT_OOD_DATA or inside the clone."""
    return Path(os.environ.get("ROBOT_OOD_DATA", str(REPO_ROOT / "data")))
