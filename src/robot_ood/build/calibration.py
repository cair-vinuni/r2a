"""C0 -- the Calibrator: demo corpus -> constants.json + version + content hash.

Zero parameters, zero gradient, one run per corpus. Everything downstream that used to carry a hand
constant reads its value from here instead, so moving to a new suite is a calibration run rather
than a code edit. The procedures it drives are robot_ood.build.measure, one class each.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from robot_ood.core.constants import CONSTANTS_JSON, Constant, Constants
from robot_ood.data.libero import DemoCorpus
from robot_ood.build.measure import (
    ActionScaleFit,
    CalibrationSpec,
    DilationReach,
    EntryPerturbation,
    EntryStepCap,
    NeighbourSpacing,
    RobotInitMagnitude,
    SeedSupport,
    SegmentEntryStratum,
    SegmentHardNegatives,
    SegmentLen,
    SegTier1M,
)
from robot_ood.data import schema

__all__ = ["CalibrationResult", "Calibrator", "OutsideGraphDir", "ensure_constants", "write_constants"]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """What one calibration run produced.

    Attributes:
        constants: The calibration itself.
        constants_path: Where constants.json was written, or None for a measure-only run.
        action_scale_path: Where action_scale.json was written, or None.
        gate_failures: One line per pre-registered gate this calibration misses; empty when clean.
    """

    constants: Constants
    constants_path: Path | None = None
    action_scale_path: Path | None = None
    gate_failures: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether every pre-registered gate passed."""
        return not self.gate_failures


class Calibrator:
    """Runs every measuring function over one corpus and assembles the calibration.

    Args:
        corpus: The suite's demos and built graph.
        spec: The quantiles and sample sizes this run uses.
        contract: The graph contract whose dilation schedule is checked.
    """

    def __init__(
        self,
        corpus: DemoCorpus,
        spec: CalibrationSpec | None = None,
        contract: schema.GraphContract = schema.DEFAULTS,
    ) -> None:
        self.corpus = corpus
        self.spec = spec or CalibrationSpec()
        self.contract = contract
        # The action-unit fit reads every demo HDF5, so run and write share the one instance.
        self._action_scale: ActionScaleFit | None = None

    def action_scale_fit(self) -> ActionScaleFit:
        """The action-unit fit for this corpus, read from disk at most once per Calibrator."""
        if self._action_scale is None:
            self._action_scale = ActionScaleFit(self.corpus, self.spec)
        return self._action_scale

    def measure(self) -> Constants:
        """Run every procedure, in dependency order.

        Returns:
            The calibration, hashed over its constants alone. Gate failures are NOT raised here:
            the numbers are still the corpus's answer and are still worth reading -- see
            gate_failures, and CalibrationResult.ok.
        """
        nodes = self.corpus.nodes

        action_scale = self.action_scale_fit()
        spacing = NeighbourSpacing(nodes)
        reach = DilationReach(nodes, self.contract)
        seed_support = SeedSupport(self.corpus, self.spec, kernel_bw=spacing.spacing)

        entries: dict[str, Constant] = {}
        for measurement in (
            action_scale, spacing, reach, seed_support,
            EntryPerturbation(nodes, self.spec),
            EntryStepCap(nodes),
            RobotInitMagnitude(),
            SegmentLen(nodes, self.spec),
            # future_temp is deliberately not measured here -- see robot_ood.build.measure.FutureTemp.
            SegmentHardNegatives(nodes, self.spec),
            SegmentEntryStratum(nodes, self.spec),
            SegTier1M(),
        ):
            entries.update(measurement.constants())

        return Constants(entries=entries, corpus=self.corpus.fingerprint())

    def gate_failures(self, constants: Constants) -> tuple[str, ...]:
        """The pre-registered gates this calibration misses, one line each.

        Args:
            constants: A calibration from measure.

        Returns:
            An empty tuple when every gate passes.
        """
        failures: list[str] = []
        try:
            self.action_scale_fit().check()
        except ValueError as err:
            failures.append(f"action-scale R2: {err}")

        reach = constants["reach_raw_frames"]
        if not reach.stats["satisfied"]:
            failures.append(
                f"dilation reach: {reach.stats['layers']} layers x max dilation "
                f"{reach.stats['max_dilation']} x coarsen {reach.stats['coarsen']} reaches "
                f"{reach.value} raw frames, short of the median demo's "
                f"{reach.stats['median_raw_frames_per_demo']:.0f} -- a query at one end of a demo "
                "can never be informed by the other. Deepen the network or coarsen further; this "
                "gate does not choose a schedule for you."
            )
        return tuple(failures)

    def run(self, out_dir: str | Path | None = None) -> CalibrationResult:
        """Measure, then write action_scale.json (and constants.json, outside a graph dir).

        Args:
            out_dir: Destination directory. None writes beside the graph artifacts, which is where
                every consumer looks.

        Returns:
            The calibration, the paths written, and any gate failures. The files are written either
            way: a calibration that misses a gate is evidence, not a crash. constants_path is None
            when target holds a graph, because a constants.json there changes that graph's hash.
        """
        constants = self.measure()
        target = Path(out_dir) if out_dir is not None else self.corpus.graph_dir
        # action_scale.json stays a separate file: robot_ood.core.action.ActionScale is loaded on the
        # control path by a sentinel that has no reason to parse the whole calibration.
        scale_path = self.action_scale_fit().scale.to_json(target / schema.ACTION_SCALE_JSON)
        # constants.json is NOT written into a graph dir: graph_hash keys on its presence there, so
        # the file would change every later stamp. write_constants puts the deployable copy outside.
        inside_graph = (target / schema.NODES_NPZ).exists()
        return CalibrationResult(
            constants=constants,
            constants_path=None if inside_graph else constants.to_json(target),
            action_scale_path=Path(scale_path),
            gate_failures=self.gate_failures(constants),
        )


class OutsideGraphDir(ValueError):
    """A constants.json destination that would re-enter one of the two recorded traps."""


def check_destination(out: Path, *, force: bool) -> None:
    """Refuse a constants.json destination that a recorded failure says not to write.

    Args:
        out: The destination FILE.
        force: Allow overwriting an existing out.

    Raises:
        OutsideGraphDir: The destination sits beside a graph, or already exists without force.
    """
    # graph_hash keys on constants.json's PRESENCE in the graph dir, so dropping one there changes
    # every future stamp and makes already-trained checkpoints unloadable against their own graph.
    if (out.parent / schema.NODES_NPZ).exists():
        raise OutsideGraphDir(
            f"refusing to write into {out.parent}: it holds {schema.NODES_NPZ}, and a "
            "constants.json beside a graph changes that graph's hash (RUNS.md 5). Pick a path "
            "outside every artifact directory."
        )
    # A constants file a measured result points at is an artifact, not scratch space. RUNS.md run
    # 10: an in-place regeneration moved entry_servo_budget 16 -> 19 underneath a scored result.
    if out.exists() and not force:
        raise OutsideGraphDir(
            f"refusing to overwrite {out}: it already exists, and RUNS.md run 10 is what happens "
            "when a shared constants file is regenerated in place -- a measured result "
            "(entry_fcsr, 94.9% on Robot_Initial_States) silently stopped being reproducible from "
            "its own recorded recipe. Pick a new destination for a new calibration, or pass force "
            "if you have confirmed nothing measured is pinned to this file's current bytes."
        )


def write_constants(calibrator: "Calibrator", out: str | Path, *,
                    force: bool = False) -> tuple[Constants, CalibrationResult]:
    """Run C0 and write constants.json somewhere that is NOT a graph artifact directory.

    Args:
        calibrator: The configured run.
        out: Destination FILE for constants.json.
        force: Allow overwriting an existing out.

    Returns:
        (the calibration reloaded through the real reader, the run's own result).

    Raises:
        OutsideGraphDir: The destination is one the recorded traps forbid.
    """
    out = Path(out).resolve()
    check_destination(out, force=force)
    # A scratch dir, so the action_scale.json the Calibrator also emits never lands beside a graph.
    with tempfile.TemporaryDirectory() as scratch:
        result = calibrator.run(out_dir=scratch)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(scratch) / CONSTANTS_JSON, out)
    # Reload through the real reader: it re-checks the content hash, so a file that would be
    # rejected at deploy is rejected here instead.
    return Constants.from_json(out), result


def ensure_constants(suite: str, graph_dir: str | Path, log: "Callable[[str], None]" = print) -> Any:
    """Install a C0 calibration for this process, measuring it when nothing resolves one.

    Args:
        suite: Suite short-name whose demo corpus is measured when needed.
        graph_dir: The artifact directory holding the built graph.
        log: Progress sink.

    Returns:
        The robot_ood.core.constants.Constants the process now resolves.
    """
    from robot_ood.data.contract import resolved_contract
    from robot_ood.core.config import calibrated_constants, set_constants

    constants = calibrated_constants()
    if constants is not None:
        log(f"[fcsr] constants: resolved externally ({constants.content_hash[:19]})")
        return constants
    log(f"[fcsr] measuring C0 over suite {suite!r} (no ROBOT_OOD_CONSTANTS, no embedded calibration)")
    calibrator = Calibrator(DemoCorpus(suite, graph_dir=Path(graph_dir)),
                            contract=resolved_contract(suite))
    constants = calibrator.measure()
    for failure in calibrator.gate_failures(constants):
        log(f"[fcsr] C0 GATE FAILED -- {failure}")
    set_constants(constants)
    log(f"[fcsr] constants: measured ({constants.content_hash[:19]})")
    return constants
