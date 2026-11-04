"""calibrate: C0 -- demo corpus -> constants.json + action_scale.json beside the graph."""

from __future__ import annotations

import dataclasses
import os
import shlex
import sys
from pathlib import Path

from robot_ood.data.store import record_artifacts
from robot_ood.cli.command import SuiteCommand, SuiteResult
from robot_ood.cli.errors import CLIError
from robot_ood.cli.render import fmt_metric
from robot_ood.data.libero import hdf5_name
from robot_ood.data import schema
from robot_ood.core.report import StageLogger

__all__ = ["CalibrateCommand"]


class CalibrateCommand(SuiteCommand):
    """C0: demo corpus -> constants.json + action_scale.json beside the graph (robot_ood.build.calibration)."""

    stage = "calibrate"
    label = "build calibrate"
    done_status = "calibrated"
    update_latest = False               # C0 reads the graph latest points at; it does not build one
    table_headers = ("suite", "status", "pos_scale", "kernel_bw", "assumed")

    def resolve_config(self, suite: str) -> dict:
        from robot_ood.core.action import DEFAULT_MIN_R2
        from robot_ood.build.measure import CalibrationSpec

        graph_dir = self.opt("graph_dir", None)
        graph_dir = Path(graph_dir) if graph_dir else Path(self.paths.graph(suite))
        hdf5_dir = self.opt("hdf5_dir", None)
        hdf5_dir = os.fspath(hdf5_dir) if hdf5_dir else os.fspath(self.paths.hdf5(hdf5_name(suite)))
        return {
            "command": "build calibrate", "suite": suite, "graph_dir": str(graph_dir),
            "hdf5_dir": hdf5_dir, "name": self.args.name,
            "spec": dataclasses.asdict(CalibrationSpec(
                n_queries=self.opt("n_queries", 256),
                min_r2=float(self.opt("min_r2", DEFAULT_MIN_R2)),
            )),
        }

    def n_stages(self, config: dict) -> int:
        return 1

    def plan_line(self, suite: str, config: dict) -> str:
        return f"planned: constants.json + action_scale.json in {config['graph_dir']}"

    def stage_inputs(self, suite: str, config: dict) -> list[str]:
        return [str(Path(config["graph_dir"]) / schema.NODES_NPZ)]

    def validate(self, suite: str, config: dict) -> None:
        self.require_graph(suite, config)

    def artifacts_present(self, suite: str, config: dict) -> bool:
        return (Path(config["graph_dir"]) / schema.ACTION_SCALE_JSON).exists()

    def execute(self, suite: str, config: dict, log: StageLogger) -> dict:
        from robot_ood.build.calibration import Calibrator
        from robot_ood.data.contract import resolved_contract
        from robot_ood.core.constants import ASSUMED
        from robot_ood.data.libero import DemoCorpus
        from robot_ood.build.measure import CalibrationSpec

        spec = CalibrationSpec(**config["spec"])
        corpus = DemoCorpus(suite, hdf5_dir=config["hdf5_dir"], graph_dir=config["graph_dir"],
                            paths=self.paths)
        log.step("measure")
        # The reach gate must measure the graph this suite builds, not the module defaults.
        result = Calibrator(corpus, spec, contract=resolved_contract(suite)).run()
        constants = result.constants
        log.metric(content_hash=constants.content_hash,
                    constants_path=str(result.constants_path or "not written (graph dir)"),
                    action_scale_path=str(result.action_scale_path))
        for name in sorted(constants.entries):
            entry = constants[name]
            log.log(f"{name} = {entry.value!r} [{entry.kind}]")
        for failure in result.gate_failures:
            log.log(f"GATE FAILED -- {failure}")
        record_artifacts(
            config["graph_dir"], stage="calibrate", command=shlex.join(sys.argv),
            filenames=[Path(result.action_scale_path).name],
        )
        if not result.ok:
            raise CLIError(
                f"{suite}: {len(result.gate_failures)} pre-registered gate(s) failed after writing "
                f"{result.action_scale_path}:\n  " + "\n  ".join(result.gate_failures),
                exit_code=2,
            )
        return {
            "pos_scale": constants.value("pos_scale"), "kernel_bw": constants.value("kernel_bw"),
            "assumed": ", ".join(constants.of_kind(ASSUMED)) or "-",
            "content_hash": constants.content_hash,
        }

    def row(self, result: SuiteResult) -> list[str]:
        m = result.metrics
        return [result.suite, result.status,
                fmt_metric(m.get("pos_scale", "-")), fmt_metric(m.get("kernel_bw", "-")),
                str(m.get("assumed", "-"))]
