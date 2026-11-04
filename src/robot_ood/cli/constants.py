"""build constants: C0 measured into a scratch constants.json, deliberately away from the graph."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from robot_ood.cli.command import SuiteCommand, SuiteResult
from robot_ood.cli.errors import CLIError
from robot_ood.data.libero import hdf5_name
from robot_ood.data import schema
from robot_ood.core.report import StageLogger

__all__ = ["ConstantsCommand"]


class ConstantsCommand(SuiteCommand):
    """The same measurement build calibrate runs, written to a path outside every graph dir."""

    stage = "constants"
    label = "build constants"
    done_status = "written"
    update_latest = False               # scratch output; it belongs to no graph
    table_headers = ("suite", "status", "constants", "hash", "gates")

    def resolve_config(self, suite: str) -> dict:
        from robot_ood.build.measure import CalibrationSpec

        graph_dir = self.opt("graph_dir", None)
        graph_dir = Path(graph_dir) if graph_dir else Path(self.paths.graph(suite))
        hdf5_dir = self.opt("hdf5_dir", None)
        hdf5_dir = os.fspath(hdf5_dir) if hdf5_dir else os.fspath(self.paths.hdf5(hdf5_name(suite)))
        dest = self.opt("dest", None) or str(self.out_root / f"constants_{suite}.json")
        return {
            "command": "build constants", "suite": suite, "graph_dir": str(graph_dir),
            "hdf5_dir": hdf5_dir, "dest": dest, "name": self.args.name,
            "spec": dataclasses.asdict(CalibrationSpec(n_queries=self.opt("n_queries", 256))),
        }

    def n_stages(self, config: dict) -> int:
        return 1

    def plan_line(self, suite: str, config: dict) -> str:
        return f"planned: {config['dest']} (measured against {config['graph_dir']})"

    def stage_inputs(self, suite: str, config: dict) -> list[str]:
        return [str(Path(config["graph_dir"]) / schema.NODES_NPZ)]

    def validate(self, suite: str, config: dict) -> None:
        from robot_ood.build.calibration import OutsideGraphDir, check_destination

        self.require_graph(suite, config)
        try:
            check_destination(Path(config["dest"]).resolve(), force=self.args.force)
        except OutsideGraphDir as err:
            raise CLIError(str(err)) from err

    def artifacts_present(self, suite: str, config: dict) -> bool:
        return Path(config["dest"]).exists()

    def execute(self, suite: str, config: dict, log: StageLogger) -> dict:
        from robot_ood.build.calibration import Calibrator, write_constants
        from robot_ood.build.measure import CalibrationSpec
        from robot_ood.data.contract import resolved_contract
        from robot_ood.data.libero import DemoCorpus

        corpus = DemoCorpus(suite, hdf5_dir=config["hdf5_dir"], graph_dir=config["graph_dir"],
                            paths=self.paths)
        log.step("measure", config["dest"])
        constants, result = write_constants(
            Calibrator(corpus, CalibrationSpec(**config["spec"]),
                       contract=resolved_contract(suite)), config["dest"],
            force=self.args.force,
        )
        log.metric(dest=config["dest"], content_hash=constants.content_hash)
        for name in sorted(constants.entries):
            entry = constants[name]
            log.log(f"{name} = {entry.value!r} [{entry.kind}]")
        # action_scale.json belongs beside the graph and build calibrate already recorded it there,
        # so the scratch copy this run made is dropped with its temp dir.
        for failure in result.gate_failures:
            log.log(f"GATE FAILED -- {failure}")
        return {"constants": len(constants.entries), "hash": constants.content_hash,
                "gates": "pass" if result.ok else f"{len(result.gate_failures)} failed"}

    def row(self, result: SuiteResult) -> list[str]:
        m = result.metrics
        return [result.suite, result.status, str(m.get("constants", "-")),
                str(m.get("hash", "-"))[:12], str(m.get("gates", "-"))]
