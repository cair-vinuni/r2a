"""build graph: demo HDF5 -> g_nodes.npz + g_edges.npz for one suite."""

from __future__ import annotations

import dataclasses
import glob
import os
import shlex
import sys
from pathlib import Path

from robot_ood.data.store import record_artifacts
from robot_ood.cli.command import SuiteCommand, SuiteResult
from robot_ood.core.build_config import build_param
from robot_ood.core.config import GraphConfig
from robot_ood.data.libero import hdf5_name
from robot_ood.build.graph import LoadLimits, build_graph
from robot_ood.data import schema
from robot_ood.core.report import StageLogger

__all__ = ["GRAPH_STAGE_LABELS", "GraphCommand"]

from robot_ood.build.graph import GRAPH_STAGE_LABELS


class GraphCommand(SuiteCommand):
    """HDF5 demos -> g_nodes.npz / g_edges.npz for one suite."""

    stage = "graph"
    label = "build graph"
    done_status = "built"
    update_latest = True                # a build IS the "new graph" event latest exists to track
    table_headers = ("suite", "status", "n_nodes", "n_edges", "n_stages")

    def resolve_config(self, suite: str) -> dict:
        cfg = GraphConfig(
            coarsen=self.opt("coarsen", build_param("graph", "coarsen", schema.COARSEN, suite=suite)),
            dilations=tuple(int(d) for d in build_param(
                "graph", "dilations", list(schema.DILATIONS), suite=suite)),
            device=self.opt("device", ""),
            virtual_per_entry=self.opt("virtual_per_entry",
                                       build_param("graph", "virtual_per_entry",
                                                   schema.VIRTUAL_PER_ENTRY, suite=suite)),
        )
        hdf5_dir = self.opt("hdf5_dir", None)
        hdf5_dir = os.fspath(hdf5_dir) if hdf5_dir else os.fspath(self.paths.hdf5(hdf5_name(suite)))
        limits = {
            k: v for k, v in (
                ("limit_demos", self.opt("limit_demos", None)),
                ("limit_tasks", self.opt("limit_tasks", None)),
            ) if v is not None
        }
        return {
            **dataclasses.asdict(cfg),
            "dilations": list(cfg.dilations),   # tuple would defeat the JSON skip comparison
            "command": "build graph", "suite": suite, "hdf5_dir": hdf5_dir,
            "seed": self.opt("seed", 0), "name": self.args.name, "limits": limits,
        }

    def n_stages(self, config: dict) -> int:
        return len(GRAPH_STAGE_LABELS)

    def plan_line(self, suite: str, config: dict) -> str:
        return f"planned stages ({len(GRAPH_STAGE_LABELS)}): {', '.join(GRAPH_STAGE_LABELS)}"

    def stage_inputs(self, suite: str, config: dict) -> list[str]:
        return sorted(glob.glob(f"{config['hdf5_dir']}/*.hdf5"))

    def skip_note(self, hit: Path, config: dict) -> str:
        return f"identical config already built at {hit}"

    def execute(self, suite: str, config: dict, log: StageLogger) -> dict:
        cfg = GraphConfig(
            coarsen=config["coarsen"], dilations=tuple(config["dilations"]),
            device=config["device"], virtual_per_entry=config["virtual_per_entry"],
        )
        limits = config["limits"]
        artifacts = build_graph(
            suite, hdf5_dir=config["hdf5_dir"], out_dir=None, cfg=cfg, coarsen=cfg.coarsen,
            device=(cfg.device or None), logger=log,
            limits=LoadLimits(tasks=limits.get("limit_tasks"), demos=limits.get("limit_demos")),
            seed=config["seed"],
        )
        record_artifacts(
            artifacts.nodes_path.parent, stage="build", command=shlex.join(sys.argv),
            filenames=[schema.NODES_NPZ, schema.EDGES_NPZ],
        )
        self._seed_canonical(suite, artifacts.nodes_path.parent, log)
        return artifacts.to_dict()

    def _seed_canonical(self, suite: str, run_dir: Path, log: StageLogger) -> None:
        """Clone the just-built tables into the one path every later stage reads.

        Args:
            suite: Suite short-name.
            run_dir: The run directory build_graph wrote into.
            log: Where the copy report goes.
        """
        from robot_ood.build.graph import seed_graph_dir
        from robot_ood.core.paths import suite_config

        configured = suite_config(suite).get("artifacts")
        if self.out_root.resolve() != Path(self.paths.outputs()).resolve():
            dest = run_dir.parent.parent / "artifacts"
        elif configured:
            dest = self.paths.outputs(configured)
        else:
            dest = self.paths.outputs(suite, "artifacts")
        if dest.resolve() == run_dir.resolve():
            return
        result = seed_graph_dir(run_dir, dest)
        dropped = f", dropped {len(result.dropped)} stale" if result.dropped else ""
        log.log(f"[build graph] seeded {dest}: {len(result.copied)} artifact(s) "
                f"from {run_dir}{dropped}")

    def row(self, result: SuiteResult) -> list[str]:
        m = result.metrics
        return [result.suite, result.status,
                str(m.get("n_nodes", "-")), str(m.get("n_edges", "-")), str(m.get("n_stages", "-"))]
