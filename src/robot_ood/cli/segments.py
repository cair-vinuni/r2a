"""build segments: an already-built graph -> g_segments.npz beside it."""

from __future__ import annotations

from pathlib import Path

from robot_ood.cli.command import SuiteCommand, SuiteResult
from robot_ood.core.build_config import build_param
from robot_ood.data import schema
from robot_ood.core.report import StageLogger

__all__ = ["SegmentsCommand"]


class SegmentsCommand(SuiteCommand):
    """C1's segment memory, a pure function of what build graph already wrote."""

    stage = "segments"
    label = "build segments"
    done_status = "built"
    update_latest = False               # segments extend a graph; they do not make a new one
    table_headers = ("suite", "status", "n_segments", "segment_len")

    def resolve_config(self, suite: str) -> dict:
        graph_dir = self.opt("graph_dir", None)
        graph_dir = Path(graph_dir) if graph_dir else Path(self.paths.graph(suite))
        return {
            "command": "build segments", "suite": suite, "graph_dir": str(graph_dir),
            "segment_len": self.opt("segment_len",
                                    build_param("segments", "segment_len", None, suite=suite)),
            "stride": self.opt("stride", build_param("segments", "stride", None, suite=suite)),
            "name": self.args.name,
        }

    def n_stages(self, config: dict) -> int:
        return 1

    def plan_line(self, suite: str, config: dict) -> str:
        return f"planned: g_segments.npz in {config['graph_dir']}"

    def stage_inputs(self, suite: str, config: dict) -> list[str]:
        return [str(Path(config["graph_dir"]) / schema.NODES_NPZ)]

    def validate(self, suite: str, config: dict) -> None:
        self.require_graph(suite, config)

    def artifacts_present(self, suite: str, config: dict) -> bool:
        from robot_ood.core.schemas import SEGMENTS_NPZ

        return (Path(config["graph_dir"]) / SEGMENTS_NPZ).exists()

    def execute(self, suite: str, config: dict, log: StageLogger) -> dict:
        from robot_ood.build.segments import build_segments

        log.step("segments", config["graph_dir"])
        result = build_segments(config["graph_dir"], segment_len=config["segment_len"],
                                stride=config["stride"])
        log.log(result.summary())
        return {"n_segments": result.n_segments, "segment_len": result.segment_len,
                "graph_hash": result.graph_hash}

    def row(self, result: SuiteResult) -> list[str]:
        m = result.metrics
        return [result.suite, result.status, str(m.get("n_segments", "-")),
                str(m.get("segment_len", "-"))]
