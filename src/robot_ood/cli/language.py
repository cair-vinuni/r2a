"""build language: a suite's recorded task phrasings -> g_lang.npz beside its graph."""

from __future__ import annotations

from pathlib import Path

from robot_ood.cli.command import SuiteCommand, SuiteResult
from robot_ood.cli.errors import CLIError
from robot_ood.cli.render import fmt_metric
from robot_ood.data import schema
from robot_ood.data.libero import hdf5_name
from robot_ood.core.report import StageLogger

__all__ = ["LanguageCommand"]


class LanguageCommand(SuiteCommand):
    """The frozen encoder over each task's recorded phrasings, and the task head fit on them."""

    stage = "language"
    label = "build language"
    done_status = "embedded"
    update_latest = False               # g_lang.npz sits outside ArtifactSchemas.NODES
    table_headers = ("suite", "status", "tasks", "cross", "gates")

    def resolve_config(self, suite: str) -> dict:
        from robot_ood.build.language import DEFAULT_MAX_CROSS

        graph_dir = self.opt("graph_dir", None)
        graph_dir = Path(graph_dir) if graph_dir else Path(self.paths.graph(suite))
        hdf5_dir = self.opt("hdf5_dir", None)
        hdf5_dir = Path(hdf5_dir) if hdf5_dir else self.paths.hdf5(hdf5_name(suite))
        return {
            "command": "build language", "suite": suite, "graph_dir": str(graph_dir),
            "hdf5_dir": str(hdf5_dir), "device": self.opt("device", "") or "cpu",
            "model_dir": self.opt("model_dir", ""), "name": self.args.name,
            "max_cross": self.opt("max_cross", DEFAULT_MAX_CROSS),
        }

    def n_stages(self, config: dict) -> int:
        return 1

    def plan_line(self, suite: str, config: dict) -> str:
        from robot_ood.models.language.encoder import LANG_NPZ

        return f"planned: {LANG_NPZ} in {config['graph_dir']}"

    def stage_inputs(self, suite: str, config: dict) -> list[str]:
        return [str(Path(config["graph_dir"]) / schema.NODES_NPZ)]

    def validate(self, suite: str, config: dict) -> None:
        self.require_graph(suite, config)

    def artifacts_present(self, suite: str, config: dict) -> bool:
        from robot_ood.models.language.encoder import LANG_NPZ

        return (Path(config["graph_dir"]) / LANG_NPZ).exists()

    def execute(self, suite: str, config: dict, log: StageLogger) -> dict:
        from robot_ood.build.language import build_task_embeddings, measure_separation, save_task_embeddings
        from robot_ood.models.language.encoder import LANG_NPZ

        log.step("embed", config["graph_dir"])
        emb = build_task_embeddings(config["graph_dir"], config["hdf5_dir"],
                                    model_dir=config["model_dir"], device=config["device"])
        measured = measure_separation(emb, config["max_cross"])
        for line in measured.lines(emb):
            log.log(line)
        # Both gates are pre-registered and fatal: an embedding that fails them cannot tilt the
        # task lane apart at all, so writing it would quietly weaken retrieval.
        if not measured.passed and not self.args.force:
            raise CLIError(f"{suite}: the separation/distinctness gate failed -- pass --force to "
                           f"write {LANG_NPZ} anyway", exit_code=2)
        written = save_task_embeddings(emb, Path(config["graph_dir"]))
        log.metric(path=str(written), passed=measured.passed)
        return {"tasks": emb.n_tasks, "cross": measured.cross_mean,
                "gates": "pass" if measured.passed else "FORCED"}

    def row(self, result: SuiteResult) -> list[str]:
        m = result.metrics
        return [result.suite, result.status, str(m.get("tasks", "-")),
                fmt_metric(m.get("cross", "-")), str(m.get("gates", "-"))]
