"""train <task>: one verb per robot_ood.train.tasks module.

Adding a task is a file plus a TRAIN_TASKS row: the subparser, the defaults and the skip guard all
read that row. Nothing here branches on which task is running.
"""

from __future__ import annotations

import argparse
import dataclasses
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from robot_ood.data.store import record_artifacts
from robot_ood.cli.command import SuiteCommand, SuiteResult
from robot_ood.cli.errors import CLIError
from robot_ood.cli.render import fmt_metric
from robot_ood.core.config import Ablation, GraphConfig
from robot_ood.core.train_config import train_param
from robot_ood.data import schema
from robot_ood.core.report import StageLogger

__all__ = ["TRAIN_TASKS", "TrainCommand", "TrainTask"]


@dataclass(frozen=True)
class TrainTask:
    """One train verb.

    Attributes:
        epochs: Default --epochs.
        outputs: The artifact filenames this verb writes.
        inputs: The artifact filenames this verb reads, beyond the node and edge tables. A change
            to any of them makes a prior run of the same config stale.
        run: Called as run(cmd, suite, config, log) and returns the summary metrics.
        flags: Called with this verb's parser to add the flags only it takes, or None.
        summary: What this verb's row shows beyond suite and status.
    """

    epochs: int
    outputs: tuple[str, ...]
    run: Callable[..., dict]
    flags: Callable[[argparse.ArgumentParser], None] | None = None
    summary: tuple[str, ...] = ("top1", "mrr")
    inputs: tuple[str, ...] = ()


# ---- the engines ---------------------------------------------------------------------------------
def _run_graph(cmd: TrainCommand, suite: str, config: dict, log: StageLogger) -> dict:
    """retrieval: the demo-graph curriculum through robot_ood.train.loop."""
    from robot_ood.train.loop import train_graph
    from robot_ood.train.types import TrainGraphSpec

    from robot_ood.core.device import resolve_device
    from robot_ood.eval.phase_skew import embed_report
    from robot_ood.cli.run_config import dump_run_config, update_run_config
    from robot_ood.core.env import env_bool
    from robot_ood.train.tasks import locator as locator_task

    cfg = cmd.env_cfg(config)
    single = env_bool("GR_SINGLE_CHECK", False)
    ablation = Ablation.from_env()
    dump_run_config(
        config["graph_dir"],
        {
            "stage_a": locator_task.STAGE_A, "qbatch": config["qbatch"],
            "joint_tol": cfg.joint_tol, "seed_topk": cfg.seed_topk, "single_check": single,
            "ablation": str(ablation), "no_memory": ablation.memory,
            "epochs": config["epochs"], "seed": config["seed"],
            "n_query": config["n_query"], "n_eval": config["n_eval"],
            "suite": suite, "device": config["device"],
        },
        suite=suite, log=log.log,
    )
    result = train_graph(
        Path(config["graph_dir"]),
        TrainGraphSpec(stages=(1,), epochs=config["epochs"], seed=config["seed"],
                       device=(config["device"] or None), save=True, cfg=cfg,
                       logger=log, n_query=config["n_query"], n_eval=config["n_eval"],
                       task="single" if single else "locator", no_memory=ablation.memory),
    )
    embed_report(config["graph_dir"], device=str(resolve_device(config["device"] or None)),
                 cfg=cfg, log=log.log)
    log.metric(checkpoint_path=result.checkpoint_path, graph_hash=result.graph_hash)
    final = _final_epoch(result.history)
    update_run_config(config["graph_dir"], {
        "best_epoch": final.get("best_epoch"),
        "metric": {k: v for k, v in final.items() if isinstance(v, (int, float))},
        "tie_break": "lowest task.select score (robot_ood.train.runner.BestEpoch)",
    })
    return {"checkpoint_path": result.checkpoint_path, **final}


def _run_fcsr(cmd: TrainCommand, suite: str, config: dict, log: StageLogger) -> dict:
    """fcsr: the retriever and the corrector as one staged objective."""
    from robot_ood.build.calibration import ensure_constants
    from robot_ood.train.fcsr import FcsrTrainSpec
    from robot_ood.train.tasks.fcsr import FcsrTask

    ensure_constants(suite, config["graph_dir"], log=log.log)
    task = FcsrTask(FcsrTrainSpec(
        epochs=config["epochs"], seed=config["seed"], device=(config["device"] or None),
        n_query=config["n_query"], n_eval=config["n_eval"],
        qbatch=None if config["fcsr_qbatch"] is None else int(config["fcsr_qbatch"]),
        language=config["seg_language"], ablation=Ablation.from_env(),
    ))
    return cmd.runner_metrics(task, config, log)



def _run_entry(cmd: TrainCommand, suite: str, config: dict, log: StageLogger) -> dict:
    """entry: §10's two t = 0 nets, fit back to back off the same node table."""
    from robot_ood.train.tasks.entry import InvDynTask, ScoreTask

    metrics: dict = {}
    for task in (ScoreTask(seed=config["seed"]), InvDynTask(seed=config["seed"])):
        log.step(task.name, f"{config['epochs']} epochs")
        metrics.update({f"{task.name}_{k}": v
                        for k, v in cmd.runner_metrics(task, config, log).items()})
    return metrics


# ---- the registry: filenames are literals because argparse needs them before a task module loads
# (test_cli.py::test_train_task_outputs_match_their_modules keeps them equal to the real constants)
TRAIN_TASKS: dict[str, TrainTask] = {
    "locator": TrainTask(30, (schema.HEAD_NPZ,), _run_graph,
                         summary=("top1_target_masked", "phase_mae_masked", "task_acc_masked", "recovery_masked"),
                         inputs=(schema.A_PI_RAW_NPZ, schema.ACTION_SCALE_JSON, "g_lang.npz")),
    "fcsr": TrainTask(30, (schema.FCSR_NPZ,),
                      _run_fcsr,
                      summary=("seg_deploy_top1", "corr_mse_learned"),
                      inputs=("g_segments.npz", schema.HEAD_NPZ)),
    "entry": TrainTask(6_000, ("g_score.npz", "g_invdyn.npz"), _run_entry, summary=("loss",)),
}


def _final_epoch(history: dict[Any, list[Any]]) -> dict:
    """The last stage's epoch == "final" record out of a training history, or {}."""
    final: dict = {}
    for stage_hist in history.values():
        if stage_hist and isinstance(stage_hist[-1], dict) and stage_hist[-1].get("epoch") == "final":
            final = stage_hist[-1]
    return final


class TrainCommand(SuiteCommand):
    """Fit one task's head over a built graph and write it beside that graph."""

    stage = "train"
    label = "train"
    done_status = "trained"
    update_latest = False               # a train run must not steal latest from the build it read

    @property
    def table_headers(self) -> tuple[str, ...]:
        """The summary header, taken from the verb's own row rather than fixed for all eight."""
        return ("suite", "status", *self.task.summary)

    @property
    def task(self) -> TrainTask:
        """The registry row for the verb this invocation named."""
        return TRAIN_TASKS[self.args.task]

    def env_cfg(self, config: dict) -> GraphConfig:
        """GraphConfig.from_env with this run's device, so every other GR_* knob stays deployed."""
        return dataclasses.replace(GraphConfig.from_env(), device=config["device"])

    def resolve_config(self, suite: str) -> dict:
        # Imported here, not at module scope: train --help and every build run would otherwise pay
        # for importing torch through robot_ood.train.loop.
        from robot_ood.core.protocol import suite_chunk
        from robot_ood.train.loop import N_EVAL, N_QUERY

        cfg = GraphConfig(device=self.opt("device", ""))
        graph_dir = self.opt("graph_dir", None)
        graph_dir = Path(graph_dir) if graph_dir else Path(self.paths.graph(suite))
        # asdict FIRST -- see BuildCommand.resolve_config: GraphConfig.graph_dir would otherwise
        # overwrite the resolved graph_dir below with "".
        return {
            **dataclasses.asdict(cfg),
            "dilations": list(cfg.dilations),   # tuple would defeat the JSON skip comparison
            "command": "train", "task": self.args.task, "suite": suite,
            "graph_dir": str(graph_dir),
            # The rows the deployed policy executes per query: the corrector is trained on chunks
            # cut to it, so a run under a different chunk is a different run.
            "policy_chunk": suite_chunk(suite),
            "epochs": self.opt("epochs", train_param(self.args.task, "epochs",
                                                     self.task.epochs, suite=suite)),
            "n_query": self.opt("n_query", N_QUERY), "n_eval": self.opt("n_eval", N_EVAL),
            "seg_language": bool(train_param("fcsr", "language", 1, suite=suite)),
            "fcsr_qbatch": train_param("fcsr", "qbatch", None, suite=suite),
            "ablation": str(Ablation.from_env()),
            "seed": self.opt("seed", 0), "name": self.args.name,
        }

    def n_stages(self, config: dict) -> int:
        return 1

    def plan_line(self, suite: str, config: dict) -> str:
        from robot_ood.core.kinematics import is_joint_space

        robot = "panda" if is_joint_space(suite) else "proxy(ee_from_obs)"
        return (f"planned: train {config['task']} for {config['epochs']} epochs "
                f"[robot={robot}] -> {', '.join(self.task.outputs)} in {config['graph_dir']}")

    def stage_inputs(self, suite: str, config: dict) -> list[str]:
        graph_dir = Path(config["graph_dir"])
        names = (schema.EDGES_NPZ, *self.task.inputs)
        return [str(graph_dir / n) for n in names if (graph_dir / n).exists()]

    def validate(self, suite: str, config: dict) -> None:
        graph_dir = Path(config["graph_dir"])
        if not (graph_dir / schema.NODES_NPZ).exists():
            raise CLIError(f"no built graph for suite {suite!r} at {graph_dir} -- run "
                           f"`python -m robot_ood.cli build graph --suite {suite}` first")

    def artifacts_present(self, suite: str, config: dict) -> bool:
        graph_dir = Path(config["graph_dir"])
        return all((graph_dir / name).exists() for name in self.task.outputs)

    def skip_note(self, hit: Path, config: dict) -> str:
        written = ", ".join(str(Path(config["graph_dir"]) / n) for n in self.task.outputs)
        return f"identical config already trained at {hit}, wrote {written}"

    def runner_metrics(self, task: Any, config: dict, log: StageLogger) -> dict:
        """Run one Task-protocol objective through the one Runner and return its final metrics.

        Args:
            task: The objective.
            config: The resolved run configuration.
            log: Where progress goes.

        Returns:
            The winning epoch's metrics.
        """
        from robot_ood.data.store import ArtifactStore
        from robot_ood.core.device import resolve_device
        from robot_ood.train.runner import Runner, RunSpec

        run = Runner(
            RunSpec(epochs=config["epochs"], seed=config["seed"], save=True,
                    log_every=max(1, config["epochs"] // 10)),
            log,
        ).run(task, ArtifactStore(config["graph_dir"]), cfg=self.env_cfg(config),
              device=resolve_device(config["device"] or None), evalset=object())
        return dict(run.final)

    def execute(self, suite: str, config: dict, log: StageLogger) -> dict:
        metrics = self.task.run(self, suite, config, log)
        record_artifacts(
            config["graph_dir"], stage=config["task"], command=shlex.join(sys.argv),
            filenames=list(self.task.outputs),
        )
        return metrics

    def row(self, result: SuiteResult) -> list[str]:
        m = result.metrics
        return [result.suite, result.status,
                *(fmt_metric(m.get(k, "-")) for k in self.task.summary)]
