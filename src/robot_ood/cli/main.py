"""The top-level parser and dispatch: python -m robot_ood.cli <command> ....

Exit codes: 0 success, 1 exception or bad input, 2 a pre-registered gate failed (build calibrate
and build language).
"""

from __future__ import annotations

import argparse
import sys

from robot_ood.cli.artifacts import ArtifactsCommand
from robot_ood.cli.bench import BenchCommand
from robot_ood.cli.build import BUILD_VERBS
from robot_ood.cli.cache_policy import CachePolicyCommand
from robot_ood.cli.errors import CLIError
from robot_ood.cli.evaluate import EvalCommand
from robot_ood.cli.pipeline import BuildAllCommand, TrainAllCommand
from robot_ood.data.store import REQUIRED_ARTIFACTS
from robot_ood.data.bucket import DEFAULT_BUCKET
from robot_ood.eval.bench import DEFAULT_BENCH, bench_ids
from robot_ood.pipeline.dag import PROFILES
from robot_ood.cli.train import TRAIN_TASKS, TrainCommand

__all__ = ["build_parser", "main"]


def _add_common_args(sub: argparse.ArgumentParser) -> None:
    """Attach the run-bookkeeping flags every subcommand shares."""
    sub.add_argument("--out", default=None, help="root run-bookkeeping dir (default: Paths.outputs(), "
                      "i.e. $ROBOT_OOD_OUTPUTS or <repo>/outputs)")
    sub.add_argument("--name", default="", help="label appended to this run's directory name")
    sub.add_argument("--seed", type=int, default=None, help="RNG seed (default: 0, or the resumed "
                      "run's seed under --resume)")
    sub.add_argument("--device", default=None, help='torch device, e.g. "cuda:0" or "cpu" '
                      '(default: auto -- cuda if available else cpu)')
    sub.add_argument("--quiet", action="store_true", help="suppress the mirrored stdout progress "
                      "lines (log.txt under outputs/ still gets every line)")
    sub.add_argument("--dry-run", action="store_true", help="print the fully-resolved config and the "
                      "planned stages, then exit 0 -- touches NOTHING under outputs/, so an expensive "
                      "run can be inspected before it starts")
    sub.add_argument("--force", action="store_true", help="re-run even if a completed run with the "
                      "identical resolved config already exists under outputs/")
    sub.add_argument("--resume", default=None, metavar="RUN_DIR", help="inherit RUN_DIR's "
                      "resolved config.json as defaults for any flag not explicitly passed here")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        A parser with command_cls in the namespace.
    """
    p = argparse.ArgumentParser(
        prog="python -m robot_ood.cli",
        description="The demonstration-graph subsystem's offline pipeline (see robot_ood.data.schema for "
        "the full WHY): build the graph, then train the retrieval network on it. Every subcommand "
        "resolves its full configuration through robot_ood.core.paths.Paths / configs/robot_ood.yaml (never a "
        "hardcoded path) and leaves a reproducible run under outputs/ (robot_ood.core.report.StageLogger).",
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    build = sub.add_parser(
        "build", help="raw demos -> the graph and everything that is a pure function of it",
        description="Bare `build --suite X` runs every build step that is stale, in order: graph, "
        "segments, calibrate, language, cache-policy. Name a verb (`build segments --suite X`) to "
        "run exactly that one. Each step re-runs when an input's HASH changed, not merely when an "
        "output file is missing; --dry-run prints each step and why it is or is not stale.",
    )
    verbs = build.add_subparsers(dest="verb", required=True, metavar="verb")
    ball = verbs.add_parser("all", help="every stale build step, in order")
    ball.add_argument("--suite", required=True,
                      help="a suite with a configs/trees/<name>.yaml, or all")
    ball.add_argument("--profile", default=None, choices=sorted(PROFILES),
                      help="full (default) | fast (1 epoch everywhere) | smoke (2 tasks x 5 demos, "
                      "its own outputs_smoke/ tree)")
    ball.add_argument("--port", type=int, default=None, help="policy server port for cache-policy")
    ball.add_argument("--gpu", default=None,
                      help="comma-separated CUDA devices for cache-policy's servers (default: the "
                           "suite's build.cache_policy.gpus)")
    _add_common_args(ball)
    ball.set_defaults(command_cls=BuildAllCommand, verb="all")
    for name, verb in BUILD_VERBS.items():
        bp = verbs.add_parser(name, help=f"-> {', '.join(verb.outputs)}",
                              description=verb.description)
        bp.add_argument("--suite", required=True,
                        help="a suite with a configs/trees/<name>.yaml, or all")
        if verb.flags is not None:
            verb.flags(bp)
        _add_common_args(bp)
        bp.set_defaults(command_cls=verb.command_cls, verb=name)

    train = sub.add_parser(
        "train", help="fit one task's head over a built graph",
        description="Bare `train --suite X` runs locator, entry and fcsr in order (stale ones "
        "only), then the deploy gate. One verb per robot_ood.train.tasks module: locator fits the "
        "network the belief comes out of, fcsr fits the retriever and the corrector the belief is "
        "carried into, and entry fits the two t = 0 nets. "
        "Each writes its checkpoints beside the graph it read. Requires `build` to have run "
        "first.",
    )
    tasks = train.add_subparsers(dest="task", required=True, metavar="task")
    tall = tasks.add_parser("all", help="locator, entry, fcsr, then the deploy gate")
    tall.add_argument("--suite", required=True, help="a suite with a configs/trees/<name>.yaml, or all")
    tall.add_argument("--profile", default=None, choices=sorted(PROFILES),
                      help="full (default) | fast (1 epoch everywhere)")
    _add_common_args(tall)
    tall.set_defaults(command_cls=TrainAllCommand, task="all")
    for name, task in TRAIN_TASKS.items():
        tp = tasks.add_parser(name, help=f"-> {', '.join(task.outputs)}")
        tp.add_argument("--suite", required=True, help="a suite with a configs/trees/<name>.yaml, or all")
        tp.add_argument("--epochs", type=int, default=None,
                        help=f"epochs (default: {task.epochs})")
        tp.add_argument("--n-query", type=int, default=None, help="training queries "
                        "(default: robot_ood.train.loop.N_QUERY)")
        tp.add_argument("--n-eval", type=int, default=None, help="held-out queries for per-epoch "
                        "metrics (default: robot_ood.train.loop.N_EVAL)")
        tp.add_argument("--graph-dir", default=None, help="override the graph dir to train on/into "
                        "(default: Paths.graph(<suite>))")
        if task.flags is not None:
            task.flags(tp)
        _add_common_args(tp)
        tp.set_defaults(command_cls=TrainCommand, task=name)

    cache = sub.add_parser(
        "cache-policy", help="the frozen policy's own chunks over the demo corpus -> a_pi_raw.npz",
        description="Bring a policy server up (unless one is already listening), cache its "
        "per-frame actions over this suite's corpus, and tear the server back down if and only "
        "if this command started it. Which server, cache client and conda env is the policy "
        "descriptor's business (evals/policies/<id>.yaml); ROBOT_OOD_PY overrides the interpreter.",
    )
    cache.add_argument("--suite", required=True, help="a suite with a configs/trees/<name>.yaml, or all")
    cache.add_argument("--policy", default=None,
                       help="policy descriptor under evals/policies/ (default: the suite's "
                       "configured policy, else stablevla)")
    cache.add_argument("--port", type=int, default=None, help="policy server port (default: 14190)")
    cache.add_argument("--gpu", default=None,
                       help="comma-separated CUDA devices the servers are placed on, round-robin "
                            "(default: the suite's build.cache_policy.gpus)")
    cache.add_argument("--graph-dir", default=None, help="override the graph dir to cache into "
                       "(default: Paths.graph(<suite>))")
    cache.add_argument("--limit-tasks", type=int, default=None, help="cap hdf5 files read -- must "
                       "match what `build graph` was limited to, or the row counts disagree")
    cache.add_argument("--limit-demos", type=int, default=None, help="cap demos read per hdf5 file")
    cache.add_argument("--clients", type=int, default=None,
                       help="replay the corpus through this many concurrent clients, demos dealt "
                            "round robin (default: the policy's placement times the GPUs)")
    cache.add_argument("--servers", type=int, default=None,
                       help="policy servers to spread those clients over, one port each "
                            "(default: the policy's placement times the GPUs)")
    _add_common_args(cache)
    cache.set_defaults(command_cls=CachePolicyCommand)

    art = sub.add_parser(
        "artifacts", help="the stage gate, seeding a graph directory, and moving one to the bucket",
        description="Directory-level artifact work, so neither verb takes --suite: check reads a "
        "directory's ARTIFACTS.json and reports whether one stage's required set is present, "
        "unchanged and consistently stamped; seed clones a directory's non-FCSR artifacts into a "
        "fresh one, carrying each file's provenance record across so the copy stays provable.",
    )
    art_verbs = art.add_subparsers(dest="verb", required=True, metavar="verb")

    check = art_verbs.add_parser("check", help="-> exit 0 when the stage's set is satisfied")
    check.add_argument("graph_dir", help="the artifact directory to check")
    check.add_argument("--stage", required=True, choices=sorted(REQUIRED_ARTIFACTS),
                       help="which required set to check")
    check.add_argument("--graph-hash", default=None,
                       help="require every stamped artifact to carry exactly this graph_hash "
                       "(default: only require the stamped artifacts to agree with each other)")
    check.set_defaults(command_cls=ArtifactsCommand, verb="check")

    seed = art_verbs.add_parser("seed", help="-> a new artifact directory with provenance")
    seed.add_argument("source_dir", help="an existing, deploy-ready artifact directory")
    seed.add_argument("dest_dir", help="the new artifact directory to seed (created if absent)")
    seed.set_defaults(command_cls=ArtifactsCommand, verb="seed")

    fetch = art_verbs.add_parser("fetch", help="-> outputs/<suite>/ from the project's bucket")
    fetch.add_argument("source", help="the subdirectory inside the bucket, e.g. long")
    fetch.add_argument("--suite", default="long", help="which outputs/<suite>/ to install into "
                       "(default: long)")
    fetch.add_argument("--bucket", default=DEFAULT_BUCKET,
                       help=f"bucket id (default: {DEFAULT_BUCKET})")
    fetch.set_defaults(command_cls=ArtifactsCommand, verb="fetch")

    push = art_verbs.add_parser("push", help="-> outputs/<suite>/ published to the project's bucket")
    push.add_argument("dest", help="the subdirectory inside the bucket, e.g. long")
    push.add_argument("--suite", default="long", help="which outputs/<suite>/ to publish "
                      "(default: long)")
    push.add_argument("--bucket", default=DEFAULT_BUCKET,
                      help=f"bucket id (default: {DEFAULT_BUCKET})")
    push.set_defaults(command_cls=ArtifactsCommand, verb="push")

    bench = sub.add_parser(
        "bench", help="the registered simulator benchmarks, and provisioning one",
        description="A benchmark is a directory under evals/ holding a bench.yaml. list reports "
        "what is registered and which checkouts are present; setup runs the provisioning script "
        "that benchmark declares, and is idempotent.",
    )
    bench_verbs = bench.add_subparsers(dest="verb", required=True, metavar="verb")

    bench_list = bench_verbs.add_parser("list", help="-> every benchmark, its axes, its install state")
    bench_list.set_defaults(command_cls=BenchCommand, verb="list")

    bench_setup = bench_verbs.add_parser("setup", help="-> a provisioned benchmark checkout")
    bench_setup.add_argument("bench", choices=bench_ids(), help="which benchmark to provision")
    bench_setup.set_defaults(command_cls=BenchCommand, verb="setup")

    ev = sub.add_parser(
        "eval", help="run the success-rate ladder (scripts/run_sr.py is the backend)",
        description="Drive one rung of configs/sr_ladder.yaml, or with --cell exactly one cell of "
        "the smoke rung -- the cheapest way to learn whether an install is complete. --cell rebuilds "
        "the reduced ladder itself, recounting expected_n against the LIBERO-plus actually checked "
        "out, because the ladder's own number is the author's.",
    )
    ev.add_argument("--suite", required=True, help="a suite with a configs/trees/<name>.yaml")
    ev.add_argument("--mode", required=True, help="base | fcsr")
    ev.add_argument("--axis", default=None, help="perturbation axis, e.g. Robot_Initial_States "
                    "(required with --cell)")
    ev.add_argument("--cell", action="store_true", help="run ONE cell of the smoke rung")
    ev.add_argument("--rung", default=None, help="ladder rung to run without --cell "
                    "(default: the benchmark's own, R1 for libero_plus)")
    ev.add_argument("--bench", default=None,
                    help=f"benchmark to evaluate on: {' | '.join(bench_ids())} "
                         f"(default: {DEFAULT_BENCH}, or $ROBOT_OOD_BENCH)")
    ev.add_argument("--trials", type=int, default=None, help="episodes per task, overriding the "
                    "benchmark's default -- the cheap way to shorten a cell")
    ev.add_argument("--seed", type=int, default=None, help="episode seed base handed to every "
                    "client, overriding the protocol's -- a validation split of the same suite")
    ev.add_argument("--variant", default=None, help="tag suffix, so a run against another artifact "
                    "tree gets its own result dir (GR_GRAPH_DIR picks the tree)")
    ev.add_argument("--gpus", default="0", help="comma-separated physical GPU indices")
    ev.add_argument("--per-gpu", type=int, default=1, help="cells run concurrently on each GPU, "
                    "each with its own policy server (a batch-1 server keeps an A100 about half busy)")
    ev.add_argument("--num-clients", type=int, default=None, help="sim clients per cell, overriding "
                    "the ladder's num_clients; one server saturates at about two GPU-rendered clients")
    ev.add_argument("--video", choices=("none", "failures", "all"), default="none",
                    help="replay videos to keep beside episodes.jsonl (default: none; the records "
                         "are the score)")
    ev.add_argument("--servers-per-cell", type=int, default=None, help="policy servers per cell, "
                    "its clients sharded over them; a lone cell spreads them over the run's idle GPUs")
    ev.add_argument("--policy", default=None, help="policy descriptor under evals/policies/ "
                    "(default: the suite tree's own, e.g. gr00t for suite long via --policy gr00t)")
    ev.add_argument("--ladder", default=None, help="path to a ladder yaml (default: "
                    "configs/sr_ladder.yaml)")
    ev.add_argument("--artifacts", default=None, help="artifact directory to evaluate "
                    "(default: the suite's canonical outputs/<suite>/artifacts); fcsr must pass "
                    "the deploy gate before anything launches")
    ev.add_argument("--results-root", default=None, help="override the results/ root -- re-running a "
                    "cell in place is refused, so point this somewhere new instead of deleting")
    ev.add_argument("--dry-run", action="store_true", help="print the schedule; launch nothing")
    ev.add_argument("--resume", action="store_true", help="skip complete cells; run a partial one's missing episodes")
    ev.add_argument("--rerun-incomplete", action="store_true",
                    help="purge and re-run partial cells")
    ev.add_argument("--score", action="store_true", help="score existing results only")
    ev.add_argument("--check-env", action="store_true", help="GPU/load/benchmark preflight "
                    "report; --suite/--mode are accepted but ignored")
    ev.add_argument("--allow-partial", action="store_true", help="include incomplete cells in totals")
    ev.set_defaults(command_cls=EvalCommand)

    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Argument list; None uses sys.argv[1:].

    Returns:
        The process exit code.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    # Bare build / train mean "run the whole verb group": route them to the all subverb.
    if argv and argv[0] in ("build", "train") and (len(argv) == 1 or argv[1].startswith("-")):
        argv.insert(1, "all")
    args = build_parser().parse_args(argv)
    try:
        return args.command_cls(args).run()
    except CLIError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
