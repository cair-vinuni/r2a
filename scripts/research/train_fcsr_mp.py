"""Data-parallel `train fcsr`: one rank per GPU, gradients averaged, rank 0 writes.

    torchrun --nproc_per_node=4 scripts/research/train_fcsr_mp.py --suite g1 --epochs 20

Like scripts/research/train_locator_mp.py: the Runner reads the launcher's environment, splits the
epoch's rows across the ranks and averages their gradients, so the epoch is the same data on any
world size.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from robot_ood.build.calibration import ensure_constants
from robot_ood.core.config import GraphConfig
from robot_ood.core.device import resolve_device
from robot_ood.core.paths import default_paths
from robot_ood.core.train_config import train_param
from robot_ood.data.store import ArtifactStore, record_artifacts
from robot_ood.train.fcsr import FcsrTrainSpec
from robot_ood.train.parallel import world
from robot_ood.train.runner import Runner, RunSpec
from robot_ood.train.tasks.fcsr import FcsrTask

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_locator_mp import RankLogger  # noqa: E402  (a sibling script, not a package)


def main(argv: list[str] | None = None) -> int:
    """Train one suite's corrector across every rank the launcher started."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True)
    p.add_argument("--epochs", type=int, default=None, help="default: the suite's own recipe")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-query", type=int, default=None, help="rows per epoch, split across the ranks")
    args = p.parse_args(argv)

    torch.set_float32_matmul_precision("high")
    par = world()
    log = RankLogger(par.rank)
    graph_dir = str(default_paths().graph(args.suite))
    device = par.device(resolve_device(None))
    epochs = int(args.epochs or train_param("fcsr", "epochs", 30, suite=args.suite))
    if par.is_main:
        ensure_constants(args.suite, graph_dir, log=log.log)
    par.barrier()
    from robot_ood.train.loop import N_EVAL, N_QUERY

    task = FcsrTask(FcsrTrainSpec(
        epochs=epochs, seed=args.seed, device=str(device),
        n_query=int(args.n_query or N_QUERY), n_eval=N_EVAL,
        qbatch=train_param("fcsr", "qbatch", None, suite=args.suite),
        language=bool(train_param("fcsr", "language", 1, suite=args.suite)),
    ))
    log.log(f"{args.suite}: fcsr {epochs} epochs on {device}, world {par.world_size}")
    run = Runner(RunSpec(epochs=epochs, seed=args.seed, save=True, log_every=1), log,
                 parallel=par).run(
        task, ArtifactStore(graph_dir),
        cfg=GraphConfig(device=str(device)), device=device, evalset=object())
    if par.is_main:
        record_artifacts(graph_dir, stage="fcsr",
                         command=f"torchrun train_fcsr_mp.py --suite {args.suite}",
                         filenames=["g_fcsr.npz"])
        log.log(f"final {dict(run.final)}")
    par.barrier()
    par.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
