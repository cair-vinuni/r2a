"""Data-parallel `train locator`: one rank per GPU, gradients averaged, rank 0 writes.

    torchrun --nproc_per_node=4 scripts/research/train_locator_mp.py --suite g1 --epochs 12

The CLI's verb stays single-process: its run directory is timestamped, so four ranks would race on
one path. This drives robot_ood.train.loop.train_graph directly; the Runner reads the launcher's
environment (robot_ood.train.parallel.world), splits each epoch's row budget across the ranks and
averages their gradients, so an epoch is the same data and the same arithmetic on any world size.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from robot_ood.core.config import GraphConfig
from robot_ood.core.device import resolve_device
from robot_ood.core.paths import default_paths
from robot_ood.core.train_config import train_param
from robot_ood.data.store import record_artifacts
from robot_ood.train.parallel import world


class RankLogger:
    """Prints one line per call, stamped with the rank and the wall clock since start."""

    def __init__(self, rank: int) -> None:
        self.rank, self.t0 = int(rank), time.monotonic()

    def log(self, message: str) -> None:
        print(f"[rank{self.rank} {time.monotonic() - self.t0:7.1f}s] {message}", flush=True)

    def step(self, *args: object, **kwargs: object) -> None:
        self.log(" ".join(str(a) for a in args))

    def metric(self, **kwargs: object) -> None:
        return None


def main(argv: list[str] | None = None) -> int:
    """Train one suite's locator across every rank the launcher started."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True)
    p.add_argument("--epochs", type=int, default=None, help="default: the suite's own recipe")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-query", type=int, default=None, help="rows per epoch, split across the ranks")
    p.add_argument("--graph-dir", default=None, help="default: the suite's artifact directory")
    args = p.parse_args(argv)

    import torch

    from robot_ood.eval.phase_skew import embed_report
    from robot_ood.train.loop import N_EVAL, N_QUERY, train_graph
    from robot_ood.train.types import TrainGraphSpec

    # A100 tensor cores for the dense layers: the same arithmetic to within float32 rounding.
    torch.set_float32_matmul_precision("high")
    par = world()
    log = RankLogger(par.rank)
    graph_dir = Path(args.graph_dir or default_paths().graph(args.suite))
    epochs = int(args.epochs or train_param("locator", "epochs", 30, suite=args.suite))
    device = par.device(resolve_device(None))
    cfg = GraphConfig(device=str(device))
    log.log(f"{args.suite}: locator {epochs} epochs on {device}, world {par.world_size}")
    result = train_graph(graph_dir, TrainGraphSpec(
        stages=(1,), epochs=epochs, seed=args.seed, device=str(device), save=True, cfg=cfg,
        logger=log, n_query=int(args.n_query or N_QUERY), n_eval=N_EVAL, task="locator"))
    if par.is_main:
        embed_report(graph_dir, device=str(device), cfg=cfg, log=log.log)
        record_artifacts(graph_dir, stage="locator",
                         command=f"torchrun train_locator_mp.py --suite {args.suite}",
                         filenames=["g_head.npz"])
        log.log(f"done: {result.checkpoint_path}")
    par.barrier()
    par.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
