"""Data-parallel training: every rank runs the same loop on its own rows, gradients averaged.

Not torch DDP: an objective here calls backward once per check, which DDP's reducer refuses. The
gradients are all-reduced explicitly between backward and the optimiser step instead, which is the
same arithmetic with none of the hooks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import torch

__all__ = ["Parallel", "world"]


@dataclass(frozen=True)
class Parallel:
    """One rank's place in the run.

    Attributes:
        rank: Global rank, 0 for a single-process run.
        world_size: Number of ranks.
        local_rank: Rank within this box, which picks the CUDA device.
    """

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        """Whether more than one rank is running."""
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """Whether this rank logs, evaluates and saves."""
        return self.rank == 0

    def device(self, fallback: Any) -> Any:
        """The device this rank trains on: its own CUDA card, or fallback when running alone."""
        return torch.device(f"cuda:{self.local_rank}") if self.enabled else fallback

    def seed(self, base: int) -> int:
        """A per-rank data seed, so the ranks draw different rows from the same corpus."""
        return int(base) + 1013 * self.rank

    def average_grads(self, params: Iterable[torch.nn.Parameter]) -> None:
        """Replace every gradient by its mean over the ranks, in place.

        Args:
            params: The parameters about to be stepped. A parameter with no gradient this step is
                given a zero one first: the ranks must all-reduce the same tensors in the same
                order, and a rank whose batch never touched a head would otherwise hang.
        """
        if not self.enabled:
            return
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            p.grad /= self.world_size

    def broadcast(self, module: torch.nn.Module) -> None:
        """Copy rank 0's weights and buffers onto every rank."""
        if not self.enabled:
            return
        for tensor in list(module.state_dict().values()):
            if tensor.is_floating_point() or tensor.dtype in (torch.int64, torch.int32):
                torch.distributed.broadcast(tensor, src=0)

    def barrier(self) -> None:
        """Wait for every rank; a no-op when running alone."""
        if self.enabled:
            torch.distributed.barrier()

    def mean(self, value: float) -> float:
        """The mean of one scalar over the ranks."""
        if not self.enabled:
            return float(value)
        t = torch.tensor([float(value)], device=self.device(torch.device("cpu")))
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return float(t.item() / self.world_size)

    def shutdown(self) -> None:
        """Tear the process group down, if one was built."""
        if self.enabled and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def world() -> Parallel:
    """This process's rank, from the launcher's environment; initialises the process group.

    Returns:
        The Parallel handle. WORLD_SIZE unset or 1 means a plain single-process run, and no
        process group is created.
    """
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if size <= 1:
        return Parallel()
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=size)
    return Parallel(rank=rank, world_size=size, local_rank=local)
