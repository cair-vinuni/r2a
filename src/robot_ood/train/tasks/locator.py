"""The locator fit: one network over check sequences, memory carried between checks under a curriculum."""

from __future__ import annotations

from typing import Any, Final, Iterator

import numpy as np
import torch

from robot_ood.core.constants import HELDOUT_STRIDE
from robot_ood.core.train_config import train_param
from robot_ood.data import schema
from robot_ood.models.retrieval.gnn import MEMORY_TOPK, save_checkpoint
from robot_ood.train.loss import NodeState, batch_loss, forward_check
from robot_ood.train.metrics import task_accuracy
from robot_ood.train.sequences import INSTR_MODES, Sequence, SequenceSampler, to_device
from robot_ood.train.task import Requirements, TrainContext, TrainUnit

__all__ = ["STAGE_A", "TEACHER_A", "SELF_B", "NONE_B", "WRONG_B", "N_EVAL_ROWS", "EVAL_CHECKS",
           "LocatorTask"]

STAGE_A: Final[int] = train_param("locator", "stage_a", 10, env="GR_STAGE_A")
TEACHER_A: Final[float] = 0.7
SELF_B: Final[float] = 0.7
NONE_B: Final[float] = 0.15
WRONG_B: Final[float] = 0.15
N_EVAL_ROWS: Final[int] = 128
EVAL_CHECKS: Final[int] = 8

NONE, TEACHER, SELF, WRONG = 0, 1, 2, 3


class LocatorTask:
    """Train the retriever over check sequences and select on the held-out self-memory rollout.

    Attributes:
        name: The task's registry name.
    """

    name = "locator"

    def __init__(self, gt: Any, sampler: SequenceSampler, net: Any, nodes: Any, edges: Any,
                 train_owners: np.ndarray, heldout_owners: np.ndarray, *, epochs: int,
                 n_rows: int, single_check: bool = False, no_memory: bool = False,
                 seed: int = 0) -> None:
        self.gt, self.sampler, self.net = gt, sampler, net
        self.nodes, self.edges = nodes, edges
        self.train_owners, self.heldout_owners = train_owners, heldout_owners
        self.epochs, self.n_rows, self.single_check = int(epochs), int(n_rows), bool(single_check)
        self.no_memory = bool(no_memory)
        self.rng = np.random.RandomState(seed)
        self.epoch = 0
        self.ctx: TrainContext | None = None
        self._eval: list[tuple[Sequence, np.ndarray]] = []

    def requires(self) -> Requirements:
        return Requirements(artifacts=(schema.NODES_NPZ, schema.EDGES_NPZ, schema.A_PI_RAW_NPZ,
                                       "g_lang.npz"))

    def build(self, ctx: TrainContext) -> TrainUnit:
        self.ctx = ctx
        self.net = self.net.to(ctx.device)
        batch = max(1, int(ctx.cfg.qbatch))
        per_regime = max(1, N_EVAL_ROWS // (batch * len(INSTR_MODES)))
        for mode in INSTR_MODES:
            for _ in range(per_regime):
                seq = self.sampler.draw(self.heldout_owners, batch, EVAL_CHECKS, rank0=True,
                                        instr_mode=mode, policy_clean=True)
                wrong = self.sampler.wrong_nodes(seq.checks[0].tgt, seq.demo)
                self._eval.append((seq, wrong))
        return TrainUnit(trained={"locator": self.net})

    def on_epoch(self, unit: TrainUnit, epoch: int) -> None:
        self.epoch = int(epoch)

    def batches(self, ctx: TrainContext, rng: Any) -> Iterator[Sequence]:
        batch = max(1, int(ctx.cfg.qbatch))
        # An epoch is a row budget, not a step count: on several ranks each draws its share, so
        # the same epoch is the same data whatever the world size.
        ranks = max(1, getattr(ctx.parallel, "world_size", 1))
        for _ in range(max(1, self.n_rows // (batch * ranks))):
            yield self.sampler.draw(self.train_owners, batch,
                                    n_checks=1 if self.single_check else None)

    # ---- the curriculum ----
    def _memory_modes(self, batch: int, first: bool) -> np.ndarray:
        rng = self.rng
        if self.epoch < STAGE_A:
            if first:
                return np.full(batch, NONE, dtype=np.int8)
            return np.where(rng.random_sample(batch) < TEACHER_A, TEACHER, NONE).astype(np.int8)
        t = (self.epoch - STAGE_A + 1) / max(1, self.epochs - STAGE_A)
        p_self = SELF_B * min(1.0, t)
        if first:
            return np.where(rng.random_sample(batch) < WRONG_B, WRONG, NONE).astype(np.int8)
        p = np.array([NONE_B, max(0.0, 1.0 - p_self - NONE_B - WRONG_B), p_self, WRONG_B])
        return rng.choice(4, size=batch, p=p / p.sum()).astype(np.int8)

    def _memory(self, modes: np.ndarray, seq: Sequence, k: int,
                prev_belief: torch.Tensor | None, dev: Any) -> tuple[torch.Tensor, torch.Tensor]:
        n_nodes = len(self.nodes)
        batch = modes.shape[0]
        mem = np.zeros((batch, n_nodes), dtype=np.float32)
        prev_tgt = seq.checks[k - 1].tgt if k > 0 else seq.checks[0].tgt
        rows = np.flatnonzero(modes == TEACHER)
        if rows.size:
            mem[rows] = self.sampler.dense_target(prev_tgt[rows], seq.has_instr[rows], n_nodes)
        rows = np.flatnonzero(modes == WRONG)
        if rows.size:
            wrong = self.sampler.wrong_nodes(prev_tgt[rows], seq.demo[rows])
            mem[rows] = self.sampler.dense_target(wrong, seq.has_instr[rows], n_nodes)
        mem_t = torch.as_tensor(mem, device=dev)
        rows = np.flatnonzero(modes == SELF)
        if rows.size and prev_belief is not None:
            mem_t[rows] = truncate(prev_belief[rows].detach())
        has = torch.as_tensor((modes != NONE).astype(np.float32), device=dev)
        return mem_t, has

    def _run(self, net: Any, seq: Sequence, memory_of, want_loss: bool,
             mask: torch.Tensor | None = None):
        """Roll the network over a sequence; yields (check index, forward) per check."""
        cfg, dev = self.ctx.cfg, self.ctx.device
        state = None if want_loss else NodeState(net, self.gt)
        instr = torch.as_tensor(seq.instr, device=dev)
        has_instr = torch.as_tensor(seq.has_instr.astype(np.float32), device=dev)
        instr_p = None if seq.instr_p is None else torch.as_tensor(seq.instr_p, device=dev)
        prev = None
        for k, check in enumerate(seq.checks):
            if want_loss:
                state = NodeState(net, self.gt)
            ck = to_device(check, dev)
            memory, has_memory = memory_of(k, prev)
            out = forward_check(net, cfg, self.gt, state, ck["qh"], ck["vh"], ck["gh"],
                                instr=instr, has_instr=has_instr, policy=ck["policy"],
                                has_policy=ck["has_policy"], memory=memory, has_memory=has_memory,
                                mask=mask, instr_p=instr_p)
            prev = out.belief()
            yield k, out

    def loss(self, unit: TrainUnit, seq: Sequence) -> tuple[torch.Tensor, dict[str, float]]:
        net = unit.trained["locator"]
        dev = self.ctx.device
        batch = len(seq)

        def memory_of(k, prev):
            if self.single_check or self.no_memory:
                return None, None
            return self._memory(self._memory_modes(batch, k == 0), seq, k, prev, dev)

        # Each check's graph is freed by its own backward; the memory between checks is detached.
        total = torch.zeros((), device=dev)
        for k, out in self._run(net, seq, memory_of, True):
            part = batch_loss(out, self.gt, seq.checks[k].tgt, seq.has_instr) / seq.n_checks
            part.backward()
            total = total + part.detach()
        return total + 0.0 * next(net.parameters()).sum(), {}

    # ---- evaluation ----
    def _hit(self, top: np.ndarray, tgt: np.ndarray, has_instr: np.ndarray,
             hidden: np.ndarray | None) -> np.ndarray:
        out = np.zeros(top.shape[0], dtype=bool)
        for i in range(top.shape[0]):
            nodes = self.gt.targets(int(tgt[i]), bool(has_instr[i])).nodes
            if hidden is not None:
                nodes = nodes[~hidden[nodes]]
            out[i] = bool(np.any(nodes == int(top[i])))
        return out

    def _rollout(self, net: Any, mask: torch.Tensor | None, suffix: str) -> dict[str, float]:
        """Self-memory rollout metrics over the fixed held-out sequences, optionally masked."""
        dev = self.ctx.device
        phase = self.gt.phase_t
        phase_np = phase.cpu().numpy()
        hidden = None if mask is None else mask.cpu().numpy()
        acc: dict[str, list[float]] = {m: [] for m in INSTR_MODES}
        hits, maes, recovery = [], [], []
        for seq, wrong in self._eval:
            batch = len(seq)

            def self_memory(k, prev):
                if self.no_memory or k == 0 or prev is None:
                    return None, None
                return truncate(prev), torch.ones(batch, device=dev)

            for k, out in self._run(net, seq, self_memory, False, mask):
                b = out.belief()
                top = b.argmax(dim=-1).cpu().numpy()
                tgt = seq.checks[k].tgt
                hits.extend(self._hit(top, tgt, seq.has_instr, hidden).tolist())
                maes.extend(np.abs((b * phase).sum(-1).cpu().numpy() - phase_np[tgt]).tolist())
                acc[INSTR_MODES[int(seq.instr_mode[0])]].append(
                    task_accuracy(b, self.gt.task_id_t, seq.task))
            if not self.single_check and not self.no_memory:
                wrong_mem = torch.as_tensor(
                    self.sampler.dense_target(wrong, seq.has_instr, len(self.nodes)), device=dev)

                def wrong_memory(k, prev):
                    if k == 0:
                        return wrong_mem, torch.ones(batch, device=dev)
                    return truncate(prev), torch.ones(batch, device=dev)

                first = np.full(batch, EVAL_CHECKS, dtype=np.int64)
                for k, out in self._run(net, seq, wrong_memory, False, mask):
                    if k == 0:
                        continue
                    hit = self._hit(out.belief().argmax(dim=-1).cpu().numpy(), seq.checks[k].tgt,
                                    seq.has_instr, hidden)
                    first = np.where((first == EVAL_CHECKS) & hit, k, first)
                recovery.extend(first.tolist())
        metrics = {f"top1_target{suffix}": float(np.mean(hits)), f"phase_mae{suffix}": float(np.mean(maes))}
        for m in INSTR_MODES:
            metrics[f"task_acc_{m}{suffix}"] = float(np.mean(acc[m])) if acc[m] else float("nan")
        metrics[f"task_acc{suffix}"] = float(np.mean([metrics[f"task_acc_{m}{suffix}"] for m in INSTR_MODES]))
        metrics[f"recovery{suffix}"] = float(np.mean(recovery)) if recovery else float("nan")
        return metrics

    def evaluate(self, unit: TrainUnit, evalset: Any) -> dict[str, float]:
        """Held-out self-memory rollouts, with the held-out demos' nodes present and hidden.

        Returns:
            top1_target, phase_mae, task_acc per regime and recovery, each also with the
            _masked suffix where the held-out demos' own nodes are removed from tier 1, the seed,
            reachability and the target set.
        """
        net = unit.trained["locator"]
        mask = (self.gt.owner_t % HELDOUT_STRIDE) == 0
        return {**self._rollout(net, None, ""), **self._rollout(net, mask, "_masked")}

    def select(self, metrics: dict[str, float]) -> float | None:
        phase, hit, acc = (metrics.get("phase_mae_masked"), metrics.get("top1_target_masked"),
                           metrics.get("task_acc_masked"))
        if phase is None or hit is None or acc is None:
            return None
        rec = metrics.get("recovery_masked")
        rec = 0.0 if rec is None or not np.isfinite(rec) else rec
        return phase + 0.1 * (1.0 - hit) + 0.1 * (1.0 - acc) + 0.01 * rec

    def save(self, unit: TrainUnit, store: Any) -> None:
        save_checkpoint(unit.trained["locator"], self.nodes, self.edges, store.path(schema.HEAD_NPZ))


def truncate(belief: torch.Tensor, k: int = MEMORY_TOPK) -> torch.Tensor:
    """[B, V] the belief kept on its top-k nodes, renormalised, zero elsewhere."""
    k = min(k, belief.shape[-1])
    top, idx = belief.topk(k, dim=-1)
    out = torch.zeros_like(belief).scatter_(-1, idx, top)
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
