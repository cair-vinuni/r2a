"""The training-query set: one device-resident draw of the make_queries mapping.

Shared by the locator and FCSR objectives; every corruption, instruction and hard-negative draw
happens at construction, so a row is a fixed example, not a per-epoch noise process.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch

from robot_ood.models.fusion.target import ChunkTargetBuilder, ChunkTargets
from robot_ood.train.augment import (
    CopyPlan, Corruption, PolicyCorruption, PolicyCorruptor, RetrievalCorruptor,
)
from robot_ood.train.data import tile_queries
from robot_ood.train.types import QueryBatch

__all__ = ["QuerySet"]


@dataclass(frozen=True)
class QuerySet:
    """A stage's training queries on device, plus the class-balance weights derived from them.

    Attributes:
        qh: Joint positions [n, H, D].
        vh: Joint velocities [n, H, D].
        gh: Gripper states [n, H].
        tgt: Target node ids [n].
        targets: The chunk-blend inputs and target, or None when they were not built.
        corrupt: Which rows retrieval is deliberately wrong on, or None.
        policy_corrupt: Which rows the policy's chunk is deliberately wrong on, or None.
        e_lang: [n, lang_dim] per-row instruction embedding on device, or None.
        seg_id: [n] i64 the row's own C1 segment id (SEGMENT rows) or -1, or None.
        neg_segments: [n, n_seg_neg] i64 mined hard negatives, -1 padded, or None.
        a_hist: [n, H, ACTION_DIM] the demo's recorded actions ending at each row's window, or None.
        phase_hat: [n] f32 the phase each row's QueryContext carries, or None.
    """

    qh: torch.Tensor
    vh: torch.Tensor
    gh: torch.Tensor
    tgt: np.ndarray
    targets: ChunkTargets | None = None
    corrupt: Corruption | None = None
    policy_corrupt: Any = None
    e_lang: torch.Tensor | None = None
    seg_id: np.ndarray | None = None
    neg_segments: np.ndarray | None = None
    pos_segments: np.ndarray | None = None
    a_hist: torch.Tensor | None = None
    phase_hat: torch.Tensor | None = None

    @classmethod
    def from_queries(
        cls, queries: Any, dev: Any, builder: ChunkTargetBuilder | None = None,
        corruptor: RetrievalCorruptor | None = None, rng: Any = None,
        language: Any = None, task_id: np.ndarray | None = None,
        policy_corruptor: PolicyCorruptor | None = None, copies: CopyPlan | None = None,
        seg_miner: Any = None, n_seg_neg: int = 0,
    ) -> "QuerySet":
        """Upload one make_queries result and build its chunk targets.

        Args:
            queries: A query mapping from make_queries.
            dev: Target device.
            builder: Chunk-target builder, or None to leave targets unbuilt.
            corruptor: Deliberately-wrong-retrieval sampler, or None to leave every row clean.
            rng: Random state the corruption is drawn from; required with corruptor.
            language: robot_ood.models.language.encoder.TaskEmbeddings, or None to build no
                instruction embeddings.
            task_id: [V] per-node task id on the host, used to look up each target's task.
            policy_corruptor: Deliberately-wrong-policy sampler, drawn independently of corruptor
                so the head sees all four right/wrong quadrants, or None.
            copies: K fixed corruption copies per source row, or None for one.
            seg_miner: robot_ood.train.negatives.HardNegativeMiner for this run's segment table,
                or None to mine no negatives.
            n_seg_neg: Hard negatives mined per SEGMENT row; unread when seg_miner is None.

        Returns:
            The device-resident query set.

        Raises:
            ValueError: A corruptor was given without an rng to draw it from, or a language table
                without the rng and task ids it needs.
        """
        if (corruptor is not None or policy_corruptor is not None) and rng is None:
            raise ValueError("QuerySet.from_queries: a corruptor needs an rng to draw from")
        if language is not None and (rng is None or task_id is None):
            raise ValueError("QuerySet.from_queries: a language table needs an rng and task_id")
        if copies is not None and copies.copies > 1:
            queries = tile_queries(queries, copies.copies)
        tgt = queries["tgt_node"]
        n = len(tgt)
        # One stream when there is one copy -- and it is the caller's own, so a K=1 run draws the
        # sequence it always drew. Copy k of a K>1 plan draws from a stream seeded by (seed, k).
        streams = [rng] if copies is None or copies.copies == 1 else [
            copies.rng(k) for k in range(copies.copies)
        ]
        src = n // len(streams)
        t_now = np.asarray(queries["t_raw_now"], dtype=np.int64)[:src]
        # Drawn at construction, then only sliced -- redrawing per epoch would make a corrupted row a
        # noise process; alpha's only consistent fit across changing pairings is the average.
        policy_corrupt = (
            None if policy_corruptor is None else PolicyCorruption.concat(
                [policy_corruptor.draw(tgt[:src], t_now, stream) for stream in streams]
            )
        )
        corrupt = (
            None if corruptor is None else Corruption.concat(
                [corruptor.draw(tgt[:src], stream) for stream in streams]
            )
        )
        e_lang = cls._draw_instructions(language, tgt[:src], task_id, streams, dev)
        seg_id = np.asarray(queries.get("seg_id", np.full(n, -1)), dtype=np.int64)
        neg_segments = None if seg_miner is None else seg_miner.mine_batch(seg_id, n_seg_neg)
        from robot_ood.train.negatives import POS_MODE
        pos_segments = seg_miner.positive_batch(seg_id) if seg_miner is not None and POS_MODE == "xdemo" else None
        return cls(
            qh=torch.as_tensor(queries["q_hist"], device=dev),
            vh=torch.as_tensor(queries["qdot_hist"], device=dev),
            gh=torch.as_tensor(queries["grip_hist"], device=dev),
            tgt=tgt,
            targets=None if builder is None else builder.build(queries, policy_corrupt),
            corrupt=corrupt,
            policy_corrupt=policy_corrupt,
            e_lang=e_lang,
            seg_id=seg_id,
            neg_segments=neg_segments,
            pos_segments=pos_segments,
            a_hist=None if "a_hist" not in queries else torch.as_tensor(queries["a_hist"], device=dev),
            phase_hat=(
                None if "phase_hat" not in queries
                else torch.as_tensor(queries["phase_hat"], device=dev)
            ),
        )

    @staticmethod
    def _draw_instructions(
        language: Any, tgt: np.ndarray, task_id: np.ndarray | None, streams: "list[Any]", dev: Any,
    ) -> torch.Tensor | None:
        """One paraphrase surrogate of each row's own task, as an embedding, per copy.

        Args:
            language: The TaskEmbeddings table, or None.
            tgt: [n] target node ids of ONE copy.
            task_id: [V] per-node task id on the host.
            streams: One random state per copy; a single-copy set passes the caller's own.
            dev: Target device.

        Returns:
            [n * len(streams), lang_dim] float32 on device, or None when no table was given.
        """
        if language is None:
            return None
        # Drawn from the target's OWN task only -- wrong-task rows already come from the corruption
        # sampler swapping what retrieval returns, not from a mismatched instruction.
        rows = np.asarray(task_id, dtype=np.int64)[np.asarray(tgt, dtype=np.int64)]
        # rng.random, not rng.integers: the one uniform draw both RandomState and Generator expose.
        drawn = np.concatenate([
            language.e_aug[language.draw_aug(rows, np.asarray(stream.random(rows.shape[0])))]
            for stream in streams
        ], axis=0)
        return torch.as_tensor(drawn, dtype=torch.float32, device=dev)

    def __len__(self) -> int:
        return len(self.tgt)

    @property
    def device(self) -> Any:
        """The device every tensor here lives on."""
        return self.qh.device

    def summary(self) -> str:
        """One-line report: the row count and the corruption branch counts."""
        corrupt = "" if self.corrupt is None else " | corruption: " + " ".join(
            f"{name}={count}" for name, count in self.corrupt.counts().items()
        )
        corrupt += "" if self.policy_corrupt is None else " | policy: " + " ".join(
            f"{name}={count}" for name, count in self.policy_corrupt.counts().items()
        )
        return f"{len(self)} rows" + corrupt

    def shuffled_batches(self, qbatch: int) -> "Iterator[QueryBatch]":
        """Yield contiguous chunks of one fresh torch.randperm permutation.

        Args:
            qbatch: Maximum queries per optimiser step. The last chunk of an epoch may be shorter;
                the loss takes its MEAN over however many rows it is given, so it stays weighted right.

        Yields:
            One robot_ood.train.types.QueryBatch per optimiser step.
        """
        n_q = len(self)
        # torch.randperm specifically, not RandomSampler/BatchSampler: it consumes the process-global
        # torch RNG manual_seed pins, so switching samplers draws a different permutation from the same seed.
        perm = torch.randperm(n_q).numpy()
        for start in range(0, n_q, qbatch):
            yield self.batch(perm[start : start + qbatch])

    def ordered_batches(self, qbatch: int) -> "Iterator[QueryBatch]":
        """Yield the set in row order, drawing no randomness. Used by the held-out evaluation.

        Args:
            qbatch: Maximum queries per batch.

        Yields:
            One robot_ood.train.types.QueryBatch per chunk, covering every row, none twice.
        """
        n_q = len(self)
        for start in range(0, n_q, qbatch):
            yield self.batch(np.arange(start, min(start + qbatch, n_q)))

    def batch(self, idx: np.ndarray) -> QueryBatch:
        """Slice the set down to the given rows.

        Args:
            idx: [b] row indices.

        Returns:
            The device-resident batch.
        """
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        chunk, corrupt = self.targets, self.corrupt
        pol = self.policy_corrupt
        return QueryBatch(
            qh=self.qh[idx_t], vh=self.vh[idx_t], gh=self.gh[idx_t],
            tgt=self.tgt[idx],
            a_policy=None if chunk is None else torch.as_tensor(chunk.a_policy[idx], device=self.device),
            a_corr=None if chunk is None else torch.as_tensor(chunk.a_corr[idx], device=self.device),
            ee_now=None if chunk is None else torch.as_tensor(chunk.ee_now[idx], device=self.device),
            ee_ref=None if chunk is None else torch.as_tensor(chunk.ee_ref[idx], device=self.device),
            corrupt_node=None if corrupt is None else torch.as_tensor(corrupt.node[idx], device=self.device),
            corrupt_offset=(
                None if corrupt is None else torch.as_tensor(corrupt.offset[idx], device=self.device)
            ),
            # int8 stays on the host: the loss uploads it per batch (the strata reweight and the two
            # cause heads' labels, robot_ood.train.loss), and the alpha diagnostics group by it.
            corrupt_kind=None if corrupt is None else corrupt.kind[idx],
            policy_kind=None if pol is None else pol.kind[idx],
            e_lang=None if self.e_lang is None else self.e_lang[idx_t],
            seg_id=None if self.seg_id is None else torch.as_tensor(
                self.seg_id[idx], device=self.device
            ),
            neg_segments=None if self.neg_segments is None else torch.as_tensor(
                self.neg_segments[idx], device=self.device
            ),
            pos_segments=None if self.pos_segments is None else torch.as_tensor(
                self.pos_segments[idx], device=self.device
            ),
            a_hist=None if self.a_hist is None else self.a_hist[idx_t],
            phase_hat=None if self.phase_hat is None else self.phase_hat[idx_t],
        )


