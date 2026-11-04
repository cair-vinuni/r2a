"""The FCSR candidate realisation the joint stage trains on.

The realize_column0 tests cover the train/deploy candidate-set skew -- the corrector must see the shortlist
SegmentRetriever.retrieve_batch actually builds, which can miss the true continuation, not
SegmentLossComputer's mined pool, which never can.
"""

from __future__ import annotations

import numpy as np
import torch

from robot_ood.core.schemas import ACTION_DIM, SEGMENT_LEN, SegmentTable
from robot_ood.train.augment import CLEAN, OFFSET, TARGET, WRONG_TASK
from robot_ood.train.fcsr import realize_column0
from robot_ood.train.negatives import segment_tensors
from robot_ood.train.types import QueryBatch


# ======================================================================================================
# realize_column0 -- the fix for the measured gap: OFFSET was labelled but never applied to what
# the corrector was handed, so the offset branch could not have differed from the clean one.
# ======================================================================================================
def _seg_t(s: int = 6, node_to_segment: np.ndarray | None = None):
    rng = np.random.RandomState(0)
    table = SegmentTable(
        start_node=np.arange(s, dtype=np.int32), end_node=np.arange(s, dtype=np.int32),
        owner=np.zeros(s, dtype=np.int32), task_id=np.zeros(s, dtype=np.int16),
        phase_start=np.linspace(0, 1, s, dtype=np.float32),
        phase_end=np.linspace(0, 1, s, dtype=np.float32),
        action_chunk=rng.randn(s, SEGMENT_LEN, ACTION_DIM).astype(np.float32),
        ee_start=rng.randn(s, 6).astype(np.float32), ee_end=rng.randn(s, 6).astype(np.float32),node_to_segment=np.arange(s, dtype=np.int32) if node_to_segment is None else node_to_segment,
    )
    return segment_tensors(table, "cpu")


def _ref_for(seg_t):
    """A reference reader: each candidate's ee_start repeated over SEGMENT_LEN rows, plus a row ramp."""
    def read(idx):
        base = seg_t.ee_start[idx][:, :, None, :].expand(-1, -1, SEGMENT_LEN, -1)
        return base + 0.01 * torch.arange(SEGMENT_LEN, dtype=base.dtype)[None, None, :, None]
    return read


def _qb(b: int, corrupt_kind=None, corrupt_node=None, corrupt_offset=None) -> QueryBatch:
    return QueryBatch(
        qh=torch.zeros(b, 1, 1), vh=torch.zeros(b, 1, 1), gh=torch.zeros(b, 1),
        tgt=np.arange(b),
        corrupt_kind=corrupt_kind, corrupt_node=corrupt_node, corrupt_offset=corrupt_offset,
    )


def test_realize_column0_applies_offset_to_geometry_not_the_action_chunk():
    """The measurement this whole change exists to fix: an OFFSET row's corrector inputs must differ
    from its clean twin's."""
    seg_t = _seg_t(s=6)
    cand_idx = torch.tensor([[0, 1], [2, 3]])
    kind = np.array([CLEAN, OFFSET], dtype=np.int8)
    node = torch.tensor([-1, -1])
    offset = torch.tensor([[0.0] * 6, [0.1, -0.2, 0.05, 0.0, 0.3, -0.1]])
    qb = _qb(2, corrupt_kind=kind, corrupt_node=node, corrupt_offset=offset)
    ref_for = _ref_for(seg_t)

    idx, ee_start, ee_end, actions, ee_ref, realized = realize_column0(seg_t, qb, cand_idx, ref_for)

    torch.testing.assert_close(idx, cand_idx)                      # OFFSET never swaps the segment
    torch.testing.assert_close(actions, seg_t.action_chunk[cand_idx])   # action chunk untouched
    clean_start, clean_end, clean_ref = seg_t.ee_start[cand_idx], seg_t.ee_end[cand_idx], ref_for(cand_idx)
    torch.testing.assert_close(ee_start[0], clean_start[0])        # clean row: bit-identical
    torch.testing.assert_close(ee_ref[0], clean_ref[0])
    torch.testing.assert_close(ee_start[1, 1], clean_start[1, 1])  # offset row, OTHER columns: untouched
    torch.testing.assert_close(ee_ref[1, 1], clean_ref[1, 1])
    # the whole segment moves: start, end and every reference row by the same displacement
    torch.testing.assert_close(ee_start[1, 0], clean_start[1, 0] + offset[1])
    torch.testing.assert_close(ee_end[1, 0], clean_end[1, 0] + offset[1])
    torch.testing.assert_close(ee_ref[1, 0], clean_ref[1, 0] + offset[1])
    assert realized.tolist() == kind.tolist()                      # OFFSET always realises


def test_realize_column0_target_moves_every_column_after_its_start():
    """A TARGET row is the scene that moved: every candidate keeps its start, and every later
    reference row and the end carry the same displacement, so the demos still agree with each
    other."""
    seg_t = _seg_t(s=6)
    cand_idx = torch.tensor([[0, 1]])
    kind = np.array([TARGET], dtype=np.int8)
    offset = torch.tensor([[0.1, -0.2, 0.05, 0.0, 0.3, -0.1]])
    qb = _qb(1, corrupt_kind=kind, corrupt_node=torch.tensor([-1]), corrupt_offset=offset)
    ref_for = _ref_for(seg_t)

    idx, ee_start, ee_end, actions, ee_ref, realized = realize_column0(seg_t, qb, cand_idx, ref_for)

    clean_start, clean_end, clean_ref = seg_t.ee_start[cand_idx], seg_t.ee_end[cand_idx], ref_for(cand_idx)
    torch.testing.assert_close(ee_start, clean_start)                       # starts untouched
    torch.testing.assert_close(ee_end, clean_end + offset[0])               # every end displaced
    torch.testing.assert_close(ee_ref[:, :, 0], clean_ref[:, :, 0])         # first rows untouched
    torch.testing.assert_close(ee_ref[:, :, 1:], clean_ref[:, :, 1:] + offset[0])   # every later row, every column
    assert realized.tolist() == [TARGET]


def test_realize_column0_swaps_the_segment_for_a_realisable_wrong_task_node():
    seg_t = _seg_t(s=6)   # node_to_segment is the identity here, so node 4 -> segment 4
    cand_idx = torch.tensor([[0, 1]])
    kind = np.array([WRONG_TASK], dtype=np.int8)
    node = torch.tensor([4])
    qb = _qb(1, corrupt_kind=kind, corrupt_node=node, corrupt_offset=torch.zeros(1, 6))
    ref_for = _ref_for(seg_t)

    idx, ee_start, ee_end, actions, ee_ref, realized = realize_column0(seg_t, qb, cand_idx, ref_for)

    assert idx[0, 0].item() == 4 and idx[0, 1].item() == 1         # only column 0 swaps
    torch.testing.assert_close(actions[0, 0], seg_t.action_chunk[4])
    torch.testing.assert_close(ee_start[0, 0], seg_t.ee_start[4])
    torch.testing.assert_close(ee_ref, ref_for(idx))                # the reference follows the swap
    assert realized.tolist() == [WRONG_TASK]


def test_realize_column0_downgrades_an_unrealisable_swap_to_clean():
    """A node with no segment of its own (node_to_segment == -1) cannot be realised -- the row must
    report CLEAN, not WRONG_TASK with clean-identical inputs, which is exactly the mislabelling bug
    this function exists to prevent."""
    node_to_segment = np.arange(6, dtype=np.int32)
    node_to_segment[4] = -1                                        # node 4 starts no segment
    seg_t = _seg_t(s=6, node_to_segment=node_to_segment)
    cand_idx = torch.tensor([[0, 1]])
    kind = np.array([WRONG_TASK], dtype=np.int8)
    node = torch.tensor([4])
    qb = _qb(1, corrupt_kind=kind, corrupt_node=node, corrupt_offset=torch.zeros(1, 6))

    idx, ee_start, ee_end, actions, ee_ref, realized = realize_column0(seg_t, qb, cand_idx, _ref_for(seg_t))

    torch.testing.assert_close(idx, cand_idx)                       # swap could not happen
    torch.testing.assert_close(actions, seg_t.action_chunk[cand_idx])
    torch.testing.assert_close(ee_start, seg_t.ee_start[cand_idx])
    assert realized.tolist() == [CLEAN]                             # downgraded, not mislabelled


def test_realize_column0_is_a_no_op_when_the_batch_carries_no_labels():
    seg_t = _seg_t(s=6)
    cand_idx = torch.tensor([[0, 1], [2, 3]])
    qb = _qb(2)   # corrupt_kind/corrupt_node/corrupt_offset all default to None

    idx, ee_start, ee_end, actions, ee_ref, realized = realize_column0(seg_t, qb, cand_idx, _ref_for(seg_t))

    torch.testing.assert_close(idx, cand_idx)
    torch.testing.assert_close(actions, seg_t.action_chunk[cand_idx])
    torch.testing.assert_close(ee_start, seg_t.ee_start[cand_idx])
    assert realized is None


def test_the_stage_saves_through_the_nets_own_writer():
    """The stage must write through FcsrNet.save, which stamps every width a reader needs.

    A trainer that hand-rolls an npz drops those stamps. That fails as a KeyError on a stamp key at
    episode 1 of a 1573-episode eval, an hour of GPU time after the mistake, and in no unit test.

    Asserted on the source because the failure is a MISSING call, and a behavioural test would have
    to reproduce a full fusion fit to see it.
    """
    import inspect

    from robot_ood.train.tasks import fcsr

    src = inspect.getsource(fcsr)
    assert "net.save(graph_dir, graph_hash=" in src, "the stage must save through FcsrNet"
    assert "np.savez(" not in src, "a checkpoint goes through FcsrNet.save, not np.savez"
