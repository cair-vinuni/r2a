"""The policy-action cache client: demo shards, batched queries, per-demo resume."""

from __future__ import annotations

import numpy as np
import pytest

from robot_ood.build.policy_cache import (ActionCacheBuilder, ChunkReplayer, DemoActions,
                                          DemoFrames, DemoSource, ShardStore, _shard)
from robot_ood.core.protocol import get_protocol


def test_a_shard_spec_names_an_index_within_its_count():
    assert _shard(None) is None
    assert _shard("0/4") == (0, 4)
    assert _shard("3/4") == (3, 4)


@pytest.mark.parametrize("spec", ["4/4", "-1/4", "1/0", "1", "a/b", ""])
def test_a_shard_spec_outside_its_count_is_refused(spec):
    with pytest.raises(ValueError):
        _shard(spec)


def _selector(shard):
    builder = ActionCacheBuilder.__new__(ActionCacheBuilder)
    builder._shard = shard
    return builder._is_mine


def test_the_shards_of_a_run_partition_its_demos_exactly_once():
    demos = range(500)
    for count in (2, 3, 24):
        claimed = [sum(_selector((i, count))(d) for i in range(count)) for d in demos]
        assert set(claimed) == {1}
        sizes = [sum(_selector((i, count))(d) for d in demos) for i in range(count)]
        assert max(sizes) - min(sizes) <= 1


def test_no_selector_claims_every_demo():
    assert all(_selector(None)(d) for d in range(500))


def _source(demo_id, n_frames, task_id=0):
    return DemoSource(demo_id=demo_id, hdf5_path=None, group_key=f"demo_{demo_id}", task_id=task_id,
                      task_name="task", instruction="put the bowl on the plate", n_frames=n_frames)


def _frames(n, views=2, state_dim=8, ee=(0.0,) * 6):
    return DemoFrames(views=tuple(np.zeros((n, 128, 128, 3), np.uint8) for _ in range(views)),
                      state=np.zeros((n, state_dim), np.float64),
                      ee=np.tile(np.asarray(ee, np.float64), (n, 1)), joints=np.zeros((n, 7), np.float32))


@pytest.mark.parametrize("protocol_id, n_frames, batch", [
    ("nvidia", 53, 16), ("openpi", 53, 4), ("openvla_oft", 8, 1), ("nvidia", 1, 16)])
def test_every_frame_takes_its_row_of_the_query_that_covers_it(protocol_id, n_frames, batch):
    protocol = get_protocol(protocol_id)
    replayer = ChunkReplayer(protocol, host="", port=0, batch=batch, stub=True)
    actions = replayer.replay(_source(0, n_frames), _frames(n_frames))
    assert len(actions) == n_frames
    assert np.array_equal(actions.commanded[:, 0], np.arange(n_frames, dtype=np.float32))
    assert replayer.chunk_size == protocol.exec_horizon


def test_the_gripper_follows_the_protocol():
    frames = _frames(4)
    binarized = ChunkReplayer(get_protocol("nvidia"), host="", port=0, stub=True).replay(_source(0, 4), frames)
    native = ChunkReplayer(get_protocol("openpi"), host="", port=0, stub=True).replay(_source(0, 4), frames)
    assert set(np.unique(binarized.commanded[:, 6])) <= {-1.0, 1.0}
    assert np.array_equal(native.commanded[:, 6], native.grip)
    assert np.array_equal(binarized.grip, native.grip)


def test_requests_carry_at_most_batch_queries():
    class Counting:
        def __init__(self):
            self.sizes = []

        def get_server_metadata(self):
            return {}

        def infer(self, payload):
            self.sizes.append(len(payload["episode_step"]))
            steps = payload["episode_step"]
            return {"data": {"actions": np.zeros((len(steps), 16, 7), np.float32)}}

    replayer = ChunkReplayer(get_protocol("nvidia"), host="", port=0, batch=4, stub=True)
    replayer._client = Counting()
    replayer.replay(_source(0, 80), _frames(80))     # 10 queries at horizon 8
    assert replayer._client.sizes == [4, 4, 2]


def test_a_shard_store_keeps_one_file_per_demo_and_rejects_a_changed_length(tmp_path):
    store = ShardStore(tmp_path / "shards")
    source = _source(3, 5)
    assert store.load(source) is None
    store.save(source, DemoActions(commanded=np.ones((5, 7), np.float32), grip=np.zeros(5, np.float32)))
    loaded = store.load(source)
    assert loaded is not None and len(loaded) == 5
    assert store.load(_source(3, 6)) is None
    assert store.path(source).parent.name == "000_task"


def test_a_bimanual_joint_chunk_is_one_lanes_steps_from_the_demo_joints():
    """Absolute joint targets of the named arm become steps from frame t's joints; the gripper reads as a flag."""
    protocol = get_protocol("pi05_robotwin")
    with pytest.raises(ValueError):
        ChunkReplayer(protocol, host="", port=0, stub=True)

    class Targets:
        def get_server_metadata(self):
            return {}

        def infer(self, payload):
            k, chunk = len(payload["episode_step"]), np.zeros((50, 14), np.float32)
            chunk[:, 0] = np.arange(1, 51) * 0.01          # left joint 0 walks 0.01 rad a step
            chunk[:, 6] = 0.9                               # left gripper open
            chunk[:, 7] = 5.0                               # right joint 0 parked at 5
            chunk[:, 13] = 0.1                              # right gripper closed
            assert len(payload["batch_images"][0]) == 3 and payload["state"][0].shape == (1, 14)
            return {"data": {"actions": np.stack([chunk] * k)}}

    frames = _frames(75, views=3, state_dim=14)
    left = ChunkReplayer(protocol, host="", port=0, stub=True, arm="left")
    left._client = Targets()
    actions = left.replay(_source(0, 75), frames)
    assert np.allclose(actions.commanded[:50, 0], 0.01) and np.allclose(actions.commanded[:, 1:6], 0)
    assert np.all(actions.commanded[:, 6] == -1.0) and np.allclose(actions.grip, 0.9)
    right = ChunkReplayer(protocol, host="", port=0, stub=True, arm="right")
    right._client = Targets()
    actions = right.replay(_source(0, 75), frames)
    assert np.allclose(actions.commanded[0, 0], 5.0) and np.allclose(actions.commanded[1:50, 0], 0.0)
    assert np.all(actions.commanded[:, 6] == 1.0)
