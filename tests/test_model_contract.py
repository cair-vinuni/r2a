"""Every trained head writes the same .npz it always wrote, and reads back exactly what it wrote.

The refactor that moves these heads onto robot_ood.models.base.CheckpointedModule must not change one
byte on disk: a checkpoint written before it has to load after it, and vice versa. Two properties
hold that shut.

  1. **Pinned bytes.** A seeded head saved to a fresh path hashes to the value recorded here. Any
     change to the key order, the dtypes, or which shape constants get stamped moves the hash.
  2. **Round trip.** load(save(m)) gives back the same tensors, and saving the loaded module
     reproduces the same bytes -- so a head that reads its own file correctly but rewrites it
     differently is caught too.

A hash here changes ONLY together with a deliberate, reviewed artifact-format change, and every
artifact on disk has to be regenerated in the same commit.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch


def _score():
    from robot_ood.models.entry.score import ScoreNet
    return ScoreNet(hidden=32)


def _invdyn():
    from robot_ood.models.entry.invdyn import InverseDynamicsNet
    return InverseDynamicsNet(hidden=32)


# head -> (factory, sha256 of its checkpoint under torch.manual_seed(0)). The keys each one stamps
# are asserted through the hash, not listed twice.
HEADS = {
    "ScoreNet": (_score, "6b69148b71a5a1286880f176d3bace1ad78b2b97095525cd2e1deea79b2c2ee8"),
    "InverseDynamicsNet": (_invdyn, "c57d6b62f5de631dc5cc88354f84259785969b8f1de1cb1ad5dcf5cd32ef9b45"),
}


def _seeded(factory):
    torch.manual_seed(0)
    return factory()


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", sorted(HEADS))
def test_checkpoint_bytes_are_pinned(name, tmp_path):
    factory, want = HEADS[name]
    written = _seeded(factory).save(tmp_path / f"{name}.npz")
    assert _sha256(written) == want, (
        f"{name}'s checkpoint layout changed. Every {written.name} on disk was written by the old "
        "layout; regenerate them in this commit, or put the layout back."
    )


@pytest.mark.parametrize("name", sorted(HEADS))
def test_load_returns_what_save_wrote(name, tmp_path):
    factory, _ = HEADS[name]
    written = _seeded(factory).save(tmp_path / f"{name}.npz")
    reloaded = type(_seeded(factory)).load(written)
    before = _seeded(factory).state_dict()
    after = reloaded.state_dict()
    assert sorted(before) == sorted(after), f"{name}: load rebuilt a different set of parameters"
    for key in before:
        np.testing.assert_array_equal(
            before[key].detach().cpu().numpy(), after[key].detach().cpu().numpy(),
            err_msg=f"{name}.{key} did not survive the round trip",
        )
    assert _sha256(reloaded.save(tmp_path / f"{name}.again.npz")) == _sha256(written), (
        f"{name} reads its own checkpoint but rewrites it differently"
    )
