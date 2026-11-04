"""FcsrNet holds the four trained pieces and writes the archive they deploy from."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from robot_ood.core import npz
from robot_ood.core.schemas import ACTION_EMBED_DIM, SEGMENT_LEN
from robot_ood.data import schema
from robot_ood.models.fcsr_net import FcsrNet, SegEncoders

SHIPPED = Path("outputs/long_frozen/artifacts") / schema.FCSR_NPZ


def _net(*, hidden: int = 8, k: int = 3, chunk: int = 4,
         future_temp: float | None = None) -> FcsrNet:
    from robot_ood.models.fusion.corrector import Corrector
    from robot_ood.models.retrieval.encoders import QueryEncoder, SegmentKeyEncoder
    from robot_ood.models.retrieval.scorer import ActionEncoder, SegmentScorer

    seg = SegEncoders(
        QueryEncoder(dim=7, hist=3, hidden=hidden, lang_dim=0),
        SegmentKeyEncoder(hidden),
        ActionEncoder(SEGMENT_LEN, out_dim=ACTION_EMBED_DIM),
    )
    net = FcsrNet(seg, SegmentScorer(hidden=hidden), Corrector(chunk, k=k, hidden=hidden))
    net.future_temp = future_temp
    return net


def test_the_state_dict_keys_are_the_three_prefixes():
    keys = set(_net().state_dict())
    assert {k.split(".")[0] for k in keys} == {"seg", "scorer", "corrector"}
    assert {k.split(".")[1] for k in keys if k.startswith("seg.")} == {"query", "key", "action"}


def test_a_round_trip_through_the_archive_returns_the_same_tensors(tmp_path):
    net = _net(future_temp=2.5)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    path = npz.write_arrays(tmp_path, schema.FCSR_NPZ, net.arrays(graph_hash="h"))

    back = FcsrNet.from_arrays(dict(np.load(path)), path)

    assert back.future_temp == pytest.approx(2.5)
    before, after = net.state_dict(), back.state_dict()
    assert set(before) == set(after)
    for k in before:
        assert torch.equal(before[k], after[k]), k


def test_the_meta_half_names_every_width_a_reader_needs(tmp_path):
    meta = _net(hidden=16, k=5, future_temp=3.0).checkpoint_meta()
    assert meta["seg.hidden"] == 16 and meta["scorer.hidden"] == 16
    assert meta["corrector.k"] == 5 and meta["seg.future_temp"] == pytest.approx(3.0)
    assert meta["seg.action_chunk_len"] == SEGMENT_LEN


def test_an_archive_missing_a_stamp_names_the_stage_that_writes_it(tmp_path):
    arrays = _net().arrays(graph_hash="h")
    del arrays["seg.hidden"]
    with pytest.raises(ValueError, match="train fcsr"):
        FcsrNet.from_arrays(arrays, tmp_path / schema.FCSR_NPZ)


@pytest.mark.skipif(not SHIPPED.is_file(), reason="no built graph beside this checkout")
def test_reading_and_rewriting_the_shipped_archive_reproduces_it_byte_for_byte(tmp_path):
    """The archive on disk is what every eval loaded. Reading it into FcsrNet and writing it back
    must land on the same bytes, or the weights that deploy are not the weights that were measured.
    """
    archive = dict(np.load(SHIPPED))
    net = FcsrNet.from_arrays(archive, SHIPPED)
    rewritten = net.save(tmp_path, graph_hash=str(archive["graph_hash"]))

    assert zipfile.ZipFile(rewritten).namelist() == zipfile.ZipFile(SHIPPED).namelist()
    assert hashlib.sha256(Path(rewritten).read_bytes()).hexdigest() == \
        hashlib.sha256(SHIPPED.read_bytes()).hexdigest()
