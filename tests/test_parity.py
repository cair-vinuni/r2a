"""The golden parity net -- proves the LIVE inference path (robot_ood.runtime.locator.GraphLocator)
produces BIT-IDENTICAL output before and after the deletion campaign this file is a safety net for.

tests/golden/parity.json is the recorded output of tests/capture_golden.py run against the
real trained artifacts at outputs/long_frozen/artifacts/ -- see that script's module docstring for
the exact recipe (fixed seed, device="cpu", every float round-tripped via float.hex() for exact
equality). This module reruns the IDENTICAL recipe (by importing the capture functions directly, never
re-deriving them, so the two can never silently drift apart) and asserts every reconstructed float
equals the golden value with ==, not pytest.approx -- the whole point is exactness, not
closeness.

Skips cleanly (not failure) when either the real artifacts or the golden file itself is missing, so a
fresh clone (or a clone of this repo without outputs/ fetched) never fails this suite.

outputs/ is a symlink into a tree shared with sibling worktrees, so the artifacts these numbers
were recorded against can be rewritten by another branch. The test therefore checks
meta.artifact_sha256 first and fail with an explicit ARTIFACT DRIFT message when the bytes on
disk differ -- a code regression and someone else's rebuild must not look the same.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

import capture_golden as cg

HAS_ARTIFACTS = cg.ARTIFACT_DIR.exists()
HAS_GOLDEN = cg.GOLDEN_PATH.exists()
skip_no_parity = pytest.mark.skipif(
    not (HAS_ARTIFACTS and HAS_GOLDEN),
    reason=(
        f"parity net needs both {cg.ARTIFACT_DIR} and {cg.GOLDEN_PATH} -- "
        "run `PYTHONPATH=src .venv-core/bin/python tests/capture_golden.py` on a checkout that has "
        "outputs/long_frozen/artifacts/ to regenerate"
    ),
)


@pytest.fixture(scope="module")
def golden() -> dict:
    with open(cg.GOLDEN_PATH) as f:
        return json.load(f)


def assert_artifacts_match_capture(meta: dict) -> None:
    """Fail loudly when the on-disk artifacts are not the bytes the golden was recorded against.

    Without this check an artifact rebuilt by a sibling worktree is indistinguishable from a genuine
    numeric regression in this repo's code.
    """
    want = meta.get("artifact_sha256")
    if not want:
        return
    got = cg.artifact_sha256()
    drifted = sorted(n for n in want if got.get(n) != want[n])
    assert not drifted, (
        f"ARTIFACT DRIFT -- {', '.join(drifted)} in {cg.ARTIFACT_DIR} no longer match the bytes the "
        "golden file was captured against. outputs/ is shared with sibling worktrees, so this is most "
        "likely another branch's rebuild, NOT a regression in this checkout's code. Confirm the "
        "artifacts are the ones you mean to ship, then re-run tests/capture_golden.py."
    )


_hexf = cg._hexf
_hex_list = cg._hex_list


# ==== 1. t=0 ENTRY -- exact reproduction ====
@skip_no_parity
def test_entry_records_are_bit_identical_to_golden(golden):
    """Replay the 64 recorded t=0 retrievals and assert every float is the recorded one."""
    from conftest import skip_if_head_is_stale

    skip_if_head_is_stale(cg.ARTIFACT_DIR)
    meta = golden["meta"]
    assert_artifacts_match_capture(meta)

    from robot_ood.core.config import GraphConfig
    from robot_ood.runtime.retrieve import GraphRetriever

    rng = np.random.RandomState(meta["seed"])
    retriever = GraphRetriever.load(cg.ARTIFACT_DIR, cfg=GraphConfig(**meta["entry_cfg"]), device="cpu")
    got, want = cg.capture_entry(retriever, rng), golden["entry"]

    assert len(got) == len(want) == meta["n_entry"]
    for i, (g, w) in enumerate(zip(got, want)):
        assert g["node_idx"] == w["node_idx"], f"entry record {i}: node index moved"
        assert g.get("unreachable") == w.get("unreachable"), (
            f"entry record {i}: whether this query reaches the entry band at all has moved"
        )
        if w.get("unreachable"):
            continue
        assert g["reached"] == w["reached"], f"entry record {i}: candidate pool size moved"
        assert g["depth_hex"] == w["depth_hex"], f"entry record {i}: depth moved"
        assert g["q_star_hex"] == w["q_star_hex"], f"entry record {i}: q_star moved"


# ======================================================================================================
# 2. t>0 SENTINEL -- exact reproduction, or a clean explanation of why it was never recorded
# ======================================================================================================
@skip_no_parity
def test_track_records_are_bit_identical_to_golden_or_were_never_captured(golden):
    from conftest import skip_if_head_is_stale

    skip_if_head_is_stale(cg.ARTIFACT_DIR)
    meta = golden["meta"]
    assert_artifacts_match_capture(meta)

    if not meta.get("track_npz_present", False):
        assert golden["track"] is None, (
            "meta says no locator was present at capture time, but a track section was recorded"
        )
        import robot_ood.data.schema as schema

        assert not (cg.ARTIFACT_DIR / schema.HEAD_NPZ).exists(), (
            "g_head.npz now EXISTS in this checkout but the golden file was captured without it "
            "-- regenerate tests/golden/parity.json to also cover the sentinel half"
        )
        pytest.skip(f"{cg.ARTIFACT_DIR / schema.HEAD_NPZ} was absent at capture time")

    from robot_ood.core.config import GraphConfig
    from robot_ood.runtime.locator import GraphLocator

    # build_golden() draws the ENTRY records first, off the SAME RandomState, before ever touching the
    # locator -- replay that exactly so this RNG is in the same state build_golden()'s was when it
    # started drawing the track half (capture_track only draws ONE int, the demo index).
    rng = np.random.RandomState(meta["seed"])
    from robot_ood.runtime.retrieve import GraphRetriever

    cfg_entry = GraphConfig(**meta["entry_cfg"])
    entry_retriever = GraphRetriever.load(cg.ARTIFACT_DIR, cfg=cfg_entry, device="cpu")
    cg.capture_entry(entry_retriever, rng)   # replays the entry draws so rng is at the SAME state

    cfg_locator = GraphConfig(**meta["track_cfg"])
    locator = GraphLocator.load(cg.ARTIFACT_DIR, cfg=cfg_locator, device="cpu")
    got = cg.capture_track(locator, rng)
    want = golden["track"]

    assert got["demo_owner"] == want["demo_owner"]
    got_records, want_records = got["records"], want["records"]
    assert len(got_records) == len(want_records) == meta["n_track_steps"]

    for i, (g, w) in enumerate(zip(got_records, want_records)):
        assert g["step"] == w["step"]
        assert g["belief_entropy_hex"] == w["belief_entropy_hex"], f"step {i}: belief_entropy changed"
        assert g["depth_hex"] == w["depth_hex"], f"step {i}: belief-view depth changed"
        assert g["phase_hat_hex"] == w["phase_hat_hex"], f"step {i}: phase_hat changed"
        assert g["q_star_hex"] == w["q_star_hex"], f"step {i}: belief-view q_star changed"
