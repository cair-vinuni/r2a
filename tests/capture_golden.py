"""capture_golden.py — build tests/golden/parity.json, the bit-identical regression net that
pins the two LIVE inference paths to bit-identical output:

  1. t=0 ENTRY  -- robot_ood.runtime.retrieve.GraphRetriever, queried with a zero-velocity window
     (q0 repeated hist times).
  2. t>0 SENTINEL -- robot_ood.runtime.locator.GraphLocator, driven step-by-step over a short
     synthetic joint trajectory sliced from one real demo's raw frames.

Everything below is DETERMINISTIC: one seeded numpy.random.RandomState (never the global
np.random state, never wall-clock/PID-derived anything) drives every draw, device="cpu"
throughout, and every float that goes into the golden file is round-tripped via float.hex() so
tests/test_parity.py can assert exact (==, not approx) equality against a fresh rerun of
this exact recipe.

Run (regenerates tests/golden/parity.json in place):

    PYTHONPATH=src .venv-core/bin/python tests/capture_golden.py

Both halves read ONLY the real trained artifacts at outputs/long_frozen/artifacts/ (never a
synthetic toy graph -- the toy fixtures in tests/test_retrieve.py exist for fast, artifact-free
unit tests of the *mechanics*; this script exists to pin the *actual* deployed numbers). A
checkout with no g_head.npz there skips the sentinel half and records that in meta, so it is
never silently compared against a golden file captured from a different arm.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from robot_ood.core.config import GraphConfig
from robot_ood.data import schema
from robot_ood.runtime.retrieve import GraphRetriever
from robot_ood.runtime.locator import GraphLocator

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "outputs" / "long_frozen" / "artifacts"
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "parity.json"

SEED = 20260805          # fixed, arbitrary -- never re-drawn, never wall-clock derived
N_ENTRY = 64
N_TRACK_STEPS = 64
TRACK_STRIDE = 2          # raw-frame stride between consecutive locator checks

# t=0 ENTRY config -- the entry-band recipe pinned here as literal kwargs, not read from the
# environment, so this script has no dependency on which env vars a particular eval mode happens to set.
ENTRY_CFG_KWARGS = dict(entry_band=0.05, move_cost=False, topm=32, hist=8, device="cpu")

# t>0 SENTINEL config -- the locator's own finalize hand-rolls its argmax reduction regardless
# of cfg.readout, so this is deliberately a separate config from the entry one above.
TRACK_CFG_KWARGS = dict(hist=8, device="cpu")


# The four artifact files the recorded numbers depend on. outputs/ is a symlink into a tree
# shared with sibling worktrees, so any of them can be rewritten by another branch; hashing them
# lets test_parity say "the artifact changed" instead of blaming the code.
ARTIFACT_FILES = ("g_nodes.npz", "g_edges.npz", "g_locator.npz", "g_head.npz")


def artifact_sha256() -> dict:
    """{filename: sha256} for each present artifact in ARTIFACT_DIR; absent files are omitted."""
    out = {}
    for name in ARTIFACT_FILES:
        p = ARTIFACT_DIR / name
        if p.exists():
            out[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _hexf(x) -> str:
    """One float -> its exact hex round-trip representation."""
    return float(x).hex()


def _hex_list(arr) -> list:
    return [_hexf(v) for v in np.asarray(arr, dtype=np.float64).ravel().tolist()]


# ======================================================================================================
# 1. t=0 ENTRY capture
# ======================================================================================================
def capture_entry(retriever: GraphRetriever, rng: np.random.RandomState) -> list[dict]:
    """64 t=0 queries: fixed strided node indices, each node's own q perturbed by the seeded RNG,
    then wrapped into a zero-velocity window (q0 repeated cfg.hist times)."""
    nodes = retriever.nodes
    v = len(nodes)
    stride = max(v // N_ENTRY, 1)
    node_idx = [min(i * stride, v - 1) for i in range(N_ENTRY)]

    records = []
    for i in node_idx:
        q0 = nodes.q[i].astype(np.float64)
        q_pert = q0 + rng.normal(scale=0.02, size=q0.shape)
        q_hist = np.repeat(q_pert[None, :], retriever.cfg.hist, axis=0)   # zero velocity, by construction
        # A strided node can sit deep inside a demo, far from every entry-band node -- observe()
        # then raises rather than returning, and THAT is the number to pin: a future head that
        # starts reaching the band from here has changed, and this is where it shows.
        try:
            result = retriever.retrieve(q_hist)
        except RuntimeError as exc:
            if "no node is reachable" not in str(exc):
                raise
            records.append({"node_idx": int(i), "unreachable": True})
            continue
        records.append({
            "node_idx": int(i),
            "q_star_hex": _hex_list(result.q_star),
            "depth_hex": _hexf(result.depth),
            "reached": int(result.info["reached"]),   # the surviving candidate pool, post every mask
        })
    return records


# ======================================================================================================
# 2. t>0 SENTINEL capture
# ======================================================================================================
def capture_track(locator: GraphLocator, rng: np.random.RandomState) -> dict:
    """64 t>0 steps: a short synthetic trajectory built by walking consecutive raw q frames of one
    (seed-chosen) demo, fed to robot_ood.runtime.locator.GraphLocator one check at a time -- the
    clean-replay regime, driven through the live step() call.
    """
    nodes = locator.retriever.nodes
    j = int(rng.randint(0, nodes.n_demos))
    demo_start, demo_end = int(nodes.raw_ptr[j]), int(nodes.raw_ptr[j + 1])
    traj = nodes.q_raw[demo_start:demo_end].astype(np.float64)

    hist = locator.retriever.cfg.hist
    needed = hist + (N_TRACK_STEPS - 1) * TRACK_STRIDE
    if traj.shape[0] < needed:
        raise RuntimeError(
            f"capture_track: demo {j} has only {traj.shape[0]} raw frames, need >= {needed} "
            f"(hist={hist}, stride={TRACK_STRIDE}, steps={N_TRACK_STEPS}) -- pick a different seed/demo"
        )

    locator.reset()
    records = []
    for s in range(N_TRACK_STEPS):
        start = s * TRACK_STRIDE
        q_hist = traj[start:start + hist]
        step = locator.step(q_hist)
        b = step.belief_view
        records.append({
            "step": s,
            "belief_entropy_hex": _hexf(step.belief_entropy),
            "q_star_hex": _hex_list(b.q_star),
            "depth_hex": _hexf(b.depth),
            "phase_hat_hex": _hexf(step.phase_hat),
        })
    return {"demo_owner": j, "records": records}


# ======================================================================================================
# entry point
# ======================================================================================================
def build_golden() -> dict:
    head_path = ARTIFACT_DIR / schema.HEAD_NPZ
    graph_hash_hex = None
    if head_path.exists():
        raw = np.load(str(head_path))
        if "graph_hash" in raw.files:
            graph_hash_hex = str(raw["graph_hash"])

    track_npz_present = head_path.exists()

    meta = {
        "artifact_dir": str(ARTIFACT_DIR.relative_to(REPO_ROOT)),
        "graph_hash": graph_hash_hex,
        "seed": SEED,
        "n_entry": N_ENTRY,
        "n_track_steps": N_TRACK_STEPS,
        "track_stride": TRACK_STRIDE,
        "entry_cfg": ENTRY_CFG_KWARGS,
        "track_cfg": TRACK_CFG_KWARGS,
        "track_npz_present": track_npz_present,
        "artifact_sha256": artifact_sha256(),
    }

    rng = np.random.RandomState(SEED)

    entry_cfg = GraphConfig(**ENTRY_CFG_KWARGS)
    entry_retriever = GraphRetriever.load(ARTIFACT_DIR, cfg=entry_cfg, device="cpu",
                                          head_npz=schema.HEAD_NPZ)
    entry_records = capture_entry(entry_retriever, rng)

    golden = {"meta": meta, "entry": entry_records}

    if track_npz_present is False:
        print(
            f"capture_golden: {ARTIFACT_DIR / schema.HEAD_NPZ} is ABSENT -- skipping the t>0 "
            "sentinel half, since there is no locator in this checkout to pin numbers against. "
            "meta.track_npz_present=False records this."
        )
        golden["track"] = None
    else:
        track_cfg = GraphConfig(**TRACK_CFG_KWARGS)
        locator = GraphLocator.load(ARTIFACT_DIR, cfg=track_cfg, device="cpu")
        golden["track"] = capture_track(locator, rng)

    return golden


def main() -> None:
    if not ARTIFACT_DIR.exists():
        raise SystemExit(f"capture_golden: {ARTIFACT_DIR} does not exist -- nothing to capture against")
    golden = build_golden()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_PATH, "w") as f:
        json.dump(golden, f, indent=2, sort_keys=True)
        f.write("\n")
    n_track = 0 if golden["track"] is None else len(golden["track"]["records"])
    print(f"wrote {GOLDEN_PATH} -- {len(golden['entry'])} entry records, {n_track} track records")


if __name__ == "__main__":
    main()
